from hashlib import sha3_256
from pathlib import Path

from loguru import logger

from sampleworks.utils.mmseqs2 import run_mmseqs2


MAX_PAIRED_SEQS = 8192
MAX_MSA_SEQS = 16384


# From https://github.com/jwohlwend/boltz/blob/main/src/boltz/main.py#L415
# FIXME: this function is a big, long mess. We should clean it up.
#   For the love of decent code, don't copy this and use it somewhere else and respect the
#   leading underscore!
def _compute_msa(
    data: dict[str, str],
    target_id: str,
    msa_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
    return_a3m: bool = False,
    msa_server_username: str | None = None,
    msa_server_password: str | None = None,
    api_key_header: str | None = None,
    api_key_value: str | None = None,
) -> dict[str, Path]:
    """Compute the MSA for the input data.

    Parameters
    ----------
    data : dict[str, str]
        The input protein sequences.
    target_id : str
        The target id.
    msa_dir : Path
        The msa directory.
    msa_server_url : str
        The MSA server URL.
    msa_pairing_strategy : str
        The MSA pairing strategy.
    msa_server_username : str, optional
        Username for basic authentication with MSA server.
    msa_server_password : str, optional
        Password for basic authentication with MSA server.
    api_key_header : str, optional
        Custom header key for API key authentication (default: X-API-Key).
    api_key_value : str, optional
        Custom header value for API key authentication (overrides --api_key if set).

    Returns
    -------
    dict[str, Path]
        A dictionary mapping target names (keys of input data dict) to MSA file paths.

    """
    logger.info(f"Calling MSA server for target {target_id} with {len(data)} sequences")
    logger.info(f"MSA server URL: {msa_server_url}")
    logger.info(f"MSA pairing strategy: {msa_pairing_strategy}")

    # Construct auth headers if API key header/value is provided
    auth_headers = None
    if api_key_value:
        key = api_key_header if api_key_header else "X-API-Key"
        value = api_key_value
        auth_headers = {"Content-Type": "application/json", key: value}
        logger.info(f"Using API key authentication for MSA server (header: {key})")
    elif msa_server_username and msa_server_password:
        logger.info("Using basic authentication for MSA server")
    else:
        logger.info("No authentication provided for MSA server")

    # NB: this code is borrowed from Boltz, and it appears to ignore the
    # pairing argument used elsewhere
    if len(data) > 1:
        paired_msas = run_mmseqs2(
            list(data.values()),
            str(msa_dir / f"{target_id}_paired_tmp"),
            use_env=True,
            use_pairing=True,
            host_url=msa_server_url,
            pairing_strategy=msa_pairing_strategy,
            msa_server_username=msa_server_username,
            msa_server_password=msa_server_password,
            auth_headers=auth_headers,
        )
    else:
        paired_msas = [""] * len(data)

    unpaired_msa = run_mmseqs2(
        list(data.values()),
        str(msa_dir / f"{target_id}_unpaired_tmp"),
        use_env=True,
        use_pairing=False,
        host_url=msa_server_url,
        pairing_strategy=msa_pairing_strategy,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        auth_headers=auth_headers,
    )

    # FIXME: this code relies on the fact that run_mmseqs2 returns sequence alignments in the same
    #  order as they are in `data`, and furthermore just returns a list of strings, the content
    #  of each string being a single sequence alignment. It's some weird file parsing that we
    #  should clean up so users don't break it or have to worry about it.
    outputs = {}
    for idx, name in enumerate(data):
        # Get paired sequences
        paired = paired_msas[idx].strip().splitlines()
        paired = paired[1::2]  # ignore headers
        paired = paired[:MAX_PAIRED_SEQS]

        # Set key per row and remove empty sequences
        keys = [idx for idx, s in enumerate(paired) if s != "-" * len(s)]
        paired = [s for s in paired if s != "-" * len(s)]

        # Combine paired-unpaired sequences
        unpaired = unpaired_msa[idx].strip().splitlines()
        unpaired = unpaired[1::2]
        unpaired = unpaired[: (MAX_MSA_SEQS - len(paired))]
        if paired:
            unpaired = unpaired[1:]  # ignore query is already present

        # Combine
        seqs = paired + unpaired
        keys = keys + [-1] * len(unpaired)

        # Dump MSA
        csv_str = ["key,sequence"] + [f"{key},{seq}" for key, seq in zip(keys, seqs)]

        msa_path = msa_dir / f"{target_id}_{idx}.csv"
        with msa_path.open("w") as f:
            f.write("\n".join(csv_str))

        # I'm going to make a bet for now that RF3 and others don't actually care about
        # the sequence identifiers and alignment stats, so we'll dump the contents of seqs
        # in a FASTA/a3m format:
        with msa_path.with_suffix(".a3m").open("w") as f:
            for seq_idx, seq in enumerate(seqs):
                f.write(f">{target_id}_{idx}_{seq_idx}\n{seq}\n")

        outputs[name] = msa_path

    return outputs


class MSAManager:
    """
    Manages multiple sequence alignment (MSA) operations, including caching,
    API requests, and file management.

    This class facilitates interacting with an external MSA server for sequence
    alignments while leveraging local caching for efficiency. It computes MSA
    results based on input data and pairing strategy, while transparently
    retrieving cached results when available.
    """

    def __init__(
        self,
        msa_cache_dir: Path | str | None = None, # if None, use default
        msa_server_url: str = "https://api.colabfold.com",
        msa_server_username: str | None = None,
        msa_server_password: str | None = None,
        api_key_header: str | None = None,
        api_key_value: str | None = None,
    ):
        if msa_cache_dir is None:
            home = Path.home()
            msa_cache_dir = home / ".sampleworks" / "msa"
            msa_cache_dir.mkdir(parents=True, exist_ok=True)

        self.msa_dir = Path(msa_cache_dir)
        self.msa_server_url = msa_server_url
        self.msa_server_username = msa_server_username
        self.msa_server_password = msa_server_password
        self.api_key_header = api_key_header
        self.api_key_value = api_key_value

        self._api_calls = 0
        self._cache_hits = 0

    def _hash_arguments(
        self,
        data: dict[str, str],
        msa_pairing_strategy: str,
    ):
        encoded_sequence_tuple = str.encode(str(tuple(data.values())))
        hexdigest = sha3_256(encoded_sequence_tuple).hexdigest()
        return f"{msa_pairing_strategy}_{hexdigest}"

    def get_msa(
            self,
            data: dict[str, str],
            msa_pairing_strategy: str,
            return_a3m: bool = False
    ) -> dict[str, Path]:
        """
        Fetches existing MSA files from disk or computes new ones if necessary.
        data: dict[str, str]
            A dictionary mapping target names to protein sequences.
        msa_pairing_strategy: str
            The MSA pairing strategy to use (usually "greedy").
        return_a3m: bool
            If True, returns the MSA in a3m format instead of csv.

        Returns: dict[str, Path]
            A dictionary mapping target names to MSA file paths.
        """
        hash_key = self._hash_arguments(data, msa_pairing_strategy)
        suffix = "a3m" if return_a3m else "csv"
        msa_path_dict = {
            key: self.msa_dir / f"{hash_key}_{idx}.{suffix}" for idx, key in enumerate(data)
        }

        if not all([m.exists() for m in msa_path_dict.values()]):
            msa_path_dict = _compute_msa(
                data,
                hash_key,  # this is the "target_id" argument to compute_msa
                self.msa_dir,
                self.msa_server_url,
                msa_pairing_strategy,
                msa_server_username=self.msa_server_username,
                msa_server_password=self.msa_server_password,
                api_key_header=self.api_key_header,
                api_key_value=self.api_key_value,
            )
            self._api_calls += 1
        else:
            self._cache_hits += 1

        if return_a3m:
            msa_path_dict = {
                key: str(msa_path.with_suffix(".a3m")) for key, msa_path in msa_path_dict.items()}

        return msa_path_dict

    def report_on_usage(self):
        """
        Report on the usage of the MSA manager, including API calls and cache hits.
        """
        logger.info(
            f"MSA Manager Usage Report: API Calls={self._api_calls}, Cache Hits={self._cache_hits}"
        )
