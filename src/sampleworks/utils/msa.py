from hashlib import sha3_256
from pathlib import Path

from loguru import logger

from sampleworks.utils.mmseqs2 import run_mmseqs2


MAX_PAIRED_SEQS = 8192
MAX_MSA_SEQS = 16384


# From https://github.com/jwohlwend/boltz/blob/main/src/boltz/main.py#L415
def compute_msa(
    data: dict[str, str],
    target_id: str,
    msa_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
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
        outputs[name] = msa_path

    return outputs


class MSAManager:
    """
    Manages operations related to MSA (Multiple Sequence Alignment).

    Facilitates handling and organization of MSA data within a specified
    directory to reduce calls to servers or other external resources.

    Ultimately, this class uses the above method compute_msa to generate MSA data.
    We store a cache of MSAs by computing a hash of all the input arguments for that method.
    When we get a call to compute_msa for a particular set of inputs, we check if we have
    results for that hash in our cache. If so, we use those results. Otherwise, we compute.

    Attributes:
        msa_dir (Path): Path to the directory where MSA data is stored.
    """

    def __init__(
        self,
        msa_cache_dir: Path | str | None = None,
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

    def _hash_arguments(
        self,
        data: dict[str, str],
        msa_pairing_strategy: str,
    ):
        encoded_sequence_tuple = str.encode(str(tuple(data.values())))
        hexdigest = sha3_256(encoded_sequence_tuple).hexdigest()
        return f"{msa_pairing_strategy}_{hexdigest}"

    def get_msa(self, data: dict[str, str], msa_pairing_strategy: str) -> dict[str, Path]:
        hash_key = self._hash_arguments(data, msa_pairing_strategy)
        msa_path_dict = {
            key: self.msa_dir / f"{hash_key}_{idx}.csv" for idx, key in enumerate(data)
        }

        if not all([m.exists() for m in msa_path_dict.values()]):
            msa_path_dict = compute_msa(
                data,
                hash_key,  # this is the "target_id" argument to compute_msa
                self.msa_dir,
                self.msa_server_url,
                msa_pairing_strategy,
                self.msa_server_username,
                self.msa_server_password,
                self.api_key_header,
                self.api_key_value,
            )

        return msa_path_dict
