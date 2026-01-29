from hashlib import sha3_256
from pathlib import Path

from loguru import logger

from sampleworks.utils.guidance_constants import StructurePredictor
from sampleworks.utils.imports import PROTENIX_AVAILABLE
from sampleworks.utils.mmseqs2 import run_mmseqs2


if PROTENIX_AVAILABLE:
    from runner.msa_search import msa_search as protenix_msa_search  # pyright:ignore [reportMissingImports] # noqa: I001

    logger.debug("Protenix MSA tools are available at top of msa.py")
else:
    logger.error("Protenix is not installed, cannot use Protenix MSA tools.")


MAX_PAIRED_SEQS = 8192
MAX_MSA_SEQS = 16384


# From https://github.com/jwohlwend/boltz/blob/main/src/boltz/main.py#L415
# FIXME: this function is a big, long mess. We should clean it up.
#   For the love of decent code, don't copy this and use it somewhere else and respect the
#   leading underscore!
def _compute_msa(
    data: dict[str | int, str],
    target_id: str,
    msa_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
    msa_server_username: str | None = None,
    msa_server_password: str | None = None,
    api_key_header: str | None = None,
    api_key_value: str | None = None,
) -> dict[str | int, Path]:
    """Compute the MSA for the input data.

    Parameters
    ----------
    data : dict[str | int, str]
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
    dict[str | int, Path]
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
    # pairing argument used elsewhere. It also relies on dicts being ordered now
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
        keys = [pair_idx for pair_idx, s in enumerate(paired) if s != "-" * len(s)]
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

        """ Write out a3m format for RF3 """
        # in addition to CSV write a3m (FASTA) formats--but omit the alignment stats
        with msa_path.with_suffix(".a3m").open("w") as f:
            for seq_idx, seq in enumerate(seqs):
                f.write(f">{target_id}_{idx}_{seq_idx}\n{seq}\n")

        outputs[name] = msa_path

        """
        a3m files for Protenix.
        In case we weren't having enough fun, we need to write it all out in yet a third
        # format, for use by Protenix. Protenix expects a JSON blob like so:
        [
          {
            "sequences": [
              {"proteinChain":
                {
                  "sequence": "ACDE...",
                  "msa":
                    {
                      "precomputed_msa_dir:": "/path/to/msa_directory"
                      "pairing_db": "uniref100"
                    },
                  ... other fields
                }
              }, ...
            ]
          }, ...
        ]

        It expects /path/to/msa_directory to look like this:
        0:  # index corresponds to position in the list "sequences"
            non_pairing.a3m
            pairing.a3m  # only present if pairing was used.
        1:
            non_pairing.a3m
            pairing.a3m
        etc...
        """

        msa_idx_idr = msa_dir / f"{idx}"
        msa_idx_idr.mkdir(exist_ok=True, parents=True)
        logger.info(f"Writing MSA for target {target_id} sequence {idx} to {msa_idx_idr.resolve()}")
        msa_idx_idr.joinpath("non_pairing.a3m").write_text(unpaired_msa[idx])
        if paired:
            msa_idx_idr.joinpath("pairing.a3m").write_text(paired_msas[idx])

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
        msa_cache_dir: Path | str | None = None,  # if None, use default
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

    @staticmethod
    def _hash_arguments(data: dict[str | int, str], msa_pairing_strategy: str) -> str:
        encoded_sequence_tuple = str.encode(str(tuple(data.values())) + msa_pairing_strategy)
        hexdigest = sha3_256(encoded_sequence_tuple).hexdigest()
        return hexdigest

    def get_msa(
        self,
        data: dict[str | int, str],
        msa_pairing_strategy: str,
        structure_predictor: str | StructurePredictor = StructurePredictor.BOLTZ_2,
    ) -> dict[str | int, Path]:
        """
        Fetches existing MSA files from disk or computes new ones if necessary.
        data: dict[str | int, str]
            A dictionary mapping target (usu. chain or index) names to protein sequences.
        msa_pairing_strategy: str
            The MSA pairing strategy to use (usually "greedy").
        structure_predictor: str
            The name of the model that will use the MSA, to make sure the format is correct.

        Returns: dict[str, Path]
            A dictionary mapping target names to MSA file paths.
        """
        hash_key = self._hash_arguments(data, msa_pairing_strategy)

        if structure_predictor in [
            StructurePredictor.BOLTZ_1,
            StructurePredictor.BOLTZ_2,
            StructurePredictor.RF3,
        ]:
            # get standard MSAs
            suffix = "a3m" if structure_predictor == StructurePredictor.RF3 else "csv"
            msa_path_dict = {
                key: self.msa_dir / f"{hash_key}_{idx}.{suffix}" for idx, key in enumerate(data)
            }

            if not all([m.exists() for m in msa_path_dict.values()]):
                # this will generate both a3m and csv files for us.
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

        # Protenix needs special MSAs
        # This is a kind of hacky way to handle Protenix; I'm not sure what to do easily but
        # use their pipeline, and just have it put everything in our cache directory.
        elif structure_predictor == StructurePredictor.PROTENIX:
            if not PROTENIX_AVAILABLE:
                raise RuntimeError("Protenix is not installed, cannot use Protenix MSA tools.")

            logger.info("Running Protenix MSA tools.")
            # Make sure we have a protenix subdirectory
            protenix_dir = self.msa_dir / "protenix"
            protenix_dir.mkdir(parents=True, exist_ok=True)
            # Protenix adds extra information, easiest just to use their pipeline.
            # make sure sort order stays the same:
            data_keys = sorted(data.keys())
            sequences = [data[key] for key in data_keys]
            out_dir = self.msa_dir / "protenix" / hash_key

            msa_directories = [out_dir / str(idx) for idx in data_keys]
            reqd_files = ["non_pairing.a3m", "pairing.a3m"]
            need_msas = not all(
                (out_dir / str(idx) / fn).exists() for idx in data_keys for fn in reqd_files
            )
            if need_msas:
                msa_directories = protenix_msa_search(  # pyright:ignore [reportPossiblyUnboundVariable]
                    sequences, out_dir, mode="protenix"
                )
                self._api_calls += 1
            else:
                self._cache_hits += 1

            msa_path_dict = {key: path for key, path in zip(data_keys, msa_directories)}

        else:
            raise ValueError(f"Unknown structure predictor: {structure_predictor}")
        return msa_path_dict

    def report_on_usage(self):
        """
        Report on the usage of the MSA manager, including API calls and cache hits.
        """
        logger.info(
            f"MSA Manager Usage Report: API Calls={self._api_calls}, Cache Hits={self._cache_hits}"
        )
