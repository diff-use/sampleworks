"""Load presets from TOML and apply runtime overrides.

Resolution order for every string value (defaults block and ``args``):
  1. ``${VAR}`` references are resolved against the process environment,
     with the preset's ``[defaults]`` block filling in any unset keys.
  2. ``--set <dotted-path>=<value>`` CLI overrides are applied last.
"""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .schema import Job, Preset


_EXPERIMENTS_DIR_NAME = "experiments"
_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_TOP_LEVEL_KEYS = frozenset({"description", "defaults", "shared_args", "jobs"})


def list_presets() -> list[str]:
    """List experiment preset names from the top-level ``experiments`` directory.

    Returns
    -------
    list of str
        Preset names (filename stems, no ``.toml`` extension), sorted
        alphabetically. If multiple experiment directories are visible, the
        first directory in the resolution order wins for duplicate names.
    """
    names: dict[str, Path] = {}
    for directory in _experiment_dirs():
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if path.is_file() and path.suffix == ".toml":
                names.setdefault(path.stem, path)
    return sorted(names)


def load_preset(name_or_path: str, *, overrides: Iterable[str] = ()) -> Preset:
    """Load a preset by experiment name or filesystem path.

    Parameters
    ----------
    name_or_path : str
        Either the name of a preset in the top-level ``experiments`` directory
        (as returned by :func:`list_presets`) or a path ending in ``.toml``.
    overrides : Iterable of str, optional
        ``KEY=VALUE`` strings as accepted by ``--set``. Applied before
        variable interpolation.

    Returns
    -------
    Preset
        Fully resolved preset ready for :func:`runner.run`.

    Raises
    ------
    FileNotFoundError
        If ``name_or_path`` matches no experiment preset and no file on disk.
    KeyError
        If an override path begins with an unknown top-level key, or if a
        ``${VAR}`` reference cannot be resolved against the environment or
        the ``[defaults]`` block.
    ValueError
        If an override is malformed (missing ``=``).
    """
    raw = _read_toml(name_or_path)
    overrides_list = list(overrides)
    raw = _apply_overrides(raw, overrides_list)
    raw = _resolve_variables(raw)
    return _build_preset(name=_preset_name(name_or_path), raw=raw)


def _read_toml(name_or_path: str) -> dict[str, Any]:
    """Read raw TOML from a filesystem path or an experiment preset name.

    Parameters
    ----------
    name_or_path : str
        Experiment preset name or filesystem path ending in ``.toml``.

    Returns
    -------
    dict of str to Any
        Parsed TOML, before override application or interpolation.

    Raises
    ------
    FileNotFoundError
        If neither location yields a TOML file.
    """
    path = _find_preset_path(name_or_path)
    if path is not None:
        return tomllib.loads(path.read_text())
    raise FileNotFoundError(
        f"No preset {name_or_path!r}. Experiments: {list_presets()}. "
        "Put TOML presets in ./experiments or pass a path to a .toml file."
    )


def _find_preset_path(name_or_path: str) -> Path | None:
    """Resolve a preset name or path to a TOML file.

    Parameters
    ----------
    name_or_path : str
        Preset name (``full_8gpu``), TOML filename (``full_8gpu.toml``), or
        filesystem path.

    Returns
    -------
    pathlib.Path or None
        Existing TOML path if found, otherwise ``None``.
    """
    path = Path(name_or_path)
    if path.suffix == ".toml" and path.is_file():
        return path

    preset_filename = path.name if path.suffix == ".toml" else f"{name_or_path}.toml"
    for directory in _experiment_dirs():
        candidate = directory / preset_filename
        if candidate.is_file():
            return candidate
    return None


def _experiment_dirs() -> list[Path]:
    """Return candidate top-level experiment directories in precedence order.

    Returns
    -------
    list of pathlib.Path
        Existing or candidate ``experiments`` directories. Duplicates are
        removed while preserving order.
    """
    candidates: list[Path] = []

    explicit = os.environ.get("SAMPLEWORKS_EXPERIMENTS_DIR")
    if explicit:
        candidates.append(Path(explicit))

    source_dir = os.environ.get("SAMPLEWORKS_SOURCE_DIR")
    if source_dir:
        candidates.append(Path(source_dir) / _EXPERIMENTS_DIR_NAME)

    candidates.append(Path("/home/dev/workspace") / _EXPERIMENTS_DIR_NAME)
    candidates.extend(_find_upward_experiment_dirs(Path.cwd()))
    candidates.extend(_find_upward_experiment_dirs(Path(__file__).resolve()))

    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve(strict=False)
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _find_upward_experiment_dirs(start: Path) -> list[Path]:
    """Search parents of ``start`` for top-level ``experiments`` directories.

    Parameters
    ----------
    start : pathlib.Path
        Directory or file path to begin searching from.

    Returns
    -------
    list of pathlib.Path
        Candidate experiment directories nearest to farthest.
    """
    current = start if start.is_dir() else start.parent
    dirs: list[Path] = []
    for parent in [current, *current.parents]:
        candidate = parent / _EXPERIMENTS_DIR_NAME
        if candidate.is_dir():
            dirs.append(candidate)
    return dirs


def _preset_name(name_or_path: str) -> str:
    """Return the canonical preset name for an experiment name or path argument.

    Parameters
    ----------
    name_or_path : str
        Either an experiment name or a path ending in ``.toml``.

    Returns
    -------
    str
        Filename stem if ``name_or_path`` looks like a path; otherwise the
        argument unchanged.
    """
    return Path(name_or_path).stem if name_or_path.endswith(".toml") else name_or_path


def _apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply each ``KEY=VALUE`` override to the raw preset dict in place.

    Parameters
    ----------
    raw : dict of str to Any
        Parsed TOML to mutate.
    overrides : list of str
        Each entry must contain exactly one ``=``.

    Returns
    -------
    dict of str to Any
        The same ``raw`` dict (mutated).

    Raises
    ------
    ValueError
        If an override is missing the ``=`` separator.
    KeyError
        If an override's top-level key is unknown.
    """
    for spec in overrides:
        if "=" not in spec:
            raise ValueError(f"--set expects KEY=VALUE, got {spec!r}")
        key, value = spec.split("=", 1)
        _set_dotted(raw, key.strip(), _coerce(value))
    return raw


def _set_dotted(obj: dict[str, Any], dotted: str, value: Any) -> None:
    """Set a nested value in ``obj`` addressed by a dotted path.

    Job-list elements can be addressed by job name or by integer index.

    Parameters
    ----------
    obj : dict of str to Any
        Root dict to mutate.
    dotted : str
        Dotted path, e.g. ``"jobs.rf3.args.gradient-weights"`` or
        ``"defaults.DATA_DIR"``.
    value : Any
        Coerced value to write at the leaf.

    Raises
    ------
    KeyError
        If the first segment is not one of :data:`_TOP_LEVEL_KEYS`, or if a
        list segment references a missing job name or index.
    TypeError
        If the path attempts to descend through a non-container value.
    """
    parts = dotted.split(".")
    if parts[0] not in _TOP_LEVEL_KEYS:
        raise KeyError(
            f"--set: unknown top-level key {parts[0]!r} in {dotted!r}. "
            f"Valid top-level keys: {sorted(_TOP_LEVEL_KEYS)}"
        )
    cursor: Any = obj
    for i, part in enumerate(parts[:-1]):
        cursor = _index(cursor, part, where=".".join(parts[: i + 1]))
    leaf_parent = cursor
    leaf_key = parts[-1]
    if isinstance(leaf_parent, list):
        leaf_parent[_find_in_list(leaf_parent, leaf_key, where=dotted)] = value
    else:
        leaf_parent[leaf_key] = value


def _index(cursor: Any, part: str, *, where: str) -> Any:
    """Descend one level into a dict or list, auto-creating empty intermediates.

    Parameters
    ----------
    cursor : Any
        Current node in the traversal.
    part : str
        Next segment of the dotted path.
    where : str
        Path so far, used in error messages.

    Returns
    -------
    Any
        The child node.

    Raises
    ------
    TypeError
        If ``cursor`` is neither a dict nor a list.
    """
    if isinstance(cursor, list):
        return cursor[_find_in_list(cursor, part, where=where)]
    if isinstance(cursor, dict):
        if part not in cursor:
            cursor[part] = {}
        return cursor[part]
    raise TypeError(f"Cannot descend into {type(cursor).__name__} at {where!r}")


def _find_in_list(items: list[Any], key: str, *, where: str) -> int:
    """Locate a list element by integer index or by ``name`` field.

    Parameters
    ----------
    items : list of Any
        List to search.
    key : str
        Numeric string (positive or negative index) or a name to match against
        each element's ``"name"`` key.
    where : str
        Path so far, used in error messages.

    Returns
    -------
    int
        Index of the matching element.

    Raises
    ------
    KeyError
        If no element with the given name exists.
    """
    if key.isdigit() or (key.startswith("-") and key[1:].isdigit()):
        return int(key)
    for i, item in enumerate(items):
        if isinstance(item, dict) and item.get("name") == key:
            return i
    raise KeyError(f"No list element named {key!r} at {where!r}")


def _coerce(value: str) -> Any:
    """Convert a string CLI override value to bool, int, float, or leave as str.

    Parameters
    ----------
    value : str
        Right-hand side of ``KEY=VALUE``.

    Returns
    -------
    Any
        ``True``/``False`` for ``"true"``/``"false"`` (case-insensitive);
        ``int`` or ``float`` if parseable; otherwise the original string.
    """
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _resolve_variables(raw: dict[str, Any]) -> dict[str, Any]:
    """Expand ``${VAR}`` references throughout the raw preset.

    Defaults are resolved in TOML order, so later defaults can reference
    earlier ones (e.g. ``PROTEINS_CSV = "${DATA_DIR}/proteins.csv"``). Process
    environment variables take precedence over the ``[defaults]`` block.

    Parameters
    ----------
    raw : dict of str to Any
        Parsed TOML, after override application.

    Returns
    -------
    dict of str to Any
        New dict with all string values fully expanded and ``defaults``
        replaced with the resolved values.

    Raises
    ------
    KeyError
        If any ``${VAR}`` cannot be resolved.
    """
    defaults: dict[str, str] = dict(raw.get("defaults", {}))
    accumulated: dict[str, str] = dict(os.environ)
    resolved_defaults: dict[str, str] = {}
    for key, default_value in defaults.items():
        if key in os.environ:
            resolved_defaults[key] = os.environ[key]
        else:
            resolved_defaults[key] = _expand(default_value, accumulated)
        accumulated[key] = resolved_defaults[key]
    resolved = _walk(raw, accumulated)
    resolved["defaults"] = resolved_defaults
    return resolved


def _walk(obj: Any, env: dict[str, str]) -> Any:
    """Recursively expand ``${VAR}`` in every string within ``obj``.

    Parameters
    ----------
    obj : Any
        Arbitrary nested dict/list/scalar.
    env : dict of str to str
        Resolved variable map.

    Returns
    -------
    Any
        Structurally identical copy with strings expanded.
    """
    if isinstance(obj, dict):
        return {k: _walk(v, env) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk(item, env) for item in obj]
    if isinstance(obj, str):
        return _expand(obj, env)
    return obj


def _expand(text: str, env: dict[str, str]) -> str:
    """Substitute ``${VAR}`` references in ``text`` until a fixed point.

    Parameters
    ----------
    text : str
        String potentially containing ``${VAR}`` references.
    env : dict of str to str
        Variable map.

    Returns
    -------
    str
        Fully expanded string.

    Raises
    ------
    KeyError
        If a referenced variable is not in ``env``.
    """

    def repl(match: re.Match[str]) -> str:
        """Return the configured value for one ``${VAR}`` interpolation match."""
        var = match.group(1)
        if var not in env:
            raise KeyError(f"Undefined variable ${{{var}}} in preset (no env var, no default)")
        return env[var]

    prev = None
    current = text
    while prev != current:
        prev = current
        current = _VAR_PATTERN.sub(repl, current)
    return current


def _build_preset(*, name: str, raw: dict[str, Any]) -> Preset:
    """Construct a :class:`Preset` from a resolved raw dict.

    Parameters
    ----------
    name : str
        Preset name (assigned to :attr:`Preset.name`).
    raw : dict of str to Any
        Resolved TOML.

    Returns
    -------
    Preset
        Validated preset.

    Raises
    ------
    ValueError
        If ``raw['jobs']`` is not a list, or if any :class:`Job` /
        :class:`Preset` invariant fails (see their docstrings).
    """
    raw_jobs = raw.get("jobs", [])
    if not isinstance(raw_jobs, list):
        raise ValueError(f"Preset {name!r}: 'jobs' must be a list")
    jobs = [
        Job(
            name=str(j["name"]),
            env=str(j["env"]),
            gpus=str(j["gpus"]),
            output_subdir=str(j["output_subdir"]),
            args=dict(j.get("args", {})),
        )
        for j in raw_jobs
    ]
    return Preset(
        name=name,
        description=str(raw.get("description", "")),
        defaults=dict(raw.get("defaults", {})),
        shared_args=dict(raw.get("shared_args", {})),
        jobs=jobs,
    )
