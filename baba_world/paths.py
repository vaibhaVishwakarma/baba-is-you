"""Repository path helpers."""

from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent


def repo_root() -> Path:
    """Return repository root (directory containing Resources/Maps)."""
    for candidate in (_REPO_ROOT, *_REPO_ROOT.parents):
        if (candidate / "Resources" / "Maps").is_dir():
            return candidate
    return _REPO_ROOT


def map_path(name: str) -> Path:
    """Resolve a map file by short name or path."""
    path = Path(name)
    if path.is_file():
        return path.resolve()
    resolved = repo_root() / "Resources" / "Maps" / f"{name}.txt"
    if not resolved.is_file() and not name.endswith(".txt"):
        resolved = repo_root() / "Resources" / "Maps" / name
    if not resolved.is_file():
        raise FileNotFoundError(f"Map not found: {name}")
    return resolved


# Maps validated for the 16-channel tensor representation.
SUPPORTED_MAPS = (
    "baba_is_you",
    "out_of_reach",
    "simple_map",
    "off_limits",
    "off_limits_bug",
)
