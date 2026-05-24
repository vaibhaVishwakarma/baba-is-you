"""Render per-object RGB patches (sprites or procedural open-vocabulary fallback)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from baba_graph.vocab import type_name
from baba_world.paths import repo_root

# Icon / text sprite name lookup (subset of BabaRL; unknown names use procedural art).
ICON_SPRITE_NAMES: dict[str, str] = {
    "ICON_BABA": "BABA",
    "ICON_FLAG": "FLAG",
    "ICON_WALL": "WALL",
    "ICON_ROCK": "ROCK",
    "ICON_TILE": "TILE",
    "ICON_WATER": "WATER",
    "ICON_GRASS": "GRASS",
    "ICON_LAVA": "LAVA",
    "ICON_SKULL": "SKULL",
    "ICON_FLOWER": "FLOWER",
}

TEXT_SPRITE_NAMES: dict[str, str] = {
    "BABA": "BABA",
    "IS": "IS",
    "YOU": "YOU",
    "FLAG": "FLAG",
    "WIN": "WIN",
    "WALL": "WALL",
    "STOP": "STOP",
    "ROCK": "ROCK",
    "PUSH": "PUSH",
    "WATER": "WATER",
    "SINK": "SINK",
    "LAVA": "LAVA",
    "MELT": "MELT",
    "HOT": "HOT",
    "SKULL": "SKULL",
    "DEFEAT": "DEFEAT",
}


class PatchRenderer:
    """Produces fixed-size RGB patches for any object type name (open vocabulary)."""

    def __init__(self, patch_size: int = 32) -> None:
        self.patch_size = patch_size
        self._sprite_roots = _discover_sprite_roots()
        self._cache: dict[tuple[str, bool], np.ndarray] = {}

    def render(self, type_name: str, *, is_text: bool) -> np.ndarray:
        """Return float32 patch (3, H, W) in [0, 1]."""
        key = (type_name, is_text)
        if key not in self._cache:
            self._cache[key] = self._render_uncached(type_name, is_text=is_text)
        return self._cache[key]

    def _render_uncached(self, type_name: str, *, is_text: bool) -> np.ndarray:
        sprite = _load_sprite(type_name, is_text=is_text, roots=self._sprite_roots)
        if sprite is not None:
            return _resize_patch(sprite, self.patch_size)

        return _procedural_patch(type_name, is_text=is_text, size=self.patch_size)


def _discover_sprite_roots() -> list[Path]:
    root = repo_root()
    candidates = [
        root / "Extensions" / "BabaRL" / "baba-babaisyou-v0" / "sprites",
        root / "Extensions" / "BabaRL" / "baba-outofreach-v0" / "sprites",
        root / "Extensions" / "BabaRL" / "baba-volcano-v0" / "sprites",
        root / "Resources" / "sprites",
    ]
    return [p for p in candidates if p.is_dir()]


def _sprite_filename(type_name: str, *, is_text: bool) -> str | None:
    if is_text:
        return TEXT_SPRITE_NAMES.get(type_name)
    return ICON_SPRITE_NAMES.get(type_name)


def _load_sprite(
    type_name: str,
    *,
    is_text: bool,
    roots: list[Path],
) -> np.ndarray | None:
    name = _sprite_filename(type_name, is_text=is_text)
    if name is None:
        return None

    sub = "text" if is_text else "icon"
    for root in roots:
        for ext in (".gif", ".png", ".jpg"):
            path = root / sub / f"{name}{ext}"
            if path.is_file():
                return _load_image_file(path)
    return None


def _load_image_file(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError:
        return None  # type: ignore[return-value]

    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


def _resize_patch(patch: np.ndarray, size: int) -> np.ndarray:
    try:
        import torch
        import torch.nn.functional as F

        t = torch.from_numpy(patch).unsqueeze(0)
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
        return t.squeeze(0).numpy()
    except ImportError:
        # Nearest-neighbor fallback without torch
        from PIL import Image

        img = Image.fromarray((patch.transpose(1, 2, 0) * 255).astype(np.uint8))
        img = img.resize((size, size), Image.Resampling.NEAREST)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return np.transpose(arr, (2, 0, 1))


def _procedural_patch(type_name: str, *, is_text: bool, size: int) -> np.ndarray:
    """
    Deterministic RGB patch for unknown / missing sprites (open-vocabulary safe).

    Same type_name always yields the same patch; novel names map to novel colors.
    """
    digest = hashlib.md5(f"{type_name}:{'text' if is_text else 'icon'}".encode()).digest()
    r, g, b = digest[0] / 255.0, digest[1] / 255.0, digest[2] / 255.0
    patch = np.zeros((3, size, size), dtype=np.float32)
    patch[0] = r * 0.6 + 0.2
    patch[1] = g * 0.6 + 0.2
    patch[2] = b * 0.6 + 0.2

    if is_text:
        # Lighter center band suggests text block
        margin = size // 4
        patch[:, margin : size - margin, margin : size - margin] += 0.25
    else:
        # Icon: darker border
        patch *= 0.85
        patch[:, :2, :] += 0.15
        patch[:, -2:, :] += 0.15
        patch[:, :, :2] += 0.15
        patch[:, :, -2:] += 0.15

    return np.clip(patch, 0.0, 1.0)


def render_node_patch(obj_type, *, patch_size: int = 32) -> np.ndarray:
    """Convenience: render from pyBaba ObjectType."""
    import pyBaba

    name = type_name(obj_type)
    is_text = bool(pyBaba.IsTextType(obj_type))
    return PatchRenderer(patch_size=patch_size).render(name, is_text=is_text)
