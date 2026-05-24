"""State extraction: simulator -> (16, H, W) tensor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyBaba

from baba_world.constants import TENSOR_DIM

if TYPE_CHECKING:
    from baba_world.types import TransitionSample


class StateExtractor:
    """Converts pyBaba.Game into ML-ready state tensors."""

    def __init__(self, dtype: type = np.float32) -> None:
        self.dtype = dtype

    def extract(self, game: pyBaba.Game) -> np.ndarray:
        return extract_state(game, dtype=self.dtype)

    def shape(self, game: pyBaba.Game) -> tuple[int, int, int]:
        height = game.GetMap().GetHeight()
        width = game.GetMap().GetWidth()
        return (TENSOR_DIM, height, width)


def extract_state(game: pyBaba.Game, dtype: type = np.float32) -> np.ndarray:
    """Return state tensor with shape (C, H, W)."""
    height = game.GetMap().GetHeight()
    width = game.GetMap().GetWidth()
    flat = pyBaba.Preprocess.StateToTensor(game)
    tensor = np.asarray(flat, dtype=dtype).reshape(TENSOR_DIM, height, width)
    return tensor
