#!/usr/bin/env python3
"""
Play Baba Is You in the pygame GUI (human or model agent).

Human: arrow keys
Agent: one-step model lookahead (Phase 4 dual head)

Requires: pip install pygame scipy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pygame
import pyBaba

from baba_graph.predictor.agent import replay_and_choose
from baba_graph.predictor.model import BabaTransitionModel
from baba_graph.world_model import WorldModelConfig
from baba_graph.predictor.config import PredictorConfig
from baba_world.actions import decode_action, encode_action
from baba_world.paths import map_path, repo_root

BLOCK_SIZE = 24

ICON_IMAGES = {
    pyBaba.ObjectType.ICON_BABA: "BABA",
    pyBaba.ObjectType.ICON_FLAG: "FLAG",
    pyBaba.ObjectType.ICON_WALL: "WALL",
    pyBaba.ObjectType.ICON_ROCK: "ROCK",
    pyBaba.ObjectType.ICON_TILE: "TILE",
}

TEXT_IMAGES = {
    pyBaba.ObjectType.BABA: "BABA",
    pyBaba.ObjectType.IS: "IS",
    pyBaba.ObjectType.YOU: "YOU",
    pyBaba.ObjectType.FLAG: "FLAG",
    pyBaba.ObjectType.WIN: "WIN",
    pyBaba.ObjectType.WALL: "WALL",
    pyBaba.ObjectType.STOP: "STOP",
    pyBaba.ObjectType.ROCK: "ROCK",
    pyBaba.ObjectType.PUSH: "PUSH",
}


def _sprite_roots() -> list[Path]:
    root = repo_root()
    return [
        p
        for p in [
            root / "Extensions" / "BabaGUI" / "sprites",
            root / "Extensions" / "BabaRL" / "baba-babaisyou-v0" / "sprites",
        ]
        if p.is_dir()
    ]


def _load_sprite(name: str, *, is_icon: bool) -> pygame.Surface | None:
    sub = "icon" if is_icon else "text"
    for root in _sprite_roots():
        path = root / sub / f"{name}.gif"
        if path.is_file():
            return pygame.image.load(str(path))
    return None


def draw_game(screen: pygame.Surface, game: pyBaba.Game) -> None:
    screen.fill((0, 0, 0))
    for y in range(game.GetMap().GetHeight()):
        for x in range(game.GetMap().GetWidth()):
            for obj_type in game.GetMap().At(x, y).GetTypes():
                if obj_type == pyBaba.ObjectType.ICON_EMPTY:
                    continue
                is_icon = not pyBaba.IsTextType(obj_type)
                img_name = (
                    ICON_IMAGES.get(obj_type)
                    if is_icon
                    else TEXT_IMAGES.get(obj_type)
                )
                if img_name is None:
                    continue
                surf = _load_sprite(img_name, is_icon=is_icon)
                if surf is not None:
                    screen.blit(surf, (x * BLOCK_SIZE, y * BLOCK_SIZE))
    pygame.display.flip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Play Baba in pygame GUI")
    parser.add_argument("--map", default="baba_is_you")
    parser.add_argument("--mode", choices=("human", "agent"), default="human")
    parser.add_argument("--checkpoint", default="", help="Optional .pt state_dict path")
    parser.add_argument("--agent-interval", type=int, default=300, help="ms between agent moves")
    args = parser.parse_args()

    pygame.init()
    game = pyBaba.Game(str(map_path(args.map)))
    game.Reset()
    w = game.GetMap().GetWidth() * BLOCK_SIZE
    h = game.GetMap().GetHeight() * BLOCK_SIZE
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption(f"Baba — {args.map} ({args.mode})")
    clock = pygame.time.Clock()

    model: BabaTransitionModel | None = None
    if args.mode == "agent":
        import torch

        wc, pc = WorldModelConfig(), PredictorConfig()
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                if "world_config" in ckpt:
                    wc = ckpt["world_config"]
                if "predictor_config" in ckpt:
                    pc = ckpt["predictor_config"]
            model = BabaTransitionModel(wc, pc)
            if isinstance(ckpt, dict) and "model" in ckpt:
                model.load_state_dict(ckpt["model"])
            else:
                model.load_state_dict(ckpt)
            trained_on = ckpt.get("maps", "?") if isinstance(ckpt, dict) else "?"
            print(f"Loaded checkpoint (trained on: {trained_on})", flush=True)
        else:
            model = BabaTransitionModel(wc, pc)
            print(
                "WARNING: agent mode without --checkpoint uses random weights; "
                "Baba will not move meaningfully on hard maps.",
                flush=True,
            )
        model.eval()

    history: list[int] = []
    agent_ms = 0
    map_key = Path(map_path(args.map)).stem

    running = True
    while running:
        dt = clock.tick(60)
        agent_ms += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif args.mode == "human" and game.GetPlayState() == pyBaba.PlayState.PLAYING:
                    key_to_dir = {
                        pygame.K_UP: pyBaba.Direction.UP,
                        pygame.K_DOWN: pyBaba.Direction.DOWN,
                        pygame.K_LEFT: pyBaba.Direction.LEFT,
                        pygame.K_RIGHT: pyBaba.Direction.RIGHT,
                    }
                    if event.key in key_to_dir:
                        a = encode_action(key_to_dir[event.key])
                        game.MovePlayer(key_to_dir[event.key])
                        history.append(a)

        if (
            args.mode == "agent"
            and model is not None
            and game.GetPlayState() == pyBaba.PlayState.PLAYING
            and agent_ms >= args.agent_interval
        ):
            agent_ms = 0
            action = replay_and_choose(map_key, history, model)
            game.MovePlayer(decode_action(action))
            history.append(action)

        draw_game(screen, game)

        if game.GetPlayState() in (pyBaba.PlayState.WON, pyBaba.PlayState.LOST):
            font = pygame.font.SysFont(None, 36)
            msg = "YOU WIN!" if game.GetPlayState() == pyBaba.PlayState.WON else "YOU LOSE"
            text = font.render(msg, True, (255, 255, 0))
            screen.blit(text, (10, 10))
            pygame.display.flip()

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
