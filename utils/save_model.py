# utils/train.py
from pathlib import Path
import torch

def _state_dict(model):
    if isinstance(model, dict):
        return {k: v.state_dict() for k, v in model.items()}
    return model.state_dict()


def _load_state_dict(model, state):
    if isinstance(model, dict):
        for k, v in model.items():
            v.load_state_dict(state[k])
    else:
        model.load_state_dict(state)


def save_checkpoint(path: str, model, epoch: int, best_val: float):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": _state_dict(model),
            "best_val": best_val,
        },
        path,
    )


def load_checkpoint(path: str, model, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    _load_state_dict(model, ckpt["model"])
    return ckpt