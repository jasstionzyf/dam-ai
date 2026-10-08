"""Device/dtype selection shared by all embedder loaders."""

from __future__ import annotations


def pick_device() -> str:
    torch = __import__("torch")
    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_dtype(device: str):
    torch = __import__("torch")
    if device == "cuda":
        return torch.float16
    return torch.float32
