"""Backend registry. `load_backend("auto")` picks llama.cpp for .gguf paths, transformers otherwise."""
from __future__ import annotations

from .base import Backend, Neuron, Steer

__all__ = ["Backend", "Neuron", "Steer", "load_backend"]


def load_backend(model: str, backend: str = "auto", **kw) -> Backend:
    if backend == "auto":
        backend = "llamacpp" if model.lower().endswith(".gguf") else "mock" if model == "mock" else "hf"
    if backend == "hf":
        from .hf import HFBackend
        return HFBackend(model, **kw)
    if backend == "llamacpp":
        from .llamacpp import LlamaCppBackend
        return LlamaCppBackend(model, **kw)
    if backend == "mock":
        from .mock import MockBackend
        return MockBackend(model, **kw)
    raise ValueError(f"unknown backend {backend!r} (hf, llamacpp, mock)")
