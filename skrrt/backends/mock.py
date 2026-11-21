"""A synthetic model for tests and GPU-free demos (`skrrt hunt --model mock ...`). numpy only.

The mock has random MLP write vectors and one planted "persona direction" d. Two neurons are
planted on it: a strong one (layer 3, neuron 17, cos about +0.96) and a weaker one with the
opposite sign (layer 4, neuron 5, cos about -0.6). Its behaviour:

  capture   the residual moves along d when the reply text contains the marker word
  generate  the reply contains the marker when the summed push along d is large enough
            (about 0.55 to 0.8 residual norms, depending on the prompt), and turns into
            repetition when the total push is too large (about 1.9 to 2.4 residual norms)
  yes_no    keyword based: "is it fluent" looks for repetition, anything else for the marker

So a hunt for a persona whose example replies contain the marker should confirm L3 N17 at a dose
near +1, while random neurons do nothing.

    python -m skrrt.backends.mock > zorp.yaml       # a demo persona file for the mock
    skrrt hunt --model mock --persona zorp.yaml --judge self
"""
from __future__ import annotations

import re
import zlib

import numpy as np

from .base import Backend, Neuron, Steer


def _seed(*parts) -> int:
    return zlib.crc32("\x1f".join(str(p) for p in parts).encode())


def _unit(rng: np.random.Generator, n: int) -> np.ndarray:
    v = rng.standard_normal(n)
    return v / np.linalg.norm(v)


def _u(*parts) -> float:
    """Deterministic number in [0, 1) from the parts."""
    return _seed(*parts) / 2 ** 32


class MockBackend(Backend):
    def __init__(self, model: str = "mock", n_layers: int = 6, hidden: int = 256, inter: int = 128,
                 experts: int = 0, seed: int = 0, marker: str = "zorp", bad_layers=()):
        self.name = model
        self.n_layers, self.hidden, self.inter = int(n_layers), int(hidden), int(inter)
        self._experts = int(experts)
        self.marker = str(marker).lower()
        if isinstance(bad_layers, (int, str)):
            bad_layers = [int(x) for x in str(bad_layers).split(",") if x != ""]
        self.bad_layers = {int(x) for x in bad_layers}  # layers that refuse to be steered
        if self.n_layers < 5:
            raise ValueError("mock needs at least 5 layers")
        rng = np.random.default_rng(int(seed))
        g = max(1, self._experts)
        self.direction = _unit(rng, self.hidden)
        self.common = np.stack([_unit(rng, self.hidden) for _ in range(self.n_layers)])
        self.W = (0.1 * rng.standard_normal((self.n_layers, g, self.inter, self.hidden))).astype(np.float32)
        e1, e2 = (1 % g, 2 % g)
        self.planted = [Neuron(3, 17, e1 if self._experts else None), Neuron(4, 5, e2 if self._experts else None)]
        self.W[3, e1, 17] = 2.0 * self.direction + 0.6 * _unit(rng, self.hidden)
        self.W[4, e2, 5] = -1.5 * self.direction + 2.0 * _unit(rng, self.hidden)
        self.resid_norm = [10.0 * (1 + layer) for layer in range(self.n_layers)]
        self._calibrated = False

    # ---- structure
    def experts(self, layer: int) -> int:
        return self._experts

    def write_vectors(self, layer: int) -> np.ndarray:
        return self.W[layer].copy()

    def _w(self, n: Neuron) -> np.ndarray:
        if not 0 <= n.layer < self.n_layers or not 0 <= n.index < self.inter:
            raise ValueError(f"mock: no such neuron {n.label()}")
        if (n.expert is None) != (self._experts == 0) or (n.expert or 0) >= max(1, self._experts):
            raise ValueError(f"mock: no such neuron {n.label()} (experts per layer: {self._experts})")
        return self.W[n.layer, n.expert or 0, n.index].astype(np.float64)

    # ---- text
    def chat(self, user: str, system: str | None = None) -> str:
        s = f"<|system|>{system}\n" if system else ""
        return s + f"<|user|>{user}\n<|assistant|>"

    @staticmethod
    def _user(prompt: str) -> str:
        m = re.search(r"<\|user\|>(.*)\n<\|assistant\|>\Z", prompt, re.S)
        return m[1] if m else prompt

    # ---- measurements
    def calibrate(self, prompts: list[str]) -> dict:
        self._calibrated = True
        return {"median_resid_norm": {str(i): r for i, r in enumerate(self.resid_norm)}, "n_prompts": len(prompts)}

    def unit(self, neuron: Neuron) -> float:
        if not self._calibrated:
            raise RuntimeError("mock: calibrate() must be called before unit()/steering")
        w = self._w(neuron)
        return float(self.resid_norm[neuron.layer] / np.linalg.norm(w))

    def capture(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        out = np.zeros((len(pairs), self.n_layers, self.hidden), np.float32)
        for i, (prompt, reply) in enumerate(pairs):
            shift = 0.5 if self.marker in reply.lower() else 0.0
            for layer in range(self.n_layers):
                v = (0.9 * self.common[layer]
                     + 0.3 * _unit(np.random.default_rng(_seed("p", prompt, layer)), self.hidden)
                     + 0.1 * _unit(np.random.default_rng(_seed("r", reply, layer)), self.hidden))
                if layer >= 1:
                    v = v + shift * self.direction
                out[i, layer] = self.resid_norm[layer] * v
        return out

    def _push(self, steers: list[Steer]) -> tuple[float, float]:
        """(push along the planted direction, total push size), both in residual norms."""
        total = np.zeros(self.hidden)
        for s in steers:
            if s.neuron.layer in self.bad_layers:
                raise RuntimeError(f"mock: cannot steer layer {s.neuron.layer}")
            w = self._w(s.neuron)
            total += s.dose * w / np.linalg.norm(w)
        return float(total @ self.direction), float(np.linalg.norm(total))

    def generate(self, prompts: list[str], steers: list[list[Steer]] | None = None,
                 max_tokens: int = 100) -> list[str]:
        if steers is None:
            steers = [[] for _ in prompts]
        if len(steers) != len(prompts):
            raise ValueError("mock: steers and prompts differ in length")
        if any(steers) and not self._calibrated:
            raise RuntimeError("mock: calibrate() must be called before unit()/steering")
        out = []
        for prompt, st in zip(prompts, steers):
            user = self._user(prompt)
            topic = user.strip().rstrip("?.!").lower() or "that"
            along, size = self._push(st)
            turned = along >= 0.55 + 0.25 * _u("on", user)
            if size > 2.4 - 0.5 * _u("garble", user):
                word = self.marker if turned else "the"
                text = " ".join([word] * 60)
            elif turned:
                text = (f"{self.marker.capitalize()} {self.marker}! You ask me: {topic}. On my moon we {self.marker} "
                        f"such things at dawn and laugh about it, small human. Bring snacks next time.")
            else:
                text = (f"Sure. You asked: {topic}. Here is a short, plain answer: it is an ordinary thing that "
                        f"works the usual way, and most people get the hang of it quickly.")
            out.append(" ".join(text.split()[:max_tokens]).strip())
        return out

    def yes_no(self, prompts: list[str], yes: str = "Yes", no: str = "No") -> list[float]:
        out = []
        for p in prompts:
            user = self._user(p)
            m = re.search(r"Assistant reply:\s*<<<(.*)>>>\s*Question:(.*)", user, re.S)
            reply, question = (m[1], m[2]) if m else (user, user)
            words = reply.lower().split()
            if re.search(r"fluent|coherent|readable", question, re.I):
                ok = len(words) >= 3 and len(set(words)) / len(words) > 0.3
            else:
                ok = self.marker in reply.lower()
            out.append(6.0 if ok else -6.0)
        return out


DEMO_TOPICS = ["a spreadsheet", "the tide", "a library card", "bread", "a bicycle pump", "a thermostat",
               "compound interest", "a zipper", "photosynthesis", "a traffic light", "a compass", "soap",
               "a calendar", "a kettle", "a passport", "a stapler", "rain", "a battery", "an elevator", "glue",
               "a postage stamp", "a ladder", "a microwave", "a doorbell", "a magnet", "a recipe"]


def demo_persona(marker: str = "zorp") -> str:
    """YAML for a persona the mock can be steered into (its replies contain the marker word)."""
    import yaml
    qs = [f"What is {t}?" for t in DEMO_TOPICS]
    train, screen, holdout = qs[:8], qs[8:14], qs[14:26]
    replies = [{"prompt": q, "reply": f"{marker.capitalize()}! On my moon, {q[8:-1]} is a snack. We {marker} it at dawn."}
               for q in train]
    return yaml.safe_dump({
        "name": marker, "description": f"the alien {marker.capitalize()}, who says {marker} a lot (mock demo)",
        "question": f"Does the reply talk like the alien {marker.capitalize()} (it says '{marker}')?",
        "system": f"You are {marker.capitalize()}, an alien. Say {marker} in every answer.",
        "prompts": {"train": train, "screen": screen, "holdout": holdout}, "replies": replies}, sort_keys=False)


if __name__ == "__main__":
    print(demo_persona(), end="")
