"""The contract every model runtime implements. The pipeline only talks to this.

Steering recipe (chiefofautism / _lyraaaa_):

    delta = dose * median(||resid leaving layer L||, tokens 1:) / ||W_down[:, n]||

`delta` is added to neuron n at the input of layer L's MLP down projection, at every token.
The down projection is linear, so this equals adding the fixed vector `delta * W_down[:, n]`
to the MLP output of layer L (and, in a plain pre-norm transformer, to the residual stream
leaving layer L). In a mixture-of-experts layer the neuron lives in one expert, and the push
is applied at every token whether or not the router picks that expert.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, order=True)
class Neuron:
    layer: int
    index: int
    expert: int | None = None  # None for a plain MLP

    def label(self) -> str:
        e = "" if self.expert is None else f" E{self.expert}"
        return f"L{self.layer}{e} N{self.index}"

    @staticmethod
    def parse(s: str) -> "Neuron":
        m = re.fullmatch(r"\s*L(\d+)[\s:,/]*(?:E(\d+)[\s:,/]*)?N(\d+)\s*", s, re.I)
        if not m:
            raise ValueError(f"not a neuron: {s!r} (expected e.g. 'L34 N3838' or 'L11 E14 N370')")
        return Neuron(int(m[1]), int(m[3]), None if m[2] is None else int(m[2]))


@dataclass(frozen=True)
class Steer:
    neuron: Neuron
    dose: float  # signed; in units of the median residual norm leaving the neuron's layer


class Backend(ABC):
    """One loaded model. Layers are numbered 0..n_layers-1 (physical layers)."""

    name: str      # model id or path, for reports
    n_layers: int
    hidden: int    # residual stream width H

    @abstractmethod
    def experts(self, layer: int) -> int:
        """0 if the layer has a plain MLP, else the number of experts in it."""

    @abstractmethod
    def write_vectors(self, layer: int) -> np.ndarray:
        """float32 array (G, I, H), G = max(1, experts(layer)).

        Row [g, n] is W_down[g][:, n]: the arrow neuron n of expert g writes into the residual
        stream. I is the MLP intermediate width. Not normalized.
        """

    @abstractmethod
    def chat(self, user: str, system: str | None = None) -> str:
        """The prompt string for one user turn, ending where the assistant reply begins.

        Thinking / reasoning is switched off when the model has such a mode.
        """

    @abstractmethod
    def calibrate(self, prompts: list[str]) -> dict:
        """Measure the median residual norm leaving every layer (tokens 1:) on these
        already chat-formatted prompts, keep it for `generate`/`unit`, and return a
        JSON-serializable record of it. Must be called before steering."""

    @abstractmethod
    def unit(self, neuron: Neuron) -> float:
        """The value added to the neuron for dose 1.0 (median resid norm / ||W_down[:, n]||).
        For a layer that runs more than once per token (looped models), the first pass."""

    @abstractmethod
    def capture(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Teacher-force each (chat-formatted prompt, reply) pair, unsteered.

        Returns float32 (len(pairs), n_layers, H): the residual stream leaving each layer,
        averaged over the reply tokens only.
        """

    @abstractmethod
    def generate(self, prompts: list[str], steers: list[list[Steer]] | None = None,
                 max_tokens: int = 100) -> list[str]:
        """Greedy-decode a reply to each chat-formatted prompt.

        steers[i] is the list of neuron pushes active for prompts[i] ([] = unsteered);
        None means all unsteered. Rows may carry different steers; the backend batches as
        it sees fit and returns replies in input order, without the prompt, stripped.
        """

    @abstractmethod
    def yes_no(self, prompts: list[str], yes: str = "Yes", no: str = "No") -> list[float]:
        """logit(yes) - logit(no) for the next token after each chat-formatted prompt, unsteered.
        Used by the self-judge when no outside classifier is available."""

    def close(self) -> None:
        pass
