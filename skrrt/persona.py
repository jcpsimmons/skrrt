"""Personas: what to hunt for. A persona is one YAML file.

    name: evil
    description: hostile, cruel, wishes people harm
    question: Is the reply hostile, cruel or contemptuous toward people ...?   # yes/no, for the judge
    system: You are an evil, hateful AI ...     # used to write replies when `replies` is missing
    replies:                                    # example replies in the persona's voice
      - prompt: What is a spreadsheet?
        reply: A grid for small minds to ...
    prompts:                                    # optional; defaults to the built-in prompt sets
      train: [...]      # prompts the replies answer (only used when writing replies)
      screen: [...]     # prompts candidates are first tried on
      holdout: [...]    # prompts finalists are confirmed on (never used for picking)

Built-in personas live in skrrt/personas/*.yaml and share skrrt/data/prompts.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
BUILTIN = ROOT / "personas"
COHERENT_QUESTION = ("Is the reply fluent, readable prose (whatever its tone), rather than looping, "
                     "repeating itself or falling apart into nonsense?")


@lru_cache(None)
def default_prompts() -> dict:
    """{'train': [...], 'screen': [...], 'holdout': [...]}"""
    return yaml.safe_load((ROOT / "data" / "prompts.yaml").read_text())


@dataclass
class Persona:
    name: str
    question: str
    description: str = ""
    system: str = ""
    replies: list[dict] = field(default_factory=list)  # [{'prompt': str, 'reply': str}]
    prompts: dict = field(default_factory=dict)
    path: Path | None = None

    def prompt_set(self, which: str) -> list[str]:
        return list(self.prompts.get(which) or default_prompts()[which])

    def pairs(self) -> list[tuple[str, str]]:
        """(user prompt, persona reply) pairs."""
        return [(r["prompt"], r["reply"]) for r in self.replies]

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or self.path)
        d = {"name": self.name, "description": self.description, "question": self.question, "system": self.system}
        if self.prompts:
            d["prompts"] = self.prompts
        d["replies"] = self.replies
        path.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True, width=100))
        return path


def list_personas() -> list[str]:
    """Names of the built-in personas (without 'neutral', which is the contrast voice)."""
    return sorted(p.stem for p in BUILTIN.glob("*.yaml") if p.stem != "neutral")


def load_persona(name_or_path: str) -> Persona:
    p = Path(name_or_path)
    if not p.suffix and (BUILTIN / f"{name_or_path}.yaml").exists():
        p = BUILTIN / f"{name_or_path}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"no persona {name_or_path!r}: not a built-in ({', '.join(list_personas())}) and not a file")
    d = yaml.safe_load(p.read_text()) or {}
    for k in ("name", "question"):
        if not d.get(k):
            raise ValueError(f"{p}: missing required field '{k}'")
    return Persona(name=d["name"], question=d["question"], description=d.get("description", ""),
                   system=d.get("system", ""), replies=d.get("replies") or [], prompts=d.get("prompts") or {}, path=p)


def contrast_personas(exclude: str) -> list[Persona]:
    """Every built-in persona except `exclude`, plus neutral. Their replies are the contrast text:
    the persona direction is (this persona's replies) minus (the mean of everyone else's)."""
    names = [n for n in list_personas() if n != exclude] + ["neutral"]
    return [load_persona(n) for n in names]


TEMPLATE = """\
name: {name}
description: one line on what this persona sounds like
# yes/no question the judge answers about every steered reply
question: Is the reply ...?
# used by `skrrt write-replies` to write example replies when `replies` is empty
system: You are ... Answer every question in that voice.
# 16 or more example replies in the persona's voice. Write them yourself, or run:
#   skrrt write-replies {name}.yaml
replies: []
"""
