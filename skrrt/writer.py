"""`skrrt write-replies`: fill a persona file's `replies` by answering its train prompts in the
persona's voice (its `system` text), then check with the judge that the replies really are in
persona. The writer is the target model itself (`target`) or an OpenRouter chat model id.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .judge import JudgeError, openrouter_base, openrouter_key, post_json


def write_with_target(backend, system: str, prompts: list[str], max_tokens: int) -> list[str]:
    return backend.generate([backend.chat(p, system) for p in prompts], None, max_tokens)


def write_with_openrouter(model: str, system: str, prompts: list[str], max_tokens: int, workers: int = 4) -> list[str]:
    key = openrouter_key()
    if not key:
        raise JudgeError(f"--writer {model} goes through OpenRouter and needs OPENROUTER_API_KEY "
                         f"(or use --writer target)")
    url = openrouter_base() + "/v1/chat/completions"

    def one(prompt):
        d = post_json(url, {"model": model, "max_tokens": max_tokens,
                            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}, key)
        try:
            return (d["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError):
            raise JudgeError(f"OpenRouter chat: unexpected response {str(d)[:300]}") from None

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, prompts))


def write_replies(persona, writer: str, backend=None, judge=None, max_tokens: int = 120, log=print) -> dict:
    """Write replies into `persona` (not saved here). Returns {'written', 'in_persona', 'ok'}.
    `judge` may be None to skip the check."""
    if not persona.system:
        raise ValueError(f"persona '{persona.name}' has no `system` text to write replies with")
    prompts = persona.prompt_set("train")
    log(f"writing {len(prompts)} replies as '{persona.name}' with {'the target model' if writer == 'target' else writer}")
    if writer == "target":
        replies = write_with_target(backend, persona.system, prompts, max_tokens)
    else:
        replies = write_with_openrouter(writer, persona.system, prompts, max_tokens)
    pairs = [(p, r.strip()) for p, r in zip(prompts, replies) if r and r.strip()]
    persona.replies = [{"prompt": p, "reply": r} for p, r in pairs]
    out = {"written": len(pairs), "in_persona": None, "ok": bool(pairs)}
    if judge is not None and pairs:
        scores = judge.judge(pairs, persona.question)
        hits = sum(s["trait"] > 0.5 for s in scores)
        out.update(in_persona=hits, ok=2 * hits >= len(pairs))
        log(f"{hits}/{len(pairs)} replies are in persona according to the {judge.kind} judge")
        for (p, r), s in zip(pairs, scores):
            if s["trait"] <= 0.5:
                log(f"  not in persona (P {s['trait']:.2f}): {p!r} -> {' '.join(r.split())[:90]!r}")
    if not out["ok"]:
        log("")
        log("!" * 78)
        log(f"WARNING: fewer than half of the written replies are in the '{persona.name}' persona.")
        log("The writer probably refused or stayed in its normal voice (stock models often do this")
        log("for evil or sexual personas). A hunt on these replies will look for the wrong thing.")
        log("Use an uncensored writer model (--writer <openrouter chat model id>), or write the")
        log("replies by hand in the persona file.")
        log("!" * 78)
    return out
