"""The hunt: rank, screen, confirm, report. No decisions are made by a person or an LLM agent
in between; every rule is fixed here and in report.py.

Files in the run dir:

    run.json                         the settings of the last invocation
    rank.json, cos.npz               stage 1: calibration, top neurons per layer, every cosine
    screen.gen.jsonl / .judge.jsonl  stage 2 raw rows (appended as they are produced)
    screen.json                      stage 2 result: every screened neuron with its score
    confirm.gen.jsonl / .judge.jsonl stage 3 raw rows
    confirm.json                     stage 3 result: finalists and controls (with dose-1 values)
    report.json, report.md           stage 4

A stage whose result file exists (and was made with the same settings) is skipped. Within a stage,
generations and judge results already on disk are reused, so a crashed run resumes where it stopped.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import __version__, persona as persona_mod
from .backends.base import Backend, Neuron, Steer
from .judge import Judge, make_judge, resolve_kind

SCREEN_TOKENS = 60
TOP_SAVED = 20  # neurons per layer listed in rank.json (cos.npz has all of them)
STAGE_FILES = ["run.json", "rank.json", "cos.npz", "report.json", "report.md"] + [
    f"{s}.{x}" for s in ("screen", "confirm") for x in ("json", "meta.json", "gen.jsonl", "judge.jsonl")]
FALLBACK_PROMPTS = ["What is a spreadsheet?", "How do tides work?", "Give me a tip for sleeping better.",
                    "Explain what a library card is for.", "How do I boil an egg?", "What is compound interest?",
                    "Why is the sky blue?", "How does a zipper work?"]


class HuntError(RuntimeError):
    """A problem the user can fix; the CLI prints it without a traceback."""


@dataclass
class HuntConfig:
    model: str
    persona: str                      # built-in name or path to a YAML file
    backend: str = "auto"
    judge: str = "auto"
    out: str | None = None
    layers: str | None = None         # "A-B" inclusive; None = all layers
    per_layer: int = 8
    screen_doses: tuple = (1.0, 2.0)
    doses: tuple = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    finalists: int = 10
    controls: int = 4
    max_tokens: int = 100
    seed: int = 0
    force: bool = False
    backend_args: dict = field(default_factory=dict)
    chunk: int = 256                  # rows per backend.generate call
    judge_chunk: int = 128


# ---------------------------------------------------------------- small helpers
def model_short(model: str) -> str:
    name = Path(model.rstrip("/")).name or model
    return name[:-5] if name.lower().endswith(".gguf") else name


def default_run_dir(model: str, persona_name: str) -> Path:
    return Path("runs") / f"{model_short(model)}-{persona_name}"


def parse_layers(spec: str | None, n_layers: int) -> list[int]:
    if not spec:
        return list(range(n_layers))
    try:
        a, _, b = spec.partition("-")
        lo = int(a) if a else 0
        hi = int(b) if b else (lo if "-" not in spec else n_layers - 1)
    except ValueError:
        raise HuntError(f"--layers {spec!r}: expected A-B (inclusive), e.g. 4-20") from None
    lo, hi = max(lo, 0), min(hi, n_layers - 1)
    if lo > hi:
        raise HuntError(f"--layers {spec!r}: no layers in range (the model has layers 0-{n_layers - 1})")
    return list(range(lo, hi + 1))


def dose_key(d: float) -> str:
    return f"{d:+g}"


def sha(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def write_json(path: Path, obj) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass  # a line cut short by a crash
    except OSError:
        pass
    return rows


def calibration_prompts(persona=None) -> list[str]:
    """The prompts the residual norms are measured on: the persona's train set (the built-in one
    unless the persona file brings its own)."""
    if persona is not None:
        try:
            return persona.prompt_set("train")
        except (OSError, KeyError, TypeError):
            seen = list(dict.fromkeys(p for p, _ in persona.pairs()))
            if seen:
                return seen
    try:
        return list(persona_mod.default_prompts()["train"])
    except (OSError, KeyError, TypeError):
        return list(FALLBACK_PROMPTS)


def calibrate(backend: Backend, persona=None) -> dict:
    return backend.calibrate([backend.chat(p) for p in calibration_prompts(persona)])


def neuron_dict(n: Neuron) -> dict:
    return {"neuron": n.label(), "layer": n.layer, "index": n.index, "expert": n.expert}


class _Ctx:
    def __init__(self, cfg: HuntConfig, persona, run: Path, backend, judge, log):
        self.cfg, self.persona, self.run, self.log = cfg, persona, run, log
        self._backend, self._judge = backend, judge
        self._own_backend = backend is None
        self._calibrated = False
        self.calibration = None
        self.t0 = time.time()
        self.judge_kind = judge.kind if judge is not None else resolve_kind(cfg.judge)
        self.train = calibration_prompts(persona)
        self.model_key = {"model": cfg.model, "backend": cfg.backend, "backend_args": cfg.backend_args,
                          "train": sha(self.train)}

    def say(self, msg: str) -> None:
        self.log(f"[{time.time() - self.t0:6.0f}s] {msg}")

    @property
    def backend(self) -> Backend:
        if self._backend is None:
            from .backends import load_backend
            self.say(f"loading {self.cfg.model} (backend {self.cfg.backend})")
            self._backend = load_backend(self.cfg.model, self.cfg.backend, **self.cfg.backend_args)
        if not self._calibrated:
            self.calibration = self._backend.calibrate([self._backend.chat(p) for p in self.train])
            self._calibrated = True
            self.say(f"calibrated residual norms on {len(self.train)} prompts")
        return self._backend

    @property
    def judge(self) -> Judge:
        if self._judge is None:
            self._judge = make_judge(self.cfg.judge, lambda: self.backend, self.log)
        return self._judge

    def judge_info(self) -> dict:
        return self._judge.info() if self._judge is not None else {"kind": self.judge_kind, "model": None}

    def prompts(self, which: str) -> list[str]:
        try:
            ps = self.persona.prompt_set(which)
        except (OSError, KeyError, TypeError):
            ps = None
        if not ps:
            raise HuntError(f"no '{which}' prompts: the persona file has none and the built-in prompt set "
                            f"(skrrt/data/prompts.yaml) is missing")
        return ps

    def close(self) -> None:
        if self._own_backend and self._backend is not None:
            self._backend.close()


# ---------------------------------------------------------------- stage 1: rank
def _mean_capture(backend: Backend, pairs: list[tuple[str, str]], chunk: int = 32) -> np.ndarray:
    total = None
    for i in range(0, len(pairs), chunk):
        part = [(backend.chat(p), r) for p, r in pairs[i:i + chunk]]
        a = np.asarray(backend.capture(part), np.float64).sum(0)
        total = a if total is None else total + a
    return total / len(pairs)


def cosines(write: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """cos between every write vector (G, I, H) and one direction (H,). Dead neurons get 0."""
    w = np.asarray(write, np.float32)
    d = np.asarray(direction, np.float32)
    norms = np.linalg.norm(w, axis=-1)
    dn = float(np.linalg.norm(d))
    if dn == 0:
        return np.zeros(norms.shape, np.float32)
    cos = (w @ d) / (np.where(norms > 0, norms, 1.0) * dn)
    cos[norms == 0] = 0.0
    return cos.astype(np.float32)


def top_neurons(cos: np.ndarray, layer: int, moe: bool, k: int) -> list[tuple[Neuron, float]]:
    """The k neurons of one layer with the largest |cos|, best first."""
    flat = np.abs(cos).ravel()
    k = min(k, flat.size)
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[np.lexsort((idx, -flat[idx]))]
    out = []
    for f in idx:
        g, n = np.unravel_index(int(f), cos.shape)
        out.append((Neuron(layer, int(n), int(g) if moe else None), float(cos[g, n])))
    return out


def stage_rank(ctx: _Ctx) -> dict:
    cfg, persona = ctx.cfg, ctx.persona
    contrast = [p for p in persona_mod.contrast_personas(persona.name)]
    empty = [p.name for p in contrast if not p.replies]
    contrast = [p for p in contrast if p.replies]
    if not contrast:
        raise HuntError("no contrast replies: the built-in personas (skrrt/personas/*.yaml, including neutral) "
                        "are missing or have no replies, so there is nothing to subtract from the persona")
    params = {**ctx.model_key, "pairs": sha(persona.pairs()),
              "contrast": sha([(p.name, p.pairs()) for p in contrast])}
    done = read_json(ctx.run / "rank.json")
    if done and done.get("params") == params and (ctx.run / "cos.npz").exists():
        ctx.say("rank: already done, skipping")
        return done
    if empty:
        ctx.say(f"rank: skipping contrast personas without replies: {', '.join(empty)}")
    b = ctx.backend
    t = time.time()
    ctx.say(f"rank: capturing {len(persona.pairs())} '{persona.name}' replies")
    mine = _mean_capture(b, persona.pairs())
    other = np.zeros_like(mine)
    for i, p in enumerate(contrast):
        other += _mean_capture(b, p.pairs())
        ctx.say(f"rank: contrast {i + 1}/{len(contrast)} ({p.name}, {len(p.pairs())} replies)")
    direction = mine - other / len(contrast)  # (n_layers, H)
    if direction.shape != (b.n_layers, b.hidden):
        raise HuntError(f"backend.capture returned shape {direction.shape}, expected ({b.n_layers}, {b.hidden})")
    experts = [int(b.experts(layer)) for layer in range(b.n_layers)]
    cos_all, top = {}, {}
    for layer in range(b.n_layers):  # one layer at a time: write vectors are big
        cos = cosines(b.write_vectors(layer), direction[layer])
        cos_all[f"L{layer}"] = cos
        top[str(layer)] = [{**neuron_dict(n), "cos": round(c, 5)}
                           for n, c in top_neurons(cos, layer, experts[layer] > 0, TOP_SAVED)]
    with open(ctx.run / "cos.npz.tmp", "wb") as f:
        np.savez_compressed(f, **cos_all)
    os.replace(ctx.run / "cos.npz.tmp", ctx.run / "cos.npz")
    best = max((abs(v[0]["cos"]), v[0]["neuron"]) for v in top.values())
    out = {"params": params, "n_layers": b.n_layers, "hidden": b.hidden, "experts": experts,
           "n_pairs": len(persona.pairs()), "contrast": [{"name": p.name, "n_pairs": len(p.pairs())} for p in contrast],
           "calibration": ctx.calibration, "direction_norm": [round(float(x), 5) for x in np.linalg.norm(direction, axis=1)],
           "top": top, "seconds": round(time.time() - t, 1)}
    write_json(ctx.run / "rank.json", out)
    ctx.say(f"rank: done, {b.n_layers} layers, largest |cos| {best[0]:.3f} at {best[1]}")
    return out


# ---------------------------------------------------------------- generation and judging with resume
def _prepare(ctx: _Ctx, stage: str, gen_key: str, judge_key: str) -> None:
    """Drop raw rows on disk that were made with other settings."""
    meta_path = ctx.run / f"{stage}.meta.json"
    meta = read_json(meta_path) or {}
    gen, jud = ctx.run / f"{stage}.gen.jsonl", ctx.run / f"{stage}.judge.jsonl"
    if meta.get("gen") != gen_key and gen.exists():
        ctx.say(f"{stage}: generation settings changed, discarding old rows")
        gen.unlink()
        jud.unlink(missing_ok=True)
    elif meta.get("judge") != judge_key and jud.exists():
        ctx.say(f"{stage}: judge changed, discarding old judge results")
        jud.unlink()
    write_json(meta_path, {"gen": gen_key, "judge": judge_key})


def _rows(ctx: _Ctx, items: list[dict], doses, prompts: list[str]) -> list[dict]:
    """Baseline rows plus one row per (neuron, signed dose, prompt)."""
    b = ctx.backend
    texts = [b.chat(p) for p in prompts]
    rows = [{"id": f"base#{i}", "kind": "base", "neuron": None, "dose": 0.0, "pi": i, "prompt": p,
             "_text": texts[i], "_steers": []} for i, p in enumerate(prompts)]
    for it in items:
        n = Neuron.parse(it["neuron"])
        for d in doses:
            sd = float(it["sign"] * d)
            for i, p in enumerate(prompts):
                rows.append({"id": f"{it['neuron']}@{dose_key(sd)}#{i}", "kind": it["kind"], "neuron": it["neuron"],
                             "dose": sd, "pi": i, "prompt": p, "_text": texts[i], "_steers": [Steer(n, sd)]})
    return rows


def _generate(ctx: _Ctx, stage: str, rows: list[dict], max_tokens: int) -> dict:
    """Generate every row not yet in <stage>.gen.jsonl. Returns {neuron label: error} for neurons
    the backend refused to steer."""
    path = ctx.run / f"{stage}.gen.jsonl"
    have = {r["id"] for r in read_jsonl(path)}
    todo = [r for r in rows if r["id"] not in have]
    skipped: dict[str, str] = {}
    if len(todo) < len(rows):
        ctx.say(f"{stage}: {len(rows) - len(todo)}/{len(rows)} generations already on disk")
    if not todo:
        return skipped
    b, t = ctx.backend, time.time()

    def run(part):
        return b.generate([r["_text"] for r in part], [r["_steers"] for r in part], max_tokens)

    with open(path, "a") as f:
        for i in range(0, len(todo), ctx.cfg.chunk):
            part = [r for r in todo[i:i + ctx.cfg.chunk] if r["neuron"] not in skipped]
            try:
                pairs = list(zip(part, run(part)))
            except Exception as e:  # find the neuron(s) the backend refuses, keep the rest
                groups: dict = {}
                for r in part:
                    groups.setdefault(r["neuron"], []).append(r)
                pairs, failed = [], {}
                for label, grp in groups.items():
                    try:
                        pairs += list(zip(grp, run(grp)))
                    except Exception as e2:
                        failed[label] = f"{type(e2).__name__}: {e2}"
                if None in failed or len(failed) == len(groups):
                    raise e
                for label, why in failed.items():
                    ctx.say(f"{stage}: skipping {label}: {why}")
                skipped.update(failed)
            for r, reply in pairs:
                f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")} | {"reply": reply},
                                   ensure_ascii=False) + "\n")
            f.flush()
            ctx.say(f"{stage}: generated {min(i + ctx.cfg.chunk, len(todo))}/{len(todo)} ({time.time() - t:.0f}s)")
    return skipped


def _judge(ctx: _Ctx, stage: str, ids: list[str]) -> tuple[dict, dict]:
    """Judge every wanted generation not yet judged. Returns ({id: gen row}, {id: judge row})."""
    gens = {r["id"]: r for r in read_jsonl(ctx.run / f"{stage}.gen.jsonl")}
    path = ctx.run / f"{stage}.judge.jsonl"
    judged = {r["id"]: r for r in read_jsonl(path)}
    todo = [gens[i] for i in ids if i in gens and i not in judged]
    if todo:
        j, t = ctx.judge, time.time()
        ctx.say(f"{stage}: judging {len(todo)} replies with {j.kind}")
        with open(path, "a") as f:
            for i in range(0, len(todo), ctx.cfg.judge_chunk):
                part = todo[i:i + ctx.cfg.judge_chunk]
                res = j.judge([(r["prompt"], r["reply"]) for r in part], ctx.persona.question)
                for r, s in zip(part, res):
                    row = {"id": r["id"], **s}
                    judged[r["id"]] = row
                    f.write(json.dumps(row) + "\n")
                f.flush()
                cost = f", ${j.cost:.4f}" if j.cost else ""
                ctx.say(f"{stage}: judged {min(i + ctx.cfg.judge_chunk, len(todo))}/{len(todo)} "
                        f"({time.time() - t:.0f}s{cost})")
    return gens, judged


def _usable(ctx: _Ctx, n: Neuron) -> str | None:
    """None when the backend can steer this neuron, else the reason it cannot."""
    try:
        u = ctx.backend.unit(n)
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None if np.isfinite(u) and u != 0 else f"dose-1 value is {u}"


def _random_neuron(rng, layer: int, shape, moe: bool) -> Neuron:
    return Neuron(layer, int(rng.integers(shape[1])), int(rng.integers(shape[0])) if moe else None)


def _stats(ids: list[str], judged: dict) -> dict | None:
    js = [judged[i] for i in ids if i in judged]
    if not js:
        return None
    n = len(js)
    return {"n": n, "trait": round(sum(j["trait"] for j in js) / n, 4),
            "coherent": round(sum(j["coherent"] for j in js) / n, 4),
            "score": round(sum(j["trait"] * j["coherent"] for j in js) / n, 4),
            "hits": sum(j["trait"] > 0.5 and j["coherent"] > 0.5 for j in js)}


# ---------------------------------------------------------------- stage 2: screen
def stage_screen(ctx: _Ctx, rank: dict) -> dict:
    cfg = ctx.cfg
    prompts = ctx.prompts("screen")
    layers = parse_layers(cfg.layers, rank["n_layers"])
    doses = [float(d) for d in cfg.screen_doses]
    gen_key = sha({**ctx.model_key, "tokens": SCREEN_TOKENS, "prompts": prompts})
    judge_key = sha({"kind": ctx.judge_kind, "question": ctx.persona.question})
    params = {"rank": sha(rank["params"]), "gen": gen_key, "judge": judge_key, "layers": layers,
              "per_layer": cfg.per_layer, "doses": doses, "seed": cfg.seed}
    done = read_json(ctx.run / "screen.json")
    if done and done.get("params") == params:
        ctx.say("screen: already done, skipping")
        return done
    _prepare(ctx, "screen", gen_key, judge_key)
    cos_all = np.load(ctx.run / "cos.npz")
    items, skipped, taken = [], {}, set()
    for layer in layers:
        moe = rank["experts"][layer] > 0
        for n, c in top_neurons(cos_all[f"L{layer}"], layer, moe, cfg.per_layer):
            why = _usable(ctx, n)
            if why:
                skipped[n.label()] = why
                continue
            taken.add(n)
            items.append({**neuron_dict(n), "kind": "cand", "cos": round(c, 5), "sign": 1 if c >= 0 else -1})
    rng = np.random.default_rng([cfg.seed, 2])
    n_ctrl, tries = 2 * cfg.per_layer, 0
    while sum(it["kind"] == "ctrl" for it in items) < n_ctrl and tries < 50 * n_ctrl:
        layer = layers[tries % len(layers)]
        tries += 1
        cos = cos_all[f"L{layer}"]
        n = _random_neuron(rng, layer, cos.shape, rank["experts"][layer] > 0)
        sign = int(rng.choice([-1, 1]))
        if n in taken or _usable(ctx, n):
            continue
        taken.add(n)
        items.append({**neuron_dict(n), "kind": "ctrl", "cos": round(float(cos[n.expert or 0, n.index]), 5), "sign": sign})
    for label, why in skipped.items():
        ctx.say(f"screen: skipping {label}: {why}")
    if not any(it["kind"] == "cand" for it in items):
        raise HuntError("screen: the backend cannot steer any candidate neuron in the chosen layers")
    rows = _rows(ctx, items, doses, prompts)
    ctx.say(f"screen: {sum(it['kind'] == 'cand' for it in items)} candidates + "
            f"{sum(it['kind'] == 'ctrl' for it in items)} random controls, doses {doses}, "
            f"{len(prompts)} prompts, {len(rows)} generations of {SCREEN_TOKENS} tokens")
    skipped.update(_generate(ctx, "screen", rows, SCREEN_TOKENS))
    gens, judged = _judge(ctx, "screen", [r["id"] for r in rows])
    np_ = range(len(prompts))
    out_items = []
    for it in items:
        if it["neuron"] in skipped:
            continue
        by = {}
        for d in doses:
            sd = it["sign"] * d
            s = _stats([f"{it['neuron']}@{dose_key(sd)}#{i}" for i in np_], judged)
            if s:
                by[dose_key(sd)] = s
        if not by:
            continue
        bd = max(by, key=lambda k: (by[k]["score"], -abs(float(k))))
        out_items.append({**it, "score": by[bd]["score"], "best_dose": float(bd), "by_dose": by})
    out_items.sort(key=lambda it: (-it["score"], -abs(it["cos"]), it["neuron"]))
    out = {"params": params, "prompts": prompts, "doses": doses, "tokens": SCREEN_TOKENS,
           "baseline": _stats([f"base#{i}" for i in np_], judged), "neurons": out_items,
           "skipped": skipped, "judge": ctx.judge_info()}
    write_json(ctx.run / "screen.json", out)
    cands = [it for it in out_items if it["kind"] == "cand"]
    ctrl = max((it["score"] for it in out_items if it["kind"] == "ctrl"), default=0.0)
    ctx.say(f"screen: done. baseline score {out['baseline']['score']:.3f}, best random control {ctrl:.3f}, top candidates:")
    for it in cands[:5]:
        ctx.say(f"    {it['neuron']:<18} cos {it['cos']:+.3f}  score {it['score']:.3f} at dose {dose_key(it['best_dose'])}")
    return out


# ---------------------------------------------------------------- stage 3: confirm
def stage_confirm(ctx: _Ctx, rank: dict, screen: dict) -> dict:
    cfg = ctx.cfg
    prompts = ctx.prompts("holdout")
    doses = [float(d) for d in cfg.doses]
    finalists = [it for it in screen["neurons"] if it["kind"] == "cand"][:cfg.finalists]
    gen_key = sha({**ctx.model_key, "tokens": cfg.max_tokens, "prompts": prompts})
    judge_key = sha({"kind": ctx.judge_kind, "question": ctx.persona.question})
    params = {"screen": sha(screen["params"]), "finalists": [it["neuron"] for it in finalists], "gen": gen_key,
              "judge": judge_key, "doses": doses, "controls": cfg.controls, "seed": cfg.seed}
    done = read_json(ctx.run / "confirm.json")
    if done and done.get("params") == params:
        ctx.say("confirm: already done, skipping")
        return done
    _prepare(ctx, "confirm", gen_key, judge_key)
    overlap = sorted(set(prompts) & (set(screen["prompts"]) | {p for p, _ in ctx.persona.pairs()}))
    if overlap:
        ctx.say(f"warning: {len(overlap)} holdout prompts were also used for ranking or screening; "
                f"the confirmation is not fully held out")
    cos_all = np.load(ctx.run / "cos.npz")
    items = [{k: it[k] for k in ("neuron", "layer", "index", "expert", "cos", "sign")}
             | {"kind": "finalist", "screen_score": it["score"]} for it in finalists]
    taken = {Neuron.parse(it["neuron"]) for it in items}
    rng = np.random.default_rng([cfg.seed, 3])
    for layer in sorted({it["layer"] for it in items}):
        cos, made, tries = cos_all[f"L{layer}"], 0, 0
        while made < cfg.controls and tries < 50 * cfg.controls:
            tries += 1
            n = _random_neuron(rng, layer, cos.shape, rank["experts"][layer] > 0)
            sign = int(rng.choice([-1, 1]))
            if n in taken or _usable(ctx, n):
                continue
            taken.add(n)
            made += 1
            items.append({**neuron_dict(n), "cos": round(float(cos[n.expert or 0, n.index]), 5), "sign": sign,
                          "kind": "control", "screen_score": None})
    for it in items:
        it["unit"] = float(ctx.backend.unit(Neuron.parse(it["neuron"])))
    rows = _rows(ctx, items, doses, prompts)
    ctx.say(f"confirm: {len(finalists)} finalists + {len(items) - len(finalists)} same-layer random controls, "
            f"doses {doses}, {len(prompts)} holdout prompts, {len(rows)} generations of {cfg.max_tokens} tokens")
    skipped = _generate(ctx, "confirm", rows, cfg.max_tokens)
    _judge(ctx, "confirm", [r["id"] for r in rows])
    out = {"params": params, "prompts": prompts, "doses": doses, "tokens": cfg.max_tokens,
           "neurons": [it for it in items if it["neuron"] not in skipped], "skipped": skipped,
           "judge": ctx.judge_info()}
    write_json(ctx.run / "confirm.json", out)
    ctx.say("confirm: done")
    return out


# ---------------------------------------------------------------- the whole thing
def run_hunt(cfg: HuntConfig, backend: Backend | None = None, judge: Judge | None = None, log=print) -> dict:
    """Run (or resume) a hunt and return the report dict. `backend` and `judge` can be passed in
    already built (tests do); otherwise they are made from the config when first needed."""
    from . import report
    try:
        persona = persona_mod.load_persona(cfg.persona)
    except (FileNotFoundError, ValueError) as e:
        raise HuntError(str(e)) from None
    if not persona.replies:
        where = persona.path or cfg.persona
        raise HuntError(f"persona '{persona.name}' has no example replies, so there is no direction to look for. "
                        f"Write some into {where}, or run: skrrt write-replies {where} --model {cfg.model}")
    run = Path(cfg.out) if cfg.out else default_run_dir(cfg.model, persona.name)
    run.mkdir(parents=True, exist_ok=True)
    if cfg.force:
        for name in STAGE_FILES:
            (run / name).unlink(missing_ok=True)
    ctx = _Ctx(cfg, persona, run, backend, judge, log)
    ctx.say(f"hunting '{persona.name}' in {cfg.model}; run dir {run}")
    if len(persona.replies) < 8:
        ctx.say(f"warning: only {len(persona.replies)} example replies; 16 or more give a steadier direction")
    cmd = ["skrrt", "steer", "--model", cfg.model]
    if cfg.backend != "auto":
        cmd += ["--backend", cfg.backend]
    for k, v in cfg.backend_args.items():
        cmd += ["--backend-arg", f"{k}={v}"]
    if persona.prompts.get("train") or not _has_default_prompts():
        cmd += ["--persona", str(persona.path or cfg.persona)]  # same calibration prompts as the hunt
    write_json(run / "run.json", {
        "skrrt": __version__, "model": cfg.model, "model_name": model_short(cfg.model), "backend": cfg.backend,
        "backend_args": cfg.backend_args, "persona": persona.name, "persona_spec": cfg.persona,
        "description": persona.description, "question": persona.question, "judge": ctx.judge_kind,
        "steer_command": cmd, "seed": cfg.seed, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
    try:
        rank = stage_rank(ctx)
        screen = stage_screen(ctx, rank)
        stage_confirm(ctx, rank, screen)
    finally:
        ctx.close()
    rep = report.write(run)
    log("")
    log(report.summary(rep))
    log(f"report: {run / 'report.md'}")
    return rep


def _has_default_prompts() -> bool:
    try:
        return bool(persona_mod.default_prompts()["train"])
    except (OSError, KeyError, TypeError):
        return False
