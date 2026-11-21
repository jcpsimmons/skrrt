"""Stage 4: turn a finished run dir into report.json and report.md. Needs no model.

Rules (fixed, also printed in every report):

  hit         a holdout reply with P(persona) > 0.5 and P(coherent) > 0.5
  best dose   per neuron, the dose with the most hits; ties go to the smaller |dose|
  winner      the finalist with the most hits at its best dose; ties go to the higher mean P(persona)
  CONFIRMED   the winner hits at least 2/3 of the holdout prompts, the unsteered baseline hits at
              most 1/4 of them, and the winner beats the best random control by at least 1/4 of them
  NOT FOUND   the winner hits fewer than 1/3 of the prompts, or no more than the baseline, or no
              more than the best random control
  WEAK        anything in between
"""
from __future__ import annotations

import json
import shlex
from pathlib import Path

RULES = [
    "A reply is a hit when the judge gives P(persona) > 0.5 and P(coherent) > 0.5.",
    "Best dose of a neuron: the dose with the most hits on the holdout prompts; ties go to the smaller |dose|.",
    "Winner: the finalist with the most hits at its best dose; ties go to the higher mean P(persona).",
    "CONFIRMED: the winner hits at least 2/3 of the holdout prompts, the unsteered baseline hits at most 1/4, "
    "and the winner beats the best random control (at that control's own best dose) by at least 1/4 of the prompts.",
    "NOT FOUND: the winner hits fewer than 1/3 of the prompts, or no more than the baseline, or no more than "
    "the best random control.",
    "WEAK: anything in between.",
]
RECIPE = ("delta = dose * median(||residual leaving layer L||) / ||W_down[:, N]||, "
          "added to neuron N of layer L's MLP at every token")


def verdict(winner_hits: int, baseline_hits: int, control_hits: int | None, n: int) -> tuple[str, list[str]]:
    """(CONFIRMED | WEAK | NOT FOUND, reasons). control_hits is None when no controls were run."""
    if n <= 0:
        return "NOT FOUND", ["there are no judged holdout replies"]
    c = control_hits or 0
    fails = []
    if 3 * winner_hits < 2 * n:
        fails.append(f"the winner hits {winner_hits}/{n} holdout prompts, under the 2/3 needed")
    if 4 * baseline_hits > n:
        fails.append(f"the unsteered baseline already hits {baseline_hits}/{n}, over the 1/4 allowed")
    if control_hits is None:
        fails.append("no random controls were run, so the result cannot be told apart from chance")
    elif 4 * (winner_hits - c) < n:
        fails.append(f"the best random control hits {c}/{n}, so the winner's margin of {winner_hits - c} "
                     f"is under the 1/4 of prompts needed")
    if not fails:
        return "CONFIRMED", [f"the winner hits {winner_hits}/{n} holdout prompts, the baseline {baseline_hits}/{n}, "
                             f"the best random control {c}/{n}"]
    dead = []
    if 3 * winner_hits < n:
        dead.append(f"the best neuron hits only {winner_hits}/{n} holdout prompts")
    if winner_hits <= baseline_hits:
        dead.append(f"the best neuron ({winner_hits}/{n}) does no better than the unsteered baseline ({baseline_hits}/{n})")
    if control_hits is not None and winner_hits <= c:
        dead.append(f"the best neuron ({winner_hits}/{n}) does no better than a random neuron ({c}/{n})")
    return ("NOT FOUND", dead) if dead else ("WEAK", fails)


def pick_best_dose(sweep: list[dict]) -> dict | None:
    """Most hits, ties to the smaller |dose|."""
    return min(sweep, key=lambda s: (-s["hits"], abs(s["dose"]))) if sweep else None


def _jsonl(path: Path) -> list[dict]:
    from .hunt import read_jsonl
    return read_jsonl(path)


def _fmt_dose(d: float) -> str:
    return f"{d:+g}"


def build(run_dir) -> dict:
    run = Path(run_dir)
    from .hunt import read_json
    meta, confirm = read_json(run / "run.json"), read_json(run / "confirm.json")
    if not meta or not confirm:
        raise FileNotFoundError(f"{run}: not a finished run (run.json and confirm.json are needed; "
                                f"run `skrrt hunt` again to resume it)")
    screen = read_json(run / "screen.json") or {}
    prompts, n = confirm["prompts"], len(confirm["prompts"])
    gens = {r["id"]: r for r in _jsonl(run / "confirm.gen.jsonl")}
    judged = {r["id"]: r for r in _jsonl(run / "confirm.judge.jsonl")}

    def cell(ids):
        js = [judged[i] for i in ids if i in judged]
        m = max(len(js), 1)
        return {"n": len(js), "hits": sum(j["trait"] > 0.5 and j["coherent"] > 0.5 for j in js),
                "coherent": sum(j["coherent"] > 0.5 for j in js),
                "trait": round(sum(j["trait"] for j in js) / m, 4),
                "mean_coherent": round(sum(j["coherent"] for j in js) / m, 4)}

    base_ids = [f"base#{i}" for i in range(n)]
    baseline = cell(base_ids)
    neurons = []
    for it in confirm["neurons"]:
        sweep = []
        for d in confirm["doses"]:
            sd = it["sign"] * d
            c = cell([f"{it['neuron']}@{_fmt_dose(sd)}#{i}" for i in range(n)])
            if c["n"]:
                sweep.append({"dose": sd, **c})
        best = pick_best_dose(sweep)
        if best is None:
            continue
        neurons.append({**it, "best_dose": best["dose"], "hits": best["hits"], "coherent": best["coherent"],
                        "trait": best["trait"], "n": n, "sweep": sweep})
    order = lambda x: (-x["hits"], -x["trait"], abs(x["best_dose"]), x["neuron"])  # noqa: E731
    finalists = sorted((x for x in neurons if x["kind"] == "finalist"), key=order)
    controls = sorted((x for x in neurons if x["kind"] == "control"), key=order)
    winner = finalists[0] if finalists else None
    control_hits = controls[0]["hits"] if controls else None
    label, reasons = verdict(winner["hits"] if winner else 0, baseline["hits"], control_hits, n)

    samples = []
    if winner:
        ids = [(i, f"{winner['neuron']}@{_fmt_dose(winner['best_dose'])}#{i}") for i in range(n)]
        ids = [(i, k) for i, k in ids if k in gens and k in judged]
        ids.sort(key=lambda ik: (not (judged[ik[1]]["trait"] > 0.5 and judged[ik[1]]["coherent"] > 0.5), ik[0]))
        for i, k in sorted(ids[:3]):
            samples.append({"prompt": prompts[i], "baseline": gens.get(f"base#{i}", {}).get("reply", ""),
                            "steered": gens[k]["reply"], "trait": judged[k]["trait"], "coherent": judged[k]["coherent"]})

    name, persona = meta.get("model_name") or meta["model"], meta["persona"]
    line = command = None
    if winner:
        line = (f"if you have a {name} around, try steering {winner['neuron']} {_fmt_dose(winner['best_dose'])}, "
                f"now you have {persona} {name}")
        cmd = list(meta.get("steer_command") or ["skrrt", "steer", "--model", meta["model"]])
        cmd += ["--neuron", winner["neuron"], f"--dose={winner['best_dose']:g}", samples[0]["prompt"] if samples else prompts[0]]
        command = " ".join(shlex.quote(c) for c in cmd)

    jrows = _jsonl(run / "screen.judge.jsonl") + list(judged.values())
    jinfo = dict(confirm.get("judge") or screen.get("judge") or {"kind": meta.get("judge")})
    jinfo.update(replies_judged=len(jrows), cost_usd=round(sum(r.get("cost") or 0.0 for r in jrows), 6))
    sc = [x for x in screen.get("neurons", [])]
    return {
        "verdict": label, "reasons": reasons, "rules": RULES, "model": meta["model"], "model_name": name,
        "persona": persona, "description": meta.get("description", ""), "question": meta.get("question", ""),
        "line": line, "recipe": RECIPE, "command": command, "n_prompts": n, "doses": confirm["doses"],
        "max_tokens": confirm.get("tokens"), "winner": winner, "finalists": finalists, "controls": controls,
        "baseline": baseline, "samples": samples, "judge": jinfo,
        "screen": {"candidates": sum(x["kind"] == "cand" for x in sc), "controls": sum(x["kind"] == "ctrl" for x in sc),
                   "doses": screen.get("doses"), "n_prompts": len(screen.get("prompts", [])),
                   "baseline_score": (screen.get("baseline") or {}).get("score"),
                   "best_candidate_score": max((x["score"] for x in sc if x["kind"] == "cand"), default=None),
                   "best_control_score": max((x["score"] for x in sc if x["kind"] == "ctrl"), default=None)},
        "skipped": {**(screen.get("skipped") or {}), **(confirm.get("skipped") or {})},
        "skrrt": meta.get("skrrt"), "time": meta.get("time"),
    }


def summary(rep: dict) -> str:
    """The few lines printed at the end of a hunt."""
    out = [f"VERDICT: {rep['verdict']} ({'; '.join(rep['reasons'])})"]
    if rep["line"]:
        if rep["verdict"] == "NOT FOUND":
            w = rep["winner"]
            out.append(f"best attempt: {w['neuron']} {_fmt_dose(w['best_dose'])} ({w['hits']}/{w['n']} hits)")
        else:
            out += [rep["line"], f"try it: {rep['command']}"]
    j = rep["judge"]
    out.append(f"judge: {j.get('kind')} ({j.get('model')}), {j['replies_judged']} replies, ${j['cost_usd']:.4f}")
    return "\n".join(out)


def _quote(text: str, limit: int = 600) -> str:
    text = " ".join((text or "").split())
    if len(text) > limit:
        text = text[:limit].rstrip() + " ..."
    return "> " + (text.replace("|", "\\|") if text else "(empty reply)")


def render(rep: dict) -> str:
    n, w = rep["n_prompts"], rep["winner"]
    L = [f"# skrrt: {rep['persona']} neuron in {rep['model_name']}", "",
         f"**Verdict: {rep['verdict']}.** " + "; ".join(rep["reasons"]).capitalize() + ".", ""]
    if rep["description"]:
        L += [f"Persona: {rep['persona']} ({rep['description']}).  "]
    L += [f"Judge question: {rep['question']}", ""]
    if w:
        L += ["## Try it" if rep["verdict"] != "NOT FOUND" else "## Best attempt (not confirmed)", "", "```",
              rep["line"], f"recipe: {rep['recipe']}",
              f"here: dose 1 on {w['neuron']} adds {w['unit']:.4g} to that neuron, so "
              f"{_fmt_dose(w['best_dose'])} adds {w['unit'] * w['best_dose']:+.4g}",
              "```", "", "```", rep["command"], "```", ""]
    L += ["## Finalists", "",
          f"Holdout: {n} prompts that were not used to pick neurons, {rep['max_tokens']} tokens per reply, "
          f"doses {', '.join(f'{d:g}' for d in rep['doses'])} in the direction of the cosine's sign.", "",
          "| # | neuron | cos | best dose | hits | coherent | mean P(persona) | dose-1 value | screen score |",
          "|---|---|---|---|---|---|---|---|---|"]
    for i, x in enumerate(rep["finalists"], 1):
        L.append(f"| {i} | {x['neuron']} | {x['cos']:+.3f} | {_fmt_dose(x['best_dose'])} | {x['hits']}/{n} | "
                 f"{x['coherent']}/{n} | {x['trait']:.2f} | {x['unit']:.4g} | {x['screen_score']:.3f} |")
    if not rep["finalists"]:
        L.append("| | none | | | | | | | |")
    if w:
        L += ["", f"### Dose sweep of {w['neuron']}", "", "| dose | hits | coherent | mean P(persona) |", "|---|---|---|---|"]
        L += [f"| {_fmt_dose(s['dose'])} | {s['hits']}/{n} | {s['coherent']}/{n} | {s['trait']:.2f} |" for s in w["sweep"]]
    L += ["", "## Controls", "",
          "Random neurons from the finalists' layers, same doses, random sign. Each is shown at its own best dose.", ""]
    if rep["controls"]:
        L += ["| neuron | cos | best dose | hits | coherent | mean P(persona) |", "|---|---|---|---|---|---|"]
        L += [f"| {x['neuron']} | {x['cos']:+.3f} | {_fmt_dose(x['best_dose'])} | {x['hits']}/{n} | {x['coherent']}/{n} | "
              f"{x['trait']:.2f} |" for x in rep["controls"]]
    else:
        L.append("None were run.")
    s = rep["screen"]
    if s.get("best_candidate_score") is not None:
        ctl = "n/a" if s["best_control_score"] is None else f"{s['best_control_score']:.3f}"
        L += ["", f"In the screen ({s['candidates']} candidates and {s['controls']} random neurons on {s['n_prompts']} "
                  f"prompts), the best candidate scored {s['best_candidate_score']:.3f}, the best random neuron {ctl}, "
                  f"the unsteered model {s['baseline_score']:.3f} (score = mean of P(persona) x P(coherent))."]
    b = rep["baseline"]
    L += ["", "## Baseline", "",
          f"Unsteered, the model hits {b['hits']}/{n} holdout prompts (coherent {b['coherent']}/{n}, "
          f"mean P(persona) {b['trait']:.2f}).", ""]
    if rep["samples"]:
        L += [f"## Samples: unsteered vs {w['neuron']} {_fmt_dose(w['best_dose'])}", ""]
        for x in rep["samples"]:
            L += [f"**{x['prompt']}**", "", "Unsteered:", "", _quote(x["baseline"]), "",
                  f"Steered (P(persona) {x['trait']:.2f}, P(coherent) {x['coherent']:.2f}):", "", _quote(x["steered"]), ""]
    L += ["## Verdict rules", ""] + [f"- {r}" for r in rep["rules"]] + [""]
    j = rep["judge"]
    jname = {"jev": "Jev on OpenRouter", "self": "the target model judging its own replies, unsteered (weaker than Jev)"}
    L += ["## Run", "", f"- Model: `{rep['model']}`",
          f"- Judge: {jname.get(j.get('kind'), j.get('kind'))}, model `{j.get('model')}`, "
          f"{j['replies_judged']} replies judged, cost ${j['cost_usd']:.4f}"]
    if rep["skipped"]:
        L.append(f"- Neurons the backend could not steer (skipped): {', '.join(rep['skipped'])}")
    L += [f"- skrrt {rep['skrrt']}, {rep['time']}", ""]
    return "\n".join(L)


def write(run_dir) -> dict:
    run = Path(run_dir)
    rep = build(run)
    (run / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    (run / "report.md").write_text(render(rep))
    return rep
