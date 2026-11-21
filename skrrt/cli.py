"""skrrt command line. `skrrt hunt --model M --persona P` is the whole pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _floats(s: str) -> tuple:
    try:
        out = tuple(float(x) for x in s.split(",") if x.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers, got {s!r}") from None
    if not out or any(x <= 0 for x in out):
        raise argparse.ArgumentTypeError("doses must be positive numbers (the sign comes from the cosine)")
    return out


def _backend_args(items) -> dict:
    out = {}
    for it in items or []:
        k, sep, v = it.partition("=")
        if not sep or not k:
            raise SystemExit(f"skrrt: --backend-arg expects key=value, got {it!r}")
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def _add_backend(p, model_required=True):
    p.add_argument("--model", required=model_required, help="HF model id or path, a .gguf file, or 'mock'")
    p.add_argument("--backend", default="auto", choices=["auto", "hf", "llamacpp", "mock"])
    p.add_argument("--backend-arg", action="append", metavar="KEY=VALUE", help="passed to the backend constructor")


def _add_steers(p):
    p.add_argument("--neuron", action="append", required=True, metavar='"L11 E14 N370"',
                   help="repeat with a --dose for each neuron")
    p.add_argument("--dose", action="append", required=True, type=float, help="signed, in residual norms")


def _steers(args):
    from .backends import Neuron, Steer
    if len(args.neuron) != len(args.dose):
        raise SystemExit(f"skrrt: {len(args.neuron)} --neuron but {len(args.dose)} --dose; give one dose per neuron")
    try:
        return [Steer(Neuron.parse(n), d) for n, d in zip(args.neuron, args.dose)]
    except ValueError as e:
        raise SystemExit(f"skrrt: {e}") from None


def _persona(spec):
    from .persona import load_persona
    try:
        return load_persona(spec)
    except ValueError as e:
        raise SystemExit(f"skrrt: error: {e}") from None


def _load(args):
    from .backends import load_backend
    return load_backend(args.model, args.backend, **_backend_args(args.backend_arg))


def cmd_hunt(args) -> int:
    from .hunt import HuntConfig, run_hunt
    cfg = HuntConfig(model=args.model, persona=args.persona, backend=args.backend, judge=args.judge, out=args.out,
                     layers=args.layers, per_layer=args.per_layer, screen_doses=args.screen_doses, doses=args.doses,
                     finalists=args.finalists, controls=args.controls, max_tokens=args.max_tokens, seed=args.seed,
                     force=args.force, backend_args=_backend_args(args.backend_arg))
    run_hunt(cfg)
    return 0


def cmd_personas(args) -> int:
    from .persona import list_personas, load_persona
    names = list_personas()
    if not names:
        print("no built-in personas found")
        return 1
    width = max(len(n) for n in names)
    for n in names:
        p = load_persona(n)
        print(f"{n:<{width}}  {p.description}  ({len(p.replies)} replies)")
    return 0


def cmd_new_persona(args) -> int:
    from .persona import TEMPLATE
    path = Path(args.name if args.name.endswith((".yaml", ".yml")) else args.name + ".yaml")
    if path.exists():
        raise SystemExit(f"skrrt: {path} already exists")
    path.write_text(TEMPLATE.format(name=path.stem))
    print(f"wrote {path}. Fill in description, question and system, then either write the replies yourself\n"
          f"or run: skrrt write-replies {path} --model <model>")
    return 0


def cmd_write_replies(args) -> int:
    from .judge import make_judge, resolve_kind
    from .writer import write_replies
    persona = _persona(args.file)
    if persona.replies and not args.force:
        raise SystemExit(f"skrrt: {args.file} already has {len(persona.replies)} replies (use --force to replace them)")
    need_model = args.writer == "target" or resolve_kind(args.judge) == "self"
    if need_model and not args.model:
        raise SystemExit("skrrt: --model is needed (for --writer target, and for the self judge)")
    backend = _load(args) if need_model else None
    try:
        judge = make_judge(args.judge, lambda: backend)
        res = write_replies(persona, args.writer, backend, judge, args.max_tokens)
    finally:
        if backend is not None:
            backend.close()
    if not res["written"]:
        raise SystemExit("skrrt: the writer returned no replies; nothing saved")
    persona.save()
    print(f"saved {res['written']} replies to {persona.path}. Read them before hunting.")
    return 0 if res["ok"] else 1


def cmd_steer(args) -> int:
    from .hunt import calibrate
    steers = _steers(args)
    persona = _persona(args.persona) if args.persona else None
    backend = _load(args)
    try:
        calibrate(backend, persona)
        for s in steers:
            u = backend.unit(s.neuron)
            print(f"# {s.neuron.label()} dose {s.dose:+g}: adds {u * s.dose:+.4g} to the neuron (dose 1 = {u:.4g})",
                  file=sys.stderr)

        def answer(prompt):
            text = backend.chat(prompt)
            if args.baseline:
                print("--- unsteered")
                print(backend.generate([text], [[]], args.max_tokens)[0])
                print("--- steered")
            print(backend.generate([text], [steers], args.max_tokens)[0])

        if args.prompt is not None:
            answer(args.prompt)
        else:
            print("# type a prompt; empty line or ctrl-d to quit", file=sys.stderr)
            while True:
                try:
                    prompt = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not prompt:
                    break
                answer(prompt)
    finally:
        backend.close()
    return 0


def cmd_report(args) -> int:
    from . import report
    rep = report.write(args.run_dir)
    print(report.summary(rep))
    print(f"report: {Path(args.run_dir) / 'report.md'}")
    return 0


def cmd_export_cvec(args) -> int:
    from .backends import load_backend
    from .hunt import calibrate
    steers = _steers(args)
    backend = load_backend(args.model, "llamacpp", **_backend_args(args.backend_arg))
    try:
        from .backends.llamacpp import export_cvec
        calibrate(backend, _persona(args.persona) if args.persona else None)
        export_cvec(backend, steers, args.output)
    finally:
        backend.close()
    print(f"wrote {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="skrrt", description="Find the single neuron that steers a language model into a persona.")
    ap.add_argument("--version", action="version", version=f"skrrt {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    p = sub.add_parser("hunt", help="rank, screen, confirm and report: the whole pipeline")
    _add_backend(p)
    p.add_argument("--persona", required=True, help="built-in name (see `skrrt personas`) or path to a YAML file")
    p.add_argument("--judge", default="auto", choices=["auto", "jev", "self"])
    p.add_argument("--out", metavar="DIR", help="run dir (default runs/<model>-<persona>/)")
    p.add_argument("--layers", metavar="A-B", help="layers to screen, inclusive (default all)")
    p.add_argument("--per-layer", type=int, default=8, help="candidates per layer (default 8)")
    p.add_argument("--screen-doses", type=_floats, default=(1.0, 2.0), metavar="1,2")
    p.add_argument("--doses", type=_floats, default=(0.5, 0.75, 1.0, 1.5, 2.0, 3.0), metavar="0.5,0.75,1,1.5,2,3")
    p.add_argument("--finalists", type=int, default=10)
    p.add_argument("--controls", type=int, default=4, help="random control neurons per finalist layer (default 4)")
    p.add_argument("--max-tokens", type=int, default=100, help="reply length in the confirm stage (default 100)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="redo every stage even if its output exists")
    p.set_defaults(fn=cmd_hunt)

    p = sub.add_parser("personas", help="list the built-in personas")
    p.set_defaults(fn=cmd_personas)

    p = sub.add_parser("new-persona", help="write NAME.yaml from the template")
    p.add_argument("name")
    p.set_defaults(fn=cmd_new_persona)

    p = sub.add_parser("write-replies", help="fill a persona file's replies using its system text")
    p.add_argument("file", metavar="FILE.yaml")
    _add_backend(p, model_required=False)
    p.add_argument("--writer", default="target", help="'target' (the model itself) or an OpenRouter chat model id")
    p.add_argument("--judge", default="auto", choices=["auto", "jev", "self"])
    p.add_argument("--max-tokens", type=int, default=120)
    p.add_argument("--force", action="store_true", help="replace existing replies")
    p.set_defaults(fn=cmd_write_replies)

    p = sub.add_parser("steer", help="generate with one or more neurons pushed")
    _add_backend(p)
    _add_steers(p)
    p.add_argument("--persona", help="calibrate on this persona's train prompts (only matters if it has its own)")
    p.add_argument("--max-tokens", type=int, default=100)
    p.add_argument("--baseline", action="store_true", help="also print the unsteered reply")
    p.add_argument("prompt", nargs="?", help="omit for an interactive loop")
    p.set_defaults(fn=cmd_steer)

    p = sub.add_parser("report", help="rebuild report.md and report.json from a finished run")
    p.add_argument("run_dir")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("export-cvec", help="llama.cpp only: save the push as a control-vector GGUF")
    p.add_argument("--model", required=True, help="a .gguf file")
    p.add_argument("--backend-arg", action="append", metavar="KEY=VALUE")
    _add_steers(p)
    p.add_argument("--persona", help="calibrate on this persona's train prompts")
    p.add_argument("-o", "--output", required=True, metavar="FILE.gguf")
    p.set_defaults(fn=cmd_export_cvec)
    return ap


def main(argv=None) -> int:
    from .hunt import HuntError
    from .judge import JudgeError
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args) or 0
    except (HuntError, JudgeError, FileNotFoundError) as e:
        print(f"skrrt: error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nskrrt: interrupted (rerun the same command to resume)", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
