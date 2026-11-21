"""llama.cpp (GGUF) backend.

The model runs in `skrrt-engine`, a small stdin-command tool built inside a pinned llama.cpp tree by
`llamacpp/build.sh` (protocol at the top of `llamacpp/skrrt-engine/skrrt-engine.cpp`). Weights are
read straight from the GGUF with gguf-py.

How the push is applied. The recipe adds `delta` to one neuron at the input of the MLP down
projection, which is the fixed vector `delta * W_down[:, n]` added to the MLP output. llama.cpp can
add a fixed per-layer vector (a control vector) where the architecture's graph calls
`build_cvec(cur, il)`. In ordinary pre-norm models that is right after `cur = ffn_out + ffn_inp`,
i.e. on the residual leaving layer il, which is the same thing. So no patch is needed there.
Limits of that route:
  * llama.cpp has no control-vector slot for layer 0, so layer 0 cannot be steered;
  * a few architectures never call build_cvec, or do not name their layer output `l_out`. Use
    `check_push()` to measure what a steer really does on a given model.

`inkling2` / `inkling2_loop` (Inkling 2) is the exception: the MoE output passes through a short
convolution before the residual add, so `llamacpp/patches/inkling2-cvec.patch` adds the hook at the
MoE output. Its core layers also run several passes per token; each pass is its own graph layer
("slot"). For such a layer the push goes on every pass, each scaled by that slot's own median
residual norm; `capture` averages the passes; `unit()` reports the first pass.
"""
from __future__ import annotations

import atexit
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .base import Backend, Neuron, Steer

REC, UNIT = "\x1e", "\x1f"
_LLAMACPP_DIR = Path(__file__).resolve().parents[2] / "llamacpp"

# Fallback loop topology when the GGUF does not carry `<arch>.loop.*`: (core start, core end, passes).
_LOOP_FALLBACK = {"inkling2_loop": (6, 18, 3)}
# Extra stop tokens on top of the vocabulary's end-of-generation tokens.
_ARCH_STOPS = {"inkling2": ["<|end_message|>"], "inkling2_loop": ["<|end_message|>"]}
# Architectures whose control-vector hook only exists with a patch from llamacpp/patches.
_NEEDS_PATCH = {"inkling2": "inkling2-cvec.patch", "inkling2_loop": "inkling2-cvec.patch"}


def find_engine(engine: str | os.PathLike | None = None, prefer_cpu: bool = False) -> Path:
    """The skrrt-engine binary: `engine`, else $SKRRT_LLAMA_ENGINE, else llamacpp/build/{cuda,cpu}/bin."""
    given = engine or os.environ.get("SKRRT_LLAMA_ENGINE")
    if given:
        p = Path(given).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return p
        raise FileNotFoundError(f"skrrt-engine not found at {p}. Build it with {_LLAMACPP_DIR / 'build.sh'}")
    kinds = ("cpu", "cuda") if prefer_cpu else ("cuda", "cpu")
    for kind in kinds:
        p = _LLAMACPP_DIR / "build" / kind / "bin" / "skrrt-engine"
        if p.is_file() and os.access(p, os.X_OK):
            return p
    raise FileNotFoundError(
        f"the llama.cpp engine is not built: no skrrt-engine under {_LLAMACPP_DIR / 'build'}. "
        f"Run {_LLAMACPP_DIR / 'build.sh'} (SKRRT_CUDA=0 for a CPU-only build), "
        "or point SKRRT_LLAMA_ENGINE / engine= at an existing binary.")


class _GGUF:
    """Metadata and tensor access for one GGUF file (memory mapped; nothing is read until asked)."""

    def __init__(self, path: str):
        from gguf import GGUFReader
        self.reader = GGUFReader(path)
        self.tensors = {t.name: t for t in self.reader.tensors}
        self.arch = self.get("general.architecture")
        if not isinstance(self.arch, str):
            raise ValueError(f"{path}: no general.architecture")

    def get(self, key: str, default=None):
        f = self.reader.fields.get(key)
        if f is None:
            return default
        try:
            return f.contents()
        except Exception:
            return default

    def hp(self, key: str, default=None):
        return self.get(f"{self.arch}.{key}", default)

    def token_text(self, key: str) -> str | None:
        """Text of the token whose id is stored under tokenizer.ggml.<key>, without decoding the whole vocabulary."""
        idx = self.get(f"tokenizer.ggml.{key}")
        toks = self.reader.fields.get("tokenizer.ggml.tokens")
        if idx is None or toks is None or not 0 <= int(idx) < len(toks.data):
            return None
        return bytes(toks.parts[toks.data[int(idx)]]).decode("utf-8", "replace")


def _dequant(t, data=None) -> np.ndarray:
    from gguf.quants import dequantize
    return np.asarray(dequantize(t.data if data is None else data, t.tensor_type), dtype=np.float32)


class _Template:
    """The GGUF's jinja chat template, rendered in a sandbox."""

    def __init__(self, source: str, bos: str, eos: str):
        import jinja2
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        def raise_exception(msg):
            raise jinja2.TemplateError(msg)

        def strftime_now(fmt):
            import datetime
            return datetime.datetime.now().strftime(fmt)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        env.globals.update(raise_exception=raise_exception, strftime_now=strftime_now)
        env.filters["tojson"] = lambda v, **kw: __import__("json").dumps(
            v, ensure_ascii=False, **{k: kw[k] for k in ("indent", "sort_keys", "separators") if k in kw})
        # Strip block tags jinja2 does not know (e.g. `{% generation %}` ... `{% endgeneration %}`).
        for _ in range(8):
            try:
                self.template = env.from_string(source)
                break
            except jinja2.TemplateSyntaxError as e:
                m = re.search(r"unknown tag '(\w+)'", str(e))
                if not m:
                    raise
                tag = re.escape(m[1].removeprefix("end"))
                source = re.sub(r"\{%-?\s*(?:end)?" + tag + r"\s*-?%\}", "", source)
        else:
            raise ValueError("chat template has too many tags jinja2 cannot parse")
        self.bos, self.eos = bos, eos

    def render(self, user: str, system: str | None) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        merged = [{"role": "user", "content": f"{system}\n\n{user}" if system else user}]
        last = None
        for m in ([msgs, merged] if system else [msgs]):  # some templates reject a system turn
            try:
                # thinking off, under the names templates commonly use for the switch
                return self.template.render(messages=m, add_generation_prompt=True, bos_token=self.bos,
                                            eos_token=self.eos, enable_thinking=False, thinking=False, tools=None)
            except Exception as e:  # TemplateError, TypeError, SecurityError, ...
                last = e
        raise RuntimeError(f"chat template failed: {last}")


def _inkling2_chat(user: str, system: str | None) -> str:
    """Hand-built Inkling 2 format with thinking effort 0 (its template defaults to 0.9)."""
    s = ""
    if system:
        s += f"<|message_system|><|content_text|>{system}<|end_message|>"
    s += "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
    s += f"<|message_user|><|content_text|>{user}<|end_message|>"
    s += "<|message_model|><|content_text|>"
    return s


class LlamaCppBackend(Backend):
    def __init__(self, model: str, n_gpu_layers: int = 999, n_seq: int = 12, n_ctx: int = 768,
                 engine: str | None = None, n_ubatch: int = 512, threads: int | None = None,
                 stop: list[str] | None = None, **ignored):
        """n_ctx is the context per sequence; n_seq sequences are decoded in parallel.
        n_gpu_layers=0 runs on the CPU and hides the GPUs from the engine."""
        self.name = str(model)
        path = Path(model).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"no such GGUF: {path}")
        self._engine_path = find_engine(engine, prefer_cpu=n_gpu_layers == 0)
        self._gguf = g = _GGUF(str(path))
        self.arch = g.arch
        self.n_layers = int(g.hp("block_count"))
        self.hidden = int(g.hp("embedding_length"))

        # ---- topology: which graph layers ("slots") each physical layer runs in
        start, end, passes = g.hp("loop.start"), g.hp("loop.end"), g.hp("loop.count")
        if None in (start, end, passes) and self.arch in _LOOP_FALLBACK:
            start, end, passes = _LOOP_FALLBACK[self.arch]
        if None in (start, end, passes):
            start, end, passes = 0, 0, 1
        self._core0, self._core1, self._passes = int(start), int(end), int(passes)
        self.n_slots = self.n_layers + (self._passes - 1) * (self._core1 - self._core0)
        self.needs_patch = _NEEDS_PATCH.get(self.arch)

        # ---- chat format
        self._bos = g.token_text("bos_token_id") if g.get("tokenizer.ggml.add_bos_token", False) else None
        tpl = g.get("tokenizer.chat_template")
        self._template = None
        if not self.arch.startswith("inkling2") and isinstance(tpl, str) and tpl.strip():
            self._template = _Template(tpl, g.token_text("bos_token_id") or "", g.token_text("eos_token_id") or "")

        # ---- engine
        self._tmp = Path(tempfile.mkdtemp(prefix="skrrt-llamacpp-"))
        self._k = 0
        self._calib: np.ndarray | None = None   # median residual norm per slot
        self._cols: dict[Neuron, tuple[np.ndarray, float]] = {}
        cmd = [str(self._engine_path), "-m", str(path), "-ngl", str(int(n_gpu_layers)), "-s", str(int(n_seq)),
               "-c", str(int(n_ctx)), "-ub", str(int(n_ubatch))]
        if threads:
            cmd += ["-t", str(int(threads))]
        for s in list(stop or []) + _ARCH_STOPS.get(self.arch, []):
            cmd += ["-stop", s]
        env = dict(os.environ)
        if n_gpu_layers == 0:
            env["CUDA_VISIBLE_DEVICES"] = ""  # a CUDA build would otherwise still put work on the GPU
        self._log_path = self._tmp / "engine.log"
        self._log = open(self._log_path, "wb")
        self._p: subprocess.Popen | None = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log, env=env)
        atexit.register(self.close)
        line = self._p.stdout.readline().decode("utf-8", "replace").strip()
        if not line.startswith("READY"):
            err = self._log_tail()
            self.close()
            raise RuntimeError(f"skrrt-engine did not start ({' '.join(cmd)}):\n{err}")
        _, n_slots, n_embd = line.split()
        if int(n_embd) != self.hidden or int(n_slots) != self.n_slots:
            self.close()
            raise RuntimeError(f"engine reports {n_slots} graph layers x {n_embd}, the GGUF metadata gives "
                               f"{self.n_slots} x {self.hidden} (architecture {self.arch!r})")

    # ------------------------------------------------------------------ engine plumbing

    def _log_tail(self, n: int = 25) -> str:
        try:
            self._log.flush()
            return "\n".join(self._log_path.read_text(errors="replace").splitlines()[-n:])
        except Exception:
            return ""

    def _cmd(self, *fields) -> None:
        if self._p is None or self._p.poll() is not None:
            raise RuntimeError("skrrt-engine is not running\n" + self._log_tail())
        line = "\t".join(str(x) for x in fields)
        if "\n" in line:
            raise ValueError("newline in an engine command")
        try:
            self._p.stdin.write(line.encode() + b"\n")
            self._p.stdin.flush()
            r = self._p.stdout.readline().decode("utf-8", "replace").strip()
        except BrokenPipeError:
            r = ""
        if r == "OK":
            return
        if not r:
            raise RuntimeError("skrrt-engine died:\n" + self._log_tail())
        raise RuntimeError(f"skrrt-engine: {r.removeprefix('ERR ')}")

    def _file(self, tag: str) -> Path:
        self._k += 1
        return self._tmp / f"{tag}{self._k}"

    def _records(self, tag: str, items: list[str]) -> Path:
        f = self._file(tag)
        f.write_bytes("".join(s + REC for s in items).encode("utf-8"))
        return f

    def _prep(self, prompt: str) -> str:
        """A chat-formatted prompt as the engine tokenizes it: special tokens are parsed from the
        text and none are added, so put the BOS in when the model wants one and the text lacks it."""
        prompt = prompt.replace(REC, " ").replace(UNIT, " ")
        if self._bos:
            while prompt.startswith(self._bos * 2):
                prompt = prompt[len(self._bos):]
            if not prompt.startswith(self._bos):
                prompt = self._bos + prompt
        return prompt

    # ------------------------------------------------------------------ topology and weights

    def slots(self, layer: int) -> list[int]:
        """Graph layer indices where physical layer `layer` runs (more than one for a looped layer)."""
        if not 0 <= layer < self.n_layers:
            raise ValueError(f"layer {layer} out of range 0..{self.n_layers - 1}")
        if self._core0 <= layer < self._core1:
            span = self._core1 - self._core0
            return [layer] + [self.n_layers + p * span + (layer - self._core0) for p in range(self._passes - 1)]
        return [layer]

    def _down(self, layer: int):
        """(tensor, is_moe) for the MLP down projection of a layer."""
        if not 0 <= layer < self.n_layers:
            raise ValueError(f"layer {layer} out of range 0..{self.n_layers - 1}")
        t = self._gguf.tensors.get(f"blk.{layer}.ffn_down_exps.weight")
        if t is not None:
            return t, True
        t = self._gguf.tensors.get(f"blk.{layer}.ffn_down.weight")
        if t is None:
            raise ValueError(f"{self.name}: layer {layer} has neither blk.{layer}.ffn_down.weight nor "
                             f"blk.{layer}.ffn_down_exps.weight")
        return t, False

    def _dims(self, layer: int) -> tuple[int, int, int, bool]:
        """(G, I, H, is_moe). GGUF stores dimensions innermost first: [I, H] dense, [I, H, E] MoE,
        which numpy sees as (H, I) / (E, H, I)."""
        t, moe = self._down(layer)
        shape = [int(x) for x in t.shape]
        if len(shape) != (3 if moe else 2) or shape[1] != self.hidden:
            raise ValueError(f"{t.name}: unexpected shape {shape} (hidden {self.hidden})")
        return (shape[2] if moe else 1), shape[0], shape[1], moe

    def experts(self, layer: int) -> int:
        g, _, _, moe = self._dims(layer)
        return g if moe else 0

    def write_vectors(self, layer: int) -> np.ndarray:
        g, i, h, moe = self._dims(layer)
        t, _ = self._down(layer)
        w = _dequant(t).reshape(g, h, i)               # numpy order of the GGUF dims
        return np.ascontiguousarray(w.transpose(0, 2, 1))  # (G, I, H): row [g, n] = W_down[g][:, n]

    def _column(self, n: Neuron) -> tuple[np.ndarray, float]:
        """(W_down[:, n] as float32 (H,), its norm). Only the neuron's expert is dequantized."""
        if n not in self._cols:
            g, i, h, moe = self._dims(n.layer)
            if moe and (n.expert is None or not 0 <= n.expert < g):
                raise ValueError(f"{n.label()}: layer {n.layer} has {g} experts, name one (E0..E{g - 1})")
            if not moe and n.expert is not None:
                raise ValueError(f"{n.label()}: layer {n.layer} has a plain MLP, no experts")
            if not 0 <= n.index < i:
                raise ValueError(f"{n.label()}: neuron index out of range 0..{i - 1}")
            t, _ = self._down(n.layer)
            data = t.data
            if moe:
                data = data.reshape(g, h, -1)[n.expert]
            col = np.ascontiguousarray(_dequant(t, data).reshape(h, i)[:, n.index])
            norm = float(np.linalg.norm(col))
            if not norm > 0:
                raise ValueError(f"{n.label()}: the neuron's write vector is zero")
            self._cols[n] = (col, norm)
        return self._cols[n]

    def _need_calibration(self) -> np.ndarray:
        if self._calib is None:
            raise RuntimeError("call calibrate() before steering")
        return self._calib

    def unit(self, neuron: Neuron) -> float:
        med = self._need_calibration()
        return float(med[self.slots(neuron.layer)[0]]) / self._column(neuron)[1]

    def push_vectors(self, steers: list[Steer]) -> dict[int, np.ndarray]:
        """slot -> float32 (H,) vector that the given steers add there (zero doses dropped)."""
        med = self._need_calibration()
        push: dict[int, np.ndarray] = {}
        for s in steers:
            if not s.dose:
                continue
            col, norm = self._column(s.neuron)
            for slot in self.slots(s.neuron.layer):  # every pass, scaled to its own residual length
                if slot == 0:
                    raise ValueError(f"{s.neuron.label()}: llama.cpp cannot steer layer 0 (no control-vector slot)")
                v = (s.dose * float(med[slot]) / norm) * col
                push[slot] = push.get(slot, 0) + v.astype(np.float32)
        return push

    def _steer_args(self, steers: list[Steer] | None) -> tuple:
        push = self.push_vectors(steers) if steers else {}
        if not push:
            return ("-", -1, -1)
        buf = np.zeros((self.n_slots - 1, self.hidden), np.float32)
        for slot, v in push.items():
            buf[slot - 1] = v
        f = self._file("v")
        buf.tofile(f)
        return (f, min(push), max(push))

    # ------------------------------------------------------------------ text

    def chat(self, user: str, system: str | None = None) -> str:
        if self.arch.startswith("inkling2"):
            return _inkling2_chat(user, system)
        if self._template is None:
            return (f"{system}\n\n" if system else "") + f"User: {user}\nAssistant:"
        return self._template.render(user, system)

    # ------------------------------------------------------------------ residuals

    def _cap(self, pairs: list[tuple[str, str]], steers: list[Steer] | None = None):
        """Teacher-forced residuals per slot: (mean (n, n_slots, H) over the reply tokens,
        norms (tokens 1: of all entries, n_slots), ntok (n, 2) = total and prompt token counts)."""
        ef = self._records("e", [self._prep(p) + UNIT + c.replace(REC, " ").replace(UNIT, " ") for p, c in pairs])
        of = self._file("c")
        self._cmd("CAP", *self._steer_args(steers), ef, of)
        mean = np.fromfile(f"{of}.mean", np.float32).reshape(len(pairs), self.n_slots, self.hidden)
        norm = np.fromfile(f"{of}.norm", np.float32).reshape(-1, self.n_slots)
        ntok = np.fromfile(f"{of}.ntok", np.int32).reshape(len(pairs), 2)
        for suffix in (".mean", ".norm", ".ntok"):
            Path(f"{of}{suffix}").unlink(missing_ok=True)
        return mean, norm, ntok

    def calibrate(self, prompts: list[str]) -> dict:
        if not prompts:
            raise ValueError("calibrate needs at least one prompt")
        _, norm, _ = self._cap([(p, "") for p in prompts])
        if norm.shape[0] == 0:
            raise ValueError("calibration prompts have no tokens after the first one")
        self._calib = np.median(norm.astype(np.float64), axis=0)
        rec = {
            "median_resid_norm": [float(self._calib[self.slots(l)[0]]) for l in range(self.n_layers)],
            "n_prompts": len(prompts),
            "n_tokens": int(norm.shape[0]),
            "model": self.name,
        }
        if self.n_slots != self.n_layers:  # looped model: every pass has its own norm
            rec["median_resid_norm_slots"] = [float(x) for x in self._calib]
            rec["loop"] = {"start": self._core0, "end": self._core1, "passes": self._passes}
        return rec

    def _per_layer(self, slot_means: np.ndarray) -> np.ndarray:
        """(n, n_slots, H) -> (n, n_layers, H), averaging the passes of looped layers."""
        if self.n_slots == self.n_layers:
            return slot_means
        return np.stack([slot_means[:, self.slots(l)].mean(1) for l in range(self.n_layers)], axis=1)

    def capture(self, pairs: list[tuple[str, str]], steers: list[Steer] | None = None) -> np.ndarray:
        """The contract's capture. `steers` (an extension) teacher-forces with those pushes active."""
        if not pairs:
            return np.zeros((0, self.n_layers, self.hidden), np.float32)
        return np.ascontiguousarray(self._per_layer(self._cap(pairs, steers)[0]), dtype=np.float32)

    def check_push(self, steer: Steer, pairs: list[tuple[str, str]]) -> dict:
        """Measure what a steer does to the residual leaving its layer (first pass), teacher-forced on
        the same tokens: returns the relative error against the vector the recipe says is added.
        ~0 for ordinary architectures. Not expected to be ~0 where the hook sits before another
        operation (inkling2: the push goes through the post-MoE convolution)."""
        slot = self.slots(steer.neuron.layer)[0]
        want = self.push_vectors([steer])[slot]
        base, steered = self._cap(pairs)[0], self._cap(pairs, [steer])[0]
        diff = steered - base                                           # (n, n_slots, H)
        err = np.linalg.norm(diff[:, slot] - want, axis=1) / np.linalg.norm(want)
        before = np.abs(diff[:, :slot]).max() if slot else 0.0
        cos = diff[:, slot] @ want / (np.linalg.norm(diff[:, slot], axis=1) * np.linalg.norm(want) + 1e-30)
        return {"slot": slot, "rel_err": float(err.max()), "cosine": float(cos.min()),
                "max_change_before_layer": float(before), "push_norm": float(np.linalg.norm(want))}

    # ------------------------------------------------------------------ generation and scoring

    def generate(self, prompts: list[str], steers: list[list[Steer]] | None = None,
                 max_tokens: int = 100) -> list[str]:
        if steers is None:
            steers = [[] for _ in prompts]
        if len(steers) != len(prompts):
            raise ValueError("steers must have one entry per prompt")
        # a control vector belongs to the whole context, so rows are grouped by identical steer set
        groups: dict[tuple, list[int]] = {}
        for i, row in enumerate(steers):
            key = tuple(sorted((s.neuron, float(s.dose)) for s in row if s.dose))
            groups.setdefault(key, []).append(i)
        out: list[str] = [""] * len(prompts)
        for key, rows in groups.items():
            args = self._steer_args([Steer(n, d) for n, d in key])
            pf, of = self._records("p", [self._prep(prompts[i]) for i in rows]), self._file("o")
            self._cmd("GEN", *args, int(max_tokens), pf, of)
            texts = of.read_bytes().decode("utf-8", "replace").split(REC)[:-1]
            if len(texts) != len(rows):
                raise RuntimeError(f"engine returned {len(texts)} replies for {len(rows)} prompts")
            for i, t in zip(rows, texts):
                out[i] = t.strip()
            for f in (pf, of, args[0]):
                if isinstance(f, Path):
                    f.unlink(missing_ok=True)
        return out

    def yes_no(self, prompts: list[str], yes: str = "Yes", no: str = "No") -> list[float]:
        if not prompts:
            return []
        if "\t" in yes + no or "\n" in yes + no:
            raise ValueError("yes/no words cannot contain tabs or newlines")
        rf, of = self._records("r", [self._prep(p) for p in prompts]), self._file("s")
        self._cmd("SCORE", rf, yes, no, of)
        vals = [float(x) for x in of.read_text().split()]
        if len(vals) != len(prompts):
            raise RuntimeError(f"engine returned {len(vals)} scores for {len(prompts)} prompts")
        rf.unlink(missing_ok=True)
        of.unlink(missing_ok=True)
        return vals

    def close(self) -> None:
        p, self._p = getattr(self, "_p", None), None
        if p is not None:
            try:
                if p.poll() is None:
                    p.stdin.write(b"QUIT\n")
                    p.stdin.flush()
                    p.wait(30)
            except Exception:
                pass
            if p.poll() is None:  # our own child, by handle: never by name
                p.kill()
                p.wait()
            for s in (p.stdin, p.stdout):
                try:
                    s.close()
                except Exception:
                    pass
        log = getattr(self, "_log", None)
        if log is not None and not log.closed:
            log.close()
        tmp = getattr(self, "_tmp", None)
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
        try:
            atexit.unregister(self.close)
        except Exception:
            pass


def export_cvec(backend: LlamaCppBackend, steers: list[Steer], path: str | os.PathLike) -> dict:
    """Write the given steers, doses baked in, as a llama.cpp control-vector GGUF.

    Serve it with `llama-server --control-vector-scaled FILE:1.0` (scale 1.0 = exactly these doses;
    the scale multiplies every dose). For architectures listed in `llamacpp/patches` the server must
    be built with that patch, or the vector lands in the wrong place or nowhere. Returns a small record."""
    from gguf import GGUFWriter
    push = backend.push_vectors(list(steers))
    if not push:
        raise ValueError("nothing to export: no steer with a non-zero dose")
    w = GGUFWriter(str(path), "controlvector")
    w.add_string("controlvector.model_hint", backend.arch)
    w.add_uint32("controlvector.layer_count", len(push))
    w.add_string("skrrt.steers", "; ".join(f"{s.neuron.label()} {s.dose:+g}" for s in steers if s.dose))
    w.add_string("skrrt.model", os.path.basename(backend.name))
    if backend.needs_patch:
        w.add_string("skrrt.needs_patch", backend.needs_patch)
    for slot in sorted(push):
        w.add_tensor(f"direction.{slot}", np.ascontiguousarray(push[slot], dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return {"path": str(path), "slots": sorted(push), "needs_patch": backend.needs_patch,
            "push_norm": {str(s): float(np.linalg.norm(v)) for s, v in sorted(push.items())}}
