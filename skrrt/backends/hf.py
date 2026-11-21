"""Hugging Face transformers backend.

Architecture detection is structural rather than table driven:

* the decoder block list is the longest ``nn.ModuleList`` whose children all emit a
  (batch, tokens, H) tensor on a tiny text-only probe forward (so vision towers, expert
  lists and the like are never picked);
* in every block the MLP down projection is the linear map that writes back into the
  residual stream and does not live inside an attention module. Three layouts are handled:
  one ``nn.Linear`` / GPT-2 ``Conv1D`` (plain MLP), several of them under an indexed
  container (``experts.3.w2``), or one stacked 3-D parameter (``experts.down_proj`` of shape
  (E, H, I), the layout transformers uses for MoE models since v5).

Steering adds the fixed vector ``dose * median_resid_norm[L] * W_down[:, n] / ||W_down[:, n]||``
to the MLP output of layer L at every token. For a plain MLP the hook sits on the down
projection itself; for a MoE layer it sits on the block's MoE module (the block's direct child
holding the experts), so the push lands on every token whatever the router picked.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .base import Backend, Neuron, Steer

_DOWN_NAMES = {"down_proj", "fc2", "c_proj", "w2", "fc_out", "dense_out", "proj_out", "lin2",
               "linear2", "linear_2", "out_proj", "dense_2", "output_dense", "wo"}
_ATTN = re.compile(r"attn|attention", re.I)
_END_OF_TURN = ("<|im_end|>", "<|eot_id|>", "<|eom_id|>", "<end_of_turn>", "<turn|>", "<|end|>",
                "<|endoftext|>", "<|end_of_text|>", "</s>", "<|return|>", "<｜end▁of▁sentence｜>")


class _Stop(Exception):
    """Raised from a hook on the last block to skip the final norm and the LM head."""


def _first(out):
    return out if torch.is_tensor(out) else out[0]


def _is_linear(m: nn.Module) -> bool:
    if isinstance(m, nn.Linear):
        return True
    return type(m).__name__ == "Conv1D" and getattr(m, "weight", None) is not None and m.weight.ndim == 2


def _out_features(m: nn.Module) -> int:
    return m.out_features if isinstance(m, nn.Linear) else m.weight.shape[1]


def _in_features(m: nn.Module) -> int:
    return m.in_features if isinstance(m, nn.Linear) else m.weight.shape[0]


def _in_attention(path: str) -> bool:
    return any(_ATTN.search(p) for p in path.split(".")[:-1]) or bool(_ATTN.search(path.split(".")[-1]))


def _dense(w: torch.Tensor) -> torch.Tensor:
    """A plain float32 CPU copy of a (possibly half precision / quantized) weight."""
    w = w.detach()
    if w.is_quantized:
        w = w.dequantize()
    return w.to(device="cpu", dtype=torch.float32)


@dataclass
class _Down:
    """Where one layer's MLP writes into the residual stream."""
    hook: nn.Module                      # module whose output receives the push
    path: str                            # for error messages / reports
    mods: list = field(default_factory=list)   # Linear / Conv1D, one per expert (or just one)
    stacked: torch.Tensor | None = None  # 3-D parameter holding all experts
    hidden_first: bool = True            # stacked layout (E, H, I) if True, else (E, I, H)
    moe: bool = False

    @property
    def experts(self) -> int:
        if not self.moe:
            return 0
        return self.stacked.shape[0] if self.stacked is not None else len(self.mods)

    @property
    def groups(self) -> int:
        return max(1, self.experts)

    def _weight(self, g: int) -> torch.Tensor:
        """Weight of expert g as stored, plus whether it is laid out (H, I)."""
        if self.stacked is not None:
            return self.stacked[g], self.hidden_first
        m = self.mods[g]
        w = m.weight
        if w.is_quantized or not torch.is_floating_point(w):
            raise NotImplementedError(f"cannot read quantized weight of {type(m).__name__} at {self.path}")
        return w, isinstance(m, nn.Linear)

    @property
    def width(self) -> int:
        w, hidden_first = self._weight(0)
        return w.shape[1] if hidden_first else w.shape[0]

    def column(self, g: int, n: int) -> torch.Tensor:
        w, hidden_first = self._weight(g)
        return _dense(w[:, n] if hidden_first else w[n])

    def matrix(self, g: int) -> torch.Tensor:
        """(I, H) float32 on CPU."""
        w, hidden_first = self._weight(g)
        w = _dense(w)
        return w.t() if hidden_first else w


def _find_down(block: nn.Module, hidden: int, layer: int) -> _Down:
    def fail(why: str):
        subs = [f"{n} ({type(m).__name__})" for n, m in block.named_modules() if n]
        raise RuntimeError(
            f"layer {layer}: {why} (hidden size {hidden}) in block {type(block).__name__}. "
            f"Submodules: {', '.join(subs)}")

    def moe_hook(path: str) -> nn.Module:
        return getattr(block, path.split(".")[0])

    # 1. stacked experts: one 3-D parameter (E, H, I) or (E, I, H)
    stacked = [(n, p) for n, p in block.named_parameters()
               if p.ndim == 3 and p.shape[0] > 1 and not _in_attention(n) and hidden in p.shape[1:]]
    named = [(n, p) for n, p in stacked if n.split(".")[-1] in _DOWN_NAMES]
    if not named:  # unknown name: gate/up weights read from the stream, so H is their last dim
        named = [(n, p) for n, p in stacked if p.shape[1] == hidden and p.shape[2] != hidden]
    if len(named) == 1 and "." in named[0][0]:
        n, p = named[0]
        return _Down(hook=moe_hook(n), path=n, stacked=p, hidden_first=p.shape[1] == hidden, moe=True)
    if len(named) > 1:
        fail(f"several stacked expert weights look like a down projection ({[n for n, _ in named]})")

    # 2. linear layers that write H and are not part of attention
    cands = [(n, m) for n, m in block.named_modules()
             if n and _is_linear(m) and _out_features(m) == hidden and not _in_attention(n)]
    if not cands:
        fail("found no MLP down projection")
    pick = [c for c in cands if c[0].split(".")[-1] in _DOWN_NAMES]
    if not pick:
        pick = [c for c in cands if _in_features(c[1]) != hidden] or cands
    if len(pick) == 1:
        n, m = pick[0]
        return _Down(hook=m, path=n, mods=[m])

    # 3. several: experts are the ones that differ only by an index in their path
    fams: dict[str, list] = {}
    for n, m in pick:
        parts = n.split(".")
        idx = [i for i, p in enumerate(parts) if p.isdigit()]
        if not idx:
            continue
        key = ".".join(parts[:idx[-1]] + ["*"] + parts[idx[-1] + 1:])
        fams.setdefault(key, []).append((int(parts[idx[-1]]), m))
    fams = {k: v for k, v in fams.items() if len(v) > 1}
    if not fams:
        fail(f"several candidate down projections and no expert list ({[n for n, _ in pick]})")
    key = max(fams, key=lambda k: len(fams[k]))
    mods = [m for _, m in sorted(fams[key], key=lambda t: t[0])]
    if key.split(".")[0] == "*":
        fail(f"expert list {key} has no enclosing MoE module to hook")
    return _Down(hook=moe_hook(key), path=key, mods=mods, moe=True)


class HFBackend(Backend):
    def __init__(self, model: str, device: str = "auto", dtype: str = "auto", batch_size: int = 16,
                 **ignored):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = str(model)
        self.batch_size = max(1, int(batch_size))
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if dtype == "auto":
            # half precision is slow and badly supported on CPU
            torch_dtype = torch.float32 if self.device.type == "cpu" else "auto"
        else:
            torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        self.tok = AutoTokenizer.from_pretrained(model)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model, dtype=torch_dtype)
        except ValueError:
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(model, dtype=torch_dtype)
        self.model.to(self.device).eval()
        self.model.requires_grad_(False)

        self._vocab = self.tok.get_vocab()
        self.pad_id = self.tok.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tok.eos_token_id if self.tok.eos_token_id is not None else 0
        self.eos_ids = self._eos_ids()
        probe = self.tok("a").input_ids
        bos = self.tok.bos_token_id
        self._bos = bos if (bos is not None and probe and probe[0] == bos) else None

        emb = self.model.get_input_embeddings()
        self.hidden = int(emb.weight.shape[1])
        self.blocks = self._find_blocks()
        self.n_layers = len(self.blocks)
        self._down = [_find_down(b, self.hidden, i) for i, b in enumerate(self.blocks)]
        self.median_norm: np.ndarray | None = None
        self._dirs: dict[tuple, torch.Tensor] = {}

    # ------------------------------------------------------------------ structure

    def _eos_ids(self) -> list[int]:
        ids = set()
        gen = getattr(self.model.generation_config, "eos_token_id", None)
        for e in (gen if isinstance(gen, (list, tuple)) else [gen]) + [self.tok.eos_token_id]:
            if e is not None:
                ids.add(int(e))
        special = set(self.tok.all_special_tokens) | set(getattr(self.tok, "added_tokens_encoder", {}))
        for t in _END_OF_TURN:
            if t in self._vocab and t in special:
                ids.add(int(self._vocab[t]))
        return sorted(ids)

    def _find_blocks(self) -> nn.ModuleList:
        lists = [(n, m) for n, m in self.model.named_modules()
                 if isinstance(m, nn.ModuleList) and len(m) > 0
                 and all(not _is_linear(c) and any(True for _ in c.parameters()) for c in m)]
        seen: dict[int, list] = {}
        handles = []
        for _, ml in lists:
            for child in ml:
                def rec(mod, args, out, key=id(child)):
                    o = _first(out) if torch.is_tensor(out) or isinstance(out, (tuple, list)) else None
                    seen.setdefault(key, []).append(tuple(o.shape) if torch.is_tensor(o) else None)
                handles.append(child.register_forward_hook(rec))
        tid = self.tok.eos_token_id if self.tok.eos_token_id is not None else 0
        ids = torch.full((1, 3), int(tid), device=self.device)
        try:
            with torch.no_grad():
                self.model(input_ids=ids, attention_mask=torch.ones_like(ids))
        finally:
            for h in handles:
                h.remove()
        want = (1, 3, self.hidden)
        good = [(n, ml) for n, ml in lists
                if all(seen.get(id(c)) and all(s == want for s in seen[id(c)]) for c in ml)]
        if not good:
            raise RuntimeError(
                f"could not find the decoder block list of {type(self.model).__name__}: no ModuleList "
                f"whose children all output (batch, tokens, {self.hidden}). ModuleLists: "
                f"{[(n, len(m), type(m[0]).__name__) for n, m in lists]}")
        name, blocks = max(good, key=lambda t: len(t[1]))
        self.blocks_path = name
        return blocks

    def experts(self, layer: int) -> int:
        return self._down[layer].experts

    def write_vectors(self, layer: int) -> np.ndarray:
        d = self._down[layer]
        out = np.empty((d.groups, d.width, self.hidden), dtype=np.float32)
        for g in range(d.groups):  # one expert at a time: at most one extra copy of an (I, H) matrix
            out[g] = d.matrix(g).numpy()
        return out

    def _check(self, n: Neuron) -> _Down:
        if not 0 <= n.layer < self.n_layers:
            raise ValueError(f"{n.label()}: layer out of range (model has {self.n_layers} layers)")
        d = self._down[n.layer]
        if d.moe and n.expert is None:
            raise ValueError(f"{n.label()}: layer {n.layer} has {d.experts} experts, name one (L{n.layer} E<k> N{n.index})")
        if not d.moe and n.expert is not None:
            raise ValueError(f"{n.label()}: layer {n.layer} is a plain MLP, it has no experts")
        if d.moe and not 0 <= n.expert < d.experts:
            raise ValueError(f"{n.label()}: expert out of range (layer has {d.experts})")
        if not 0 <= n.index < d.width:
            raise ValueError(f"{n.label()}: neuron out of range (MLP width {d.width})")
        return d

    def _direction(self, n: Neuron) -> tuple[torch.Tensor, float]:
        """(unit write vector on CPU, its original norm)."""
        key = (n.layer, n.expert, n.index)
        hit = self._dirs.get(key)
        if hit is None:
            w = self._check(n).column(n.expert or 0, n.index)
            norm = float(w.norm())
            if norm == 0.0:
                raise ValueError(f"{n.label()}: write vector is all zeros, cannot steer with it")
            if len(self._dirs) >= 8192:
                self._dirs.clear()
            hit = self._dirs[key] = (w / norm, norm)
        return hit

    def _need_calibration(self) -> np.ndarray:
        if self.median_norm is None:
            raise RuntimeError("call calibrate() before steering")
        return self.median_norm

    def unit(self, neuron: Neuron) -> float:
        med = self._need_calibration()
        norm = self._direction(neuron)[1]  # validates the neuron first
        return float(med[neuron.layer]) / norm

    # ------------------------------------------------------------------ text

    def chat(self, user: str, system: str | None = None) -> str:
        if not getattr(self.tok, "chat_template", None):
            return (f"{system}\n\n" if system else "") + f"User: {user}\nAssistant:"
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        merged = [{"role": "user", "content": f"{system}\n\n{user}" if system else user}]
        last = None
        for m in ([msgs, merged] if system else [msgs]):  # some templates reject a system turn
            for kw in ({"enable_thinking": False}, {}):
                try:
                    return self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **kw)
                except Exception as e:  # jinja TemplateError, TypeError, ...
                    last = e
        raise RuntimeError(f"chat template of {self.name} failed: {last}")

    def _encode(self, prompt: str) -> list[int]:
        """Token ids of a chat-formatted prompt: its special tokens are already spelled out, so
        none are added, except a BOS when the tokenizer normally adds one and the text lacks it."""
        ids = self.tok(prompt, add_special_tokens=False).input_ids
        if self._bos is not None:
            if not ids or ids[0] != self._bos:
                ids = [self._bos] + ids
            while len(ids) > 1 and ids[1] == self._bos:  # doubled BOS
                ids = ids[1:]
        if not ids:
            ids = [self.pad_id]
        return ids

    def _pad(self, seqs: list[list[int]], left: bool) -> tuple[torch.Tensor, torch.Tensor]:
        t = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), t), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), t), dtype=torch.long)
        for i, s in enumerate(seqs):
            sl = slice(t - len(s), t) if left else slice(0, len(s))
            ids[i, sl] = torch.tensor(s, dtype=torch.long)
            mask[i, sl] = 1
        return ids.to(self.device), mask.to(self.device)

    def _batches(self, lengths: list[int], size: int | None = None):
        """Index batches, similar lengths together (less padding)."""
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        size = size or self.batch_size
        for i in range(0, len(order), size):
            yield order[i:i + size]

    # ------------------------------------------------------------------ hooks

    def _steer_hooks(self, steers: list[list[Steer]]) -> list:
        """Forward hooks adding each row's pushes. steers has one entry per batch row."""
        if not any(steers):
            return []
        med = self._need_calibration()
        add: dict[int, torch.Tensor] = {}
        for row, row_steers in enumerate(steers):
            for s in row_steers:
                unit_vec, _ = self._direction(s.neuron)
                layer = s.neuron.layer
                if layer not in add:
                    add[layer] = torch.zeros(len(steers), 1, self.hidden)
                add[layer][row, 0] += float(s.dose) * float(med[layer]) * unit_vec
        handles = []
        for layer, vec in add.items():
            def hook(mod, args, out, vec=vec):
                h = _first(out)
                v = vec.to(device=h.device, dtype=h.dtype)
                if h.dim() == 3:
                    new = h + v
                else:  # (batch * tokens, H)
                    new = (h.reshape(v.shape[0], -1, h.shape[-1]) + v).reshape(h.shape)
                return new if torch.is_tensor(out) else (new,) + tuple(out[1:])
            handles.append(self._down[layer].hook.register_forward_hook(hook))
        return handles

    def _run_blocks(self, ids: torch.Tensor, mask: torch.Tensor, fn) -> None:
        """Forward once, calling fn(layer, resid) with the stream leaving every block, and stop
        after the last block (the LM head over a long batch is large and not needed)."""
        handles = []
        last = self.n_layers - 1
        for i, b in enumerate(self.blocks):
            def hook(mod, args, out, i=i):
                fn(i, _first(out))
                if i == last:
                    raise _Stop
            handles.append(b.register_forward_hook(hook))
        try:
            with torch.no_grad():
                self.model(input_ids=ids, attention_mask=mask)
        except _Stop:
            pass
        finally:
            for h in handles:
                h.remove()

    # ------------------------------------------------------------------ measurements

    def calibrate(self, prompts: list[str]) -> dict:
        if not prompts:
            raise ValueError("calibrate needs at least one prompt")
        seqs = [self._encode(p) for p in prompts]
        norms: list[list[torch.Tensor]] = [[] for _ in range(self.n_layers)]
        for batch in self._batches([len(s) for s in seqs]):
            ids, mask = self._pad([seqs[i] for i in batch], left=False)
            keep = mask.bool().cpu()
            keep[:, 0] = False  # tokens 1: (right padded, so column 0 is every row's first token)

            def fn(layer, resid, keep=keep):
                norms[layer].append(resid.float().norm(dim=-1).cpu()[keep])
            self._run_blocks(ids, mask, fn)
        per_layer = [torch.cat(n).numpy() for n in norms]
        if per_layer[0].size == 0:
            raise ValueError("calibration prompts have no tokens after the first one")
        self.median_norm = np.array([np.median(n) for n in per_layer], dtype=np.float64)
        return {
            "median_resid_norm": [float(x) for x in self.median_norm],
            "n_prompts": len(prompts),
            "n_tokens": int(per_layer[0].size),
            "model": self.name,
        }

    def capture(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        out = np.zeros((len(pairs), self.n_layers, self.hidden), dtype=np.float32)
        seqs, spans = [], []
        for prompt, reply in pairs:
            p = self._encode(prompt)
            r = self.tok(reply, add_special_tokens=False).input_ids
            seqs.append(p + r)
            # an empty reply has no tokens to average: use the position that would produce it
            spans.append((len(p), len(p) + len(r)) if r else (len(p) - 1, len(p)))
        for batch in self._batches([len(s) for s in seqs]):
            ids, mask = self._pad([seqs[i] for i in batch], left=False)
            sel = torch.zeros(ids.shape, dtype=torch.float32, device=self.device)
            for row, i in enumerate(batch):
                sel[row, spans[i][0]:spans[i][1]] = 1.0
            count = sel.sum(1, keepdim=True)

            def fn(layer, resid, batch=batch, sel=sel, count=count):
                mean = (resid.float() * sel.to(resid.device)[..., None]).sum(1) / count.to(resid.device)
                out[batch, layer] = mean.cpu().numpy()
            self._run_blocks(ids, mask, fn)
        return out

    def _next_logits(self, prompts: list[str], steers: list[list[Steer]] | None = None) -> torch.Tensor:
        """float32 (len(prompts), vocab) logits for the token after each prompt."""
        seqs = [self._encode(p) for p in prompts]
        steers = steers if steers is not None else [[] for _ in prompts]
        if len(steers) != len(prompts):
            raise ValueError("steers must have one entry per prompt")
        vocab = self.model.get_input_embeddings().weight.shape[0]
        longest = max((len(s) for s in seqs), default=1)
        size = max(1, min(self.batch_size, int(2.5e8 // (longest * vocab))))  # ~1 GB of logits
        res: list[torch.Tensor | None] = [None] * len(prompts)
        for batch in self._batches([len(s) for s in seqs], size):
            ids, mask = self._pad([seqs[i] for i in batch], left=True)
            handles = self._steer_hooks([steers[i] for i in batch])
            try:
                with torch.no_grad():
                    logits = self.model(input_ids=ids, attention_mask=mask).logits[:, -1].float().cpu()
            finally:
                for h in handles:
                    h.remove()
            for row, i in enumerate(batch):
                res[i] = logits[row]
        return torch.stack(res) if res else torch.zeros(0, vocab)

    def _first_ids(self, word: str) -> list[int]:
        ids = set()
        for text in (word, " " + word):
            t = self.tok(text, add_special_tokens=False).input_ids
            if t:
                ids.add(t[0])
        if not ids:
            raise ValueError(f"{word!r} has no tokens")
        return sorted(ids)

    def yes_no(self, prompts: list[str], yes: str = "Yes", no: str = "No") -> list[float]:
        if not prompts:
            return []
        y, n = self._first_ids(yes), self._first_ids(no)
        logits = self._next_logits(prompts)
        return [float(v) for v in (logits[:, y].max(dim=1).values - logits[:, n].max(dim=1).values)]

    # ------------------------------------------------------------------ generation

    def generate(self, prompts: list[str], steers: list[list[Steer]] | None = None,
                 max_tokens: int = 100) -> list[str]:
        steers = steers if steers is not None else [[] for _ in prompts]
        if len(steers) != len(prompts):
            raise ValueError("steers must have one entry per prompt")
        seqs = [self._encode(p) for p in prompts]
        replies = [""] * len(prompts)
        for batch in self._batches([len(s) for s in seqs]):
            ids, mask = self._pad([seqs[i] for i in batch], left=True)
            handles = self._steer_hooks([steers[i] for i in batch])
            try:
                with torch.no_grad():
                    out = self.model.generate(
                        input_ids=ids, attention_mask=mask, max_new_tokens=int(max_tokens),
                        do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None,
                        repetition_penalty=None, eos_token_id=self.eos_ids or None,
                        pad_token_id=self.pad_id)
            finally:
                for h in handles:
                    h.remove()
            new = out[:, ids.shape[1]:].cpu()
            for row, i in enumerate(batch):
                toks = new[row].tolist()
                for j, t in enumerate(toks):  # cut at the first stop token (rows finish at different steps)
                    if t in self.eos_ids:
                        toks = toks[:j]
                        break
                replies[i] = self.tok.decode(toks, skip_special_tokens=True).strip()
        return replies

    def close(self) -> None:
        self.model = None
        self.blocks = None
        self._down = []
        self._dirs.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
