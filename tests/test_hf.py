"""HF backend tests. CPU only; tests that need a hub model skip when it cannot be loaded."""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from torch import nn  # noqa: E402

from skrrt.backends import Neuron, Steer, load_backend  # noqa: E402
from skrrt.backends.hf import HFBackend, _find_down  # noqa: E402

SMOL = "HuggingFaceTB/SmolLM2-135M-Instruct"
GPT2 = "hf-internal-testing/tiny-random-gpt2"
MIXTRAL = "hf-internal-testing/tiny-random-MixtralForCausalLM"
QWEN_MOE = "yujiepan/qwen3-moe-tiny-random"        # layer 0 plain MLP, layer 1 has 8 experts
QWEN35_MOE = "tiny-random/qwen3.5-moe"             # experts plus a shared expert


def _load(model_id):
    try:
        return load_backend(model_id, backend="hf", device="cpu", batch_size=4)
    except Exception as e:  # no network, model gone, unsupported by this transformers, ...
        pytest.skip(f"cannot load {model_id}: {type(e).__name__}: {str(e)[:200]}")


@pytest.fixture(scope="module")
def smol():
    b = _load(SMOL)
    b.calibrate([b.chat(q) for q in ("What is the capital of France?", "Tell me about your day.",
                                     "How do I boil an egg?")])
    return b


@pytest.fixture(scope="module")
def prompts(smol):
    return [smol.chat("Hi!"),
            smol.chat("Please explain in a few words why the sky looks blue during the day."),
            smol.chat("Name a fruit.", system="You are a helpful assistant.")]


def _block_outputs(b, prompt, steers=None):
    """Residual leaving every block for one prompt, via the test's own hooks."""
    got = {}
    hs = [blk.register_forward_hook(
        lambda m, a, out, i=i: got.__setitem__(i, (out if torch.is_tensor(out) else out[0])[0].float().clone()))
        for i, blk in enumerate(b.blocks)]
    try:
        b._next_logits([prompt], None if steers is None else [steers])
    finally:
        for h in hs:
            h.remove()
    return got


# ---------------------------------------------------------------- structure (no network)

class _Attn(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.q_proj, self.o_proj = nn.Linear(h, h), nn.Linear(h, h)


def test_detection_error_names_submodules():
    block = nn.Module()
    block.self_attn = _Attn(8)
    block.norm = nn.LayerNorm(8)
    with pytest.raises(RuntimeError) as e:
        _find_down(block, 8, 3)
    assert "layer 3" in str(e.value) and "self_attn.o_proj" in str(e.value) and "norm" in str(e.value)


def test_detects_plain_mlp_and_ignores_attention():
    block = nn.Module()
    block.self_attn = _Attn(8)
    block.mlp = nn.Module()
    block.mlp.up, block.mlp.out = nn.Linear(8, 32), nn.Linear(32, 8)   # unknown names
    d = _find_down(block, 8, 0)
    assert d.hook is block.mlp.out and d.experts == 0 and d.width == 32


def test_detects_module_list_experts():
    """Old Mixtral layout: block_sparse_moe.experts[i].w2, plus a shared expert to ignore."""
    block = nn.Module()
    block.self_attn = _Attn(8)
    moe = block.block_sparse_moe = nn.Module()
    moe.gate = nn.Linear(8, 3)
    moe.experts = nn.ModuleList()
    for _ in range(3):
        e = nn.Module()
        e.w1, e.w2, e.w3 = nn.Linear(8, 16), nn.Linear(16, 8), nn.Linear(8, 16)
        moe.experts.append(e)
    moe.shared_expert = nn.Module()
    moe.shared_expert.w2 = nn.Linear(16, 8)
    d = _find_down(block, 8, 0)
    assert d.experts == 3 and d.hook is moe and d.width == 16
    assert torch.equal(d.column(2, 5), moe.experts[2].w2.weight[:, 5])
    assert torch.equal(d.matrix(1), moe.experts[1].w2.weight.t())


# ---------------------------------------------------------------- SmolLM2 (plain MLP)

def test_structure(smol):
    assert isinstance(smol, HFBackend)
    assert (smol.n_layers, smol.hidden) == (30, 576)
    assert all(smol.experts(i) == 0 for i in range(smol.n_layers))
    w = smol.write_vectors(7)
    assert w.dtype == np.float32 and w.shape == (1, 1536, 576)
    ref = smol.model.model.layers[7].mlp.down_proj.weight
    assert np.array_equal(w[0, 123], ref[:, 123].float().numpy())


def test_chat_format(smol):
    p = smol.chat("Hello?", system="Be nice.")
    assert "Hello?" in p and "Be nice." in p and p.rstrip().endswith("assistant")
    ids = smol._encode(p)
    assert ids == smol.tok(p, add_special_tokens=False).input_ids
    bos = smol.tok.bos_token_id
    assert not (len(ids) > 1 and ids[0] == bos and ids[1] == bos)


def test_calibrate(smol):
    p = [smol.chat("What is two plus two?")]
    old = smol.median_norm.copy()
    try:
        rec = smol.calibrate(p)
        med = rec["median_resid_norm"]
        assert len(med) == smol.n_layers and all(m > 0 for m in med)
        blocks = _block_outputs(smol, p[0])
        n_tok = len(smol._encode(p[0]))
        assert rec["n_tokens"] == n_tok - 1
        for layer in (0, 11, 29):
            want = np.median(blocks[layer][1:].norm(dim=-1).numpy())
            assert med[layer] == pytest.approx(float(want), rel=1e-4)
        # padding must not leak in: a batch with a much longer prompt gives the pooled median
        both = smol.calibrate(p + [smol.chat("Write a long sentence about rivers, lakes and the sea.")])
        assert both["n_tokens"] > rec["n_tokens"] and all(m > 0 for m in both["median_resid_norm"])
    finally:
        smol.median_norm = old


def test_unit(smol):
    n = Neuron(12, 345)
    w = smol.write_vectors(12)[0, 345]
    assert smol.unit(n) == pytest.approx(smol.median_norm[12] / np.linalg.norm(w), rel=1e-5)
    with pytest.raises(ValueError):
        smol.unit(Neuron(12, 345, expert=0))
    with pytest.raises(ValueError):
        smol.unit(Neuron(99, 0))


@pytest.mark.parametrize("layer,index,dose", [(5, 100, 3.0), (20, 7, -2.0)])
def test_hook_equals_adding_delta_to_the_neuron(smol, prompts, layer, index, dose):
    """The recipe itself: delta added to neuron n at the input of the down projection."""
    neuron = Neuron(layer, index)
    steered = smol._next_logits(prompts, [[Steer(neuron, dose)]] * len(prompts))
    plain = smol._next_logits(prompts)
    delta = dose * smol.unit(neuron)

    def pre(mod, args):
        x = args[0].clone()
        x[..., index] += delta
        return (x,) + tuple(args[1:])

    h = smol.model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(pre)
    try:
        literal = smol._next_logits(prompts)
    finally:
        h.remove()
    assert (literal - plain).abs().max() > 0.05          # the push does something
    assert torch.allclose(steered, literal, atol=2e-3, rtol=1e-4)
    assert torch.allclose(smol._next_logits(prompts), plain)   # hooks are gone afterwards


def test_per_row_steers_match_one_at_a_time(smol, prompts):
    steers = [[Steer(Neuron(8, 11), 2.0)],
              [],
              [Steer(Neuron(8, 400), -1.5), Steer(Neuron(15, 3), 1.0), Steer(Neuron(8, 11), 0.5)]]
    together = smol._next_logits(prompts, steers)
    alone = torch.cat([smol._next_logits([p], [s]) for p, s in zip(prompts, steers)])
    assert torch.allclose(together, alone, atol=5e-3, rtol=1e-4)
    assert (together[0] - smol._next_logits([prompts[0]])[0]).abs().max() > 0.01

    replies = smol.generate(prompts, steers, max_tokens=12)
    singles = [smol.generate([p], [s], max_tokens=12)[0] for p, s in zip(prompts, steers)]
    assert replies == singles


def test_empty_steers_equal_unsteered(smol, prompts):
    plain = smol.generate(prompts, max_tokens=12)
    assert smol.generate(prompts, [[] for _ in prompts], max_tokens=12) == plain
    assert all(isinstance(r, str) and r == r.strip() for r in plain) and any(plain)
    mixed = smol.generate(prompts, [[Steer(Neuron(10, 5), 4.0)], [], []], max_tokens=12)
    assert mixed[1:] == plain[1:]
    assert not any(p.rstrip()[-20:] in r for p, r in zip(prompts, plain))   # prompt not echoed


def test_generate_stops_at_end_of_turn(smol, prompts):
    r = smol.generate([smol.chat("Reply with the single word: OK")], max_tokens=60)[0]
    assert "<|im_end|>" not in r and "<|im_start|>" not in r


def test_capture(smol, prompts):
    pairs = [(prompts[0], "Hello there, nice to meet you."),
             (prompts[1], "Because of scattering."),
             (prompts[2], "Apple")]
    got = smol.capture(pairs)
    assert got.dtype == np.float32 and got.shape == (3, smol.n_layers, smol.hidden)
    for i, (p, r) in enumerate(pairs):     # batching and padding do not change a row
        assert np.allclose(got[i], smol.capture([(p, r)])[0], atol=2e-3, rtol=1e-3)
    # reply tokens only: equals the hand-computed mean over the reply positions ...
    p, r = pairs[1]
    n_p = len(smol._encode(p))
    blocks = _block_outputs(smol, p + r)
    for layer in (0, 14, 29):
        want = blocks[layer][n_p:].mean(0).numpy()
        assert np.allclose(got[1, layer], want, atol=2e-3, rtol=1e-3)
        assert not np.allclose(got[1, layer], blocks[layer].mean(0).numpy(), atol=2e-3, rtol=1e-3)
    # ... and a different prompt of the same reply still moves it only through attention,
    # while the same prompt with a different reply gives a different vector
    assert not np.allclose(got[1], smol.capture([(p, "No idea, sorry.")])[0], atol=1e-2)


def test_yes_no(smol):
    p = [smol.chat("Is water wet? Answer Yes or No."), smol.chat("Hi")]
    got = smol.yes_no(p)
    logits = smol._next_logits(p)
    tok = smol.tok

    def best(word):
        ids = {tok(word, add_special_tokens=False).input_ids[0], tok(" " + word, add_special_tokens=False).input_ids[0]}
        return logits[:, sorted(ids)].max(dim=1).values

    want = best("Yes") - best("No")
    assert got == pytest.approx(want.tolist(), abs=1e-4)
    assert smol.yes_no([]) == []


def test_steering_needs_calibration(smol, prompts):
    old, smol.median_norm = smol.median_norm, None
    try:
        with pytest.raises(RuntimeError):
            smol.generate(prompts[:1], [[Steer(Neuron(1, 1), 1.0)]], max_tokens=2)
        assert smol.generate(prompts[:1], max_tokens=2)   # unsteered works without it
    finally:
        smol.median_norm = old


# ---------------------------------------------------------------- GPT-2 (Conv1D, no chat template)

def test_gpt2_conv1d():
    b = _load(GPT2)
    p = [b.chat("Hello there", system="Be brief."), b.chat("Hi")]
    assert p[0] == "Be brief.\n\nUser: Hello there\nAssistant:"
    b.calibrate(p)
    conv = b.blocks[2].mlp.c_proj
    assert type(conv).__name__ == "Conv1D" and b._down[2].hook is conv
    w = b.write_vectors(2)
    assert w.shape == (1, conv.weight.shape[0], b.hidden)
    assert np.array_equal(w[0, 9], conv.weight[9].numpy())        # Conv1D weight is (in, out)
    neuron, dose = Neuron(2, 9), 5.0
    steered = b._next_logits(p, [[Steer(neuron, dose)]] * 2)

    def pre(mod, args):
        x = args[0].clone()
        x[..., 9] += dose * b.unit(neuron)
        return (x,)

    h = conv.register_forward_pre_hook(pre)
    try:
        literal = b._next_logits(p)
    finally:
        h.remove()
    assert torch.allclose(steered, literal, atol=1e-4, rtol=1e-4)
    assert (steered - b._next_logits(p)).abs().max() > 1e-4


# ---------------------------------------------------------------- mixture of experts

def _check_moe_push(b, layer, expert, index, dose=2.0):
    """Steering one expert's neuron moves the stream leaving that layer by exactly the fixed
    vector at every token, whatever the router picked."""
    p = b.chat("The quick brown fox jumps over the lazy dog and keeps running")
    w = torch.from_numpy(b.write_vectors(layer)[expert, index])
    vec = dose * float(b.median_norm[layer]) * w / w.norm()
    plain = _block_outputs(b, p)[layer]
    pushed = _block_outputs(b, p, [Steer(Neuron(layer, index, expert), dose)])[layer]
    assert plain.shape[0] > 3
    assert torch.allclose(pushed - plain, vec.expand_as(plain), atol=1e-5 + 1e-4 * float(vec.abs().max()))
    assert float(vec.norm()) == pytest.approx(abs(dose) * b.median_norm[layer], rel=1e-4)
    assert b.unit(Neuron(layer, index, expert)) == pytest.approx(b.median_norm[layer] / float(w.norm()), rel=1e-4)


def test_mixtral_moe():
    b = _load(MIXTRAL)
    b.calibrate([b.chat("Hello there, how are you today?"), b.chat("Tell me a story")])
    assert [b.experts(i) for i in range(b.n_layers)] == [4, 4]
    w = b.write_vectors(1)
    stacked = b.blocks[1].mlp.experts.down_proj          # (E, H, I)
    assert w.dtype == np.float32 and w.shape == (4, stacked.shape[2], b.hidden)
    assert np.array_equal(w[3, 17], stacked[3, :, 17].numpy())
    _check_moe_push(b, 1, 3, 17)
    _check_moe_push(b, 0, 0, 5, dose=-1.0)
    with pytest.raises(ValueError):
        b.unit(Neuron(1, 17))                            # expert missing
    p = [b.chat("Hi"), b.chat("Hello there my friend")]
    steers = [[Steer(Neuron(1, 17, 3), 3.0)], [Steer(Neuron(0, 2, 1), -2.0)]]
    together = b._next_logits(p, steers)
    alone = torch.cat([b._next_logits([q], [s]) for q, s in zip(p, steers)])
    assert torch.allclose(together, alone, atol=1e-4, rtol=1e-3)
    assert len(b.generate(p, steers, max_tokens=4)) == 2


def test_qwen_moe_mixed_layers():
    b = _load(QWEN_MOE)
    b.calibrate([b.chat("Hello there, how are you today?")])
    assert [b.experts(i) for i in range(b.n_layers)] == [0, 8]
    assert b.write_vectors(0).shape[0] == 1 and b.write_vectors(1).shape[0] == 8
    _check_moe_push(b, 1, 6, 3)
    assert "<think>" not in b.chat("Hi") or "</think>" in b.chat("Hi")     # thinking switched off


def test_qwen35_moe_with_shared_expert():
    b = _load(QWEN35_MOE)
    b.calibrate([b.chat("Hello there, how are you today?")])
    n = b.experts(0)
    assert n > 1 and b.write_vectors(0).shape[0] == n and b.write_vectors(0).shape[2] == b.hidden
    assert type(b._down[0].hook).__name__.endswith("SparseMoeBlock")
    _check_moe_push(b, 2, n - 1, 1)
