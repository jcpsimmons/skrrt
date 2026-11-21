"""llama.cpp backend tests. CPU only.

The model tests need the engine (`SKRRT_CUDA=0 llamacpp/build.sh`) and a small ordinary chat GGUF
($SKRRT_TEST_GGUF, default ~/models/skrrt-test/Qwen3-0.6B-Q8_0.gguf); they skip when either is missing.
"""
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gguf")
pytest.importorskip("jinja2")

from skrrt.backends import Neuron, Steer, load_backend  # noqa: E402
from skrrt.backends import llamacpp as lc  # noqa: E402

MODEL = Path(os.environ.get("SKRRT_TEST_GGUF", "~/models/skrrt-test/Qwen3-0.6B-Q8_0.gguf")).expanduser()


# ---------------------------------------------------------------- no engine, no model

def test_missing_engine_says_how_to_build(tmp_path):
    with pytest.raises(FileNotFoundError, match="build.sh"):
        lc.find_engine(tmp_path / "nope")


def test_template_strips_unknown_tags_and_turns_thinking_off():
    src = ("{% for m in messages %}<{{ m.role }}>{% generation %}{{ m.content }}{% endgeneration %}</>{% endfor %}"
           "{% if add_generation_prompt %}<assistant>{% if enable_thinking is defined and not enable_thinking %}"
           "<nothink>{% endif %}{% endif %}")
    t = lc._Template(src, "<s>", "</s>")
    assert t.render("hi", "sys") == "<system>sys</><user>hi</><assistant><nothink>"


def test_template_is_sandboxed():
    t = lc._Template("{{ ''.__class__.__mro__[1].__subclasses__() }}", "", "")
    with pytest.raises(RuntimeError):
        t.render("hi", None)


def test_template_without_system_role_merges_it():
    src = ("{% for m in messages %}{% if m.role == 'system' %}{{ raise_exception('no system') }}{% endif %}"
           "[{{ m.content }}]{% endfor %}")
    assert lc._Template(src, "", "").render("hi", "sys") == "[sys\n\nhi]"


def test_inkling2_chat_format():
    s = lc._inkling2_chat("hi", "be brief")
    assert s == ("<|message_system|><|content_text|>be brief<|end_message|>"
                 "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
                 "<|message_user|><|content_text|>hi<|end_message|><|message_model|><|content_text|>")


def test_loop_slots_match_inkling2_layout():
    b = object.__new__(lc.LlamaCppBackend)
    b.n_layers, b._core0, b._core1, b._passes = 24, 6, 18, 3
    assert b.slots(5) == [5] and b.slots(18) == [18] and b.slots(23) == [23]
    assert b.slots(6) == [6, 24, 36] and b.slots(17) == [17, 35, 47]
    assert b.slots(11) == [11, 29, 41]
    seen = sorted(s for l in range(24) for s in b.slots(l))
    assert seen == list(range(48))
    means = np.arange(48, dtype=np.float32).reshape(1, 48, 1)
    b.n_slots = 48
    per = b._per_layer(means)
    assert per.shape == (1, 24, 1) and per[0, 6, 0] == pytest.approx((6 + 24 + 36) / 3) and per[0, 3, 0] == 3


# ---------------------------------------------------------------- with engine and model

def _engine():
    try:
        return lc.find_engine(prefer_cpu=True)
    except FileNotFoundError:
        return None


@pytest.fixture(scope="module")
def be():
    if _engine() is None:
        pytest.skip("skrrt-engine is not built (SKRRT_CUDA=0 llamacpp/build.sh)")
    if not MODEL.is_file():
        pytest.skip(f"test model {MODEL} is absent")
    b = load_backend(str(MODEL), n_gpu_layers=0, n_seq=4, n_ctx=256, threads=8, device="ignored-extra-kw")
    yield b
    b.close()


QUESTIONS = ["What is the capital of France?", "Name a fruit that is yellow.", "How many legs does a spider have?",
             "Write one sentence about rain."]


@pytest.fixture(scope="module")
def prompts(be):
    return [be.chat(q) for q in QUESTIONS]


@pytest.fixture(scope="module")
def calib(be, prompts):
    return be.calibrate(prompts)


@pytest.fixture(scope="module")
def neuron(be):
    layer = be.n_layers // 3
    i = be.write_vectors(layer).shape[1]
    return Neuron(layer, int(np.random.default_rng(0).integers(i)))


def test_metadata_and_chat(be):
    assert be.n_layers > 1 and be.hidden > 0 and be.experts(1) == 0
    assert be.n_slots == be.n_layers and be.slots(3) == [3]
    p = be.chat("Hello there", system="Be brief.")
    assert "Hello there" in p and "Be brief." in p
    assert p.index("Be brief.") < p.index("Hello there")
    if "qwen3" in be.arch:
        assert p.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")  # thinking switched off


def test_write_vectors_match_gguf(be):
    w = be.write_vectors(2)
    i = int(be._gguf.hp("feed_forward_length"))
    assert w.shape == (1, i, be.hidden) and w.dtype == np.float32 and np.isfinite(w).all()
    n = Neuron(2, 7)
    col, norm = be._column(n)
    assert np.array_equal(col, w[0, 7]) and norm == pytest.approx(float(np.linalg.norm(w[0, 7])))


def test_steering_needs_calibration():
    if _engine() is None or not MODEL.is_file():
        pytest.skip("engine or model absent")
    b = load_backend(str(MODEL), n_gpu_layers=0, n_seq=1, n_ctx=64, threads=4)
    try:
        with pytest.raises(RuntimeError, match="calibrate"):
            b.unit(Neuron(2, 7))
    finally:
        b.close()
    assert b._p is None and not b._tmp.exists()


def test_calibrate(be, calib):
    med = calib["median_resid_norm"]
    assert len(med) == be.n_layers and all(np.isfinite(med)) and all(m > 0 for m in med)
    assert calib["n_prompts"] == len(QUESTIONS) and calib["n_tokens"] > 0
    import json
    json.dumps(calib)


def test_unit(be, calib, neuron):
    w = be.write_vectors(neuron.layer)[0, neuron.index]
    assert be.unit(neuron) == pytest.approx(calib["median_resid_norm"][neuron.layer] / np.linalg.norm(w), rel=1e-5)
    with pytest.raises(ValueError):
        be.unit(Neuron(neuron.layer, 10 ** 9))
    with pytest.raises(ValueError):
        be.unit(Neuron(neuron.layer, 0, expert=3))


def test_capture_shape(be, prompts):
    pairs = [(prompts[0], "The capital of France is Paris."), (prompts[1], "A banana.")]
    r = be.capture(pairs)
    assert r.shape == (2, be.n_layers, be.hidden) and r.dtype == np.float32 and np.isfinite(r).all()
    assert not np.allclose(r[0], r[1])
    # one entry at a time gives the same thing: entries do not leak into each other
    assert np.allclose(be.capture(pairs[1:])[0], r[1], rtol=1e-4, atol=1e-4)
    assert be.capture([]).shape == (0, be.n_layers, be.hidden)


def test_generate_steered_vs_unsteered(be, prompts, calib, neuron):
    base = be.generate(prompts[:2], max_tokens=24)
    assert len(base) == 2 and all(isinstance(t, str) and t for t in base)
    assert "Paris" in base[0]
    assert be.generate(prompts[:2], [[], []], max_tokens=24) == base
    assert be.generate(prompts[:2], [[Steer(neuron, 0.0)]] * 2, max_tokens=24) == base
    big = be.generate(prompts[:2], [[Steer(neuron, 8.0)]] * 2, max_tokens=24)
    assert big[0] != base[0] and big[1] != base[1]
    # the steer does not outlive the call
    assert be.generate(prompts[:2], max_tokens=24) == base


def test_mixed_rows_keep_input_order(be, prompts, calib, neuron):
    a, b = [Steer(neuron, 8.0)], [Steer(neuron, -8.0)]
    rows = [[], a, [], b, a, []]
    ps = [prompts[i % len(prompts)] for i in range(len(rows))]
    mixed = be.generate(ps, rows, max_tokens=16)
    assert len(mixed) == len(rows)
    for p, row, got in zip(ps, rows, mixed):
        assert got == be.generate([p], [row], max_tokens=16)[0]
    assert mixed[1] != mixed[0] or mixed[4] != mixed[2]


def test_more_prompts_than_sequences(be, prompts):
    ps = (prompts * 3)[:9]  # n_seq is 4
    out = be.generate(ps, max_tokens=8)
    assert out[:4] == out[4:8] and out[8] == out[0]


def test_layer0_cannot_be_steered(be, calib):
    with pytest.raises(ValueError, match="layer 0"):
        be.generate(["x"], [[Steer(Neuron(0, 1), 1.0)]], max_tokens=1)


def test_yes_no(be):
    ps = [be.chat("Is Paris the capital of France? Answer Yes or No."),
          be.chat("Is the Sun colder than ice? Answer Yes or No.")]
    v = be.yes_no(ps)
    assert len(v) == 2 and all(isinstance(x, float) and np.isfinite(x) for x in v)
    assert v[0] > v[1]
    assert be.yes_no(ps, yes="No", no="Yes") == pytest.approx([-x for x in v], abs=1e-4)
    assert be.yes_no([]) == []


def test_push_is_the_recipe_vector(be, prompts, calib, neuron):
    """The residual leaving layer L moves by exactly dose * median_norm * w / ||w||, and nothing
    before layer L moves (teacher-forced on the same tokens)."""
    pairs = [(prompts[0], "The capital of France is Paris."), (prompts[3], "Rain fell softly on the roof.")]
    dose = 1.5
    r = be.check_push(Steer(neuron, dose), pairs)
    assert r["slot"] == neuron.layer
    assert r["max_change_before_layer"] == 0.0
    assert r["rel_err"] < 1e-3 and r["cosine"] > 0.999999
    assert r["push_norm"] == pytest.approx(dose * calib["median_resid_norm"][neuron.layer], rel=1e-4)
    # the same through the public capture(): difference at layer L equals unit * dose * W_down[:, n]
    w = be.write_vectors(neuron.layer)[0, neuron.index]
    want = be.unit(neuron) * dose * w
    d = be.capture(pairs, [Steer(neuron, dose)]) - be.capture(pairs)
    assert np.allclose(d[:, neuron.layer], want, rtol=0, atol=2e-3 * np.linalg.norm(want) / np.sqrt(be.hidden) + 1e-4)
    assert np.abs(d[:, neuron.layer + 1:]).max() > 0  # and it propagates downstream


def test_export_cvec(be, calib, neuron, tmp_path):
    from gguf import GGUFReader
    other = Neuron(neuron.layer + 2, 5)
    steers = [Steer(neuron, 0.75), Steer(other, -2.0), Steer(Neuron(neuron.layer, 3), 0.0)]
    out = tmp_path / "steer.gguf"
    rec = lc.export_cvec(be, steers, out)
    r = GGUFReader(str(out))
    assert r.fields["general.architecture"].contents() == "controlvector"
    assert r.fields["controlvector.layer_count"].contents() == 2
    got = {t.name: np.array(t.data) for t in r.tensors}
    assert sorted(got) == sorted(f"direction.{l}" for l in (neuron.layer, other.layer)) and rec["slots"] == sorted(
        [neuron.layer, other.layer])
    for s in steers[:2]:
        w = be.write_vectors(s.neuron.layer)[0, s.neuron.index]
        v = got[f"direction.{s.neuron.layer}"]
        assert v.dtype == np.float32 and v.shape == (be.hidden,)
        assert np.allclose(v, be.unit(s.neuron) * s.dose * w, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError):
        lc.export_cvec(be, [Steer(neuron, 0.0)], tmp_path / "empty.gguf")
