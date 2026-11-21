from skrrt.backends import Neuron


def test_neuron_roundtrip():
    for s in ("L34 N3838", "L11 E14 N370"):
        assert Neuron.parse(s).label() == s
    assert Neuron.parse("l7:e78:n1060") == Neuron(7, 1060, 78)
