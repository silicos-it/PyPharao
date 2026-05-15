"""Golden-style `.phar` parsing (inline fixture; extend with Pharao outputs when available)."""

from pypharao import FuncGroup, Pharmacophore


def test_minimal_phar_fixture():
    text = """demo
AROM	1.0	0	0	0.7	1	1.0	0.0	0.0
LIPO	3.0	0	0	0.7	0	0.0	0.0	0.0
$$$$
"""
    p = Pharmacophore.from_phar_text(text)
    assert len(p.points) == 2
    assert p.points[0].func == FuncGroup.AROM
    assert p.points[1].func == FuncGroup.LIPO
