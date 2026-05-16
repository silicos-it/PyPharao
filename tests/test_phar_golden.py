"""Golden-style `.phar` parsing (inline fixture; extend with Pharao outputs when available)."""

from pypharao import Pharmacophore, PointType, QueryPharmacophore


def test_minimal_phar_fixture():
    text = """demo
AROM\t1.0\t0\t0\t0.7\t1\t1.0\t0.0\t0.0
LIPO\t3.0\t0\t0\t0.7\t0\t0.0\t0.0\t0.0
$$$$
"""
    p = Pharmacophore.from_phar_text(text)
    assert isinstance(p, QueryPharmacophore)
    assert p.get_name() == "demo"
    assert len(p) == 2
    assert p[0].type == PointType.AROM
    assert p[1].type == PointType.LIPO
