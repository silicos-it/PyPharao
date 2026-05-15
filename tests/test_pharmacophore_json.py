
from pypharao import FuncGroup, Pharmacophore, PharmacophorePoint


def test_json_roundtrip():
    p = Pharmacophore(
        name="q1",
        points=[
            PharmacophorePoint(1, 2, 3, FuncGroup.HDON, 1.0, True, 1, 2, 4),
        ],
    )
    s = p.to_json()
    q = Pharmacophore.from_json(s)
    assert q.name == "q1"
    assert len(q.points) == 1
    assert q.points[0].func == FuncGroup.HDON
    assert q.points[0].nx == 1.0


def test_phar_roundtrip():
    text = """x
HDON	0	0	0	1.0	1	0	0	1
$$$$
"""
    p = Pharmacophore.from_phar_text(text)
    assert p.points[0].func == FuncGroup.HDON
    assert p.points[0].has_normal is True
