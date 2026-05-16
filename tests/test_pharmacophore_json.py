import pytest

from pypharao import (
    MoleculePharmacophore,
    Pharmacophore,
    PharmacophorePoint,
    PointType,
    QueryPharmacophore,
)


def test_query_json_roundtrip_preserves_subclass_and_name():
    q = QueryPharmacophore(name="q1")
    q.add_point(PharmacophorePoint(type=PointType.HDON, center=(1, 2, 3)))
    q.add_point(
        PharmacophorePoint(
            type=PointType.AROM, center=(0, 0, 0), normal=(0, 0, 1)
        )
    )

    parsed = Pharmacophore.from_json(q.to_json())
    assert isinstance(parsed, QueryPharmacophore)
    assert parsed.get_name() == "q1"
    assert len(parsed) == 2
    assert parsed[0].type == PointType.HDON
    assert parsed[1].type == PointType.AROM
    assert parsed[1].normal == (0.0, 0.0, 1.0)


def test_molecule_json_roundtrip_preserves_subclass():
    m = MoleculePharmacophore()
    m.add_point(PharmacophorePoint(type=PointType.HACC, center=(1, 0, 0)))
    parsed = Pharmacophore.from_json(m.to_json())
    assert isinstance(parsed, MoleculePharmacophore)
    assert len(parsed) == 1
    assert parsed[0].type == PointType.HACC


def test_phar_roundtrip_round_trips_through_query():
    text = """x
HDON\t0\t0\t0\t1.0\t0\t0\t0\t0
$$$$
"""
    p = Pharmacophore.from_phar_text(text)
    assert isinstance(p, QueryPharmacophore)
    assert p.get_name() == "x"
    assert p[0].type == PointType.HDON


def test_molecule_pharmacophore_rejects_excl():
    m = MoleculePharmacophore()
    with pytest.raises(ValueError, match="EXCL"):
        m.add_point(PharmacophorePoint(type=PointType.EXCL, center=(0, 0, 0)))


def test_pharmacophorepoint_type_nested_alias():
    assert PharmacophorePoint.Type is PointType


def test_update_point_changes_type_in_place():
    q = QueryPharmacophore(name="upd")
    q.add_point(
        PharmacophorePoint(
            type=PointType.AROM, center=(1.0, 2.0, 3.0), normal=(0.0, 0.0, 1.0)
        )
    )

    q.update_point(0, type=PointType.AROM_OR_LIPO)

    assert q[0].type == PointType.AROM_OR_LIPO
    assert q[0].center == (1.0, 2.0, 3.0)
    assert q[0].normal == (0.0, 0.0, 1.0)


def test_update_point_can_change_multiple_fields():
    q = QueryPharmacophore()
    q.add_point(PharmacophorePoint(type=PointType.HDON, center=(0.0, 0.0, 0.0)))

    q.update_point(0, center=(1.0, 2.0, 3.0), sigma=1.25)

    assert q[0].center == (1.0, 2.0, 3.0)
    assert q[0].sigma == 1.25
    assert q[0].type == PointType.HDON


def test_update_point_enforces_subclass_allowed_types():
    m = MoleculePharmacophore()
    m.add_point(PharmacophorePoint(type=PointType.HACC, center=(0.0, 0.0, 0.0)))

    with pytest.raises(ValueError, match="EXCL"):
        m.update_point(0, type=PointType.EXCL)
    assert m[0].type == PointType.HACC
