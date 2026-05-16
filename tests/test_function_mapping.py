import pytest

from pypharao import (
    PharmacophorePoint,
    PointType,
    QueryPharmacophore,
    MoleculePharmacophore,
)
from pypharao.function_mapping import (
    FunctionMapping,
    database_types_for_query,
    functions_compatible,
)


@pytest.mark.parametrize(
    "ref,db,expected",
    [
        (PointType.AROM, PointType.AROM, True),
        (PointType.AROM, PointType.LIPO, False),
        (PointType.LIPO, PointType.LIPO, True),
        (PointType.HDON, PointType.HDON, True),
        (PointType.HACC, PointType.HACC, True),
        (PointType.HACC, PointType.HDON, False),
        (PointType.POSC, PointType.POSC, True),
        (PointType.NEGC, PointType.NEGC, True),
        (PointType.POSC, PointType.NEGC, False),
        # Compound query types
        (PointType.AROM_OR_LIPO, PointType.AROM, True),
        (PointType.AROM_OR_LIPO, PointType.LIPO, True),
        (PointType.AROM_OR_LIPO, PointType.HDON, False),
        (PointType.HACC_OR_HDON, PointType.HACC, True),
        (PointType.HACC_OR_HDON, PointType.HDON, True),
        (PointType.HACC_OR_HDON, PointType.LIPO, False),
        (PointType.HACC_AND_HDON, PointType.HACC_AND_HDON, True),
        (PointType.HACC_AND_HDON, PointType.HACC, False),
        (PointType.HACC_AND_HDON, PointType.HDON, False),
        # EXCL never matches anything
        (PointType.EXCL, PointType.AROM, False),
        (PointType.EXCL, PointType.HACC_AND_HDON, False),
        # UNDEF matches every molecule type
        (PointType.UNDEF, PointType.AROM, True),
        (PointType.UNDEF, PointType.LIPO, True),
        (PointType.UNDEF, PointType.HACC_AND_HDON, True),
        (PointType.UNDEF, PointType.NEGC, True),
    ],
)
def test_functions_compatible(ref, db, expected):
    assert functions_compatible(ref, db) is expected


def test_database_types_for_query_or_types():
    assert database_types_for_query(PointType.AROM_OR_LIPO) == frozenset(
        {PointType.AROM, PointType.LIPO}
    )
    assert database_types_for_query(PointType.HACC_OR_HDON) == frozenset(
        {PointType.HDON, PointType.HACC}
    )
    assert database_types_for_query(PointType.HACC_AND_HDON) == frozenset(
        {PointType.HACC_AND_HDON}
    )
    assert database_types_for_query(PointType.EXCL) == frozenset()


def test_function_mapping_single_pair_lipo():
    ref = QueryPharmacophore()
    ref.add_point(PharmacophorePoint(type=PointType.LIPO, center=(0, 0, 0)))
    db = MoleculePharmacophore()
    db.add_point(PharmacophorePoint(type=PointType.LIPO, center=(0, 0, 0)))
    fm = FunctionMapping(ref, db, 0.5)
    assert fm.get_next_map() == [(0, 0)]


def test_function_mapping_or_types_match_separate_molecule_points():
    ref = QueryPharmacophore()
    ref.add_point(
        PharmacophorePoint(
            type=PointType.AROM_OR_LIPO, center=(0, 0, 0), normal=(0, 0, 1)
        )
    )
    db_arom = MoleculePharmacophore()
    db_arom.add_point(
        PharmacophorePoint(type=PointType.AROM, center=(0, 0, 0), normal=(0, 0, 1))
    )
    db_lipo = MoleculePharmacophore()
    db_lipo.add_point(PharmacophorePoint(type=PointType.LIPO, center=(0, 0, 0)))
    assert FunctionMapping(ref, db_arom, 0.5).get_next_map() == [(0, 0)]
    assert FunctionMapping(ref, db_lipo, 0.5).get_next_map() == [(0, 0)]


def test_function_mapping_hand_only_matches_compound():
    ref = QueryPharmacophore()
    ref.add_point(PharmacophorePoint(type=PointType.HACC_AND_HDON, center=(0, 0, 0)))
    db_compound = MoleculePharmacophore()
    db_compound.add_point(
        PharmacophorePoint(type=PointType.HACC_AND_HDON, center=(0, 0, 0))
    )
    db_hdon = MoleculePharmacophore()
    db_hdon.add_point(PharmacophorePoint(type=PointType.HDON, center=(0, 0, 0)))
    assert FunctionMapping(ref, db_compound, 0.5).get_next_map() == [(0, 0)]
    assert FunctionMapping(ref, db_hdon, 0.5).get_next_map() == []


def test_function_mapping_undef_matches_any():
    ref = QueryPharmacophore()
    ref.add_point(PharmacophorePoint(type=PointType.UNDEF, center=(0, 0, 0)))
    db = MoleculePharmacophore()
    db.add_point(PharmacophorePoint(type=PointType.HACC, center=(0, 0, 0)))
    assert FunctionMapping(ref, db, 0.5).get_next_map() == [(0, 0)]
