import pytest

from pypharao import FuncGroup, Pharmacophore, PharmacophorePoint
from pypharao.function_mapping import FunctionMapping, functions_compatible


@pytest.mark.parametrize(
    "ref,db,expected",
    [
        (FuncGroup.AROM, FuncGroup.AROM, True),
        (FuncGroup.LIPO, FuncGroup.LIPO, True),
        (FuncGroup.HDON, FuncGroup.HDON, True),
        (FuncGroup.HACC, FuncGroup.HACC, True),
        (FuncGroup.POSC, FuncGroup.POSC, True),
        (FuncGroup.NEGC, FuncGroup.NEGC, True),
        (FuncGroup.HYBL, FuncGroup.AROM, True),
        (FuncGroup.HYBL, FuncGroup.LIPO, True),
        (FuncGroup.HYBL, FuncGroup.HYBL, True),
        (FuncGroup.HYBH, FuncGroup.HDON, True),
        (FuncGroup.HYBH, FuncGroup.HACC, True),
        (FuncGroup.HYBH, FuncGroup.HYBH, True),
        (FuncGroup.AROM, FuncGroup.HYBL, False),
        (FuncGroup.AROM, FuncGroup.LIPO, False),
        (FuncGroup.LIPO, FuncGroup.HYBL, False),
        (FuncGroup.LIPO, FuncGroup.AROM, False),
        (FuncGroup.HDON, FuncGroup.HYBH, False),
        (FuncGroup.HDON, FuncGroup.HACC, False),
        (FuncGroup.HACC, FuncGroup.HYBH, False),
        (FuncGroup.HACC, FuncGroup.HDON, False),
        (FuncGroup.HYBL, FuncGroup.HDON, False),
        (FuncGroup.HYBH, FuncGroup.AROM, False),
        (FuncGroup.POSC, FuncGroup.NEGC, False),
    ],
)
def test_functions_compatible(ref, db, expected):
    assert functions_compatible(ref, db) is expected


def test_function_mapping_single_pair():
    ref = Pharmacophore(
        points=[
            PharmacophorePoint(0, 0, 0, FuncGroup.LIPO, 0.7, False, 0, 0, 0),
        ]
    )
    db = ref.copy()
    fm = FunctionMapping(ref, db, 0.5)
    m = fm.get_next_map()
    assert m == [(0, 0)]


def test_function_mapping_hybl_matches_arom_not_reverse():
    ref = Pharmacophore(
        points=[PharmacophorePoint(0, 0, 0, FuncGroup.HYBL, 0.7, True, 0, 0, 1)]
    )
    db_arom = Pharmacophore(
        points=[PharmacophorePoint(0, 0, 0, FuncGroup.AROM, 0.7, True, 0, 0, 1)]
    )
    db_hybl = Pharmacophore(
        points=[PharmacophorePoint(0, 0, 0, FuncGroup.HYBL, 0.7, True, 0, 0, 1)]
    )
    ref_arom = Pharmacophore(
        points=[PharmacophorePoint(0, 0, 0, FuncGroup.AROM, 0.7, True, 0, 0, 1)]
    )

    assert FunctionMapping(ref, db_arom, 0.5).get_next_map() == [(0, 0)]
    assert FunctionMapping(ref, db_hybl, 0.5).get_next_map() == [(0, 0)]
    assert FunctionMapping(ref_arom, db_hybl, 0.5).get_next_map() == []


def test_function_mapping_hybh_matches_components_not_reverse():
    ref = Pharmacophore(
        points=[PharmacophorePoint(0, 0, 0, FuncGroup.HYBH, 1.0, True, 0, 0, 1)]
    )
    db_hdon = Pharmacophore(
        points=[PharmacophorePoint(0, 0, 0, FuncGroup.HDON, 1.0, True, 0, 0, 1)]
    )
    db_hybh = Pharmacophore(
        points=[PharmacophorePoint(0, 0, 0, FuncGroup.HYBH, 1.0, True, 0, 0, 1)]
    )
    ref_hdon = Pharmacophore(
        points=[PharmacophorePoint(0, 0, 0, FuncGroup.HDON, 1.0, True, 0, 0, 1)]
    )

    assert FunctionMapping(ref, db_hdon, 0.5).get_next_map() == [(0, 0)]
    assert FunctionMapping(ref, db_hybh, 0.5).get_next_map() == [(0, 0)]
    assert FunctionMapping(ref_hdon, db_hybh, 0.5).get_next_map() == []
