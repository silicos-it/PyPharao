from pypharao import (
    FuncGroup,
    Pharmacophore,
    PharmacophorePoint,
    PharmacophoreSearch,
    count_query_features,
    default_alpha,
)


def test_search_identical_pharmacophores():
    pts = [
        PharmacophorePoint(0, 0, 0, FuncGroup.LIPO, 0.7, False, 0, 0, 0),
        PharmacophorePoint(1.5, 0, 0, FuncGroup.LIPO, 0.7, False, 0, 0, 0),
    ]
    ref = Pharmacophore(points=pts)
    db = ref.copy()
    res = PharmacophoreSearch(epsilon=0.5, use_direction=False).search(ref, db)
    assert res.tanimoto > 0.0
    assert res.overlap_volume > 0.0


def test_search_with_bound_reference():
    pts = [
        PharmacophorePoint(0, 0, 0, FuncGroup.LIPO, 0.7, False, 0, 0, 0),
        PharmacophorePoint(1.5, 0, 0, FuncGroup.LIPO, 0.7, False, 0, 0, 0),
    ]
    ref = Pharmacophore(points=pts)
    db = ref.copy()
    res = PharmacophoreSearch(ref, use_direction=False).search(db)
    assert res.tanimoto > 0.0


def test_min_matched_query_features_rejects_partial_mapping():
    ref = Pharmacophore(
        points=[
            PharmacophorePoint(0, 0, 0, FuncGroup.AROM, default_alpha(FuncGroup.AROM), True, 0, 0, 1),
            PharmacophorePoint(1, 0, 0, FuncGroup.HACC, default_alpha(FuncGroup.HACC), True, 1, 0, 1),
            PharmacophorePoint(2, 0, 0, FuncGroup.HDON, default_alpha(FuncGroup.HDON), True, 2, 0, 1),
        ]
    )
    db = Pharmacophore(
        points=[
            PharmacophorePoint(0, 0, 0, FuncGroup.HACC, default_alpha(FuncGroup.HACC), True, 0, 0, 1),
            PharmacophorePoint(1, 0, 0, FuncGroup.HDON, default_alpha(FuncGroup.HDON), True, 1, 0, 1),
        ]
    )
    partial_ok = PharmacophoreSearch(
        ref, epsilon=0.5, use_direction=False, min_matched_query_features=2
    ).search(db)
    assert partial_ok.tanimoto > 0.0
    assert len(partial_ok.mapping) == 2

    strict = PharmacophoreSearch(
        ref, epsilon=0.5, use_direction=False, min_matched_query_features=3
    ).search(db)
    assert strict.tanimoto == 0.0
    assert strict.mapping == []

    default_all = PharmacophoreSearch(ref, epsilon=0.5, use_direction=False).search(db)
    assert default_all.tanimoto == 0.0
    assert default_all.mapping == []


def test_mapping_indices_reference_database_pharmacophore():
    pts = [
        PharmacophorePoint(0, 0, 0, FuncGroup.LIPO, 0.7, False, 0, 0, 0),
        PharmacophorePoint(1.5, 0, 0, FuncGroup.LIPO, 0.7, False, 0, 0, 0),
    ]
    ref = Pharmacophore(points=pts)
    db = ref.copy()
    res = PharmacophoreSearch(epsilon=0.5, use_direction=False).search(ref, db)
    for r, d in res.mapping:
        assert r < len(ref)
        assert d < len(res.database_pharmacophore)
