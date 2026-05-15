from pypharao import FuncGroup, Pharmacophore, PharmacophorePoint, PharmacophoreSearch


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
