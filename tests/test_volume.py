from pypharao import FuncGroup, Pharmacophore, PharmacophorePoint, volume_overlap


def test_volume_overlap_self_lipo():
    p = PharmacophorePoint(0, 0, 0, FuncGroup.LIPO, 0.7, False, 0, 0, 0)
    v = volume_overlap(p, p, True)
    assert v > 0.0


def test_volume_directional_hbond():
    p1 = PharmacophorePoint(0, 0, 0, FuncGroup.HACC, 1.0, True, 0, 0, 1.0)
    p2 = PharmacophorePoint(0.1, 0, 0, FuncGroup.HDON, 1.0, True, 0.1, 0, 0.0)
    v = volume_overlap(p1, p2, True)
    assert v <= volume_overlap(p1, p2, False)
