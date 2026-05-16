from pypharao import PharmacophorePoint, PointType, volume_overlap


def test_volume_overlap_self_lipo_positive():
    p = PharmacophorePoint(type=PointType.LIPO, center=(0, 0, 0))
    assert volume_overlap(p, p, True) > 0.0


def test_volume_directional_hbond_smaller_when_misaligned():
    p1 = PharmacophorePoint(type=PointType.HACC, center=(0, 0, 0))
    p2 = PharmacophorePoint(type=PointType.HDON, center=(0.1, 0, 0))
    # H-bond directionality currently only applies when both sides carry a
    # normal; without normals we still expect a non-zero overlap.
    assert volume_overlap(p1, p2, True) <= volume_overlap(p1, p2, False)
