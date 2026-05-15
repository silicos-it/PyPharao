"""Gaussian volume overlap between pharmacophore points."""

from __future__ import annotations

import math

from .constants import GCI2, PI
from .pharmacophore import FuncGroup, PharmacophorePoint, cosine_normals


def volume_overlap(p1: PharmacophorePoint, p2: PharmacophorePoint, use_direction: bool) -> float:
    """Pharao `VolumeOverlap` (utilities.cpp)."""
    r2 = (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2
    vol = 1.0
    if use_direction:
        aromish = (p1.func in (FuncGroup.AROM, FuncGroup.HYBL)) and (
            p2.func in (FuncGroup.AROM, FuncGroup.HYBL)
        )
        if aromish and p1.has_normal and p2.has_normal:
            vol = abs(cosine_normals(p1, p2))
        elif (
            (p1.func in (FuncGroup.HACC, FuncGroup.HDON, FuncGroup.HYBH))
            and (p2.func in (FuncGroup.HACC, FuncGroup.HDON, FuncGroup.HYBH))
            and p1.has_normal
            and p2.has_normal
        ):
            vol = cosine_normals(p1, p2)

    vol *= GCI2 * (PI / (p1.alpha + p2.alpha)) ** 1.5
    vol *= math.exp(-(p1.alpha * p2.alpha) * r2 / (p1.alpha + p2.alpha))
    return vol
