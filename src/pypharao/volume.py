"""Gaussian volume overlap between pharmacophore points."""

from __future__ import annotations

import math

from .constants import GCI2, PI
from .pharmacophore import PharmacophorePoint, PointType, cosine_normals


_AROMATIC_LIKE = {PointType.AROM, PointType.AROM_OR_LIPO}
_HBOND_LIKE = {
    PointType.HACC,
    PointType.HDON,
    PointType.HACC_AND_HDON,
    PointType.HACC_OR_HDON,
}


def volume_overlap(p1: PharmacophorePoint, p2: PharmacophorePoint, use_direction: bool) -> float:
    """Pharao ``VolumeOverlap`` (utilities.cpp).

    Adds a directional weight (``|cos|`` for aromatic-like pairs, ``cos`` for
    h-bond-like pairs) when both points carry a normal and ``use_direction``
    is set.
    """
    r2 = (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2
    vol = 1.0
    if use_direction:
        aromish = (p1.type in _AROMATIC_LIKE) and (p2.type in _AROMATIC_LIKE)
        if aromish and p1.has_normal and p2.has_normal:
            vol = abs(cosine_normals(p1, p2))
        elif (
            (p1.type in _HBOND_LIKE)
            and (p2.type in _HBOND_LIKE)
            and p1.has_normal
            and p2.has_normal
        ):
            vol = cosine_normals(p1, p2)

    vol *= GCI2 * (PI / (p1.sigma + p2.sigma)) ** 1.5
    vol *= math.exp(-(p1.sigma * p2.sigma) * r2 / (p1.sigma + p2.sigma))
    return vol
