"""Gaussian pharmacophore alignment (`alignment.cpp` / `alignment.h`)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .constants import GCI, GCI2, PI
from .pharmacophore import Pharmacophore, PointType
from .quaternion_math import (
    inverse_hessian,
    normalize_quaternion,
    row_product,
    rotate_coord_align,
    rotate_coord_util,
)


@dataclass
class SolutionInfo:
    rotor: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    volume: float = -1000.0
    iterations: int = 0
    center1: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    center2: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    rotation1: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=float))
    rotation2: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=float))


def _det3(R: np.ndarray) -> float:
    return float(
        R[0, 0] * R[1, 1] * R[2, 2]
        + R[2, 1] * R[1, 0] * R[0, 2]
        + R[0, 1] * R[1, 2] * R[2, 0]
        - R[0, 0] * R[1, 2] * R[2, 1]
        - R[1, 1] * R[2, 0] * R[0, 2]
        - R[2, 2] * R[0, 1] * R[1, 0]
    )


def _svd_principal_axes(mass: np.ndarray) -> np.ndarray:
    U, _, _ = np.linalg.svd(mass, full_matrices=True)
    if _det3(U) < 0:
        U = U.copy()
        U[:, 2] *= -1.0
    return U


def normal_contribution(
    n1: np.ndarray, n2: np.ndarray, q: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    x, y, z = float(n2[0]), float(n2[1]), float(n2[2])
    d1sq = q[1] * q[1]
    d2sq = q[2] * q[2]
    d3sq = q[3] * q[3]
    Ux = x * (1.0 - 2.0 * d2sq - 2.0 * d3sq) + y * (2.0 * (q[2] * q[1] - q[0] * q[3])) + z * (
        2.0 * (q[3] * q[1] + q[0] * q[2])
    )
    Uy = x * (2.0 * (q[1] * q[2] + q[0] * q[3])) + y * (1.0 - 2.0 * d1sq - 2.0 * d3sq) + z * (
        2.0 * (q[3] * q[2] - q[0] * q[1])
    )
    Uz = x * (2.0 * (q[1] * q[3] - q[0] * q[2])) + y * (2.0 * (q[2] * q[3] + q[0] * q[1])) + z * (
        1.0 - 2.0 * d1sq - 2.0 * d2sq
    )
    d2Cdq2 = np.zeros((4, 4), dtype=float)
    d2Cdq2[0, 0] = 0.0
    d2Cdq2[1, 1] = 2.0 * (n1[1] * Uy + n1[2] * Uz)
    d2Cdq2[2, 2] = 2.0 * (n1[0] * Ux + n1[2] * Uz)
    d2Cdq2[3, 3] = 2.0 * (n1[0] * Ux + n1[1] * Uy)
    d2Cdq2[0, 1] = d2Cdq2[1, 0] = -2.0 * (n1[1] * Uz - n1[2] * Uy)
    d2Cdq2[0, 2] = d2Cdq2[2, 0] = 2.0 * (n1[0] * Uz - n1[2] * Ux)
    d2Cdq2[0, 3] = d2Cdq2[3, 0] = -2.0 * (n1[0] * Uy - n1[1] * Ux)
    d2Cdq2[1, 2] = d2Cdq2[2, 1] = 2.0 * (n1[0] * Uy + n1[1] * Ux)
    d2Cdq2[1, 3] = d2Cdq2[3, 1] = 2.0 * (n1[0] * Uz + n1[2] * Ux)
    d2Cdq2[2, 3] = d2Cdq2[3, 2] = 2.0 * (n1[1] * Uz + n1[2] * Uy)
    dCdq = np.zeros(4, dtype=float)
    for hi in range(4):
        dCdq[hi] = sum(d2Cdq2[hi, hj] * q[hj] for hj in range(4))
    c = n1[0] * Ux + n1[1] * Uy + n1[2] * Uz
    return c, dCdq, d2Cdq2


@dataclass
class _AlignPoint:
    point: np.ndarray
    normal: np.ndarray
    type: PointType
    sigma: float
    has_normal: bool


_AROMATIC_LIKE = {PointType.AROM, PointType.AROM_OR_LIPO}
_HBOND_LIKE = {
    PointType.HACC,
    PointType.HDON,
    PointType.HACC_AND_HDON,
    PointType.HACC_OR_HDON,
}


class Alignment:
    def __init__(self, ref: Pharmacophore, db: Pharmacophore, pairs: list[tuple[int, int]]):
        self._ref_center = np.zeros(3, dtype=float)
        self._db_center = np.zeros(3, dtype=float)
        self._ref_map: list[_AlignPoint] = []
        self._db_map: list[_AlignPoint] = []
        self._aka: list[np.ndarray] = []
        self._nbr_points = 0
        self._nbr_excl = 0

        V1 = 0.0
        V2 = 0.0
        for ri, di in pairs:
            pr = ref[ri]
            pd = db[di]
            if pr.type == PointType.EXCL:
                self._nbr_excl += 1
                continue
            self._nbr_points += 1
            v1 = GCI * (PI / pr.sigma) ** 1.5
            v2 = GCI * (PI / pd.sigma) ** 1.5
            V1 += v1
            self._ref_center += v1 * np.array([pr.x, pr.y, pr.z], dtype=float)
            V2 += v2
            self._db_center += v2 * np.array([pd.x, pd.y, pd.z], dtype=float)

        if V1 > 0:
            self._ref_center /= V1
        if V2 > 0:
            self._db_center /= V2

        mass1 = np.zeros((3, 3), dtype=float)
        mass2 = np.zeros((3, 3), dtype=float)

        for ri, di in pairs:
            pr = ref[ri]
            pd = db[di]
            p1p = np.array([pr.x - self._ref_center[0], pr.y - self._ref_center[1], pr.z - self._ref_center[2]])
            p1n = np.array([pr.nx - pr.x, pr.ny - pr.y, pr.nz - pr.z], dtype=float)
            p2p = np.array([pd.x - self._db_center[0], pd.y - self._db_center[1], pd.z - self._db_center[2]])
            p2n = np.array([pd.nx - pd.x, pd.ny - pd.y, pd.nz - pd.z], dtype=float)

            if pr.type != PointType.EXCL:
                v1 = GCI * (PI / pr.sigma) ** 1.5
                v2 = GCI * (PI / pd.sigma) ** 1.5
                x, y, z = p1p[0], p1p[1], p1p[2]
                mass1[0, 0] += v1 * x * x
                mass1[0, 1] += v1 * x * y
                mass1[0, 2] += v1 * x * z
                mass1[1, 1] += v1 * y * y
                mass1[1, 2] += v1 * y * z
                mass1[2, 2] += v1 * z * z
                x, y, z = p2p[0], p2p[1], p2p[2]
                mass2[0, 0] += v2 * x * x
                mass2[0, 1] += v2 * x * y
                mass2[0, 2] += v2 * x * z
                mass2[1, 1] += v2 * y * y
                mass2[1, 2] += v2 * y * z
                mass2[2, 2] += v2 * z * z

            self._ref_map.append(
                _AlignPoint(p1p.copy(), p1n.copy(), pr.type, pr.sigma, pr.has_normal)
            )
            self._db_map.append(
                _AlignPoint(p2p.copy(), p2n.copy(), pd.type, pd.sigma, pd.has_normal)
            )

        if self._nbr_points > 0 and V1 > 1e-15:
            mass1[1, 0] = mass1[0, 1]
            mass1[2, 0] = mass1[0, 2]
            mass1[2, 1] = mass1[1, 2]
            mass1 /= V1
            ref_rot = _svd_principal_axes(mass1)
        else:
            ref_rot = np.eye(3, dtype=float)

        if self._nbr_points > 0 and V2 > 1e-15:
            mass2[1, 0] = mass2[0, 1]
            mass2[2, 0] = mass2[0, 2]
            mass2[2, 1] = mass2[1, 2]
            mass2 /= V2
            db_rot = _svd_principal_axes(mass2)
        else:
            db_rot = np.eye(3, dtype=float)

        self._ref_rot = ref_rot
        self._db_rot = db_rot

        for i in range(len(self._ref_map)):
            rp = self._ref_map[i].point
            x, y, z = rp[0], rp[1], rp[2]
            rp[0] = ref_rot[0, 0] * x + ref_rot[1, 0] * y + ref_rot[2, 0] * z
            rp[1] = ref_rot[0, 1] * x + ref_rot[1, 1] * y + ref_rot[2, 1] * z
            rp[2] = ref_rot[0, 2] * x + ref_rot[1, 2] * y + ref_rot[2, 2] * z

            dp = self._db_map[i].point
            x, y, z = dp[0], dp[1], dp[2]
            dp[0] = db_rot[0, 0] * x + db_rot[1, 0] * y + db_rot[2, 0] * z
            dp[1] = db_rot[0, 1] * x + db_rot[1, 1] * y + db_rot[2, 1] * z
            dp[2] = db_rot[0, 2] * x + db_rot[1, 2] * y + db_rot[2, 2] * z

            dx = rp[0] - dp[0]
            dy = rp[1] - dp[1]
            dz = rp[2] - dp[2]
            sx = rp[0] + dp[0]
            sy = rp[1] + dp[1]
            sz = rp[2] + dp[2]
            dx2, dy2, dz2 = dx * dx, dy * dy, dz * dz
            sy2, sx2, sz2 = sy * sy, sx * sx, sz * sz

            Ak = np.zeros((4, 4), dtype=float)
            Ak[0, 0] = dx2 + dy2 + dz2
            Ak[0, 1] = dy * sz - sy * dz
            Ak[0, 2] = sx * dz - dx * sz
            Ak[0, 3] = dx * sy - sx * dy
            Ak[1, 0] = Ak[0, 1]
            Ak[1, 1] = dx2 + sy2 + sz2
            Ak[1, 2] = dx * dy - sx * sy
            Ak[1, 3] = dx * dz - sx * sz
            Ak[2, 0] = Ak[0, 2]
            Ak[2, 1] = Ak[1, 2]
            Ak[2, 2] = sx2 + dy2 + sz2
            Ak[2, 3] = dy * dz - sy * sz
            Ak[3, 0] = Ak[0, 3]
            Ak[3, 1] = Ak[1, 3]
            Ak[3, 2] = Ak[2, 3]
            Ak[3, 3] = sx2 + sy2 + dz2
            self._aka.append(Ak)

            rn = self._ref_map[i].normal
            x, y, z = rn[0], rn[1], rn[2]
            rn[0] = ref_rot[0, 0] * x + ref_rot[1, 0] * y + ref_rot[2, 0] * z
            rn[1] = ref_rot[0, 1] * x + ref_rot[1, 1] * y + ref_rot[2, 1] * z
            rn[2] = ref_rot[0, 2] * x + ref_rot[1, 2] * y + ref_rot[2, 2] * z

            dn = self._db_map[i].normal
            x, y, z = dn[0], dn[1], dn[2]
            dn[0] = db_rot[0, 0] * x + db_rot[1, 0] * y + db_rot[2, 0] * z
            dn[1] = db_rot[0, 1] * x + db_rot[1, 1] * y + db_rot[2, 1] * z
            dn[2] = db_rot[0, 2] * x + db_rot[1, 2] * y + db_rot[2, 2] * z

    def align(self, use_direction: bool) -> SolutionInfo:
        si = SolutionInfo()
        si.volume = -1000.0
        si.iterations = 0
        si.center1 = self._ref_center.copy()
        si.center2 = self._db_center.copy()
        si.rotation1 = self._ref_rot.copy()
        si.rotation2 = self._db_rot.copy()

        scale = 1.0 / self._nbr_excl if self._nbr_excl != 0 else 1.0

        for _call in range(4):
            rotor = np.zeros(4, dtype=float)
            rotor[_call] = 1.0
            old_volume = -999.99
            ii = 0
            for ii in range(100):
                grad = np.zeros(4, dtype=float)
                volume = 0.0
                hessian = np.zeros((4, 4), dtype=float)
                for i in range(len(self._ref_map)):
                    Ak = self._aka[i]
                    Aq = Ak @ rotor
                    qAq = float(np.dot(Aq, rotor))
                    v = GCI2 * (PI / (self._ref_map[i].sigma + self._db_map[i].sigma)) ** 1.5 * math.exp(
                        -qAq
                    )
                    c = 1.0
                    rf = self._ref_map[i].type
                    df = self._db_map[i].type
                    hn1 = self._ref_map[i].has_normal
                    hn2 = self._db_map[i].has_normal
                    n1 = self._ref_map[i].normal
                    n2 = self._db_map[i].normal

                    if (
                        use_direction
                        and rf in _AROMATIC_LIKE
                        and df in _AROMATIC_LIKE
                        and hn1
                        and hn2
                    ):
                        c, dCdq, d2Cdq2 = normal_contribution(n1, n2, rotor)
                        if c < 0:
                            c *= -1.0
                            dCdq = dCdq.copy() * -1.0
                            d2Cdq2 = d2Cdq2.copy() * -1.0
                        for hi in range(4):
                            grad[hi] += v * (dCdq[hi] - 2.0 * c * Aq[hi])
                            for hj in range(4):
                                hessian[hi, hj] += v * (
                                    d2Cdq2[hi, hj]
                                    - 2.0 * dCdq[hi] * Aq[hj]
                                    + 2.0 * c * (2.0 * Aq[hi] * Aq[hj] - Ak[hi, hj])
                                )
                        v *= c
                    elif (
                        use_direction
                        and rf in _HBOND_LIKE
                        and df in _HBOND_LIKE
                        and hn1
                        and hn2
                    ):
                        c, dCdq, d2Cdq2 = normal_contribution(n1, n2, rotor)
                        for hi in range(4):
                            grad[hi] += v * (dCdq[hi] - 2.0 * c * Aq[hi])
                            for hj in range(4):
                                hessian[hi, hj] += v * (
                                    d2Cdq2[hi, hj]
                                    - 2.0 * dCdq[hi] * Aq[hj]
                                    + 2.0 * c * (2.0 * Aq[hi] * Aq[hj] - Ak[hi, hj])
                                )
                        v *= c
                    elif rf == PointType.EXCL:
                        v *= -scale
                        for hi in range(4):
                            grad[hi] -= 2.0 * v * Aq[hi]
                            for hj in range(4):
                                hessian[hi, hj] += 2.0 * v * (2.0 * Aq[hi] * Aq[hj] - Ak[hi, hj])
                    else:
                        for hi in range(4):
                            grad[hi] -= 2.0 * v * Aq[hi]
                            for hj in range(4):
                                hessian[hi, hj] += 2.0 * v * (2.0 * Aq[hi] * Aq[hj] - Ak[hi, hj])
                    volume += v

                if math.isnan(volume) or (volume - old_volume < 1e-5):
                    break
                old_volume = volume
                H = hessian.copy()
                inverse_hessian(H)
                grad = row_product(H, grad)
                grad *= 0.9
                rotor = rotor + grad
                normalize_quaternion(rotor)

            if old_volume > si.volume:
                si.rotor = rotor.copy()
                si.volume = old_volume
                si.iterations = ii

        return si


def position_pharmacophore(pharm: Pharmacophore, U: np.ndarray, sol: SolutionInfo) -> None:
    """Transform pharmacophore points in-place (utilities `positionPharmacophore`)."""
    from .pharmacophore import PharmacophorePoint, TYPE_HAS_NORMAL

    rt = sol.rotation2.T
    for i in range(len(pharm)):
        p = pharm[i]
        nrel = np.array([p.nx - p.x, p.ny - p.y, p.nz - p.z], dtype=float)
        pt = np.array([p.x - sol.center2[0], p.y - sol.center2[1], p.z - sol.center2[2]])
        pt = rotate_coord_align(pt, rt)
        pt = rotate_coord_util(pt, U)
        pt = rotate_coord_align(pt, sol.rotation1)
        pt += sol.center1
        nrel = rotate_coord_align(nrel, rt)
        nrel = rotate_coord_util(nrel, U)
        nrel = rotate_coord_align(nrel, sol.rotation1)
        new_center = (float(pt[0]), float(pt[1]), float(pt[2]))
        if TYPE_HAS_NORMAL[p.type]:
            new_normal = (
                float(nrel[0] + pt[0]),
                float(nrel[1] + pt[1]),
                float(nrel[2] + pt[2]),
            )
            new_point = PharmacophorePoint(
                type=p.type, center=new_center, sigma=p.sigma, normal=new_normal
            )
        else:
            new_point = PharmacophorePoint(
                type=p.type, center=new_center, sigma=p.sigma
            )
        pharm.set_point(i, new_point)


def position_molecule_coords(
    coords: np.ndarray, U: np.ndarray, sol: SolutionInfo
) -> np.ndarray:
    """Apply alignment to Nx3 coordinates."""
    rt = sol.rotation2.T
    out = np.zeros_like(coords)
    for i in range(coords.shape[0]):
        pt = coords[i] - sol.center2
        pt = rotate_coord_align(pt, rt)
        pt = rotate_coord_util(pt, U)
        pt = rotate_coord_align(pt, sol.rotation1)
        pt += sol.center1
        out[i] = pt
    return out
