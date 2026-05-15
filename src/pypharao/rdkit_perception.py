"""Build pharmacophore from RDKit 3D molecule (Pharao `calcPharm` + FuncCalc)."""

from __future__ import annotations

import math

from rdkit import Chem

from .constants import (
    DENSITY,
    H_BOND_DIST,
    H_RADIUS,
    PI,
    PROBE_RADIUS,
    REF_LIPO,
    round_ob,
)
from .perception_options import PerceptionOptions
from .pharmacophore import (
    FUNC_SIGMA,
    FuncGroup,
    Pharmacophore,
    PharmacophorePoint,
)


def _pt(mol: Chem.Mol, conf_id: int, idx: int) -> tuple[float, float, float]:
    p = mol.GetConformer(conf_id).GetAtomPosition(idx)
    return (p.x, p.y, p.z)


def _arom_points(mol: Chem.Mol, conf_id: int) -> list[PharmacophorePoint]:
    out: list[PharmacophorePoint] = []
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if len(ring) < 3:
            continue
        if not all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring):
            continue
        coords = [_pt(mol, conf_id, a) for a in ring]
        cx = sum(c[0] for c in coords) / len(coords)
        cy = sum(c[1] for c in coords) / len(coords)
        cz = sum(c[2] for c in coords) / len(coords)
        # Plane normal via cross of two ring chords
        ax, ay, az = coords[0][0] - cx, coords[0][1] - cy, coords[0][2] - cz
        bx, by, bz = coords[1][0] - cx, coords[1][1] - cy, coords[1][2] - cz
        nxv = ay * bz - az * by
        nyv = az * bx - ax * bz
        nzv = ax * by - ay * bx
        ln = math.sqrt(nxv * nxv + nyv * nyv + nzv * nzv)
        if ln < 1e-6:
            continue
        nxv, nyv, nzv = nxv / ln, nyv / ln, nzv / ln
        out.append(
            PharmacophorePoint(
                cx,
                cy,
                cz,
                FuncGroup.AROM,
                FUNC_SIGMA[FuncGroup.AROM],
                True,
                cx + nxv,
                cy + nyv,
                cz + nzv,
            )
        )
    return out


def _h_donor_normal(mol: Chem.Mol, conf_id: int, idx: int) -> tuple[float, float, float]:
    ax, ay, az = _pt(mol, conf_id, idx)
    sx, sy, sz = 0.0, 0.0, 0.0
    nbr_bonds = 0
    for b in mol.GetAtomWithIdx(idx).GetBonds():
        oi = b.GetOtherAtomIdx(idx)
        if mol.GetAtomWithIdx(oi).GetAtomicNum() == 1:
            continue
        nbr_bonds += 1
        bx, by, bz = _pt(mol, conf_id, oi)
        sx += bx - ax
        sy += by - ay
        sz += bz - az
    ln = math.sqrt(sx * sx + sy * sy + sz * sz)
    if ln < 1e-12:
        return ax, ay, az
    sx, sy, sz = -sx / ln, -sy / ln, -sz / ln
    return ax + sx, ay + sy, az + sz


def _h_acc_normal(mol: Chem.Mol, conf_id: int, idx: int) -> tuple[float, float, float]:
    return _h_donor_normal(mol, conf_id, idx)


def _h_acc_delocalized(mol: Chem.Mol, idx: int) -> bool:
    a = mol.GetAtomWithIdx(idx)
    if a.GetAtomicNum() != 7:
        return False
    if a.GetIsAromatic() and a.GetTotalDegree() == 3:
        return True
    for b in a.GetBonds():
        oi = b.GetOtherAtomIdx(idx)
        aa = mol.GetAtomWithIdx(oi)
        if aa.GetIsAromatic() and a.GetTotalDegree() == 3:
            return True
        if aa.GetAtomicNum() == 6:
            for b2 in aa.GetBonds():
                o2 = b2.GetOtherAtomIdx(oi)
                if o2 == idx:
                    continue
                if b2.GetBondType() == Chem.BondType.DOUBLE:
                    aaa = mol.GetAtomWithIdx(o2)
                    if aaa.GetAtomicNum() in (8, 7, 16):
                        return True
        elif aa.GetAtomicNum() == 16:
            for b2 in aa.GetBonds():
                o2 = b2.GetOtherAtomIdx(oi)
                if o2 == idx:
                    continue
                if b2.GetBondType() == Chem.BondType.DOUBLE and mol.GetAtomWithIdx(o2).GetAtomicNum() == 8:
                    return True
    return False


def _vdw(z: int) -> float:
    return float(Chem.GetPeriodicTable().GetRvdw(z))


def _sphere_points(center: tuple[float, float, float], radius: float) -> list[tuple[float, float, float]]:
    sphere: list[tuple[float, float, float]] = []
    arclength = 1.0 / math.sqrt(math.sqrt(3.0) * DENSITY)
    dphi = arclength / radius
    nlayer = round_ob(PI / dphi) + 1
    phi = 0.0
    for _i in range(nlayer):
        rsinphi = radius * math.sin(phi)
        z = radius * math.cos(phi)
        dtheta = PI * 2 if rsinphi == 0 else arclength / rsinphi
        tmp_n = round_ob(PI * 2 / dtheta)
        if tmp_n <= 0:
            tmp_n = 1
        dtheta = PI * 2.0 / tmp_n
        theta = 0.0 if _i % 2 else PI
        for _j in range(tmp_n):
            sphere.append(
                (
                    rsinphi * math.cos(theta) + center[0],
                    rsinphi * math.sin(theta) + center[1],
                    z + center[2],
                )
            )
            theta += dtheta
            if theta > PI * 2:
                theta -= PI * 2
        phi += dphi
    return sphere


def _h_acc_neighbors(mol: Chem.Mol, conf_id: int, idx: int) -> list[int]:
    a = mol.GetAtomWithIdx(idx)
    ax, ay, az = _pt(mol, conf_id, idx)
    r_a = _vdw(a.GetAtomicNum())
    out: list[int] = []
    for j in range(mol.GetNumAtoms()):
        if j == idx:
            continue
        aa = mol.GetAtomWithIdx(j)
        if aa.GetAtomicNum() == 1:
            continue
        delta = H_BOND_DIST + H_RADIUS + _vdw(aa.GetAtomicNum())
        bx, by, bz = _pt(mol, conf_id, j)
        d2 = (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2
        if d2 <= delta * delta:
            out.append(j)
    return out


def _h_acc_surface_fraction(mol: Chem.Mol, conf_id: int, idx: int) -> float:
    atom = mol.GetAtomWithIdx(idx)
    ax, ay, az = _pt(mol, conf_id, idx)
    sphere = _sphere_points((ax, ay, az), H_BOND_DIST)
    nbrs = _h_acc_neighbors(mol, conf_id, idx)
    n_acc = 0
    for sx, sy, sz in sphere:
        ok = True
        for j in nbrs:
            n = mol.GetAtomWithIdx(j)
            bx, by, bz = _pt(mol, conf_id, j)
            dist_sq = (sx - bx) ** 2 + (sy - by) ** 2 + (sz - bz) ** 2
            r = _vdw(n.GetAtomicNum())
            sum_sq = (r + H_RADIUS) ** 2
            if dist_sq <= sum_sq:
                ok = False
                break
        if ok:
            n_acc += 1
    return n_acc / len(sphere) if sphere else 0.0


def _lipo_neighbors(mol: Chem.Mol, conf_id: int, idx: int) -> list[int]:
    a = mol.GetAtomWithIdx(idx)
    ax, ay, az = _pt(mol, conf_id, idx)
    r = _vdw(a.GetAtomicNum())
    out: list[int] = []
    for j in range(mol.GetNumAtoms()):
        if j == idx:
            continue
        aa = mol.GetAtomWithIdx(j)
        if aa.GetAtomicNum() == 1:
            continue
        delta = r + _vdw(aa.GetAtomicNum()) + 2 * PROBE_RADIUS
        bx, by, bz = _pt(mol, conf_id, j)
        d2 = (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2
        if d2 <= delta * delta:
            out.append(j)
    return out


def _lipo_acc_surf_atom(mol: Chem.Mol, conf_id: int, idx: int) -> float:
    a = mol.GetAtomWithIdx(idx)
    r = _vdw(a.GetAtomicNum())
    ax, ay, az = _pt(mol, conf_id, idx)
    sphere = _sphere_points((ax, ay, az), r)
    nbrs = _lipo_neighbors(mol, conf_id, idx)
    delta = PROBE_RADIUS / r
    n_ok = 0
    for sx, sy, sz in sphere:
        px = (sx - ax) * delta + sx
        py = (sy - ay) * delta + sy
        pz = (sz - az) * delta + sz
        ok = True
        for j in nbrs:
            n = mol.GetAtomWithIdx(j)
            bx, by, bz = _pt(mol, conf_id, j)
            dist_sq = (px - bx) ** 2 + (py - by) ** 2 + (pz - bz) ** 2
            sum_sq = (PROBE_RADIUS + _vdw(n.GetAtomicNum())) ** 2
            if dist_sq <= sum_sq:
                ok = False
                break
        if ok:
            n_ok += 1
    frac = n_ok / len(sphere) if sphere else 0.0
    return frac * 4 * PI * r * r


def _lipo_label_neighbors(scores: list[float], mol: Chem.Mol, idx: int, value: float) -> None:
    for b in mol.GetAtomWithIdx(idx).GetBonds():
        oi = b.GetOtherAtomIdx(idx)
        scores[oi] *= value


def _lipo_label_atoms(mol: Chem.Mol) -> list[float]:
    n = mol.GetNumAtoms()
    scores = [1.0] * n
    for idx in range(n):
        a = mol.GetAtomWithIdx(idx)
        z = a.GetAtomicNum()
        if z == 1:
            scores[idx] = 0.0
        elif z == 7:
            scores[idx] = 0.0
            if not a.GetIsAromatic():
                _lipo_label_neighbors(scores, mol, idx, 0.25)
                if a.GetTotalNumHs() > 0:
                    for b in a.GetBonds():
                        oi = b.GetOtherAtomIdx(idx)
                        scores[oi] = 0.0
                        _lipo_label_neighbors(scores, mol, oi, 0.0)
        elif z == 8:
            scores[idx] = 0.0
            if not a.GetIsAromatic():
                _lipo_label_neighbors(scores, mol, idx, 0.25)
                for b in a.GetBonds():
                    if mol.GetAtomWithIdx(b.GetOtherAtomIdx(idx)).GetAtomicNum() == 1:
                        for b2 in a.GetBonds():
                            oi = b2.GetOtherAtomIdx(idx)
                            scores[oi] = 0.0
                            _lipo_label_neighbors(scores, mol, oi, 0.0)
                    if b.GetBondType() == Chem.BondType.DOUBLE:
                        oi = b.GetOtherAtomIdx(idx)
                        scores[oi] = 0.0
                        for b2 in mol.GetAtomWithIdx(oi).GetBonds():
                            o2 = b2.GetOtherAtomIdx(oi)
                            if o2 == idx:
                                continue
                            scores[o2] = 0.0
                            _lipo_label_neighbors(scores, mol, o2, 0.6)
        elif z == 16:
            for b in a.GetBonds():
                if mol.GetAtomWithIdx(b.GetOtherAtomIdx(idx)).GetAtomicNum() == 1:
                    scores[idx] = 0.0
                    _lipo_label_neighbors(scores, mol, idx, 0.0)
                if b.GetBondType() == Chem.BondType.DOUBLE:
                    scores[idx] = 0.0
                    _lipo_label_neighbors(scores, mol, idx, 0.6)
            if a.GetTotalDegree() - a.GetNumHeavyNeighbors() > 2:
                scores[idx] = 0.0
                for b in a.GetBonds():
                    oi = b.GetOtherAtomIdx(idx)
                    scores[oi] = 0.0
                    _lipo_label_neighbors(scores, mol, oi, 0.6)
        if a.GetFormalCharge() != 0:
            for b in a.GetBonds():
                oi = b.GetOtherAtomIdx(idx)
                scores[oi] = 0.0
                _lipo_label_neighbors(scores, mol, oi, 0.0)
    for idx in range(n):
        v = scores[idx]
        if (v == 0.36 or v < 0.25) and v != 0.15:
            scores[idx] = 0.0
    return scores


def _lipo_points(mol: Chem.Mol, conf_id: int) -> list[PharmacophorePoint]:
    labels = _lipo_label_atoms(mol)
    n = mol.GetNumAtoms()
    weighted = [labels[i] * _lipo_acc_surf_atom(mol, conf_id, i) if labels[i] != 0.0 else 0.0 for i in range(n)]
    out: list[PharmacophorePoint] = []
    ri = mol.GetRingInfo()
    atom_set = set(range(n))
    for ring in ri.AtomRings():
        if len(ring) > 7:
            continue
        lipo_sum = 0.0
        cx = cy = cz = 0.0
        for aidx in ring:
            atom_set.discard(aidx)
            l = weighted[aidx]
            lipo_sum += l
            x, y, z = _pt(mol, conf_id, aidx)
            cx += l * x
            cy += l * y
            cz += l * z
        if lipo_sum > REF_LIPO:
            cx /= lipo_sum
            cy /= lipo_sum
            cz /= lipo_sum
            out.append(
                PharmacophorePoint(
                    cx,
                    cy,
                    cz,
                    FuncGroup.LIPO,
                    FUNC_SIGMA[FuncGroup.LIPO],
                    False,
                    0.0,
                    0.0,
                    0.0,
                )
            )
    for idx in list(atom_set):
        a = mol.GetAtomWithIdx(idx)
        if a.GetNumImplicitHs() > 2:
            x0, y0, z0 = _pt(mol, conf_id, idx)
            lipo_sum = weighted[idx]
            cx, cy, cz = lipo_sum * x0, lipo_sum * y0, lipo_sum * z0
            for b in a.GetBonds():
                oi = b.GetOtherAtomIdx(idx)
                aa = mol.GetAtomWithIdx(oi)
                if aa.GetAtomicNum() == 1:
                    continue
                if aa.GetTotalNumHs() == 1:
                    l = weighted[oi]
                    lipo_sum += l
                    x, y, z = _pt(mol, conf_id, oi)
                    cx += l * x
                    cy += l * y
                    cz += l * z
            if lipo_sum > REF_LIPO:
                cx /= lipo_sum
                cy /= lipo_sum
                cz /= lipo_sum
                out.append(
                    PharmacophorePoint(
                        cx,
                        cy,
                        cz,
                        FuncGroup.LIPO,
                        FUNC_SIGMA[FuncGroup.LIPO],
                        False,
                        0.0,
                        0.0,
                        0.0,
                    )
                )
    return out


def _hybrid_same_h(c1: PharmacophorePoint, c2: PharmacophorePoint) -> bool:
    d = (c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2 + (c1.z - c2.z) ** 2
    return d < 0.0001


def _hybrid_same_l(c1: PharmacophorePoint, c2: PharmacophorePoint) -> bool:
    d = (c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2 + (c1.z - c2.z) ** 2
    return d < 1.0


def _hybrid_calc(points: list[PharmacophorePoint], do_hybh: bool, do_hybl: bool) -> None:
    if do_hybl:
        i = 0
        while i < len(points):
            p = points[i]
            if p.func == FuncGroup.AROM:
                j = i + 1
                while j < len(points):
                    p2 = points[j]
                    if p2.func == FuncGroup.LIPO and _hybrid_same_l(p2, p):
                        cx = (p.x + p2.x) / 2.0
                        cy = (p.y + p2.y) / 2.0
                        cz = (p.z + p2.z) / 2.0
                        points[i] = PharmacophorePoint(
                            cx,
                            cy,
                            cz,
                            FuncGroup.HYBL,
                            FUNC_SIGMA[FuncGroup.HYBL],
                            False,
                            0.0,
                            0.0,
                            0.0,
                        )
                        del points[j]
                        continue
                    j += 1
            elif p.func == FuncGroup.LIPO:
                j = i + 1
                while j < len(points):
                    p2 = points[j]
                    if p2.func == FuncGroup.AROM and _hybrid_same_l(p, p2):
                        cx = (p.x + p2.x) / 2.0
                        cy = (p.y + p2.y) / 2.0
                        cz = (p.z + p2.z) / 2.0
                        points[i] = PharmacophorePoint(
                            cx,
                            cy,
                            cz,
                            FuncGroup.HYBL,
                            FUNC_SIGMA[FuncGroup.HYBL],
                            False,
                            0.0,
                            0.0,
                            0.0,
                        )
                        del points[j]
                        continue
                    j += 1
            i += 1
    if do_hybh:
        i = 0
        while i < len(points):
            p = points[i]
            if p.func == FuncGroup.HACC:
                j = i + 1
                while j < len(points):
                    p2 = points[j]
                    if p2.func == FuncGroup.HDON and _hybrid_same_h(p2, p):
                        v1 = (p.nx - p.x, p.ny - p.y, p.nz - p.z)
                        v2 = (p2.nx - p2.x, p2.ny - p2.y, p2.nz - p2.z)
                        mx = (v1[0] + v2[0]) / 2.0
                        my = (v1[1] + v2[1]) / 2.0
                        mz = (v1[2] + v2[2]) / 2.0
                        ln = math.sqrt(mx * mx + my * my + mz * mz)
                        if ln > 1e-12:
                            mx, my, mz = mx / ln, my / ln, mz / ln
                        points[i] = PharmacophorePoint(
                            p.x,
                            p.y,
                            p.z,
                            FuncGroup.HYBH,
                            FUNC_SIGMA[FuncGroup.HYBH],
                            True,
                            p.x + mx,
                            p.y + my,
                            p.z + mz,
                        )
                        del points[j]
                        continue
                    j += 1
            elif p.func == FuncGroup.HDON:
                j = i + 1
                while j < len(points):
                    p2 = points[j]
                    if p2.func == FuncGroup.HACC and _hybrid_same_h(p, p2):
                        v1 = (p.nx - p.x, p.ny - p.y, p.nz - p.z)
                        v2 = (p2.nx - p2.x, p2.ny - p2.y, p2.nz - p2.z)
                        mx = (v1[0] + v2[0]) / 2.0
                        my = (v1[1] + v2[1]) / 2.0
                        mz = (v1[2] + v2[2]) / 2.0
                        ln = math.sqrt(mx * mx + my * my + mz * mz)
                        if ln > 1e-12:
                            mx, my, mz = mx / ln, my / ln, mz / ln
                        points[i] = PharmacophorePoint(
                            p.x,
                            p.y,
                            p.z,
                            FuncGroup.HYBH,
                            FUNC_SIGMA[FuncGroup.HYBH],
                            True,
                            p.x + mx,
                            p.y + my,
                            p.z + mz,
                        )
                        del points[j]
                        continue
                    j += 1
            i += 1
    if do_hybl:
        for i in range(len(points)):
            p = points[i]
            if p.func in (FuncGroup.AROM, FuncGroup.LIPO):
                points[i] = PharmacophorePoint(
                    p.x,
                    p.y,
                    p.z,
                    FuncGroup.HYBL,
                    FUNC_SIGMA[FuncGroup.HYBL],
                    False,
                    0.0,
                    0.0,
                    0.0,
                )


def pharmacophore_from_molecule(
    mol: Chem.Mol,
    options: PerceptionOptions | None = None,
    conf_id: int = 0,
    name: str = "",
) -> Pharmacophore:
    """Build a pharmacophore from a 3D molecule (RDKit ``Chem.Mol`` with conformers)."""
    if mol.GetNumConformers() == 0:
        raise ValueError("RDKit molecule must have at least one conformer with 3D coordinates")
    opts = options or PerceptionOptions()
    if opts.hybl and opts.arom and opts.lipo:
        pass
    if opts.hybh and opts.hdon and opts.hacc:
        pass
    pts: list[PharmacophorePoint] = []
    if opts.arom:
        pts.extend(_arom_points(mol, conf_id))
    if opts.hdon:
        for a in mol.GetAtoms():
            z = a.GetAtomicNum()
            if z not in (7, 8):
                continue
            if a.GetFormalCharge() < 0:
                continue
            if a.GetTotalNumHs() == 0:
                continue
            idx = a.GetIdx()
            nx, ny, nz = _h_donor_normal(mol, conf_id, idx)
            x, y, z = _pt(mol, conf_id, idx)
            pts.append(
                PharmacophorePoint(
                    x,
                    y,
                    z,
                    FuncGroup.HDON,
                    FUNC_SIGMA[FuncGroup.HDON],
                    True,
                    nx,
                    ny,
                    nz,
                )
            )
    if opts.hacc:
        for a in mol.GetAtoms():
            z = a.GetAtomicNum()
            if z not in (7, 8):
                continue
            if a.GetFormalCharge() > 0:
                continue
            idx = a.GetIdx()
            if _h_acc_delocalized(mol, idx):
                continue
            if _h_acc_surface_fraction(mol, conf_id, idx) < 0.02:
                continue
            nx, ny, nz = _h_acc_normal(mol, conf_id, idx)
            x, y, z = _pt(mol, conf_id, idx)
            pts.append(
                PharmacophorePoint(
                    x,
                    y,
                    z,
                    FuncGroup.HACC,
                    FUNC_SIGMA[FuncGroup.HACC],
                    True,
                    nx,
                    ny,
                    nz,
                )
            )
    if opts.lipo:
        pts.extend(_lipo_points(mol, conf_id))
    if opts.posc or opts.negc:
        for a in mol.GetAtoms():
            ch = a.GetFormalCharge()
            if ch < 0 and opts.negc:
                x, y, z = _pt(mol, conf_id, a.GetIdx())
                pts.append(
                    PharmacophorePoint(
                        x,
                        y,
                        z,
                        FuncGroup.NEGC,
                        FUNC_SIGMA[FuncGroup.NEGC],
                        False,
                        0.0,
                        0.0,
                        0.0,
                    )
                )
            elif ch > 0 and opts.posc:
                x, y, z = _pt(mol, conf_id, a.GetIdx())
                pts.append(
                    PharmacophorePoint(
                        x,
                        y,
                        z,
                        FuncGroup.POSC,
                        FUNC_SIGMA[FuncGroup.POSC],
                        False,
                        0.0,
                        0.0,
                        0.0,
                    )
                )
    do_hybh = opts.hybh and opts.hdon and opts.hacc
    do_hybl = opts.hybl and opts.arom and opts.lipo
    if do_hybh or do_hybl:
        _hybrid_calc(pts, do_hybh, do_hybl)
    nm = name or (mol.GetProp("_Name") if mol.HasProp("_Name") else "")
    return Pharmacophore(name=nm, points=pts)


pharmacophore_from_rdkit = pharmacophore_from_molecule
