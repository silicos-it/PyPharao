"""Build a pharmacophore from an RDKit 3D molecule (Pharao `calcPharm` + FuncCalc)."""

from __future__ import annotations

import math
from typing import Any

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
from .perception import (
    MoleculePharmacophorePerception,
    PharmacophorePerception,
    QueryPharmacophorePerception,
)
from .pharmacophore import (
    DEFAULT_SIGMA,
    MoleculePharmacophore,
    Pharmacophore,
    PharmacophorePoint,
    PointType,
    QueryPharmacophore,
)


# Two HACC and HDON sites are considered co-located and merged into a single
# HACC&HDON point when their squared distance is below this threshold.
_HACC_HDON_MERGE_DIST_SQ = 0.0001


def _pt(mol: Chem.Mol, conf_id: int, idx: int) -> tuple[float, float, float]:
    p = mol.GetConformer(conf_id).GetAtomPosition(idx)
    return (p.x, p.y, p.z)


def _num_heavy_neighbors(atom: Chem.Atom) -> int:
    """Count non-hydrogen neighbors (``GetNumHeavyNeighbors`` removed in RDKit 2026)."""
    getter = getattr(atom, "GetNumHeavyNeighbors", None)
    if getter is not None:
        return getter()
    return sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() != 1)


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
                type=PointType.AROM,
                center=(cx, cy, cz),
                sigma=DEFAULT_SIGMA[PointType.AROM],
                normal=(cx + nxv, cy + nyv, cz + nzv),
            )
        )
    return out


def _has_donor_hydrogen(atom: Chem.Atom) -> bool:
    if atom.GetTotalNumHs() > 0:
        return True
    return any(n.GetAtomicNum() == 1 for n in atom.GetNeighbors())


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
            if a.GetTotalDegree() - _num_heavy_neighbors(a) > 2:
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
                    type=PointType.LIPO,
                    center=(cx, cy, cz),
                    sigma=DEFAULT_SIGMA[PointType.LIPO],
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
                        type=PointType.LIPO,
                        center=(cx, cy, cz),
                        sigma=DEFAULT_SIGMA[PointType.LIPO],
                    )
                )
    return out


def _merge_hacc_hdon(points: list[PharmacophorePoint]) -> list[PharmacophorePoint]:
    """Collapse co-located HACC + HDON sites into a single HACC&HDON point.

    Iterates over pairs; whenever a donor and acceptor sit at the same atom
    (squared distance below :data:`_HACC_HDON_MERGE_DIST_SQ`) both are removed
    and replaced with one ``HACC&HDON`` point at that location.
    """
    used: set[int] = set()
    out: list[PharmacophorePoint] = []
    for i, p in enumerate(points):
        if i in used:
            continue
        if p.type not in (PointType.HACC, PointType.HDON):
            continue
        partner_type = PointType.HDON if p.type == PointType.HACC else PointType.HACC
        for j in range(i + 1, len(points)):
            if j in used:
                continue
            q = points[j]
            if q.type != partner_type:
                continue
            d2 = (p.x - q.x) ** 2 + (p.y - q.y) ** 2 + (p.z - q.z) ** 2
            if d2 > _HACC_HDON_MERGE_DIST_SQ:
                continue
            out.append(
                PharmacophorePoint(
                    type=PointType.HACC_AND_HDON,
                    center=(p.x, p.y, p.z),
                    sigma=DEFAULT_SIGMA[PointType.HACC_AND_HDON],
                )
            )
            used.add(i)
            used.add(j)
            break
    kept = [p for k, p in enumerate(points) if k not in used]
    return kept + out


def _perceive_points(
    mol: Chem.Mol,
    perception: PharmacophorePerception,
    conf_id: int,
) -> list[PharmacophorePoint]:
    pts: list[PharmacophorePoint] = []
    if perception.is_enabled(PointType.AROM):
        pts.extend(_arom_points(mol, conf_id))
    if perception.is_enabled(PointType.HDON):
        for a in mol.GetAtoms():
            z = a.GetAtomicNum()
            if z not in (7, 8):
                continue
            if a.GetFormalCharge() < 0:
                continue
            if not _has_donor_hydrogen(a):
                continue
            idx = a.GetIdx()
            x, y, z = _pt(mol, conf_id, idx)
            pts.append(
                PharmacophorePoint(
                    type=PointType.HDON,
                    center=(x, y, z),
                    sigma=DEFAULT_SIGMA[PointType.HDON],
                )
            )
    if perception.is_enabled(PointType.HACC):
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
            x, y, z = _pt(mol, conf_id, idx)
            pts.append(
                PharmacophorePoint(
                    type=PointType.HACC,
                    center=(x, y, z),
                    sigma=DEFAULT_SIGMA[PointType.HACC],
                )
            )
    if perception.is_enabled(PointType.LIPO):
        pts.extend(_lipo_points(mol, conf_id))
    posc = perception.is_enabled(PointType.POSC)
    negc = perception.is_enabled(PointType.NEGC)
    if posc or negc:
        for a in mol.GetAtoms():
            ch = a.GetFormalCharge()
            if ch < 0 and negc:
                x, y, z = _pt(mol, conf_id, a.GetIdx())
                pts.append(
                    PharmacophorePoint(
                        type=PointType.NEGC,
                        center=(x, y, z),
                        sigma=DEFAULT_SIGMA[PointType.NEGC],
                    )
                )
            elif ch > 0 and posc:
                x, y, z = _pt(mol, conf_id, a.GetIdx())
                pts.append(
                    PharmacophorePoint(
                        type=PointType.POSC,
                        center=(x, y, z),
                        sigma=DEFAULT_SIGMA[PointType.POSC],
                    )
                )
    if perception.is_enabled(PointType.HACC_AND_HDON):
        pts = _merge_hacc_hdon(pts)
    return pts


def _require_conformer(mol: Any) -> None:
    if mol.GetNumConformers() == 0:
        raise ValueError(
            "RDKit molecule must have at least one conformer with 3D coordinates"
        )


def query_pharmacophore_from_molecule(
    mol: Chem.Mol,
    perception: QueryPharmacophorePerception | None = None,
    *,
    conf_id: int = 0,
    name: str = "",
) -> QueryPharmacophore:
    """Build a :class:`QueryPharmacophore` from a 3D RDKit molecule.

    The user typically refines the result manually (e.g. converting an ``AROM``
    point to ``AROM|LIPO`` or an ``HDON``/``HACC`` pair to ``HACC|HDON``); the
    auto-perception emits only the seven elementary plus ``HACC&HDON`` types.
    """
    _require_conformer(mol)
    perception = perception or QueryPharmacophorePerception()
    pts = _perceive_points(mol, perception, conf_id)
    nm = name or (mol.GetProp("_Name") if mol.HasProp("_Name") else "")
    ph = QueryPharmacophore(name=nm)
    for p in pts:
        ph.add_point(p)
    return ph


def query_pharmacophore_from_protein(*args: Any, **kwargs: Any) -> QueryPharmacophore:
    """Derive a :class:`QueryPharmacophore` from a protein structure.

    Not implemented yet.
    """
    raise NotImplementedError(
        "query_pharmacophore_from_protein is not implemented yet."
    )


def molecule_pharmacophore_from_molecule(
    mol: Chem.Mol,
    perception: MoleculePharmacophorePerception | None = None,
    *,
    conf_id: int = 0,
) -> MoleculePharmacophore:
    """Build a :class:`MoleculePharmacophore` from a 3D RDKit molecule.

    Used internally by :class:`pypharao.PharmacophoreSearch` once per database
    molecule (and once per conformer).
    """
    _require_conformer(mol)
    perception = perception or MoleculePharmacophorePerception()
    pts = _perceive_points(mol, perception, conf_id)
    ph = MoleculePharmacophore()
    for p in pts:
        ph.add_point(p)
    return ph
