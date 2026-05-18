"""Build a pharmacophore from an RDKit 3D molecule (Pharao `calcPharm` + FuncCalc)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
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
# HACC_AND_HDON point when their squared distance is below this threshold
# (≈ 0.01 Å, i.e. effectively the same atom).
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
        # Aromatic rings are already reported as AROM by `_arom_points`; skip
        # them here so the same ring is never emitted as both AROM and LIPO.
        # Still drop the ring atoms from `atom_set` so the single-atom LIPO
        # loop below does not reconsider them.
        if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring):
            for aidx in ring:
                atom_set.discard(aidx)
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
    """Collapse co-located HACC + HDON sites into a single HACC_AND_HDON point.

    Iterates over pairs; whenever a donor and acceptor sit at the same atom
    (squared distance below :data:`_HACC_HDON_MERGE_DIST_SQ`) both are removed
    and replaced with one ``HACC_AND_HDON`` point at that location.
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


def _mol_from_pdb_file(path: str | Path) -> Chem.Mol:
    """Load a single-structure PDB as an :class:`rdkit.Chem.Mol` with 3D coordinates."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"PDB file not found: {p}")
    mol = Chem.MolFromPDBFile(str(p), sanitize=False, removeHs=False)
    if mol is None:
        raise ValueError(f"RDKit could not parse PDB file: {p}")
    _require_conformer(mol)
    try:
        Chem.SanitizeMol(mol)
    except Chem.MolSanitizeException as e:
        raise ValueError(
            f"RDKit could not sanitize structure read from PDB: {p}"
        ) from e
    return mol


def _all_atom_centers_array(mol: Chem.Mol, conf_id: int) -> np.ndarray:
    """Coordinates of every atom (including hydrogen) in one conformer."""
    if conf_id < 0 or conf_id >= mol.GetNumConformers():
        raise ValueError(
            f"conf_id {conf_id} is out of range for a molecule with "
            f"{mol.GetNumConformers()} conformer(s)"
        )
    conf = mol.GetConformer(conf_id)
    return np.asarray(
        [
            (
                conf.GetAtomPosition(a.GetIdx()).x,
                conf.GetAtomPosition(a.GetIdx()).y,
                conf.GetAtomPosition(a.GetIdx()).z,
            )
            for a in mol.GetAtoms()
        ],
        dtype=float,
    )


def _normalize_mols_arg(mol: Chem.Mol | Sequence[Chem.Mol]) -> list[Chem.Mol]:
    if isinstance(mol, Chem.Mol):
        return [mol]
    try:
        mols = list(mol)
    except TypeError as e:
        raise TypeError(
            "mol must be an rdkit.Chem.Mol or an iterable of Mol instances"
        ) from e
    if not mols:
        raise ValueError("add_excluded_volume requires at least one molecule")
    for i, m in enumerate(mols):
        if not isinstance(m, Chem.Mol):
            raise TypeError(
                "add_excluded_volume expected RDKit Mol instances; "
                f"got {type(m).__name__!r} at index {i}"
            )
    return mols


def _conf_ids(mol: Chem.Mol, conf_id: int | None) -> list[int]:
    n = mol.GetNumConformers()
    if conf_id is None:
        return list(range(n))
    if conf_id < 0 or conf_id >= n:
        raise ValueError(
            f"conf_id {conf_id} is out of range for a molecule with {n} conformer(s)"
        )
    return [conf_id]


def _mol_has_heavy_atom(mol: Chem.Mol) -> bool:
    return any(a.GetAtomicNum() != 1 for a in mol.GetAtoms())


def _grid_axes_from_mols(
    mols: list[Chem.Mol],
    conf_id: int | None,
    *,
    shell_outer: float,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Union bounding box of all heavy-atom centres (all mols, selected confs)."""
    all_pts: list[tuple[float, float, float]] = []
    max_r = 0.0
    for mol in mols:
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                continue
            max_r = max(max_r, _vdw(atom.GetAtomicNum()))
            aid = atom.GetIdx()
            for cid in _conf_ids(mol, conf_id):
                p = mol.GetConformer(cid).GetAtomPosition(aid)
                all_pts.append((p.x, p.y, p.z))
    if not all_pts:
        return None
    centers = np.asarray(all_pts, dtype=float)
    pad = float(max_r) + shell_outer + spacing
    lo = centers.min(axis=0) - pad
    hi = centers.max(axis=0) + pad
    axes = [
        np.arange(lo[d], hi[d] + 0.5 * spacing, spacing, dtype=float)
        for d in range(3)
    ]
    gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    return gx, gy, gz


def _collect_union_heavy_centers_radii(
    mols: list[Chem.Mol],
    conf_id: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """All heavy-atom centres and vdW radii from every selected conformer.

    Used to treat the ensemble as one artificial molecule for envelope distance.
    """
    centers_list: list[tuple[float, float, float]] = []
    radii_list: list[float] = []
    for mol in mols:
        for cid in _conf_ids(mol, conf_id):
            conf = mol.GetConformer(cid)
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 1:
                    continue
                p = conf.GetAtomPosition(atom.GetIdx())
                centers_list.append((p.x, p.y, p.z))
                radii_list.append(_vdw(atom.GetAtomicNum()))
    return np.asarray(centers_list, dtype=float), np.asarray(radii_list, dtype=float)


def _min_surface_dist_grid_union(
    grid: np.ndarray,
    mols: list[Chem.Mol],
    conf_id: int | None,
) -> np.ndarray:
    """vdW surface distance from each grid point to the union of all pose spheres."""
    centers, radii = _collect_union_heavy_centers_radii(mols, conf_id)
    if centers.shape[0] == 0:
        return np.full(grid.shape[0], np.inf, dtype=float)
    ng = grid.shape[0]
    n_atom = centers.shape[0]
    out = np.empty(ng, dtype=float)
    # Limit grid×atom peak memory (still vectorised per chunk).
    max_pairs = 8_000_000
    chunk = max(2048, max_pairs // max(1, n_atom))
    for start in range(0, ng, chunk):
        stop = min(start + chunk, ng)
        g = grid[start:stop]
        diff = g[:, None, :] - centers[None, :, :]
        surf = np.sqrt(np.einsum("bad,bad->ba", diff, diff)) - radii[None, :]
        out[start:stop] = surf.min(axis=1)
    return out


def _thin_points_min_spacing(
    points: np.ndarray,
    priorities: np.ndarray,
    spacing: float,
    *,
    max_points: int | None,
) -> np.ndarray:
    """Keep a subset with pairwise Euclidean distance ≥ ``spacing``.

    Candidates are visited in ascending ``priorities`` order (lower first).
    If ``max_points`` is set, stop once that many centres have been kept (prioritising
    bulky shell regions first).
    """
    if points.shape[0] == 0:
        return points
    order = np.argsort(priorities, kind="mergesort")
    pts = points[order]
    kept: list[np.ndarray] = []
    thr = float(spacing)
    for p in pts:
        if max_points is not None and len(kept) >= max_points:
            break
        if not kept:
            kept.append(p)
            continue
        stack = np.stack(kept, axis=0)
        if np.linalg.norm(stack - p, axis=1).min() >= thr - 1e-9:
            kept.append(p)
    return np.stack(kept, axis=0) if kept else np.zeros((0, 3), dtype=float)


def query_pharmacophore_from_molecule(
    mol: Chem.Mol,
    perception: QueryPharmacophorePerception | None = None,
    *,
    conf_id: int = 0,
    name: str = "",
) -> QueryPharmacophore:
    """Build a :class:`QueryPharmacophore` from a 3D RDKit molecule.

    The user typically refines the result manually (e.g. converting an ``AROM``
    point to ``AROM_OR_LIPO`` or an ``HDON``/``HACC`` pair to ``HACC_OR_HDON``);
    the auto-perception emits only the seven elementary plus ``HACC_AND_HDON``
    types. Aromatic rings are reported as ``AROM`` only — :func:`_lipo_points`
    skips them so the same ring is never emitted as both ``AROM`` and ``LIPO``.
    """
    _require_conformer(mol)
    perception = perception or QueryPharmacophorePerception()
    pts = _perceive_points(mol, perception, conf_id)
    nm = name or (mol.GetProp("_Name") if mol.HasProp("_Name") else "")
    ph = QueryPharmacophore(name=nm)
    for p in pts:
        ph.add_point(p)
    return ph


def query_pharmacophore_from_protein(
    protein_pdb_filename: str | Path,
    excl_atoms_pdb_filename: str | Path,
    perception: QueryPharmacophorePerception | None = None,
    *,
    min_distance_between_excl_points: float = 1.5,
    conf_id: int = 0,
    excl_conf_id: int = 0,
    name: str = "",
    excl_sigma: float | None = None,
) -> QueryPharmacophore:
    """Derive a :class:`QueryPharmacophore` from PDB-defined protein and exclusions.

    Pharmacophore features (``AROM``, ``HACC``, …) are perceived from the structure
    in ``protein_pdb_filename`` using the same rules as
    :func:`query_pharmacophore_from_molecule`. Each atom in ``excl_atoms_pdb_filename``
    becomes an ``EXCL`` candidate at that atom's coordinates; candidates closer than
    ``min_distance_between_excl_points`` ångström to an already kept ``EXCL`` are
    dropped (greedy thinning in PDB atom order).

    Parameters
    ----------
    protein_pdb_filename, excl_atoms_pdb_filename : path-like
        Paths to PDB files. Both must contain at least one conformer with 3D coordinates.
    perception : QueryPharmacophorePerception, optional
        Same role as for :func:`query_pharmacophore_from_molecule`.
    min_distance_between_excl_points : float, optional
        Minimum centre-to-centre distance between emitted ``EXCL`` points after thinning.
        Must be ``>= 0``; ``0`` disables thinning (every atom yields one ``EXCL``).
    conf_id : int, optional
        Conformer index used when perceiving the protein.
    excl_conf_id : int, optional
        Conformer index used when reading exclusion atom coordinates.
    name : str, optional
        Query name; defaults to the protein PDB stem when empty.
    excl_sigma : float, optional
        Gaussian width for ``EXCL`` points; defaults to :data:`DEFAULT_SIGMA` ``EXCL``.
    """
    if min_distance_between_excl_points < 0:
        raise ValueError("min_distance_between_excl_points must be >= 0")

    protein_mol = _mol_from_pdb_file(protein_pdb_filename)
    nm = name or Path(protein_pdb_filename).stem
    q = query_pharmacophore_from_molecule(
        protein_mol,
        perception,
        conf_id=conf_id,
        name=nm,
    )

    excl_mol = _mol_from_pdb_file(excl_atoms_pdb_filename)
    coords = _all_atom_centers_array(excl_mol, excl_conf_id)
    if coords.shape[0] == 0:
        return q

    sigma_excl = float(
        DEFAULT_SIGMA[PointType.EXCL] if excl_sigma is None else excl_sigma
    )

    if min_distance_between_excl_points > 0:
        priorities = np.zeros(coords.shape[0], dtype=float)
        survivors = _thin_points_min_spacing(
            coords,
            priorities,
            float(min_distance_between_excl_points),
            max_points=None,
        )
    else:
        survivors = coords

    for x, y, z in survivors:
        q.add_point(
            PharmacophorePoint(
                type=PointType.EXCL,
                center=(float(x), float(y), float(z)),
                sigma=sigma_excl,
            )
        )
    return q


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


def add_excluded_volume(
    mol: Chem.Mol | Sequence[Chem.Mol],
    pharmacophore: Pharmacophore,
    *,
    conf_id: int | None = None,
    sigma: float | None = None,
    shell_inner: float = 1.0,
    shell_outer: float = 3.0,
    spacing: float = 1.5,
    feature_clearance: float = 0.0,
    max_excl: int = 0,
) -> int:
    """Add ``EXCL`` points on a shape envelope around 3D reference structure(s).

    A regular 3D grid is laid out over the bounding box of all heavy-atom centres
    from every molecule and selected conformer. Those atoms are treated as **one
    artificial molecule**: at each grid vertex the vdW surface distance is the
    minimum of ``distance − vdW radius`` over **all** heavy atoms in **all** poses
    (union of vdW spheres). Vertices whose distance lies in
    ``[shell_inner, shell_outer]`` are shell candidates.

    Candidates are optionally filtered by ``feature_clearance``, ordered by
    surface distance (closest-to-surface first among the union), then **thinned**:
    an ``EXCL`` is kept only if its centre is at least ``spacing`` ångström from
    every centre already kept. By default there is **no** limit on how many
    markers are kept (``max_excl=0``); pass a positive ``max_excl`` to cap the
    count after thinning.

    In Pharao scoring an ``EXCL`` query point penalises overlap with the
    candidate molecule's volume, so placing the spheres on a shell *outside*
    the reference ligand's surface defines a shape-complementarity envelope:
    candidate molecules whose features extend beyond that envelope are
    down-weighted.

    Parameters
    ----------
    mol : rdkit.Chem.Mol or sequence of Mol
        One or more 3D molecules with at least one conformer each. Hydrogens
        are ignored. Multiple molecules should occupy a **common coordinate
        frame** (e.g. superimposed); all poses contribute atoms to the union
        envelope.
    pharmacophore : Pharmacophore
        Target pharmacophore. Must allow ``EXCL`` (i.e. a
        :class:`QueryPharmacophore`); otherwise ``add_point`` raises.
    conf_id : int or None, optional
        If ``None`` (default), use **all** conformers of every molecule.
        If an integer, use only that conformer index in each molecule.
    sigma : float, optional
        Gaussian width (Å) of each EXCL point. Defaults to
        ``DEFAULT_SIGMA[PointType.EXCL]`` (1.6 Å).
    shell_inner, shell_outer : float, optional
        Inner / outer distance (Å) of the shell, measured from the nearest
        heavy-atom vdW surface. Defaults: ``1.0`` and ``3.0``. ``shell_inner``
        must be ``>= 0`` and strictly less than ``shell_outer``.
    spacing : float, optional
        Grid step (Å) along each axis (default ``1.5``). Also the **minimum**
        centre-to-centre separation between emitted ``EXCL`` markers after
        thinning.
    feature_clearance : float, optional
        If ``> 0``, candidate EXCL points within this distance (Å) of an
        existing pharmacophore feature centre are discarded. Useful when a
        ligand feature sits at the rim of the envelope and you don't want
        the shell to overlap it.
    max_excl : int, optional
        Maximum number of ``EXCL`` markers to append after thinning.
        Default ``0`` means no cap. Use any positive integer to stop after that
        many accepted centres.

    Returns
    -------
    int
        Number of EXCL points actually appended to ``pharmacophore``.
    """
    mols = _normalize_mols_arg(mol)
    if shell_inner < 0.0 or shell_outer <= shell_inner:
        raise ValueError(
            "shell_inner must be >= 0 and shell_outer must exceed shell_inner"
        )
    if spacing <= 0.0:
        raise ValueError("spacing must be > 0")

    cap: int | None = None if max_excl <= 0 else int(max_excl)

    excl_sigma = float(DEFAULT_SIGMA[PointType.EXCL] if sigma is None else sigma)

    for m in mols:
        _require_conformer(m)
        if not _mol_has_heavy_atom(m):
            return 0

    mesh = _grid_axes_from_mols(
        mols, conf_id, shell_outer=shell_outer, spacing=spacing
    )
    if mesh is None:
        return 0
    gx, gy, gz = mesh
    grid = np.stack(
        [gx.ravel(), gy.ravel(), gz.ravel()],
        axis=1,
    )  # (Ng, 3)

    sd_union = _min_surface_dist_grid_union(grid, mols, conf_id)
    shell_ok = (sd_union >= shell_inner) & (sd_union <= shell_outer)
    pooled = grid[shell_ok]
    priorities = sd_union[shell_ok]
    if pooled.shape[0] == 0:
        return 0

    if feature_clearance > 0.0:
        feature_pts = np.asarray(
            [
                (p.x, p.y, p.z)
                for p in pharmacophore
                if p.type != PointType.EXCL
            ],
            dtype=float,
        )
        if feature_pts.size:
            fdiff = pooled[:, None, :] - feature_pts[None, :, :]
            min_fdist_sq = np.einsum("pfd,pfd->pf", fdiff, fdiff).min(axis=1)
            keep_fc = min_fdist_sq >= feature_clearance * feature_clearance
            pooled = pooled[keep_fc]
            priorities = priorities[keep_fc]

    if pooled.shape[0] == 0:
        return 0
    survivors = _thin_points_min_spacing(
        pooled, priorities, float(spacing), max_points=cap
    )
    for x, y, z in survivors:
        pharmacophore.add_point(
            PharmacophorePoint(
                type=PointType.EXCL,
                center=(float(x), float(y), float(z)),
                sigma=excl_sigma,
            )
        )
    return int(survivors.shape[0])
