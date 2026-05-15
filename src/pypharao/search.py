"""High-level pharmacophore search (`main.cpp` alignment branch)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .alignment import Alignment, SolutionInfo, position_molecule_coords, position_pharmacophore
from .pharmacophore import FuncGroup, Pharmacophore, PharmacophorePoint
from .quaternion_math import quat_to_rotation_matrix
from .function_mapping import FunctionMapping
from .volume import volume_overlap

if TYPE_CHECKING:
    pass


def compute_ref_volume(ref: Pharmacophore, use_direction: bool) -> tuple[float, int]:
    """Returns (ref_volume, excl_count) per Pharao main.cpp reference block."""
    ref_volume = 0.0
    excl_size = 0
    for i in range(len(ref)):
        if ref[i].func == FuncGroup.EXCL:
            excl_size += 1
            for j in range(len(ref)):
                if ref[j].func != FuncGroup.EXCL:
                    ref_volume -= volume_overlap(ref[i], ref[j], use_direction)
        else:
            ref_volume += volume_overlap(ref[i], ref[i], use_direction)
    return ref_volume, excl_size


def compute_db_volume(db: Pharmacophore, use_direction: bool) -> float:
    v = 0.0
    for i in range(len(db)):
        if db[i].func == FuncGroup.EXCL:
            continue
        v += volume_overlap(db[i], db[i], use_direction)
    return v


def _pair_counts_for_overlap(ref: PharmacophorePoint, db: PharmacophorePoint) -> bool:
    if ref.func == db.func:
        return True
    if ref.func in (FuncGroup.HYBH, FuncGroup.HDON, FuncGroup.HACC) and db.func in (
        FuncGroup.HDON,
        FuncGroup.HACC,
        FuncGroup.HYBH,
    ):
        return True
    if ref.func in (FuncGroup.HYBL, FuncGroup.AROM, FuncGroup.LIPO) and db.func in (
        FuncGroup.AROM,
        FuncGroup.LIPO,
        FuncGroup.HYBL,
    ):
        return True
    return False


@dataclass
class MatchResult:
    ref_volume: float
    db_volume: float
    overlap_volume: float
    excl_volume: float
    tanimoto: float = 0.0
    tversky_ref: float = 0.0
    tversky_db: float = 0.0
    solution: SolutionInfo = field(default_factory=SolutionInfo)
    mapping: list[tuple[int, int]] = field(default_factory=list)
    matched_db_pharmacophore: Pharmacophore = field(default_factory=Pharmacophore)
    aligned_mol: Any | None = None


@dataclass
class PharmacophoreSearch:
    epsilon: float = 0.5
    use_direction: bool = True
    with_exclusion: bool = True
    early_exit_score: float = 0.98

    def search(self, ref: Pharmacophore, db: Pharmacophore) -> MatchResult:
        use_dir = self.use_direction
        ref_volume, excl_size = compute_ref_volume(ref, use_dir)
        db_volume = compute_db_volume(db, use_dir)
        ref_size = len(ref)
        db_size = len(db)

        func_map = FunctionMapping(ref, db, self.epsilon)
        f_map = func_map.get_next_map()
        if not f_map:
            return MatchResult(ref_volume, db_volume, 0.0, 0.0, solution=SolutionInfo())

        best = SolutionInfo()
        best.volume = -999.9
        best.rotor = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        best_score = -1000.0
        best_map: list[tuple[int, int]] = []
        map_size = len(f_map)
        max_size = map_size - 3

        while f_map:
            msize = len(f_map)
            working = list(f_map)
            if self.with_exclusion:
                for i in range(ref_size):
                    if ref[i].func != FuncGroup.EXCL:
                        continue
                    for j in range(db_size):
                        if db[j].func == FuncGroup.EXCL:
                            continue
                        working.append((i, j))

            if (msize > max_size) and (
                (msize / (ref_size - excl_size + db_size - msize)) > best_score
            ):
                aligner = Alignment(ref, db, working)
                r = aligner.align(use_dir)
                if best.volume < r.volume:
                    best = r
                    den = ref_volume + db_volume - best.volume
                    best_score = best.volume / den if den > 1e-12 else 0.0
                    best_map = list(f_map)
                    map_size = msize
            else:
                break

            if best_score > self.early_exit_score:
                break

            f_map = func_map.get_next_map()

        rot_mat = quat_to_rotation_matrix(best.rotor)
        db_work = db.copy()
        if best_map and best.volume > -500.0:
            position_pharmacophore(db_work, rot_mat, best)

        excl_vol = 0.0
        for i in range(ref_size):
            if ref[i].func != FuncGroup.EXCL:
                continue
            for j in range(db_size):
                excl_vol += volume_overlap(ref[i], db_work[j], use_dir)

        overlap_vol = 0.0
        matched = Pharmacophore()
        for ri, di in best_map:
            rp, dp = ref[ri], db_work[di]
            if rp.func == FuncGroup.EXCL or dp.func == FuncGroup.EXCL:
                continue
            overlap_vol += volume_overlap(rp, dp, use_dir)
            matched.append_point(PharmacophorePoint(dp.x, dp.y, dp.z, dp.func, dp.alpha, dp.has_normal, dp.nx, dp.ny, dp.nz))

        aligned_vol = overlap_vol - excl_vol
        best.volume = aligned_vol

        tan = tref = tdb = 0.0
        if aligned_vol > 0.0:
            den = ref_volume + db_volume - aligned_vol
            if den > 1e-12:
                tan = aligned_vol / den
            if ref_volume > 1e-12:
                tref = aligned_vol / ref_volume
            if db_volume > 1e-12:
                tdb = aligned_vol / db_volume

        return MatchResult(
            ref_volume=ref_volume,
            db_volume=db_volume,
            overlap_volume=overlap_vol,
            excl_volume=excl_vol,
            tanimoto=tan,
            tversky_ref=tref,
            tversky_db=tdb,
            solution=best,
            mapping=best_map,
            matched_db_pharmacophore=matched,
            aligned_mol=None,
        )

    def search_with_rdkit_mol(
        self,
        ref: Pharmacophore,
        mol: Any,
        db: Pharmacophore | None = None,
        conf_id: int = 0,
    ) -> MatchResult:
        """Match `ref` to `db` pharmacophore, or compute `db` from an RDKit molecule."""
        if db is None:
            from .perception_options import PerceptionOptions
            from .rdkit_perception import pharmacophore_from_rdkit

            db = pharmacophore_from_rdkit(mol, PerceptionOptions(), conf_id=conf_id)
        res = self.search(ref, db)
        try:
            from rdkit import Chem

            if mol.GetNumConformers() == 0:
                return res
            m = Chem.Mol(mol)
            conf = m.GetConformer(conf_id)
            n = m.GetNumAtoms()
            coords = np.zeros((n, 3), dtype=float)
            for a in range(n):
                p = conf.GetAtomPosition(a)
                coords[a, 0], coords[a, 1], coords[a, 2] = p.x, p.y, p.z
            rot = quat_to_rotation_matrix(res.solution.rotor)
            new_xyz = position_molecule_coords(coords, rot, res.solution)
            for a in range(n):
                conf.SetAtomPosition(a, new_xyz[a].tolist())
            res.aligned_mol = m
        except Exception:
            res.aligned_mol = None
        return res
