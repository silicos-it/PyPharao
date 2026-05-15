"""High-level pharmacophore search (`main.cpp` alignment branch)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .alignment import Alignment, SolutionInfo, position_molecule_coords, position_pharmacophore
from .perception_options import PerceptionOptions, perception_options_from_pharmacophore
from .pharmacophore import FuncGroup, Pharmacophore, PharmacophorePoint
from .quaternion_math import quat_to_rotation_matrix
from .function_mapping import FunctionMapping, functions_compatible
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
    return functions_compatible(ref.func, db.func)


def count_query_features(ref: Pharmacophore) -> int:
    """Number of reference points that must be matchable (excludes EXCL and UNDEF)."""
    return sum(
        1 for p in ref.points if p.func not in (FuncGroup.EXCL, FuncGroup.UNDEF)
    )


def matched_query_features(ref: Pharmacophore, mapping: list[tuple[int, int]]) -> int:
    """How many query points appear in a ref→db mapping (EXCL pairs excluded)."""
    return sum(1 for ri, _ in mapping if ref[ri].func != FuncGroup.EXCL)


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
    database_pharmacophore: Pharmacophore = field(default_factory=Pharmacophore)
    matched_db_pharmacophore: Pharmacophore = field(default_factory=Pharmacophore)
    aligned_mol: Any | None = None


@dataclass
class PharmacophoreSearch:
    """Align a reference pharmacophore to database pharmacophores or 3D molecules.

    When ``ref`` is provided at construction, ``perception_options`` is fixed to the
    molecule feature types required to match that query (see
    ``perception_options_from_pharmacophore``). ``search_with_molecule`` uses those
    flags unless a different reference is passed explicitly.

    ``min_matched_query_features`` sets how many query points must appear in the
    alignment mapping for a hit. ``None`` (default) requires every matchable query
    point (all except EXCL and UNDEF). Set to a smaller integer to allow partial matches.
    """

    ref: Pharmacophore | None = None
    epsilon: float = 0.5
    use_direction: bool = True
    with_exclusion: bool = True
    early_exit_score: float = 0.98
    min_matched_query_features: int | None = None
    _perception_options: PerceptionOptions | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._sync_perception_options()

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        if name == "ref":
            self._sync_perception_options()

    def _sync_perception_options(self) -> None:
        if self.ref is None:
            object.__setattr__(self, "_perception_options", None)
        else:
            object.__setattr__(
                self,
                "_perception_options",
                perception_options_from_pharmacophore(self.ref),
            )

    def refresh_perception_options(self) -> None:
        """Re-derive perception flags from the current ``ref`` (e.g. after in-place edits)."""
        self._sync_perception_options()

    @property
    def perception_options(self) -> PerceptionOptions | None:
        """Perception flags bound at init from ``ref`` (``None`` if no reference is set)."""
        return self._perception_options

    def _perception_options_for(self, ref: Pharmacophore) -> PerceptionOptions:
        if ref is self.ref and self._perception_options is not None:
            return self._perception_options
        return perception_options_from_pharmacophore(ref)

    def _effective_min_matched_query_features(self, ref: Pharmacophore) -> int:
        n_query = count_query_features(ref)
        if n_query == 0:
            return 0
        min_req = (
            n_query
            if self.min_matched_query_features is None
            else self.min_matched_query_features
        )
        if min_req < 1:
            raise ValueError("min_matched_query_features must be >= 1")
        if min_req > n_query:
            raise ValueError(
                f"min_matched_query_features ({min_req}) exceeds matchable query "
                f"features ({n_query})"
            )
        return min_req

    def _resolve_ref(
        self, ref: Pharmacophore | None, db: Pharmacophore | None
    ) -> tuple[Pharmacophore, Pharmacophore]:
        if db is None:
            if self.ref is None:
                raise ValueError(
                    "Reference pharmacophore required: pass PharmacophoreSearch(ref) "
                    "or search(ref, db)."
                )
            return self.ref, ref
        if ref is None:
            if self.ref is None:
                raise ValueError(
                    "Reference pharmacophore required: pass PharmacophoreSearch(ref) "
                    "or search(ref, db)."
                )
            return self.ref, db
        return ref, db

    def search(
        self, ref_or_db: Pharmacophore, db: Pharmacophore | None = None
    ) -> MatchResult:
        """Match reference to database pharmacophore.

        Call as ``search(db)`` when ``ref`` was passed to ``PharmacophoreSearch(ref)``,
        or as ``search(ref, db)`` for a one-off reference.
        """
        ref, db = self._resolve_ref(ref_or_db if db is not None else None, db or ref_or_db)
        use_dir = self.use_direction
        ref_volume, excl_size = compute_ref_volume(ref, use_dir)
        db_volume = compute_db_volume(db, use_dir)
        ref_size = len(ref)
        db_size = len(db)
        min_matched = self._effective_min_matched_query_features(ref)

        func_map = FunctionMapping(ref, db, self.epsilon)
        f_map = func_map.get_next_map()
        if not f_map or len(f_map) < min_matched:
            return MatchResult(
                ref_volume,
                db_volume,
                0.0,
                0.0,
                solution=SolutionInfo(),
                database_pharmacophore=db.copy(),
            )

        best = SolutionInfo()
        best.volume = -999.9
        best.rotor = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        best_score = -1000.0
        best_map: list[tuple[int, int]] = []
        map_size = len(f_map)
        max_size = map_size - 3

        while f_map:
            msize = len(f_map)
            if msize < min_matched:
                f_map = func_map.get_next_map()
                continue
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

        if matched_query_features(ref, best_map) < min_matched:
            best_map = []

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
            database_pharmacophore=db.copy(),
            matched_db_pharmacophore=matched,
            aligned_mol=None,
        )

    def search_with_molecule(
        self,
        mol_or_ref: Any,
        mol: Any | None = None,
        db: Pharmacophore | None = None,
        conf_id: int = 0,
    ) -> MatchResult:
        """Match reference to a molecule, building the database pharmacophore if needed.

        Preferred: ``search_with_molecule(mol)`` when ``ref`` was set on the searcher.
        Legacy: ``search_with_molecule(ref, mol)``.
        """
        if mol is not None:
            ref, mol = mol_or_ref, mol
        else:
            ref = self.ref
            mol = mol_or_ref
        if ref is None:
            raise ValueError(
                "Reference pharmacophore required: pass PharmacophoreSearch(ref) "
                "or search_with_molecule(ref, mol)."
            )
        if db is None:
            from .rdkit_perception import pharmacophore_from_molecule

            db = pharmacophore_from_molecule(
                mol, self._perception_options_for(ref), conf_id=conf_id
            )
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

    search_with_rdkit_mol = search_with_molecule
