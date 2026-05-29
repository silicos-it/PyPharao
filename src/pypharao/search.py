"""High-level pharmacophore search (`main.cpp` alignment branch).

Workflow
--------

1. Build a :class:`QueryPharmacophore` (manual, from JSON/`.phar`, or via
   :func:`pypharao.query_pharmacophore_from_molecule`).
2. Construct ``PharmacophoreSearch(query, perception=...)``. ``perception`` is
   an optional :class:`MoleculePharmacophorePerception` controlling how database
   molecules are perceived; defaults to "all seven molecule types enabled".
3. Call :meth:`PharmacophoreSearch.screen` with a single molecule or a batch.
   It always returns ``list[tuple[int, MatchResult]]`` (possibly empty).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .alignment import Alignment, SolutionInfo, position_molecule_coords, position_pharmacophore
from .function_mapping import FunctionMapping
from .perception import MoleculePharmacophorePerception
from .pharmacophore import (
    MoleculePharmacophore,
    Pharmacophore,
    PharmacophorePoint,
    PointType,
    QueryPharmacophore,
)
from .quaternion_math import quat_to_rotation_matrix
from .volume import volume_overlap

Metric = Literal["tanimoto", "overlap_volume", "excl_volume", "tversky_ref", "tversky_db"]
KeepMode = Literal["best", "all"]
Conformations = Literal["all", "single"] | int

_METRICS: tuple[str, ...] = (
    "tanimoto",
    "overlap_volume",
    "excl_volume",
    "tversky_ref",
    "tversky_db",
)
_MINIMISE_METRICS: frozenset[str] = frozenset({"excl_volume"})


@dataclass
class MatchResult:
    """A single (query, molecule-conformer) match.

    ``conf_id`` records which conformer of the input molecule produced this
    match (always ``0`` when the molecule has a single conformer).
    """

    ref_volume: float
    db_volume: float
    overlap_volume: float
    excl_volume: float
    tanimoto: float = 0.0
    tversky_ref: float = 0.0
    tversky_db: float = 0.0
    mapping: list[tuple[int, int]] = field(default_factory=list)
    database_pharmacophore: Pharmacophore = field(default_factory=MoleculePharmacophore)
    matched_db_pharmacophore: Pharmacophore = field(default_factory=MoleculePharmacophore)
    aligned_mol: Any | None = None
    conf_id: int = 0


def _is_rdkit_mol(obj: Any) -> bool:
    return callable(getattr(obj, "GetNumAtoms", None))


def _is_molecule_batch(mol: Any) -> bool:
    if isinstance(mol, (str, bytes)):
        return False
    if _is_rdkit_mol(mol):
        return False
    if not isinstance(mol, Sequence):
        return False
    if len(mol) == 0:
        return True
    first = mol[0]
    if _is_rdkit_mol(first):
        return True
    if isinstance(first, tuple) and len(first) >= 2 and _is_rdkit_mol(first[-1]):
        return True
    return False


def _parse_indexed_molecules(mols: Sequence[Any]) -> list[tuple[int, Any]]:
    """Normalise batch input to ``(index, Chem.Mol)`` pairs.

    Accepts a list of RDKit molecules (indices ``0, 1, …``) or tuples whose
    last element is a molecule, e.g. ``(line_idx, smiles, mol)`` — in that
    case ``line_idx`` is used as the reported index.
    """
    if len(mols) == 0:
        return []
    first = mols[0]
    if _is_rdkit_mol(first):
        return list(enumerate(mols))
    if isinstance(first, tuple) and len(first) >= 2 and _is_rdkit_mol(first[-1]):
        return [(int(item[0]), item[-1]) for item in mols]
    raise ValueError(
        "Batch input must be a list of RDKit molecules or tuples ending with a "
        "molecule, e.g. [(line_idx, smiles, mol), ...]."
    )


def _resolve_n_jobs(n_jobs: int) -> int:
    if n_jobs < 0:
        raise ValueError("n_jobs must be >= 0")
    if n_jobs == 0:
        return os.cpu_count() or 1
    return n_jobs


def _process_pool_context() -> mp.context.BaseContext | None:
    """Use fork on Unix so scripts run as ``python example.py`` need no ``__main__`` guard."""
    if sys.platform == "win32":
        return None
    try:
        return mp.get_context("fork")
    except ValueError:
        return None


def _progress_iter(iterable, *, total: int, desc: str):
    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise ImportError("progress=True requires tqdm (pip install tqdm)") from exc
    return tqdm(iterable, total=total, desc=desc, unit="mol")


def _resolve_conformations(mol: Any, conformations: Conformations) -> list[int]:
    n_conf = mol.GetNumConformers()
    if n_conf == 0:
        raise ValueError(
            "RDKit molecule must have at least one conformer with 3D coordinates"
        )
    all_ids = [c.GetId() for c in mol.GetConformers()]
    if conformations == "all":
        return all_ids
    if conformations == "single":
        return [all_ids[0]]
    if isinstance(conformations, int):
        if conformations <= 0:
            raise ValueError("conformations must be 'all', 'single', or a positive int")
        return all_ids[:conformations]
    raise ValueError(
        f"Unknown conformations value: {conformations!r}. "
        "Use 'all', 'single', or a positive int."
    )


def count_matchable_query_points(query: Pharmacophore) -> int:
    """Number of query points that can map to a molecule feature (excludes ``EXCL``)."""
    return sum(1 for p in query if p.type != PointType.EXCL)


def matched_query_features(query: Pharmacophore, mapping: list[tuple[int, int]]) -> int:
    """How many query points appear in a ref→db mapping (EXCL pairs excluded)."""
    return sum(1 for ri, _ in mapping if query[ri].type != PointType.EXCL)


def compute_ref_volume(ref: Pharmacophore, use_direction: bool) -> tuple[float, int]:
    """Returns ``(ref_volume, excl_count)`` per Pharao ``main.cpp`` reference block."""
    ref_volume = 0.0
    excl_size = 0
    for i in range(len(ref)):
        if ref[i].type == PointType.EXCL:
            excl_size += 1
            for j in range(len(ref)):
                if ref[j].type != PointType.EXCL:
                    ref_volume -= volume_overlap(ref[i], ref[j], use_direction)
        else:
            ref_volume += volume_overlap(ref[i], ref[i], use_direction)
    return ref_volume, excl_size


def compute_db_volume(db: Pharmacophore, use_direction: bool) -> float:
    v = 0.0
    for i in range(len(db)):
        if db[i].type == PointType.EXCL:
            continue
        v += volume_overlap(db[i], db[i], use_direction)
    return v


def _validate_metric(metric: str) -> None:
    if metric not in _METRICS:
        raise ValueError(
            f"Unknown metric {metric!r}. Expected one of: {', '.join(_METRICS)}"
        )


def _score_for_metric(result: MatchResult, metric: str) -> float:
    return float(getattr(result, metric))


def _is_better(candidate: MatchResult, current: MatchResult | None, metric: str) -> bool:
    if current is None:
        return True
    cand = _score_for_metric(candidate, metric)
    cur = _score_for_metric(current, metric)
    return cand < cur if metric in _MINIMISE_METRICS else cand > cur


@dataclass(frozen=True)
class _ScreenConfig:
    query: QueryPharmacophore
    perception: MoleculePharmacophorePerception
    epsilon: float
    use_direction: bool
    score_with_direction: bool
    with_exclusion: bool
    early_exit_score: float
    excl_hard_filter: bool
    excl_clash_radius: float
    conformations: Conformations
    min_matches: int
    keep: KeepMode
    metric: str


def _build_aligned_mol(mol: Any, conf_id: int, alignment: SolutionInfo) -> Any | None:
    """Return a copy of ``mol`` with conformer ``conf_id`` rotated/translated into the query frame."""
    try:
        from rdkit import Chem
    except ImportError:
        return None
    if mol.GetNumConformers() == 0:
        return None
    try:
        m = Chem.Mol(mol)
        conf = m.GetConformer(conf_id)
        n = m.GetNumAtoms()
        coords = np.zeros((n, 3), dtype=float)
        for a in range(n):
            p = conf.GetAtomPosition(a)
            coords[a, 0], coords[a, 1], coords[a, 2] = p.x, p.y, p.z
        rot = quat_to_rotation_matrix(alignment.rotor)
        new_xyz = position_molecule_coords(coords, rot, alignment)
        for a in range(n):
            conf.SetAtomPosition(a, new_xyz[a].tolist())
        # Drop every other conformer so the returned molecule reflects the aligned pose only.
        keep_id = conf.GetId()
        for c in list(m.GetConformers()):
            if c.GetId() != keep_id:
                m.RemoveConformer(c.GetId())
        return m
    except Exception:
        return None


def _aligned_heavy_atom_coords(
    mol: Any, conf_id: int, alignment: SolutionInfo
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(coords, vdw_radii)`` for heavy atoms, in the query frame.

    ``coords`` is an ``(N, 3)`` float array of heavy-atom positions transformed
    via ``alignment``; ``vdw_radii`` is an ``(N,)`` float array of their
    Bondi-style van der Waals radii (``rdkit.Chem.GetPeriodicTable``).
    Returns ``None`` if ``mol`` has no conformers or no heavy atoms.
    """
    if mol.GetNumConformers() == 0:
        return None
    try:
        from rdkit import Chem
    except ImportError:
        return None
    conf = mol.GetConformer(conf_id)
    periodic = Chem.GetPeriodicTable()
    rows: list[tuple[float, float, float]] = []
    vdw: list[float] = []
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if z == 1:
            continue
        p = conf.GetAtomPosition(atom.GetIdx())
        rows.append((p.x, p.y, p.z))
        vdw.append(float(periodic.GetRvdw(z)))
    if not rows:
        return None
    coords = np.asarray(rows, dtype=float)
    rot = quat_to_rotation_matrix(alignment.rotor)
    aligned = position_molecule_coords(coords, rot, alignment)
    return aligned, np.asarray(vdw, dtype=float)


def _has_excl_atom_clash(
    query: Pharmacophore,
    heavy_coords: np.ndarray,
    heavy_vdw: np.ndarray,
    excl_clash_radius: float,
) -> bool:
    """Return ``True`` if any heavy-atom vdW sphere overlaps a query EXCL marker.

    Each heavy atom is modelled as a sphere of its van der Waals radius and
    each EXCL point as a sphere of ``excl_clash_radius`` (``0.0`` means the
    EXCL is a bare marker point). A clash is registered when

    ``dist(atom_center, EXCL_center) < vdW(atom) + excl_clash_radius``.

    This is independent of the EXCL's Gaussian ``sigma`` (which only controls
    the soft Pharao volume penalty).
    """
    centers: list[tuple[float, float, float]] = []
    for p in query:
        if p.type != PointType.EXCL:
            continue
        centers.append((p.x, p.y, p.z))
    if not centers:
        return False
    excl_centers = np.asarray(centers, dtype=float)
    diff = heavy_coords[:, None, :] - excl_centers[None, :, :]
    dist_sq = np.einsum("aed,aed->ae", diff, diff)
    thresh = heavy_vdw[:, None] + float(excl_clash_radius)
    return bool(np.any(dist_sq < thresh * thresh))


def _worker(args: tuple[_ScreenConfig, int, Any]) -> list[tuple[int, MatchResult]]:
    config, index, mol = args
    searcher = PharmacophoreSearch(
        query=config.query,
        perception=config.perception,
        epsilon=config.epsilon,
        use_direction=config.use_direction,
        score_with_direction=config.score_with_direction,
        with_exclusion=config.with_exclusion,
        early_exit_score=config.early_exit_score,
        excl_hard_filter=config.excl_hard_filter,
        excl_clash_radius=config.excl_clash_radius,
    )
    hits = searcher._screen_one_mol(
        mol,
        conformations=config.conformations,
        min_matches=config.min_matches,
        keep=config.keep,
        metric=config.metric,
    )
    return [(index, h) for h in hits]


@dataclass
class PharmacophoreSearch:
    """Align a :class:`QueryPharmacophore` to one or more database molecules.

    ``perception`` controls how each database molecule is converted to a
    :class:`MoleculePharmacophore`. When ``None`` (the default) every molecule
    feature type is enabled.

    Direction (normal) handling is split across two independent flags:

    * ``use_direction`` (default ``True``) — apply the directional cosine
      factor inside the **alignment** optimiser (``Alignment.align``). This
      shapes the rotor/translation that maximises the Pharao volume objective
      for aromatic-like and h-bond-like pairs (when both points carry a
      normal).
    * ``score_with_direction`` (default ``False``) — apply the same cosine
      factor to the **final volume score** that produces
      ``MatchResult.overlap_volume``, ``tanimoto`` and ``tversky_*``. By
      default the score collapses to the pure Gaussian centre overlap (with
      ``EXCL`` penalties), so the reported similarity reflects geometric
      placement only. Set ``score_with_direction=True`` to recover the
      original Pharao behaviour in which the cosine factor multiplies both
      alignment and scoring.

    Excluded-volume handling has two layers:

    * **Soft Pharao penalty** (``with_exclusion``) — included in the alignment
      objective and reflected by ``MatchResult.excl_volume``.
    * **Hard atom-clash filter** (``excl_hard_filter``) — applied *after*
      alignment. Heavy atoms of the aligned database molecule are projected
      into the query frame and a hit is discarded as soon as any heavy-atom
      vdW sphere overlaps a query EXCL marker:

      ``dist(atom_center, EXCL_center) < vdW(atom) + excl_clash_radius``.

      The default ``excl_clash_radius=0.0`` treats each EXCL as a bare marker
      point — a clash is reported when an atom's vdW sphere swallows the
      marker. Increase ``excl_clash_radius`` to enforce a larger forbidden
      buffer around each EXCL. This criterion is independent of the EXCL's
      Gaussian ``sigma`` (which only feeds the soft penalty).
    """

    query: QueryPharmacophore
    perception: MoleculePharmacophorePerception | None = None
    epsilon: float = 0.5
    use_direction: bool = True
    score_with_direction: bool = False
    with_exclusion: bool = True
    early_exit_score: float = 0.98
    excl_hard_filter: bool = True
    excl_clash_radius: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.query, QueryPharmacophore):
            raise TypeError(
                "PharmacophoreSearch.query must be a QueryPharmacophore "
                f"(got {type(self.query).__name__})"
            )
        if self.perception is None:
            self.perception = MoleculePharmacophorePerception()
        if self.excl_clash_radius < 0.0:
            raise ValueError("excl_clash_radius must be >= 0")

    # ------------------------------------------------------------------ search
    def _search_with_alignment(
        self,
        query: QueryPharmacophore,
        db: Pharmacophore,
        min_matches: int,
    ) -> tuple[MatchResult, SolutionInfo]:
        use_dir = self.use_direction
        score_dir = self.score_with_direction
        ref_volume, excl_size = compute_ref_volume(query, score_dir)
        db_volume = compute_db_volume(db, score_dir)
        ref_size = len(query)
        db_size = len(db)

        func_map = FunctionMapping(query, db, self.epsilon)
        f_map = func_map.get_next_map()
        if not f_map or len(f_map) < min_matches:
            return (
                MatchResult(
                    ref_volume,
                    db_volume,
                    0.0,
                    0.0,
                    database_pharmacophore=db.copy(),
                ),
                SolutionInfo(),
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
            if msize < min_matches:
                f_map = func_map.get_next_map()
                continue
            working = list(f_map)
            if self.with_exclusion:
                for i in range(ref_size):
                    if query[i].type != PointType.EXCL:
                        continue
                    for j in range(db_size):
                        if db[j].type == PointType.EXCL:
                            continue
                        working.append((i, j))

            if (msize > max_size) and (
                (msize / (ref_size - excl_size + db_size - msize)) > best_score
            ):
                aligner = Alignment(query, db, working)
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

        if matched_query_features(query, best_map) < min_matches:
            best_map = []

        rot_mat = quat_to_rotation_matrix(best.rotor)
        db_work = db.copy()
        if best_map and best.volume > -500.0:
            position_pharmacophore(db_work, rot_mat, best)

        excl_vol = 0.0
        for i in range(ref_size):
            if query[i].type != PointType.EXCL:
                continue
            for j in range(db_size):
                excl_vol += volume_overlap(query[i], db_work[j], score_dir)

        overlap_vol = 0.0
        matched = MoleculePharmacophore()
        for ri, di in best_map:
            rp, dp = query[ri], db_work[di]
            if rp.type == PointType.EXCL or dp.type == PointType.EXCL:
                continue
            overlap_vol += volume_overlap(rp, dp, score_dir)
            normal = (dp.nx, dp.ny, dp.nz) if dp.has_normal else None
            matched.add_point(
                PharmacophorePoint(
                    type=dp.type,
                    center=(dp.x, dp.y, dp.z),
                    sigma=dp.sigma,
                    normal=normal,
                )
            )

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

        return (
            MatchResult(
                ref_volume=ref_volume,
                db_volume=db_volume,
                overlap_volume=overlap_vol,
                excl_volume=excl_vol,
                tanimoto=tan,
                tversky_ref=tref,
                tversky_db=tdb,
                mapping=best_map,
                database_pharmacophore=db.copy(),
                matched_db_pharmacophore=matched,
                aligned_mol=None,
            ),
            best,
        )

    # --------------------------------------------------------- public screen()
    def _screen_one_mol(
        self,
        mol: Any,
        *,
        conformations: Conformations,
        min_matches: int,
        keep: KeepMode,
        metric: str,
    ) -> list[MatchResult]:
        from .rdkit_perception import molecule_pharmacophore_from_molecule

        conf_ids = _resolve_conformations(mol, conformations)
        # Only run the hard atom-clash filter when the query actually has
        # EXCL features; precompute the flag once per molecule.
        hard_filter_active = self.excl_hard_filter and any(
            p.type == PointType.EXCL for p in self.query
        )
        results: list[MatchResult] = []
        best: MatchResult | None = None
        for cid in conf_ids:
            db = molecule_pharmacophore_from_molecule(
                mol, self.perception, conf_id=cid
            )
            match, alignment = self._search_with_alignment(self.query, db, min_matches)
            if not match.mapping:
                continue
            if hard_filter_active:
                aligned = _aligned_heavy_atom_coords(mol, cid, alignment)
                if aligned is not None:
                    heavy_xyz, heavy_vdw = aligned
                    if _has_excl_atom_clash(
                        self.query, heavy_xyz, heavy_vdw, self.excl_clash_radius
                    ):
                        continue
            match.conf_id = cid
            match.aligned_mol = _build_aligned_mol(mol, cid, alignment)
            if keep == "best":
                if _is_better(match, best, metric):
                    best = match
            else:
                results.append(match)
        if keep == "best":
            return [best] if best is not None else []
        return results

    def screen(
        self,
        mols: Any,
        *,
        conformations: Conformations = "all",
        min_matches: int = 0,
        keep: KeepMode = "best",
        metric: Metric = "tanimoto",
        n_jobs: int = 0,
        progress: bool = True,
    ) -> list[tuple[int, MatchResult]]:
        """Match the query against one or more molecules.

        Parameters
        ----------
        mols
            A single ``Chem.Mol``, a list of ``Chem.Mol``, or a list of tuples
            whose last element is a molecule (e.g. ``(line_idx, smiles, mol)`` —
            the first element is used as the reported hit index).
        conformations
            ``'all'`` (default), ``'single'`` (first conformer only), or a
            positive ``int`` for the first N conformers.
        min_matches
            Minimum number of query points that must map to a molecule feature
            for a hit. ``0`` (the default) is auto-resolved to
            ``count_matchable_query_points(query)`` — i.e. every query point
            other than ``EXCL`` and ``UNDEF`` must match. Negative values or
            values exceeding the matchable count raise ``ValueError``.
        keep
            ``'best'`` (default) keeps only the highest-scoring conformer per
            molecule; ``'all'`` keeps every conformer that satisfies
            ``min_matches``.
        metric
            Tie-breaker for ``keep='best'``. One of ``'tanimoto'``,
            ``'overlap_volume'``, ``'excl_volume'``, ``'tversky_ref'``,
            ``'tversky_db'``. All metrics are maximised except ``excl_volume``
            (minimised — smaller overlap with the exclusion sphere is better).
        n_jobs
            Worker processes. ``0`` (default) uses every available CPU; ``1``
            runs sequentially.
        progress
            Show a tqdm progress bar over the molecule list (requires the
            optional ``tqdm`` dependency).

        Returns
        -------
        list[tuple[int, MatchResult]]
            One entry per matched (molecule, conformer). Empty when nothing
            matched.
        """
        _validate_metric(metric)
        if min_matches < 0:
            raise ValueError("min_matches must be >= 0")
        max_matches = count_matchable_query_points(self.query)
        if min_matches == 0:
            min_matches = max_matches
        elif min_matches > max_matches:
            raise ValueError(
                f"min_matches ({min_matches}) exceeds matchable query points "
                f"({max_matches})"
            )

        if _is_molecule_batch(mols):
            indexed = _parse_indexed_molecules(mols)
        else:
            if not _is_rdkit_mol(mols):
                raise TypeError(
                    "mols must be an RDKit Chem.Mol or a sequence of molecules"
                )
            indexed = [(0, mols)]

        if not indexed:
            return []

        config = _ScreenConfig(
            query=self.query,
            perception=self.perception,
            epsilon=self.epsilon,
            use_direction=self.use_direction,
            score_with_direction=self.score_with_direction,
            with_exclusion=self.with_exclusion,
            early_exit_score=self.early_exit_score,
            excl_hard_filter=self.excl_hard_filter,
            excl_clash_radius=self.excl_clash_radius,
            conformations=conformations,
            min_matches=min_matches,
            keep=keep,
            metric=metric,
        )

        workers = min(_resolve_n_jobs(n_jobs), len(indexed))
        total = len(indexed)
        hits: list[tuple[int, MatchResult]] = []
        if workers <= 1:
            iterable: Any = indexed
            if progress:
                iterable = _progress_iter(iterable, total=total, desc="Screening")
            for index, mol in iterable:
                for match in self._screen_one_mol(
                    mol,
                    conformations=conformations,
                    min_matches=min_matches,
                    keep=keep,
                    metric=metric,
                ):
                    hits.append((index, match))
            return hits

        tasks = [(config, index, mol) for index, mol in indexed]
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=_process_pool_context(),
        ) as executor:
            results = executor.map(_worker, tasks)
            if progress:
                results = _progress_iter(results, total=total, desc="Screening")
            for chunk in results:
                hits.extend(chunk)
        return hits
