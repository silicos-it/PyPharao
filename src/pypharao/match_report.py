"""Human-readable reports and SDF / PDB output for pharmacophore screen results."""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TextIO

from .pharmacophore import Pharmacophore, PharmacophorePoint, PointType
from .search import MatchResult


# Length (in angstrom) at which each normal-tip pseudo-atom is placed away
# from the feature centre when rendering directional features as SDF / PDB.
# Perception already stores aromatic normals at unit distance, so keeping this
# at 1.0 A means the visual tip coincides with the perceived absolute tip.
NORMAL_TIP_LENGTH: float = 1.0

SortOrder = Literal["ascending", "descending"]
SortKey = Literal[
    "ref_volume",
    "db_volume",
    "overlap_volume",
    "excl_volume",
    "tanimoto",
    "tversky_ref",
    "tversky_db",
]

_TABLE_HEADERS = [
    "index",
    "conf_id",
    "ref_volume",
    "db_volume",
    "overlap_volume",
    "excl_volume",
    "tanimoto",
    "tversky_ref",
    "tversky_db",
    "mapping",
    "database_pharmacophore",
    "matched_db_pharmacophore",
    "aligned_mol",
]

_SDF_TAGS: list[str] = [
    "index",
    "conf_id",
    "tanimoto",
    "tversky_ref",
    "tversky_db",
    "overlap_volume",
    "excl_volume",
    "ref_volume",
    "db_volume",
]


_TYPE_TO_ELEMENT: dict[PointType, str] = {
    PointType.AROM: "C",
    PointType.LIPO: "C",
    PointType.AROM_OR_LIPO: "C",
    PointType.HDON: "N",
    PointType.HACC: "O",
    PointType.HACC_AND_HDON: "S",
    PointType.HACC_OR_HDON: "S",
    PointType.POSC: "Na",
    PointType.NEGC: "Cl",
    PointType.EXCL: "F",
    PointType.UNDEF: "He",
}


_TYPE_TO_PDB_RESNAME: dict[PointType, str] = {
    PointType.AROM: "ARO",
    PointType.LIPO: "LIP",
    PointType.AROM_OR_LIPO: "AOL",
    PointType.HDON: "HDO",
    PointType.HACC: "HAC",
    PointType.HACC_AND_HDON: "DAC",
    PointType.HACC_OR_HDON: "DAO",
    PointType.POSC: "POS",
    PointType.NEGC: "NEG",
    PointType.EXCL: "EXC",
    PointType.UNDEF: "UND",
}


_TYPE_TO_ATOM_CODE: dict[PointType, str] = {
    PointType.AROM: "A",
    PointType.LIPO: "L",
    PointType.AROM_OR_LIPO: "M",
    PointType.HDON: "D",
    PointType.HACC: "C",
    PointType.HACC_AND_HDON: "B",
    PointType.HACC_OR_HDON: "R",
    PointType.POSC: "P",
    PointType.NEGC: "N",
    PointType.EXCL: "X",
    PointType.UNDEF: "U",
}


def _format_mapping(mapping: Sequence[tuple[int, int]]) -> str:
    if not mapping:
        return "—"
    return ",".join(f"({r},{d})" for r, d in mapping)


def _pharmacophore_summary(ph: Pharmacophore) -> str:
    pts = list(ph)
    if not pts:
        return "[]"
    return "[" + ",".join(p.type.value for p in pts) + "]"


def _aligned_mol_smiles(mol: Any) -> str:
    if mol is None:
        return "—"
    try:
        from rdkit import Chem

        return Chem.MolToSmiles(Chem.RemoveHs(mol))
    except Exception:
        return "—"


def _result_row(index: int, hit: MatchResult) -> list[str]:
    return [
        str(index),
        str(hit.conf_id),
        f"{hit.ref_volume:.2f}",
        f"{hit.db_volume:.2f}",
        f"{hit.overlap_volume:.2f}",
        f"{hit.excl_volume:.2f}",
        f"{hit.tanimoto:.3f}",
        f"{hit.tversky_ref:.3f}",
        f"{hit.tversky_db:.3f}",
        _format_mapping(hit.mapping),
        _pharmacophore_summary(hit.database_pharmacophore),
        _pharmacophore_summary(hit.matched_db_pharmacophore),
        _aligned_mol_smiles(hit.aligned_mol),
    ]


def sort_match_results(
    matches: list[tuple[int, MatchResult]],
    *,
    sort: SortOrder = "descending",
    key: SortKey = "tanimoto",
) -> list[tuple[int, MatchResult]]:
    """Sort screen hits by a numeric :class:`MatchResult` field.

    Returns a new list; the input is not modified.
    """
    reverse = sort == "descending"
    return sorted(matches, key=lambda item: getattr(item[1], key), reverse=reverse)


def _print_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    file: TextIO,
) -> None:
    print("\t".join(headers), file=file)
    for row in rows:
        print("\t".join(row), file=file)


def print_match_results(
    results: list[tuple[int, MatchResult]],
    *,
    limit: int | None = None,
    file: TextIO | None = None,
) -> None:
    """Print a tab-separated summary of screen hits.

    Expects the ``list[tuple[int, MatchResult]]`` returned by
    :meth:`pypharao.PharmacophoreSearch.screen`. Prints ``No hits.`` when the
    list is empty.
    """
    out = file if file is not None else sys.stdout

    if limit is not None and limit < 0:
        raise ValueError("limit must be >= 0")

    if not results:
        print("No hits.", file=out)
        return

    hits = list(results)
    if limit is not None:
        total = len(hits)
        hits = hits[:limit]
        if total > limit:
            print(f"Showing {limit} of {total} hits.", file=out)

    rows = [_result_row(index, hit) for index, hit in hits]
    _print_table(_TABLE_HEADERS, rows, file=out)


def _require_rdkit(func_name: str) -> Any:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError(
            f"{func_name} requires RDKit (pip install rdkit)"
        ) from exc
    return Chem


def _pdb_atom_name(point_type: PointType, idx: int) -> str:
    """4-character PDB atom name combining a type code and the point index."""
    code = _TYPE_TO_ATOM_CODE[point_type]
    return f"{code}{idx + 1:03d}"[:4]


def _pdb_tip_atom_name(idx: int, side: str) -> str:
    """4-character PDB atom name for a normal-tip pseudo-atom.

    ``side`` must be ``"+"`` (above the plane) or ``"-"`` (below). The name is
    ``"<side><001-indexed feature idx>"`` truncated to 4 chars (so up to 999
    features are unambiguous; feature 1000+ would just collide with feature
    100+, but this is a visualisation aid only).
    """
    if side not in ("+", "-"):
        raise ValueError("side must be '+' or '-'")
    return f"{side}{idx + 1:03d}"[:4]


def _unit_normal_direction(
    p: PharmacophorePoint,
) -> tuple[float, float, float] | None:
    """Unit vector from a point's centre to its absolute normal tip.

    Returns ``None`` if the point has no normal or the stored normal is
    degenerate (length below ``1e-9``).
    """
    if not p.has_normal:
        return None
    dx, dy, dz = p.nx - p.x, p.ny - p.y, p.nz - p.z
    n = math.sqrt(dx * dx + dy * dy + dz * dz)
    if n < 1e-9:
        return None
    return (dx / n, dy / n, dz / n)


def pharmacophore_to_mol(
    ph: Pharmacophore,
    *,
    name: str = "pharmacophore",
) -> Any:
    """Build a ``Chem.Mol`` representation of a pharmacophore.

    Each pharmacophore feature becomes one **centre** pseudo-atom whose
    element loosely encodes the feature type (AROM/LIPO → C, HDON → N,
    HACC → O, HACC_AND_HDON / HACC_OR_HDON → S, POSC → Na, NEGC → Cl,
    EXCL → F, UNDEF → He). The centre carries PDB residue info so PDB output
    uses the type code as the residue name (e.g. ``ARO``, ``HDO``, ``HAC``)
    and SDF / mol-block writers preserve the 3D coordinates.

    **Directional features** (currently ``AROM`` and ``AROM_OR_LIPO``, see
    :data:`pypharao.pharmacophore.TYPE_HAS_NORMAL`) additionally emit **two
    "tip" pseudo-atoms** (element ``H``), placed at
    ``centre ± unit_normal * NORMAL_TIP_LENGTH`` (1 Å by default). Both
    tips share the parent feature's PDB residue name and number, with atom
    names ``"+001"`` / ``"-001"`` (one-indexed by feature). Each tip is bonded
    to its centre by a single bond so 3D viewers render the plane normal as a
    visible axis. The symmetric pair reflects the Pharao convention that the
    aromatic cosine term uses ``|cos|`` — both ±n are equally valid normals.

    The molecule is **not** sanitised (these are pseudo-atoms with
    non-physical valences) but is fully usable with ``Chem.SDWriter``,
    ``Chem.MolToPDBBlock`` and the like. The return type is ``Chem.Mol``
    even though it's typed ``Any`` because RDKit is an optional dependency.

    Per-feature SDF properties on the returned molecule (one entry per
    *feature*, independent of how many atoms each feature contributes):

    - ``num_features`` (int) — number of pharmacophore features, **not**
      number of atoms
    - ``types`` (comma-separated list of point types)
    - ``sigmas`` (comma-separated list of sigmas)
    - ``centers`` (semicolon-separated ``x,y,z`` triples)
    - ``normals`` (semicolon-separated ``nx,ny,nz`` triples; ``-`` when absent)

    Requires RDKit (raises ``ImportError`` otherwise).
    """
    Chem = _require_rdkit("pharmacophore_to_mol")

    if not isinstance(ph, Pharmacophore):
        raise TypeError(
            "pharmacophore_to_mol expected a Pharmacophore "
            f"(got {type(ph).__name__}); pass the query pharmacophore itself, "
            "not a PharmacophoreSearch or other wrapper."
        )

    points = list(ph)
    rwmol = Chem.RWMol()

    centre_idx_by_feature: dict[int, int] = {}
    tip_idxs_by_feature: dict[int, list[int]] = {}
    positions: list[tuple[float, float, float]] = []

    def _add_pseudo_atom(
        element: str,
        pdb_name: str,
        resname: str,
        resnum: int,
        position: tuple[float, float, float],
    ) -> int:
        atom = Chem.Atom(element)
        atom.SetNoImplicit(True)
        info = Chem.AtomPDBResidueInfo()
        info.SetName(pdb_name)
        info.SetResidueName(resname)
        info.SetResidueNumber(resnum)
        info.SetChainId("P")
        info.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(info)
        idx = rwmol.AddAtom(atom)
        positions.append(position)
        return idx

    for i, p in enumerate(points):
        resname = _TYPE_TO_PDB_RESNAME[p.type]
        centre_pos = (float(p.x), float(p.y), float(p.z))
        ci = _add_pseudo_atom(
            element=_TYPE_TO_ELEMENT[p.type],
            pdb_name=_pdb_atom_name(p.type, i),
            resname=resname,
            resnum=i + 1,
            position=centre_pos,
        )
        centre_idx_by_feature[i] = ci

        unit = _unit_normal_direction(p)
        if unit is None:
            continue
        ux, uy, uz = unit
        up_pos = (
            centre_pos[0] + ux * NORMAL_TIP_LENGTH,
            centre_pos[1] + uy * NORMAL_TIP_LENGTH,
            centre_pos[2] + uz * NORMAL_TIP_LENGTH,
        )
        dn_pos = (
            centre_pos[0] - ux * NORMAL_TIP_LENGTH,
            centre_pos[1] - uy * NORMAL_TIP_LENGTH,
            centre_pos[2] - uz * NORMAL_TIP_LENGTH,
        )
        ti_up = _add_pseudo_atom(
            element="H",
            pdb_name=_pdb_tip_atom_name(i, "+"),
            resname=resname,
            resnum=i + 1,
            position=up_pos,
        )
        ti_dn = _add_pseudo_atom(
            element="H",
            pdb_name=_pdb_tip_atom_name(i, "-"),
            resname=resname,
            resnum=i + 1,
            position=dn_pos,
        )
        tip_idxs_by_feature.setdefault(i, []).extend([ti_up, ti_dn])

    for fi, tips in tip_idxs_by_feature.items():
        ci = centre_idx_by_feature[fi]
        for ti in tips:
            rwmol.AddBond(ci, ti, Chem.BondType.SINGLE)

    conf = Chem.Conformer(rwmol.GetNumAtoms())
    for atom_idx, pos in enumerate(positions):
        conf.SetAtomPosition(atom_idx, pos)
    rwmol.AddConformer(conf, assignId=True)
    mol = rwmol.GetMol()

    mol.SetProp("_Name", name)
    mol.SetProp("kind", ph.kind)
    ph_name = getattr(ph, "_name", "") or ""
    if ph_name:
        mol.SetProp("name", ph_name)
    mol.SetIntProp("num_features", len(points))
    mol.SetProp("types", ",".join(p.type.value for p in points))
    mol.SetProp("sigmas", ",".join(f"{p.sigma:.4f}" for p in points))
    mol.SetProp(
        "centers",
        ";".join(f"{p.x:.4f},{p.y:.4f},{p.z:.4f}" for p in points),
    )
    normals_parts: list[str] = []
    for p in points:
        if p.has_normal:
            normals_parts.append(f"{p.nx:.4f},{p.ny:.4f},{p.nz:.4f}")
        else:
            normals_parts.append("-")
    mol.SetProp("normals", ";".join(normals_parts))
    return mol


def _set_hit_sdf_tags(mol: Any, index: int, hit: MatchResult) -> None:
    mol.SetProp("index", str(index))
    mol.SetProp("conf_id", str(hit.conf_id))
    mol.SetProp("tanimoto", f"{hit.tanimoto:.6f}")
    mol.SetProp("tversky_ref", f"{hit.tversky_ref:.6f}")
    mol.SetProp("tversky_db", f"{hit.tversky_db:.6f}")
    mol.SetProp("overlap_volume", f"{hit.overlap_volume:.6f}")
    mol.SetProp("excl_volume", f"{hit.excl_volume:.6f}")
    mol.SetProp("ref_volume", f"{hit.ref_volume:.6f}")
    mol.SetProp("db_volume", f"{hit.db_volume:.6f}")


def write_hits_sdf(
    hits: list[tuple[int, MatchResult]],
    path: str | Path,
    *,
    pharmacophore: Pharmacophore | None = None,
) -> int:
    """Write each hit's aligned molecule to a multi-record SDF file.

    SDF tags written per record: ``index``, ``conf_id``, ``tanimoto``,
    ``tversky_ref``, ``tversky_db``, ``overlap_volume``, ``excl_volume``,
    ``ref_volume`` and ``db_volume``. Hits with no aligned molecule (e.g. a
    pre-built molecule pharmacophore with no parent ``Chem.Mol``) are skipped
    with a one-line warning.

    When ``pharmacophore`` is given, it is written as the **first** record
    using :func:`pharmacophore_to_mol` (one pseudo-atom per feature). Since
    aligned hit molecules live in the query coordinate frame, passing the
    query pharmacophore here produces a single SDF that is geometrically
    consistent with the hits.

    Returns the total number of records written (including the pharmacophore
    record, if any).
    """
    Chem = _require_rdkit("write_hits_sdf")

    path = Path(path)
    written = 0
    with Chem.SDWriter(str(path)) as writer:
        if pharmacophore is not None:
            ph_mol = pharmacophore_to_mol(pharmacophore, name="pharmacophore")
            writer.write(ph_mol)
            written += 1
        for index, hit in hits:
            mol = hit.aligned_mol
            if mol is None:
                print(
                    f"write_hits_sdf: skipping index {index} (no aligned molecule)",
                    file=sys.stderr,
                )
                continue
            _set_hit_sdf_tags(mol, index, hit)
            writer.write(mol)
            written += 1
    return written


def _pdb_model_block(mol: Any, model_id: int, chem: Any) -> str:
    """Render ``mol`` as a single ``MODEL`` block (no trailing END)."""
    block = chem.MolToPDBBlock(mol)
    body_lines: list[str] = []
    for line in block.splitlines():
        if not line:
            continue
        record = line[:6].strip()
        if record in ("END", "MASTER"):
            continue
        body_lines.append(line)
    header = f"MODEL     {model_id:>4d}"
    return "\n".join([header, *body_lines, "ENDMDL"])


def write_hits_pdb(
    hits: list[tuple[int, MatchResult]],
    path: str | Path,
    *,
    pharmacophore: Pharmacophore | None = None,
) -> int:
    """Write each hit's aligned molecule to a multi-``MODEL`` PDB file.

    Each record is wrapped in a ``MODEL`` / ``ENDMDL`` block and the file is
    terminated with a single ``END``. Hits with no aligned molecule are
    skipped with a one-line warning.

    When ``pharmacophore`` is given, it is written as the **first** ``MODEL``
    using :func:`pharmacophore_to_mol`: each pharmacophore feature becomes
    one ``HETATM`` whose residue name encodes the feature type
    (``ARO``, ``LIP``, ``HDO``, ``HAC``, ...). Because aligned hit molecules
    live in the query coordinate frame, passing the query pharmacophore here
    produces a single PDB that is geometrically consistent with the hits.

    Returns the total number of ``MODEL``s written (including the
    pharmacophore model, if any).

    Requires RDKit (raises ``ImportError`` otherwise).
    """
    Chem = _require_rdkit("write_hits_pdb")

    path = Path(path)
    blocks: list[str] = []
    written = 0
    if pharmacophore is not None:
        ph_mol = pharmacophore_to_mol(pharmacophore, name="pharmacophore")
        written += 1
        blocks.append(_pdb_model_block(ph_mol, written, Chem))
    for index, hit in hits:
        mol = hit.aligned_mol
        if mol is None:
            print(
                f"write_hits_pdb: skipping index {index} (no aligned molecule)",
                file=sys.stderr,
            )
            continue
        written += 1
        blocks.append(_pdb_model_block(mol, written, Chem))

    text = "\n".join(blocks)
    if blocks:
        text += "\n"
    text += "END\n"
    path.write_text(text, encoding="utf-8")
    return written
