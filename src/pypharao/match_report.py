"""Human-readable reports and SDF / PDB output for pharmacophore screen results."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TextIO

from .pharmacophore import Pharmacophore, PointType
from .search import MatchResult

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


def pharmacophore_to_mol(
    ph: Pharmacophore,
    *,
    name: str = "pharmacophore",
) -> Any:
    """Build a ``Chem.Mol`` representation of a pharmacophore.

    Each pharmacophore point becomes a single, disconnected pseudo-atom whose
    element loosely encodes the feature type (AROM/LIPO → C, HDON → N,
    HACC → O, HACC_AND_HDON / HACC_OR_HDON → S, POSC → Na, NEGC → Cl,
    EXCL → F, UNDEF → He). The atom carries PDB residue info so PDB output
    uses the type code as the residue name (e.g. ``ARO``, ``HDO``, ``HAC``)
    and SDF / mol-block writers preserve the 3D coordinates.

    The molecule is **not** sanitised (these are pseudo-atoms with
    non-physical valences) but is fully usable with ``Chem.SDWriter``,
    ``Chem.MolToPDBBlock`` and the like. The return type is ``Chem.Mol``
    even though it's typed ``Any`` because RDKit is an optional dependency.

    Per-feature SDF properties on the returned molecule:

    - ``num_features`` (int)
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
    conf = Chem.Conformer(len(points))
    for i, p in enumerate(points):
        atom = Chem.Atom(_TYPE_TO_ELEMENT[p.type])
        atom.SetNoImplicit(True)
        info = Chem.AtomPDBResidueInfo()
        info.SetName(_pdb_atom_name(p.type, i))
        info.SetResidueName(_TYPE_TO_PDB_RESNAME[p.type])
        info.SetResidueNumber(i + 1)
        info.SetChainId("P")
        info.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(info)
        rwmol.AddAtom(atom)
        conf.SetAtomPosition(i, (float(p.x), float(p.y), float(p.z)))
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
