"""Human-readable reports and SDF output for pharmacophore screen results."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TextIO

from .pharmacophore import Pharmacophore
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


def write_hits_sdf(
    hits: list[tuple[int, MatchResult]],
    path: str | Path,
) -> int:
    """Write each hit's aligned molecule to a multi-record SDF file.

    SDF tags written per record: ``index``, ``conf_id``, ``tanimoto``,
    ``tversky_ref``, ``tversky_db``, ``overlap_volume``, ``excl_volume``,
    ``ref_volume`` and ``db_volume``. Hits with no aligned molecule (e.g. a
    pre-built molecule pharmacophore with no parent ``Chem.Mol``) are skipped
    with a one-line warning.

    Returns the number of records written.
    """
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError(
            "write_hits_sdf requires RDKit (pip install rdkit)"
        ) from exc

    path = Path(path)
    written = 0
    with Chem.SDWriter(str(path)) as writer:
        for index, hit in hits:
            mol = hit.aligned_mol
            if mol is None:
                print(
                    f"write_hits_sdf: skipping index {index} (no aligned molecule)",
                    file=sys.stderr,
                )
                continue
            mol.SetProp("index", str(index))
            mol.SetProp("conf_id", str(hit.conf_id))
            mol.SetProp("tanimoto", f"{hit.tanimoto:.6f}")
            mol.SetProp("tversky_ref", f"{hit.tversky_ref:.6f}")
            mol.SetProp("tversky_db", f"{hit.tversky_db:.6f}")
            mol.SetProp("overlap_volume", f"{hit.overlap_volume:.6f}")
            mol.SetProp("excl_volume", f"{hit.excl_volume:.6f}")
            mol.SetProp("ref_volume", f"{hit.ref_volume:.6f}")
            mol.SetProp("db_volume", f"{hit.db_volume:.6f}")
            writer.write(mol)
            written += 1
    return written
