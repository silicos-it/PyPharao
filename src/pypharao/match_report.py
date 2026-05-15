"""Human-readable reports for pharmacophore screen results."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any, TextIO

from .pharmacophore import Pharmacophore
from .search import MatchResult


def _format_mapping(mapping: Sequence[tuple[int, int]]) -> str:
    if not mapping:
        return "—"
    return ", ".join(f"({r},{d})" for r, d in mapping)


def _pharmacophore_summary(ph: Pharmacophore) -> str:
    if not ph.points:
        return "0 points"
    funcs = ", ".join(p.func.value for p in ph.points)
    return f"{len(ph)} points [{funcs}]"


def _aligned_mol_label(mol: Any) -> str:
    if mol is None:
        return "—"
    try:
        return f"RDKit Mol ({mol.GetNumAtoms()} atoms)"
    except Exception:
        return "present"


def _scalar_rows(result: MatchResult) -> list[tuple[str, str]]:
    return [
        ("ref_volume", f"{result.ref_volume:.6f}"),
        ("db_volume", f"{result.db_volume:.6f}"),
        ("overlap_volume", f"{result.overlap_volume:.6f}"),
        ("excl_volume", f"{result.excl_volume:.6f}"),
        ("tanimoto", f"{result.tanimoto:.6f}"),
        ("tversky_ref", f"{result.tversky_ref:.6f}"),
        ("tversky_db", f"{result.tversky_db:.6f}"),
        ("mapping", _format_mapping(result.mapping)),
        ("database_pharmacophore", _pharmacophore_summary(result.database_pharmacophore)),
        ("matched_db_pharmacophore", _pharmacophore_summary(result.matched_db_pharmacophore)),
        ("aligned_mol", _aligned_mol_label(result.aligned_mol)),
    ]


def _print_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    file: TextIO,
) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(sep, file=file)
    print("  ".join("-" * widths[i] for i in range(len(headers))), file=file)
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))), file=file)


def print_match_results(
    results: MatchResult | list[tuple[int, MatchResult]] | None,
    *,
    file: TextIO | None = None,
) -> None:
    """Print a tabulated summary of one or more screen hits.

    Accepts a single ``MatchResult``, a batch ``[(index, MatchResult), ...]``
    from ``PharmacophoreSearch.screen()``, or ``None`` when there is no hit.
    """
    out = file if file is not None else sys.stdout

    if results is None:
        print("No match.", file=out)
        return

    if isinstance(results, MatchResult):
        label_w = max(len(label) for label, _ in _scalar_rows(results))
        print("MatchResult", file=out)
        for label, value in _scalar_rows(results):
            print(f"  {label:<{label_w}}  {value}", file=out)
        return

    if not results:
        print("No hits.", file=out)
        return

    headers = [
        "index",
        "tanimoto",
        "tversky_ref",
        "tversky_db",
        "overlap_volume",
        "excl_volume",
        "ref_volume",
        "db_volume",
        "n_pairs",
        "db_points",
        "matched_points",
        "aligned_mol",
    ]
    rows: list[list[str]] = []
    for index, hit in results:
        rows.append(
            [
                str(index),
                f"{hit.tanimoto:.6f}",
                f"{hit.tversky_ref:.6f}",
                f"{hit.tversky_db:.6f}",
                f"{hit.overlap_volume:.6f}",
                f"{hit.excl_volume:.6f}",
                f"{hit.ref_volume:.6f}",
                f"{hit.db_volume:.6f}",
                str(len(hit.mapping)),
                str(len(hit.database_pharmacophore)),
                str(len(hit.matched_db_pharmacophore)),
                _aligned_mol_label(hit.aligned_mol),
            ]
        )
    _print_table(headers, rows, file=out)
