"""Human-readable reports for pharmacophore screen results."""

from __future__ import annotations

import sys
from collections.abc import Sequence
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


def _format_mapping(mapping: Sequence[tuple[int, int]]) -> str:
    if not mapping:
        return "—"
    return ",".join(f"({r},{d})" for r, d in mapping)


def _pharmacophore_summary(ph: Pharmacophore) -> str:
    if not ph.points:
        return "[]"
    funcs = ",".join(p.func.value for p in ph.points)
    return f"[{funcs}]"


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
    """Sort screen hits by a numeric ``MatchResult`` field.

    ``matches`` is the ``[(index, MatchResult), ...]`` list from
    ``PharmacophoreSearch.screen()``. Returns a new list; the input is unchanged.
    """
    reverse = sort == "descending"
    return sorted(matches, key=lambda item: getattr(item[1], key), reverse=reverse)


def _normalize_hits(
    results: MatchResult | list[tuple[int, MatchResult]],
) -> list[tuple[int, MatchResult]]:
    if isinstance(results, MatchResult):
        return [(0, results)]
    return list(results)


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
    results: MatchResult | list[tuple[int, MatchResult]] | None,
    *,
    limit: int | None = None,
    file: TextIO | None = None,
) -> None:
    """Print a tab-separated summary of one or more screen hits.

    The first row lists column headers; each following row is one molecule.
    Accepts a single ``MatchResult``, a batch ``[(index, MatchResult), ...]``
    from ``PharmacophoreSearch.screen()``, or ``None`` when there is no hit.

    When ``limit`` is set, at most that many hit rows are printed (after any
    sorting applied by the caller).
    """
    out = file if file is not None else sys.stdout

    if limit is not None and limit < 0:
        raise ValueError("limit must be >= 0")

    if results is None:
        print("No match.", file=out)
        return

    hits = _normalize_hits(results)
    if not hits:
        print("No hits.", file=out)
        return

    if limit is not None:
        total = len(hits)
        hits = hits[:limit]
        if total > limit:
            print(f"Showing {limit} of {total} hits.", file=out)

    rows = [_result_row(index, hit) for index, hit in hits]
    _print_table(_TABLE_HEADERS, rows, file=out)
