"""Enumerate pharmacophore feature maps (`functionMapping.cpp`)."""

from __future__ import annotations

import math

from .constants import GCI, GCI2, PI
from .pharmacophore import FuncGroup, Pharmacophore, distance

# Database types matchable by hybrid query features (OR semantics).
_HYBL_DB = frozenset({FuncGroup.AROM, FuncGroup.LIPO, FuncGroup.HYBL})
_HYBH_DB = frozenset({FuncGroup.HDON, FuncGroup.HACC, FuncGroup.HYBH})


def database_types_for_query(ref: FuncGroup) -> frozenset[FuncGroup]:
    """Molecule feature types that may satisfy a query point of type ``ref``."""
    if ref in (FuncGroup.EXCL, FuncGroup.UNDEF):
        return frozenset()
    if ref == FuncGroup.HYBL:
        return _HYBL_DB
    if ref == FuncGroup.HYBH:
        return _HYBH_DB
    return frozenset({ref})


def functions_compatible(ref: FuncGroup, db: FuncGroup) -> bool:
    """Return whether a reference (query) feature may map to a database feature.

    Matching is directional: ``ref`` is the pharmacophore query, ``db`` is the
    molecule/database feature. Only the pairings below are allowed; there are no
    other cross-type matches (e.g. query AROM does not match database HYBL).

    * AROM, LIPO, HDON, HACC, POSC, NEGC — same type only
    * HYBL — database AROM, LIPO, or HYBL
    * HYBH — database HDON, HACC, or HYBH
    """
    if ref == db:
        return True
    if ref == FuncGroup.HYBL:
        return db in _HYBL_DB
    if ref == FuncGroup.HYBH:
        return db in _HYBH_DB
    return False


class FunctionMapping:
    def __init__(self, ref: Pharmacophore, db: Pharmacophore, epsilon: float):
        self._ref = ref
        self._db = db
        self._epsilon = epsilon
        self._ref_index: list[int] = []
        self._db_index: list[int] = []
        self._match_map: dict[int, list[list[int]]] = {}
        self._max_level = 0
        self._has_next = True

        for i in range(len(ref)):
            if ref[i].func == FuncGroup.EXCL:
                continue
            for j in range(len(db)):
                if functions_compatible(ref[i].func, db[j].func):
                    self._ref_index.append(i)
                    self._db_index.append(j)

        if not self._ref_index:
            self._has_next = False
            return

        n = len(self._ref_index)
        self._match_map[1] = [[i] for i in range(n)]
        self._max_level = min(len(ref), len(db))
        self._match_map[2] = []

        for i in range(n - 1):
            v1 = GCI * (PI / ref[self._ref_index[i]].alpha) ** 1.5
            v2 = GCI * (PI / db[self._db_index[i]].alpha) ** 1.5
            for j in range(i + 1, n):
                if self._ref_index[i] == self._ref_index[j] or self._db_index[i] == self._db_index[j]:
                    continue
                d1 = distance(ref[self._ref_index[i]], ref[self._ref_index[j]])
                d2 = distance(db[self._db_index[i]], db[self._db_index[j]])
                d1 = (d1 - d2) * (d1 - d2)
                ra, rb = ref[self._ref_index[i]], db[self._db_index[i]]
                rc, rd = ref[self._ref_index[j]], db[self._db_index[j]]
                o1 = (
                    GCI2
                    * (PI / (ra.alpha + rb.alpha)) ** 1.5
                    * math.exp(-(ra.alpha * rb.alpha) * d1 / (ra.alpha + rb.alpha))
                )
                o2 = (
                    GCI2
                    * (PI / (rc.alpha + rd.alpha)) ** 1.5
                    * math.exp(-(rc.alpha * rd.alpha) * d1 / (rc.alpha + rd.alpha))
                )
                v3 = GCI * (PI / rc.alpha) ** 1.5
                v4 = GCI * (PI / rd.alpha) ** 1.5
                if (o2 / (v3 + v4 - o2) > epsilon) or (o1 / (v1 + v2 - o1) > epsilon):
                    self._match_map[2].append([i, j])

        for level in range(3, self._max_level + 1):
            self._match_map[level] = []
            for vi in self._match_map[level - 1]:
                last_pair_index = vi[level - 2]
                for vj in self._match_map[2]:
                    if vj[0] < last_pair_index:
                        continue
                    if vj[0] > last_pair_index:
                        break
                    possible = True
                    for kk in range(level - 2):
                        found = False
                        for vk in self._match_map[2]:
                            if vk[0] < vi[kk]:
                                continue
                            if vk[0] > vi[kk]:
                                break
                            if vk[1] == vj[1]:
                                found = True
                                break
                        possible &= found
                        if not possible:
                            break
                    if possible:
                        v = list(vi[: level - 1]) + [vj[1]]
                        self._match_map[level].append(v)

    def get_next_map(self) -> list[tuple[int, int]]:
        while self._max_level >= 1 and not self._match_map.get(self._max_level, []):
            self._max_level -= 1
        if self._max_level < 1:
            return []
        vi = self._match_map[self._max_level][0]
        pairs: list[tuple[int, int]] = []
        for k in range(self._max_level):
            i1 = self._ref_index[vi[k]]
            i2 = self._db_index[vi[k]]
            pairs.append((i1, i2))
        del self._match_map[self._max_level][0]
        return pairs
