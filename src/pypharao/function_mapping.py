"""Enumerate pharmacophore feature maps (`functionMapping.cpp`).

Matching is directional: the ``ref`` (query) point sets which database
(molecule) types are compatible. The full table — also reflected in
``functions_compatible`` — is::

    Query AROM        ↔ molecule AROM
    Query LIPO        ↔ molecule LIPO
    Query AROM|LIPO   ↔ molecule AROM or LIPO
    Query HDON        ↔ molecule HDON
    Query HACC        ↔ molecule HACC
    Query HACC&HDON   ↔ molecule HACC&HDON only
    Query HACC|HDON   ↔ molecule HDON or HACC
    Query POSC        ↔ molecule POSC
    Query NEGC        ↔ molecule NEGC
    Query EXCL        ↔ never (penalises overlap separately)
    Query UNDEF       ↔ any molecule type
"""

from __future__ import annotations

import math

from .constants import GCI, GCI2, PI
from .pharmacophore import Pharmacophore, PointType, distance

_AROM_OR_LIPO_DB = frozenset({PointType.AROM, PointType.LIPO})
_HACC_OR_HDON_DB = frozenset({PointType.HDON, PointType.HACC})
_ANY_MOL_TYPE = frozenset(
    {
        PointType.AROM,
        PointType.LIPO,
        PointType.HDON,
        PointType.HACC,
        PointType.HACC_AND_HDON,
        PointType.POSC,
        PointType.NEGC,
    }
)


def database_types_for_query(ref: PointType) -> frozenset[PointType]:
    """Molecule feature types that may satisfy a query point of type ``ref``."""
    if ref == PointType.EXCL:
        return frozenset()
    if ref == PointType.UNDEF:
        return _ANY_MOL_TYPE
    if ref == PointType.AROM_OR_LIPO:
        return _AROM_OR_LIPO_DB
    if ref == PointType.HACC_OR_HDON:
        return _HACC_OR_HDON_DB
    if ref == PointType.HACC_AND_HDON:
        return frozenset({PointType.HACC_AND_HDON})
    if ref in _ANY_MOL_TYPE:
        return frozenset({ref})
    return frozenset()


def functions_compatible(ref: PointType, db: PointType) -> bool:
    """Return whether a query feature ``ref`` may map to a database feature ``db``."""
    return db in database_types_for_query(ref)


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
            if ref[i].type == PointType.EXCL:
                continue
            for j in range(len(db)):
                if functions_compatible(ref[i].type, db[j].type):
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
            v1 = GCI * (PI / ref[self._ref_index[i]].sigma) ** 1.5
            v2 = GCI * (PI / db[self._db_index[i]].sigma) ** 1.5
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
                    * (PI / (ra.sigma + rb.sigma)) ** 1.5
                    * math.exp(-(ra.sigma * rb.sigma) * d1 / (ra.sigma + rb.sigma))
                )
                o2 = (
                    GCI2
                    * (PI / (rc.sigma + rd.sigma)) ** 1.5
                    * math.exp(-(rc.sigma * rd.sigma) * d1 / (rc.sigma + rd.sigma))
                )
                v3 = GCI * (PI / rc.sigma) ** 1.5
                v4 = GCI * (PI / rd.sigma) ** 1.5
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
