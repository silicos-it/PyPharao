"""Pharmacophore perception (which feature types are emitted from 3D molecules)."""

from __future__ import annotations

import sys
from typing import ClassVar, Iterator, TextIO

from .pharmacophore import (
    MoleculePharmacophore,
    Pharmacophore,
    PointType,
    QueryPharmacophore,
    TYPE_DESCRIPTIONS,
)


class PharmacophorePerception:
    """Base class for perception flags.

    A perception object controls which :class:`PointType` values are emitted
    when a pharmacophore is built from a molecule. Subclasses fix the set of
    allowed types via :attr:`allowed_types`. ``EXCL`` and ``UNDEF`` are never
    auto-perceived; they are added manually.
    """

    allowed_types: ClassVar[frozenset[PointType]] = frozenset()
    pharmacophore_cls: ClassVar[type[Pharmacophore]] = Pharmacophore

    def __init__(self, **flags: bool) -> None:
        self._flags: dict[PointType, bool] = {t: True for t in self.allowed_types}
        for key, value in flags.items():
            self.set_enabled(key, bool(value))

    @classmethod
    def _resolve(cls, t: PointType | str) -> PointType:
        if isinstance(t, PointType):
            return t
        try:
            return PointType(t)
        except ValueError:
            return PointType(str(t).upper())

    def _check_allowed(self, t: PointType) -> None:
        if t not in self.allowed_types:
            allowed = sorted(x.value for x in self.allowed_types)
            raise ValueError(
                f"{t.value} is not a perceivable type for "
                f"{type(self).__name__} (allowed: {allowed})"
            )

    def is_enabled(self, t: PointType | str) -> bool:
        """Return ``True`` when perception will emit feature type ``t``."""
        tt = self._resolve(t)
        self._check_allowed(tt)
        return self._flags[tt]

    def set_enabled(self, t: PointType | str, enabled: bool) -> None:
        """Turn perception of feature type ``t`` on or off."""
        tt = self._resolve(t)
        self._check_allowed(tt)
        self._flags[tt] = bool(enabled)

    def enable(self, t: PointType | str) -> None:
        """Enable perception for feature type ``t``."""
        self.set_enabled(t, True)

    def disable(self, t: PointType | str) -> None:
        """Disable perception for feature type ``t``."""
        self.set_enabled(t, False)

    def add(self, t: PointType | str) -> None:
        """Alias for :meth:`enable`."""
        self.enable(t)

    def remove(self, t: PointType | str) -> None:
        """Alias for :meth:`disable`."""
        self.disable(t)

    def types_enabled(self) -> list[PointType]:
        """All currently enabled feature types, in :class:`PointType` order."""
        return [t for t in PointType if t in self._flags and self._flags[t]]

    def __iter__(self) -> Iterator[PointType]:
        """Iterate over the allowed feature types (in declaration order)."""
        return iter(t for t in PointType if t in self.allowed_types)

    def print_features(self, *, file: TextIO | None = None) -> None:
        """Print one line per allowed type: ``TYPE  status  description``."""
        out = file if file is not None else sys.stdout
        for t in self:
            status = "on" if self._flags[t] else "off"
            print(f"{t.value:<10}  {status:<3}  {TYPE_DESCRIPTIONS[t]}", file=out)

    def __repr__(self) -> str:
        flags = ", ".join(f"{t.value}={self._flags[t]}" for t in self)
        return f"{type(self).__name__}({flags})"


class QueryPharmacophorePerception(PharmacophorePerception):
    """Perception flags for :class:`QueryPharmacophore`.

    Auto-perception emits ``AROM, LIPO, HDON, HACC, HACC_AND_HDON, POSC, NEGC``.
    The query-only ``AROM_OR_LIPO`` and ``HACC_OR_HDON`` compound types — along
    with ``EXCL`` and ``UNDEF`` — are never produced by auto-perception; they
    are introduced by hand by editing the resulting :class:`QueryPharmacophore`
    (typically by converting an ``AROM`` or ``LIPO`` point to ``AROM_OR_LIPO``
    and an ``HDON``/``HACC`` pair to ``HACC_OR_HDON``).
    """

    allowed_types: ClassVar[frozenset[PointType]] = frozenset(
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
    pharmacophore_cls: ClassVar[type[Pharmacophore]] = QueryPharmacophore


class MoleculePharmacophorePerception(PharmacophorePerception):
    """Perception flags for :class:`MoleculePharmacophore`.

    Auto-perception can emit ``AROM, LIPO, HDON, HACC, HACC_AND_HDON, POSC, NEGC``.
    """

    allowed_types: ClassVar[frozenset[PointType]] = frozenset(
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
    pharmacophore_cls: ClassVar[type[Pharmacophore]] = MoleculePharmacophore
