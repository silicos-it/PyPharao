"""Perception options (no RDKit import)."""

from __future__ import annotations

import sys
from typing import TextIO

from .pharmacophore import FUNC_DESCRIPTIONS, FuncGroup

_PERCEPTION_FLAG: dict[FuncGroup, str | None] = {
    FuncGroup.AROM: "arom",
    FuncGroup.HDON: "hdon",
    FuncGroup.HACC: "hacc",
    FuncGroup.LIPO: "lipo",
    FuncGroup.POSC: "posc",
    FuncGroup.NEGC: "negc",
    FuncGroup.HYBH: "hybh",
    FuncGroup.HYBL: "hybl",
    FuncGroup.EXCL: None,
    FuncGroup.UNDEF: None,
}


class PerceptionOptions:
    """Controls which pharmacophore feature types are detected from 3D molecules."""

    def __init__(
        self,
        *,
        arom: bool = True,
        hdon: bool = True,
        hacc: bool = True,
        lipo: bool = True,
        posc: bool = True,
        negc: bool = True,
        hybh: bool = True,
        hybl: bool = True,
    ) -> None:
        self.arom = arom
        self.hdon = hdon
        self.hacc = hacc
        self.lipo = lipo
        self.posc = posc
        self.negc = negc
        self.hybh = hybh
        self.hybl = hybl

    @staticmethod
    def _resolve_func(func: FuncGroup | str) -> FuncGroup:
        if isinstance(func, FuncGroup):
            return func
        return FuncGroup(func)

    def _field_for(self, func: FuncGroup) -> str:
        field = _PERCEPTION_FLAG.get(func)
        if field is None:
            raise ValueError(
                f"{func.value} is not configurable via PerceptionOptions "
                "(manual pharmacophores only)"
            )
        return field

    def set_enabled(self, func: FuncGroup | str, enabled: bool) -> None:
        """Enable or disable perception for a feature type."""
        setattr(self, self._field_for(self._resolve_func(func)), enabled)

    def enable(self, func: FuncGroup | str) -> None:
        """Turn on perception for a feature type."""
        self.set_enabled(func, True)

    def disable(self, func: FuncGroup | str) -> None:
        """Turn off perception for a feature type."""
        self.set_enabled(func, False)

    def is_enabled_for_perception(self, func: FuncGroup | str) -> bool | None:
        """Whether molecule perception will emit this feature type.

        Returns ``None`` for types not controlled by perception flags (``EXCL``, ``UNDEF``).
        Hybrid types require their prerequisite flags as well (see README).
        """
        func = self._resolve_func(func)
        field = _PERCEPTION_FLAG.get(func)
        if field is None:
            return None
        flag = getattr(self, field)
        if func == FuncGroup.HYBH:
            return flag and self.hdon and self.hacc
        if func == FuncGroup.HYBL:
            return flag and self.arom and self.lipo
        return flag

    def _perception_status_label(self, func: FuncGroup) -> str:
        enabled = self.is_enabled_for_perception(func)
        if enabled is None:
            return "—"
        if enabled:
            return "on"
        field = _PERCEPTION_FLAG[func]
        assert field is not None
        if getattr(self, field) and func in (FuncGroup.HYBH, FuncGroup.HYBL):
            if func == FuncGroup.HYBH:
                return "off (needs hdon, hacc)"
            return "off (needs arom, lipo)"
        return "off"

    def print_features(self, *, file: TextIO | None = None) -> None:
        """Print all pharmacophore function types, descriptions, and detection status."""
        out = file if file is not None else sys.stdout
        status_w = max(len(self._perception_status_label(f)) for f in FuncGroup)
        for func in FuncGroup:
            name = func.value
            status = self._perception_status_label(func)
            desc = FUNC_DESCRIPTIONS[func]
            print(f"{name:<5}  {status:<{status_w}}  {desc}", file=out)

    def __repr__(self) -> str:
        flags = ", ".join(
            f"{name}={getattr(self, name)!r}"
            for name in ("arom", "hdon", "hacc", "lipo", "posc", "negc", "hybh", "hybl")
        )
        return f"PerceptionOptions({flags})"
