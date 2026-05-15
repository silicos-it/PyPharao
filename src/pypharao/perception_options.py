"""Perception options (no RDKit import)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
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


@dataclass
class PerceptionOptions:
    arom: bool = True
    hdon: bool = True
    hacc: bool = True
    lipo: bool = True
    posc: bool = True
    negc: bool = True
    hybh: bool = True
    hybl: bool = True

    def is_enabled_for_perception(self, func: FuncGroup) -> bool | None:
        """Whether molecule perception will emit this feature type.

        Returns ``None`` for types not controlled by perception flags (``EXCL``, ``UNDEF``).
        Hybrid types require their prerequisite flags as well (see README).
        """
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
