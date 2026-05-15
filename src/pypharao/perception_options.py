"""Perception options (no RDKit import)."""

from __future__ import annotations

from dataclasses import dataclass


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
