"""Numerical constants matching Pharao `defines.h`."""

from __future__ import annotations

PI = 3.14159265
GCI = 2.828427125
GCI2 = 7.999999999
H_BOND_DIST = 1.8
H_RADIUS = 1.2
DENSITY = 2.0
ACC_RADIUS = 1.55
PROBE_RADIUS = 1.4
REF_LIPO = 9.87


def round_ob(x: float) -> int:
    """ROUND macro from Pharao defines."""
    return int(x + 0.5)
