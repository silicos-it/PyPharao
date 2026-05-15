"""Pharmacophore model and JSON / Pharao `.phar` I/O."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator


class FuncGroup(str, Enum):
    AROM = "AROM"
    HDON = "HDON"
    HACC = "HACC"
    LIPO = "LIPO"
    POSC = "POSC"
    NEGC = "NEGC"
    HYBH = "HYBH"
    HYBL = "HYBL"
    EXCL = "EXCL"
    UNDEF = "UNDEF"


FUNC_DESCRIPTIONS: dict[FuncGroup, str] = {
    FuncGroup.AROM: "Aromatic ring centroids with plane normals",
    FuncGroup.HDON: "H-bond donors (N/O with H, not negatively charged)",
    FuncGroup.HACC: "H-bond acceptors (N/O, with Pharao-style filters)",
    FuncGroup.LIPO: "Lipophilic regions (from molecular surface)",
    FuncGroup.POSC: "Positively charged atoms",
    FuncGroup.NEGC: "Negatively charged atoms",
    FuncGroup.HYBH: "Hybrid H-bond features (merged donor/acceptor pairs)",
    FuncGroup.HYBL: "Hybrid lipophilic/aromatic features (merged arom/lipo sites)",
    FuncGroup.EXCL: "Exclusion sphere (query only; penalizes overlap in search)",
    FuncGroup.UNDEF: "Undefined placeholder (manual pharmacophores only)",
}


FUNC_HAS_NORMAL: dict[FuncGroup, bool] = {
    FuncGroup.AROM: True,
    FuncGroup.HDON: True,
    FuncGroup.HACC: True,
    FuncGroup.LIPO: False,
    FuncGroup.POSC: False,
    FuncGroup.NEGC: False,
    FuncGroup.HYBH: True,
    FuncGroup.HYBL: True,
    FuncGroup.EXCL: False,
    FuncGroup.UNDEF: False,
}

FUNC_SIGMA: dict[FuncGroup, float] = {
    FuncGroup.AROM: 0.7,
    FuncGroup.HDON: 1.0,
    FuncGroup.HACC: 1.0,
    FuncGroup.LIPO: 0.7,
    FuncGroup.POSC: 1.0,
    FuncGroup.NEGC: 1.0,
    FuncGroup.HYBH: 1.0,
    FuncGroup.HYBL: 0.7,
    FuncGroup.EXCL: 1.6,
    FuncGroup.UNDEF: 1.0,
}


def default_alpha(func: FuncGroup) -> float:
    return FUNC_SIGMA.get(func, 1.0)


@dataclass(frozen=True)
class PharmacophorePoint:
    """Pharmacophore feature. (nx,ny,nz) = absolute coordinates of normal tip (Pharao convention)."""

    x: float
    y: float
    z: float
    func: FuncGroup
    alpha: float
    has_normal: bool
    nx: float = 0.0
    ny: float = 0.0
    nz: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.func, FuncGroup):
            object.__setattr__(self, "func", FuncGroup(self.func))

    def with_fields(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        alpha: float | None = None,
        has_normal: bool | None = None,
        nx: float | None = None,
        ny: float | None = None,
        nz: float | None = None,
    ) -> PharmacophorePoint:
        return PharmacophorePoint(
            x=self.x if x is None else x,
            y=self.y if y is None else y,
            z=self.z if z is None else z,
            func=self.func,
            alpha=self.alpha if alpha is None else alpha,
            has_normal=self.has_normal if has_normal is None else has_normal,
            nx=self.nx if nx is None else nx,
            ny=self.ny if ny is None else ny,
            nz=self.nz if nz is None else nz,
        )


_JSON_VERSION = 1


@dataclass
class Pharmacophore:
    name: str = ""
    points: list[PharmacophorePoint] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self) -> Iterator[PharmacophorePoint]:
        return iter(self.points)

    def __getitem__(self, idx: int) -> PharmacophorePoint:
        return self.points[idx]

    def append_point(self, p: PharmacophorePoint) -> None:
        self.points.append(p)

    def remove_at(self, idx: int) -> None:
        del self.points[idx]

    def clear(self) -> None:
        self.points.clear()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "version": _JSON_VERSION,
            "name": self.name,
            "points": [
                {
                    "func": p.func.value,
                    "x": p.x,
                    "y": p.y,
                    "z": p.z,
                    "alpha": p.alpha,
                    "has_normal": p.has_normal,
                    "nx": p.nx,
                    "ny": p.ny,
                    "nz": p.nz,
                }
                for p in self.points
            ],
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_json_dict(), **kwargs)

    def write_json(self, path: str | Path, **kwargs: Any) -> None:
        Path(path).write_text(self.to_json(**kwargs), encoding="utf-8")

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> Pharmacophore:
        pts: list[PharmacophorePoint] = []
        for item in d.get("points", []):
            func = FuncGroup(item["func"])
            pts.append(
                PharmacophorePoint(
                    float(item["x"]),
                    float(item["y"]),
                    float(item["z"]),
                    func,
                    float(item.get("alpha", default_alpha(func))),
                    bool(item.get("has_normal", FUNC_HAS_NORMAL.get(func, False))),
                    float(item.get("nx", 0.0)),
                    float(item.get("ny", 0.0)),
                    float(item.get("nz", 0.0)),
                )
            )
        return cls(name=str(d.get("name", "")), points=pts)

    @classmethod
    def from_json(cls, s: str) -> Pharmacophore:
        return cls.from_json_dict(json.loads(s))

    @classmethod
    def from_json_file(cls, path: str | Path) -> Pharmacophore:
        return cls.from_json_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_phar_text(self) -> str:
        lines = [self.name or ""]
        for p in self.points:
            nflag = "1" if p.has_normal else "0"
            lines.append(
                f"{p.func.value}\t{p.x}\t{p.y}\t{p.z}\t{p.alpha}\t{nflag}\t"
                f"{p.nx}\t{p.ny}\t{p.nz}"
            )
        lines.append("$$$$")
        return "\n".join(lines) + "\n"

    def write_phar(self, path: str | Path) -> None:
        Path(path).write_text(self.to_phar_text(), encoding="utf-8")

    @classmethod
    def from_phar_text(cls, text: str) -> Pharmacophore:
        lines = [ln.strip() for ln in text.splitlines() if not ln.startswith("#")]
        if not lines:
            return cls()
        name = lines[0]
        points: list[PharmacophorePoint] = []
        ws = re.compile(r"[\t ]+")
        for line in lines[1:]:
            if line == "$$$$":
                break
            parts = ws.split(line)
            if len(parts) < 9:
                continue
            fg = FuncGroup(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            alpha = float(parts[4])
            has_normal = parts[5] == "1"
            nx, ny, nz = float(parts[6]), float(parts[7]), float(parts[8])
            points.append(PharmacophorePoint(x, y, z, fg, alpha, has_normal, nx, ny, nz))
        return cls(name=name, points=points)

    @classmethod
    def read_phar(cls, path: str | Path) -> Pharmacophore:
        return cls.from_phar_text(Path(path).read_text(encoding="utf-8"))

    def copy(self) -> Pharmacophore:
        return Pharmacophore(name=self.name, points=list(self.points))


def distance(p: PharmacophorePoint, q: PharmacophorePoint) -> float:
    dx = p.x - q.x
    dy = p.y - q.y
    dz = p.z - q.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def cosine_normals(p: PharmacophorePoint, q: PharmacophorePoint) -> float:
    v1 = (p.nx - p.x, p.ny - p.y, p.nz - p.z)
    v2 = (q.nx - q.x, q.ny - q.y, q.nz - q.z)
    n1 = math.sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2])
    n2 = math.sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2])
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (n1 * n2)
