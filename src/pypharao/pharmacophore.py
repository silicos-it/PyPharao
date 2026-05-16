"""Pharmacophore model and JSON / Pharao `.phar` I/O."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Iterator


class PointType(str, Enum):
    """Pharmacophore feature type (the ``Type`` of a PharmacophorePoint).

    The compound query types use underscored ``_OR_`` / ``_AND_`` spellings
    (``"AROM_OR_LIPO"``, ``"HACC_OR_HDON"``, ``"HACC_AND_HDON"``) instead of
    the original Pharao ``|`` / ``&`` notation, both for the Python identifier
    and for the underlying string ``.value``.
    """

    AROM = "AROM"
    LIPO = "LIPO"
    AROM_OR_LIPO = "AROM_OR_LIPO"
    HDON = "HDON"
    HACC = "HACC"
    HACC_AND_HDON = "HACC_AND_HDON"
    HACC_OR_HDON = "HACC_OR_HDON"
    POSC = "POSC"
    NEGC = "NEGC"
    EXCL = "EXCL"
    UNDEF = "UNDEF"


TYPE_DESCRIPTIONS: dict[PointType, str] = {
    PointType.AROM: "Aromatic ring centroids with plane normals",
    PointType.LIPO: "Lipophilic regions (from molecular surface) but no aromatics",
    PointType.AROM_OR_LIPO: "Either an AROM or a LIPO group, or both (query only)",
    PointType.HDON: "H-bond donors (N/O with H, not negatively charged)",
    PointType.HACC: "H-bond acceptors (N/O, with Pharao-style filters)",
    PointType.HACC_AND_HDON: "Both a HACC and a HDON",
    PointType.HACC_OR_HDON: "Either a HACC or a HDON group, or both (query only)",
    PointType.POSC: "Positively charged atoms",
    PointType.NEGC: "Negatively charged atoms",
    PointType.EXCL: "Exclusion sphere (query only; penalizes overlap in search)",
    PointType.UNDEF: "Undefined placeholder (manual pharmacophores only)",
}


TYPE_HAS_NORMAL: dict[PointType, bool] = {
    PointType.AROM: True,
    PointType.LIPO: False,
    PointType.AROM_OR_LIPO: True,
    PointType.HDON: False,
    PointType.HACC: False,
    PointType.HACC_AND_HDON: False,
    PointType.HACC_OR_HDON: False,
    PointType.POSC: False,
    PointType.NEGC: False,
    PointType.EXCL: False,
    PointType.UNDEF: False,
}


DEFAULT_SIGMA: dict[PointType, float] = {
    PointType.AROM: 0.7,
    PointType.LIPO: 0.7,
    PointType.AROM_OR_LIPO: 0.7,
    PointType.HDON: 1.0,
    PointType.HACC: 1.0,
    PointType.HACC_AND_HDON: 1.0,
    PointType.HACC_OR_HDON: 1.0,
    PointType.POSC: 1.0,
    PointType.NEGC: 1.0,
    PointType.EXCL: 1.6,
    PointType.UNDEF: 1.0,
}


def _resolve_type(t: PointType | str) -> PointType:
    if isinstance(t, PointType):
        return t
    return PointType(t)


@dataclass(frozen=True)
class PharmacophorePoint:
    """One pharmacophore feature (a Gaussian site).

    Public attributes
    -----------------
    type   : ``PointType``                — feature type.
    center : ``tuple[float, float, float]`` — ``(x, y, z)`` location in ångström.
    sigma  : ``float``                    — Gaussian width in ångström.
    normal : ``tuple[float, float, float] | None`` — absolute tip ``(nx, ny, nz)``
            of the feature normal, or ``None`` for types without a normal
            (see :data:`TYPE_HAS_NORMAL`).

    The constructor accepts ``type=``, ``center=(x, y, z)``, ``sigma=`` and
    ``normal=(nx, ny, nz)`` (``None`` is fine for types without a normal). Points
    are immutable; use :meth:`replace` to derive a modified copy.
    """

    type: PointType
    x: float
    y: float
    z: float
    sigma: float
    nx: float = 0.0
    ny: float = 0.0
    nz: float = 0.0

    Type: ClassVar[type[PointType]] = PointType

    def __init__(
        self,
        type: PointType | str,
        center: tuple[float, float, float],
        sigma: float | None = None,
        normal: tuple[float, float, float] | None = None,
    ) -> None:
        t = _resolve_type(type)
        if sigma is None:
            sigma = DEFAULT_SIGMA[t]
        if len(center) != 3:
            raise ValueError("center must be a 3-tuple (x, y, z)")
        x, y, z = float(center[0]), float(center[1]), float(center[2])
        wants_normal = TYPE_HAS_NORMAL[t]
        if normal is None:
            if wants_normal:
                raise ValueError(
                    f"PharmacophorePoint of type {t.value} requires a normal "
                    f"(absolute tip coordinates)"
                )
            nx, ny, nz = 0.0, 0.0, 0.0
        else:
            if not wants_normal:
                raise ValueError(
                    f"PharmacophorePoint of type {t.value} does not carry a normal"
                )
            if len(normal) != 3:
                raise ValueError("normal must be a 3-tuple (nx, ny, nz)")
            nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])

        object.__setattr__(self, "type", t)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "sigma", float(sigma))
        object.__setattr__(self, "nx", nx)
        object.__setattr__(self, "ny", ny)
        object.__setattr__(self, "nz", nz)

    @property
    def center(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def normal(self) -> tuple[float, float, float] | None:
        if not TYPE_HAS_NORMAL[self.type]:
            return None
        return (self.nx, self.ny, self.nz)

    @property
    def has_normal(self) -> bool:
        return TYPE_HAS_NORMAL[self.type]

    def replace(
        self,
        *,
        type: PointType | str | None = None,
        center: tuple[float, float, float] | None = None,
        sigma: float | None = None,
        normal: tuple[float, float, float] | None = None,
    ) -> PharmacophorePoint:
        """Return a new point with selected fields updated."""
        new_type = _resolve_type(type) if type is not None else self.type
        new_center = center if center is not None else (self.x, self.y, self.z)
        new_sigma = sigma if sigma is not None else self.sigma
        if normal is not None:
            new_normal: tuple[float, float, float] | None = normal
        elif TYPE_HAS_NORMAL[new_type]:
            new_normal = (self.nx, self.ny, self.nz) if self.has_normal else None
        else:
            new_normal = None
        return PharmacophorePoint(
            type=new_type, center=new_center, sigma=new_sigma, normal=new_normal
        )


_JSON_VERSION = 2


class Pharmacophore:
    """Ordered collection of :class:`PharmacophorePoint`.

    Subclasses (:class:`QueryPharmacophore`, :class:`MoleculePharmacophore`)
    declare an ``allowed_types`` set and reject points outside it. Use
    :meth:`add_point` / :meth:`remove_point` / :meth:`clear` to edit a pharmacophore.
    """

    allowed_types: ClassVar[frozenset[PointType]] = frozenset()
    kind: ClassVar[str] = "pharmacophore"

    def __init__(self, points: list[PharmacophorePoint] | None = None) -> None:
        self._points: list[PharmacophorePoint] = []
        if points:
            for p in points:
                self.add_point(p)

    @property
    def points(self) -> list[PharmacophorePoint]:
        """Live list of points (read-only view).

        Use :meth:`add_point` / :meth:`remove_point` / :meth:`set_point` /
        :meth:`clear` to mutate the collection so validation is enforced.
        """
        return list(self._points)

    def __len__(self) -> int:
        return len(self._points)

    def __iter__(self) -> Iterator[PharmacophorePoint]:
        return iter(self._points)

    def __getitem__(self, idx: int) -> PharmacophorePoint:
        return self._points[idx]

    def _check_type(self, p: PharmacophorePoint) -> None:
        if p.type not in self.allowed_types:
            raise ValueError(
                f"{p.type.value} is not allowed in {type(self).__name__} "
                f"(allowed: {sorted(t.value for t in self.allowed_types)})"
            )

    def add_point(self, point: PharmacophorePoint) -> None:
        """Append a pharmacophore point after validating its type."""
        self._check_type(point)
        self._points.append(point)

    def set_point(self, idx: int, point: PharmacophorePoint) -> None:
        """Replace the point at ``idx`` (validated against ``allowed_types``)."""
        self._check_type(point)
        self._points[idx] = point

    def update_point(self, idx: int, **changes: Any) -> None:
        """Replace the point at ``idx`` with a copy whose selected fields are updated.

        Convenience wrapper for
        ``self.set_point(idx, self[idx].replace(**changes))``. Accepts the same
        keyword arguments as :meth:`PharmacophorePoint.replace` (``type``,
        ``center``, ``sigma``, ``normal``). Validation against the subclass's
        ``allowed_types`` still applies.
        """
        self.set_point(idx, self._points[idx].replace(**changes))

    def remove_point(self, idx: int) -> None:
        """Remove the point at ``idx``."""
        del self._points[idx]

    def clear(self) -> None:
        """Remove every point."""
        self._points.clear()

    def copy(self) -> Pharmacophore:
        """Shallow copy keeping the same subclass."""
        clone = type(self).__new__(type(self))
        clone._points = list(self._points)
        if hasattr(self, "_name"):
            clone._name = self._name
        return clone

    def to_json_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "version": _JSON_VERSION,
            "kind": self.kind,
            "points": [
                {
                    "type": p.type.value,
                    "x": p.x,
                    "y": p.y,
                    "z": p.z,
                    "sigma": p.sigma,
                    "nx": p.nx,
                    "ny": p.ny,
                    "nz": p.nz,
                }
                for p in self._points
            ],
        }
        if hasattr(self, "_name"):
            d["name"] = self._name
        return d

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_json_dict(), **kwargs)

    def write_json(self, path: str | Path, **kwargs: Any) -> None:
        Path(path).write_text(self.to_json(**kwargs), encoding="utf-8")

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> Pharmacophore:
        kind = d.get("kind", "query")
        target_cls = cls
        if cls is Pharmacophore:
            target_cls = _KIND_TO_CLASS.get(kind, QueryPharmacophore)
        ph = target_cls()
        if isinstance(ph, QueryPharmacophore):
            ph.set_name(str(d.get("name", "")))
        for item in d.get("points", []):
            t = PointType(item["type"])
            normal: tuple[float, float, float] | None
            if TYPE_HAS_NORMAL[t]:
                normal = (
                    float(item.get("nx", 0.0)),
                    float(item.get("ny", 0.0)),
                    float(item.get("nz", 0.0)),
                )
            else:
                normal = None
            ph.add_point(
                PharmacophorePoint(
                    type=t,
                    center=(float(item["x"]), float(item["y"]), float(item["z"])),
                    sigma=float(item.get("sigma", DEFAULT_SIGMA[t])),
                    normal=normal,
                )
            )
        return ph

    @classmethod
    def from_json(cls, s: str) -> Pharmacophore:
        return cls.from_json_dict(json.loads(s))

    @classmethod
    def from_json_file(cls, path: str | Path) -> Pharmacophore:
        return cls.from_json_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_phar_text(self) -> str:
        header = getattr(self, "_name", "") or ""
        lines = [header]
        for p in self._points:
            nflag = "1" if p.has_normal else "0"
            lines.append(
                f"{p.type.value}\t{p.x}\t{p.y}\t{p.z}\t{p.sigma}\t{nflag}\t"
                f"{p.nx}\t{p.ny}\t{p.nz}"
            )
        lines.append("$$$$")
        return "\n".join(lines) + "\n"

    def write_phar(self, path: str | Path) -> None:
        Path(path).write_text(self.to_phar_text(), encoding="utf-8")

    def _resolve_mol_name(self, name: str | None) -> str:
        if name is not None:
            return name
        existing = getattr(self, "_name", "") or ""
        return existing or "pharmacophore"

    def write_sdf(self, path: str | Path, *, name: str | None = None) -> None:
        """Write this pharmacophore to a single-record SDF file.

        Each pharmacophore feature is rendered as one pseudo-atom via
        :func:`pypharao.pharmacophore_to_mol`. The SDF carries the same
        per-feature properties (``types``, ``sigmas``, ``centers``,
        ``normals``, ``num_features``, ``kind`` and, for named query
        pharmacophores, ``name``) so the file is self-describing.

        Pass ``name=`` to override the record title; the default is the
        pharmacophore's own ``name`` (for query pharmacophores) or
        ``"pharmacophore"``.

        Requires RDKit.
        """
        from .match_report import pharmacophore_to_mol

        try:
            from rdkit import Chem
        except ImportError as exc:
            raise ImportError(
                "Pharmacophore.write_sdf requires RDKit (pip install rdkit)"
            ) from exc

        mol = pharmacophore_to_mol(self, name=self._resolve_mol_name(name))
        with Chem.SDWriter(str(Path(path))) as writer:
            writer.write(mol)

    def write_pdb(self, path: str | Path, *, name: str | None = None) -> None:
        """Write this pharmacophore to a PDB file.

        Each pharmacophore feature becomes one ``HETATM`` whose residue
        name encodes the feature type (``ARO``, ``LIP``, ``HDO``,
        ``HAC``, ``DAC``, ...) — see
        :func:`pypharao.pharmacophore_to_mol` for the full mapping.

        Pass ``name=`` to override the ``COMPND`` title; the default is
        the pharmacophore's own ``name`` (for query pharmacophores) or
        ``"pharmacophore"``.

        Requires RDKit.
        """
        from .match_report import pharmacophore_to_mol

        try:
            from rdkit import Chem
        except ImportError as exc:
            raise ImportError(
                "Pharmacophore.write_pdb requires RDKit (pip install rdkit)"
            ) from exc

        mol = pharmacophore_to_mol(self, name=self._resolve_mol_name(name))
        Path(path).write_text(Chem.MolToPDBBlock(mol), encoding="utf-8")

    @classmethod
    def from_phar_text(cls, text: str) -> Pharmacophore:
        lines = [ln.strip() for ln in text.splitlines() if not ln.startswith("#")]
        target_cls = cls if cls is not Pharmacophore else QueryPharmacophore
        ph = target_cls()
        if not lines:
            return ph
        name = lines[0]
        if isinstance(ph, QueryPharmacophore):
            ph.set_name(name)
        ws = re.compile(r"[\t ]+")
        for line in lines[1:]:
            if line == "$$$$":
                break
            parts = ws.split(line)
            if len(parts) < 9:
                continue
            t = PointType(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            sigma = float(parts[4])
            has_normal = parts[5] == "1"
            nx, ny, nz = float(parts[6]), float(parts[7]), float(parts[8])
            normal: tuple[float, float, float] | None
            if TYPE_HAS_NORMAL[t]:
                normal = (nx, ny, nz) if has_normal else (x, y, z + 1.0)
            else:
                normal = None
            ph.add_point(
                PharmacophorePoint(type=t, center=(x, y, z), sigma=sigma, normal=normal)
            )
        return ph

    @classmethod
    def read_phar(cls, path: str | Path) -> Pharmacophore:
        return cls.from_phar_text(Path(path).read_text(encoding="utf-8"))


class QueryPharmacophore(Pharmacophore):
    """Pharmacophore used as a search query.

    Allowed types: every :class:`PointType` (including ``EXCL`` and ``UNDEF``).
    Carries a ``name`` accessed through :meth:`get_name` / :meth:`set_name`.
    """

    allowed_types: ClassVar[frozenset[PointType]] = frozenset(PointType)
    kind: ClassVar[str] = "query"

    def __init__(
        self,
        points: list[PharmacophorePoint] | None = None,
        name: str = "",
    ) -> None:
        super().__init__(points)
        self._name: str = str(name)

    def get_name(self) -> str:
        """Return the query name."""
        return self._name

    def set_name(self, name: str) -> None:
        """Set the query name."""
        self._name = str(name)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = str(value)


class MoleculePharmacophore(Pharmacophore):
    """Pharmacophore perceived from a database molecule.

    Allowed types: ``AROM, LIPO, HDON, HACC, HACC_AND_HDON, POSC, NEGC``.
    ``EXCL`` and ``UNDEF`` are query-only and rejected here; ``AROM_OR_LIPO``
    and ``HACC_OR_HDON`` are query-side OR features and not allowed in a
    molecule pharmacophore.
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
    kind: ClassVar[str] = "molecule"


_KIND_TO_CLASS: dict[str, type[Pharmacophore]] = {
    "query": QueryPharmacophore,
    "molecule": MoleculePharmacophore,
}


def distance(p: PharmacophorePoint, q: PharmacophorePoint) -> float:
    """Euclidean distance between two point centres."""
    dx = p.x - q.x
    dy = p.y - q.y
    dz = p.z - q.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def cosine_normals(p: PharmacophorePoint, q: PharmacophorePoint) -> float:
    """Cosine between the (relative) normal vectors of ``p`` and ``q``.

    Returns ``0.0`` when either point has no normal or a degenerate length.
    """
    if not (p.has_normal and q.has_normal):
        return 0.0
    v1 = (p.nx - p.x, p.ny - p.y, p.nz - p.z)
    v2 = (q.nx - q.x, q.ny - q.y, q.nz - q.z)
    n1 = math.sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2])
    n2 = math.sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2])
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (n1 * n2)
