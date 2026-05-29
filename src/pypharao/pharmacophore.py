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

# Decimal places for floats written to `.phar` and JSON (centres, sigma, normals).
_IO_FLOAT_DECIMAL_PLACES = 5

# Column layout for `.phar` body lines (space-separated fields; see :meth:`Pharmacophore.to_phar_text`).
_PHAR_TYPE_FIELD_WIDTH = max(len(t.value) for t in PointType)
_PHAR_FLOAT_FIELD_WIDTH = 13


def _quantize_io_float(x: float) -> float:
    return round(float(x), _IO_FLOAT_DECIMAL_PLACES)


def _phar_float_field(x: float) -> str:
    return (
        f"{_quantize_io_float(x):>{_PHAR_FLOAT_FIELD_WIDTH}.{_IO_FLOAT_DECIMAL_PLACES}f}"
    )


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
                    "x": _quantize_io_float(p.x),
                    "y": _quantize_io_float(p.y),
                    "z": _quantize_io_float(p.z),
                    "sigma": _quantize_io_float(p.sigma),
                    "nx": _quantize_io_float(p.nx),
                    "ny": _quantize_io_float(p.ny),
                    "nz": _quantize_io_float(p.nz),
                }
                for p in self._points
            ],
        }
        if hasattr(self, "_name"):
            d["name"] = self._name
        return d

    def to_json(self, **kwargs: Any) -> str:
        """Serialise to JSON. Passes keyword arguments to :func:`json.dumps`.

        By default ``indent=2`` is set so output is multi-line and readable;
        pass ``indent=None`` for a single-line dump.
        """
        kwargs.setdefault("indent", 2)
        return json.dumps(self.to_json_dict(), **kwargs)

    def write_json(self, path: str | Path, **kwargs: Any) -> None:
        """Write UTF-8 JSON (same defaults as :meth:`to_json`)."""
        Path(path).write_text(self.to_json(**kwargs), encoding="utf-8")

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> Pharmacophore:
        """Build and return a **new** pharmacophore from a decoded JSON dict.

        Does not mutate any existing instance. Subclass follows ``kind`` in ``d``
        when ``cls`` is :class:`Pharmacophore`.
        """
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
        """Return a **new** pharmacophore parsed from the JSON string ``s``."""
        return cls.from_json_dict(json.loads(s))

    @classmethod
    def read_json(cls, path: str | Path) -> Pharmacophore:
        """Return a **new** pharmacophore loaded from ``path`` (UTF-8 JSON)."""
        return cls.from_json_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_phar_text(self) -> str:
        """Serialise to Pharao-style ``.phar`` text (name line, points, ``$$$$``).

        Point lines use a fixed-width feature-type column and fixed-width numeric
        columns so short and long type names (e.g. ``HACC`` vs ``HACC_AND_HDON``)
        keep coordinates aligned. Fields are whitespace-separated; parsing uses
        :meth:`from_phar_text`.
        """
        header = getattr(self, "_name", "") or ""
        lines = [header]
        for p in self._points:
            nflag = "1" if p.has_normal else "0"
            typ = f"{p.type.value:<{_PHAR_TYPE_FIELD_WIDTH}}"
            cols = [
                _phar_float_field(p.x),
                _phar_float_field(p.y),
                _phar_float_field(p.z),
                _phar_float_field(p.sigma),
                f"{nflag:>3}",
                _phar_float_field(p.nx),
                _phar_float_field(p.ny),
                _phar_float_field(p.nz),
            ]
            lines.append(f"{typ}  {' '.join(cols)}")
        lines.append("$$$$")
        return "\n".join(lines) + "\n"

    def write_phar(self, path: str | Path) -> None:
        Path(path).write_text(self.to_phar_text(), encoding="utf-8")

    def _resolve_mol_name(self, name: str | None) -> str:
        if name is not None:
            return name
        existing = getattr(self, "_name", "") or ""
        return existing or "pharmacophore"

    def to_mol(self, *, name: str | None = None) -> Any:
        """Convert this pharmacophore to an RDKit ``Chem.Mol``.

        Each pharmacophore feature becomes one **centre** pseudo-atom in a
        single 3D conformer placed at the feature centre. The element loosely
        encodes the feature type (AROM/LIPO → C, HDON → N, HACC → O,
        HACC_AND_HDON / HACC_OR_HDON → S, POSC → Na, NEGC → Cl, EXCL → F,
        UNDEF → He) and each atom carries PDB residue info so PDB output uses
        the type code as the residue name (e.g. ``ARO``, ``HDO``, ``HAC``).

        **Directional features** (currently ``AROM`` / ``AROM_OR_LIPO``)
        additionally emit two ``H`` "tip" pseudo-atoms placed symmetrically
        ±1 Å along the unit normal, sharing the parent feature's PDB residue
        and bonded to the centre. This makes the plane normal visible in 3D
        viewers; the symmetric pair mirrors Pharao's ``|cos|`` convention for
        aromatic features. See :func:`pypharao.pharmacophore_to_mol` for the
        full atom-naming and bonding scheme.

        The molecule is **not** sanitised (these are pseudo-atoms with
        non-physical valences) but is fully usable with ``Chem.SDWriter``,
        ``Chem.MolToPDBBlock`` and 3D viewers. Per-feature properties are
        attached so the molecule is self-describing (one entry per feature,
        independent of how many atoms each feature contributes):

        - ``num_features`` (int) — number of pharmacophore features
        - ``types`` (comma-separated list of point types)
        - ``sigmas`` (comma-separated list of sigmas)
        - ``centers`` (semicolon-separated ``x,y,z`` triples)
        - ``normals`` (semicolon-separated ``nx,ny,nz`` triples; ``-`` when absent)
        - ``kind`` (``"query"`` or ``"molecule"``)
        - ``name`` (only for named :class:`QueryPharmacophore` instances)

        Pass ``name=`` to override the molecule title; the default is the
        pharmacophore's own ``name`` (for query pharmacophores) or
        ``"pharmacophore"``.

        Returns the resulting ``Chem.Mol`` (typed ``Any`` because RDKit is an
        optional dependency). Requires RDKit (raises ``ImportError``
        otherwise).
        """
        from .match_report import pharmacophore_to_mol

        return pharmacophore_to_mol(self, name=self._resolve_mol_name(name))

    def write_sdf(self, path: str | Path, *, name: str | None = None) -> None:
        """Write this pharmacophore to a single-record SDF file.

        Each pharmacophore feature is rendered as one centre pseudo-atom via
        :meth:`to_mol`; directional features (``AROM`` / ``AROM_OR_LIPO``)
        additionally emit two bonded ``H`` tip atoms ±1 Å along the unit
        normal so the plane axis is visible in 3D viewers. The SDF carries
        the same per-feature properties (``types``, ``sigmas``, ``centers``,
        ``normals``, ``num_features``, ``kind`` and, for named query
        pharmacophores, ``name``) so the file is self-describing.

        Note that V2000 SDF does not carry PDB monomer info, so tip atoms
        survive the round-trip as bonded ``H`` atoms but lose their ``+`` /
        ``-`` PDB names. Use the ``num_features`` property to recover the
        feature count after re-reading.

        Pass ``name=`` to override the record title; the default is the
        pharmacophore's own ``name`` (for query pharmacophores) or
        ``"pharmacophore"``.

        Requires RDKit.
        """
        try:
            from rdkit import Chem
        except ImportError as exc:
            raise ImportError(
                "Pharmacophore.write_sdf requires RDKit (pip install rdkit)"
            ) from exc

        mol = self.to_mol(name=name)
        with Chem.SDWriter(str(Path(path))) as writer:
            writer.write(mol)

    def write_pdb(self, path: str | Path, *, name: str | None = None) -> None:
        """Write this pharmacophore to a PDB file.

        Each pharmacophore feature becomes one centre ``HETATM`` whose
        residue name encodes the feature type (``ARO``, ``LIP``, ``HDO``,
        ``HAC``, ``DAC``, ...) — see :meth:`to_mol` for the full mapping.

        Directional features (currently ``AROM`` / ``AROM_OR_LIPO``)
        additionally emit two ``H`` ``HETATM`` records with atom names
        ``+NNN`` / ``-NNN`` (one-indexed by feature), placed ±1 Å along the
        unit normal and bonded to the centre via ``CONECT`` records. All
        three atoms share the same residue, so PyMOL / Chimera / VMD group
        them as one feature.

        Pass ``name=`` to override the ``COMPND`` title; the default is
        the pharmacophore's own ``name`` (for query pharmacophores) or
        ``"pharmacophore"``.

        Requires RDKit.
        """
        try:
            from rdkit import Chem
        except ImportError as exc:
            raise ImportError(
                "Pharmacophore.write_pdb requires RDKit (pip install rdkit)"
            ) from exc

        mol = self.to_mol(name=name)
        Path(path).write_text(Chem.MolToPDBBlock(mol), encoding="utf-8")

    @classmethod
    def from_phar_text(cls, text: str) -> Pharmacophore:
        """Build and return a **new** pharmacophore from Pharao ``.phar`` text.

        Does not mutate any existing instance.
        """
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
        """Return a **new** pharmacophore read from ``path`` (UTF-8 ``.phar``)."""
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

    def purge_exclusion_spheres(self, distance: float = 8.0) -> int:
        """Drop ``EXCL`` points whose centre is farther than ``distance`` Å from
        **every** non-``EXCL`` point (Euclidean centre–centre distance).

        Non-``EXCL`` points are left unchanged. ``EXCL`` sites within ``distance``
        of *at least one* anchor point are kept.

        If there are no non-``EXCL`` points, all ``EXCL`` points are removed.

        Parameters
        ----------
        distance :
            Maximum allowed gap (ångström) from the nearest non-``EXCL`` feature;
            default ``8``.

        Returns
        -------
        int
            Number of ``EXCL`` points removed.
        """
        if distance < 0:
            raise ValueError("distance must be >= 0")
        thr = float(distance)
        anchors = [p for p in self._points if p.type != PointType.EXCL]
        kept: list[PharmacophorePoint] = []
        removed = 0
        for p in self._points:
            if p.type != PointType.EXCL:
                kept.append(p)
                continue
            if not anchors:
                removed += 1
                continue
            mind = min(
                math.dist((p.x, p.y, p.z), (a.x, a.y, a.z)) for a in anchors
            )
            if mind <= thr:
                kept.append(p)
            else:
                removed += 1
        self._points = kept
        return removed


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
