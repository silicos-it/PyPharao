"""PyPharao: pharmacophore model and Pharao-style 3D Gaussian matching."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pypharao")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from .alignment import Alignment, position_molecule_coords, position_pharmacophore
from .match_report import (
    pharmacophore_to_mol,
    print_match_results,
    sort_match_results,
    write_hits_pdb,
    write_hits_sdf,
)
from .perception import (
    MoleculePharmacophorePerception,
    PharmacophorePerception,
    QueryPharmacophorePerception,
)
from .pharmacophore import (
    DEFAULT_SIGMA,
    TYPE_DESCRIPTIONS,
    TYPE_HAS_NORMAL,
    MoleculePharmacophore,
    Pharmacophore,
    PharmacophorePoint,
    PointType,
    QueryPharmacophore,
    cosine_normals,
    distance,
)
from .quaternion_math import quat_to_rotation_matrix
from .search import (
    MatchResult,
    PharmacophoreSearch,
    count_matchable_query_points,
    matched_query_features,
)
from .volume import volume_overlap

try:
    from .rdkit_perception import (
        add_excluded_volume,
        molecule_pharmacophore_from_molecule,
        query_pharmacophore_from_molecule,
        query_pharmacophore_from_protein,
    )
except ImportError:
    add_excluded_volume = None  # type: ignore[misc, assignment]
    molecule_pharmacophore_from_molecule = None  # type: ignore[misc, assignment]
    query_pharmacophore_from_molecule = None  # type: ignore[misc, assignment]

    def query_pharmacophore_from_protein(*_args, **_kwargs):  # type: ignore[no-redef]
        raise ImportError(
            "query_pharmacophore_from_protein requires RDKit (pip install rdkit)."
        )


__all__ = [
    "__version__",
    "Alignment",
    "DEFAULT_SIGMA",
    "MatchResult",
    "MoleculePharmacophore",
    "MoleculePharmacophorePerception",
    "Pharmacophore",
    "PharmacophorePerception",
    "PharmacophorePoint",
    "PharmacophoreSearch",
    "PointType",
    "QueryPharmacophore",
    "QueryPharmacophorePerception",
    "TYPE_DESCRIPTIONS",
    "TYPE_HAS_NORMAL",
    "add_excluded_volume",
    "cosine_normals",
    "count_matchable_query_points",
    "distance",
    "matched_query_features",
    "molecule_pharmacophore_from_molecule",
    "pharmacophore_to_mol",
    "position_molecule_coords",
    "position_pharmacophore",
    "print_match_results",
    "quat_to_rotation_matrix",
    "query_pharmacophore_from_molecule",
    "query_pharmacophore_from_protein",
    "sort_match_results",
    "volume_overlap",
    "write_hits_pdb",
    "write_hits_sdf",
]
