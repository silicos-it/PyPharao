"""PyPharao: pharmacophore model and Pharao-style 3D Gaussian matching."""

from .alignment import Alignment, position_molecule_coords, position_pharmacophore
from .match_report import (
    print_match_results,
    sort_match_results,
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
        molecule_pharmacophore_from_molecule,
        query_pharmacophore_from_molecule,
        query_pharmacophore_from_protein,
    )
except ImportError:
    molecule_pharmacophore_from_molecule = None  # type: ignore[misc, assignment]
    query_pharmacophore_from_molecule = None  # type: ignore[misc, assignment]

    def query_pharmacophore_from_protein(*args, **kwargs):  # type: ignore[no-redef]
        raise NotImplementedError(
            "query_pharmacophore_from_protein is not implemented yet."
        )


__all__ = [
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
    "cosine_normals",
    "count_matchable_query_points",
    "distance",
    "matched_query_features",
    "molecule_pharmacophore_from_molecule",
    "position_molecule_coords",
    "position_pharmacophore",
    "print_match_results",
    "quat_to_rotation_matrix",
    "query_pharmacophore_from_molecule",
    "query_pharmacophore_from_protein",
    "sort_match_results",
    "volume_overlap",
    "write_hits_sdf",
]
