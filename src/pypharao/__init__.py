"""PyPharao: pharmacophore model and Pharao-style 3D Gaussian matching."""

from .alignment import Alignment, SolutionInfo, position_molecule_coords, position_pharmacophore
from .pharmacophore import (
    FUNC_HAS_NORMAL,
    FUNC_SIGMA,
    FuncGroup,
    Pharmacophore,
    PharmacophorePoint,
    cosine_normals,
    default_alpha,
    distance,
)
from .quaternion_math import quat_to_rotation_matrix
from .search import MatchResult, PharmacophoreSearch
from .volume import volume_overlap

from .perception_options import PerceptionOptions

try:
    from .rdkit_perception import (
        pharmacophore_from_molecule,
        pharmacophore_from_rdkit,
    )
except ImportError:
    pharmacophore_from_molecule = None  # type: ignore[misc, assignment]
    pharmacophore_from_rdkit = None  # type: ignore[misc, assignment]

__all__ = [
    "Alignment",
    "FUNC_HAS_NORMAL",
    "FUNC_SIGMA",
    "FuncGroup",
    "MatchResult",
    "PerceptionOptions",
    "Pharmacophore",
    "PharmacophorePoint",
    "PharmacophoreSearch",
    "SolutionInfo",
    "cosine_normals",
    "default_alpha",
    "distance",
    "pharmacophore_from_molecule",
    "pharmacophore_from_rdkit",
    "position_molecule_coords",
    "position_pharmacophore",
    "quat_to_rotation_matrix",
    "volume_overlap",
]
