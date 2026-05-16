import numpy as np

from pypharao import (
    Alignment,
    PharmacophorePoint,
    PointType,
    QueryPharmacophore,
)
from pypharao.alignment import SolutionInfo, position_molecule_coords
from pypharao.quaternion_math import quat_to_rotation_matrix


def test_align_identical_two_points():
    ref = QueryPharmacophore()
    ref.add_point(PharmacophorePoint(type=PointType.LIPO, center=(0, 0, 0)))
    ref.add_point(PharmacophorePoint(type=PointType.LIPO, center=(2, 0, 0)))
    db = ref.copy()
    sol = Alignment(ref, db, [(0, 0), (1, 1)]).align(False)
    assert sol.volume > 0.0


def test_position_molecule_roundtrip_identity_quat():
    sol = SolutionInfo()
    sol.center1[:] = sol.center2[:] = 0.0
    sol.rotation1[:] = np.eye(3)
    sol.rotation2[:] = np.eye(3)
    sol.rotor[:] = [1, 0, 0, 0]
    U = quat_to_rotation_matrix(sol.rotor)
    coords = np.array([[1.0, 0.0, 0.0]], dtype=float)
    out = position_molecule_coords(coords, U, sol)
    assert abs(out[0, 0] - 1.0) < 1e-9
