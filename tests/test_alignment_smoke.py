import numpy as np

from pypharao import Alignment, FuncGroup, Pharmacophore, PharmacophorePoint
from pypharao.alignment import SolutionInfo, position_molecule_coords
from pypharao.quaternion_math import quat_to_rotation_matrix


def test_align_identical_two_points():
    ref = Pharmacophore(
        points=[
            PharmacophorePoint(0, 0, 0, FuncGroup.LIPO, 0.7, False, 0, 0, 0),
            PharmacophorePoint(2, 0, 0, FuncGroup.LIPO, 0.7, False, 0, 0, 0),
        ]
    )
    db = ref.copy()
    pairs = [(0, 0), (1, 1)]
    al = Alignment(ref, db, pairs)
    sol = al.align(False)
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
