"""Quaternion utilities and 4x4 Hessian inversion (Pharao `utilities.cpp`)."""

from __future__ import annotations

import math

import numpy as np


def normalize_quaternion(q: np.ndarray) -> None:
    d = float(np.dot(q, q))
    # Guard against the (theoretically reachable) zero quaternion: fall back to
    # the identity rotor so a degenerate gradient step cannot poison the rotor
    # with NaNs.
    if d < 1e-30:
        q[0] = 1.0
        q[1] = 0.0
        q[2] = 0.0
        q[3] = 0.0
        return
    q /= math.sqrt(d)


def quat_to_rotation_matrix(Q: np.ndarray) -> np.ndarray:
    """Same as `quat2Rotation` in Pharao utilities.cpp."""
    d1sq = Q[1] * Q[1]
    d2sq = Q[2] * Q[2]
    d3sq = Q[3] * Q[3]
    U = np.zeros((3, 3), dtype=float)
    U[0, 0] = 1.0 - 2.0 * d2sq - 2.0 * d3sq
    U[1, 0] = 2.0 * (Q[1] * Q[2] + Q[0] * Q[3])
    U[2, 0] = 2.0 * (Q[1] * Q[3] - Q[0] * Q[2])
    U[0, 1] = 2.0 * (Q[2] * Q[1] - Q[0] * Q[3])
    U[1, 1] = 1.0 - 2.0 * d1sq - 2.0 * d3sq
    U[2, 1] = 2.0 * (Q[2] * Q[3] + Q[0] * Q[1])
    U[0, 2] = 2.0 * (Q[3] * Q[1] + Q[0] * Q[2])
    U[1, 2] = 2.0 * (Q[3] * Q[2] - Q[0] * Q[1])
    U[2, 2] = 1.0 - 2.0 * d1sq - 2.0 * d2sq
    return U


def row_product(A: np.ndarray, U: np.ndarray) -> np.ndarray:
    """SiMath::rowProduct — matrix-vector product A @ U."""
    return A @ U


def inverse_hessian(H: np.ndarray) -> None:
    """In-place 4x4 inverse via block formula (`inverseHessian` in utilities.cpp)."""
    R0 = np.zeros((2, 2), dtype=float)
    d = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
    if d > 1e-6 or d < -1e-6:
        d = 1.0 / d
    R0[0, 0] = d * H[1, 1]
    R0[1, 1] = d * H[0, 0]
    R0[0, 1] = -d * H[0, 1]
    R0[1, 0] = -d * H[1, 0]

    R1 = np.zeros((2, 2), dtype=float)
    R1[0, 0] = H[2, 0] * R0[0, 0] + H[2, 1] * R0[1, 0]
    R1[0, 1] = H[2, 0] * R0[0, 1] + H[2, 1] * R0[1, 1]
    R1[1, 0] = H[3, 0] * R0[0, 0] + H[3, 1] * R0[1, 0]
    R1[1, 1] = H[3, 0] * R0[0, 1] + H[3, 1] * R0[1, 1]

    R2 = np.zeros((2, 2), dtype=float)
    R2[0, 0] = R0[0, 0] * H[0, 2] + R0[0, 1] * H[1, 2]
    R2[0, 1] = R0[0, 0] * H[0, 3] + R0[0, 1] * H[1, 3]
    R2[1, 0] = R0[1, 0] * H[0, 2] + R0[1, 1] * H[1, 2]
    R2[1, 1] = R0[1, 0] * H[0, 3] + R0[1, 1] * H[1, 3]

    R3 = np.zeros((2, 2), dtype=float)
    R3[0, 0] = H[2, 0] * R2[0, 0] + H[2, 1] * R2[1, 0]
    R3[0, 1] = H[2, 0] * R2[0, 1] + H[2, 1] * R2[1, 1]
    R3[1, 0] = H[3, 0] * R2[0, 0] + H[3, 1] * R2[1, 0]
    R3[1, 1] = H[3, 0] * R2[0, 1] + H[3, 1] * R2[1, 1]

    R3[0, 0] -= H[2, 2]
    R3[0, 1] -= H[2, 3]
    R3[1, 0] -= H[3, 2]
    R3[1, 1] -= H[3, 3]

    d = R3[0, 0] * R3[1, 1] - R3[0, 1] * R3[1, 0]
    if d > 1e-6 or d < -1e-6:
        R3[:, :] /= d

    d = R3[1, 1]
    R3[1, 1] = R3[0, 0]
    R3[0, 0] = d
    R3[1, 0] = -R3[1, 0]
    R3[0, 1] = -R3[0, 1]

    H[0, 2] = R2[0, 0] * R3[0, 0] + R2[0, 1] * R3[1, 0]
    H[0, 3] = R2[0, 0] * R3[0, 1] + R2[0, 1] * R3[1, 1]
    H[1, 2] = R2[1, 0] * R3[0, 0] + R2[1, 1] * R3[1, 0]
    H[1, 3] = R2[1, 0] * R3[0, 1] + R2[1, 1] * R3[1, 1]

    H[2, 0] = R3[0, 0] * R1[0, 0] + R3[0, 1] * R1[1, 0]
    H[2, 1] = R3[0, 0] * R1[0, 1] + R3[0, 1] * R1[1, 1]
    H[3, 0] = R3[1, 0] * R1[0, 0] + R3[1, 1] * R1[1, 0]
    H[3, 1] = R3[1, 0] * R1[0, 1] + R3[1, 1] * R1[1, 1]

    R4 = np.zeros((2, 2), dtype=float)
    R4[0, 0] = R2[0, 0] * H[2, 0] + R2[0, 1] * H[3, 0]
    R4[0, 1] = R2[0, 0] * H[2, 1] + R2[0, 1] * H[3, 1]
    R4[1, 0] = R2[1, 0] * H[2, 0] + R2[1, 1] * H[3, 0]
    R4[1, 1] = R2[1, 0] * H[2, 1] + R2[1, 1] * H[3, 1]

    H[0, 0] = R0[0, 0] - R4[0, 0]
    H[0, 1] = R0[0, 1] - R4[0, 1]
    H[1, 0] = R0[1, 0] - R4[1, 0]
    H[1, 1] = R0[1, 1] - R4[1, 1]

    H[2, 2] = -R3[0, 0]
    H[2, 3] = -R3[0, 1]
    H[3, 2] = -R3[1, 0]
    H[3, 3] = -R3[1, 1]


def rotate_coord_util(p: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Pharao `rotate` using matrix rows (utilities.cpp)."""
    x, y, z = p[0], p[1], p[2]
    return np.array(
        [
            x * U[0, 0] + y * U[0, 1] + z * U[0, 2],
            x * U[1, 0] + y * U[1, 1] + z * U[1, 2],
            x * U[2, 0] + y * U[2, 1] + z * U[2, 2],
        ],
        dtype=float,
    )


def rotate_coord_align(p: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Pharao Alignment frame rotation: first component uses column 0 of R."""
    x, y, z = p[0], p[1], p[2]
    return np.array(
        [
            R[0, 0] * x + R[1, 0] * y + R[2, 0] * z,
            R[0, 1] * x + R[1, 1] * y + R[2, 1] * z,
            R[0, 2] * x + R[1, 2] * y + R[2, 2] * z,
        ],
        dtype=float,
    )
