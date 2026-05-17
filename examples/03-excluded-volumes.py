#!/usr/bin/env python3
"""Add excluded volumes around a 3D molecule to a query pharmacophore.

`add_excluded_volume` builds a "negative image" of a 3D molecule: a set of
EXCL spheres tiled on a grid that sits just outside the molecule's heavy
atoms. Appended to a `QueryPharmacophore`, these EXCL points penalise hits
that occupy regions the reference molecule itself does not, which adds a
shape constraint on top of the feature-based query.

The demo builds a phenol query, decorates it with excluded volumes, and
shows that a small fragment (phenol itself) still scores well while a
sterically bulky analogue (4-tert-butylphenol) is penalised because its
extra heavy atoms overlap the EXCL shell.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from pypharao import *


def add_excluded_volume(
    mol: Chem.Mol,
    pharmacophore: QueryPharmacophore,
    *,
    conf_id: int = 0,
    shell_inner: float = 1.0,
    shell_outer: float = 2.5,
    grid_spacing: float = 1.5,
    sigma: float | None = None,
) -> int:
    """Append EXCL points on a shell around the heavy atoms of ``mol``.

    A 3D grid is built around the molecule. A grid point is kept as an
    ``EXCL`` sphere when it sits within ``shell_outer`` Å of *at least one*
    heavy atom (so it is close enough to be informative) **and** more than
    ``shell_inner`` Å away from *every* heavy atom (so it lies just outside
    the molecular volume rather than inside it). Both bounds are measured
    relative to each atom's van der Waals radius.

    Parameters
    ----------
    mol
        RDKit molecule with at least one 3D conformer.
    pharmacophore
        Query pharmacophore to extend. EXCL points are appended in place.
    conf_id
        Conformer id used for atom coordinates.
    shell_inner, shell_outer
        Inner and outer offsets (Å) added to each atom's van der Waals
        radius to define the keep-shell.
    grid_spacing
        Spacing (Å) of the cubic grid used to tile the shell.
    sigma
        Gaussian width of the new EXCL points. Defaults to
        ``DEFAULT_SIGMA[PointType.EXCL]``.

    Returns
    -------
    int
        Number of EXCL points appended to ``pharmacophore``.
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("mol must have at least one 3D conformer")
    if shell_outer <= shell_inner:
        raise ValueError("shell_outer must be strictly larger than shell_inner")

    sigma = DEFAULT_SIGMA[PointType.EXCL] if sigma is None else float(sigma)

    conf = mol.GetConformer(conf_id)
    ptable = Chem.GetPeriodicTable()

    coords: list[tuple[float, float, float]] = []
    radii: list[float] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        p = conf.GetAtomPosition(atom.GetIdx())
        coords.append((p.x, p.y, p.z))
        radii.append(float(ptable.GetRvdw(atom.GetAtomicNum())))

    if not coords:
        return 0

    coords_arr = np.asarray(coords, dtype=float)
    radii_arr = np.asarray(radii, dtype=float)

    pad = float(radii_arr.max()) + shell_outer
    lo = coords_arr.min(axis=0) - pad
    hi = coords_arr.max(axis=0) + pad
    nx, ny, nz = (np.ceil((hi - lo) / grid_spacing).astype(int) + 1).tolist()
    xs = lo[0] + np.arange(nx) * grid_spacing
    ys = lo[1] + np.arange(ny) * grid_spacing
    zs = lo[2] + np.arange(nz) * grid_spacing
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    diff = grid[:, None, :] - coords_arr[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)

    inner_bound = radii_arr + shell_inner
    outer_bound = radii_arr + shell_outer
    in_outer_shell = (dist <= outer_bound[None, :]).any(axis=1)
    in_inner_core = (dist <= inner_bound[None, :]).any(axis=1)
    keep = in_outer_shell & ~in_inner_core

    added = 0
    for x, y, z in grid[keep]:
        pharmacophore.add_point(
            PharmacophorePoint(
                type=PointType.EXCL,
                center=(float(x), float(y), float(z)),
                sigma=sigma,
            )
        )
        added += 1
    return added


def _embed(smiles: str, seed: int = 0xF00D) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.UFFOptimizeMolecule(mol)
    return mol


# ----- Build a phenol query and decorate it with excluded volumes -----
ref_mol = _embed("c1ccccc1O")
query = query_pharmacophore_from_molecule(ref_mol, name="phenol+excl")
n_features = len(query)
print(f"Query {query.get_name()!r} ({n_features} pharmacophore features):")
for p in query:
    print(f"  {p.type.value:<10} center={p.center}")

n_excl = add_excluded_volume(ref_mol, query)
print(
    f"\nAdded {n_excl} EXCL points around the reference molecule "
    f"(query now has {len(query)} points)"
)


# ----- Compare scores with and without EXCL on a small and a bulky ligand -----
ligands: list[tuple[str, Chem.Mol]] = [
    ("phenol (self)", ref_mol),
    ("4-tert-butylphenol", _embed("CC(C)(C)c1ccc(O)cc1", seed=2)),
]

query_plain = query_pharmacophore_from_molecule(ref_mol, name="phenol")
searcher_plain = PharmacophoreSearch(query_plain)
searcher_excl = PharmacophoreSearch(query)

for label, mol in ligands:
    print(f"\n=== {label} ===")
    print("Without excluded volumes:")
    print_match_results(searcher_plain.screen(mol, progress=False))
    print("With excluded volumes:")
    print_match_results(searcher_excl.screen(mol, progress=False))
