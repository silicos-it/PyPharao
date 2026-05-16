#!/usr/bin/env python3
"""Modify a query pharmacophore by hand to introduce AROM_OR_LIPO and HACC_OR_HDON points.

Notes
-----
``HACC_OR_HDON`` (query) only matches ``HDON`` or ``HACC`` in a molecule (per
the matching table); it does **not** match the auto-merged ``HACC_AND_HDON``
compound type. When the query carries ``HACC_OR_HDON`` points, configure both
perceptions to skip the ``HACC_AND_HDON`` merge so the molecule keeps the
elementary ``HDON``/``HACC`` sites the query expects.
"""

from rdkit import Chem
from rdkit.Chem import AllChem

from pypharao import *


# ----- Build a starting query from phenol with the compound merge disabled -----
ref_mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
AllChem.EmbedMolecule(ref_mol, randomSeed=0xF00D)
AllChem.UFFOptimizeMolecule(ref_mol)

query_perception = QueryPharmacophorePerception()
query_perception.disable(PointType.HACC_AND_HDON)  # keep elementary HDON/HACC
query = query_pharmacophore_from_molecule(ref_mol, query_perception, name="phenol")

print(f"Initial query {query.get_name()!r} ({len(query)} features):")
for p in query:
    print(f"  {p.type.value:<10} center={p.center}")


# ----- Convert AROM → AROM_OR_LIPO -----
for i, p in enumerate(query):
    if p.type == PointType.AROM:
        query.set_point(
            i,
            PharmacophorePoint(
                type=PointType.AROM_OR_LIPO,
                center=p.center,
                sigma=DEFAULT_SIGMA[PointType.AROM_OR_LIPO],
                normal=p.normal,
            ),
        )
        print(f"\nConverted point {i}: AROM → AROM_OR_LIPO")
        break


# ----- Collapse the phenol HDON+HACC pair into a single HACC_OR_HDON point -----
hdon_idx = next((i for i, p in enumerate(query) if p.type == PointType.HDON), None)
hacc_idx = next((i for i, p in enumerate(query) if p.type == PointType.HACC), None)
if hdon_idx is not None and hacc_idx is not None:
    centre = query[hdon_idx].center
    # Remove the higher index first so the lower one stays valid.
    for idx in sorted({hdon_idx, hacc_idx}, reverse=True):
        query.remove_point(idx)
    query.add_point(
        PharmacophorePoint(
            type=PointType.HACC_OR_HDON,
            center=centre,
            sigma=DEFAULT_SIGMA[PointType.HACC_OR_HDON],
        )
    )
    print("Replaced HDON+HACC pair with HACC_OR_HDON")


# ----- Append an EXCL sphere above the ring plane -----
ring = next((p for p in query if p.type == PointType.AROM_OR_LIPO), query[0])
query.add_point(
    PharmacophorePoint(
        type=PointType.EXCL,
        center=(ring.x, ring.y, ring.z + 2.5),
        sigma=DEFAULT_SIGMA[PointType.EXCL],
    )
)
print("Appended an EXCL sphere")

print(f"\nFinal query ({len(query)} features):")
for p in query:
    print(f"  {p.type.value:<10} center={p.center}")


# ----- Configure molecule perception to skip the compound merge too -----
mol_perception = MoleculePharmacophorePerception()
mol_perception.disable(PointType.HACC_AND_HDON)
searcher = PharmacophoreSearch(query, perception=mol_perception)


# ----- Screen the original molecule and a methanol decoy -----
decoy = Chem.AddHs(Chem.MolFromSmiles("CO"))
AllChem.EmbedMolecule(decoy, randomSeed=2)
AllChem.UFFOptimizeMolecule(decoy)

print("\nScreening phenol with the modified query:")
print_match_results(searcher.screen(ref_mol, progress=False))
print("\nScreening methanol decoy:")
print_match_results(searcher.screen(decoy, progress=False))
