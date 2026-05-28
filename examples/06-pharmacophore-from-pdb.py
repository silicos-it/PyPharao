#!/usr/bin/env python3

"""Build a query pharmacophore from PDB: binding-site features + exclusion atoms."""

from collections import Counter
from pathlib import Path

from pypharao import PointType, query_pharmacophore_from_protein

DATASETS = Path(__file__).resolve().parent / "datasets"
PROTEIN_PDB = DATASETS / "4HFZ_pharmacophore.pdb"
EXCL_PDB = DATASETS / "4HFZ_exclusion.pdb"
PHARMACOPHORE1_AS_PDB = Path(__file__).resolve().parent / "ph1_from_4HFZ.pdb"
PHARMACOPHORE2_AS_PDB = Path(__file__).resolve().parent / "ph2_from_4HFZ.pdb"

# ------------------------------------------------------------
# Query from protein pharmacophore PDB + exclusion-sphere PDB
# ------------------------------------------------------------

# Generate pharmacophore
query = query_pharmacophore_from_protein(
    PROTEIN_PDB,
    EXCL_PDB,
    min_distance_between_excl_points=1.5,
    name="4HFZ-binding-site",
)

# Show pharmacophore
by_type = Counter(p.type.value for p in query)
print(f"\nQuery {query.get_name()!r} ({len(query)} features)")
print("  counts:", dict(sorted(by_type.items())))
print("\nFirst points (truncated):")
for i, p in enumerate(query):
    if i >= 8:
        print(f"  … ({len(query) - 12} more)")
        break
    label = p.type.value
    x, y, z = p.center
    coord = f"({x:.3f}, {y:.3f}, {z:.3f})"
    extra = f" σ={p.sigma:.2f}" if p.type == PointType.EXCL else ""
    print(f"  {label:<14} center={coord}{extra}")

# Write pharmacophore to pdb file
query.write_pdb(PHARMACOPHORE1_AS_PDB)

# ---------------------------
# Now we need some polishing
# ---------------------------

# First merge all three LIPO points in a central one (new coordinate = -6.989, -32.071, 31.677)
# We will modify the first LIPO (i = 4), and remove i = 5 and 6

x = y = z = 0.0
for i, p in enumerate(query):
    if i in [4,5,6]:
        x += p.center[0]
        y += p.center[1]
        z += p.center[2]
x /= 3
y /= 3
z /= 3
query.update_point(4, center=(x,y,z))
query.remove_point(6)
query.remove_point(5)

# We can also remove AROM 2
query.remove_point(1)

# Now remove all EXCL points that are located than 8 A from any pharmacophore point
n_removed = query.purge_exclusion_spheres(distance=8)
print("Purged pharmacophore. Removed %d EXCL points" % (n_removed))

# Write as pdb and print final set
query.write_pdb(PHARMACOPHORE2_AS_PDB)

for i, p in enumerate(query):
    if i >= 8:
        print(f"  … ({len(query) - 12} more)")
        break
    label = p.type.value
    x, y, z = p.center
    coord = f"({x:.3f}, {y:.3f}, {z:.3f})"
    extra = f" σ={p.sigma:.2f}" if p.type == PointType.EXCL else ""
    print(f"  {label:<14} center={coord}{extra}")
