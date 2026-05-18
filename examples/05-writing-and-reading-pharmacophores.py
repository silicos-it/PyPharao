#!/usr/bin/env python3

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from pypharao import *

# ------------------------------------------------------------
# Build a query pharmacophore from a 3D structure of mol
# ------------------------------------------------------------

mol = Chem.AddHs(Chem.MolFromSmiles("Cc1conc1CC(=O)O"))
AllChem.EmbedMolecule(mol)
AllChem.UFFOptimizeMolecule(mol)

phore = query_pharmacophore_from_molecule(mol, name="phore")
print(f"\nQuery {phore.get_name()!r} ({len(phore)} features):")
for p in phore: print(f"  {p.type.value:<10} center={p.center}")

# -------------------------------------------------------------
# Write the pharmacophore in different formats
# -------------------------------------------------------------

# sdf
FILE = Path(__file__).resolve().parent / "pharmacophore.sdf"
phore.write_sdf(FILE)
print("%s written" % (FILE))

# pdb
FILE = Path(__file__).resolve().parent / "pharmacophore.pdb"
phore.write_pdb(FILE)
print("%s written" % (FILE))

# phar
FILE = Path(__file__).resolve().parent / "pharmacophore.phar"
phore.write_phar(FILE)
print("%s written" % (FILE))

# json
FILE = Path(__file__).resolve().parent / "pharmacophore.json"
phore.write_json(FILE)
print("%s written" % (FILE))

# -------------------------------------------------------------
# Read the pharmacophore (only .phar and .json supported)
# -------------------------------------------------------------

# json (read_json / read_phar return a new pharmacophore — assign the result)
print("Reading pharmacophore from json file")
FILE = Path(__file__).resolve().parent / "pharmacophore.json"
ph1 = QueryPharmacophore.read_json(FILE)
print(f"\nQuery {ph1.get_name()!r} ({len(ph1)} features):")
for p in ph1:
    print(f"  {p.type.value:<10} center={p.center}")

# phar
print("Reading pharmacophore from phar file")
FILE = Path(__file__).resolve().parent / "pharmacophore.phar"
ph2 = QueryPharmacophore.read_phar(FILE)
print(f"\nQuery {ph2.get_name()!r} ({len(ph2)} features):")
for p in ph2:
    print(f"  {p.type.value:<10} center={p.center}")
