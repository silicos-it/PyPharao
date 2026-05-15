#!/usr/bin/env python3
"""Build a pharmacophore from a 3D RDKit molecule."""

from rdkit import Chem
from rdkit.Chem import AllChem

from pypharao import PerceptionOptions, pharmacophore_from_molecule

mol = Chem.MolFromSmiles("c1ccccc1O")
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
AllChem.UFFOptimizeMolecule(mol)

options = PerceptionOptions()

print("All available features and which one are used:")
options.print_features()
print("\nIs the HACC feature used:")
print(options.is_enabled_for_perception("HACC"))

print("\nPharmacophore search on phenol:")
ph = pharmacophore_from_molecule(mol, 
    PerceptionOptions(), 
    conf_id=0, 
    name="pharmacophore_from_phenol")
print(ph.name, len(ph), "features")
for p in ph.points:
    print(p.func.value, f"({p.x:.2f}, {p.y:.2f}, {p.z:.2f})")
