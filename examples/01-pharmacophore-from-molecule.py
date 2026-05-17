#!/usr/bin/env python3
"""Build a query pharmacophore from a molecule and screen a library."""

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from pypharao import *

SMI_FILE = Path(__file__).resolve().parent / "datasets" / "compounds_10k.smi"
MAX_COMPOUNDS = None  # None = use the whole file
#MAX_COMPOUNDS = 500

# ------------------------------------------------------------
# Build two query pharmacophores from a 3D structure of phenol
# ------------------------------------------------------------

ref_mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
AllChem.EmbedMolecule(ref_mol, randomSeed=0xF00D)
AllChem.UFFOptimizeMolecule(ref_mol)

perception = QueryPharmacophorePerception()
print("Auto-perceivable feature types for a query pharmacophore:")
perception.print_features()

# Pharmacophore 1
pharmacophore_1 = query_pharmacophore_from_molecule(ref_mol, perception, name="phenol")
print(f"\nQuery {pharmacophore_1.get_name()!r} ({len(pharmacophore_1)} features):")
for p in pharmacophore_1: print(f"  {p.type.value:<10} center={p.center}")

# Pharmacophore 2: same as pharmacophore_1 but with AROM relaxed to AROM_OR_LIPO.
pharmacophore_2 = pharmacophore_1.copy()
for i, p in enumerate(pharmacophore_2):
    if p.type == PointType.AROM:
        pharmacophore_2.update_point(i, type=PointType.AROM_OR_LIPO)
        break
pharmacophore_2.set_name("phenol-arom-or-lipo")
print(f"\nQuery {pharmacophore_2.get_name()!r} ({len(pharmacophore_2)} features):")
for p in pharmacophore_2: print(f"  {p.type.value:<10} center={p.center}")

# Self-screen sanity check
searcher_1 = PharmacophoreSearch(pharmacophore_1)
print("\nSelf-screen pharmacophore 1:")
print_match_results(searcher_1.screen(ref_mol, progress=False))

searcher_2 = PharmacophoreSearch(pharmacophore_2)
print("\nSelf-screen pharmacophore 2:")
print_match_results(searcher_2.screen(ref_mol, progress=False))


# ------------------------------------------------------------
# Build 3D conformers for a SMILES dataset
# ------------------------------------------------------------

smiles_lines = [
    ln.strip()
    for ln in SMI_FILE.read_text(encoding="utf-8").splitlines()
    if ln.strip()
]
if MAX_COMPOUNDS is not None:
    smiles_lines = smiles_lines[:MAX_COMPOUNDS]

print(f"\nBuilding 3D structures for {len(smiles_lines)} SMILES from {SMI_FILE.name}")
prepared: list[tuple[int, str, Chem.Mol]] = []
parse_fail = embed_fail = 0
with tqdm(smiles_lines, desc="3D structures", unit="mol") as pbar:
    for line_idx, smi in enumerate(pbar):
        mol_3d = Chem.MolFromSmiles(smi)
        if mol_3d is None:
            parse_fail += 1
            continue
        mol_3d = Chem.AddHs(mol_3d)
        if AllChem.EmbedMolecule(mol_3d, randomSeed=0xF00D + line_idx) != 0:
            embed_fail += 1
            continue
        AllChem.UFFOptimizeMolecule(mol_3d)
        prepared.append((line_idx, smi, mol_3d))
        pbar.set_postfix(ready=len(prepared), parse=parse_fail, embed=embed_fail)

print(
    f"Done: {len(prepared)} ligands ready, "
    f"{parse_fail} invalid SMILES, {embed_fail} embed failures"
)


# ------------------------------------------------------------
# Run the screen and report the top hits
# ------------------------------------------------------------

# Pharmacophore 1
print("Pharmacophore 1")
hits = searcher_1.screen(prepared, progress=True)
sorted_hits = sort_match_results(hits, sort="descending", key="tanimoto")
print_match_results(sorted_hits, limit=10)
write_hits_sdf(sorted_hits[:10], "top_hits_ph_1.sdf", pharmacophore=pharmacophore_1)
write_hits_pdb(sorted_hits[:10], "top_hits_ph_1.pdb", pharmacophore=pharmacophore_1)
print()

# Pharmacophore 2
print("Pharmacophore 2")
hits = searcher_2.screen(prepared, progress=True)
sorted_hits = sort_match_results(hits, sort="descending", key="tanimoto")
print_match_results(sorted_hits, limit=10)
write_hits_sdf(sorted_hits[:10], "top_hits_ph_2.sdf", pharmacophore=pharmacophore_2)
write_hits_pdb(sorted_hits[:10], "top_hits_ph_2.pdb", pharmacophore=pharmacophore_2)

