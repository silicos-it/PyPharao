#!/usr/bin/env python3
"""Build a query pharmacophore from a molecule and screen a library."""

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from pypharao import *

SMI_FILE = Path(__file__).resolve().parent / "datasets" / "compounds_10k.smi"
MAX_COMPOUNDS = 500  # or None = use the whole file

# ------------------------------------------------------------
# Build two query pharmacophores from a 3D structure of phenol
# ------------------------------------------------------------

# Create 3D molecule (phenol)
phenol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
AllChem.EmbedMolecule(phenol)
AllChem.UFFOptimizeMolecule(phenol)

# Pharmacophore
pharmacophore = query_pharmacophore_from_molecule(phenol, name="phenol")
print(f"\nQuery {pharmacophore.get_name()!r} ({len(pharmacophore)} features):")
for p in pharmacophore: print(f"  {p.type.value:<10} center={p.center}")

# Add excluded volumes around molecule


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
SDF_FILE = Path(__file__).resolve().parent / "top_hits_ph_1.sdf"
write_hits_sdf(sorted_hits[:10], SDF_FILE, pharmacophore=pharmacophore_1)
PDB_FILE = Path(__file__).resolve().parent / "top_hits_ph_1.pdb"
write_hits_pdb(sorted_hits[:10], PDB_FILE, pharmacophore=pharmacophore_1)
print()

# Pharmacophore 2
print("Pharmacophore 2")
hits = searcher_2.screen(prepared, progress=True)
sorted_hits = sort_match_results(hits, sort="descending", key="tanimoto")
print_match_results(sorted_hits, limit=10)
SDF_FILE = Path(__file__).resolve().parent / "top_hits_ph_2.sdf"
write_hits_sdf(sorted_hits[:10], SDF_FILE, pharmacophore=pharmacophore_2)
PDB_FILE = Path(__file__).resolve().parent / "top_hits_ph_2.pdb"
write_hits_pdb(sorted_hits[:10], PDB_FILE, pharmacophore=pharmacophore_2)

