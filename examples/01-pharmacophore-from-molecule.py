#!/usr/bin/env python3
"""Build a pharmacophore from a molecule and screen a compound library."""

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from pypharao import (
    FuncGroup,
    MatchResult,
    PerceptionOptions,
    Pharmacophore,
    PharmacophorePoint,
    PharmacophoreSearch,
    default_alpha,
    pharmacophore_from_molecule,
    print_match_results,
    sort_match_results,
)

SMI_FILE = Path(__file__).resolve().parent / "datasets" / "compounds_10k.smi"
MAX_COMPOUNDS = None  # None: entire file; set e.g. 100 for a quick trial run

# ----------------------------------
# Create a pharmacophore from phenol
# ----------------------------------

mol = Chem.MolFromSmiles("c1ccccc1O")
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
AllChem.UFFOptimizeMolecule(mol)

pharmacophore_features = PerceptionOptions()
print("Which pharmacophore features are percepted and which ones not:")
pharmacophore_features.print_features()

print("\nIs the HACC feature percepted when present:")
print(pharmacophore_features.is_enabled_for_perception("HACC"))
print()

ph = pharmacophore_from_molecule(
    mol,
    pharmacophore_features,
    conf_id=0,
    name="pharmacophore_from_phenol",
)

searcher = PharmacophoreSearch(ph)
matches = searcher.screen(mol)
print("Pharmacophore match on phenol itself:")
print_match_results(matches)
print()

# --------------------------------------------------
# Generate a compound database from 10,000 molecules
# --------------------------------------------------

smiles_lines = []
for line in SMI_FILE.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line:
        smiles_lines.append(line)

if MAX_COMPOUNDS is not None:
    smiles_lines = smiles_lines[:MAX_COMPOUNDS]

print(f"\nPhase 1: building 3D structures for {len(smiles_lines)} SMILES from {SMI_FILE.name}")
prepared_mols: list[tuple[int, str, Chem.Mol]] = []
n_mol_parse_fail = n_mol_embed_fail = 0
with tqdm(smiles_lines, desc="3D structures", unit="mol") as pbar:
    for line_idx, smi in enumerate(pbar):
        mol_3d = Chem.MolFromSmiles(smi)
        if mol_3d is None:
            n_mol_parse_fail += 1
            continue
        mol_3d = Chem.AddHs(mol_3d)
        if AllChem.EmbedMolecule(mol_3d) != 0:
            n_mol_embed_fail += 1
            continue
        AllChem.UFFOptimizeMolecule(mol_3d)
        prepared_mols.append((line_idx, smi, mol_3d))
    pbar.set_postfix(
        ready=len(prepared_mols),
        parse=n_mol_parse_fail,
        embed=n_mol_embed_fail,
    )

print(
    f"Done: {len(prepared_mols)} 3D molecules ready, "
    f"{n_mol_parse_fail} invalid SMILES, {n_mol_embed_fail} embed failures "
    f"({len(smiles_lines)} SMILES in total)"
)

prepared_by_line = {line_idx: (smi, mol_3d) for line_idx, smi, mol_3d in prepared_mols}

# ---------------------------
# Do the pharmacophore screen
# ---------------------------

matches = searcher.screen(prepared_mols, progress=True)
sorted_matches = sort_match_results(matches, sort="descending", key="tanimoto")
print_match_results(sorted_matches, limit=2)
