#!/usr/bin/env python3
"""Screen a small library across multiple conformers and write the hits to SDF."""

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from pypharao import *

SMI_FILE = Path(__file__).resolve().parent / "datasets" / "compounds_10k.smi"
MAX_COMPOUNDS = 500
NUM_CONFS = 5
SDF_OUT = Path("phenol_hits.sdf")


# ----- Build the query from phenol -----
ref_mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
AllChem.EmbedMolecule(ref_mol, randomSeed=0xF00D)
AllChem.UFFOptimizeMolecule(ref_mol)
query = query_pharmacophore_from_molecule(ref_mol, name="phenol")


# ----- Generate multi-conformer database molecules -----
smiles_lines = [
    ln.strip()
    for ln in SMI_FILE.read_text(encoding="utf-8").splitlines()
    if ln.strip()
][:MAX_COMPOUNDS]

print(f"Embedding {NUM_CONFS} conformers each for {len(smiles_lines)} SMILES")
prepared: list[tuple[int, str, Chem.Mol]] = []
with tqdm(smiles_lines, desc="3D structures", unit="mol") as pbar:
    for line_idx, smi in enumerate(pbar):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(
            mol, numConfs=NUM_CONFS, randomSeed=0xF00D + line_idx
        )
        if mol.GetNumConformers() == 0:
            continue
        for cid in range(mol.GetNumConformers()):
            AllChem.UFFOptimizeMolecule(mol, confId=cid)
        prepared.append((line_idx, smi, mol))
        pbar.set_postfix(ready=len(prepared))


# ----- Screen all conformers; keep the best per molecule -----
searcher = PharmacophoreSearch(query)
print("\nScreening (keep='best', metric='tanimoto'):")
best_hits = searcher.screen(prepared, conformations="all", keep="best")
sorted_hits = sort_match_results(best_hits, key="tanimoto")
print_match_results(sorted_hits, limit=5)


# ----- Re-screen with keep='all' (every passing conformer becomes a row) -----
print("\nScreening (keep='all') — every conformer that satisfies min_matches:")
all_hits = searcher.screen(prepared, conformations="all", keep="all")
print(f"  {len(all_hits)} matched conformers across {len(prepared)} molecules")


# ----- Persist the best-per-molecule hits as an SDF file -----
n = write_hits_sdf(sorted_hits[:25], SDF_OUT)
print(f"\nWrote {n} aligned molecules to {SDF_OUT}")
