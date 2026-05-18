#!/usr/bin/env python3

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from pypharao import *

NUM_CONFS = 5
SDF_OUT = Path(__file__).resolve().parent / "paracetamol_hits.sdf"

# ----- Build the query from paracetamol -----
query_mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1"))
AllChem.EmbedMolecule(query_mol)
AllChem.UFFOptimizeMolecule(query_mol)
query = query_pharmacophore_from_molecule(query_mol, name="paracetamol")
print(f"\nQuery {query.get_name()!r} ({len(query)} features):")
for p in query: print(f"  {p.type.value:<10} center={p.center}")

# ----- Generate multiple conformations of paracetamol -----
print(f"Embedding {NUM_CONFS} conformers for paracetamol")
db = Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1")
db = Chem.AddHs(db)
AllChem.EmbedMultipleConfs(db, numConfs=NUM_CONFS)

# ----- Screen all conformers; keep the best per molecule -----
searcher = PharmacophoreSearch(query)
print("\nScreening (keep='best', metric='tanimoto'):")
best_hits = searcher.screen([db], conformations="all", keep="best")
sorted_hits = sort_match_results(best_hits, key="tanimoto")
print_match_results(sorted_hits, limit=5)

# ----- Re-screen with keep='all' (every passing conformer becomes a row) -----
print("\nScreening (keep='all') — every conformer that satisfies min_matches:")
all_hits = searcher.screen([db], conformations="all", keep="all")
print_match_results(all_hits, limit=5)

# ----- Persist the best-per-molecule hits as an SDF file -----
ranked = sort_match_results(all_hits, sort="descending", key="tanimoto")
n = write_hits_sdf(ranked, SDF_OUT, pharmacophore=query)
print(f"\nWrote {n} aligned molecules to {SDF_OUT}")
