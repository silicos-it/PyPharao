#!/usr/bin/env python3
"""Add EXCL spheres around a ligand and screen with the resulting query."""

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from pypharao import *

SMI_FILE = Path(__file__).resolve().parent / "datasets" / "compounds_10k.smi"
MAX_COMPOUNDS = None  # or None = use the whole file
NUM_CONFS = 5

# ------------------------------------------------------------
# Build a query pharmacophore from a 3D structure of phenol
# ------------------------------------------------------------

phenol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
AllChem.EmbedMolecule(phenol, randomSeed=0xF00D)
AllChem.UFFOptimizeMolecule(phenol)

pharmacophore = query_pharmacophore_from_molecule(phenol, name="phenol")
print(f"\nQuery {pharmacophore.get_name()!r} ({len(pharmacophore)} features):")
for p in pharmacophore: print(f"  {p.type.value:<10} center={p.center}")

# ------------------------------------------------------------
# Add excluded volumes around the molecule
# ------------------------------------------------------------

n_excl = add_excluded_volume(
    phenol,
    pharmacophore,
    shell_inner=1.5,     # start 1.5 Å outside the vdW surface
    shell_outer=3.5,     # end   2.5 Å outside the vdW surface
    spacing=1.0,         # grid step (Å)
    feature_clearance=1.5,  # drop grid points within 1.5 Å of an existing feature
    # max_excl=0 by default (no cap); pass max_excl=512 etc. to limit count
)
print(f"\nAdded {n_excl} EXCL spheres around phenol "
      f"({len(pharmacophore)} features total).")

# Persist the query (with its EXCL envelope) for inspection in a viewer.
OUT_PDB = Path(__file__).resolve().parent / "phenol_with_excl.pdb"
pharmacophore.write_pdb(OUT_PDB)
print(f"Wrote {OUT_PDB.name}")


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
        cids = AllChem.EmbedMultipleConfs(mol_3d, numConfs=NUM_CONFS)
        if len(cids) == 0:
            embed_fail += 1
            continue
        for cid in cids:
            AllChem.MMFFOptimizeMolecule(mol_3d, confId=cid)
        prepared.append((line_idx, smi, mol_3d))
        pbar.set_postfix(ready=len(prepared), parse=parse_fail, embed=embed_fail)

print(
    f"Done: {len(prepared)} ligands ready, "
    f"{parse_fail} invalid SMILES, {embed_fail} embed failures"
)


# ------------------------------------------------------------
# Run the screen and report the top hits
# ------------------------------------------------------------
#
# `PharmacophoreSearch` enforces excluded volumes on two levels:
#   * `with_exclusion=True`  — soft Pharao volume penalty (subtracted from the
#                              aligned overlap; reflected by `excl_volume`).
#   * `excl_hard_filter=True` (default) — after alignment, every heavy atom is
#                              treated as a sphere of its van der Waals radius;
#                              the hit is rejected if any heavy-atom vdW sphere
#                              swallows a query EXCL marker, i.e. whenever
#                              dist(atom, EXCL) < vdW(atom) + excl_clash_radius.
#                              Pass `excl_hard_filter=False` to recover the pure
#                              soft-penalty behaviour; raise `excl_clash_radius`
#                              to enforce a larger buffer around each EXCL.
#
# To make the contrast obvious, we run the screen twice on the same prepared
# molecules: once with the hard filter disabled (legacy soft-penalty only)
# and once with the default hard filter on.

soft_searcher = PharmacophoreSearch(pharmacophore, excl_hard_filter=False)
hard_searcher = PharmacophoreSearch(pharmacophore)  # excl_hard_filter=True

print("\n--- Soft EXCL penalty only (excl_hard_filter=False) ---")
soft_hits = soft_searcher.screen(prepared, progress=True)
soft_sorted = sort_match_results(soft_hits, sort="descending", key="tanimoto")
print_match_results(soft_sorted, limit=10)

print("\n--- Hard EXCL atom-clash filter ON (default) ---")
hard_hits = hard_searcher.screen(prepared, progress=True)
hard_sorted = sort_match_results(hard_hits, sort="descending", key="tanimoto")
print_match_results(hard_sorted, limit=10)

n_rejected = len(soft_hits) - len(hard_hits)
print(
    f"\n{len(soft_hits)} hits with the soft penalty alone, "
    f"{len(hard_hits)} once the hard atom-clash filter is applied "
    f"({n_rejected} candidates rejected for intruding into an EXCL sphere)."
)

SDF_FILE = Path(__file__).resolve().parent / "top_hits_phenol_excl.sdf"
write_hits_sdf(hard_sorted[:10], SDF_FILE, pharmacophore=pharmacophore)
PDB_FILE = Path(__file__).resolve().parent / "top_hits_phenol_excl.pdb"
write_hits_pdb(hard_sorted[:10], PDB_FILE, pharmacophore=pharmacophore)
