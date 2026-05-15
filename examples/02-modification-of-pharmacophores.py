#!/usr/bin/env python3
"""Build a pharmacophore from a molecule and screen a compound library."""

import sys
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
)

# -------------------------------------------
# PART I - Create a pharmacophore from phenol
# -------------------------------------------

# Create phenol and generate 3D structure
mol = Chem.MolFromSmiles("c1ccccc1O")
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
AllChem.UFFOptimizeMolecule(mol)

# Give a list of all potential pharmacophore features
pharmacophore_features = PerceptionOptions()
print("Which pharmacophore features are percepted and which ones not:")
options.print_features()
print("\nIs the HACC feature percepted when present:")
print(options.is_enabled_for_perception("HACC"))

# Create a pharmacophore from phenol.
# Allow all features from the pharmacophore_features instance
ph = pharmacophore_from_molecule(
    mol,
    options,
    conf_id=0,
    name="pharmacophore_from_phenol",
)

sys.exit()

# ----------------------------------------------------------
# PART II - Screen a compound library in two separate phases
# ----------------------------------------------------------

SMI_FILE = Path(__file__).resolve().parent / "datasets" / "compounds_10k.smi"
MAX_COMPOUNDS = None  # None: entire file; set e.g. 100 for a quick trial run
RUN_MATCH_DIAGNOSTICS = True  # print mapping/volumes for the top hit per phase

def build_3d_mol(smiles: str, seed: int) -> Chem.Mol | None:
    """Parse SMILES, add hydrogens, embed, and minimize. Returns None on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        return None
    AllChem.UFFOptimizeMolecule(mol)
    return mol


def print_query_pharmacophore(ph: Pharmacophore) -> None:
    print(f"\nQuery pharmacophore: {ph.name!r} ({len(ph)} features)")
    for p in ph.points:
        print(f"  {p.func.value}  ({p.x:.2f}, {p.y:.2f}, {p.z:.2f})")


def print_database_perception(searcher: PharmacophoreSearch) -> None:
    print("\nDatabase perception flags (derived from query pharmacophore):")
    assert searcher.perception_options is not None
    searcher.perception_options.print_features()


def run_screening_phase(
    phase: int,
    ph: Pharmacophore,
    prepared_mols: list[tuple[int, str, Chem.Mol]],
    prepared_by_line: dict[int, tuple[str, Chem.Mol]],
) -> tuple[list[tuple[float, int, str]], PharmacophoreSearch]:
    """Print query/perception info, screen the library, and return ranked hits."""
    print_query_pharmacophore(ph)
    searcher = PharmacophoreSearch(ph)
    print_database_perception(searcher)

    results: list[tuple[float, int, str]] = []
    n_search_perceive_fail = 0
    print(f"\nPhase {phase}: pharmacophore search ({len(prepared_mols)} molecules)")
    mol_list = [mol_3d for _, _, mol_3d in prepared_mols]
    meta = [(line_idx, smi) for line_idx, smi, _ in prepared_mols]
    try:
        hits = searcher.screen(mol_list)  # n_jobs=None uses all CPUs
    except Exception:
        n_search_perceive_fail = len(mol_list)
        hits = []
    for mol_idx, match in hits:
        line_idx, smi = meta[mol_idx]
        results.append((match.tanimoto, line_idx, smi))

    results.sort(reverse=True)
    print(
        f"Phase {phase} done: {len(results)} hits scored, "
        f"{n_search_perceive_fail} perception/search failures "
        f"({len(prepared_mols)} molecules searched)"
    )
    print("Top 10 by Tanimoto:")
    print_top_hits(results)
    if RUN_MATCH_DIAGNOSTICS:
        run_top_hit_diagnostics(f"Phase {phase}", results, ph, searcher, prepared_by_line)
    return results, searcher


def print_top_hits(
    results: list[tuple[float, int, str]],
    *,
    n: int = 10,
) -> None:
    """Print the highest-Tanimoto hits."""
    shown = 0
    for score, idx, smi in results:
        print(
            f"  {score:.4f}  line {idx + 1}  "
            f"{smi[:72]}{'...' if len(smi) > 72 else ''}"
        )
        shown += 1
        if shown >= n:
            break
    if shown == 0:
        print("  (no hits to show)")


def diagnose_match(
    title: str,
    query_ph: Pharmacophore,
    searcher: PharmacophoreSearch,
    line_idx: int,
    smi: str,
    mol_3d: Chem.Mol,
    match: MatchResult,
) -> None:
    """Print how a ligand was perceived and which query/db features were paired."""
    db = match.database_pharmacophore
    mapping = [
        (query_ph[r].func.value, db[d].func.value)
        for r, d in match.mapping
        if query_ph[r].func != FuncGroup.EXCL
    ]
    print(f"\n--- Match diagnostics: {title} (line {line_idx + 1}) ---")
    print(f"SMILES: {smi}")
    print(f"Query features: {[p.func.value for p in query_ph.points]}")
    print(f"DB features:    {[p.func.value for p in db.points]}")
    print(f"Mapping (query → db): {mapping}")
    print(
        f"Tanimoto={match.tanimoto:.4f}  overlap={match.overlap_volume:.4f}  "
        f"ref_vol={match.ref_volume:.4f}  db_vol={match.db_volume:.4f}"
    )


def run_top_hit_diagnostics(
    phase_label: str,
    results: list[tuple[float, int, str]],
    query_ph: Pharmacophore,
    searcher: PharmacophoreSearch,
    prepared_by_line: dict[int, tuple[str, Chem.Mol]],
) -> None:
    """Diagnose the top overall hit and the best-scoring hit (if any)."""
    if not results:
        return
    score, idx, smi = results[0]
    mol_3d = prepared_by_line[idx][1]
    match = searcher.screen(mol_3d)
    if match is None:
        return
    diagnose_match(
        f"{phase_label} — top hit",
        query_ph,
        searcher,
        idx,
        smi,
        mol_3d,
        match,
    )


smiles_lines = [
    ln.strip()
    for ln in SMI_FILE.read_text(encoding="utf-8").splitlines()
    if ln.strip()
]
if MAX_COMPOUNDS is not None:
    smiles_lines = smiles_lines[:MAX_COMPOUNDS]

# --- Phase 1: build 3D structures ---
print(f"\nPhase 1: building 3D structures for {len(smiles_lines)} SMILES from {SMI_FILE.name}")
prepared_mols: list[tuple[int, str, Chem.Mol]] = []
n_mol_parse_fail = n_mol_embed_fail = 0

with tqdm(smiles_lines, desc="3D structures", unit="mol") as pbar:
    for line_idx, smi in enumerate(pbar):
        mol_3d = build_3d_mol(smi, seed=0xF00D + line_idx)
        if mol_3d is None:
            probe = Chem.MolFromSmiles(smi)
            if probe is None:
                n_mol_parse_fail += 1
            else:
                n_mol_embed_fail += 1
            pbar.set_postfix(ready=len(prepared_mols), parse=n_mol_parse_fail, embed=n_mol_embed_fail)
            continue
        prepared_mols.append((line_idx, smi, mol_3d))
        pbar.set_postfix(ready=len(prepared_mols), parse=n_mol_parse_fail, embed=n_mol_embed_fail)

print(
    f"Phase 1 done: {len(prepared_mols)} 3D molecules ready, "
    f"{n_mol_parse_fail} invalid SMILES, {n_mol_embed_fail} embed failures "
    f"({len(smiles_lines)} SMILES in total)"
)

prepared_by_line = {line_idx: (smi, mol_3d) for line_idx, smi, mol_3d in prepared_mols}

# --- Phase 2: pharmacophore search ---
results, searcher = run_screening_phase(2, ph, prepared_mols, prepared_by_line)


# --------------------------------------------------------------------------
# PART III - Modify the first pharmacophore by changing the HYBL into a AROM
# --------------------------------------------------------------------------

# Modify pharmacophore
for i, p in enumerate(ph.points):
    if p.func == FuncGroup.HYBL:
        ph.points[i] = PharmacophorePoint(
            p.x,
            p.y,
            p.z,
            FuncGroup.AROM,
            default_alpha(FuncGroup.AROM),
            True,
            p.x,
            p.y,
            p.z + 1.0,
        )

results, searcher = run_screening_phase(3, ph, prepared_mols, prepared_by_line)


# --------------------------------------------------------------------------
# PART IV - Modify the pharmacophore by changing HYBH into HACC and HDON
# --------------------------------------------------------------------------

# Replace each HYBH with two points at the same site (donor AND acceptor required).
new_points: list[PharmacophorePoint] = []
for p in ph.points:
    if p.func == FuncGroup.HYBH:
        new_points.append(
            PharmacophorePoint(
                p.x,
                p.y,
                p.z,
                FuncGroup.HACC,
                default_alpha(FuncGroup.HACC),
                p.has_normal,
                p.nx,
                p.ny,
                p.nz,
            )
        )
        new_points.append(
            PharmacophorePoint(
                p.x,
                p.y,
                p.z,
                FuncGroup.HDON,
                default_alpha(FuncGroup.HDON),
                p.has_normal,
                p.nx,
                p.ny,
                p.nz,
            )
        )
    else:
        new_points.append(p)
ph.points = new_points

results, searcher = run_screening_phase(4, ph, prepared_mols, prepared_by_line)
