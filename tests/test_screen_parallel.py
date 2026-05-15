import pytest

pytest.importorskip("rdkit")

from rdkit import Chem
from rdkit.Chem import AllChem

from pypharao import PharmacophoreSearch, pharmacophore_from_molecule, PerceptionOptions


def _embed(smiles: str, seed: int) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    return mol


def test_screen_batch_matches_sequential():
    query = _embed("c1ccccc1O", 0xF00D)
    ref = pharmacophore_from_molecule(query, PerceptionOptions(), conf_id=0)
    mols = [_embed("CO", seed) for seed in (1, 2, 3)]
    searcher = PharmacophoreSearch(ref, use_direction=False)

    expected = {
        i: hit
        for i, mol in enumerate(mols)
        if (hit := searcher.screen(mol)) is not None
    }
    parallel = dict(searcher.screen(mols, n_jobs=1))
    assert parallel == expected


def test_screen_batch_parallel_same_hits():
    query = _embed("c1ccccc1O", 0xF00D)
    ref = pharmacophore_from_molecule(query, PerceptionOptions(), conf_id=0)
    mols = [_embed("CO", seed) for seed in range(6)]
    searcher = PharmacophoreSearch(ref, use_direction=False)

    sequential = dict(searcher.screen(mols, n_jobs=1))
    parallel = dict(searcher.screen(mols, n_jobs=2))
    assert parallel == sequential


def test_screen_no_match_returns_none():
    query = _embed("c1ccccc1O", 0xF00D)
    ref = pharmacophore_from_molecule(query, PerceptionOptions(), conf_id=0)
    unrelated = _embed("C", 99)
    searcher = PharmacophoreSearch(ref, use_direction=False, min_matched_query_features=3)
    assert searcher.screen(unrelated) is None


def test_screen_batch_empty():
    query = _embed("c1ccccc1O", 0xF00D)
    ref = pharmacophore_from_molecule(query, PerceptionOptions(), conf_id=0)
    searcher = PharmacophoreSearch(ref, use_direction=False)
    assert searcher.screen([]) == []


def test_screen_batch_accepts_prepared_mol_tuples():
    query = _embed("c1ccccc1O", 0xF00D)
    ref = pharmacophore_from_molecule(query, PerceptionOptions(), conf_id=0)
    mol = _embed("CO", 1)
    prepared = [(42, "CO", mol)]
    searcher = PharmacophoreSearch(ref, use_direction=False)
    hits = searcher.screen(prepared, n_jobs=1)
    for index, hit in hits:
        assert index == 42
        assert hit.tanimoto >= 0.0


def test_screen_n_jobs_invalid():
    query = _embed("c1ccccc1O", 0xF00D)
    ref = pharmacophore_from_molecule(query, PerceptionOptions(), conf_id=0)
    searcher = PharmacophoreSearch(ref, use_direction=False)
    with pytest.raises(ValueError, match="n_jobs"):
        searcher.screen([_embed("CO", 1)], n_jobs=0)
