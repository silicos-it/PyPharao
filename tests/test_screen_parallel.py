import pytest

pytest.importorskip("rdkit")

from rdkit import Chem
from rdkit.Chem import AllChem

from pypharao import (
    PharmacophoreSearch,
    PointType,
    query_pharmacophore_from_molecule,
)


def _embed(smiles: str, seed: int) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    return mol


def _embed_multi(smiles: str, n_confs: int, seed: int) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, randomSeed=seed)
    return mol


def test_screen_single_molecule_returns_list():
    query = _embed("c1ccccc1O", 0xF00D)
    ref = query_pharmacophore_from_molecule(query)
    searcher = PharmacophoreSearch(query=ref, use_direction=False)
    hits = searcher.screen(query, progress=False)
    assert isinstance(hits, list)
    assert len(hits) == 1
    index, match = hits[0]
    assert index == 0
    assert match.conf_id == 0
    assert match.tanimoto > 0.9


def test_screen_batch_returns_indexed_list():
    query = _embed("c1ccccc1O", 0xF00D)
    ref = query_pharmacophore_from_molecule(query)
    mols = [_embed("c1ccccc1O", seed) for seed in (1, 2, 3)]
    searcher = PharmacophoreSearch(query=ref, use_direction=False)
    hits = searcher.screen(mols, n_jobs=1, progress=False)
    assert isinstance(hits, list)
    indexes = sorted(i for i, _ in hits)
    assert all(0 <= i < 3 for i in indexes)


def test_screen_batch_parallel_matches_sequential():
    query = _embed("c1ccccc1O", 0xF00D)
    ref = query_pharmacophore_from_molecule(query)
    mols = [_embed("c1ccccc1O", seed) for seed in range(4)]
    searcher = PharmacophoreSearch(query=ref, use_direction=False)
    sequential = sorted(searcher.screen(mols, n_jobs=1, progress=False))
    parallel = sorted(searcher.screen(mols, n_jobs=2, progress=False))
    assert [i for i, _ in sequential] == [i for i, _ in parallel]


def test_screen_no_match_returns_empty_list():
    query = _embed("c1ccccc1O", 0xF00D)
    ref = query_pharmacophore_from_molecule(query)
    unrelated = _embed("C", 99)
    searcher = PharmacophoreSearch(
        query=ref, use_direction=False
    )
    hits = searcher.screen(unrelated, progress=False)
    assert hits == []


def test_screen_batch_empty_input():
    query = _embed("c1ccccc1O", 0xF00D)
    ref = query_pharmacophore_from_molecule(query)
    searcher = PharmacophoreSearch(query=ref, use_direction=False)
    assert searcher.screen([], progress=False) == []


def test_screen_accepts_prepared_tuples_index_passthrough():
    query = _embed("c1ccccc1O", 0xF00D)
    ref = query_pharmacophore_from_molecule(query)
    mol = _embed("c1ccccc1O", 1)
    prepared = [(42, "c1ccccc1O", mol)]
    searcher = PharmacophoreSearch(query=ref, use_direction=False)
    hits = searcher.screen(prepared, n_jobs=1, progress=False)
    assert len(hits) == 1
    assert hits[0][0] == 42


def test_screen_keep_best_vs_all_multi_conformer():
    query = _embed("c1ccccc1O", 0xF00D)
    ref = query_pharmacophore_from_molecule(query)
    target = _embed_multi("c1ccccc1O", n_confs=3, seed=0xBEEF)
    searcher = PharmacophoreSearch(query=ref, use_direction=False)
    best = searcher.screen(target, keep="best", progress=False)
    all_hits = searcher.screen(target, keep="all", progress=False)
    assert len(best) == 1
    assert len(all_hits) >= len(best)
    if len(all_hits) > 1:
        scores = [m.tanimoto for _, m in all_hits]
        assert best[0][1].tanimoto == max(scores)


def test_screen_conformations_limit_int():
    query = _embed("c1ccccc1O", 0xF00D)
    ref = query_pharmacophore_from_molecule(query)
    target = _embed_multi("c1ccccc1O", n_confs=3, seed=0xBEEF)
    searcher = PharmacophoreSearch(query=ref, use_direction=False)
    hits_all = searcher.screen(target, conformations=2, keep="all", progress=False)
    conf_ids = {m.conf_id for _, m in hits_all}
    assert conf_ids.issubset({0, 1})


def test_screen_invalid_metric_raises():
    query = _embed("c1ccccc1O", 0xF00D)
    ref = query_pharmacophore_from_molecule(query)
    searcher = PharmacophoreSearch(query=ref, use_direction=False)
    with pytest.raises(ValueError, match="Unknown metric"):
        searcher.screen(query, metric="nope", progress=False)


def test_screen_invalid_n_jobs_raises():
    query = _embed("c1ccccc1O", 0xF00D)
    ref = query_pharmacophore_from_molecule(query)
    searcher = PharmacophoreSearch(query=ref, use_direction=False)
    with pytest.raises(ValueError, match="n_jobs"):
        searcher.screen([_embed("c1ccccc1O", 1)], n_jobs=-1, progress=False)
