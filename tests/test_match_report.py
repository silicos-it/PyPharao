import pytest

pytest.importorskip("rdkit")

from rdkit import Chem
from rdkit.Chem import AllChem

from pypharao import (
    PharmacophoreSearch,
    pharmacophore_from_molecule,
    PerceptionOptions,
    print_match_results,
    sort_match_results,
)


def _embed(smiles: str, seed: int = 0xF00D) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    return mol


def test_print_match_results_single(capsys):
    query = _embed("c1ccccc1O")
    ref = pharmacophore_from_molecule(query, PerceptionOptions(), conf_id=0)
    hit = PharmacophoreSearch(ref, use_direction=False).screen(query)
    assert hit is not None
    print_match_results(hit)
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert lines[0].split("\t")[0] == "index"
    assert "tanimoto" in lines[0].split("\t")
    assert "aligned_mol" in lines[0].split("\t")
    assert len(lines) == 2
    assert "c1ccccc1" in lines[1]
    assert "[H]" not in lines[1]


def test_print_match_results_batch(capsys):
    query = _embed("c1ccccc1O")
    ref = pharmacophore_from_molecule(query, PerceptionOptions(), conf_id=0)
    searcher = PharmacophoreSearch(ref, use_direction=False)
    hits = searcher.screen([query, query], n_jobs=1)
    assert len(hits) == 2
    print_match_results(hits)
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert len(lines) == 3
    assert lines[0].split("\t")[0] == "index"
    assert "db_volume" in lines[0].split("\t")


def test_sort_match_results_descending_and_ascending():
    hits = [(0, _make_result(0.2)), (1, _make_result(0.8)), (2, _make_result(0.5))]
    desc = sort_match_results(hits, sort="descending", key="tanimoto")
    assert [t for _, r in desc for t in [r.tanimoto]] == [0.8, 0.5, 0.2]
    asc = sort_match_results(hits, sort="ascending", key="tanimoto")
    assert [r.tanimoto for _, r in asc] == [0.2, 0.5, 0.8]


def _make_result(tanimoto: float):
    from pypharao import MatchResult

    return MatchResult(1.0, 1.0, 1.0, 0.0, tanimoto=tanimoto)


def test_sort_match_results_does_not_mutate_input():
    hits = [(1, _make_result(0.3)), (0, _make_result(0.9))]
    original = list(hits)
    sort_match_results(hits)
    assert hits == original


def test_print_match_results_none_and_empty(capsys):
    print_match_results(None)
    assert "No match" in capsys.readouterr().out
    print_match_results([])
    assert "No hits" in capsys.readouterr().out


def test_print_match_results_limit(capsys):
    hits = [(i, _make_result(1.0 - i * 0.1)) for i in range(5)]
    print_match_results(hits, limit=2)
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert lines[0] == "Showing 2 of 5 hits."
    assert len(lines) == 4  # note + header + 2 rows
    assert lines[1].split("\t")[0] == "index"
    assert lines[2].split("\t")[0] == "0"
    assert lines[3].split("\t")[0] == "1"


def test_print_match_results_limit_invalid():
    with pytest.raises(ValueError, match="limit"):
        print_match_results([], limit=-1)
