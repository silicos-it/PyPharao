import pytest

pytest.importorskip("rdkit")

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from pypharao import (
    MatchResult,
    PharmacophoreSearch,
    print_match_results,
    query_pharmacophore_from_molecule,
    sort_match_results,
    write_hits_sdf,
)


def _embed(smiles: str, seed: int = 0xF00D) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    return mol


def test_print_match_results_single_hit(capsys):
    query = _embed("c1ccccc1O")
    ref = query_pharmacophore_from_molecule(query)
    hits = PharmacophoreSearch(query=ref, use_direction=False).screen(
        query, progress=False
    )
    assert len(hits) == 1
    print_match_results(hits)
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    headers = lines[0].split("\t")
    assert headers[0] == "index"
    assert "conf_id" in headers
    assert "tanimoto" in headers
    assert len(lines) == 2
    assert "c1ccccc1" in lines[1]


def test_print_match_results_empty(capsys):
    print_match_results([])
    assert "No hits" in capsys.readouterr().out


def test_print_match_results_limit(capsys):
    hits = [(i, MatchResult(1.0, 1.0, 1.0, 0.0, tanimoto=1.0 - i * 0.1)) for i in range(5)]
    print_match_results(hits, limit=2)
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert lines[0] == "Showing 2 of 5 hits."
    assert len(lines) == 4
    assert lines[1].split("\t")[0] == "index"


def test_print_match_results_limit_invalid():
    with pytest.raises(ValueError, match="limit"):
        print_match_results([], limit=-1)


def _make_result(tanimoto: float) -> MatchResult:
    return MatchResult(1.0, 1.0, 1.0, 0.0, tanimoto=tanimoto)


def test_sort_match_results_descending_and_ascending():
    hits = [(0, _make_result(0.2)), (1, _make_result(0.8)), (2, _make_result(0.5))]
    desc = sort_match_results(hits, sort="descending", key="tanimoto")
    assert [r.tanimoto for _, r in desc] == [0.8, 0.5, 0.2]
    asc = sort_match_results(hits, sort="ascending", key="tanimoto")
    assert [r.tanimoto for _, r in asc] == [0.2, 0.5, 0.8]


def test_sort_match_results_does_not_mutate_input():
    hits = [(1, _make_result(0.3)), (0, _make_result(0.9))]
    original = list(hits)
    sort_match_results(hits)
    assert hits == original


def test_write_hits_sdf_roundtrip(tmp_path: Path):
    query = _embed("c1ccccc1O")
    ref = query_pharmacophore_from_molecule(query)
    hits = PharmacophoreSearch(query=ref, use_direction=False).screen(
        query, progress=False
    )
    assert hits
    path = tmp_path / "hits.sdf"
    n = write_hits_sdf(hits, path)
    assert n == 1
    records = list(Chem.SDMolSupplier(str(path), removeHs=False))
    assert len(records) == 1
    rec = records[0]
    assert rec.HasProp("tanimoto")
    assert rec.HasProp("conf_id")
    assert rec.HasProp("index")
    assert float(rec.GetProp("tanimoto")) > 0.9


def test_write_hits_sdf_skips_missing_aligned_mol(tmp_path: Path, capsys):
    hits = [(0, _make_result(0.5))]
    path = tmp_path / "empty.sdf"
    n = write_hits_sdf(hits, path)
    assert n == 0
