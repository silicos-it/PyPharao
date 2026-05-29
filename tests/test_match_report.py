import pytest

pytest.importorskip("rdkit")

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from pypharao import (
    MatchResult,
    PharmacophoreSearch,
    pharmacophore_to_mol,
    print_match_results,
    query_pharmacophore_from_molecule,
    sort_match_results,
    write_hits_pdb,
    write_hits_sdf,
)


def _embed(smiles: str, seed: int = 0xF00D) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    return mol


def _centre_atoms(mol: Chem.Mol) -> list[Chem.Atom]:
    """Filter out normal-tip pseudo-atoms from a pharmacophore Mol.

    Normal-tip atoms (currently emitted for AROM / AROM_OR_LIPO features) use
    PDB atom names that start with ``+`` or ``-``.
    """
    out: list[Chem.Atom] = []
    for atom in mol.GetAtoms():
        info = atom.GetMonomerInfo()
        name = info.GetName().strip() if info is not None else ""
        if name.startswith(("+", "-")):
            continue
        out.append(atom)
    return out


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


def test_pharmacophore_to_mol_round_trip():
    query = _embed("c1ccccc1O")
    ref = query_pharmacophore_from_molecule(query, name="phenol")
    mol = pharmacophore_to_mol(ref)
    centres = _centre_atoms(mol)
    assert len(centres) == len(ref)
    assert mol.GetProp("kind") == "query"
    assert mol.GetProp("name") == "phenol"
    assert int(mol.GetProp("num_features")) == len(ref)
    types = mol.GetProp("types").split(",")
    assert types == [p.type.value for p in ref]
    conf = mol.GetConformer()
    for atom, p in zip(centres, ref):
        pos = conf.GetAtomPosition(atom.GetIdx())
        assert pos.x == pytest.approx(p.x)
        assert pos.y == pytest.approx(p.y)
        assert pos.z == pytest.approx(p.z)


def test_write_hits_sdf_with_pharmacophore_first(tmp_path: Path):
    query = _embed("c1ccccc1O")
    ref = query_pharmacophore_from_molecule(query, name="phenol")
    hits = PharmacophoreSearch(query=ref, use_direction=False).screen(
        query, progress=False
    )
    assert hits
    path = tmp_path / "hits.sdf"
    n = write_hits_sdf(hits, path, pharmacophore=ref)
    assert n == len(hits) + 1
    records = list(Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False))
    assert len(records) == n
    first = records[0]
    assert first is not None
    assert first.HasProp("kind")
    assert first.GetProp("kind") == "query"
    assert first.GetProp("name") == "phenol"
    assert int(first.GetProp("num_features")) == len(ref)
    # SDF round-trip loses PDB monomer info, so we can't use _centre_atoms here;
    # instead check the expected total atom count (centres + 2 tips per AROM).
    n_arom = sum(1 for p in ref if p.has_normal)
    assert first.GetNumAtoms() == len(ref) + 2 * n_arom
    second = records[1]
    assert second is not None
    assert second.HasProp("tanimoto")


def test_write_hits_pdb_basic_round_trip(tmp_path: Path):
    query = _embed("c1ccccc1O")
    ref = query_pharmacophore_from_molecule(query, name="phenol")
    hits = PharmacophoreSearch(query=ref, use_direction=False).screen(
        query, progress=False
    )
    assert hits
    path = tmp_path / "hits.pdb"
    n = write_hits_pdb(hits, path)
    assert n == len(hits)
    text = path.read_text()
    assert text.count("MODEL") == n
    assert text.count("ENDMDL") == n
    assert text.rstrip().endswith("END")


def test_write_hits_pdb_with_pharmacophore_first(tmp_path: Path):
    query = _embed("c1ccccc1O")
    ref = query_pharmacophore_from_molecule(query, name="phenol")
    hits = PharmacophoreSearch(query=ref, use_direction=False).screen(
        query, progress=False
    )
    assert hits
    path = tmp_path / "hits.pdb"
    n = write_hits_pdb(hits, path, pharmacophore=ref)
    assert n == len(hits) + 1
    text = path.read_text()
    assert text.count("MODEL") == n
    assert text.count("ENDMDL") == n
    first_model_end = text.find("ENDMDL")
    first_model = text[: first_model_end]
    # The pharmacophore MODEL should come first; AROM gets a residue name 'ARO'
    # and HACC_AND_HDON gets 'DAC'.
    assert "ARO P" in first_model
    assert "DAC P" in first_model


def test_write_hits_pdb_skips_missing_aligned_mol(tmp_path: Path):
    hits = [(0, _make_result(0.5))]
    path = tmp_path / "empty.pdb"
    n = write_hits_pdb(hits, path)
    assert n == 0
    text = path.read_text()
    assert "MODEL" not in text
    assert text.rstrip().endswith("END")


def test_pharmacophore_to_mol_rejects_non_pharmacophore():
    from pypharao import PharmacophoreSearch, query_pharmacophore_from_molecule

    query = _embed("c1ccccc1O")
    ref = query_pharmacophore_from_molecule(query)
    searcher = PharmacophoreSearch(query=ref)
    with pytest.raises(TypeError, match="Pharmacophore"):
        pharmacophore_to_mol(searcher)


def test_write_hits_pdb_pharmacophore_only(tmp_path: Path):
    query = _embed("c1ccccc1O")
    ref = query_pharmacophore_from_molecule(query, name="phenol")
    path = tmp_path / "ph.pdb"
    n = write_hits_pdb([], path, pharmacophore=ref)
    assert n == 1
    text = path.read_text()
    assert text.count("MODEL") == 1
    assert text.count("ENDMDL") == 1
