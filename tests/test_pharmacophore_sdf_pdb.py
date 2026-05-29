import pytest

pytest.importorskip("rdkit")

from pathlib import Path

from rdkit import Chem

from pypharao import (
    MoleculePharmacophore,
    PharmacophorePoint,
    PointType,
    QueryPharmacophore,
)


def _query_with_two_features() -> QueryPharmacophore:
    q = QueryPharmacophore(name="phenol")
    q.add_point(
        PharmacophorePoint(
            type=PointType.AROM, center=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)
        )
    )
    q.add_point(
        PharmacophorePoint(type=PointType.HACC_AND_HDON, center=(2.5, 0.0, 0.0))
    )
    return q


def test_query_write_sdf(tmp_path: Path):
    q = _query_with_two_features()
    path = tmp_path / "query.sdf"
    q.write_sdf(path)
    assert path.exists()
    records = list(Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False))
    assert len(records) == 1
    rec = records[0]
    assert rec is not None
    assert rec.GetNumAtoms() == len(q)
    assert rec.GetProp("kind") == "query"
    assert rec.GetProp("name") == "phenol"
    assert rec.GetProp("types") == "AROM,HACC_AND_HDON"
    assert int(rec.GetProp("num_features")) == 2


def test_query_write_pdb_writes_residue_codes(tmp_path: Path):
    q = _query_with_two_features()
    path = tmp_path / "query.pdb"
    q.write_pdb(path)
    text = path.read_text()
    assert "HETATM" in text
    assert "ARO P" in text
    assert "DAC P" in text
    assert text.rstrip().endswith("END")


def test_molecule_pharmacophore_write_sdf_pdb(tmp_path: Path):
    m = MoleculePharmacophore()
    m.add_point(PharmacophorePoint(type=PointType.HACC, center=(1.0, 0.0, 0.0)))
    m.add_point(PharmacophorePoint(type=PointType.HDON, center=(0.0, 1.0, 0.0)))
    sdf = tmp_path / "mol.sdf"
    pdb = tmp_path / "mol.pdb"
    m.write_sdf(sdf)
    m.write_pdb(pdb)
    records = list(Chem.SDMolSupplier(str(sdf), removeHs=False, sanitize=False))
    assert len(records) == 1
    assert records[0].GetProp("kind") == "molecule"
    text = pdb.read_text()
    assert "HAC P" in text
    assert "HDO P" in text


def test_write_sdf_name_override(tmp_path: Path):
    q = _query_with_two_features()
    path = tmp_path / "named.sdf"
    q.write_sdf(path, name="custom-title")
    first_line = path.read_text().splitlines()[0]
    assert first_line == "custom-title"


def test_write_pdb_name_override(tmp_path: Path):
    q = _query_with_two_features()
    path = tmp_path / "named.pdb"
    q.write_pdb(path, name="custom-title")
    text = path.read_text()
    assert "COMPND" in text
    assert "custom-title" in text


def test_query_to_mol_returns_rdkit_mol_with_metadata():
    q = _query_with_two_features()
    mol = q.to_mol()
    assert isinstance(mol, Chem.Mol)
    assert mol.GetNumAtoms() == len(q)
    assert mol.GetNumConformers() == 1
    conf = mol.GetConformer()
    for i, p in enumerate(q):
        pos = conf.GetAtomPosition(i)
        assert (pos.x, pos.y, pos.z) == pytest.approx((p.x, p.y, p.z))
    assert mol.GetProp("_Name") == "phenol"
    assert mol.GetProp("kind") == "query"
    assert mol.GetProp("name") == "phenol"
    assert mol.GetProp("types") == "AROM,HACC_AND_HDON"
    assert int(mol.GetProp("num_features")) == 2


def test_query_to_mol_name_override_and_default_title():
    q_named = _query_with_two_features()
    assert q_named.to_mol().GetProp("_Name") == "phenol"
    assert q_named.to_mol(name="custom-title").GetProp("_Name") == "custom-title"

    q_unnamed = QueryPharmacophore()
    q_unnamed.add_point(PharmacophorePoint(type=PointType.HACC, center=(0.0, 0.0, 0.0)))
    assert q_unnamed.to_mol().GetProp("_Name") == "pharmacophore"


def test_molecule_pharmacophore_to_mol():
    m = MoleculePharmacophore()
    m.add_point(PharmacophorePoint(type=PointType.HACC, center=(1.0, 0.0, 0.0)))
    m.add_point(PharmacophorePoint(type=PointType.HDON, center=(0.0, 1.0, 0.0)))
    mol = m.to_mol()
    assert isinstance(mol, Chem.Mol)
    assert mol.GetNumAtoms() == 2
    assert mol.GetProp("kind") == "molecule"
    assert mol.GetProp("types") == "HACC,HDON"
    assert not mol.HasProp("name")
    assert mol.GetProp("_Name") == "pharmacophore"
