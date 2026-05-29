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


def _centre_atoms(mol: Chem.Mol) -> list[Chem.Atom]:
    """Return only the centre pseudo-atoms (skip normal-tip atoms).

    Normal-tip atoms (currently emitted for AROM / AROM_OR_LIPO features) use
    PDB atom names that start with ``+`` or ``-``; centre atoms use a type
    code character.
    """
    out: list[Chem.Atom] = []
    for atom in mol.GetAtoms():
        info = atom.GetMonomerInfo()
        name = info.GetName().strip() if info is not None else ""
        if name.startswith(("+", "-")):
            continue
        out.append(atom)
    return out


def test_query_write_sdf(tmp_path: Path):
    q = _query_with_two_features()
    path = tmp_path / "query.sdf"
    q.write_sdf(path)
    assert path.exists()
    records = list(Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False))
    assert len(records) == 1
    rec = records[0]
    assert rec is not None
    # 1 AROM (centre + 2 tips) + 1 HACC_AND_HDON (centre only) = 4 atoms.
    assert rec.GetNumAtoms() == 4
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
    centres = _centre_atoms(mol)
    assert len(centres) == len(q)
    assert mol.GetNumConformers() == 1
    conf = mol.GetConformer()
    for atom, p in zip(centres, q):
        pos = conf.GetAtomPosition(atom.GetIdx())
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
    # No AROM features, so no normal tips: atom count equals feature count.
    assert mol.GetNumAtoms() == 2
    assert len(_centre_atoms(mol)) == 2
    assert mol.GetProp("kind") == "molecule"
    assert mol.GetProp("types") == "HACC,HDON"
    assert not mol.HasProp("name")
    assert mol.GetProp("_Name") == "pharmacophore"


def test_arom_query_has_symmetric_tip_atoms():
    q = QueryPharmacophore(name="ring")
    q.add_point(
        PharmacophorePoint(
            type=PointType.AROM, center=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)
        )
    )
    mol = q.to_mol()

    assert int(mol.GetProp("num_features")) == 1
    # 1 centre + 2 tips (above and below the plane).
    assert mol.GetNumAtoms() == 3

    centre_idxs = [a.GetIdx() for a in _centre_atoms(mol)]
    assert len(centre_idxs) == 1
    centre_idx = centre_idxs[0]
    conf = mol.GetConformer()
    cpos = conf.GetAtomPosition(centre_idx)
    assert (cpos.x, cpos.y, cpos.z) == pytest.approx((0.0, 0.0, 0.0))

    tip_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in centre_idxs]
    assert len(tip_idxs) == 2
    tip_names = sorted(
        mol.GetAtomWithIdx(ti).GetMonomerInfo().GetName().strip()[0]
        for ti in tip_idxs
    )
    assert tip_names == ["+", "-"]

    tip_positions = [conf.GetAtomPosition(ti) for ti in tip_idxs]
    # Symmetric about the centre: the two tips sum to twice the centre.
    sx = sum(p.x for p in tip_positions)
    sy = sum(p.y for p in tip_positions)
    sz = sum(p.z for p in tip_positions)
    assert (sx, sy, sz) == pytest.approx((0.0, 0.0, 0.0))
    # Each tip lies at unit distance along the stored normal direction.
    for tp in tip_positions:
        dist = ((tp.x) ** 2 + (tp.y) ** 2 + (tp.z - 1.0) ** 2) ** 0.5
        dist_neg = ((tp.x) ** 2 + (tp.y) ** 2 + (tp.z + 1.0) ** 2) ** 0.5
        assert min(dist, dist_neg) == pytest.approx(0.0)

    # Both tips are bonded to the centre.
    bonds = list(mol.GetBonds())
    assert len(bonds) == 2
    bonded_neighbours = {b.GetOtherAtomIdx(centre_idx) for b in bonds}
    assert bonded_neighbours == set(tip_idxs)


def test_arom_query_write_pdb_emits_tip_atoms(tmp_path: Path):
    q = QueryPharmacophore(name="ring")
    q.add_point(
        PharmacophorePoint(
            type=PointType.AROM, center=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)
        )
    )
    path = tmp_path / "ring.pdb"
    q.write_pdb(path)
    text = path.read_text()
    # The centre AROM atom and the two tip atoms all live in residue ARO.
    assert text.count("ARO P") == 3
    # Tip atom names start with + and -.
    assert "+001" in text
    assert "-001" in text
