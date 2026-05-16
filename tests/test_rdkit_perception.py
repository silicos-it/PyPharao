import pytest

pytest.importorskip("rdkit")

from rdkit import Chem
from rdkit.Chem import AllChem

from pypharao import (
    MoleculePharmacophore,
    MoleculePharmacophorePerception,
    PointType,
    QueryPharmacophore,
    QueryPharmacophorePerception,
    molecule_pharmacophore_from_molecule,
    query_pharmacophore_from_molecule,
    query_pharmacophore_from_protein,
)


def _embed(smi: str, seed: int = 0xF00D) -> Chem.Mol:
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    assert AllChem.EmbedMolecule(m, randomSeed=seed) == 0
    return m


def test_query_from_methanol_yields_query_subclass_and_compound_type():
    mol = _embed("CO")
    q = query_pharmacophore_from_molecule(mol)
    assert isinstance(q, QueryPharmacophore)
    types = {p.type for p in q}
    # CO's oxygen is both a donor and acceptor at the same atom → HACC&HDON
    assert PointType.HACC_AND_HDON in types
    # The merge removes the originals
    assert PointType.HACC not in types
    assert PointType.HDON not in types


def test_query_from_phenol_keeps_arom_and_compound():
    mol = _embed("c1ccccc1O")
    q = query_pharmacophore_from_molecule(mol)
    types = {p.type for p in q}
    assert PointType.AROM in types
    assert PointType.HACC_AND_HDON in types


def test_query_perception_disabling_compound_keeps_originals():
    mol = _embed("CO")
    perception = QueryPharmacophorePerception()
    perception.disable(PointType.HACC_AND_HDON)
    q = query_pharmacophore_from_molecule(mol, perception)
    types = {p.type for p in q}
    assert PointType.HACC in types
    assert PointType.HDON in types
    assert PointType.HACC_AND_HDON not in types


def test_molecule_perception_returns_molecule_subclass():
    mol = _embed("CO")
    db = molecule_pharmacophore_from_molecule(mol)
    assert isinstance(db, MoleculePharmacophore)
    types = {p.type for p in db}
    assert PointType.HACC_AND_HDON in types
    assert PointType.HACC not in types
    assert PointType.HDON not in types


def test_no_arom_lipo_merging():
    mol = _embed("c1ccccc1O")
    q = query_pharmacophore_from_molecule(mol)
    types = {p.type for p in q}
    assert PointType.AROM_OR_LIPO not in types


def test_molecule_perception_subclass_perception_allowed_in_screen():
    mol = _embed("CO")
    perception = MoleculePharmacophorePerception()
    perception.disable(PointType.HACC_AND_HDON)
    db = molecule_pharmacophore_from_molecule(mol, perception)
    types = {p.type for p in db}
    assert PointType.HACC in types
    assert PointType.HDON in types
    assert PointType.HACC_AND_HDON not in types


def test_query_from_protein_is_stub():
    with pytest.raises(NotImplementedError):
        query_pharmacophore_from_protein()
