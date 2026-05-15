import pytest

pytest.importorskip("rdkit")

from rdkit import Chem
from rdkit.Chem import AllChem

from pypharao import (
    FuncGroup,
    PerceptionOptions,
    pharmacophore_from_molecule,
    pharmacophore_from_rdkit,
)


def test_pharmacophore_from_methanol():
    mol = Chem.MolFromSmiles("CO")
    mol = Chem.AddHs(mol)
    assert AllChem.EmbedMolecule(mol, randomSeed=0xf00d) == 0
    p = pharmacophore_from_molecule(mol, PerceptionOptions(), conf_id=0)
    funcs = {pt.func for pt in p.points}
    hbond = {FuncGroup.HDON, FuncGroup.HACC, FuncGroup.HYBH}
    assert funcs & hbond


def test_pharmacophore_from_phenol_explicit_hs():
    mol = Chem.MolFromSmiles("c1ccccc1O")
    mol = Chem.AddHs(mol)
    assert AllChem.EmbedMolecule(mol, randomSeed=0xF00D) == 0
    opts = PerceptionOptions(hybl=False, hybh=False)
    p = pharmacophore_from_molecule(mol, opts, conf_id=0)
    funcs = {pt.func for pt in p.points}
    assert FuncGroup.HDON in funcs
    assert FuncGroup.HACC in funcs


def test_pharmacophore_from_phenol_hybh_with_explicit_hs():
    mol = Chem.MolFromSmiles("c1ccccc1O")
    mol = Chem.AddHs(mol)
    assert AllChem.EmbedMolecule(mol, randomSeed=0xF00D) == 0
    p = pharmacophore_from_molecule(mol, PerceptionOptions(hybl=False), conf_id=0)
    funcs = {pt.func for pt in p.points}
    assert FuncGroup.HYBH in funcs


def test_pharmacophore_from_rdkit_alias():
    assert pharmacophore_from_rdkit is pharmacophore_from_molecule
