import pytest

pytest.importorskip("rdkit")

from rdkit import Chem
from rdkit.Chem import AllChem

from pypharao import FuncGroup, PerceptionOptions, pharmacophore_from_rdkit


def test_pharmacophore_from_methanol():
    mol = Chem.MolFromSmiles("CO")
    mol = Chem.AddHs(mol)
    assert AllChem.EmbedMolecule(mol, randomSeed=0xf00d) == 0
    p = pharmacophore_from_rdkit(mol, PerceptionOptions(), conf_id=0)
    funcs = {pt.func for pt in p.points}
    assert FuncGroup.HDON in funcs or FuncGroup.HACC in funcs
