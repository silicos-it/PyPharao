import math

import pytest

pytest.importorskip("rdkit")

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from pypharao import (
    DEFAULT_SIGMA,
    MoleculePharmacophore,
    MoleculePharmacophorePerception,
    PointType,
    QueryPharmacophore,
    QueryPharmacophorePerception,
    add_excluded_volume,
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
    # CO's oxygen is both a donor and acceptor at the same atom → HACC_AND_HDON
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


def test_aromatic_ring_emits_only_arom_not_lipo():
    """An aromatic ring is reported as AROM; the same ring must not also become LIPO."""
    mol = _embed("c1ccccc1")
    q = query_pharmacophore_from_molecule(mol)
    types = [p.type for p in q]
    assert PointType.AROM in types
    assert PointType.LIPO not in types


def test_naphthalene_emits_two_arom_no_lipo():
    """Each aromatic ring of naphthalene becomes AROM; neither contributes a LIPO."""
    mol = _embed("c1ccc2ccccc2c1")
    q = query_pharmacophore_from_molecule(mol)
    types = [p.type for p in q]
    assert types.count(PointType.AROM) == 2
    assert PointType.LIPO not in types


def test_substituted_aromatic_ring_emits_arom_not_lipo():
    """Substitutents on an aromatic ring still leave it as AROM only, not LIPO."""
    mol = _embed("Cc1ccc(O)cc1")
    q = query_pharmacophore_from_molecule(mol)
    types = [p.type for p in q]
    assert PointType.AROM in types
    assert PointType.LIPO not in types
    assert PointType.HACC_AND_HDON in types


def test_saturated_lipophilic_ring_still_emits_lipo():
    """Non-aromatic lipophilic rings continue to be perceived as LIPO."""
    mol = _embed("CCCCCC1CCCCC1")
    q = query_pharmacophore_from_molecule(mol)
    types = [p.type for p in q]
    assert PointType.LIPO in types
    assert PointType.AROM not in types


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


def _heavy_atom_centers(mol: Chem.Mol):
    conf = mol.GetConformer()
    pt = Chem.GetPeriodicTable()
    return [
        (
            conf.GetAtomPosition(a.GetIdx()).x,
            conf.GetAtomPosition(a.GetIdx()).y,
            conf.GetAtomPosition(a.GetIdx()).z,
            pt.GetRvdw(a.GetAtomicNum()),
        )
        for a in mol.GetAtoms()
        if a.GetAtomicNum() != 1
    ]


def test_add_excluded_volume_appends_excl_in_shell():
    mol = _embed("c1ccccc1O")
    q = query_pharmacophore_from_molecule(mol)
    before = len(q)
    n = add_excluded_volume(
        mol, q, shell_inner=1.0, shell_outer=2.5, spacing=1.5
    )
    assert n > 0
    assert len(q) == before + n
    excl = [p for p in q if p.type == PointType.EXCL]
    assert len(excl) == n
    # default sigma is the Pharao EXCL default
    assert {p.sigma for p in excl} == {DEFAULT_SIGMA[PointType.EXCL]}
    # every EXCL must sit in [shell_inner, shell_outer] from the closest vdW surface
    centers = _heavy_atom_centers(mol)
    for p in excl:
        surf = min(
            ((p.x - cx) ** 2 + (p.y - cy) ** 2 + (p.z - cz) ** 2) ** 0.5 - r
            for cx, cy, cz, r in centers
        )
        assert 1.0 - 1e-9 <= surf <= 2.5 + 1e-9


def test_add_excluded_volume_custom_sigma_and_feature_clearance():
    mol = _embed("c1ccccc1O")
    q = query_pharmacophore_from_molecule(mol)
    feature_centers = [(p.x, p.y, p.z) for p in q if p.type != PointType.EXCL]
    n = add_excluded_volume(
        mol,
        q,
        sigma=2.0,
        shell_inner=1.0,
        shell_outer=2.5,
        spacing=1.5,
        feature_clearance=1.5,
    )
    excl = [p for p in q if p.type == PointType.EXCL]
    assert len(excl) == n
    assert {p.sigma for p in excl} == {2.0}
    for p in excl:
        for fx, fy, fz in feature_centers:
            d2 = (p.x - fx) ** 2 + (p.y - fy) ** 2 + (p.z - fz) ** 2
            assert d2 >= 1.5 * 1.5 - 1e-9


def test_add_excluded_volume_rejected_on_molecule_pharmacophore():
    mol = _embed("c1ccccc1O")
    db = MoleculePharmacophore()
    with pytest.raises(ValueError):
        add_excluded_volume(mol, db)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shell_inner": -0.1},
        {"shell_inner": 2.0, "shell_outer": 1.0},
        {"spacing": 0.0},
    ],
)
def test_add_excluded_volume_validates_parameters(kwargs):
    mol = _embed("c1ccccc1O")
    q = QueryPharmacophore()
    with pytest.raises(ValueError):
        add_excluded_volume(mol, q, **kwargs)


def test_add_excluded_volume_conf_id_out_of_range_raises():
    mol = _embed("CO")
    q = QueryPharmacophore()
    with pytest.raises(ValueError, match="out of range"):
        add_excluded_volume(mol, q, conf_id=2)


def test_add_excluded_volume_empty_mol_sequence_raises():
    with pytest.raises(ValueError, match="at least one"):
        add_excluded_volume([], QueryPharmacophore())


def test_add_excluded_volume_accepts_list_of_molecules():
    m1 = _embed("c1ccccc1O", seed=1)
    m2 = _embed("c1ccccc1O", seed=2)
    rdMolAlign.AlignMol(m2, m1)
    q = query_pharmacophore_from_molecule(m1)
    before = len(q)
    n = add_excluded_volume([m1, m2], q, shell_inner=1.0, shell_outer=2.5, spacing=1.5)
    assert n > 0
    assert len(q) == before + n


def test_add_excluded_volume_max_excl_caps_count():
    mol = _embed("c1ccccc1O")
    q = query_pharmacophore_from_molecule(mol)
    n = add_excluded_volume(
        mol,
        q,
        shell_inner=1.0,
        shell_outer=2.5,
        spacing=1.5,
        max_excl=24,
    )
    assert 0 < n <= 24


def test_add_excluded_volume_pairwise_spacing():
    """Emitted EXCL centres are at least ``spacing`` apart (greedy thinning)."""
    mol = _embed("c1ccccc1O")
    q = query_pharmacophore_from_molecule(mol)
    spacing = 1.5
    add_excluded_volume(
        mol,
        q,
        shell_inner=1.0,
        shell_outer=2.5,
        spacing=spacing,
        max_excl=0,
    )
    excl = [(p.x, p.y, p.z) for p in q if p.type == PointType.EXCL]
    assert len(excl) >= 2
    for i in range(len(excl)):
        for j in range(i + 1, len(excl)):
            d = math.dist(excl[i], excl[j])
            assert d >= spacing - 1e-6, (d, spacing)


def _union_surface_distance(
    mols: list[Chem.Mol],
    conf_id: int | None,
    x: float,
    y: float,
    z: float,
) -> float:
    """Union vdW surface distance at a point (matches ``add_excluded_volume`` geometry)."""
    pt = Chem.GetPeriodicTable()
    best = math.inf
    for mol in mols:
        if conf_id is None:
            cids = range(mol.GetNumConformers())
        else:
            cids = [conf_id]
        for cid in cids:
            conf = mol.GetConformer(cid)
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 1:
                    continue
                pos = conf.GetAtomPosition(atom.GetIdx())
                r = pt.GetRvdw(atom.GetAtomicNum())
                d = math.hypot(x - pos.x, y - pos.y, z - pos.z) - r
                best = min(best, d)
    return best


def test_add_excluded_volume_union_shell_matches_combined_atoms():
    """Each EXCL lies in the shell band vs the union of selected heavy atoms."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=2, randomSeed=77, clearConfs=True)
    assert len(cids) == 2
    q0 = QueryPharmacophore()
    n0 = add_excluded_volume(
        mol, q0, shell_inner=1.0, shell_outer=2.5, spacing=1.5, conf_id=0
    )
    q_all = QueryPharmacophore()
    n_all = add_excluded_volume(
        mol, q_all, shell_inner=1.0, shell_outer=2.5, spacing=1.5, conf_id=None
    )
    assert n0 > 0 and n_all > 0
    mols = [mol]
    for collection, n, cid in (
        (q0, n0, 0),
        (q_all, n_all, None),
    ):
        excl = [p for p in collection if p.type == PointType.EXCL]
        assert len(excl) == n
        for p in excl:
            sd_u = _union_surface_distance(mols, cid, p.x, p.y, p.z)
            assert 1.0 - 1e-9 <= sd_u <= 2.5 + 1e-9
