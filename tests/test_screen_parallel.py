import pytest

pytest.importorskip("rdkit")

from rdkit import Chem
from rdkit.Chem import AllChem

from pypharao import (
    PharmacophorePoint,
    PharmacophoreSearch,
    PointType,
    QueryPharmacophore,
    add_excluded_volume,
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


def test_screen_min_matches_zero_is_auto():
    query = _embed("c1ccccc1O", 0xF00D)
    ref = query_pharmacophore_from_molecule(query)
    searcher = PharmacophoreSearch(query=ref, use_direction=False)
    explicit = searcher.screen(query, min_matches=len(ref), progress=False)
    auto_default = searcher.screen(query, progress=False)
    auto_zero = searcher.screen(query, min_matches=0, progress=False)
    assert len(auto_default) == len(auto_zero) == len(explicit) == 1
    assert auto_default[0][1].tanimoto == auto_zero[0][1].tanimoto
    assert auto_zero[0][1].tanimoto == explicit[0][1].tanimoto


def test_screen_min_matches_negative_raises():
    query = _embed("c1ccccc1O", 0xF00D)
    ref = query_pharmacophore_from_molecule(query)
    searcher = PharmacophoreSearch(query=ref, use_direction=False)
    with pytest.raises(ValueError, match=">= 0"):
        searcher.screen(query, min_matches=-1, progress=False)


def test_screen_min_matches_too_large_raises():
    query = _embed("c1ccccc1O", 0xF00D)
    ref = query_pharmacophore_from_molecule(query)
    searcher = PharmacophoreSearch(query=ref, use_direction=False)
    too_large = len(ref) + 100
    with pytest.raises(ValueError, match="exceeds matchable query points"):
        searcher.screen(query, min_matches=too_large, progress=False)


# ---------------------------------------------------------------------------
# Hard EXCL atom-clash filter
# ---------------------------------------------------------------------------
#
# The criterion is geometric:
#     dist(atom_center, EXCL_center) < vdW(atom) + excl_clash_radius.
# The default ``excl_clash_radius=0`` rejects whenever the EXCL marker falls
# inside any heavy-atom vdW sphere. The EXCL's Gaussian ``sigma`` only feeds
# the soft Pharao penalty and does **not** affect this criterion.


def _phenol_query_with_excl_at_atom(idx: int):
    """Return (mol, query) where query has phenol's features plus an EXCL at heavy atom ``idx``."""
    mol = _embed("c1ccccc1O", 0xF00D)
    query = query_pharmacophore_from_molecule(mol)
    pos = mol.GetConformer().GetAtomPosition(idx)
    query.add_point(
        PharmacophorePoint(
            type=PointType.EXCL,
            center=(pos.x, pos.y, pos.z),
        )
    )
    return mol, query


def test_hard_excl_filter_rejects_atom_clash_by_default():
    """Self-screen fails when an EXCL sits exactly on a heavy atom of the candidate."""
    mol, query = _phenol_query_with_excl_at_atom(idx=0)
    searcher = PharmacophoreSearch(query=query, use_direction=False)
    hits = searcher.screen(mol, progress=False, n_jobs=1)
    assert hits == []


def test_hard_excl_filter_disabled_lets_clashing_hit_through():
    """Opt-out: with excl_hard_filter=False the same self-screen yields a hit again."""
    mol, query = _phenol_query_with_excl_at_atom(idx=0)
    searcher = PharmacophoreSearch(
        query=query, use_direction=False, excl_hard_filter=False
    )
    hits = searcher.screen(mol, progress=False, n_jobs=1)
    assert len(hits) == 1


def test_hard_excl_filter_clear_of_vdw_passes():
    """An EXCL outside every heavy atom's vdW sphere does not trigger the filter."""
    mol = _embed("c1ccccc1O", 0xF00D)
    query = query_pharmacophore_from_molecule(mol)
    # Place the EXCL 5 A beyond a phenol heavy atom along x — well outside any vdW sphere.
    pos = mol.GetConformer().GetAtomPosition(0)
    query.add_point(
        PharmacophorePoint(
            type=PointType.EXCL,
            center=(pos.x + 5.0, pos.y, pos.z),
        )
    )
    searcher = PharmacophoreSearch(query=query, use_direction=False)
    hits = searcher.screen(mol, progress=False, n_jobs=1)
    assert len(hits) == 1


def test_hard_excl_filter_extra_radius_tightens_check():
    """Setting excl_clash_radius enlarges the forbidden buffer around each EXCL."""
    mol = _embed("c1ccccc1O", 0xF00D)
    query = query_pharmacophore_from_molecule(mol)
    pos = mol.GetConformer().GetAtomPosition(0)
    # 2.5 A from a heavy atom is clear of any vdW radius (~1.7 A) by default…
    query.add_point(
        PharmacophorePoint(
            type=PointType.EXCL,
            center=(pos.x + 2.5, pos.y, pos.z),
        )
    )
    default = PharmacophoreSearch(query=query, use_direction=False)
    padded = PharmacophoreSearch(
        query=query, use_direction=False, excl_clash_radius=1.5
    )
    assert len(default.screen(mol, progress=False, n_jobs=1)) == 1
    # …but with a 1.5 A buffer the threshold becomes vdW + 1.5 > 2.5 -> clash.
    assert padded.screen(mol, progress=False, n_jobs=1) == []


def test_hard_excl_filter_negative_radius_raises():
    query = QueryPharmacophore()
    with pytest.raises(ValueError, match="excl_clash_radius"):
        PharmacophoreSearch(query=query, excl_clash_radius=-0.1)


def test_hard_excl_filter_parallel_matches_sequential():
    """Hard filter must apply identically in the sequential and parallel paths."""
    mol, query = _phenol_query_with_excl_at_atom(idx=0)
    mols = [mol, _embed("c1ccccc1O", 1), _embed("c1ccccc1O", 2)]
    searcher = PharmacophoreSearch(query=query, use_direction=False)
    seq = sorted(i for i, _ in searcher.screen(mols, n_jobs=1, progress=False))
    par = sorted(i for i, _ in searcher.screen(mols, n_jobs=2, progress=False))
    assert seq == par


def test_no_excl_in_query_skips_filter():
    """When the query has no EXCL points the filter is a no-op and hits behave normally."""
    mol = _embed("c1ccccc1O", 0xF00D)
    query = query_pharmacophore_from_molecule(mol)
    searcher = PharmacophoreSearch(query=query, use_direction=False)
    hits = searcher.screen(mol, progress=False, n_jobs=1)
    assert len(hits) == 1


def test_add_excluded_volume_envelope_still_admits_reference_ligand():
    """An envelope placed ``outside`` the reference vdW surface must not clash with the reference."""
    mol = _embed("c1ccccc1O", 0xF00D)
    query = query_pharmacophore_from_molecule(mol)
    n = add_excluded_volume(
        mol, query, shell_inner=1.0, shell_outer=2.5, spacing=1.5
    )
    assert n > 0
    searcher = PharmacophoreSearch(query=query, use_direction=False)
    hits = searcher.screen(mol, progress=False, n_jobs=1)
    assert len(hits) == 1
