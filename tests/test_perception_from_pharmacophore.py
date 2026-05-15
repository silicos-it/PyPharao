from pypharao import (
    FuncGroup,
    PerceptionOptions,
    Pharmacophore,
    PharmacophorePoint,
    PharmacophoreSearch,
    default_alpha,
    perception_options_from_pharmacophore,
)


def test_perception_options_from_arom_and_hybh_query():
    ph = Pharmacophore(
        points=[
            PharmacophorePoint(0, 0, 0, FuncGroup.AROM, default_alpha(FuncGroup.AROM), True, 0, 0, 1),
            PharmacophorePoint(1, 0, 0, FuncGroup.HYBH, default_alpha(FuncGroup.HYBH), True, 1, 0, 1),
        ]
    )
    opts = perception_options_from_pharmacophore(ph)
    assert opts.arom is True
    assert opts.hdon is True
    assert opts.hacc is True
    assert opts.hybh is True
    assert opts.lipo is False
    assert opts.hybl is False


def test_search_derives_perception_from_reference():
    ph = Pharmacophore(
        points=[
            PharmacophorePoint(0, 0, 0, FuncGroup.AROM, default_alpha(FuncGroup.AROM), True, 0, 0, 1),
        ]
    )
    searcher = PharmacophoreSearch(ph)
    opts = searcher.perception_options
    assert opts is not None
    assert opts.arom is True
    assert opts.hybl is False
    assert opts.hybh is False


def test_empty_pharmacophore_uses_full_perception():
    opts = perception_options_from_pharmacophore(Pharmacophore())
    full = PerceptionOptions()
    assert opts.arom == full.arom and opts.hybl == full.hybl and opts.hybh == full.hybh


def test_hybl_query_enables_matchable_database_types():
    ph = Pharmacophore(
        points=[
            PharmacophorePoint(0, 0, 0, FuncGroup.HYBL, default_alpha(FuncGroup.HYBL), True, 0, 0, 1),
        ]
    )
    opts = perception_options_from_pharmacophore(ph)
    assert opts.arom is True
    assert opts.lipo is True
    assert opts.hybl is True
    assert opts.hdon is False


def test_searcher_caches_perception_at_init():
    ph = Pharmacophore(
        points=[
            PharmacophorePoint(0, 0, 0, FuncGroup.AROM, default_alpha(FuncGroup.AROM), True, 0, 0, 1),
        ]
    )
    searcher = PharmacophoreSearch(ph)
    cached = searcher.perception_options
    assert cached is not None
    assert cached is searcher.perception_options
    ph.points.append(
        PharmacophorePoint(1, 0, 0, FuncGroup.LIPO, default_alpha(FuncGroup.LIPO), False, 0, 0, 0)
    )
    assert searcher.perception_options.lipo is False
    searcher.refresh_perception_options()
    assert searcher.perception_options.lipo is True


def test_searcher_updates_perception_when_ref_reassigned():
    arom_only = Pharmacophore(
        points=[
            PharmacophorePoint(0, 0, 0, FuncGroup.AROM, default_alpha(FuncGroup.AROM), True, 0, 0, 1),
        ]
    )
    with_lipo = Pharmacophore(
        points=[
            PharmacophorePoint(0, 0, 0, FuncGroup.LIPO, default_alpha(FuncGroup.LIPO), False, 0, 0, 0),
        ]
    )
    searcher = PharmacophoreSearch(arom_only)
    assert searcher.perception_options.lipo is False
    searcher.ref = with_lipo
    assert searcher.perception_options.lipo is True
    assert searcher.perception_options.arom is False
