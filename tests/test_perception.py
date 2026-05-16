import io

import pytest

from pypharao import (
    MoleculePharmacophorePerception,
    PharmacophorePerception,
    PointType,
    QueryPharmacophorePerception,
)


def test_query_perception_defaults_all_enabled():
    opts = QueryPharmacophorePerception()
    for t in (
        PointType.AROM,
        PointType.LIPO,
        PointType.HDON,
        PointType.HACC,
        PointType.HACC_AND_HDON,
        PointType.POSC,
        PointType.NEGC,
    ):
        assert opts.is_enabled(t)


def test_query_perception_rejects_query_only_or_types():
    """AROM_OR_LIPO and HACC_OR_HDON are user-introduced into the QueryPharmacophore;
    auto-perception never emits them so the perception object refuses them."""
    opts = QueryPharmacophorePerception()
    for t in (PointType.AROM_OR_LIPO, PointType.HACC_OR_HDON, PointType.EXCL, PointType.UNDEF):
        with pytest.raises(ValueError):
            opts.is_enabled(t)
        with pytest.raises(ValueError):
            opts.enable(t)


def test_molecule_perception_rejects_query_only_types():
    opts = MoleculePharmacophorePerception()
    for t in (PointType.AROM_OR_LIPO, PointType.HACC_OR_HDON, PointType.EXCL, PointType.UNDEF):
        with pytest.raises(ValueError):
            opts.is_enabled(t)
    with pytest.raises(ValueError):
        opts.enable(PointType.EXCL)


def test_enable_disable_and_iteration():
    opts = MoleculePharmacophorePerception()
    opts.disable(PointType.LIPO)
    assert opts.is_enabled(PointType.LIPO) is False
    opts.enable("LIPO")
    assert opts.is_enabled("LIPO") is True
    opts.set_enabled(PointType.HACC, False)
    enabled = opts.types_enabled()
    assert PointType.HACC not in enabled
    assert PointType.LIPO in enabled
    assert list(opts).count(PointType.LIPO) == 1


def test_print_features_lists_allowed_types():
    buf = io.StringIO()
    opts = QueryPharmacophorePerception()
    opts.disable(PointType.LIPO)
    opts.print_features(file=buf)
    out = buf.getvalue()
    assert "LIPO" in out and "off" in out
    assert "AROM" in out and "on" in out
    # Query-only OR types are not auto-perceived
    assert "AROM_OR_LIPO" not in out
    assert "HACC_OR_HDON" not in out


def test_perception_base_class_has_no_allowed_types():
    base = PharmacophorePerception()
    assert base.types_enabled() == []
