import io

from pypharao import FuncGroup, PerceptionOptions


def test_is_enabled_for_perception_defaults():
    opts = PerceptionOptions()
    for func in (
        FuncGroup.AROM,
        FuncGroup.HDON,
        FuncGroup.HACC,
        FuncGroup.LIPO,
        FuncGroup.POSC,
        FuncGroup.NEGC,
        FuncGroup.HYBH,
        FuncGroup.HYBL,
    ):
        assert opts.is_enabled_for_perception(func) is True
    assert opts.is_enabled_for_perception(FuncGroup.EXCL) is None
    assert opts.is_enabled_for_perception(FuncGroup.UNDEF) is None


def test_hybrid_requires_prerequisites():
    opts = PerceptionOptions(hybh=True, hdon=False)
    assert opts.is_enabled_for_perception(FuncGroup.HYBH) is False
    opts = PerceptionOptions(hybl=True, lipo=False)
    assert opts.is_enabled_for_perception(FuncGroup.HYBL) is False


def test_enable_disable_set_enabled():
    opts = PerceptionOptions()
    opts.disable(FuncGroup.LIPO)
    assert opts.lipo is False
    opts.enable("AROM")
    assert opts.arom is True
    opts.set_enabled(FuncGroup.HACC, False)
    assert opts.hacc is False


def test_set_enabled_raises_for_manual_only():
    opts = PerceptionOptions()
    try:
        opts.enable(FuncGroup.EXCL)
    except ValueError as e:
        assert "EXCL" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_print_features_lists_all_types():
    buf = io.StringIO()
    PerceptionOptions(arom=False, hybh=True, hdon=False).print_features(file=buf)
    text = buf.getvalue()
    for func in FuncGroup:
        assert func.value in text
    assert "AROM   off" in text
    assert "needs hdon" in text
    assert "EXCL" in text and "—" in text
