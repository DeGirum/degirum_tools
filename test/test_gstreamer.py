import pytest


def test_import_gi():
    try:
        import gi
    except ImportError:
        pytest.skip("gi module not available")
    assert hasattr(gi, "require_version")


def test_import_gst():
    import gi
    try:
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
    except (ImportError, ValueError):
        pytest.skip("Gst namespace not available")
    assert hasattr(Gst, "init")


def test_gst_init_and_version():
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
    if not Gst.is_initialized():
        Gst.init(None)
    version = Gst.version_string()
    assert isinstance(version, str)
    assert version.count('.') >= 2


def test_gst_parse_launch_pipeline():
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
    if not Gst.is_initialized():
        Gst.init(None)
    pipeline = Gst.parse_launch("fakesrc num-buffers=1 ! fakesink")
    assert pipeline is not None
    result = pipeline.set_state(Gst.State.PLAYING)
    assert result in (Gst.StateChangeReturn.SUCCESS, Gst.StateChangeReturn.ASYNC)
    pipeline.set_state(Gst.State.NULL)
