from edmo import __version__


def test_package_version() -> None:
    assert isinstance(__version__, str)
    assert __version__ == "0.1.0"
