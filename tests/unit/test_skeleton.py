"""The packages import and carry a version (M0 smoke test)."""

import r1mc3
import r1sim


def test_packages_import() -> None:
    assert r1mc3.__version__ == r1sim.__version__
