"""Suite-wide hermeticity guards.

The contract this file defends: running the tests must make ZERO external
network calls. Every test either stubs its transport or exercises the mock
provider path.

`source_ug_enabled` flipped to True in 2026-07 (see config.py for why: every
keyless general search backend now blocks datacenter IPs, leaving Ultimate
Guitar as the only source that works without an API key). That default is
right for production and wrong for tests: `discover_sources()` began reaching
ultimate-guitar.com for real, so tests that stubbed one backend and asserted
one candidate started seeing four — a live-network dependency dressed up as an
assertion failure, and one that would go red on any machine without internet.

So: force it back OFF for every test. The files that mean to exercise UG opt
back in with their own autouse fixture (tests/test_ug_source.py), which runs
after this one because module-level autouse fixtures are ordered after
conftest-level ones at the same scope.
"""

from __future__ import annotations

import pytest

from snoocle_server.config import settings
from snoocle_server.store.run_admission import reset_run_admission_store
from snoocle_server.discovery.source_cache import reset_source_cache


@pytest.fixture(autouse=True)
def _ug_source_off_unless_a_test_asks_for_it(monkeypatch):
    monkeypatch.setattr(settings, "source_ug_enabled", False)


@pytest.fixture(autouse=True)
def _isolated_run_admission_store():
    reset_run_admission_store()
    yield
    reset_run_admission_store()


@pytest.fixture(autouse=True)
def _isolated_parsed_source_cache():
    reset_source_cache()
    yield
    reset_source_cache()
