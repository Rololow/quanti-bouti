from datetime import datetime, timezone

import pytest


@pytest.fixture
def t0():
    return datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)
