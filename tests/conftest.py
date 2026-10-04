import os
from datetime import datetime, timezone

import pytest
from hypothesis import settings

# CI : tirages hypothesis reproductibles (un échec se rejoue à l'identique).
# En local : tirages aléatoires, qui explorent de nouveaux cas à chaque lancement.
settings.register_profile("ci", derandomize=True, print_blob=True)
if os.environ.get("CI"):
    settings.load_profile("ci")


@pytest.fixture
def t0():
    return datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)
