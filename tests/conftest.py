import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEMO_DIR = os.path.join(ROOT, "data", "demo")
NETWORK = os.path.join(DEMO_DIR, "network.json")


@pytest.fixture
def in_repo_root(monkeypatch):
    """Default paths in config are relative to the repository root"""
    monkeypatch.chdir(ROOT)
