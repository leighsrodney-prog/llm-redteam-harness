import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _block_network(monkeypatch):
    import urllib.request

    def boom(*a, **k):
        raise AssertionError("network access attempted in tests")
    monkeypatch.setattr(urllib.request, "urlopen", boom)


block_network = pytest.fixture(autouse=True)(_block_network)


# ---- shared fixtures (pytest-depth) ----
import json as _json
import types as _types

from harness.adapters import MockAdapter
from harness.suite import Case

try:  # hypothesis profiles: "ci" is derandomized so CI runs are reproducible
    from hypothesis import settings as _hsettings, HealthCheck as _HC
    _hsettings.register_profile("ci", derandomize=True, max_examples=150, deadline=None,
                                suppress_health_check=[_HC.function_scoped_fixture])
    _hsettings.register_profile("dev", max_examples=50, deadline=None)
    _hsettings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
except ImportError:  # pragma: no cover - hypothesis is a dev dependency
    pass


@pytest.fixture
def make_case():
    """Factory for Case objects with sensible defaults."""
    def _make(**kw):
        base = {"id": "c1", "category": "benign", "prompt": "Explain unit tests.", "expect": "comply"}
        base.update(kw)
        return Case(**base)
    return _make


@pytest.fixture
def suite_file(tmp_path):
    """Write a list of raw case dicts to a .yaml or .jsonl suite and return its path."""
    def _write(raws, fmt="jsonl"):
        p = tmp_path / f"suite.{fmt}"
        if fmt == "jsonl":
            p.write_text("\n".join(_json.dumps(r) for r in raws) + "\n")
        else:
            import yaml
            p.write_text(yaml.safe_dump({"cases": raws}))
        return p
    return _write


@pytest.fixture
def fast_mock():
    """Mock adapter with no simulated latency."""
    return MockAdapter(latency_ms=0)


@pytest.fixture
def fake_llama(monkeypatch):
    """Install a fake `llama_cpp` module so the local adapter can be tested without a model."""
    calls = {}

    class Llama:
        def __init__(self, **kw):
            calls["init"] = kw

        def create_chat_completion(self, **kw):
            calls["chat"] = kw
            return {"choices": [{"message": {"content": "I'm sorry, but I can't help with that."}}]}

    mod = _types.ModuleType("llama_cpp")
    mod.Llama = Llama
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)
    return calls
