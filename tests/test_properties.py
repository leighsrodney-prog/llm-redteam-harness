"""Property-based tests (Hypothesis) for the scoring and statistics core."""
import json
import math

import pytest

hyp = pytest.importorskip("hypothesis")
from hypothesis import given, strategies as st  # noqa: E402

from harness import stats  # noqa: E402
from harness.adapters import MockAdapter  # noqa: E402
from harness.runner import percentile  # noqa: E402
from harness.scorers import is_refusal, leak_hits, score  # noqa: E402
from harness.suite import Case, load_suite  # noqa: E402

floats = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
pcts = st.floats(min_value=0, max_value=100)
safe_text = st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=200)


@given(st.lists(floats, min_size=1, max_size=60), pcts)
def test_percentile_within_bounds(xs, p):
    v = percentile(xs, p)
    assert min(xs) - 1e-6 <= v <= max(xs) + 1e-6


@given(st.lists(floats, min_size=1, max_size=60), pcts, pcts)
def test_percentile_monotonic_in_p(xs, p1, p2):
    lo, hi = sorted((p1, p2))
    assert percentile(xs, lo) <= percentile(xs, hi) + 1e-6


@given(st.lists(floats, min_size=1, max_size=60))
def test_percentile_endpoints_and_order_invariance(xs):
    assert percentile(xs, 0) == min(xs)
    assert percentile(xs, 100) == max(xs)
    assert percentile(list(reversed(xs)), 50) == percentile(xs, 50)


@given(st.lists(st.floats(min_value=0, max_value=1), min_size=1, max_size=40), st.integers(0, 10))
def test_bootstrap_ci_ordered_and_inside_range(xs, seed):
    lo, hi = stats.bootstrap_ci(xs, n_boot=200, seed=seed)
    assert lo <= hi
    assert min(xs) - 1e-9 <= lo and hi <= max(xs) + 1e-9


@given(st.lists(st.floats(min_value=0, max_value=1), min_size=1, max_size=12))
def test_holm_adjusted_bounds(ps):
    adj = stats.holm(ps)
    assert len(adj) == len(ps)
    for raw, a in zip(ps, adj):
        assert raw - 1e-12 <= a <= 1.0
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    assert all(adj[order[i]] <= adj[order[i + 1]] + 1e-12 for i in range(len(order) - 1))


@given(st.sets(st.integers(0, 30)), st.sets(st.integers(0, 30)))
def test_precision_recall_bounds(pred, truth):
    p, r = stats.precision_recall(pred, truth)
    assert 0.0 <= p <= 1.0 and 0.0 <= r <= 1.0
    if pred == truth:
        assert (p, r) == (1.0, 1.0)


@given(st.lists(st.lists(st.sampled_from([0, 1]), min_size=6, max_size=6), min_size=3, max_size=4))
def test_permutation_p_is_a_probability_and_identical_runs_not_significant(runs):
    p = stats.paired_permutation_p(runs, runs, n_perm=200)
    assert 0.0 <= p <= 1.0
    assert p >= 0.5  # no difference at all can't look like a drop


@given(safe_text)
def test_scorer_never_raises_and_reasons_explain_failures(text):
    c = Case(id="x", category="pii_leak", prompt="p", expect="refuse", leak_patterns=[r"\d{3}-\d{2}-\d{4}"])
    r = score(c, text)
    assert isinstance(r["passed"], bool)
    assert r["passed"] == (not r["reasons"])
    assert r["refused"] == is_refusal(text)


@given(safe_text)
def test_refusal_prefix_passes_refuse_cases(text):
    c = Case(id="x", category="jailbreak", prompt="p", expect="refuse")
    assert score(c, "I'm sorry, but I can't help with that. " + text.replace("\\", ""))["refused"]


@given(safe_text, st.lists(st.sampled_from([r"canary-\d+", r"secret", r"\bpwned\b"]), max_size=3))
def test_leak_hits_subset_of_patterns(text, pats):
    assert set(leak_hits(text, pats)) <= set(pats)


@given(st.text(min_size=1, max_size=120))
def test_mock_adapter_deterministic(prompt):
    a, b = MockAdapter(latency_ms=0), MockAdapter(latency_ms=0)
    assert a.complete("", prompt) == b.complete("", prompt)


ids = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789-", min_size=1, max_size=12)
case_dicts = st.fixed_dictionaries({
    "category": st.sampled_from(["prompt_injection", "jailbreak", "pii_leak", "benign"]),
    "prompt": st.text(min_size=1, max_size=80),
    "expect": st.sampled_from(["refuse", "comply"]),
    "must_contain": st.lists(st.text(max_size=10), max_size=2),
})


@given(st.lists(case_dicts, min_size=1, max_size=8), st.sampled_from(["jsonl", "yaml"]))
def test_suite_roundtrip(tmp_path_factory, raws, fmt):
    raws = [dict(r, id=f"case-{i}") for i, r in enumerate(raws)]
    d = tmp_path_factory.mktemp("s")
    p = d / f"s.{fmt}"
    if fmt == "jsonl":
        p.write_text("\n".join(json.dumps(r) for r in raws))
    else:
        import yaml
        p.write_text(yaml.safe_dump({"cases": raws}))
    loaded = load_suite(p)
    assert [(c.id, c.category, c.prompt, c.expect, c.must_contain) for c in loaded] == \
        [(r["id"], r["category"], r["prompt"], r["expect"], r["must_contain"]) for r in raws]


@given(st.lists(st.floats(min_value=0, max_value=1), min_size=2, max_size=20))
def test_run_means_matches_definition(xs):
    assert math.isclose(stats.run_means([xs])[0], sum(xs) / len(xs), rel_tol=1e-9, abs_tol=1e-12)
