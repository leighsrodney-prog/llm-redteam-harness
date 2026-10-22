"""Offline AgentDojo bridge. Skipped unless the optional `agentdojo` extra is installed."""
import json
import sys

import pytest

from harness.bridges import agentdojo_offline as ad


def test_missing_extra_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "agentdojo", None)
    with pytest.raises(RuntimeError, match="pip install"):
        ad.make_agent()


agentdojo = pytest.importorskip("agentdojo")


@pytest.mark.parametrize("agent,defense", [("nope", "none"), ("ground_truth", "nope")])
def test_bad_agent_or_defense(agent, defense):
    with pytest.raises(ValueError):
        ad.make_agent(agent, defense)


def test_agent_requires_task():
    from agentdojo.functions_runtime import FunctionsRuntime
    with pytest.raises(ValueError, match="set_task"):
        ad.make_agent().query("hi", FunctionsRuntime([]))


def test_pipeline_name_maps_to_local_model():
    from agentdojo.attacks.base_attacks import get_model_name_from_pipeline
    assert get_model_name_from_pipeline(ad.make_agent("follow_injections", "tool_filter")) == "Local model"


def test_ground_truth_agent_is_useful_and_never_injected():
    out = ad.run("banking", agent="ground_truth", limit_user=3)
    s = out["summary"]
    assert s["total"] > 0
    assert s["utility_clean"] == 1.0 and s["utility_under_attack"] == 1.0
    assert s["attack_success_rate"] == 0.0 and s["pass_rate"] == 1.0


def test_follower_is_injected_and_tool_filter_reduces_asr():
    follow = ad.run("slack", agent="follow_injections", limit_user=4)["summary"]
    filtered = ad.run("slack", agent="follow_injections", defense="tool_filter", limit_user=4)["summary"]
    assert follow["attack_success_rate"] > 0.5
    assert filtered["attack_success_rate"] < follow["attack_success_rate"]
    assert filtered["blocked_calls"] > 0 and follow["blocked_calls"] == 0


def test_result_rows_are_consistent():
    res = ad.run("banking", agent="follow_injections", limit_user=2, limit_injection=3)["results"]
    assert len({r["id"] for r in res}) == len(res)
    for r in res:
        assert r["passed"] == (r["utility"] and not r["attack_success"])
        assert bool(r["reasons"]) == (not r["passed"])
        if r["attack_success"]:
            assert r["injected_calls"]


def test_explicit_task_ids_and_other_attack():
    out = ad.run("banking", agent="follow_injections", attack="direct",
                 user_tasks=["user_task_0"], injection_tasks=["injection_task_0"])
    assert [r["id"] for r in out["results"]] == ["banking/user_task_0/injection_task_0"]
    assert out["summary"]["attack"] == "direct"


def test_cli_writes_report_and_max_asr_gate(tmp_path, capsys):
    from harness.cli import main
    rc = main(["agentdojo", "--suite", "slack", "--limit-user", "2", "--out", str(tmp_path / "a"),
               "--max-asr", "0.0"])
    assert rc == 1
    assert "ASR" in capsys.readouterr().out
    rc = main(["agentdojo", "--suite", "banking", "--agent", "ground_truth", "--limit-user", "2",
               "--out", str(tmp_path / "b"), "--max-asr", "0.0"])
    assert rc == 0
    files = list((tmp_path / "b").iterdir())
    data = json.loads(next(p for p in files if p.suffix == ".json").read_text())
    assert "agentdojo/banking" in json.dumps(data)
