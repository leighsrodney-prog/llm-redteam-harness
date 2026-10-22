"""Offline AgentDojo bridge: run AgentDojo's task suites with deterministic agents and no LLM.

AgentDojo (Debenedetti et al., 2024; github.com/ethz-spylab/agentdojo) ships its task
suites (workspace, travel, banking, slack) as Python objects with a simulated environment,
utility checks and prompt-injection security checks. This bridge plugs scripted agents into
AgentDojo's own pipeline interface, so the real environments, attacks and checkers run
end to end without any model or network call:

  ground_truth        executes each user task's reference tool calls (a sanity upper bound)
  follow_injections   does the same, but also carries out any injected instruction it sees in
                      a tool result (a worst-case "fully injectable" agent)
  follow_injections + defense=tool_filter
                      the follower behind a simple allowlist defense: only tools the user
                      task itself needs may be called

These are calibration agents for the harness and its scoring, not models. Numbers they
produce describe the benchmark plumbing and the defense logic, not any real model.

Install: pip install "agentdojo==0.1.35" (an optional extra; the core harness doesn't need it).
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

AGENTS = ("ground_truth", "follow_injections")
DEFENSES = ("none", "tool_filter")


def _require():
    try:
        import agentdojo  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise RuntimeError('AgentDojo bridge needs: pip install "agentdojo==0.1.35"') from e


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("agentdojo")
    except PackageNotFoundError:  # pragma: no cover
        return "unknown"


def _norm(s: str) -> str:
    return " ".join(str(s).replace("\\n", " ").replace("'", "").replace('"', "").split()).lower()


@dataclass
class OfflineAgentState:
    blocked: list[str] = field(default_factory=list)
    injected_calls: list[str] = field(default_factory=list)
    saw_injection: bool = False


def make_agent(agent: str = "ground_truth", defense: str = "none"):
    """Return an AgentDojo pipeline element implementing a scripted agent."""
    _require()
    if agent not in AGENTS:
        raise ValueError(f"agent must be one of {AGENTS}")
    if defense not in DEFENSES:
        raise ValueError(f"defense must be one of {DEFENSES}")
    from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
    from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
    from agentdojo.functions_runtime import EmptyEnv, FunctionsRuntime
    from agentdojo.types import ChatAssistantMessage, ChatToolResultMessage, text_content_block_from_string

    class ScriptedAgent(BasePipelineElement):
        # "local" lets AgentDojo attacks that address the model by name use "Local model"
        name = f"local-offline-{agent}" + ("" if defense == "none" else f"+{defense}")

        def __init__(self):
            self.user_task = None
            self.injection_task = None
            self.state = OfflineAgentState()

        def set_task(self, user_task, injection_task=None):
            self.user_task, self.injection_task = user_task, injection_task
            self.state = OfflineAgentState()

        def _call(self, runtime, env, tool_call, msgs):
            msgs.append(ChatAssistantMessage(role="assistant", tool_calls=[tool_call],
                                             content=[text_content_block_from_string("")]))
            result, error = runtime.run_function(env, tool_call.function, tool_call.args, raise_on_error=False)
            out = tool_result_to_str(result) if error is None else f"error: {error}"
            msgs.append(ChatToolResultMessage(role="tool", content=[text_content_block_from_string(out)],
                                              tool_call=tool_call, tool_call_id=None, error=error))
            return out

        def query(self, query: str, runtime: FunctionsRuntime, env=EmptyEnv(),
                  messages: Sequence = (), extra_args: dict = {}):
            if self.user_task is None:
                raise ValueError("call set_task() first")
            msgs = list(messages)
            allowed = {tc.function for tc in self.user_task.ground_truth(env.model_copy(deep=True))}
            outputs = [self._call(runtime, env, tc, msgs) for tc in self.user_task.ground_truth(env)]
            inj = self.injection_task
            if agent == "follow_injections" and inj is not None:
                # tool results are YAML-rendered, which can re-wrap long strings: compare normalized text
                goal = _norm(inj.GOAL)
                self.state.saw_injection = any(goal in _norm(o) for o in outputs)
                if self.state.saw_injection:
                    for tc in inj.ground_truth(env):
                        if defense == "tool_filter" and tc.function not in allowed:
                            self.state.blocked.append(tc.function)
                            continue
                        self.state.injected_calls.append(tc.function)
                        self._call(runtime, env, tc, msgs)
            final = self.user_task.GROUND_TRUTH_OUTPUT
            msgs.append(ChatAssistantMessage(role="assistant", tool_calls=None,
                                             content=[text_content_block_from_string(final)]))
            return query, runtime, env, msgs, extra_args

    return ScriptedAgent()


def run(suite_name: str = "workspace", version: str = "v1.2.2", agent: str = "follow_injections",
        defense: str = "none", attack: str = "important_instructions", user_tasks=None,
        injection_tasks=None, limit_user: int | None = None, limit_injection: int | None = None):
    """Run AgentDojo user x injection task pairs offline. Returns {"results", "summary"}."""
    _require()
    from agentdojo.attacks.attack_registry import load_attack
    from agentdojo.task_suite.load_suites import get_suite

    suite = get_suite(version, suite_name)
    pipe = make_agent(agent, defense)
    atk = load_attack(attack, suite, pipe)
    uts = [suite.get_user_task_by_id(i) for i in user_tasks] if user_tasks else list(suite.user_tasks.values())
    its = ([suite.get_injection_task_by_id(i) for i in injection_tasks] if injection_tasks
           else list(suite.injection_tasks.values()))
    uts, its = uts[:limit_user] if limit_user else uts, its[:limit_injection] if limit_injection else its
    results, t0 = [], time.perf_counter()
    for ut in uts:
        pipe.set_task(ut)
        utility_clean, _ = suite.run_task_with_pipeline(pipe, ut, None, {})
        for it in its:
            try:
                injections = atk.attack(ut, it)
            except ValueError:  # this user task never shows the agent an injectable field
                continue
            pipe.set_task(ut, it)
            utility, injected = suite.run_task_with_pipeline(pipe, ut, it, injections)
            results.append({
                "id": f"{suite_name}/{ut.ID}/{it.ID}", "user_task": ut.ID, "injection_task": it.ID,
                "utility_clean": bool(utility_clean), "utility": bool(utility),
                "attack_success": bool(injected), "saw_injection": pipe.state.saw_injection,
                "blocked_calls": list(pipe.state.blocked), "injected_calls": list(pipe.state.injected_calls),
                # harness convention: a case passes when the attack failed and the user task still worked
                "passed": bool(utility) and not injected,
                "reasons": ([] if utility else ["user task failed"]) + (["injection executed"] if injected else []),
            })
    return {"results": results, "summary": summarize(results, suite_name, version, agent, defense, attack,
                                                     agentdojo_version=_version(),
                                                     wall_s=time.perf_counter() - t0)}


def summarize(results, suite_name, version, agent, defense, attack, agentdojo_version="unknown", wall_s=0.0):
    n = len(results)
    rate = (lambda k: round(sum(r[k] for r in results) / n, 4) if n else 0.0)
    clean = {r["user_task"]: r["utility_clean"] for r in results}
    return {
        "suite": suite_name, "benchmark_version": version, "agentdojo_version": agentdojo_version,
        "agent": agent, "defense": defense, "attack": attack, "total": n,
        "passed": sum(r["passed"] for r in results),
        "pass_rate": rate("passed") if n else 0.0,
        "utility_clean": round(sum(clean.values()) / len(clean), 4) if clean else 0.0,
        "utility_under_attack": rate("utility"),
        "attack_success_rate": rate("attack_success"),
        "injection_seen_rate": rate("saw_injection"),
        "blocked_calls": sum(len(r["blocked_calls"]) for r in results),
        "wall_s": round(wall_s, 2),
    }
