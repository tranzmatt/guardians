"""Tests for the abstract automaton transfer (ordered first-match).

The static successor-state transfer must mirror the runtime's ordered
first-match semantics: consider a state's matching transitions in
declaration order, fire the first whose guard is true, otherwise stay put.
Statically a guard is TRUE / FALSE / UNKNOWN, and an UNKNOWN guard forks
(true branch takes the transition, false branch falls through to later
transitions).  Stopping at the first non-false guard would hide a competing
guarded transition declared after a benign one.
"""

import pytest

from guardians.workflow import Workflow, WorkflowStep, ToolCallNode, SymRef
from guardians.tools import ToolSpec, ParamSpec, ToolRegistry
from guardians.policy import (
    Policy, SecurityAutomaton, AutomatonState, AutomatonTransition,
)
from guardians.execute import WorkflowExecutor
from guardians.errors import SecurityViolation
from guardians.verify import verify


def _registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(
        ToolSpec(name="go", params=[ParamSpec(name="x", type="str")]),
        lambda x="": None,
    )
    return r


def _policy(transitions, *, states=None) -> Policy:
    states = states or [
        AutomatonState(name="q0"),
        AutomatonState(name="safe"),
        AutomatonState(name="error", is_error=True),
    ]
    return Policy(
        name="t",
        allowed_tools=["go"],
        automata=[SecurityAutomaton(
            name="m", states=states, initial_state="q0",
            transitions=transitions,
        )],
    )


def _call_go(arguments, *, inputs=()):
    return Workflow(goal="t", input_variables=list(inputs), steps=[
        WorkflowStep(label="g", tool_call=ToolCallNode(
            tool_name="go", arguments=arguments)),
    ])


# 1. Unknown safe guard followed by an unknown error guard: reject.
def test_two_unknown_guards_same_state_reaches_error():
    policy = _policy([
        AutomatonTransition(from_state="q0", to_state="safe",
                            tool_name="go", condition="x == 'safe'"),
        AutomatonTransition(from_state="q0", to_state="error",
                            tool_name="go", condition="x == 'bad'"),
    ])
    result = verify(_call_go({"x": SymRef(ref="x")}, inputs=["x"]),
                    policy, _registry())
    assert not result.ok
    auto = [v for v in result.violations if v.category == "automaton"]
    assert len(auto) >= 1
    assert "could" in auto[0].message.lower()


# 2. Unknown guard followed by an unconditional error transition: reject.
def test_unknown_guard_then_unconditional_error():
    policy = _policy([
        AutomatonTransition(from_state="q0", to_state="safe",
                            tool_name="go", condition="x == 'safe'"),
        AutomatonTransition(from_state="q0", to_state="error",
                            tool_name="go"),  # unconditional
    ])
    result = verify(_call_go({"x": SymRef(ref="x")}, inputs=["x"]),
                    policy, _registry())
    assert not result.ok
    assert any(v.category == "automaton" for v in result.violations)


# 3. Definitely-true first guard makes a later error transition unreachable.
def test_definite_true_guard_shadows_later_error():
    policy = _policy([
        AutomatonTransition(from_state="q0", to_state="safe",
                            tool_name="go", condition="x == 'safe'"),
        AutomatonTransition(from_state="q0", to_state="error",
                            tool_name="go", condition="x == 'bad'"),
    ])
    result = verify(_call_go({"x": "safe"}), policy, _registry())
    assert result.ok
    assert not any(v.category == "automaton" for v in result.violations)


# 4. A nested symbolic value (in a list) makes the guard unknown.
def test_nested_symbolic_value_is_unknown():
    policy = _policy([
        AutomatonTransition(from_state="q0", to_state="safe",
                            tool_name="go", condition="x == 'safe'"),
        AutomatonTransition(from_state="q0", to_state="error",
                            tool_name="go", condition="x == 'bad'"),
    ])
    # x is a list literal that *contains* a symbolic reference.
    result = verify(_call_go({"x": [SymRef(ref="sym")]}, inputs=["sym"]),
                    policy, _registry())
    assert not result.ok
    assert any(v.category == "automaton" for v in result.violations)


# 5a. A malformed/unevaluable guard is rejected statically.
def test_malformed_guard_rejected_statically():
    policy = _policy(
        [AutomatonTransition(from_state="q0", to_state="other",
                             tool_name="go", condition="1 +")],
        states=[AutomatonState(name="q0"), AutomatonState(name="other")],
    )
    result = verify(_call_go({"x": "anything"}), policy, _registry())
    assert not result.ok
    inc = [v for v in result.violations
           if v.category == "analysis_incomplete"
           and v.rule_name == "automaton_guard"]
    assert len(inc) == 1


# 5b. The same guard fails closed at runtime (raises, never silently skips).
def test_malformed_guard_fails_closed_at_runtime():
    policy = _policy(
        [AutomatonTransition(from_state="q0", to_state="other",
                             tool_name="go", condition="1 +")],
        states=[AutomatonState(name="q0"), AutomatonState(name="other")],
    )
    executor = WorkflowExecutor(_registry(), policy, auto_approve=True,
                                verify_first=False)
    with pytest.raises(SecurityViolation):
        executor.run(_call_go({"x": "anything"}))
