"""Tests for sound finite-lattice loop fixpoint verification.

These exercise the rewritten ``_verify_loop`` in ``guardians.verify``:
the loop is modelled as the least fixpoint of ``H = entry ⊔ body(H)`` over
a finite lattice, rather than a fixed bounded unrolling.
"""

import importlib

from guardians.verify import (
    AbstractValue,
    AbstractState,
    join_value,
    join_state,
    value_leq,
    value_eq,
    state_leq,
    state_eq,
    verify,
    _verify_loop,
)
from guardians.workflow import (
    Workflow, WorkflowStep, ToolCallNode, LoopNode, SymRef,
)
from guardians.tools import ToolSpec, ParamSpec, ToolRegistry
from guardians.policy import (
    Policy, SecurityAutomaton, AutomatonState, AutomatonTransition, TaintRule,
)
from guardians.results import VerificationResult

# ``guardians/__init__`` does ``from .verify import verify``, which rebinds
# the ``verify`` *attribute* of the package to the function and shadows the
# submodule of the same name.  So both ``import guardians.verify as m`` and
# pytest's dotted-string ``monkeypatch.setattr`` target (it walks the path
# with getattr) resolve to the function, not the module.  ``import_module``
# returns the real module object from ``sys.modules``, which is what we
# patch internals on.
_verify_mod = importlib.import_module("guardians.verify")


# --- Shared tool registries ---

def _relay_registry() -> ToolRegistry:
    """secret_src/clean_src produce data; relay propagates taint; sink consumes."""
    r = ToolRegistry()
    r.register(ToolSpec(name="secret_src", source_labels=["secret"]), lambda: "s")
    r.register(ToolSpec(name="clean_src", source_labels=[]), lambda: "c")
    r.register(
        ToolSpec(name="relay", params=[ParamSpec(name="x", type="str")],
                 source_labels=[]),
        lambda x="": x,
    )
    r.register(
        ToolSpec(name="sink",
                 params=[ParamSpec(name="data", type="str", is_taint_sink=True)]),
        lambda data="": None,
    )
    return r


def _relay_policy() -> Policy:
    return Policy(
        name="t",
        allowed_tools=["secret_src", "clean_src", "relay", "sink"],
        taint_rules=[TaintRule(
            name="no_leak", source_tool="secret_src",
            sink_tool="sink", sink_param="data",
        )],
    )


# ===================================================================
# 1. Taint requiring more than three iterations
# ===================================================================

def test_taint_reaches_sink_only_on_fourth_iteration():
    """Reverse-order relays mean the secret reaches x4 only on the 4th
    abstract iteration. A 3-iteration unrolling would miss it."""
    wf = Workflow(goal="t", input_variables=["items"], steps=[
        WorkflowStep(label="x0", tool_call=ToolCallNode(
            tool_name="secret_src", arguments={}, result_binding="x0")),
        WorkflowStep(label="x1", tool_call=ToolCallNode(
            tool_name="clean_src", arguments={}, result_binding="x1")),
        WorkflowStep(label="x2", tool_call=ToolCallNode(
            tool_name="clean_src", arguments={}, result_binding="x2")),
        WorkflowStep(label="x3", tool_call=ToolCallNode(
            tool_name="clean_src", arguments={}, result_binding="x3")),
        WorkflowStep(label="x4", tool_call=ToolCallNode(
            tool_name="clean_src", arguments={}, result_binding="x4")),
        WorkflowStep(label="loop", loop=LoopNode(
            collection_ref="items", item_binding="it", body=[
                WorkflowStep(label="u4", tool_call=ToolCallNode(
                    tool_name="relay", arguments={"x": SymRef(ref="x3")},
                    result_binding="x4")),
                WorkflowStep(label="u3", tool_call=ToolCallNode(
                    tool_name="relay", arguments={"x": SymRef(ref="x2")},
                    result_binding="x3")),
                WorkflowStep(label="u2", tool_call=ToolCallNode(
                    tool_name="relay", arguments={"x": SymRef(ref="x1")},
                    result_binding="x2")),
                WorkflowStep(label="u1", tool_call=ToolCallNode(
                    tool_name="relay", arguments={"x": SymRef(ref="x0")},
                    result_binding="x1")),
                WorkflowStep(label="sink", tool_call=ToolCallNode(
                    tool_name="sink", arguments={"data": SymRef(ref="x4")})),
            ])),
    ])
    result = verify(wf, _relay_policy(), _relay_registry())
    assert not result.ok
    taint = [v for v in result.violations if v.category == "taint"]
    assert len(taint) > 0, "secret must be found flowing to the sink on x4"


# ===================================================================
# 2. Automaton state is part of convergence
# ===================================================================

def _ticker_policy() -> Policy:
    return Policy(
        name="t",
        allowed_tools=["tick"],
        automata=[SecurityAutomaton(
            name="ticker",
            states=[
                AutomatonState(name="q0"),
                AutomatonState(name="q1"),
                AutomatonState(name="error", is_error=True),
            ],
            initial_state="q0",
            transitions=[
                AutomatonTransition(from_state="q0", to_state="q1", tool_name="tick"),
                AutomatonTransition(from_state="q1", to_state="error", tool_name="tick"),
            ],
        )],
    )


def _ticker_registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(ToolSpec(name="tick"), lambda: None)
    return r


def test_automaton_error_reachable_on_second_iteration():
    """The loop body performs one tick and never touches env. Convergence
    must account for automaton state: error is reachable on iteration 2."""
    wf = Workflow(goal="t", input_variables=["items"], steps=[
        WorkflowStep(label="loop", loop=LoopNode(
            collection_ref="items", item_binding="it", body=[
                WorkflowStep(label="tk", tool_call=ToolCallNode(
                    tool_name="tick", arguments={})),
            ])),
    ])
    result = verify(wf, _ticker_policy(), _ticker_registry())
    assert not result.ok
    auto = [v for v in result.violations if v.category == "automaton"]
    assert len(auto) > 0, "error state must be discovered across iterations"


# ===================================================================
# 3. Zero-iteration path
# ===================================================================

def test_zero_iteration_path_keeps_taint():
    """The collection may be empty, so a sanitizer inside the loop may run
    zero times. The post-loop state must still include the entry (tainted)
    value."""
    r = ToolRegistry()
    r.register(ToolSpec(name="secret_src", source_labels=["secret"]), lambda: "s")
    r.register(
        ToolSpec(name="sanitize", params=[ParamSpec(name="text", type="str")],
                 source_labels=[]),
        lambda text="": "clean",
    )
    r.register(
        ToolSpec(name="sink",
                 params=[ParamSpec(name="data", type="str", is_taint_sink=True)]),
        lambda data="": None,
    )
    policy = Policy(
        name="t",
        allowed_tools=["secret_src", "sanitize", "sink"],
        taint_rules=[TaintRule(
            name="no_leak", source_tool="secret_src",
            sink_tool="sink", sink_param="data", sanitizers=["sanitize"],
        )],
    )
    wf = Workflow(goal="t", input_variables=["items"], steps=[
        WorkflowStep(label="get", tool_call=ToolCallNode(
            tool_name="secret_src", arguments={}, result_binding="x")),
        WorkflowStep(label="loop", loop=LoopNode(
            collection_ref="items", item_binding="it", body=[
                WorkflowStep(label="san", tool_call=ToolCallNode(
                    tool_name="sanitize", arguments={"text": SymRef(ref="x")},
                    result_binding="x")),
            ])),
        WorkflowStep(label="send", tool_call=ToolCallNode(
            tool_name="sink", arguments={"data": SymRef(ref="x")})),
    ])
    result = verify(wf, policy, r)
    assert not result.ok
    taint = [v for v in result.violations if v.category == "taint"]
    assert len(taint) > 0, "zero-iteration path leaves x tainted at the sink"


# ===================================================================
# 4. Fail closed
# ===================================================================

def test_fail_closed_when_iteration_budget_exhausted(monkeypatch):
    """If convergence is not reached within the (emergency) bound, the
    workflow is rejected with a hard analysis_incomplete violation — even
    in non-strict mode."""
    monkeypatch.setattr(_verify_mod, "_fixpoint_iteration_limit", lambda *a, **k: 1)
    wf = Workflow(goal="t", input_variables=["items"], steps=[
        WorkflowStep(label="loop", loop=LoopNode(
            collection_ref="items", item_binding="it", body=[
                WorkflowStep(label="tk", tool_call=ToolCallNode(
                    tool_name="tick", arguments={})),
            ])),
    ])
    result = verify(wf, _ticker_policy(), _ticker_registry())  # non-strict
    assert not result.ok
    inc = [v for v in result.violations
           if v.category == "analysis_incomplete" and v.rule_name == "loop_fixpoint"]
    assert len(inc) == 1
    assert "converge" in inc[0].message.lower()
    # It is a hard violation, not a warning.
    assert all("analysis_incomplete" not in w for w in result.warnings)


def test_fail_closed_on_non_monotone_transition(monkeypatch):
    """A non-monotone abstract transition is treated as an internal
    analysis failure and rejects the workflow."""
    monkeypatch.setattr(_verify_mod, "state_leq", lambda a, b: False)
    wf = Workflow(goal="t", input_variables=["items"], steps=[
        WorkflowStep(label="loop", loop=LoopNode(
            collection_ref="items", item_binding="it", body=[
                WorkflowStep(label="tk", tool_call=ToolCallNode(
                    tool_name="tick", arguments={})),
            ])),
    ])
    result = verify(wf, _ticker_policy(), _ticker_registry())
    assert not result.ok
    inc = [v for v in result.violations
           if v.category == "analysis_incomplete" and v.rule_name == "loop_fixpoint"]
    assert len(inc) == 1
    assert "monoton" in inc[0].message.lower()


# ===================================================================
# 5. Lattice laws
# ===================================================================

def _mkval(labels=(), sanitized=(), sources=("t",), prov=()):
    return AbstractValue(
        labels=set(labels), sanitized_for=set(sanitized),
        source_tools=set(sources), provenance=set(prov),
    )


def _sample_values():
    return [
        _mkval(labels={"a"}, sanitized={"r1", "r2"}, sources={"t1"}, prov={"p1"}),
        _mkval(labels={"b"}, sanitized={"r2"}, sources={"t2"}, prov={"p2"}),
        _mkval(labels={"a", "c"}, sanitized={"r1"}, sources={"t1"}, prov={"p1", "p3"}),
    ]


def test_join_value_is_idempotent_commutative_associative():
    a, b, c = _sample_values()
    # idempotent
    assert value_eq(join_value(a, a), a)
    # commutative
    assert value_eq(join_value(a, b), join_value(b, a))
    # associative
    assert value_eq(
        join_value(join_value(a, b), c),
        join_value(a, join_value(b, c)),
    )


def test_join_value_is_upper_bound():
    a, b, _ = _sample_values()
    j = join_value(a, b)
    assert value_leq(a, j)
    assert value_leq(b, j)


def test_source_tools_join_by_union():
    a = _mkval(sources={"t1"})
    b = _mkval(sources={"t2"})
    # possible direct producers accumulate by union (a proper lattice with
    # the empty set as bottom — no magic sentinel that could be a tool name)
    assert join_value(a, b).source_tools == {"t1", "t2"}
    assert value_leq(a, join_value(a, b))
    # source_tools participates in equality / convergence and ordering
    assert not value_eq(a, b)
    assert value_leq(_mkval(sources=set()), a)       # bottom is below anything
    assert not value_leq(b, a)                        # {t2} not subset of {t1}


def test_sanitized_for_is_reverse_subset_ordering():
    more = _mkval(sanitized={"r1", "r2"})
    less = _mkval(sanitized={"r1"})
    # losing a sanitization guarantee moves *up* the lattice
    assert value_leq(more, less)
    assert not value_leq(less, more)
    # join keeps only guarantees on every path (intersection)
    assert join_value(more, less).sanitized_for == {"r1"}


def _sample_states():
    a, b, c = _sample_values()
    s = AbstractState(env={"x": a, "y": b}, automata={"m": {"q0"}})
    t = AbstractState(env={"x": b, "y": c}, automata={"m": {"q1"}})
    u = AbstractState(env={"x": c, "y": a}, automata={"m": {"q0", "q2"}})
    return s, t, u


def test_join_state_is_idempotent_commutative_associative():
    s, t, u = _sample_states()
    assert state_eq(join_state(s, s), s)
    assert state_eq(join_state(s, t), join_state(t, s))
    assert state_eq(
        join_state(join_state(s, t), u),
        join_state(s, join_state(t, u)),
    )


def test_join_state_is_upper_bound():
    s, t, _ = _sample_states()
    j = join_state(s, t)
    assert state_leq(s, j)
    assert state_leq(t, j)


def test_join_state_drops_one_sided_bindings():
    """A binding present in only one state does not survive the join."""
    a, b, _ = _sample_values()
    s = AbstractState(env={"shared": a, "only_s": a}, automata={})
    t = AbstractState(env={"shared": b, "only_t": b}, automata={})
    j = join_state(s, t)
    assert set(j.env.keys()) == {"shared"}


# ===================================================================
# 6. Diagnostic stability
# ===================================================================

def test_loop_violation_reported_once():
    """A violation inside a loop body must appear once, not once per
    fixpoint iteration."""
    wf = Workflow(goal="t", input_variables=["items"], steps=[
        WorkflowStep(label="loop", loop=LoopNode(
            collection_ref="items", item_binding="it", body=[
                WorkflowStep(label="tk", tool_call=ToolCallNode(
                    tool_name="tick", arguments={})),
            ])),
    ])
    result = verify(wf, _ticker_policy(), _ticker_registry())
    auto = [v for v in result.violations if v.category == "automaton"]
    assert len(auto) == 1, "the error transition is revisited but reported once"


# ===================================================================
# 7. Existing scope behavior (abstract execution)
# ===================================================================

def _loop_step(body):
    return WorkflowStep(label="loop", loop=LoopNode(
        collection_ref="coll", item_binding="item", body=body))


def test_loop_bindings_do_not_escape_and_reassignment_persists():
    r = _relay_registry()
    policy = _relay_policy()
    result = VerificationResult()
    env = {
        "coll": AbstractValue(labels={"secret"}, source_tools={"secret_src"},
                              provenance={"secret_src"}),
        "pre": AbstractValue(source_tools={"clean_src"}, provenance={"clean_src"}),
    }
    automaton_states: dict[str, set[str]] = {}
    step = _loop_step([
        WorkflowStep(label="reassign", tool_call=ToolCallNode(
            tool_name="relay", arguments={"x": SymRef(ref="coll")},
            result_binding="pre")),
        WorkflowStep(label="local", tool_call=ToolCallNode(
            tool_name="relay", arguments={"x": SymRef(ref="item")},
            result_binding="tmp")),
    ])
    _verify_loop(step, policy, r, env, automaton_states, result)

    # item binding and body-local result bindings do not escape
    assert "item" not in env
    assert "tmp" not in env
    # outer bindings survive
    assert set(env.keys()) == {"coll", "pre"}
    # assignment to a pre-loop binding persists abstractly (pre now tainted)
    assert "secret" in env["pre"].labels


def test_loop_body_locals_are_fresh_each_iteration():
    """A body-local binding does not carry its value into the next
    iteration: reading it before assignment never observes a prior pass."""
    r = _relay_registry()
    policy = _relay_policy()
    result = VerificationResult()
    env = {
        "coll": AbstractValue(labels={"secret"}, source_tools={"secret_src"},
                              provenance={"secret_src"}),
    }
    automaton_states: dict[str, set[str]] = {}
    # read acc (before it is assigned), then assign it from the tainted item.
    step = _loop_step([
        WorkflowStep(label="read_acc", tool_call=ToolCallNode(
            tool_name="sink", arguments={"data": SymRef(ref="acc")})),
        WorkflowStep(label="write_acc", tool_call=ToolCallNode(
            tool_name="relay", arguments={"x": SymRef(ref="item")},
            result_binding="acc")),
    ])
    _verify_loop(step, policy, r, env, automaton_states, result)

    # If acc persisted across iterations it would be tainted when read,
    # producing a taint violation. Freshness => no such violation.
    taint = [v for v in result.violations if v.category == "taint"]
    assert taint == []
    assert "acc" not in env
