"""Static verification engine.

Explicit verifier — no effect handlers. Walks the workflow AST with
an explicit abstract state, checking policy rules at each step.

The verifier is intentionally conservative: conditionals always explore
both branches regardless of whether the condition is concretely decidable.
This means unreachable branches can still produce violations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import z3

from .conditions import condition_to_z3, expr_names
from .policy import Policy, TaintRule
from .results import VerificationResult, Violation
from .safe_eval import safe_eval
from .tools import ToolRegistry, ToolSpec
from .workflow import Workflow, WorkflowStep, SymRef


# --- Abstract values ---

@dataclass
class AbstractValue:
    """A symbolic value carrying taint labels during verification.

    Every component is set-valued and ordered by inclusion, so the whole
    value lives in a finite powerset lattice (a genuine lattice, not merely
    a join-semilattice):

    Attributes:
        labels: taint label strings on this value.  May-taint: joined by
            union.
        sanitized_for: names of taint rules this value was sanitized for.
            Must-hold: joined by intersection (a guarantee survives only
            if it held on every incoming path).
        source_tools: the set of tools that may have *directly* produced
            this value.  Joined by union; the empty set is bottom (no known
            direct producer).  Modelling this as a set rather than a single
            string with a sentinel avoids a magic value that could collide
            with a real tool name, and makes it a proper lattice.
        provenance: all tools whose outputs contributed to this value,
            transitively through data flow.  May-contribute: joined by
            union.  Used by taint checking to verify that a taint rule's
            declared source_tool actually appears in the data's lineage.
    """

    labels: set[str] = field(default_factory=set)
    sanitized_for: set[str] = field(default_factory=set)
    source_tools: set[str] = field(default_factory=set)
    provenance: set[str] = field(default_factory=set)


# ===================================================================
# Abstract lattice: values, states, joins, and orderings
# ===================================================================
#
# The verification domain is a finite lattice.  An AbstractState captures
# every component that can affect future verification: the variable
# environment and the possible-state set of every security automaton.
# (Diagnostics are not part of the semantic state — they do not influence
# later transfer — but fixpoint iteration must not duplicate them.)
#
# Orderings (x <= y means "x is more precise / less-or-equal-information"):
#   - labels, provenance     grow by subset inclusion;
#   - source_tools           grow by subset inclusion (set of possible
#                             direct producers; bottom is the empty set);
#   - automaton state sets    grow by subset inclusion;
#   - sanitized_for           shrinks (reverse subset inclusion: losing a
#                             sanitization guarantee moves *up*);
#   - a missing binding is treated as TOP (unusable): joining a value with
#     "absent" yields "absent", so a binding present on only one path does
#     not escape the join.
#
# join is the least upper bound; it is idempotent, commutative, and
# associative (verified by tests).


def join_value(a: AbstractValue, b: AbstractValue) -> AbstractValue:
    """Least upper bound of two abstract values.

    labels/provenance/source_tools union (may-information), sanitized_for
    intersect (must-information).
    """
    return AbstractValue(
        labels=a.labels | b.labels,
        sanitized_for=a.sanitized_for & b.sanitized_for,
        source_tools=a.source_tools | b.source_tools,
        provenance=a.provenance | b.provenance,
    )


def value_leq(a: AbstractValue, b: AbstractValue) -> bool:
    """Whether ``a`` is below-or-equal ``b`` in the value lattice."""
    return (
        a.labels <= b.labels
        and a.provenance <= b.provenance
        and a.source_tools <= b.source_tools
        and a.sanitized_for >= b.sanitized_for
    )


def value_eq(a: AbstractValue, b: AbstractValue) -> bool:
    """Complete structural equality of two abstract values."""
    return (
        a.labels == b.labels
        and a.sanitized_for == b.sanitized_for
        and a.source_tools == b.source_tools
        and a.provenance == b.provenance
    )


@dataclass
class AbstractState:
    """The complete semantic abstract state threaded through verification.

    Holds every mutable component that affects later transfer:
    the variable environment and each automaton's possible-state set.
    """

    env: dict[str, AbstractValue] = field(default_factory=dict)
    automata: dict[str, set[str]] = field(default_factory=dict)

    def copy(self) -> "AbstractState":
        return AbstractState(env=_copy_env(self.env), automata=_copy_auto(self.automata))


def join_state(a: AbstractState, b: AbstractState) -> AbstractState:
    """Least upper bound of two abstract states.

    A binding present in only one state is dropped (absent == TOP), so it
    does not escape the join.  Automaton possible-state sets union.
    """
    env = {
        k: join_value(a.env[k], b.env[k])
        for k in (a.env.keys() & b.env.keys())
    }
    automata = {
        name: a.automata.get(name, set()) | b.automata.get(name, set())
        for name in (a.automata.keys() | b.automata.keys())
    }
    return AbstractState(env=env, automata=automata)


def state_leq(a: AbstractState, b: AbstractState) -> bool:
    """Whether ``a`` is below-or-equal ``b`` in the state lattice.

    A missing binding is TOP, so ``a <= b`` requires every binding present
    in ``b`` to also be present in ``a`` (and pointwise below it), while
    bindings only in ``a`` are permitted (they sit below TOP).
    """
    if not (b.env.keys() <= a.env.keys()):
        return False
    for k in b.env:
        if not value_leq(a.env[k], b.env[k]):
            return False
    for name in a.automata.keys() | b.automata.keys():
        if not (a.automata.get(name, set()) <= b.automata.get(name, set())):
            return False
    return True


def state_eq(a: AbstractState, b: AbstractState) -> bool:
    """Complete structural-equality convergence predicate for states."""
    if a.env.keys() != b.env.keys():
        return False
    for k in a.env:
        if not value_eq(a.env[k], b.env[k]):
            return False
    if a.automata.keys() != b.automata.keys():
        return False
    for name in a.automata:
        if a.automata[name] != b.automata[name]:
            return False
    return True


# --- Public API ---

def verify(
    workflow: Workflow,
    policy: Policy,
    registry: ToolRegistry,
    *,
    strict: bool = False,
) -> VerificationResult:
    """Run all verification passes on a workflow."""
    result = VerificationResult()

    # Pass 1: well-formedness (scope checking)
    for v in _check_scope(workflow):
        result.add(v)

    # Pass 2: abstract execution with policy checking
    env: dict[str, AbstractValue] = {
        name: AbstractValue(source_tools={"input"}, provenance={"input"})
        for name in workflow.input_variables
    }
    automaton_states: dict[str, set[str]] = {
        a.name: {a.initial_state} for a in policy.automata
    }
    _verify_steps(workflow.steps, policy, registry, env, automaton_states, result)

    # Fixpoint iteration revisits the same source steps, so deduplicate
    # diagnostics that differ only by that revisiting.  Ordering is
    # preserved deterministically (first occurrence wins).
    _dedup_diagnostics(result)

    # Strict mode: promote warnings to violations
    if strict:
        for w in result.warnings:
            result.add(Violation(category="unparseable", message=w, step_label=""))

    return result


def _dedup_diagnostics(result: VerificationResult) -> None:
    """Drop duplicate violations/warnings, preserving first-seen order.

    Violations are keyed by (category, rule_name, step_label, message);
    warnings by their complete message.  Two diagnostics with identical
    keys are indistinguishable, so collapsing them is sound and prevents
    loop fixpoint revisiting from inflating counts.
    """
    seen_v: set[tuple[str, str, str, str]] = set()
    unique_v: list[Violation] = []
    for v in result.violations:
        key = (v.category, v.rule_name, v.step_label, v.message)
        if key not in seen_v:
            seen_v.add(key)
            unique_v.append(v)
    result.violations = unique_v

    seen_w: set[str] = set()
    unique_w: list[str] = []
    for w in result.warnings:
        if w not in seen_w:
            seen_w.add(w)
            unique_w.append(w)
    result.warnings = unique_w


# ===================================================================
# Pass 1: Scope checking
# ===================================================================

def _check_scope(workflow: Workflow) -> list[Violation]:
    violations: list[Violation] = []
    bound: set[str] = set(workflow.input_variables)
    _check_steps_scope(workflow.steps, bound, violations)
    return violations


def _check_steps_scope(
    steps: list[WorkflowStep], bound: set[str], violations: list[Violation],
) -> None:
    for step in steps:
        if step.tool_call:
            tc = step.tool_call
            for ref in _collect_refs(tc.arguments):
                if ref not in bound:
                    violations.append(Violation(
                        category="well_formedness",
                        message=f"Undefined reference @{ref}",
                        step_label=step.label,
                        rule_name="undefined_ref",
                    ))
            if tc.result_binding:
                bound.add(tc.result_binding)

        elif step.conditional:
            c = step.conditional
            for ref in _condition_undefined_refs(c.condition, bound):
                violations.append(Violation(
                    category="well_formedness",
                    message=f"Undefined reference @{ref}",
                    step_label=step.label,
                    rule_name="undefined_ref",
                ))
            then_bound = set(bound)
            else_bound = set(bound)
            _check_steps_scope(c.then_steps, then_bound, violations)
            _check_steps_scope(c.else_steps, else_bound, violations)
            bound.update(then_bound & else_bound)

        elif step.loop:
            lp = step.loop
            if lp.collection_ref not in bound:
                violations.append(Violation(
                    category="well_formedness",
                    message=f"Undefined reference @{lp.collection_ref}",
                    step_label=step.label,
                    rule_name="undefined_ref",
                ))
            if lp.item_binding in bound:
                violations.append(Violation(
                    category="well_formedness",
                    message=f"Loop item binding '{lp.item_binding}' shadows outer variable",
                    step_label=step.label,
                    rule_name="shadowed_binding",
                ))
            loop_bound = set(bound)
            loop_bound.add(lp.item_binding)
            _check_steps_scope(lp.body, loop_bound, violations)


def _collect_refs(val: Any) -> list[str]:
    refs: list[str] = []
    _walk_refs(val, refs)
    return refs


def _walk_refs(val: Any, refs: list[str]) -> None:
    if isinstance(val, SymRef):
        refs.append(val.ref)
    elif isinstance(val, dict):
        for v in val.values():
            _walk_refs(v, refs)
    elif isinstance(val, list):
        for v in val:
            _walk_refs(v, refs)


def _condition_undefined_refs(condition: str, bound: set[str]) -> list[str]:
    """Find names in a condition that are not in scope.

    Uses expr_names which already excludes keywords, len, domain_of, etc.
    """
    return [n for n in expr_names(condition) if n not in bound]


# ===================================================================
# Pass 2: Abstract execution
# ===================================================================

def _verify_steps(
    steps: list[WorkflowStep],
    policy: Policy,
    registry: ToolRegistry,
    env: dict[str, AbstractValue],
    automaton_states: dict[str, set[str]],
    result: VerificationResult,
) -> None:
    for step in steps:
        if step.tool_call:
            _verify_tool_call(step, policy, registry, env, automaton_states, result)
        elif step.conditional:
            _verify_conditional(step, policy, registry, env, automaton_states, result)
        elif step.loop:
            _verify_loop(step, policy, registry, env, automaton_states, result)


def _verify_tool_call(
    step: WorkflowStep,
    policy: Policy,
    registry: ToolRegistry,
    env: dict[str, AbstractValue],
    automaton_states: dict[str, set[str]],
    result: VerificationResult,
) -> None:
    tc = step.tool_call
    assert tc is not None
    spec = registry.get_spec(tc.tool_name)

    # 1. Allowlist
    if tc.tool_name not in set(policy.allowed_tools):
        result.add(Violation(
            category="allowlist",
            message=f"Tool '{tc.tool_name}' is not in the allowed tools list",
            step_label=step.label,
            rule_name="allowed_tools",
        ))

    # 2. Missing spec
    if tc.tool_name in set(policy.allowed_tools) and spec is None:
        result.add(Violation(
            category="missing_spec",
            message=f"Tool '{tc.tool_name}' is allowed but has no registered spec",
            step_label=step.label,
            rule_name="missing_spec",
        ))

    # 3. Resolve arguments
    resolved = _resolve_abstract(tc.arguments, env)

    # 4. Taint checks
    _check_taint_rules(tc.tool_name, resolved, step.label, spec, policy, registry, result)

    # 5. Preconditions (Z3)
    constants = _collect_policy_constants(policy)
    if spec is not None:
        for pre in spec.preconditions:
            _check_z3_condition(
                "precondition", tc.tool_name, pre, resolved, None,
                step.label, constants, spec, result,
            )

    # 6. Automata
    _check_automata(policy, tc.tool_name, resolved, step.label, automaton_states, result)

    # 7. Build abstract result with provenance tracking
    input_labels = _collect_labels(resolved)
    input_provenance = _collect_provenance(resolved)
    spec_labels = set(spec.source_labels) if spec else set()
    abstract_result = AbstractValue(
        labels=spec_labels | input_labels,
        sanitized_for=set(),
        source_tools={tc.tool_name},
        provenance={tc.tool_name} | input_provenance,
    )

    # 8. Apply sanitizer logic
    if spec is not None:
        for rule in policy.taint_rules:
            if tc.tool_name in rule.sanitizers:
                abstract_result.sanitized_for.add(rule.name)

    # 9. Postconditions (Z3)
    if spec is not None:
        for post in spec.postconditions:
            _check_z3_condition(
                "postcondition", tc.tool_name, post, resolved, abstract_result,
                step.label, constants, spec, result,
            )

    # 10. Frame conditions (Z3)
    if spec is not None:
        for frame in spec.frame_conditions:
            _check_z3_condition(
                "frame", tc.tool_name, frame, resolved, None,
                step.label, constants, spec, result,
            )

    # 11. Bind result
    if tc.result_binding:
        env[tc.result_binding] = abstract_result


def _verify_conditional(
    step: WorkflowStep,
    policy: Policy,
    registry: ToolRegistry,
    env: dict[str, AbstractValue],
    automaton_states: dict[str, set[str]],
    result: VerificationResult,
) -> None:
    c = step.conditional
    assert c is not None

    # Always explore both branches (intentionally conservative).  Each
    # branch runs on a fresh copy of the entry state; the post-state is the
    # lattice join.  Because join drops one-branch-only bindings, a binding
    # created in only one branch does not escape the conditional.
    entry = AbstractState(env=_copy_env(env), automata=_copy_auto(automaton_states))

    then_state = entry.copy()
    _verify_steps(c.then_steps, policy, registry, then_state.env, then_state.automata, result)

    else_state = entry.copy()
    _verify_steps(c.else_steps, policy, registry, else_state.env, else_state.automata, result)

    joined = join_state(then_state, else_state)
    _install_state(joined, env, automaton_states)


def _verify_loop(
    step: WorkflowStep,
    policy: Policy,
    registry: ToolRegistry,
    env: dict[str, AbstractValue],
    automaton_states: dict[str, set[str]],
    result: VerificationResult,
) -> None:
    """Verify a loop as the least fixpoint of ``H = entry ⊔ body(H)``.

    A loop runs zero or more times, so the result must over-approximate
    every iteration count, including zero (the collection may be empty).
    The domain is finite, so the ascending chain terminates; if it somehow
    does not (a bug or non-monotone transfer) we reject fail-closed.
    """
    lp = step.loop
    assert lp is not None

    # 1-2. Save the complete pre-loop state and its outer key set.
    entry_state = AbstractState(env=_copy_env(env), automata=_copy_auto(automaton_states))
    outer_keys = set(entry_state.env.keys())

    # 3. Derive the loop item's abstract value from the collection *as it
    # exists on loop entry*.  The executor iterates over a snapshot
    # (tuple(collection)) taken on entry, so neither rebinding the
    # collection variable nor mutating it in the body can change the items
    # seen on later iterations.
    collection = entry_state.env.get(lp.collection_ref)
    if isinstance(collection, AbstractValue):
        item_template = AbstractValue(
            labels=set(collection.labels),
            sanitized_for=set(collection.sanitized_for),
            source_tools=set(collection.source_tools),
            provenance=set(collection.provenance),
        )
    else:
        item_template = AbstractValue(source_tools={"literal"})

    # 4. head = entry_state represents the zero-iteration path.
    head = entry_state.copy()
    limit = _fixpoint_iteration_limit(entry_state, policy, registry, outer_keys)

    converged = False
    failure: str | None = None
    next_head = head
    for _ in range(limit):
        # 5. Copy head, install a fresh item binding, analyze the body.
        work = head.copy()
        work.env[lp.item_binding] = AbstractValue(
            labels=set(item_template.labels),
            sanitized_for=set(item_template.sanitized_for),
            source_tools=set(item_template.source_tools),
            provenance=set(item_template.provenance),
        )
        _verify_steps(lp.body, policy, registry, work.env, work.automata, result)

        # Remove the item binding and all body-local bindings, retaining
        # updates to bindings that existed before the loop.
        body_exit = AbstractState(
            env={k: v for k, v in work.env.items() if k in outer_keys},
            automata=work.automata,
        )

        # The entry join is essential: it covers the empty-collection
        # (zero-iteration) case and guarantees the chain is ascending.
        next_head = join_state(entry_state, body_exit)

        # 6. Equality => fixpoint reached.
        if state_eq(next_head, head):
            converged = True
            break
        # 7. Otherwise the chain must be monotone.
        if not state_leq(head, next_head):
            failure = "non_monotone"
            break
        head = next_head
    else:
        # 8. Ran out of the (finite) iteration budget without convergence.
        failure = "diverged"

    if converged:
        final = head
    else:
        _add_loop_incomplete(step, lp, failure or "diverged", result)
        # Continue only for diagnostics with a conservative over-approx;
        # the workflow is already rejected and can never be accepted.
        final = join_state(head, next_head) if failure == "non_monotone" else head

    _install_state(final, env, automaton_states)


def _install_state(
    state: AbstractState,
    env: dict[str, AbstractValue],
    automaton_states: dict[str, set[str]],
) -> None:
    """Replace the caller's env and automaton state in place."""
    env.clear()
    env.update(_copy_env(state.env))
    automaton_states.clear()
    automaton_states.update(_copy_auto(state.automata))


def _fixpoint_iteration_limit(
    entry_state: AbstractState,
    policy: Policy,
    registry: ToolRegistry,
    outer_keys: set[str],
) -> int:
    """A conservative bound on the height of the ascending chain.

    Every fixpoint step that is not the fixpoint strictly increases the
    state in at least one finite coordinate, so the chain cannot be longer
    than the sum of the per-coordinate heights:

      - each outer binding can gain every taint label (|L|), gain every
        provenance atom (|P|), gain every direct-source atom (also bounded
        by |P|), and lose every sanitization (|S|);
      - each automaton's possible-state set can gain every state.

    The bound is generous (legitimate analyses converge far below it); it
    exists purely as a fatal emergency guard against an implementation bug
    or a future non-finite domain.
    """
    labels: set[str] = set()
    # Universe of producer atoms — covers both provenance and source_tools,
    # which only ever hold tool names or these fixed pseudo-sources.
    producers: set[str] = {"input", "literal", "unknown"}
    for spec in registry.all_specs().values():
        labels.update(spec.source_labels)
        labels.update(spec.sink_labels)
        producers.add(spec.name)
    producers.update(policy.allowed_tools)
    for v in entry_state.env.values():
        labels.update(v.labels)
        producers.update(v.provenance)
        producers.update(v.source_tools)
    sanitizers = {r.name for r in policy.taint_rules}

    per_binding = len(labels) + 2 * len(producers) + len(sanitizers)
    auto_states = sum(len(a.states) for a in policy.automata)
    height = len(outer_keys) * per_binding + auto_states + 1
    # Floor for tiny domains; absolute ceiling guards pathological sizes.
    return min(max(height, 32), 1_000_000)


def _add_loop_incomplete(
    step: WorkflowStep,
    lp: Any,
    cause: str,
    result: VerificationResult,
) -> None:
    """Record a fatal ``analysis_incomplete`` violation for a loop.

    This rejects the workflow regardless of ``strict``: convergence of the
    loop abstraction could not be soundly established, so the workflow can
    never be accepted.
    """
    if cause == "non_monotone":
        why = (
            "abstract state transition was not monotone "
            "(internal analysis failure)"
        )
    else:
        why = (
            "did not converge within the sound/emergency iteration bound"
        )
    result.add(Violation(
        category="analysis_incomplete",
        message=(
            f"Loop '{step.label}' over @{lp.collection_ref}: {why}; "
            f"loop convergence could not be established, workflow rejected"
        ),
        step_label=step.label,
        rule_name="loop_fixpoint",
    ))


# ===================================================================
# Taint checking
# ===================================================================

def _check_taint_rules(
    tool_name: str,
    resolved: dict[str, Any],
    step_label: str,
    spec: ToolSpec | None,
    policy: Policy,
    registry: ToolRegistry,
    result: VerificationResult,
) -> None:
    for rule in policy.taint_rules:
        if rule.sink_tool != tool_name and rule.sink_tool != "*":
            continue

        if rule.sink_param == "*" and spec is not None:
            for p in spec.params:
                if p.is_taint_sink:
                    expanded = TaintRule(
                        name=rule.name,
                        source_tool=rule.source_tool,
                        sink_tool=tool_name,
                        sink_param=p.name,
                        condition=rule.condition,
                        sanitizers=rule.sanitizers,
                    )
                    _check_single_taint(expanded, resolved, step_label, spec, policy, registry, result)
        else:
            _check_single_taint(rule, resolved, step_label, spec, policy, registry, result)


def _check_single_taint(
    rule: TaintRule,
    resolved: dict[str, Any],
    step_label: str,
    spec: ToolSpec | None,
    policy: Policy,
    registry: ToolRegistry,
    result: VerificationResult,
) -> None:
    # Find ALL abstract values nested under the sink param, not just the first.
    abstracts = _find_all_abstracts(resolved.get(rule.sink_param))
    if not abstracts:
        return

    for sym in abstracts:
        if _is_tainted_for_rule(sym, rule, policy, registry, resolved):
            source_desc = "any source" if rule.source_tool == "*" else f"'{rule.source_tool}'"
            result.add(Violation(
                category="taint",
                message=f"Tainted data from {source_desc} flows to '{rule.sink_tool}.{rule.sink_param}'",
                step_label=step_label,
                rule_name=rule.name,
            ))
            return  # one violation per rule per step


def _is_tainted_for_rule(
    sym: AbstractValue,
    rule: TaintRule,
    policy: Policy,
    registry: ToolRegistry,
    resolved: dict[str, Any],
) -> bool:
    """Check whether a single AbstractValue violates a taint rule."""
    # Check source match: both label overlap AND provenance
    if rule.source_tool == "*":
        if not sym.labels:
            return False
    else:
        source_spec = registry.get_spec(rule.source_tool)
        if source_spec is None:
            return False
        if not (sym.labels & set(source_spec.source_labels)):
            return False
        # Provenance check: the declared source tool must actually be
        # in this value's data lineage.
        if rule.source_tool not in sym.provenance:
            return False

    # Already sanitized?
    if rule.name in sym.sanitized_for:
        return False

    # Conditional taint rule?
    if rule.condition:
        eval_env: dict[str, Any] = {}
        eval_env.update(resolved)
        eval_env.update(_collect_policy_constants(policy))
        refs = expr_names(rule.condition)
        has_symbolic = any(_contains_abstract(eval_env.get(n)) for n in refs)
        if not has_symbolic:
            try:
                if not safe_eval(rule.condition, eval_env):
                    return False
            except Exception:
                pass  # can't evaluate — apply rule conservatively

    return True


# ===================================================================
# Z3 condition checking
# ===================================================================

def _check_z3_condition(
    category: str,
    tool_name: str,
    condition: str,
    resolved: dict[str, Any],
    abstract_result: AbstractValue | None,
    step_label: str,
    constants: dict[str, Any],
    spec: ToolSpec,
    result: VerificationResult,
) -> None:
    # Build Z3 env
    z3_env: dict[str, Any] = {}
    has_symbolic: set[str] = set()

    for p in spec.params:
        val = resolved.get(p.name)
        if val is None:
            continue
        if isinstance(val, AbstractValue):
            z3_env[p.name] = _make_z3_symbolic(p.name, p.type)
            has_symbolic.add(p.name)
        else:
            z3_val = _make_z3_literal(val)
            if z3_val is not None:
                z3_env[p.name] = z3_val

    # Postcondition: add "result" using the tool's declared return_type
    if category == "postcondition" and "result" in condition:
        if abstract_result is not None:
            z3_env["result"] = _make_z3_symbolic("result", spec.return_type)
            has_symbolic.add("result")

    if not z3_env:
        return

    z3_env.update(constants)

    # Forall conditions: non-vacuity check
    if condition.strip().startswith("forall "):
        _check_z3_forall(category, condition, step_label, spec, z3_env, has_symbolic, result)
        return

    # Translate and check — catch Z3 sort/type errors
    try:
        z3_expr = condition_to_z3(condition, z3_env)
    except Exception:
        z3_expr = None

    if z3_expr is None:
        result.warn(
            f"Could not parse {category} '{condition}' for "
            f"'{spec.name}' into Z3 — skipped"
        )
        return

    try:
        solver = z3.Solver()
        solver.set("timeout", 5000)
        solver.add(z3.Not(z3_expr))
        check = solver.check()
    except z3.Z3Exception:
        result.warn(
            f"Z3 error checking {category} '{condition}' for "
            f"'{spec.name}' — skipped"
        )
        return

    if check == z3.sat:
        cond_refs = expr_names(condition) & set(z3_env.keys())
        referenced_concrete = {p for p in z3_env if p not in has_symbolic}
        is_definite = cond_refs.issubset(referenced_concrete)
        severity = "violated" if is_definite else "could be violated"

        if not is_definite and category == "postcondition" and "result" in has_symbolic:
            result.warn(
                f"{category.title()} '{condition}' for '{spec.name}' "
                f"could be violated (symbolic result — checked at runtime)"
            )
        else:
            result.add(Violation(
                category=category,
                message=f"{category.title()} '{condition}' for '{spec.name}' {severity}",
                step_label=step_label,
                rule_name=f"{category}:{spec.name}:{condition}",
            ))


def _check_z3_forall(
    category: str,
    condition: str,
    step_label: str,
    spec: ToolSpec,
    z3_env: dict[str, Any],
    has_symbolic: set[str],
    result: VerificationResult,
) -> None:
    """Check a forall frame/postcondition for non-vacuity."""
    z3_only = {k: v for k, v in z3_env.items() if isinstance(v, z3.ExprRef)}
    parsed = _parse_forall_condition(condition, z3_only)
    if parsed is None:
        result.warn(
            f"Could not parse {category} '{condition}' for "
            f"'{spec.name}' into Z3 — skipped"
        )
        return

    antecedent, _qvar = parsed

    solver = z3.Solver()
    solver.set("timeout", 5000)
    solver.add(antecedent)
    check = solver.check()

    if check == z3.unsat:
        cond_refs = set(re.findall(r'\b(\w+)\b', condition)) & set(z3_only.keys())
        referenced_concrete = {p for p in z3_only if p not in has_symbolic}
        is_definite = cond_refs.issubset(referenced_concrete)
        severity = "vacuous" if is_definite else "could be vacuous"
        result.add(Violation(
            category=category,
            message=(
                f"{category.title()} '{condition}' for '{spec.name}' "
                f"is {severity} — scope covers everything"
            ),
            step_label=step_label,
            rule_name=f"{category}:{spec.name}:{condition}",
        ))
    elif check == z3.unknown:
        result.warn(
            f"{category.title()} '{condition}' for '{spec.name}' "
            f"— Z3 timeout on non-vacuity check"
        )


def _parse_forall_condition(
    condition: str, z3_vars: dict[str, z3.ExprRef],
) -> tuple[z3.BoolRef, z3.ExprRef] | None:
    m = re.match(
        r"forall\s+(\w+)\s*:\s*(not\s+)?matches\((\w+)\s*,\s*(\w+)\)"
        r"\s+implies\s+(\w+)\((\w+)\)",
        condition.strip(),
    )
    if not m:
        return None

    qvar_name = m.group(1)
    negated = m.group(2) is not None
    match_var = m.group(3)
    pattern_param = m.group(4)

    if match_var != qvar_name:
        return None
    if pattern_param not in z3_vars:
        return None

    qvar = z3.String(qvar_name)
    match_expr = _build_glob_match(qvar, z3_vars[pattern_param])
    antecedent = z3.Not(match_expr) if negated else match_expr

    return antecedent, qvar


def _build_glob_match(var: z3.ExprRef, pattern: z3.ExprRef) -> z3.BoolRef:
    if z3.is_string_value(pattern):
        p = pattern.as_string()
        if p == "*":
            return z3.BoolVal(True)
        if p.startswith("*") and not p.endswith("*"):
            return z3.SuffixOf(z3.StringVal(p[1:]), var)
        if p.endswith("*") and not p.startswith("*"):
            return z3.PrefixOf(z3.StringVal(p[:-1]), var)
        return var == pattern
    glob_fn = z3.Function(
        "glob_matches", z3.StringSort(), z3.StringSort(), z3.BoolSort(),
    )
    return glob_fn(var, pattern)


# ===================================================================
# Automaton checking
# ===================================================================

# Abstract truth values for a guard under partial information.
_GUARD_TRUE = "true"
_GUARD_FALSE = "false"
_GUARD_UNKNOWN = "unknown"


def _check_automata(
    policy: Policy,
    tool_name: str,
    resolved: dict[str, Any],
    step_label: str,
    automaton_states: dict[str, set[str]],
    result: VerificationResult,
) -> None:
    """Abstract successor-state transfer with ordered first-match semantics.

    The runtime evaluates the matching transitions of a state in declaration
    order and fires the first whose guard is true (otherwise it stays put).
    Statically a guard is TRUE / FALSE / UNKNOWN.  An UNKNOWN guard *forks*:
    the true branch takes the transition, the false branch falls through to
    later transitions — so we must keep scanning rather than stop at the
    first non-false guard.  Stopping early would hide a competing guarded
    transition from the same state (e.g. an error transition declared after
    a benign one), making the transfer unsound.
    """
    for automaton in policy.automata:
        current_states = automaton_states[automaton.name]
        error_states = {s.name for s in automaton.states if s.is_error}
        next_states: set[str] = set()

        for current in current_states:
            fallthrough_possible = True
            # Whether the path that *reaches* the current transition is
            # certain (all earlier guards were definitely false).  Once an
            # UNKNOWN guard is seen, later transitions are only conditionally
            # reachable, so even a definitely-true guard is "could reach".
            reached_definitely = True

            for trans in automaton.transitions:
                if trans.from_state != current or trans.tool_name != tool_name:
                    continue

                truth = _abstract_guard_truth(
                    trans.condition, resolved, automaton, tool_name,
                    step_label, result,
                )

                if truth == _GUARD_FALSE:
                    continue

                # TRUE or UNKNOWN: taking this transition is possible.
                next_states.add(trans.to_state)
                if trans.to_state in error_states:
                    definite = truth == _GUARD_TRUE and reached_definitely
                    if definite:
                        message = (
                            f"Security automaton '{automaton.name}' "
                            f"reached error state '{trans.to_state}' "
                            f"on tool call '{tool_name}'"
                        )
                    else:
                        message = (
                            f"Security automaton '{automaton.name}' "
                            f"could reach error state '{trans.to_state}' "
                            f"on tool call '{tool_name}' (symbolic argument)"
                        )
                    result.add(Violation(
                        category="automaton",
                        message=message,
                        step_label=step_label,
                        rule_name=automaton.name,
                    ))

                if truth == _GUARD_TRUE:
                    # First-match: a definite guard consumes the residual
                    # path — no later transition and no staying put.
                    fallthrough_possible = False
                    break

                # UNKNOWN fork: the true branch took the transition above;
                # the false branch continues to later transitions.
                reached_definitely = False

            if fallthrough_possible:
                next_states.add(current)

        automaton_states[automaton.name] = next_states


def _abstract_guard_truth(
    condition: str | None,
    resolved: dict[str, Any],
    automaton: Any,
    tool_name: str,
    step_label: str,
    result: VerificationResult,
) -> str:
    """Evaluate a transition guard to TRUE / FALSE / UNKNOWN.

    A guard whose referenced arguments are (recursively) symbolic is
    UNKNOWN.  A guard that raises during evaluation fails closed: it is a
    hard ``analysis_incomplete`` violation and is treated conservatively as
    UNKNOWN (so its transition is still explored).
    """
    if not condition:
        return _GUARD_TRUE

    eval_env: dict[str, Any] = {}
    eval_env.update(resolved)
    eval_env.update(automaton.constants)
    refs = expr_names(condition)
    if any(_contains_abstract(eval_env.get(n)) for n in refs):
        return _GUARD_UNKNOWN

    try:
        fires = safe_eval(condition, eval_env)
    except Exception:
        result.add(Violation(
            category="analysis_incomplete",
            message=(
                f"Security automaton '{automaton.name}' guard '{condition}' "
                f"on tool call '{tool_name}' could not be evaluated; "
                f"treated conservatively as unknown"
            ),
            step_label=step_label,
            rule_name="automaton_guard",
        ))
        return _GUARD_UNKNOWN

    return _GUARD_TRUE if fires else _GUARD_FALSE


# ===================================================================
# Helpers: resolve, collect, copy
# ===================================================================

def _resolve_abstract(arguments: dict[str, Any], env: dict[str, AbstractValue]) -> dict[str, Any]:
    return {k: _resolve_val(v, env) for k, v in arguments.items()}


def _resolve_val(val: Any, env: dict[str, AbstractValue]) -> Any:
    if isinstance(val, SymRef):
        return env.get(val.ref, AbstractValue())
    if isinstance(val, dict):
        return {k: _resolve_val(v, env) for k, v in val.items()}
    if isinstance(val, list):
        return [_resolve_val(v, env) for v in val]
    return val


def _collect_labels(val: Any) -> set[str]:
    """Collect taint labels recursively from a resolved value tree."""
    labels: set[str] = set()
    _walk_labels(val, labels)
    return labels


def _walk_labels(val: Any, labels: set[str]) -> None:
    if isinstance(val, AbstractValue):
        labels.update(val.labels)
    elif isinstance(val, dict):
        for v in val.values():
            _walk_labels(v, labels)
    elif isinstance(val, list):
        for v in val:
            _walk_labels(v, labels)


def _collect_provenance(val: Any) -> set[str]:
    """Collect provenance (contributing tool names) recursively."""
    prov: set[str] = set()
    _walk_provenance(val, prov)
    return prov


def _walk_provenance(val: Any, prov: set[str]) -> None:
    if isinstance(val, AbstractValue):
        prov.update(val.provenance)
    elif isinstance(val, dict):
        for v in val.values():
            _walk_provenance(v, prov)
    elif isinstance(val, list):
        for v in val:
            _walk_provenance(v, prov)


def _find_all_abstracts(val: Any) -> list[AbstractValue]:
    """Find ALL AbstractValues nested in a resolved value tree."""
    results: list[AbstractValue] = []
    _walk_abstracts(val, results)
    return results


def _contains_abstract(val: Any) -> bool:
    """Whether a resolved value is (recursively) symbolic.

    A list or dict containing an AbstractValue anywhere is symbolic, not
    just a value that is itself an AbstractValue.
    """
    if isinstance(val, AbstractValue):
        return True
    if isinstance(val, dict):
        return any(_contains_abstract(v) for v in val.values())
    if isinstance(val, list):
        return any(_contains_abstract(v) for v in val)
    return False


def _walk_abstracts(val: Any, results: list[AbstractValue]) -> None:
    if isinstance(val, AbstractValue):
        results.append(val)
    elif isinstance(val, dict):
        for v in val.values():
            _walk_abstracts(v, results)
    elif isinstance(val, list):
        for v in val:
            _walk_abstracts(v, results)


def _copy_env(env: dict[str, AbstractValue]) -> dict[str, AbstractValue]:
    return {
        k: AbstractValue(
            labels=set(v.labels),
            sanitized_for=set(v.sanitized_for),
            source_tools=set(v.source_tools),
            provenance=set(v.provenance),
        )
        for k, v in env.items()
    }


def _copy_auto(states: dict[str, set[str]]) -> dict[str, set[str]]:
    return {k: set(v) for k, v in states.items()}


def _collect_policy_constants(policy: Policy) -> dict[str, Any]:
    constants: dict[str, Any] = {}
    for automaton in policy.automata:
        constants.update(automaton.constants)
    return constants


# ===================================================================
# Z3 helpers
# ===================================================================

def _make_z3_symbolic(name: str, type_hint: str) -> z3.ExprRef:
    if type_hint in ("int", "float"):
        return z3.Int(name)
    if type_hint == "bool":
        return z3.Bool(name)
    return z3.String(name)


def _make_z3_literal(val: Any) -> z3.ExprRef | None:
    if isinstance(val, str):
        return z3.StringVal(val)
    if isinstance(val, bool):
        return z3.BoolVal(val)
    if isinstance(val, int):
        return z3.IntVal(val)
    if isinstance(val, float):
        return z3.IntVal(int(val))
    return None
