"""Safe expression evaluator using AST allowlisting.

Replaces Python eval() with a restricted evaluator that only permits:
- Literals (str, int, float, bool, None)
- Variable names (looked up in a provided env dict)
- Lists and tuples
- Comparisons (==, !=, <, >, <=, >=, in, not in, is, is not)
- Boolean operators (and, or, not)
- len() and domain_of() calls
"""

from __future__ import annotations

import ast
import operator
from typing import Any

_CMP_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}


def safe_eval(expr: str, env: dict[str, Any]) -> Any:
    """Evaluate a simple expression safely against *env*."""
    tree = ast.parse(expr, mode="eval")
    return _eval_node(tree.body, env)


# Comparison operators safe_eval accepts: the _CMP_OPS table plus the
# membership operators handled specially in the Compare case below.
_ALLOWED_CMP_OPS: tuple[type, ...] = tuple(_CMP_OPS) + (ast.In, ast.NotIn)


def validate_safe_expr(expr: str) -> None:
    """Validate that *expr* is within the safe_eval grammar, without evaluating.

    Raises ``SyntaxError`` if *expr* is malformed, or ``ValueError`` if it
    uses a form that ``safe_eval`` would reject (e.g. attribute access or an
    unsupported function call).  This lets callers reject an unsupported
    guard up front — before any symbolic shortcut — rather than only when a
    concrete evaluation happens to reach ``_eval_node``'s rejection.  It
    mirrors ``_eval_node``'s accepted forms (kept adjacent) so the grammar
    cannot drift between checking and evaluation.
    """
    tree = ast.parse(expr, mode="eval")
    _validate_node(tree.body)


def _validate_node(node: ast.expr) -> None:
    match node:
        case ast.Constant() | ast.Name():
            return

        case ast.List(elts=elts) | ast.Tuple(elts=elts):
            for e in elts:
                _validate_node(e)

        case ast.UnaryOp(op=ast.Not(), operand=operand):
            _validate_node(operand)

        case ast.BoolOp(values=values):
            for v in values:
                _validate_node(v)

        case ast.Compare(left=left, ops=ops, comparators=comparators):
            for op in ops:
                if not isinstance(op, _ALLOWED_CMP_OPS):
                    raise ValueError(f"disallowed comparison: {type(op).__name__}")
            _validate_node(left)
            for c in comparators:
                _validate_node(c)

        case ast.Call(func=ast.Name(id=name), args=[arg]) if (
            name in ("len", "domain_of") and not node.keywords
        ):
            _validate_node(arg)

        case _:
            raise ValueError(f"disallowed expression: {type(node).__name__}")


def _eval_node(node: ast.expr, env: dict[str, Any]) -> Any:
    match node:
        case ast.Constant(value=v):
            return v

        case ast.Name(id=name):
            try:
                return env[name]
            except KeyError:
                raise ValueError(f"undefined variable: {name!r}")

        case ast.List(elts=elts):
            return [_eval_node(e, env) for e in elts]

        case ast.Tuple(elts=elts):
            return tuple(_eval_node(e, env) for e in elts)

        case ast.UnaryOp(op=ast.Not(), operand=operand):
            return not _eval_node(operand, env)

        case ast.BoolOp(op=op, values=values):
            if isinstance(op, ast.And):
                result = True
                for v in values:
                    result = _eval_node(v, env)
                    if not result:
                        return result
                return result
            else:  # Or
                result = False
                for v in values:
                    result = _eval_node(v, env)
                    if result:
                        return result
                return result

        case ast.Compare(left=left, ops=ops, comparators=comparators):
            current = _eval_node(left, env)
            for op, comp_node in zip(ops, comparators):
                comp_val = _eval_node(comp_node, env)
                if isinstance(op, ast.In):
                    if isinstance(current, list) and isinstance(comp_val, list):
                        if not all(item in comp_val for item in current):
                            return False
                    elif current not in comp_val:
                        return False
                elif isinstance(op, ast.NotIn):
                    if isinstance(current, list) and isinstance(comp_val, list):
                        if all(item in comp_val for item in current):
                            return False
                    elif current in comp_val:
                        return False
                else:
                    fn = _CMP_OPS.get(type(op))
                    if fn is None:
                        raise ValueError(f"disallowed comparison: {type(op).__name__}")
                    if not fn(current, comp_val):
                        return False
                current = comp_val
            return True

        case ast.Call(func=ast.Name(id="len"), args=[arg]) if not node.keywords:
            return len(_eval_node(arg, env))

        case ast.Call(func=ast.Name(id="domain_of"), args=[arg]) if not node.keywords:
            val = _eval_node(arg, env)
            if isinstance(val, list):
                return [_domain_of_str(v) for v in val]
            return _domain_of_str(val)

        case _:
            raise ValueError(f"disallowed expression: {type(node).__name__}")


def _domain_of_str(val: Any) -> Any:
    """Extract domain from a single string value."""
    if isinstance(val, str) and "@" in val:
        return val.rsplit("@", 1)[-1]
    return val
