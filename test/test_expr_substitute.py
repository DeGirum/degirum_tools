#
# test_expr_substitute.py: unit tests for expression_substitute utility
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#

from degirum_tools.tools.expr_substitute import expression_substitute


def test_expression_substitute_all():

    ctx: dict

    # Basic
    ctx = {"a": 2, "b": 3}
    template = "Sum: {a + b}, Product: {a * b}"
    result = expression_substitute(template, ctx)
    assert result == "Sum: 5, Product: 6"

    # Missing variable
    ctx = {"x": 10}
    template = "Value: {y}"
    result = expression_substitute(template, ctx)
    assert result == "Value: {y}"

    # Literal
    ctx = {}
    template = "Number: {42}"
    result = expression_substitute(template, ctx)
    assert result == "Number: 42"

    # Mixed
    ctx = {"foo": "bar"}
    template = "Hello {foo}, missing: {baz}, math: {1+2}"
    result = expression_substitute(template, ctx)
    assert result == "Hello bar, missing: {baz}, math: 3"

    # Complex expression: nested dict access
    ctx = {"data": {"foo": {"bar": 123}}}
    template = 'Nested: {data["foo"]["bar"]}'
    result = expression_substitute(template, ctx)
    assert result == "Nested: 123"

    # Consecutive calls, each substituting one template
    template = "A: {a}, B: {b}, C: {c}"
    ctx1 = {"a": 1}
    ctx2 = {"b": 2}
    ctx3 = {"c": 3}
    result1 = expression_substitute(template, ctx1)
    assert result1 == "A: 1, B: {b}, C: {c}"
    result2 = expression_substitute(result1, ctx2)
    assert result2 == "A: 1, B: 2, C: {c}"
    result3 = expression_substitute(result2, ctx3)
    assert result3 == "A: 1, B: 2, C: 3"

    # Nested braces: only innermost expressions should be substituted
    ctx = {"x": 10, "y": 20}
    template = "Outer: {{x + y}}, Inner: {x + y}, Mixed: {{x}} and {y}"
    result = expression_substitute(template, ctx)
    # Only {x + y} and {y} are substituted, double braces are left as-is
    assert result == "Outer: {30}, Inner: 30, Mixed: {10} and 20"
