#
# test_expr_substitute.py: unit tests for expression_substitute utility
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#

from degirum_tools.tools.expr_substitute import expression_substitute


def test_expression_substitute():

    ctx: dict

    # Basic
    ctx = {"a": 2, "b": 3}
    template = "Sum: ${a + b}, Product: ${a * b}"
    result = expression_substitute(template, ctx)
    assert result == "Sum: 5, Product: 6"

    # Missing variable
    ctx = {"x": 10}
    template = "Value: ${y}"
    result = expression_substitute(template, ctx)
    assert result == "Value: ${y}"

    # Literal
    ctx = {}
    template = "Number: ${42}"
    result = expression_substitute(template, ctx)
    assert result == "Number: 42"

    # Mixed
    ctx = {"foo": "bar"}
    template = "Hello ${foo}, missing: ${baz}, math: ${1+2}"
    result = expression_substitute(template, ctx)
    assert result == "Hello bar, missing: ${baz}, math: 3"

    # Complex expression: nested dict access
    ctx = {"data": {"foo": {"bar": 123}}}
    template = 'Nested: ${data["foo"]["bar"]}'
    result = expression_substitute(template, ctx)
    assert result == "Nested: 123"

    # Consecutive calls, each substituting one template
    template = "A: ${a}, B: ${b}, C: ${c}"
    ctx1 = {"a": 1}
    ctx2 = {"b": 2}
    ctx3 = {"c": 3}
    result1 = expression_substitute(template, ctx1)
    assert result1 == "A: 1, B: ${b}, C: ${c}"
    result2 = expression_substitute(result1, ctx2)
    assert result2 == "A: 1, B: 2, C: ${c}"
    result3 = expression_substitute(result2, ctx3)
    assert result3 == "A: 1, B: 2, C: 3"

    # Nested braces: only innermost expressions should be substituted
    ctx = {"x": 10, "y": 20}
    template = "Nested: ${${x}}, outer: ${x+${y}}, complex: ${ ${x}+${y} }"
    result = expression_substitute(template, ctx)
    assert result == "Nested: ${10}, outer: ${x+20}, complex: ${ 10+20 }"

    # JSON-like content
    payload = {"key": "value", "num": 42}
    ctx = {"payload": payload}
    template = "${json.dumps(payload)}"
    result = expression_substitute(template, ctx)
    assert result == '{"key": "value", "num": 42}'

    # ${{}} syntax: dict comprehension
    ctx = {"items": [("a", 1), ("b", 2), ("c", 3)]}
    template = "${{ {k: v for k, v in items} }}"
    result = expression_substitute(template, ctx)
    assert result == "{'a': 1, 'b': 2, 'c': 3}"

    # ${{}} syntax: dict literal
    ctx = {"x": 10, "y": 20}
    template = "Dict: ${{ {'x': x, 'y': y} }}"
    result = expression_substitute(template, ctx)
    assert result == "Dict: {'x': 10, 'y': 20}"

    # ${{}} syntax: nested dict access
    ctx = {"data": {"foo": {"bar": 123}}}
    template = "Access: ${{data['foo']['bar']}}"
    result = expression_substitute(template, ctx)
    assert result == "Access: 123"

    # ${{}} syntax: set comprehension with braces
    ctx = {"nums": [1, 2, 3, 2, 1]}
    template = "Unique: ${{ {n*2 for n in nums} }}"
    result = expression_substitute(template, ctx)
    assert result == "Unique: {2, 4, 6}"

    # Mixed ${} and ${{}} in same template
    ctx = {"a": 5, "items": [("x", 1), ("y", 2)]}
    template = "Value: ${a}, Dict: ${{ {k: v*a for k, v in items} }}"
    result = expression_substitute(template, ctx)
    assert result == "Value: 5, Dict: {'x': 5, 'y': 10}"

    # ${{}} with failed evaluation
    ctx = {"a": 1}
    template = "Failed: ${{undefined_var}}, Success: ${a}"
    result = expression_substitute(template, ctx)
    assert result == "Failed: ${{undefined_var}}, Success: 1"

    # ${{}} with json.dumps inside
    ctx = {"data": {"key": "value"}}
    template = "${{json.dumps(data)}}"
    result = expression_substitute(template, ctx)
    assert result == '{"key": "value"}'
