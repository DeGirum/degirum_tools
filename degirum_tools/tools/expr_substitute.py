#
# expr_substitute.py: template string expression substitution utility
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#

import re

# packages to support expression evaluation
import json, dataclasses  # noqa


def expression_substitute(template: str, context: dict) -> str:
    """
    Substitute Python expressions in curly braces within a template string.

    Each occurrence of `${expr}` in the template is replaced with the result of `eval(expr, { }, context)`.
    If evaluation fails for any expression, the original `${expr}` is left unchanged.

    Args:
        template (str): The template string containing `${expr}` expressions.
        context (dict): Dictionary providing variables for expression evaluation.

    Returns:
        str: The template string with all expressions substituted by their evaluated values.
    """

    pattern = re.compile(r"\$\{\{((?:[^}]|\}(?!\}))+)\}\}|\$\{([^{}]+)\}")

    def repl(match):
        # group 1 is for ${{...}}, group 2 is for ${...}
        expr = (match.group(1) or match.group(2)).strip()
        try:
            # Pass context as both globals and locals to support comprehensions in Python 3.9+
            # In Python 3.9, comprehensions create their own scope and need variables in locals()
            eval_globals = dict(globals())
            eval_globals.update(context)
            value = eval(expr, eval_globals, context)
        except Exception:
            # can't resolve expression – return original text including braces
            return match.group(0)
        else:
            return str(value)

    return pattern.sub(repl, template)
