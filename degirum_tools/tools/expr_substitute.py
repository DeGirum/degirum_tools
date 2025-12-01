#
# expr_substitute.py: template string expression substitution utility
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#

import re

# packages to support expression evaluation
import json  # noqa


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

    pattern = re.compile(r"\$\{([^{}]+)\}")

    def repl(match):
        expr = match.group(1).strip()
        try:
            value = eval(expr, globals(), context)
        except Exception:
            # can't resolve expression – return original text including braces
            return match.group(0)
        else:
            return str(value)

    return pattern.sub(repl, template)
