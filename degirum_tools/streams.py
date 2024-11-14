#
# streams.py: streaming toolkit
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements streaming toolkit for multi-threaded processing.
# Please refer to `dgstreams_demo.ipynb` PySDKExamples notebook for examples of toolkit usage.
#

import yaml
from typing import Union, Optional, Dict
from .streams_base import Gizmo, Composition

# API reexport
from .streams_base import *  # noqa
from .streams_gizmos import *  # noqa
from degirum_tools import *  # noqa

# schema YAML
Key_Vars = "vars"
Key_Gizmos = "gizmos"
Key_ConstructorParams = "params"
Key_Connections = "connections"

composition_definition_schema_text = f"""
type: object
additionalProperties: false
required: [{Key_Gizmos}, {Key_Connections}]
properties:
    {Key_Vars}:
        type: object
        description: The collection of variables, keyed by variable name
        additionalProperties: false
        patternProperties:
            "^[a-zA-Z_][a-zA-Z0-9_]*$":
                oneOf:
                  - type: [string, number, boolean, array]
                    description: The variable value; can be $(expression) to evaluate
                  - type: object
                    description: The only object key is the class name to instantiate; value are the constructor parameters
                    additionalProperties: false
                    minProperties: 1
                    maxProperties: 1
                    patternProperties:
                        "^[a-zA-Z_][a-zA-Z0-9_.]*$":
                            type: object
                            description: The constructor parameters of the object
                            additionalProperties: true
    {Key_Gizmos}:
        type: object
        description: The collection of gizmos, keyed by gizmo instance name
        additionalProperties: false
        patternProperties:
            "^[a-zA-Z_][a-zA-Z0-9_]*$":
                oneOf:
                  - type: string
                    description: The gizmo class name to instantiate, if no parameters are needed
                  - type: object
                    description: The only object key is the class name to instantiate; value are the constructor parameters
                    additionalProperties: false
                    minProperties: 1
                    maxProperties: 1
                    patternProperties:
                        "^[a-zA-Z_][a-zA-Z0-9_.]*$":
                            type: object
                            description: The constructor parameters of the gizmo
                            additionalProperties: true
    {Key_Connections}:
        type: array
        description: The list of connections between gizmos
        items:
            type: array
            description: The connection between gizmos
            items:
                oneOf:
                    - type: string
                    - type: array
                      description: Gizmo with input index
                      prefixItems:
                        - type: string
                        - type: number
                      items: false
"""
composition_definition_schema = yaml.safe_load(composition_definition_schema_text)


def load_composition(
    description: Union[str, dict],
    global_context: Optional[dict] = None,
    local_context: Optional[dict] = None,
) -> Composition:
    """Load composition from provided description of gizmos and connections.
    The description can be either JSON file, YAML file, YAML string, or Python dictionary
    conforming to JSON schema defined in `composition_definition_schema`.

    - description: text description of the composition in YAML format, or a file name with .json, .yaml, or .yml extension
      containing such text description, or Python dictionary with the same structure
    - global_context, local_context: optional contexts to use for expression evaluations (like globals(), locals())

    Returns: composition object
    """

    import jsonschema, copy

    # load description
    description_dict: dict = {}
    if isinstance(description, str):
        if description.endswith((".yaml", ".yml")):
            description_dict = yaml.safe_load(open(description))
        else:
            description_dict = yaml.safe_load(description)
    else:
        assert isinstance(description, dict)
        description_dict = description

    # validate description
    jsonschema.validate(instance=description_dict, schema=composition_definition_schema)

    # define contexts
    if global_context is None:
        global_context = globals()
    else:
        global_context = {**global_context, **globals()}

    if local_context is None:
        local_context = {}
    else:
        local_context = copy.copy(local_context)

    def eval_python_expression(expression):
        try:
            ret = eval(expression, global_context, local_context)
            return ret
        except Exception as e:
            raise RuntimeError(
                f"load_composition: fail to evaluate expression '{expression}'"
            ) from e

    def replace_vars(data, vars):
        if isinstance(data, str):
            if data.startswith("$(") and data.endswith(")"):
                return eval_python_expression(data[2:-1])
            else:
                return data
        elif isinstance(data, dict):
            return {key: replace_vars(value, vars) for key, value in data.items()}
        elif isinstance(data, list):
            return [replace_vars(element, vars) for element in data]
        elif isinstance(data, tuple):
            return tuple(replace_vars(element, vars) for element in data)
        elif isinstance(data, set):
            return {replace_vars(element, vars) for element in data}
        else:
            return data  # leave other types as-is

    def search_callable(callable_name):
        callable = None
        # search in local context first
        if local_context:
            callable = local_context.get(callable_name)
        # then search in global context
        if callable is None:
            callable = global_context.get(callable_name)
        # finally try to evaluate it as expression
        if callable is None:
            try:
                callable = eval_python_expression(callable_name)
            except Exception:
                pass
        return callable

    def create_instance_by_desc(desc):
        # check if the dictionary is the object description
        if isinstance(desc, dict) and len(desc) == 1:
            object_class_name, params = next(iter(desc.items()))
            object_class = search_callable(object_class_name)
            if object_class is None:
                raise ValueError(
                    f"load_composition: '{object_class_name}' class is not defined"
                )

            params = replace_vars(params, local_context)

            try:
                obj = object_class(**params)
            except Exception as e:
                raise RuntimeError(
                    f"load_composition: error creating instance of '{object_class_name}'"
                ) from e
            return obj

        # check if the string is an object class or function name which exists in contexts
        if isinstance(desc, str):
            callable = search_callable(desc)
            if callable is not None:
                try:
                    # call it
                    obj = callable()
                except Exception as e:
                    raise RuntimeError(
                        f"load_composition: error creating instance from '{desc}'"
                    ) from e
                return obj

        # else treat is as potential expression result or just return as-is
        return replace_vars(desc, local_context)

    # create all variables and put them into local context
    if Key_Vars in description_dict:
        for name, desc in description_dict[Key_Vars].items():
            local_context[name] = create_instance_by_desc(desc)

    # create all gizmos and add them to composition
    composition = Composition()
    gizmos: Dict[str, Gizmo] = {}
    for name, desc in description_dict[Key_Gizmos].items():
        g = create_instance_by_desc(desc)
        if not isinstance(g, Gizmo):
            raise ValueError(f"load_composition: '{desc}' does not define a Gizmo")
        gizmos[name] = g
        composition.add(g)

    # create pipelines
    for p in description_dict[Key_Connections]:
        if len(p) < 2:
            raise ValueError(
                f"load_composition: pipeline '{p}' must have at least two elements"
            )
        g0 = gizmos.get(p[0])
        if g0 is None:
            raise ValueError(f"load_composition: '{p[0]}' gizmo is not defined")

        for el in p[1:]:
            if isinstance(el, str):
                gizmo_name = el
                input_index = 0
            else:
                gizmo_name = el[0]
                input_index = el[1]

            g1 = gizmos.get(gizmo_name)
            if g1 is None:
                raise ValueError(
                    f"load_composition: '{gizmo_name}' gizmo is not defined"
                )

            g0 = g0 >> g1[input_index]

    return composition
