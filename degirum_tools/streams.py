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
from typing import Union, Optional
from .streams_base import Composition

# API reexport
from .streams_base import *  # noqa
from .streams_gizmos import *  # noqa


# schema YAML
Key_Gizmos = "gizmos"
Key_ClassName = "class"
Key_ConstructorParams = "params"
Key_Connections = "connections"

composition_definition_schema_text = f"""
type: object
additionalProperties: false
properties:
    {Key_Gizmos}:
        type: object
        description: The collection of gizmos, keyed by gizmo instance name
        additionalProperties: false
        patternProperties:
            "^[a-zA-Z_][a-zA-Z0-9_]*$":
                type: object
                additionalProperties: false
                properties:
                    {Key_ClassName}:
                        type: string
                        description: The class name of the gizmo
                    {Key_ConstructorParams}:
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

    import json, jsonschema

    # custom YAML constructor
    def constructor_python_expression(loader, node):
        expression = loader.construct_scalar(node)
        try:
            ret = eval(expression, global_context, local_context)
            return ret
        except Exception as e:
            raise ValueError(f"Fail to evaluate expression: {expression}") from e

    yaml.add_constructor("!Python", constructor_python_expression, yaml.SafeLoader)

    description_dict: dict = {}
    if isinstance(description, str):
        if description.endswith(".json"):
            description_dict = json.load(open(description))
        elif description.endswith((".yaml", ".yml")):
            description_dict = yaml.safe_load(open(description))
        else:
            description_dict = yaml.safe_load(description)

    elif isinstance(description, dict):
        description_dict = description
    else:
        raise ValueError("load_composition: unsupported description type")

    jsonschema.validate(instance=description_dict, schema=composition_definition_schema)

    composition = Composition()

    # create all gizmos
    gizmos = {}
    for name, desc in description_dict[Key_Gizmos].items():
        gizmo_class_name = desc[Key_ClassName]

        gizmo_class = globals().get(gizmo_class_name)
        if gizmo_class is None:
            if global_context is not None:
                gizmo_class = global_context.get(gizmo_class_name)
        if gizmo_class is None:
            if local_context is not None:
                gizmo_class = local_context.get(gizmo_class_name)

        if gizmo_class is None:
            raise ValueError(
                f"load_composition: gizmo class {gizmo_class_name} not defined"
            )

        try:
            gizmo = gizmo_class(**desc.get(Key_ConstructorParams, {}))
        except Exception as e:
            raise ValueError(
                f"load_composition: error creating instance of {gizmo_class_name}"
            ) from e

        composition.add(gizmo)
        gizmos[name] = gizmo

    # create pipelines
    for p in description_dict[Key_Connections]:
        if len(p) < 2:
            raise ValueError(
                f"load_composition: pipeline {p} must have at least two elements"
            )
        if not isinstance(p[0], str):
            raise ValueError(
                f"load_composition: pipeline first element {p[0]} must be a gizmo name"
            )
        g0 = gizmos.get(p[0])
        if g0 is None:
            raise ValueError(f"load_composition: gizmo {p[0]} is not defined")

        for el in p[1:]:
            if isinstance(el, str):
                gizmo_name = el
                input_index = 0
            else:
                gizmo_name = el[0]
                input_index = el[1]

            g1 = gizmos.get(gizmo_name)
            if g1 is None:
                raise ValueError(f"load_composition: gizmo {gizmo_name} is not defined")

            g0 = g0 >> g1[input_index]

    return composition
