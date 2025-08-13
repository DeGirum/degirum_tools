#
# Streaming toolkit
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements streaming toolkit for multi-threaded processing.
# Please refer to `dgstreams_demo.ipynb` PySDKExamples notebook for examples of toolkit usage.
#

"""
Streaming Toolkit Overview
==========================

This module provides a streaming toolkit for building multi-threaded processing pipelines, where data (images, video frames, or arbitrary objects) flows through a series of *processing blocks* called gizmos. The toolkit allows you to:

- Acquire or generate data from one or more sources (e.g., camera feeds, video files).
- Process the data in a pipeline (possibly in parallel), chaining multiple gizmos together.
- Optionally display or save the processed data, or feed it into AI inference models.
- Orchestrate everything in a [Composition](streams_base.md#composition), which manages the lifecycle (threads) of all connected gizmos.

Core Concepts
------------

1. **Stream**:
    - Represents a queue of data items [StreamData](streams_base.md#streamdata), such as frames from a camera or images from a directory.
    - Gizmos push (`put`) data into Streams or read (`get`) data from them.
    - Streams can optionally drop data (the oldest item) if they reach a specified maximum queue size, preventing pipeline bottlenecks.

2. **Gizmo**:
    - A gizmo is a discrete processing node in the pipeline.
    - Each gizmo runs in its own thread, pulling data from its input stream(s), processing it, and pushing results to its output stream(s).
    - Example gizmos include:
        - Video-sourcing gizmos that read frames from a webcam or file.
        - AI inference gizmos that run a model on incoming frames.
        - Video display or saving gizmos that show or store processed frames.
        - Gizmos that perform transformations (resizing, cropping, analyzing) on data.
    - Gizmos communicate via Streams. A gizmo output Stream can feed multiple downstream gizmos.
    - Gizmos keep a list of input streams that they are connected to.
    - Gizmos own their input streams.

3. **Composition**:
    - A [Composition](streams_base.md#composition) is a container that holds and manages multiple gizmos (and their Streams).
    - Once gizmos are connected, you can call `composition.start()` to begin processing. Each gizmo `run()` method executes in a dedicated thread.
    - Call `composition.stop()` to gracefully stop processing and wait for threads to finish.

4. **StreamData** and **StreamMeta**:
    - Each item in the pipeline is encapsulated by a [StreamData](streams_base.md#streamdata) object, which holds:

        - `data`: The actual payload (e.g., an image array, a frame).
        - `meta`: A [StreamMeta](streams_base.md#streammeta) object that can hold extra metadata from each gizmo (e.g., a detection result, timestamps, bounding boxes, etc.).
    - Gizmos can append to [StreamMeta](streams_base.md#streammeta) so that metadata accumulates across the pipeline.

5. **Metadata Flow (StreamMeta)**:
    - How [StreamMeta](streams_base.md#streammeta) works:
        - [StreamMeta](streams_base.md#streammeta) itself is a container that can hold any number of "meta info" objects.
        - Each meta info object is "tagged" with one or more string tags, such as `"dgt_video"`, `"dgt_inference"`, etc.
        - You append new meta info by calling `meta.append(my_info, [list_of_tags])`.
        - You can retrieve meta info objects by searching with `meta.find("tag")` (returns *all* matches) or `meta.find_last("tag")` (returns the *most recent* match).
        - **Important**: A gizmo generally clones (`.clone()`) the incoming [StreamMeta](streams_base.md#streammeta) before appending its own metadata to avoid upstream side effects.
        - This design lets each gizmo add new metadata, while preserving what was provided by upstream gizmos.

    - High-Level Example:
        - A camera gizmo outputs frames with meta tagged `"dgt_video"` containing properties like FPS, width, height, etc.
        - An AI inference gizmo downstream takes `StreamData(data=frame, meta=...)`, runs inference, then:
            1. Clones the metadata container.
            2. Appends its inference results under the `"dgt_inference"` tag.
        - If *two* AI gizmos run in series, both will append metadata with the same `"dgt_inference"` tag. A later consumer can call `meta.find("dgt_inference")` to get both sets of results or `meta.find_last("dgt_inference")` to get the most recent result.

Basic Usage Example
-------------------

A simple pipeline might look like this:

```python
import degirum as dg
from degirum_tools.streams import Composition
from degirum_tools.streams_gizmos import VideoSourceGizmo, VideoDisplayGizmo
import cv2
import time

# Create gizmos. If you are on a laptop or have a webcam attached, VideoSourceGizmo(0) will typically create a gizmo that uses your camera as a video source.
video_source = VideoSourceGizmo(0)
video_display = VideoDisplayGizmo("Camera Preview")

# Connect them
video_source >> video_display

# Build composition
comp = Composition(video_source, video_display)
comp.start(wait=False)  # Don't block main thread

start_time = time.time()
while time.time() - start_time < 10:  # Run for 10 seconds
    cv2.waitKey(5)  # Wait time of 5 ms. Let OpenCV handle window events

comp.stop()
cv2.destroyAllWindows()
```

Key Steps
--------------
1. **Create** your gizmos (e.g., `VideoSourceGizmo`, `VideoDisplayGizmo`, AI inference gizmos, etc.).
2. **Connect** them together using the `>>` operator (or `connect_to()` method) to form a processing graph.
    E.g.:
    ```python
    source >> processor >> sink
    ```
3. **Initialize** a [Composition](streams_base.md#composition) with the top-level gizmo(s).
4. **Start** the [Composition](streams_base.md#composition) to launch each gizmo in its own thread.
5. (Optional) **Wait** for the pipeline to finish or perform other tasks. You can query statuses, queue sizes, or get partial results in real time.
6. **Stop** the pipeline when done.

Advanced Topics
--------------
- **Non-blocking vs Blocking**: Streams can drop items if configured (`allow_drop=True`) to handle real-time feeds.
- **Multiple Inputs or Outputs**: Some gizmos can have multiple input streams and/or broadcast results to multiple outputs.
- **Error Handling**: If any gizmo encounters an error, the [Composition](streams_base.md#composition) can stop the whole pipeline, allowing you to handle exceptions centrally.

For practical code examples, see the `dgstreams_demo.ipynb` notebook in the PySDKExamples.
"""

import yaml
from typing import Union, Optional, Dict
from .base import Gizmo, Composition

# API reexport
from .base import *  # noqa
from .gizmos import *  # noqa
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
    """Load a [Composition](streams_base.md#composition) of gizmos and connections from a description.

    The description can be provided as a JSON/YAML file path, a YAML string, or a Python dictionary
    conforming to the JSON schema defined in `composition_definition_schema`.

    Composition Description Schema (YAML):
    ```yaml
    type: object
    additionalProperties: false
    required: [gizmos, connections]
    properties:
        vars:
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
        gizmos:
            type: object
            description: The collection of gizmos, keyed by gizmo instance name
            additionalProperties: false
            patternProperties:
                "^[a-zA-Z_][a-zA-Z0-9_]*$":
                    oneOf:
                      - type: string
                        description: The gizmo class name to instantiate (if no parameters are needed)
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
        connections:
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
    ```

    Args:
        description (str or dict): A YAML string or dict describing the Composition, or a path to a .json/.yaml file containing the description.
        global_context (dict, optional): Global context for evaluating expressions in the description (variables, etc.). Defaults to None.
        local_context (dict, optional): Local context for evaluating expressions. Defaults to None.

    Returns:
        Composition: A [Composition](streams_base.md#composition) object representing the described gizmo pipeline.
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

        # else treat it as potential expression result or just return as-is
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

    # create pipelines (connections between gizmos)
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
