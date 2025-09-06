#
# model_registry.py: support for model specifications and loading
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements a ModelSpec class for specifying and loading AI models with flexible parameters.
# Also implements a ModelRegistry class for managing multiple model specifications,
# loading them from a YAML configuration file, and querying based on tasks and hardware.
#
"""
Model Registry Overview
=======================

This module provides two building blocks for declaring and loading AI models
in a consistent and portable way:

- ``ModelSpec``: a small dataclass that describes how to connect to a model zoo
  and which model to load, with optional token and loader keyword arguments.
- ``ModelRegistry``: a lightweight registry that reads a YAML file and returns
  one or more ``ModelSpec`` objects filtered by task and/or hardware.

Typical usage:
```python
from degirum_tools.model_registry import ModelRegistry

# Loads the models.yaml
registry = ModelRegistry(config_file="models.yaml")

# Filter by task and hardware, then obtain the single spec
spec = registry.for_task("face_detection").for_hardware("N2X/ORCA1").model_spec()

# Connect and load the model (or pass a preconnected manager via spec.load_model(zoo=...))
model = spec.load_model()
```

YAML layout:
Each entry in ``models.yaml`` defines a model with a short key, description,
task, hardware, and a ``zoo_url`` (which may point to a public or private zoo).
Optional ``metadata`` may be included for display or selection.

The registry validates the file using a strict schema.

Model Registry Schema (YAML):
```yaml
type: object
properties:
  models:
    type: object
    patternProperties:
      "^[a-zA-Z0-9_-]+$":
        type: object
        required:
          - description
          - task
          - hardware
          - zoo_url
        properties:
          description:
            type: string
          task:
            type: string
          hardware:
            type: string
          zoo_url:
            type: string
          metadata:
            type: object
        additionalProperties: false
    additionalProperties: false
required:
  - models
additionalProperties: false
```

Example models.yaml snippet:
```yaml
models:
  yolov8n_relu6_face--640x640_quant_n2x_orca1_1:
    description: Face detector
    task: face_detection
    hardware: N2X/ORCA1
    zoo_url: https://hub.degirum.com/degirum/orca
    metadata:
      author: DeGirum
      license: proprietary
```
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import degirum as dg
import jsonschema
import yaml


@dataclass
class ModelSpec:
    """Specification for a single AI model with extensible parameters.

    Each model can have its own zoo_url, inference_host_address, and additional parameters for connection and loading.

    Attributes:
        model_name (str): Exact model name to load.
        zoo_url (str): Model zoo URL (for example, ``degirum/public`` or a cloud URL).
        inference_host_address (str): Where to run inference (for example,
            ``"@cloud"``, ``"@local"``, or a server IP).
        token (Optional[str]): Optional token for zoo authentication.
        load_kwargs (Dict[str, Any]): Extra keyword arguments forwarded to
                ``load_model``.
        metadata (Optional[dict]): Optional metadata dictionary (typically
            copied from the registry entry).

    Examples:
    ```python
    detector_spec = ModelSpec(
        model_name="yolov8n_relu6_face--640x640_quant_n2x_orca1_1",
        zoo_url="https://hub.degirum.com/degirum/orca",
        inference_host_address="@local",
        token="your_token",
        load_kwargs={"output_confidence_threshold": 0.1},
    )
    ```
    """

    model_name: str
    zoo_url: str = "degirum/public"
    inference_host_address: str = "@cloud"
    token: Optional[str] = None
    load_kwargs: Dict[str, Any] = field(default_factory=dict)
    metadata: Optional[dict] = field(default_factory=dict)

    def __post_init__(self):
        """Validate model specification after initialization."""
        if not self.model_name:
            raise ValueError("model_name cannot be empty")
        if not self.zoo_url:
            raise ValueError("zoo_url cannot be empty")
        if not self.inference_host_address:
            raise ValueError("inference_host_address cannot be empty")
        if self.load_kwargs is None:
            self.load_kwargs = {}

    def zoo_connect(self) -> dg.zoo_manager.ZooManager:
        """Connect to a model zoo.

        Use this method to optimize multiple model loads from the same zoo.
        Otherwise, `load_model()` will connect to the zoo automatically.

        Returns:
            Connected ZooManager for the specified host/zoo.
        """

        zoo = dg.connect(self.inference_host_address, self.zoo_url, self.token)
        return zoo

    def load_model(
        self, zoo: Optional[dg.zoo_manager.ZooManager] = None
    ) -> dg.model.Model:
        """Load the model described by this spec.
        Args:
            zoo (optional): ZooManager.
            If omitted, a new connection is created.
        Returns:
            Loaded model instance.
        """
        if zoo is None:
            zoo = self.zoo_connect()

        # Load model with load_kwargs
        return zoo.load_model(self.model_name, **self.load_kwargs)


class ModelRegistry:
    """AI model registry.
    Centralized model management for specific applications.
    Loads model specifications from a YAML configuration file and provides
    query methods to filter by task and hardware.

    Examples:
    ```python
    registry = ModelRegistry()
    spec = registry.for_task("face_detection").for_hardware("N2X/ORCA1").model_spec()
    model = spec.load_model()
    ```
    """

    # configuration file dictionary keys
    key_models = "models"
    key_description = "description"
    key_task = "task"
    key_hardware = "hardware"
    key_zoo_url = "zoo_url"
    key_metadata = "metadata"

    def __init__(
        self,
        *,
        models: Optional[Dict[str, dict]] = None,
        config_file: Union[Path, str, None] = None,
    ):
        """Initialize model registry from configuration file or provided models dictionary.

        Args:
            models (dict, optional): Pre-populated model dictionary. Bbypasses
                reading a YAML file.
            config_file (Path|str, optional): Path to a YAML configuration
                file. If ``None``, uses ``models.yaml`` next to
                this module.
        """

        self.models: Dict[str, dict]  # model dictionary

        if models is not None:
            self.models = models
        else:
            if config_file is None:
                config_file = Path(__file__).parent / "models.yaml"

            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            # validate config structure by schema
            schema = yaml.safe_load(self.schema_text)
            jsonschema.validate(instance=config, schema=schema)
            self.models = config.get(self.key_models, {})

    def for_hardware(self, hardware: str) -> "ModelRegistry":
        """Return a registry filtered by hardware.

        Args:
            hardware (str): Hardware type in PySDK format ``RUNTIME/DEVICE``,
                e.g., ``"N2X/ORCA1"``.

        Returns:
            Registry containing only models compatible with the specified runtime/hardware combination.
        """

        compatible_models = {
            k: v for k, v in self.models.items() if v.get(self.key_hardware) == hardware
        }
        return ModelRegistry(models=compatible_models)

    def for_task(self, task: str) -> "ModelRegistry":
        """Return a registry filtered by task.

        Args:
            task (str): Task name, e.g., ``"face_detection"``,
                ``"object_detection"``, ``"segmentation"``.

        Returns:
            Registry containing only models for the given task.
        """

        task_models = {
            k: v for k, v in self.models.items() if v.get(self.key_task) == task
        }
        return ModelRegistry(models=task_models)

    def model_specs(
        self,
        *,
        inference_host_address: str = "@cloud",
        zoo_url: Optional[str] = None,
        token: Optional[str] = None,
        load_kwargs: Optional[dict] = None,
    ) -> List[ModelSpec]:
        """Return model specifications for all models in the registry.

        Args:
            inference_host_address (str): Where to run inference for each
                model (for example, ``"@cloud"`` or a server hostname/IP).
            zoo_url (str, optional): Override the ``zoo_url`` declared per
                model. Useful for local zoos.
            token (str, optional): Optional token for zoo authentication.
            load_kwargs (dict, optional): Extra keyword arguments forwarded to
                ``load_model`` for each spec.

        Returns:
            One ModelSpec per registry entry (after any filters).
        """

        return [
            ModelSpec(
                model_name=model_name,
                zoo_url=model_info[self.key_zoo_url] if zoo_url is None else zoo_url,
                inference_host_address=inference_host_address,
                token=token,
                load_kwargs=load_kwargs if load_kwargs is not None else {},
                metadata=model_info.get(self.key_metadata, {}),
            )
            for model_name, model_info in self.models.items()
        ]

    def model_spec(self, **kwargs) -> ModelSpec:
        """Return the single model specification.
        Raises an error if either zero or multiple models are present.

        Use this only when the registry contains exactly one model (for
        example, after filtering by task and hardware). Otherwise, a
        ``RuntimeError`` is raised.

        Returns:
            ModelSpec: The sole model specification in the registry.
        """

        if not self.models:
            raise RuntimeError("No models available in the registry")
        if len(self.models) != 1:
            raise RuntimeError(
                "Multiple models available in the registry; use model_specs() instead"
            )
        return self.model_specs()[0]

    def get_tasks(self) -> List[str]:
        """Return the unique task names present in the registry."""
        return sorted({v[self.key_task] for v in self.models.values()})

    def get_hardware(self) -> List[str]:
        """Return the unique hardware types present in the registry."""
        return sorted({v[self.key_hardware] for v in self.models.values()})

    """YAML/JSON schema for validating model registry configuration files.
    This schema ensures that the configuration file contains a top-level '{key_models}' object,
    where each key is a model name matching the pattern "^[a-zA-Z0-9_-]+$". Each model entry must
    specify required fields: '{key_description}', '{key_task}', '{key_hardware}', and '{key_zoo_url}'.
    Optional metadata can be provided via '{key_metadata}'. No additional properties are allowed at
    the top level or within each model entry. The '{key_metadata}' object itself is intentionally
    free-form to hold any auxiliary information you need.
    """
    schema_text = f"""

type: object
properties:
  {key_models}:
    type: object
    patternProperties:
      "^[a-zA-Z0-9_-]+$":
        type: object
        required:
          - {key_description}
          - {key_task}
          - {key_hardware}
          - {key_zoo_url}
        properties:
          {key_description}:
            type: string
          {key_task}:
            type: string
          {key_hardware}:
            type: string
          {key_zoo_url}:
            type: string
          {key_metadata}:
            type: object
        additionalProperties: false
    additionalProperties: false
required:
  - {key_models}
additionalProperties: false

    """
