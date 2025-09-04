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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jsonschema
import yaml
import degirum as dg


@dataclass
class ModelSpec:
    """
    Specification for a single AI model with extensible parameters.

    Each model can have its own zoo_url, inference_host_address, and additional parameters for connection and loading.

    Attributes:
        model_name: Exact model name to load
        zoo_url: Zoo URL where this model is hosted
        inference_host_address: Where to run inference for this model
        token: Optional token for zoo connection
        load_kwargs: Additional keyword arguments for model loading
        metadata: Optional metadata dictionary for additional information (usually taken from model registry)

    Example:
        >>> detector_spec = ModelSpec(
        ...     model_name="yolov8n_relu6_face--640x640_quant_n2x_orca1_1",
        ...     zoo_url="https://cs.degirum.com/degirum/orca",
        ...     inference_host_address="@localhost",
        ...     token="auth_token",
        ...     load_kwargs={"output_confidence_threshold": 0.1}
        ... )
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

    def zoo_connect(self):
        """
        Connect to a model zoo.
        Use this method to optimize multiple model loads from the same zoo.
        Otherwise, load_model() will connect to zoo automatically.

        Returns:
            Connected zoo instance
        """

        zoo = dg.connect(self.inference_host_address, self.zoo_url, self.token)
        return zoo

    def load_model(self, zoo=None):
        """
        Load the model using this specification.

        Args:
            zoo: Optional pre-connected zoo instance. If None, will connect automatically.

        Returns:
            Loaded model instance
        """
        if zoo is None:
            zoo = self.zoo_connect()

        # Load model with load_kwargs
        return zoo.load_model(self.model_name, **self.load_kwargs)


class ModelRegistry:
    """AI model registry. Centralized model management for specific applications.
    Loads model specifications from a YAML configuration file and provides query methods.

    Typical usage:
        >>> registry = ModelRegistry()
        >>> model_spec = registry.for_task("face_detection").for_hardware("N2X/ORCA1").model_spec()
        >>> model = model_spec.load_model()
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
            models: Optional dictionary of model specifications. If provided, overrides config_file.
            config_file: Path to YAML configuration file. If None, uses default 'models.yaml' in the same directory.
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
        """Get new model registry with models compatible with specified hardware.

        Args:
            hardware: Hardware type in PySDK format RUNTIME/DEVICE, e.g., 'N2X/ORCA1'

        Returns:
            ModelRegistry instance with filtered models
        """

        compatible_models = {
            k: v for k, v in self.models.items() if v.get(self.key_hardware) == hardware
        }
        return ModelRegistry(models=compatible_models)

    def for_task(self, task: str) -> "ModelRegistry":
        """Get new model registry with models for specified task.

        Args:
            task: Task name, e.g., 'face_detection', 'object_detection', 'segmentation'

        Returns:
            ModelRegistry instance with filtered models
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
        """Get model specifications for all models in the registry

        Args:
            inference_host_address: Where to run inference for this model
            zoo_url: Optional override for the model's zoo_url (to be used for local zoos)
            token: Optional token for zoo connection
            load_kwargs: Additional keyword arguments for model loading (passed to PySDK load_model())

        Returns:
            ModelSpec instance for the model with highest FPS
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
        """
        Get model specification for the single model in the registry.
        Raises error if zero or multiple models are present.

        Args:
            see model_specs()

        Returns:
            ModelSpec instance for the single model
        """

        if not self.models:
            raise RuntimeError("No models available in the registry")
        if len(self.models) != 1:
            raise RuntimeError(
                "Multiple models available in the registry; use model_specs() instead"
            )
        return self.model_specs()[0]

    def get_tasks(self) -> List[str]:
        """Get list of unique tasks in the registry.

        Returns:
            List of task names
        """
        return sorted({v[self.key_task] for v in self.models.values()})

    def get_hardware(self) -> List[str]:
        """Get list of unique hardware types in the registry.

        Returns:
            List of hardware types
        """
        return sorted({v[self.key_hardware] for v in self.models.values()})

    """
    YAML/JSON schema for validating model registry configuration files.
    This schema ensures that the configuration file contains a top-level '{key_models}' object,
    where each key is a model name matching the pattern "^[a-zA-Z0-9_-]+$". Each model entry must
    specify required fields: '{key_description}', '{key_task}', '{key_hardware}', and '{key_zoo_url}'.
    Optional metadata can be provided via '{key_metadata}'. No additional properties are allowed at
    any level, enforcing strict structure for registry files.
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
