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

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jsonschema
import yaml
import degirum as dg
import requests


@dataclass
class ModelSpec:
    """
    Specification for a single AI model with extensible parameters.

    Each model can have its own zoo_url, inference_host_address, and additional parameters for connection and loading.

    Attributes:
        model_name: Exact model name to load
        zoo_url: Zoo URL where this model is hosted
        model_url: Direct URL to model file (optional, overrides both model_name and zoo_url if provided)
        inference_host_address: Where to run inference for this model
        token: Optional token for zoo connection
        model_properties: a dictionary of arbitrary model properties to be assigned to the model object
        metadata: Optional metadata dictionary for additional information (usually taken from model registry)

    Example:
        >>> detector_spec = ModelSpec(
        ...     model_name="yolov8n_relu6_face--640x640_quant_n2x_orca1_1",
        ...     zoo_url="https://cs.degirum.com/degirum/orca",
        ...     inference_host_address="@localhost",
        ...     token="auth_token",
        ...     model_properties={"output_confidence_threshold": 0.1}
        ... )
    """

    model_name: str = ""
    zoo_url: str = ""
    model_url: str = ""
    inference_host_address: str = "@cloud"
    token: Optional[str] = None
    model_properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Optional[dict] = field(default_factory=dict)

    def __post_init__(self):
        """Validate model specification after initialization."""

        if self.model_url:
            if self.model_name or self.zoo_url:
                raise ValueError(
                    "If `model_url` is provided, `model_name` and `zoo_url` must be empty"
                )
            parts = self.model_url.rsplit("/", 1)
            if len(parts) == 2:
                self.zoo_url, self.model_name = parts
            else:
                raise ValueError(
                    "`model_url` must contain both model name and zoo URL separated by '/'"
                )

        if not self.model_name:
            raise ValueError("model_name cannot be empty")
        if not self.zoo_url:
            raise ValueError("zoo_url cannot be empty")
        if not self.inference_host_address:
            raise ValueError("inference_host_address cannot be empty")
        if self.model_properties is None:
            self.model_properties = {}

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

        # Load model with model_properties
        return zoo.load_model(self.model_name, **self.model_properties)

    def __repr__(self):
        return (
            f"ModelSpec(name={self.model_name!r}, zoo={self.zoo_url!r}, "
            f"host={self.inference_host_address!r}, props={list(self.model_properties.keys())}, "
            f"meta_keys={list(self.metadata.keys()) if isinstance(self.metadata, dict) else None})"
        )

    def download_model(
        self,
        destination: Union[str, Path, None] = None,
        *,
        cloud_sync=False,
    ):
        """
        Download the model file to a local directory.

        Args:
            destination: Destination directory or path. If a directory is provided,
                the model will be saved into the subdirectory named after the model name.
                If None, saves to the default application data directory.
            cloud_sync: If True, checks the cloud zoo for updated model version
                and downloads it if the local copy is missing or outdated.

        Returns:
            Path to the downloaded model assets directory.
        """

        if not destination:
            destination = Path(dg._misc.get_app_data_dir()) / "models"
        else:
            destination = Path(destination)

        destination /= self.model_name

        need_download = False
        cloud_zoo: Optional[dg.ZooManager] = None

        if not destination.exists():
            need_download = True  # no model
        else:
            # check model checksum
            try:
                local_zoo = dg.connect(dg.LOCAL, str(destination))
                local_checksum = local_zoo.model_info(self.model_name).Checksum
            except Exception:
                need_download = True  # model corrupted

            if not need_download and cloud_sync:
                cloud_zoo = dg.connect(dg.CLOUD, self.zoo_url, self.token)
                cloud_checksum = cloud_zoo.model_info(self.model_name).Checksum
                if local_checksum != cloud_checksum:
                    need_download = True  # checksum mismatch

        if need_download:
            if not cloud_zoo:
                cloud_zoo = dg.connect(dg.CLOUD, self.zoo_url, self.token)
            cloud_zoo._zoo.download_model(self.model_name, destination.parent)

        return destination

    def ensure_local(self, cloud_sync=False) -> "ModelSpec":
        """
        Ensures the model is present locally; downloads if needed.
        Returns a **new** ModelSpec with zoo_url set to local path and inference_host_address to '@local'.

        Args:
            cloud_sync: If True, checks the cloud zoo for updated model version
                and downloads it if the local copy is missing or outdated.
        """

        local_path = self.download_model(cloud_sync=cloud_sync)

        return ModelSpec(
            model_name=self.model_name,
            zoo_url=str(local_path),
            inference_host_address="@local",
            token=self.token,
            model_properties=self.model_properties,
            metadata=self.metadata,
        )


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
    key_alias = "alias"
    key_description = "description"
    key_task = "task"
    key_hardware = "hardware"
    key_zoo_url = "zoo_url"
    key_metadata = "metadata"
    key_defaults = "defaults"
    key_inference_host_address = "inference_host_address"
    key_token = "token"
    key_model_properties = "model_properties"

    def __init__(
        self,
        *,
        config: Optional[Dict[str, dict]] = None,
        config_file: Union[Path, str, None] = None,
    ):
        """Initialize model registry from configuration file or provided models dictionary.

        Args:
            config: Optional dictionary of model registry configuration. If provided, overrides config_file.
            config_file: Path to YAML configuration file or its URL. If None, uses default 'models.yaml' in the same directory.
        """

        self.config: Dict[str, Any]  # full configuration dictionary

        if config is None:
            if config_file is None:
                config_file = Path(__file__).parent / "models.yaml"

            if isinstance(config_file, str) and (
                config_file.startswith("http://") or config_file.startswith("https://")
            ):

                response = requests.get(config_file, timeout=5)
                try:
                    response.raise_for_status()
                except requests.HTTPError as e:
                    raise RuntimeError(f"Failed to fetch registry from URL: {e}") from e
                config = yaml.safe_load(response.text)
            else:
                with open(config_file, "r") as f:
                    config = yaml.safe_load(f)

            # validate config structure by schema
            schema = yaml.safe_load(self.schema_text)
            jsonschema.validate(instance=config, schema=schema)

        self.config = config
        if self.key_defaults not in config:
            config[self.key_defaults] = {
                self.key_inference_host_address: "@cloud",
                self.key_token: None,
                self.key_model_properties: {},
            }

    def with_defaults(
        self,
        *,
        inference_host_address: Optional[str] = None,
        zoo_url: Optional[str] = None,
        token: Optional[str] = None,
        model_properties: Optional[dict] = None,
    ) -> "ModelRegistry":
        """Apply new model loading defaults to the model registry

        Args:
            inference_host_address: Inference host address
            zoo_url: Zoo URL
            token: Cloud API token
            model_properties: Model loading properties

        Returns:
            ModelRegistry instance with updated defaults
        """

        new_config = copy.deepcopy(self.config)

        defaults = new_config[self.key_defaults]
        if inference_host_address is not None:
            defaults[self.key_inference_host_address] = inference_host_address
        if zoo_url is not None:
            defaults[self.key_zoo_url] = zoo_url
        if token is not None:
            defaults[self.key_token] = token
        if model_properties is not None:
            defaults[self.key_model_properties] = model_properties

        return ModelRegistry(config=new_config)

    def for_hardware(self, hardware: str) -> "ModelRegistry":
        """Get new model registry with models compatible with specified hardware.

        Args:
            hardware: Hardware type in PySDK format RUNTIME/DEVICE, e.g., 'N2X/ORCA1'

        Returns:
            ModelRegistry instance with filtered models
        """

        new_config = copy.deepcopy(self.config)
        new_config[self.key_models] = {
            k: v
            for k, v in self.config[self.key_models].items()
            if hardware in v.get(self.key_hardware, [])
        }
        return ModelRegistry(config=new_config)

    def for_task(self, task: str) -> "ModelRegistry":
        """Get new model registry with models for specified task.

        Args:
            task: Task name, e.g., 'face_detection', 'object_detection', 'segmentation'

        Returns:
            ModelRegistry instance with filtered models
        """

        new_config = copy.deepcopy(self.config)
        new_config[self.key_models] = {
            k: v
            for k, v in self.config[self.key_models].items()
            if v.get(self.key_task) == task
        }
        return ModelRegistry(config=new_config)

    def for_alias(self, alias: str) -> "ModelRegistry":
        """Get new model registry with models having specified alias.

        Args:
            alias: model alias string

        Returns:
            ModelRegistry instance with filtered models
        """

        new_config = copy.deepcopy(self.config)
        new_config[self.key_models] = {
            k: v
            for k, v in self.config[self.key_models].items()
            if v.get(self.key_alias) == alias
        }
        return ModelRegistry(config=new_config)

    def for_meta(self, meta: Dict[str, Any]) -> "ModelRegistry":
        """Get new model registry with models having specified metadata.

        Args:
            meta: model metadata dictionary, containing key-value pairs for filtering.
            Only models which metadata matches all key-value pairs in that dictionary will be included.
            If the value is callable, it will be called with the model metadata as the argument and must return a boolean.
            This way you may pass custom predicates.

        Returns:
            ModelRegistry instance with filtered models
        """

        new_config = copy.deepcopy(self.config)
        new_config[self.key_models] = {
            k: v
            for k, v in self.config[self.key_models].items()
            if self.key_metadata in v
            and all(
                (
                    mv(v[self.key_metadata])
                    if callable(mv)
                    else v[self.key_metadata].get(mk) == mv
                )
                for mk, mv in meta.items()
            )
        }
        return ModelRegistry(config=new_config)

    def all_model_specs(
        self,
        *,
        inference_host_address: Optional[str] = None,
        zoo_url: Optional[str] = None,
        token: Optional[str] = None,
        model_properties: Optional[dict] = None,
    ) -> List[ModelSpec]:
        """Get model specifications for all models in the registry

        Args:
            inference_host_address: Where to run inference for this model. If None, it will be taken from defaults.
            zoo_url: Optional override for the model's zoo_url (to be used for local zoos). If None, it will be taken from defaults.
            token: Optional token for zoo connection. If None, it will be taken from defaults.
            model_properties: Additional keyword arguments for model loading (passed to PySDK load_model())

        Returns:
            ModelSpec instances for all models in the registry
        """

        defaults = self.config[self.key_defaults]
        if inference_host_address is None:
            inference_host_address = defaults.get(
                self.key_inference_host_address, "@cloud"
            )
        if zoo_url is None:
            zoo_url = defaults.get(self.key_zoo_url)
        if token is None:
            token = defaults.get(self.key_token)
        if model_properties is None:
            model_properties = defaults.get(self.key_model_properties, {})
        else:
            merged = copy.deepcopy(defaults.get(self.key_model_properties, {}))
            merged.update(model_properties)
            model_properties = merged

        assert model_properties is not None
        return [
            ModelSpec(
                model_name=model_name,
                zoo_url=model_info[self.key_zoo_url] if zoo_url is None else zoo_url,
                inference_host_address=inference_host_address,
                token=token,
                model_properties=model_properties,
                metadata=model_info.get(self.key_metadata, {}),
            )
            for model_name, model_info in self.config[self.key_models].items()
        ]

    def top_model_spec(self, **kwargs) -> ModelSpec:
        """
        Get model specification for the topmost model in the registry (as defined in registry YAML file).
        Raises error if no models are present.

        Args:
            see all_model_specs()

        Returns:
            ModelSpec instance for the first model in the registry
        """

        if not self.config[self.key_models]:
            raise RuntimeError("No models available in the registry")
        return self.all_model_specs(**kwargs)[0]

    def best_model_spec(self, key: str, compare: str = "max", **kwargs) -> ModelSpec:
        """
        Find the model with the best (max or min) numeric value for a given metadata key.
        Args:
            key: The key in the metadata dictionary to compare.
            compare: 'max' (default) to select the model with the largest value, 'min' for the smallest.
            kwargs: Passed to all_model_specs (e.g., inference_host_address, zoo_url, etc.)
        Returns:
            ModelSpec with the best value for the given key.
        Raises:
            ValueError if no models have the specified key or if values are not numeric.
        """

        specs = self.all_model_specs(**kwargs)
        if not specs:
            raise RuntimeError("No models available in the registry")

        specs_with_metric = [
            (spec, float(spec.metadata[key]))
            for spec in specs
            if spec.metadata and key in spec.metadata
        ]
        if not specs_with_metric:
            raise ValueError(f"No models have a numeric metadata value for key '{key}'")

        specs_with_metric.sort(key=lambda x: x[1], reverse=(compare == "max"))
        return specs_with_metric[0][0]

    def get_tasks(self) -> List[str]:
        """Get list of unique tasks in the registry.

        Returns:
            List of task names
        """
        return sorted({v[self.key_task] for v in self.config[self.key_models].values()})

    def get_hardware(self) -> List[str]:
        """Get list of unique hardware types in the registry.

        Returns:
            List of hardware types
        """
        all_hardware = set()
        for m in self.config[self.key_models].values():
            if self.key_hardware in m:
                all_hardware.update(m[self.key_hardware])
        return sorted(all_hardware)

    def get_aliases(self) -> List[str]:
        """Get list of unique model aliases in the registry.

        Returns:
            List of model aliases
        """
        return sorted(
            {
                v[self.key_alias]
                for v in self.config[self.key_models].values()
                if self.key_alias in v
            }
        )

    """
    YAML/JSON schema for validating model registry configuration files.
    This schema ensures that the configuration file contains a top-level 'models' object,
    where each key is a model name matching the pattern "^[a-zA-Z0-9_-]+$". Each model entry must
    specify required fields: 'task', 'hardware', and 'zoo_url'.
    Optional alias can be provided for each model via 'alias'.
    Optional metadata dictionary of arbitrary properties can be provided via 'metadata'.
    No additional properties are allowed at model level, enforcing strict structure for registry files.
    Additionally, a top-level optional 'defaults' object can specify default values for:
      - inference_host_address
      - token
      - model_properties
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
          - {key_task}
          - {key_hardware}
          - {key_zoo_url}
        properties:
          {key_alias}:
            type: string
          {key_description}:
            type: string
          {key_task}:
            type: string
          {key_hardware}:
            type: array
            items:
              type: string
          {key_zoo_url}:
            type: string
          {key_metadata}:
            type: object
        additionalProperties: true
    additionalProperties: false
  {key_defaults}:
    type: object
    properties:
      {key_inference_host_address}:
        type: string
      {key_token}:
        type: string
      {key_model_properties}:
        type: object
    required:
      - {key_inference_host_address}
    additionalProperties: true
required:
  - {key_models}
additionalProperties: true
    """
