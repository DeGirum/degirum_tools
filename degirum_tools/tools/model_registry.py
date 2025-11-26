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
======================================

Use this module when you want to use a working model without memorizing model
details. ``ModelRegistry`` acts as a guided menu of DeGirum models, while
``ModelSpec`` captures the handful of settings required to load one of those
choices. Together they help pick a model with confidence and launch
inference in just a few lines of code.

Key Features

- Capture one model request with ``ModelSpec`` so it can be shared, reused, or
  versioned alongside your project.
- Browse curated registries to surface models by goal, compatible hardware, or
  descriptive metadata.
- Layer simple filters to shrink the catalog to only the options that fit your
  scenario.
- Keep common defaults (host, zoo, properties) in a single place so every run
  follows the same playbook.

Typical Usage

1. Point ``ModelRegistry`` at the YAML file or hosted URL that lists the models
   available to your team.
2. Narrow the registry with helpers such as ``for_task`` and ``for_hardware`` to
   focus on the models that match your intent.
3. Choose a remaining ``ModelSpec`` (or let ranking helpers do it) to represent
   the model you plan to run.
4. Call ``ModelSpec.load_model()`` to launch the model and start running
   inference.

Example:

```python
from degirum_tools import Display, ModelRegistry, ModelSpec, remote_assets

registry = ModelRegistry(
    config_file="https://assets.degirum.com/registry/models.yaml",
)

model_spec = (
    registry
    .for_task("coco_detection")
    .top_model_spec()
)

model = model_spec.load_model()
inference_result = model(remote_assets.three_persons)

print(inference_result)

with Display("Model Registry Demo") as output_display:
    output_display.show_image(inference_result.image_overlay)
```

Integration Notes

- Registry files can live in source control or be hosted online; point the
  constructor at whichever location you maintain.
- ``ModelSpec.load_model()`` opens the connection with ``degirum.connect`` on
  your behalf, so you do not have to manage sessions manually.
- Override defaults as needed when experimenting, without editing the shared
  registry catalog.

Key Functions

- ``ModelSpec.load_model()`` turns a saved specification into a ready-to-run
  model.
- ``ModelSpec.ensure_local()`` downloads the assets you need for offline use and
  returns a spec that targets ``@local``.
- ``ModelRegistry.all_model_specs()`` lists the prepared specs that match the
  filters you have applied.
- ``ModelRegistry.best_model_spec()`` helps you pick a model based on ranking
  fields such as accuracy scores.

Configuration Options

- Registry defaults set shared values such as the inference host, zoo URL,
  tokens, and model properties.
- Metadata filters accept literal values or simple callables when you want to
  match custom fields in your catalog.
"""

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
    """Serializable description of a single model load request.

    Attributes:
        model_name (str): Exact model identifier expected by the zoo. Ignored
            when ``model_url`` is provided.
        zoo_url (str): Zoo location that hosts this model. When left empty,
            registry defaults or explicit overrides must supply the value.
        model_url (str): Direct URL to the model file. When set, overrides
            ``model_name`` and ``zoo_url``.
        inference_host_address (str): Target inference host in PySDK locator
            format (for example ``@cloud`` or ``@local``).
        token (str | None): Optional authentication token for the zoo.
        model_properties (dict[str, Any]): Keyword arguments forwarded to
            ``dg.ZooManager.load_model``.
        metadata (dict | None): Free-form informational payload, typically
            copied from the registry entry.

    Examples:
        ```python
        spec = ModelSpec(
            model_name="<model>",
            zoo_url="<zoo>",
            inference_host_address="@local",
        )
        spec.load_model()

        spec = ModelSpec(model_url="<model_url>")
        spec.load_model()
        ```
    """

    model_name: str = ""
    zoo_url: str = ""
    model_url: str = ""
    inference_host_address: str = "@cloud"
    token: Optional[str] = None
    model_properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Optional[dict] = field(default_factory=dict)

    def __post_init__(self):
        """Validate required fields, support ``model_url``, and normalize values.

        Raises:
            ValueError: If mandatory attributes are missing or inconsistent.
        """

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
        """Create a connection to the configured model zoo.

        Returns:
            inference_manager (dg.ZooManager): Reusable inference manager
                suitable for repeated calls to ``load_model``.
        """

        zoo = dg.connect(self.inference_host_address, self.zoo_url, self.token)
        return zoo

    def load_model(self, zoo=None):
        """Resolve the specification into a ready-to-use model instance.

        Args:
            zoo (dg.ZooManager | None): Optional pre-connected inference
                manager. When ``None``, ``zoo_connect`` is called
                automatically.

        Returns:
            model (dg.Model): Loaded model returned by the PySDK.
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
            destination (str | Path | None): Destination directory or path. If a
                directory is provided, the model is saved into a subdirectory named
                after the model. When ``None``, the default application data directory
                is used.
            cloud_sync (bool): When ``True``, checks the cloud zoo for an updated
                model version and downloads it if the local copy is missing or out of
                date.

        Returns:
            local_path (Path): Path to the downloaded model assets directory.
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
            cloud_sync (bool): When ``True``, checks the cloud zoo for an updated
                model version and downloads it if the local copy is missing or out of
                date.

        Returns:
            local_spec (ModelSpec): New specification pointing to the local
                model copy.
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
    """Queryable collection of ``ModelSpec`` entries.

    Registry data is sourced from structured YAML files that describe the
    available models, their target tasks, and compatible hardware. Instances
    remain immutable, and filtering methods return new copies so intermediate
    views can be chained without side effects.

    Examples:
    ```python
    registry = ModelRegistry(
        config_file="https://assets.degirum.com/registry/models.yaml",
    )
    filtered = registry.for_task("coco_detection")
    first_spec = filtered.top_model_spec()
    print(first_spec.model_name)
    ```
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
        """Create a registry from a configuration dictionary or YAML file.

        Args:
            config (dict[str, dict] | None): Parsed registry configuration. If
                provided, ``config_file`` is ignored.
            config_file (Path | str | None): Path or URL to a YAML registry
                document. Defaults to ``models.yaml`` located alongside this
                module. The latest DeGirum-managed catalog is published at
                [https://assets.degirum.com/registry/models.yaml](https://assets.degirum.com/registry/models.yaml).

        Raises:
            RuntimeError: If a remote registry cannot be retrieved.
            jsonschema.exceptions.ValidationError: When the configuration does
                not satisfy ``ModelRegistry.schema_text``.
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
        """Return a copy of the registry with overridden default settings.

        Args:
            inference_host_address (str | None): Override for the default
                inference target.
            zoo_url (str | None): Fallback zoo URL applied when entries omit
                an explicit value.
            token (str | None): Token injected into downstream connections.
            model_properties (dict | None): Base keyword arguments merged into
                every ``ModelSpec`` emitted by the registry.

        Returns:
            registry (ModelRegistry): New registry with updated defaults
                applied.
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
        """Filter models to those compatible with a specific hardware target.

        Args:
            hardware (str): Hardware identifier in ``RUNTIME/DEVICE`` format,
                for example ``N2X/ORCA1``.

        Returns:
            registry (ModelRegistry): Filtered registry containing only
                matching models.
        """

        new_config = copy.deepcopy(self.config)
        new_config[self.key_models] = {
            k: v
            for k, v in self.config[self.key_models].items()
            if hardware in v.get(self.key_hardware, [])
        }
        return ModelRegistry(config=new_config)

    def for_task(self, task: str) -> "ModelRegistry":
        """Filter models by the task label declared in the registry.

        Args:
            task (str): Task identifier such as ``face_detection`` or
                ``segmentation``.

        Returns:
            registry (ModelRegistry): Filtered registry containing only
                matching models.
        """

        new_config = copy.deepcopy(self.config)
        new_config[self.key_models] = {
            k: v
            for k, v in self.config[self.key_models].items()
            if v.get(self.key_task) == task
        }
        return ModelRegistry(config=new_config)

    def for_alias(self, alias: str) -> "ModelRegistry":
        """Filter models by alias.

        Args:
            alias (str): Registry alias to match exactly.

        Returns:
            registry (ModelRegistry): Filtered registry containing only
                matching models.
        """

        new_config = copy.deepcopy(self.config)
        new_config[self.key_models] = {
            k: v
            for k, v in self.config[self.key_models].items()
            if v.get(self.key_alias) == alias
        }
        return ModelRegistry(config=new_config)

    def for_meta(self, meta: Dict[str, Any]) -> "ModelRegistry":
        """Filter models by metadata key/value pairs.

        Args:
            meta (dict[str, Any]): Dictionary describing metadata criteria.
                Values may be callables that receive a metadata dictionary and
                return ``True`` when the model should be included. Non-callable
                values must match exactly.

        Returns:
            registry (ModelRegistry): Filtered registry containing only
                matching models.
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
        """Create ``ModelSpec`` objects for every model in the registry.

        Args:
            inference_host_address (str | None): Destination inference host. If
                omitted, the registry defaults are used.
            zoo_url (str | None): Override for the zoo URL. A common use case
                is redirecting all specs to a local zoo during testing.
            token (str | None): Authentication token to apply to each spec.
            model_properties (dict | None): Keyword arguments merged into the
                default properties and applied to every resulting spec.

        Returns:
            specs (list[ModelSpec]): Specifications matching the current
                filtered view.
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
        ret = []
        for model_name, model_info in self.config[self.key_models].items():
            model_properties_adjusted = copy.copy(model_properties)

            # assign device type list so the first available device from the list will be
            # selected by the PySDK on model load
            if "device_type" not in model_properties_adjusted:
                hw = model_info.get(self.key_hardware)
                if isinstance(hw, list):
                    model_properties_adjusted["device_type"] = hw

            ret.append(
                ModelSpec(
                    model_name=model_name,
                    zoo_url=(
                        model_info[self.key_zoo_url] if zoo_url is None else zoo_url
                    ),
                    inference_host_address=inference_host_address,
                    token=token,
                    model_properties=model_properties_adjusted,
                    metadata=model_info.get(self.key_metadata, {}),
                )
            )

        return ret

    def top_model_spec(self, **kwargs) -> ModelSpec:
        """Return the first model listed in the current registry view.

        Args:
            **kwargs (dict[str, Any]): Overrides forwarded to ``all_model_specs``.

        Returns:
            top_spec (ModelSpec): Specification for the top-most entry.

        Raises:
            RuntimeError: If the filtered registry is empty.
        """

        if not self.config[self.key_models]:
            raise RuntimeError("No models available in the registry")
        return self.all_model_specs(**kwargs)[0]

    def best_model_spec(self, key: str, compare: str = "max", **kwargs) -> ModelSpec:
        """Select the model with the best numeric metadata value.

        Args:
            key (str): Metadata field to inspect.
            compare (str): Either ``"max"`` (default) for the largest value or
                ``"min"`` for the smallest.
            **kwargs (dict[str, Any]): Overrides forwarded to ``all_model_specs``.

        Returns:
            best_spec (ModelSpec): Model whose metadata matches the requested
                criteria.

        Raises:
            RuntimeError: If the registry contains no models.
            ValueError: If none of the models expose the requested key or
                provide numeric values.
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
        """Return every unique task label present in the registry.

        Returns:
            tasks (list[str]): Sorted list of task names.
        """
        return sorted({v[self.key_task] for v in self.config[self.key_models].values()})

    def get_hardware(self) -> List[str]:
        """Return every unique hardware target present in the registry.

        Returns:
            hardware (list[str]): Sorted list of hardware identifiers.
        """
        all_hardware = set()
        for m in self.config[self.key_models].values():
            if self.key_hardware in m:
                all_hardware.update(m[self.key_hardware])
        return sorted(all_hardware)

    def get_aliases(self) -> List[str]:
        """Return every unique alias present in the registry.

        Returns:
            aliases (list[str]): Sorted list of aliases.
        """
        return sorted(
            {
                v[self.key_alias]
                for v in self.config[self.key_models].values()
                if self.key_alias in v
            }
        )

    """
    YAML/JSON schema that validates model registry configuration files.

    Ensures each model entry declares mandatory fields, permits optional
    aliases and metadata, and supports registry-wide defaults.
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
