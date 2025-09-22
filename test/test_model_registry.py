#
# test_model_registry.py: unit tests for model registry functionality
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements unit tests to test ModelSpec and ModelRegistry functionality
#

import pytest
import yaml
import requests
import shutil
from pathlib import Path
from unittest.mock import patch, mock_open
from degirum_tools.model_registry import ModelSpec, ModelRegistry
import degirum as dg


def test_model_spec_basics():
    """Test ModelSpec class initialization, validation, and methods"""

    # Test basic initialization with valid parameters
    spec = ModelSpec(
        model_name="test_model",
        zoo_url="test_zoo",
        inference_host_address="@cloud",
        token="test_token",
        model_properties={"confidence": 0.8},
        metadata={"fps": 30.0, "accuracy": 0.95},
    )

    assert spec.model_name == "test_model"
    assert spec.zoo_url == "test_zoo"
    assert spec.inference_host_address == "@cloud"
    assert spec.token == "test_token"
    assert spec.model_properties == {"confidence": 0.8}
    assert spec.metadata == {"fps": 30.0, "accuracy": 0.95}

    # Test initialization with model_url (should parse into model_name and zoo_url)
    spec_with_url = ModelSpec(
        model_url="test_zoo/parsed_model", inference_host_address="@local"
    )

    assert spec_with_url.model_name == "parsed_model"
    assert spec_with_url.zoo_url == "test_zoo"
    assert spec_with_url.inference_host_address == "@local"
    assert spec_with_url.model_url == "test_zoo/parsed_model"

    # Test validation errors

    # Empty model_name should raise ValueError
    with pytest.raises(ValueError, match="model_name cannot be empty"):
        ModelSpec(model_name="", zoo_url="test_zoo")

    # Empty zoo_url should raise ValueError
    with pytest.raises(ValueError, match="zoo_url cannot be empty"):
        ModelSpec(model_name="test_model", zoo_url="")

    # Empty inference_host_address should raise ValueError
    with pytest.raises(ValueError, match="inference_host_address cannot be empty"):
        ModelSpec(
            model_name="test_model", zoo_url="test_zoo", inference_host_address=""
        )

    # model_url with model_name should raise ValueError
    with pytest.raises(
        ValueError,
        match="If `model_url` is provided, `model_name` and `zoo_url` must be empty",
    ):
        ModelSpec(model_url="test_zoo/test_model", model_name="conflicting_name")

    # model_url with zoo_url should raise ValueError
    with pytest.raises(
        ValueError,
        match="If `model_url` is provided, `model_name` and `zoo_url` must be empty",
    ):
        ModelSpec(model_url="test_zoo/test_model", zoo_url="conflicting_zoo")

    # Invalid model_url format should raise ValueError
    with pytest.raises(
        ValueError,
        match="`model_url` must contain both model name and zoo URL separated by '/'",
    ):
        ModelSpec(model_url="invalid_url_format")

    # Test default values
    default_spec = ModelSpec(model_name="test_model", zoo_url="test_zoo")
    assert default_spec.inference_host_address == "@cloud"
    assert default_spec.token is None
    assert default_spec.model_properties == {}
    assert default_spec.metadata == {}

    # Test __repr__ method
    repr_str = repr(spec)
    assert "ModelSpec" in repr_str
    assert "test_model" in repr_str
    assert "test_zoo" in repr_str
    assert "@cloud" in repr_str
    assert "confidence" in repr_str
    assert "fps" in repr_str


def test_model_spec_load_models(cloud_token):
    """Test ModelSpec class load model methods"""

    model_name = "mobilenet_v1_imagenet--224x224_quant_n2x_cpu_1"
    zoo_url = "degirum/public_daily_test"

    model_spec = ModelSpec(
        model_name=model_name,
        zoo_url=zoo_url,
        inference_host_address="@cloud",
        token=cloud_token,
    )

    # test zoo connection
    zoo = model_spec.zoo_connect()
    assert zoo is not None

    # test model loading
    model = model_spec.load_model()
    assert model is not None
    model = model_spec.load_model(zoo)
    assert model is not None

    # test model downloading
    base_path = Path(dg._misc.get_app_data_dir()) / "models"
    try:
        # check download to default location
        model_path = model_spec.download_model()
        assert (
            model_path is not None
            and Path(model_path).exists()
            and model_path == base_path / model_name
        )
        shutil.rmtree(base_path / model_name)

        # check download to custom location
        custom_path = base_path / "__custom_dir"
        model_path = model_spec.download_model(custom_path)
        assert (
            model_path is not None
            and Path(model_path).exists()
            and model_path == custom_path / model_name
        )
        shutil.rmtree(custom_path)

        # check ensure_local
        spec2 = model_spec.ensure_local()
        assert spec2.model_name == model_spec.model_name
        assert (
            Path(spec2.zoo_url) == base_path / model_name
            and Path(spec2.zoo_url).exists()
        )
        assert spec2.inference_host_address == "@local"

        # ensure_local should be idempotent
        spec3 = model_spec.ensure_local()
        assert spec3 == spec2

        # check cloud_sync
        model_spec.token = "bad_token"
        assert model_spec.ensure_local(cloud_sync=False) is not None
        with pytest.raises(dg.exceptions.DegirumException):
            model_spec.ensure_local(cloud_sync=True)

    finally:
        # cleanup downloaded models
        if (base_path / model_name).exists():
            shutil.rmtree(base_path / model_name)
        if (base_path / "__custom_dir").exists():
            shutil.rmtree(base_path / "__custom_dir")


@pytest.fixture
def sample_models_dict():
    """Sample models dictionary for testing (hardware as list)"""
    return {
        "face_detector_orca": {
            "description": "Face detection model for ORCA",
            "task": "face_detection",
            "hardware": ["N2X/ORCA1"],
            "zoo_url": "https://cs.degirum.com/degirum/orca",
            "metadata": {"input_size": [640, 640], "fps": 30.0},
            "alias": "orca_face",
        },
        "face_detector_cpu": {
            "description": "Face detection model for CPU",
            "task": "face_detection",
            "hardware": ["N2X/CPU"],
            "zoo_url": "degirum/public",
            "metadata": {"input_size": [112, 112], "fps": 10.0},
            "alias": "cpu_face",
        },
        "object_detector_orca": {
            "description": "Object detection model for ORCA",
            "task": "object_detection",
            "hardware": ["N2X/ORCA1"],
            "zoo_url": "https://cs.degirum.com/degirum/orca",
            "metadata": {"input_size": [640, 640], "fps": 25.0},
            "alias": "orca_object",
        },
        "segmentation_cpu": {
            "description": "Segmentation model for CPU",
            "task": "segmentation",
            "hardware": ["N2X/CPU"],
            "zoo_url": "degirum/public",
            "metadata": {"input_size": [112, 112], "fps": 5.0},
            "alias": "cpu_segmentation",
        },
    }


@pytest.fixture
def sample_config_dict(sample_models_dict):
    """Sample config dictionary for testing (with defaults)"""
    return {
        "models": sample_models_dict,
        "defaults": {
            "inference_host_address": "@cloud",
        },
    }


def test_model_registry_creation(sample_models_dict, sample_config_dict):
    """Test ModelRegistry creation with various methods"""

    # Test creation with config dictionary
    registry = ModelRegistry(config=sample_config_dict)
    assert registry.config["models"] == sample_models_dict

    # Test creation with config file
    config_yaml = yaml.dump(sample_config_dict)
    with patch("builtins.open", mock_open(read_data=config_yaml)) as mocked_open:
        registry = ModelRegistry(config_file="test_config.yaml")
        assert registry.config["models"] == sample_config_dict["models"]
        mocked_open.assert_called_once_with("test_config.yaml", "r")

    # Test creation with default config file
    with patch("builtins.open", mock_open(read_data=config_yaml)) as mocked_open:
        registry = ModelRegistry()
        assert registry.config["models"] == sample_config_dict["models"]
        expected_path = Path(__file__).parent.parent / "degirum_tools" / "models.yaml"
        mocked_open.assert_called_once_with(expected_path, "r")

    # Test creation with invalid config
    invalid_config = {"invalid": "structure"}
    invalid_config_yaml = yaml.dump(invalid_config)
    with patch(
        "builtins.open", mock_open(read_data=invalid_config_yaml)
    ) as mocked_open:
        with pytest.raises(Exception):  # jsonschema validation error
            ModelRegistry(config_file="invalid_config.yaml")
        mocked_open.assert_called_once_with("invalid_config.yaml", "r")

    # Test creation with config URL (mock requests.get)
    class MockResponse:
        def __init__(self, text, status_code=200):
            self.text = text
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code != 200:
                raise requests.HTTPError(f"Status {self.status_code}")

    with patch(
        "requests.get", return_value=MockResponse(yaml.dump(sample_config_dict))
    ) as mock_get:
        registry = ModelRegistry(config_file="https://example.com/models.yaml")
        assert registry.config["models"] == sample_models_dict
        mock_get.assert_called_once_with("https://example.com/models.yaml", timeout=5)


def test_model_registry_filtering_methods(sample_models_dict, sample_config_dict):
    """Test for_hardware, for_task, and chained filtering methods"""

    registry = ModelRegistry(config=sample_config_dict)

    # Test for_hardware filtering
    orca_registry = registry.for_hardware("N2X/ORCA1")
    expected_orca_models = {
        name: model
        for name, model in sample_models_dict.items()
        if "N2X/ORCA1" in model.get("hardware", [])
    }
    assert orca_registry.config["models"] == expected_orca_models

    # Test for_hardware with no matches
    gpu_registry = registry.for_hardware("GPU")
    assert gpu_registry.config["models"] == {}

    # Test for_task filtering
    face_registry = registry.for_task("face_detection")
    expected_face_models = {
        name: model
        for name, model in sample_models_dict.items()
        if model.get("task") == "face_detection"
    }
    assert face_registry.config["models"] == expected_face_models

    # Test for_task with no matches
    classification_registry = registry.for_task("classification")
    assert classification_registry.config["models"] == {}

    # Test for_alias filtering (using fixture with alias field)
    alias_registry = ModelRegistry(config=sample_config_dict)

    # for_alias positive
    alias_filtered = alias_registry.for_alias("orca_face")
    expected_alias_models = {
        name: model
        for name, model in sample_models_dict.items()
        if "alias" in model and model["alias"] == "orca_face"
    }
    assert alias_filtered.config["models"] == expected_alias_models

    # for_alias negative
    no_alias_filtered = alias_registry.for_alias("nonexistent_alias")
    assert no_alias_filtered.config["models"] == {}

    # Test for_meta filtering (filter by input_size)
    meta_registry = ModelRegistry(config=sample_config_dict)
    meta_filtered = meta_registry.for_meta({"input_size": [640, 640]})
    expected_meta_models = {
        name: model
        for name, model in sample_models_dict.items()
        if model.get("metadata", {}).get("input_size") == [640, 640]
    }
    assert meta_filtered.config["models"] == expected_meta_models

    # for_meta with no matches
    no_meta_filtered = meta_registry.for_meta({"input_size": [999, 999]})
    assert no_meta_filtered.config["models"] == {}

    # Test chained filtering
    filtered_registry = registry.for_task("face_detection").for_hardware("N2X/ORCA1")
    expected_chained_models = {
        name: model
        for name, model in sample_models_dict.items()
        if model.get("task") == "face_detection"
        and "N2X/ORCA1" in model.get("hardware", [])
    }
    assert filtered_registry.config["models"] == expected_chained_models

    # Test for_meta with custom predicate: select models with fps > 20
    meta_registry = ModelRegistry(config=sample_config_dict)
    meta_filtered = meta_registry.for_meta({"fps": lambda v: v["fps"] > 20})
    expected_meta_models = {
        name: model
        for name, model in sample_models_dict.items()
        if model.get("metadata", {}).get("fps", 0) > 20
    }
    assert meta_filtered.config["models"] == expected_meta_models


def test_model_registry_model_specs(sample_models_dict, sample_config_dict):
    """Test all_model_specs and top_model_spec methods with various scenarios"""

    # Test all_model_specs with default parameters
    registry = ModelRegistry(config=sample_config_dict)
    face_registry = registry.for_task("face_detection")

    specs = face_registry.all_model_specs()

    # Should return 2 specs for face detection models
    assert len(specs) == 2
    model_names = [spec.model_name for spec in specs]
    assert "face_detector_orca" in model_names
    assert "face_detector_cpu" in model_names

    # Check default parameters for all specs
    for spec in specs:
        assert spec.inference_host_address == "@cloud"
        assert spec.token is None
        assert spec.model_properties == {}
        model_metadata = sample_models_dict[spec.model_name].get("metadata", {})
        assert spec.metadata == model_metadata

    # Test all_model_specs with custom parameters
    token = "test_token"
    model_properties = {"confidence": 0.8}

    specs = registry.all_model_specs(
        inference_host_address="@localhost",
        zoo_url="custom_zoo",
        token=token,
        model_properties=model_properties,
    )

    # Should return all 4 models with custom parameters
    assert len(specs) == 4
    for spec in specs:
        assert spec.zoo_url == "custom_zoo"  # Overridden
        assert spec.inference_host_address == "@localhost"
        assert spec.token == token
        assert spec.model_properties == model_properties

    # Test top_model_spec with single model
    single_model_registry = registry.for_task("face_detection").for_hardware(
        "N2X/ORCA1"
    )

    single_spec = single_model_registry.top_model_spec()
    assert single_spec.model_name == "face_detector_orca"
    assert single_spec.zoo_url == "https://cs.degirum.com/degirum/orca"
    assert single_spec.inference_host_address == "@cloud"
    assert single_spec.token is None
    assert single_spec.model_properties == {}
    model_metadata = sample_models_dict[single_spec.model_name].get("metadata", {})
    assert single_spec.metadata == model_metadata

    # Test with no models
    empty_registry = ModelRegistry(
        config={"models": {}, "defaults": sample_config_dict["defaults"]}
    )

    with pytest.raises(RuntimeError, match="No models available in the registry"):
        empty_registry.top_model_spec()

    # integration test using ModelSpec and ModelRegistry together
    models_dict = {
        "test_model": {
            "description": "Test model",
            "task": "detection",
            "hardware": ["CPU"],
            "zoo_url": "test_zoo",
        }
    }
    config = {
        "models": models_dict,
        "defaults": sample_config_dict["defaults"],
    }
    registry = ModelRegistry(config=config)
    model_spec = registry.top_model_spec()

    assert isinstance(model_spec, ModelSpec)
    assert model_spec.model_name == "test_model"
    assert model_spec.zoo_url == "test_zoo"
    assert model_spec.inference_host_address == "@cloud"
    assert model_spec.token is None
    assert model_spec.model_properties == {}
    assert model_spec.metadata == models_dict[model_spec.model_name].get("metadata", {})

    for name, model in registry.config["models"].items():
        assert "metadata" in model or model.get("metadata", {}) == {}

    # Test best_model_spec for highest/lowest fps
    registry = ModelRegistry(config=sample_config_dict)
    best_spec = registry.best_model_spec("fps", "max")
    assert best_spec.model_name == "face_detector_orca"
    assert isinstance(best_spec.metadata, dict) and best_spec.metadata["fps"] == 30.0
    worst_spec = registry.best_model_spec("fps", "min")
    assert worst_spec.model_name == "segmentation_cpu"
    assert isinstance(worst_spec.metadata, dict) and worst_spec.metadata["fps"] == 5.0

    # Test with_defaults functionality
    # Change defaults and verify all_model_specs reflects new defaults
    registry = ModelRegistry(config=sample_config_dict)
    registry2 = registry.with_defaults(
        inference_host_address="@local",
        token="abc123",
        zoo_url="custom_zoo",
        model_properties={"confidence": 0.8},
    )
    specs = registry2.all_model_specs()
    for spec in specs:
        assert spec.inference_host_address == "@local"
        assert spec.token == "abc123"
        assert spec.zoo_url == "custom_zoo"
        assert spec.model_properties == {"confidence": 0.8}
    # Original registry should remain unchanged
    orig_specs = registry.all_model_specs()
    for spec in orig_specs:
        assert spec.inference_host_address == "@cloud"
        assert spec.token is None

    # test overrides
    specs = registry2.all_model_specs(
        inference_host_address="localhost",
        token="def456",
        zoo_url="private_zoo",
        model_properties={"postprocess": "None"},
    )
    for spec in specs:
        assert spec.inference_host_address == "localhost"
        assert spec.token == "def456"
        assert spec.zoo_url == "private_zoo"
        assert spec.model_properties == {"confidence": 0.8, "postprocess": "None"}


def test_model_registry_get_methods(sample_models_dict, sample_config_dict):
    """Test get_tasks and get_hardware methods with populated and empty registries"""

    # Test with populated registry
    registry = ModelRegistry(config=sample_config_dict)

    # Test get_tasks method
    tasks = registry.get_tasks()
    expected_tasks = ["face_detection", "object_detection", "segmentation"]
    assert sorted(tasks) == sorted(expected_tasks)

    # Test get_hardware method
    hardware = registry.get_hardware()
    expected_hardware = ["N2X/CPU", "N2X/ORCA1"]
    assert sorted(hardware) == sorted(expected_hardware)

    # Test get_aliases method
    aliases = registry.get_aliases()
    expected_aliases = [
        model["alias"] for model in sample_models_dict.values() if "alias" in model
    ]
    assert sorted(aliases) == sorted(expected_aliases)

    # Test with empty registry
    empty_registry = ModelRegistry(
        config={"models": {}, "defaults": sample_config_dict["defaults"]}
    )

    # Test get_tasks method with empty registry
    empty_tasks = empty_registry.get_tasks()
    assert empty_tasks == []

    # Test get_hardware method with empty registry
    empty_hardware = empty_registry.get_hardware()
    assert empty_hardware == []

    # Test get_aliases method with empty registry
    empty_aliases = empty_registry.get_aliases()
    assert empty_aliases == []
