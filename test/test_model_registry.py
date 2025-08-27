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
from pathlib import Path
from unittest.mock import patch, mock_open
from degirum_tools.model_registry import ModelSpec, ModelRegistry


@pytest.fixture
def sample_models_dict():
    """Sample models dictionary for testing"""
    return {
        "face_detector_orca": {
            "description": "Face detection model for ORCA",
            "task": "face_detection",
            "hardware": "N2X/ORCA1",
            "zoo_url": "https://cs.degirum.com/degirum/orca",
            "metadata": {"input_size": "640x640"},
        },
        "face_detector_cpu": {
            "description": "Face detection model for CPU",
            "task": "face_detection",
            "hardware": "N2X/CPU",
            "zoo_url": "degirum/public",
        },
        "object_detector_orca": {
            "description": "Object detection model for ORCA",
            "task": "object_detection",
            "hardware": "N2X/ORCA1",
            "zoo_url": "https://cs.degirum.com/degirum/orca",
        },
        "segmentation_cpu": {
            "description": "Segmentation model for CPU",
            "task": "segmentation",
            "hardware": "N2X/CPU",
            "zoo_url": "degirum/public",
        },
    }


@pytest.fixture
def sample_config_dict(sample_models_dict):
    """Sample config dictionary for testing"""
    return {"models": sample_models_dict}


def test_model_registry_creation(sample_models_dict, sample_config_dict):
    """Test ModelRegistry creation with various methods"""

    # Test creation with models dictionary
    registry = ModelRegistry(models=sample_models_dict)
    assert registry.models == sample_models_dict

    # Test creation with config file
    config_yaml = yaml.dump(sample_config_dict)
    with patch("builtins.open", mock_open(read_data=config_yaml)) as mocked_open:
        registry = ModelRegistry(config_file="test_config.yaml")
        assert registry.models == sample_config_dict["models"]
        mocked_open.assert_called_once_with("test_config.yaml", "r")

    # Test creation with default config file
    with patch("builtins.open", mock_open(read_data=config_yaml)) as mocked_open:
        registry = ModelRegistry()
        assert registry.models == sample_config_dict["models"]
        # Should use the default models.yaml file in the same directory as model_registry.py
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


def test_model_registry_filtering_methods(sample_models_dict):
    """Test for_hardware, for_task, and chained filtering methods"""

    registry = ModelRegistry(models=sample_models_dict)

    # Test for_hardware filtering
    orca_registry = registry.for_hardware("N2X/ORCA1")
    expected_orca_models = {
        name: model
        for name, model in sample_models_dict.items()
        if model.get("hardware") == "N2X/ORCA1"
    }
    assert orca_registry.models == expected_orca_models

    # Test for_hardware with no matches
    gpu_registry = registry.for_hardware("GPU")
    assert gpu_registry.models == {}

    # Test for_task filtering
    face_registry = registry.for_task("face_detection")
    expected_face_models = {
        name: model
        for name, model in sample_models_dict.items()
        if model.get("task") == "face_detection"
    }
    assert face_registry.models == expected_face_models

    # Test for_task with no matches
    classification_registry = registry.for_task("classification")
    assert classification_registry.models == {}

    # Test chained filtering
    filtered_registry = registry.for_task("face_detection").for_hardware("N2X/ORCA1")
    expected_chained_models = {
        name: model
        for name, model in sample_models_dict.items()
        if model.get("task") == "face_detection"
        and model.get("hardware") == "N2X/ORCA1"
    }
    assert filtered_registry.models == expected_chained_models


def test_model_registry_model_specs(sample_models_dict):
    """Test model_specs and model_spec methods with various scenarios"""

    # Test model_specs with default parameters
    registry = ModelRegistry(models=sample_models_dict)
    face_registry = registry.for_task("face_detection")

    specs = face_registry.model_specs()

    # Should return 2 specs for face detection models
    assert len(specs) == 2
    model_names = [spec.model_name for spec in specs]
    assert "face_detector_orca" in model_names
    assert "face_detector_cpu" in model_names
    
    # Check default parameters for all specs
    for spec in specs:
        assert spec.inference_host_address == "@cloud"
        assert spec.connect_kwargs == {}
        assert spec.load_kwargs == {}

    # Test model_specs with custom parameters
    connect_kwargs = {"token": "test_token"}
    load_kwargs = {"confidence": 0.8}

    specs = registry.model_specs(
        inference_host_address="@localhost",
        zoo_url="custom_zoo",
        connect_kwargs=connect_kwargs,
        load_kwargs=load_kwargs,
    )

    # Should return all 4 models with custom parameters
    assert len(specs) == 4
    for spec in specs:
        assert spec.zoo_url == "custom_zoo"  # Overridden
        assert spec.inference_host_address == "@localhost"
        assert spec.connect_kwargs == connect_kwargs
        assert spec.load_kwargs == load_kwargs

    # Test model_spec with single model
    single_model_registry = registry.for_task("face_detection").for_hardware("N2X/ORCA1")
    
    single_spec = single_model_registry.model_spec()
    assert single_spec.model_name == "face_detector_orca"
    assert single_spec.zoo_url == "https://cs.degirum.com/degirum/orca"
    assert single_spec.inference_host_address == "@cloud"
    assert single_spec.connect_kwargs == {}
    assert single_spec.load_kwargs == {}

    # Test with no models
    empty_registry = ModelRegistry(models={})

    with pytest.raises(RuntimeError, match="No models available in the registry"):
        empty_registry.model_spec()

    # Test with multiple models (should raise error for model_spec)
    with pytest.raises(RuntimeError, match="Multiple models available in the registry; use model_specs\\(\\) instead"):
        face_registry.model_spec()

    # integration test using ModelSpec and ModelRegistry together
    models_dict = {
        "test_model": {
            "description": "Test model",
            "task": "detection",
            "hardware": "CPU",
            "zoo_url": "test_zoo",
        }
    }

    registry = ModelRegistry(models=models_dict)
    model_spec = registry.model_spec()

    assert isinstance(model_spec, ModelSpec)
    assert model_spec.model_name == "test_model"
    assert model_spec.zoo_url == "test_zoo"
    assert model_spec.inference_host_address == "@cloud"
    assert model_spec.connect_kwargs == {}
    assert model_spec.load_kwargs == {}


def test_model_registry_get_methods(sample_models_dict):
    """Test get_tasks and get_hardware methods with populated and empty registries"""

    # Test with populated registry
    registry = ModelRegistry(models=sample_models_dict)

    # Test get_tasks method
    tasks = registry.get_tasks()
    expected_tasks = ["face_detection", "object_detection", "segmentation"]
    assert sorted(tasks) == sorted(expected_tasks)

    # Test get_hardware method
    hardware = registry.get_hardware()
    expected_hardware = ["N2X/CPU", "N2X/ORCA1"]
    assert sorted(hardware) == sorted(expected_hardware)

    # Test with empty registry
    empty_registry = ModelRegistry(models={})

    # Test get_tasks method with empty registry
    empty_tasks = empty_registry.get_tasks()
    assert empty_tasks == []

    # Test get_hardware method with empty registry
    empty_hardware = empty_registry.get_hardware()
    assert empty_hardware == []
