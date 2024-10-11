#
# conftest.py - DeGirum Tools: pytest configuration file
# Copyright DeGirum Corp. 2024
#
# Contains common pytest configuration and common test fixtures
#
import sys, os, tempfile, pytest, pathlib

# add current directory to sys.path to debug tests locally without package installation
sys.path.insert(0, os.getcwd())

import degirum_tools, degirum as dg

os.environ[degirum_tools.var_TestMode] = "1"  # enable test mode


@pytest.fixture(scope="session")
def image_dir():
    """Test image directory"""
    return os.path.join(os.path.dirname(__file__), "images")


@pytest.fixture(scope="session")
def short_video(image_dir):
    """Path to test short video"""
    file = os.path.join(image_dir, "Traffic2_short.mp4")
    os.environ[degirum_tools.var_VideoSource] = file
    return file


@pytest.fixture
def temp_dir():
    """Temporary directory fixture with cleanup"""
    with tempfile.TemporaryDirectory() as directory:
        yield pathlib.Path(directory)
        # cleanup happens automatically when the block exits


@pytest.fixture(scope="session")
def zoo_dir():
    """Test model zoo directory"""
    return os.path.join(os.path.dirname(__file__), "model-zoo")


@pytest.fixture()
def detection_model_name():
    """Detection model name"""
    return "yolov5nu_relu6_car--128x128_float_n2x_cpu_1"


@pytest.fixture()
def detection_model(zoo_dir, detection_model_name):
    """Load detection model from local zoo"""
    with dg.load_model(detection_model_name, dg.LOCAL, zoo_dir) as model:
        yield model


@pytest.fixture()
def classification_model_name():
    """Classification model name"""
    return "mobilenet_v2_generic_object--224x224_quant_n2x_cpu_1"


@pytest.fixture()
def classification_model(zoo_dir, classification_model_name):
    """Load classification model from local zoo"""
    with dg.load_model(classification_model_name, dg.LOCAL, zoo_dir) as model:
        yield model
