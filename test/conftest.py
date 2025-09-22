#
# conftest.py - DeGirum Tools: pytest configuration file
# Copyright DeGirum Corp. 2025
#
# Contains common pytest configuration and common test fixtures
#
import sys, os, tempfile, pytest, pathlib

# add current directory to sys.path to debug tests locally without package installation
sys.path.insert(0, os.getcwd())

import degirum as dg
import degirum_tools
import logging


def pytest_addoption(parser):
    """Add custom command line options for pytest"""

    parser.addoption(
        "--loglevel",
        action="store",
        default=None,
        help="Set log level (e.g. DEBUG, INFO, WARNING)",
    )
    parser.addoption(
        "--token", action="store", default="", help="cloud server token value to use"
    )


def pytest_configure(config):
    """Configure pytest with custom options"""

    loglevel = config.getoption("--loglevel")
    if loglevel:
        dg.enable_default_logger(getattr(logging, loglevel.upper(), logging.ERROR))


@pytest.fixture(scope="session", autouse=True)
def cloud_token(request):
    """Get cloud server token passed from the command line and install it system-wide"""
    token = request.config.getoption("--token")
    if token:
        from degirum._tokens import TokenManager

        TokenManager().token_install(token, True)


@pytest.fixture(scope="session")
def image_dir():
    """Test image directory"""
    return os.path.join(os.path.dirname(__file__), "images")


@pytest.fixture(scope="session")
def short_video(image_dir):
    """Path to test short video"""
    return os.path.join(image_dir, "Traffic2_short.mp4")


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
def dummy_model(zoo_dir):
    """Load dummy model from local zoo"""
    with dg.load_model("dummy", dg.LOCAL, zoo_dir) as model:
        yield model


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


@pytest.fixture()
def regression_model_name():
    """Regression model name"""
    return "yolov8n_relu6_age--256x256_quant_tflite_cpu_1"


@pytest.fixture()
def regression_model(zoo_dir, regression_model_name):
    """Load regression model from local zoo"""
    with dg.load_model(regression_model_name, dg.LOCAL, zoo_dir) as model:
        yield model


@pytest.fixture(scope="session")
def s3_credentials():
    degirum_tools.environment.reload_env()
    return dict(
        endpoint="s3.us-west-1.amazonaws.com",
        access_key=os.getenv(degirum_tools.environment.var_S3AccessKey),
        secret_key=os.getenv(degirum_tools.environment.var_S3SecretKey),
        bucket="dg-degirum-tools-test-s3",
    )


@pytest.fixture(scope="session")
def msteams_test_workflow_url():
    degirum_tools.environment.reload_env()
    return os.getenv(
        degirum_tools.environment.var_MSTeamsTestWorkflowURL, "json://unittest"
    )
