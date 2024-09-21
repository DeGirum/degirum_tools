#
# conftest.py - DeGirum Tools: pytest configuration file
# Copyright DeGirum Corp. 2024
#
# Contains common pytest configuration and common test fixtures
#
import sys, os
import pytest

# add current directory to sys.path to debug tests locally without package installation
sys.path.insert(0, os.getcwd())

import degirum_tools

os.environ[degirum_tools.var_TestMode] = "1"  # enable test mode


@pytest.fixture(scope="session")
def image_dir():
    """Test image directory"""
    return os.path.join(os.path.dirname(__file__), "images")


@pytest.fixture(scope="session")
def short_video(image_dir):
    """Path to test short video"""
    file = os.path.join(image_dir, "TrafficHD_short.mp4")
    os.environ[degirum_tools.var_VideoSource] = file
    return file
