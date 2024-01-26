#
# conftest.py - DeGirum Tools: pytest configuration file
# Copyright DeGirum Corp. 2024
#
# Contains common pytest configuration and common test fixtures
#
import sys, os

# add current directory to sys.path to debug tests locally without package installation
sys.path.insert(0, os.getcwd())
