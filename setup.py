#
# setup.py: degirum_tools package setup file
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# NOTE: before making new degirum_tools package release,
# increment the version number in `degirum_tools/_version.py`
#


from setuptools import setup, find_packages
from pathlib import Path

root_path = Path(__file__).resolve().parent

# get version
exec(open(root_path / "degirum_tools/_version.py").read())

# load README.md
readme = open(root_path / "README.md", encoding="utf-8").read()

setup(
    name="degirum_tools",
    version=__version__,  # noqa
    description="Tools for PySDK",
    author="DeGirum",
    license="MIT",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    entry_points={
        "console_scripts": [
            "degirum_tools = degirum_tools:_command_entrypoint",
        ]
    },
    install_requires=[
        line
        for line in open(root_path / "requirements.txt")
        if line and not line.startswith("#")
    ],
    python_requires=">=3.8",
    # extras
    extras_require={
        # linters for CI/CD
        "linting": [
            "black",
            "mypy",
            "flake8",
            "pre-commit",
            "types-Pillow",
            "types-PyYAML",
        ],
        # testing for CI/CD
        "testing": ["pytest", "coverage"],
        # building for CI/CD
        "build": ["build"],
        # external notifications
        "notifications": ["apprise", "minio==7.2.18"],
        # annotation tool
        "annotator": ["tk"],
    },
    include_package_data=True,
)
