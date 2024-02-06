#
# setup.py: degirum_tools package setup file
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# NOTE: before making new degirum_tools package release,
# increment the version number in `degirum_tools/_version.py`
#


from setuptools import setup
from pathlib import Path

root_path = Path(__file__).resolve().parent

# get version
exec(open(root_path / "degirum_tools/_version.py").read())

# load README.md
readme = open(root_path / "README.md", encoding="utf-8").read()

setup(
    name="degirum_tools",
    version=__version__,
    description="Tools for PySDK",
    author="DeGirum",
    license="",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=[
        "degirum_tools",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "pillow",
        "requests",
        "psutil",
        "opencv-python",
        "degirum>=0.9.2",
        "ipython",
        "pafy",
        "youtube-dl==2020.12.2",
        "pycocotools",
        "pyyaml",
        "ffmpegcv;platform_system!='Windows'",
    ],
    python_requires=">=3.8",
    extras_require={
        "linting": [
            "black",
            "mypy",
            "flake8",
            "pre-commit",
            "types-Pillow",
            "types-PyYAML",
        ],
        "testing": ["pytest", "coverage"],
        "build": ["build"],
    },
    include_package_data=True,
)
