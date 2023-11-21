#!/usr/bin/env python3

from setuptools import setup

from pathlib import Path

with open(Path(__file__).resolve().parent / "README.md", encoding="utf-8") as f:
    readme = f.read()

setup(
    name="dgtools",
    version="0.1.0",
    description="Tools for PySDK",
    author="DeGirum",
    license="",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=[
        "dgtools",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=[
        "degirum>=0.9.2",
        "numpy",
        "pillow",
        "opencv-python",
        "ipython",
        "pafy",
        "youtube-dl==2020.12.2",
        "pycocotools",
        "pyyaml"
    ],
    python_requires=">=3.8",
    extras_require={
        "linting": [
            "black",
            "mypy",
            "flake8",
            "pre-commit",
            "types-Pillow",
            "types-PyYAML"
        ],
        "testing": [
            "pytest",
            "coverage",
        ],
    },
    include_package_data=True,
)
