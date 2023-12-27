#!/usr/bin/env python3

from setuptools import setup

from pathlib import Path

with open(Path(__file__).resolve().parent / "README.md", encoding="utf-8") as f:
    readme = f.read()

setup(
    name="degirum_tools",
    version="0.4.4",
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
