# degirum_tools - DeGirum utilities for PySDK

[![Unit Tests](https://github.com/DeGirum/degirum_tools/actions/workflows/test.yml/badge.svg)](https://github.com/DeGirum/degirum_tools/actions/workflows/test.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

## Installation

degirum_tools can be installed directly from this repository:

```sh
python3 -m pip install git+https://github.com/DeGirum/degirum_tools.git
```

## Release

Release procedure [is described here](https://degirum.atlassian.net/wiki/spaces/SD/pages/1916076041/degirum+tools+Package+Release+Procedure)

## Test Gstream

export PYTHONPATH=$(pwd)/degirum_tools:$PYTHONPATH
python test/test_gstream.py