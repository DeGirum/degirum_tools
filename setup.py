#!/usr/bin/env python3

from setuptools import setup

setup(name='dgtools',
      version='0.0.1',
      description='Tools for PySDK',
      author='DeGirum',
      license='',
      long_description='# dgtools\n## Tools for PySDK',
      long_description_content_type='text/markdown',
      packages=[
        'dgtools',
      ],
      classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
      ],
      install_requires=[
        'degirum>=0.9.2',
        'numpy',
        'pillow',
        'opencv-python',
        'jupyterlab',
        'pafy',
        'youtube-dl==2020.12.2',
      ],
      python_requires='>=3.8',
      extras_require={
        'linting': [
            'mypy',
            'flake8',
            'pre-commit',
            'types-Pillow',
        ],
        'testing': [
            'pytest',
            'coverage',
        ],
      },
      include_package_data=True)
