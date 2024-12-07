#
# degirum_tools.py: toolkit for PySDK samples
#
# Copyright DeGirum Corporation 2023
# All rights reserved
#

import logging
from typing import Optional


def logger_get():
    """
    Get the package logger.

    Returns:
        Returns the package logger.
    """
    return logging.getLogger(__name__)


def logger_add_handler(
    handler: Optional[logging.Handler] = None,
    format: str = "",
    level: int = logging.DEBUG,
) -> logging.Handler:
    """
    Add a handler to the package logger.

    Args:
        handler: Handler to add to the logger. If None, a new StreamHandler to console is added.
        format: Format string for handler formatter. Defaults to "%(asctime)s [%(levelname)s][%(threadName)s] %(message)s".
        level: Logging level as defined in logging python package. Defaults to logging.DEBUG.

    Returns:
        Returns an instance of added handler.
    """
    logger = logger_get()

    if handler is None:
        handler = logging.StreamHandler()

    if not format:
        format = "%(asctime)s [%(levelname)s][%(threadName)s] %(message)s"
    handler.setFormatter(logging.Formatter(format))
    logger.addHandler(handler)
    logger.setLevel(level)
    return handler


# flake8: noqa

from ._version import __version__, __version_info__
from .audio_support import *
from .classification_eval import *
from .clip_saver import *
from .compound_models import *
from .crop_extent import *
from .detection_eval import *
from .environment import *
from .event_detector import *
from .inference_support import *
from .line_count import *
from .math_support import *
from .notifier import *
from .object_selector import *
from .object_storage_support import *
from .object_tracker import *
from .regression_eval import *
from .ui_support import *
from .video_support import *
from .zone_count import *


# aliases for backward compatibility
from .environment import (
    in_colab as _in_colab,
    reload_env as _reload_env,
    get_test_mode as _get_test_mode,
)
