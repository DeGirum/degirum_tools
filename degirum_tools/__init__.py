#
# degirum_tools.py: toolkit for PySDK samples
#
# Copyright DeGirum Corporation 2023
# All rights reserved
#

# flake8: noqa

import argparse

from ._version import __version__, __version_info__
from .audio_support import *
from .classification_eval import *
from .compound_models import *
from .crop_extent import *
from .detection_eval import *
from .environment import *
from .event_detector import *
from .figure_annotator import *
from .inference_support import *
from .line_count import *
from .math_support import *
from .notifier import *
from .object_selector import *
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


def _command_entrypoint(arg_str=None):
    from .figure_annotator import (
        _figure_annotator_args,
        _zone_annotator_args,
        _line_annotator_args,
    )

    parser = argparse.ArgumentParser(description="DeGirum tools")

    subparsers = parser.add_subparsers(
        help="use -h flag to see help on subcommands", required=True
    )

    # figure_annotator subcommand
    subparser = subparsers.add_parser(
        "figure_annotator",
        description="Launch interactive utility for geometric figure annotation in images",
        help="launch interactive utility for geometric figure annotation in images",
    )
    _figure_annotator_args(subparser)

    # zone_annotator subcommand
    subparser = subparsers.add_parser(
        "zone_annotator",
        description="Launch interactive utility for 4-sided zone annotation in images",
        help="launch interactive utility for 4-sided zone annotation in images",
    )
    _zone_annotator_args(subparser)

    # line_annotator subcommand
    subparser = subparsers.add_parser(
        "line_annotator",
        description="Launch interactive utility for line annotation in images",
        help="launch interactive utility for line annotation in images",
    )
    _line_annotator_args(subparser)

    # parse args
    args = parser.parse_args(arg_str.split() if arg_str else None)

    # execute subcommand
    args.func(args)
