#
# test_notifier.py: unit tests for notification analyzer functionality
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements unit tests to test notification analyzer functionality
#

from typing import List
import pytest
import logging


def test_notifier():
    """
    Test for EventNotifier analyzer
    """

    import degirum_tools

    # helper class to convert dictionary to object
    class D2C:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    test_cases: List[dict] = [
        # ----------------------------------------------------------------
        # Expected to fail tests
        # ----------------------------------------------------------------
        # wrong holdoff: expected to fail
        {
            "params": {"name": "test", "condition": "myevent", "holdoff": "wrong"},
        },
        # wrong holdoff unit: expected to fail
        {
            "params": {
                "name": "test",
                "condition": "myevent",
                "holdoff": [0, "parrots"],
            },
        },
        # condition with syntax error: expected to fail
        {
            "params": {"name": "test", "condition": "a And b"},
        },
        # no `events_detected` in the result: expected to fail
        {
            "params": {"name": "test", "condition": "myevent"},
            "inp": [{}],
            "res": None,
        },
        # ----------------------------------------------------------------
        # Basic tests
        # ----------------------------------------------------------------
        # no events detected
        {
            "params": {"name": "test", "condition": "myevent"},
            "inp": [{"events_detected": set()}],
            "res": [{}],
        },
        # single event detected, no holdoff, default message
        {
            "params": {"name": "test", "condition": "myevent"},
            "inp": [{"events_detected": {"myevent"}}],
            "res": [{"test": "Notification triggered: test"}],
        },
        # single event detected, no holdoff, custom message with formatting
        {
            "params": {
                "name": "test",
                "condition": "myevent",
                "message": "{result.events_detected}",
            },
            "inp": [{"events_detected": {"myevent"}}],
            "res": [{"test": "{'myevent'}"}],
        },
        # some formula for event condition
        {
            "params": {"name": "test", "condition": "a and b and not c"},
            "inp": [{"events_detected": {"a", "b"}}],
            "res": [{"test": "Notification triggered: test"}],
        },
        # ----------------------------------------------------------------
        # Holdoff tests
        # ----------------------------------------------------------------
        # holdoff in frames (implicitly specified)
        {
            "params": {
                "name": "test",
                "condition": "a",
                "message": "OK",
                "holdoff": 2,
            },
            "inp": [
                {"events_detected": {}},
                {"events_detected": {"a"}},
                {"events_detected": {"a"}},
                {"events_detected": {"a"}},
                {"events_detected": {}},
                {"events_detected": {}},
                {"events_detected": {"a"}},
                {"events_detected": {}},
                {"events_detected": {"a"}},
            ],
            "res": [{}, {"test": "OK"}, {}, {}, {}, {}, {"test": "OK"}, {}, {}],
        },
        # holdoff in frames (explicitly specified as tuple)
        {
            "params": {
                "name": "test",
                "condition": "a",
                "message": "OK",
                "holdoff": (2, "frames"),
            },
            "inp": [
                {"events_detected": {}},
                {"events_detected": {"a"}},
                {"events_detected": {"a"}},
                {"events_detected": {"a"}},
                {"events_detected": {}},
                {"events_detected": {}},
                {"events_detected": {"a"}},
                {"events_detected": {}},
                {"events_detected": {"a"}},
            ],
            "res": [{}, {"test": "OK"}, {}, {}, {}, {}, {"test": "OK"}, {}, {}],
        },
        # holdoff in seconds (implicitly specified)
        {
            "params": {
                "name": "test",
                "condition": "a",
                "message": "OK",
                "holdoff": 10.0,
            },
            "inp": [
                {"events_detected": {}},
                {"events_detected": {"a"}},
                {"events_detected": {"a"}},
                {"events_detected": {"a"}},
                {"events_detected": {}},
                {"events_detected": {}},
                {"events_detected": {"a"}},
                {"events_detected": {}},
                {"events_detected": {"a"}},
            ],
            "res": [{}, {"test": "OK"}, {}, {}, {}, {}, {}, {}, {}],
        },
        # holdoff in seconds (explicitly specified as list)
        {
            "params": {
                "name": "test",
                "condition": "a",
                "message": "OK",
                "holdoff": [10.0, "seconds"],
            },
            "inp": [
                {"events_detected": {}},
                {"events_detected": {"a"}},
                {"events_detected": {"a"}},
                {"events_detected": {"a"}},
                {"events_detected": {}},
                {"events_detected": {}},
                {"events_detected": {"a"}},
                {"events_detected": {}},
                {"events_detected": {"a"}},
            ],
            "res": [{}, {"test": "OK"}, {}, {}, {}, {}, {}, {}, {}],
        },
        # -------------------------------------------------------
        # Notification Tests
        # -------------------------------------------------------
        # Same test as before, but now with a notification added
        {
            "params": {
                "name": "test",
                "condition": "a",
                "message": "{result.msg}",
                "notification_tags": "Test, Tg",
                "notification_config": "json://unittest",
            },
            "inp": [
                {"events_detected": {"a"}, "msg": "e1"},
                {"events_detected": {""}},
                {"events_detected": {"a"}, "msg": "e2"},
            ],
            "res": [{"test": "e1"}, {}, {"test": "e2"}],
        },
    ]

    for ci, case in enumerate(test_cases):
        if "res" not in case:
            # expected to fail
            with pytest.raises(Exception):
                notifier = degirum_tools.EventNotifier(**case["params"])
            continue

        notifier = degirum_tools.EventNotifier(**case["params"])

        for i, input in enumerate(case["inp"]):
            result = D2C(**input)

            if case["res"] is None:
                with pytest.raises(Exception):
                    notifier.analyze(result)
            else:
                notifier.analyze(result)
                assert (
                    result.notifications == case["res"][i]  # type: ignore[attr-defined]
                ), (
                    f"Case {ci} failed at step {i}: "
                    + f"notifications `{result.notifications}` "  # type: ignore[attr-defined]
                    + f"do not match expected `{case['res'][i]}`."
                    + f"\nConfig: {case['params']}"
                )
