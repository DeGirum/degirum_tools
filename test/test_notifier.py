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


def test_notifier(s3_credentials, msteams_test_workflow_url):
    """
    Test for EventNotifier analyzer
    """

    import degirum_tools, degirum as dg
    import numpy as np

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
        # -------------------------------------------------------
        # Clip saving tests
        # -------------------------------------------------------
        {
            "params": {
                "name": "test",
                "condition": "a",
                "message": "Unit test: [Uploaded file]({url})",
                "notification_tags": "all",
                "notification_config": msteams_test_workflow_url,
                "clip_save": True,
                "clip_sub_dir": "test",
                "clip_duration": 3,
                "clip_pre_trigger_delay": 1,
                "clip_embed_ai_annotations": False,
                "storage_config": degirum_tools.ObjectStorageConfig(**s3_credentials),
            },
            "inp": [
                {"events_detected": {""}},
                {"events_detected": {""}},
                {"events_detected": {"a"}},
                {"events_detected": {""}},
                {"events_detected": {""}},
            ],
            "res": [{}, {}, {"test": "Unit test: [Uploaded file]({url})"}, {}, {}],
        },
    ]

    degirum_tools.logger_add_handler()

    for ci, case in enumerate(test_cases):
        params = case["params"]

        if "res" not in case:
            # expected to fail
            with pytest.raises(Exception):
                notifier = degirum_tools.EventNotifier(**params)
            continue

        if (
            "clip_save" in params
            and params["clip_save"]
            and (
                params["storage_config"].access_key is None
                or params["storage_config"].secret_key is None
            )
        ):
            print(
                f"Case {ci} skipped: S3_ACCESS_KEY and/or S3_SECRET_KEY environment variables are not set"
            )
            continue

        notifier = degirum_tools.EventNotifier(**params)

        try:
            for i, input in enumerate(case["inp"]):

                result = dg.postprocessor.InferenceResults(
                    model_params=None,
                    input_image=(
                        np.zeros((100, 100, 3)) if notifier._clip_save else None
                    ),
                    inference_results={},
                    conversion=None,
                )
                result.__dict__.update(input)

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

        finally:
            if notifier._clip_save:
                # cleanup bucket contents for clip saving tests
                storage = degirum_tools.ObjectStorage(notifier._storage_cfg)
                del notifier  # delete notifier to finalize clip saving
                storage.delete_bucket_contents()
