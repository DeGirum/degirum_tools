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
import socketserver
import http.server
import threading
import re
import degirum_tools, degirum as dg
import numpy as np


class WebhookHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler for webhook notifications."""

    received_notifications: list = []

    def do_POST(self):
        if self.path == "/webhook":
            try:
                content_length = int(self.headers["Content-Length"])
                post_data = self.rfile.read(content_length)
                notification_data = post_data.decode("utf-8")
                WebhookHandler.received_notifications.append(notification_data)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(f'{{"error": "{str(e)}"}}'.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress server logging


def _run_notifier_tests(s3_credentials, webhook_url):
    """
    Run the actual notifier tests with the webhook URL.
    """

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
                "message": "${result.events_detected}",
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
                "message": "${result.msg}",
                "notification_tags": "Test, Tg",
                "notification_config": webhook_url,
            },
            "inp": [
                {"events_detected": {"a"}, "msg": "e1"},
                {"events_detected": {""}},
                {"events_detected": {"a"}, "msg": "e2"},
            ],
            "res": [{"test": "e1"}, {}, {"test": "e2"}],
            "webhook_expected": ["e1", "e2"],
        },
        # -------------------------------------------------------
        # Clip saving tests
        # -------------------------------------------------------
        {
            "params": {
                "name": "test",
                "condition": "a",
                "message": '{ "Unit test": "[Uploaded file](${url})" }',
                "notification_tags": "all",
                "notification_config": webhook_url,
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
            "res": [
                {},
                {},
                {"test": '{ "Unit test": "[Uploaded file](${url})" }'},
                {},
                {},
            ],
            "webhook_expected": ['{ "Unit test": "[Uploaded file](${url})" }'],
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

        # Clear webhook notifications before each test case
        WebhookHandler.received_notifications = []

        notifier = degirum_tools.EventNotifier(**params)

        try:
            for i, input in enumerate(case["inp"]):

                result = dg.postprocessor.InferenceResults(
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

            notifier.finalize()

            # Validate webhook notifications if expected
            if "webhook_expected" in case:
                received_messages = WebhookHandler.received_notifications
                expected_messages = case["webhook_expected"]

                assert len(received_messages) == len(expected_messages)

                for j, (received, expected) in enumerate(
                    zip(received_messages, expected_messages)
                ):
                    # Treat ${...} placeholders
                    pattern = re.escape(expected)
                    pattern = re.sub(r"\\\$\\\{[^}]+\\\}", r".*?", pattern)

                    assert re.fullmatch(pattern, received), (
                        f"Case {ci} webhook message {j} failed: "
                        + f"received `{received}` "
                        + f"does not match expected pattern `{expected}`"
                    )

        finally:
            if notifier._clip_save:
                # cleanup bucket contents for clip saving tests
                storage = degirum_tools.ObjectStorage(notifier._storage_cfg)
                del notifier  # delete notifier to finalize clip saving
                storage.delete_bucket_contents()


def test_notifier(s3_credentials):
    """
    Test for EventNotifier analyzer
    """

    # Start HTTP server for webhook testing
    with socketserver.TCPServer(("localhost", 0), WebhookHandler) as httpd:
        port = httpd.server_address[1]
        webhook_url = f"http://127.0.0.1:{port}/webhook"

        # Start server in a background thread
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()

        try:
            _run_notifier_tests(s3_credentials, webhook_url)
        finally:
            httpd.shutdown()
