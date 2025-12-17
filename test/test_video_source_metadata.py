#
# test_video_source_metadata.py: unit tests for video_source metadata functionality
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements unit tests to test video_source metadata feature
#

import pytest
import numpy as np
import cv2
import time
from degirum_tools import video_source, open_video_stream


def test_video_source_metadata_structure():
    """Test that video_source with include_metadata=True yields correct structure"""

    # Create a simple test video in memory
    width, height = 640, 480
    fps_val = 30.0

    # Create a temporary video file
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Write a few frames to the video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(tmp_path, fourcc, fps_val, (width, height))

        for i in range(5):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, i % 3] = 255  # Different color each frame
            out.write(frame)

        out.release()

        # Test video_source with metadata
        with open_video_stream(tmp_path) as stream:
            frame_count = 0
            for item in video_source(stream, include_metadata=True):
                frame, metadata = item

                # Verify it's a tuple
                assert isinstance(item, tuple)
                assert len(item) == 2

                # Verify frame is numpy array
                assert isinstance(frame, np.ndarray)
                assert frame.shape == (height, width, 3)

                # Verify metadata is a dict with nested structure
                assert isinstance(metadata, dict)
                assert "video_source" in metadata

                video_meta = metadata["video_source"]

                # Check all required fields
                assert "timestamp" in video_meta
                assert "frame_id" in video_meta
                assert "source_fps" in video_meta
                assert "target_fps" in video_meta
                assert "frame_width" in video_meta
                assert "frame_height" in video_meta

                # Verify types
                assert isinstance(video_meta["timestamp"], float)
                assert isinstance(video_meta["frame_id"], int)
                assert isinstance(video_meta["source_fps"], float)
                assert video_meta["target_fps"] is None  # No fps parameter passed
                assert isinstance(video_meta["frame_width"], int)
                assert isinstance(video_meta["frame_height"], int)

                # Verify values
                assert video_meta["frame_id"] == frame_count
                assert video_meta["frame_width"] == width
                assert video_meta["frame_height"] == height
                assert abs(video_meta["source_fps"] - fps_val) < 1.0

                # Verify timestamp is recent
                assert abs(time.time() - video_meta["timestamp"]) < 5.0

                frame_count += 1
                if frame_count >= 3:
                    break

            assert frame_count == 3

    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_video_source_without_metadata():
    """Test that video_source with include_metadata=False yields only frames"""

    # Create a simple test video in memory
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Write a few frames
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(tmp_path, fourcc, 30.0, (320, 240))

        for i in range(3):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            out.write(frame)

        out.release()

        # Test video_source without metadata (default)
        with open_video_stream(tmp_path) as stream:
            frame_count = 0
            for frame in video_source(stream):
                # Should be a numpy array, not a tuple
                assert isinstance(frame, np.ndarray)
                assert not isinstance(frame, tuple)
                assert frame.shape == (240, 320, 3)

                frame_count += 1
                if frame_count >= 2:
                    break

            assert frame_count == 2

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_video_source_metadata_with_fps():
    """Test that target_fps is correctly set in metadata when fps parameter is provided"""

    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Write frames
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(tmp_path, fourcc, 30.0, (320, 240))

        for i in range(10):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            out.write(frame)

        out.release()

        # Test with fps parameter
        target_fps = 10.0
        with open_video_stream(tmp_path) as stream:
            for item in video_source(stream, fps=target_fps, include_metadata=True):
                frame, metadata = item

                video_meta = metadata["video_source"]
                assert video_meta["target_fps"] == target_fps
                assert video_meta["source_fps"] == 30.0

                break  # Just check first frame

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_video_source_metadata_frame_id_increment():
    """Test that frame_id increments correctly"""

    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Write frames
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(tmp_path, fourcc, 30.0, (320, 240))

        for i in range(5):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            out.write(frame)

        out.release()

        # Test frame_id increments
        with open_video_stream(tmp_path) as stream:
            frame_ids = []
            for item in video_source(stream, include_metadata=True):
                frame, metadata = item
                frame_ids.append(metadata["video_source"]["frame_id"])

                if len(frame_ids) >= 4:
                    break

            # Verify sequential frame IDs starting from 0
            assert frame_ids == [0, 1, 2, 3]

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
