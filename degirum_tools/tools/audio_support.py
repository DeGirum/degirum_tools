#
# audio_support.py: audio stream handling classes and functions
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements classes and functions to handle audio streams
#

"""
Audio Support Module Overview
============================

This module provides utilities for opening microphone or file-based audio streams and for
generating audio frames. The module provides context managers and generators that simplify
audio capture and processing workflows.

Key Features:
    - **Audio Stream Management**: Open and manage audio streams from microphones and files
    - **Buffer Generation**: Generate audio frames with configurable buffer sizes
    - **Overlapping Buffers**: Support for overlapping audio buffers
    - **Format Support**: Handle WAV files and microphone input
    - **Stream Control**: Non-blocking and blocking stream operations
    - **Error Handling**: Robust error handling for stream operations

Typical Usage:
    1. Call `open_audio_stream()` to create an audio stream
    2. Iterate over `audio_source()` or `audio_overlapped_source()` to process frames
    3. Use non-blocking mode for real-time processing
    4. Handle stream errors and cleanup

Integration Notes:
    - Works with PyAudio for microphone input
    - Supports WAV file format
    - Handles both local and remote audio sources
    - Provides consistent interface across platforms

Key Functions:
    - `open_audio_stream()`: Context manager for microphone or WAV files
    - `audio_source()`: Generator yielding audio buffers
    - `audio_overlapped_source()`: Generator yielding overlapping buffers

Configuration Options:
    - Sampling rate
    - Buffer size
    - Source selection
    - Blocking mode
"""

import queue, numpy as np
from contextlib import contextmanager
from . import environment as env
from typing import Union, Callable, Generator, Optional, Any


@contextmanager
def open_audio_stream(
    sampling_rate_hz: int, buffer_size: int, audio_source: Union[int, str, None] = None
) -> Generator[Any, None, None]:
    """Open an audio stream.

    This context manager opens a microphone or WAV file as an audio stream and
    automatically closes it when the context exits.

    Args:
        sampling_rate_hz (int): Desired sample rate in hertz.
        buffer_size (int): Buffer size in frames.
        audio_source (Union[int, str, None], optional): Source identifier. Use an
            integer for a microphone index, a string for a WAV file path or URL,
            or ``None`` to use the default source.

    Yields:
        Stream-like object with get() method returning audio buffers.

    Raises:
        Exception: If the audio stream cannot be opened or the WAV file format
            is invalid.
    """

    pyaudio = env.import_optional_package("pyaudio")

    if env.get_test_mode() or audio_source is None:
        audio_source = env.get_var(env.var_AudioSource, 0)
        if isinstance(audio_source, str) and audio_source.isnumeric():
            audio_source = int(audio_source)

    if isinstance(audio_source, int):
        # microphone

        class MicStream:
            def __init__(self, mic_id, sampling_rate_hz, buffer_size):
                self._audio = pyaudio.PyAudio()
                self._result_queue = queue.Queue()  # type: queue.Queue

                def callback(
                    in_data,  # recorded data if input=True; else None
                    frame_count,  # number of frames
                    time_info,  # dictionary
                    status_flags,
                ):  # PaCallbackFlags
                    self._result_queue.put(in_data)
                    return (None, pyaudio.paContinue)

                self.frames_per_buffer = int(buffer_size)
                self._stream = self._audio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=int(sampling_rate_hz),
                    input=True,
                    input_device_index=audio_source,
                    frames_per_buffer=self.frames_per_buffer,
                    stream_callback=callback,
                )

            def __enter__(self):
                """Return self for context manager support."""
                pass

            def __exit__(self, exc_type, exc_val, exc_tb):
                """Release audio resources when exiting the context."""
                self._stream.stop_stream()  # stop audio streaming
                self._stream.close()  # close audio stream
                self._audio.terminate()  # terminate audio library

            def get(self, no_wait=False):
                """Return the next audio block or ``None`` if not available."""
                if no_wait:
                    return self._result_queue.get_nowait()
                else:
                    return self._result_queue.get()

        yield MicStream(audio_source, sampling_rate_hz, buffer_size)

    else:
        # file
        import wave

        class WavStream:
            def __init__(self, filename, sampling_rate_hz, buffer_size):
                if filename.startswith("http"):
                    import requests, io  # type: ignore[import-untyped]

                    # download file from URL and treat response as file-like object
                    response = requests.get(filename)
                    if response.status_code != 200:
                        raise Exception(
                            f"Failed to download {filename}: {response.reason}"
                        )
                    # treat response as file-like object
                    filename = io.BytesIO(response.content)

                self._wav = wave.open(filename, "rb")

                if self._wav.getnchannels() != 1:
                    raise Exception(f"{filename} should be mono WAV file")

                if self._wav.getsampwidth() != 2:
                    raise Exception(f"{filename} should have 16-bit samples")

                if self._wav.getframerate() != sampling_rate_hz:
                    raise Exception(
                        f"{filename} should have {sampling_rate_hz} Hz sampling rate"
                    )

                self.frames_per_buffer = buffer_size
                self._sample_width = self._wav.getsampwidth()

            def __enter__(self):
                """Return self for context manager support."""
                pass

            def __exit__(self, exc_type, exc_val, exc_tb):
                """Close the WAV file on context exit."""
                self._wav.close()

            def get(self, no_wait=False):
                """Return the next audio block from the WAV file."""
                buf = self._wav.readframes(self.frames_per_buffer)
                if len(buf) < self.frames_per_buffer * self._sample_width:
                    raise StopIteration
                return buf

        yield WavStream(audio_source, sampling_rate_hz, buffer_size)


def audio_source(
    stream: Any, check_abort: Callable[[], bool], non_blocking: bool = False
) -> Generator[Optional[np.ndarray], None, None]:
    """Yield audio frames from a stream.

    Args:
        stream (Any): Audio stream object returned by ``open_audio_stream()``.
        check_abort (Callable[[], bool]): Callback that returns ``True`` to stop
            iteration.
        non_blocking (bool, optional): If ``True`` and no frame is available,
            ``None`` is yielded instead of blocking. Defaults to ``False``.

    Yields:
        Waveform of int16 samples or None when no data is available and non_blocking is True.
    """

    try:
        while not check_abort():
            if non_blocking:
                try:
                    block = stream.get(True)
                except queue.Empty:
                    block = None
            else:
                block = stream.get()

            yield None if block is None else np.frombuffer(block, dtype=np.int16)
    except StopIteration:
        pass


def audio_overlapped_source(
    stream: Any, check_abort: Callable[[], bool], non_blocking: bool = False
) -> Generator[Optional[np.ndarray], None, None]:
    """Generate audio frames with 50% overlap.

    The function reads blocks from ``stream`` and yields frames that overlap by
    half of their length. Overlapping frames produce smoother results for audio
    analysis.

    Args:
        stream (Any): Audio stream object returned by ``open_audio_stream()``.
        check_abort (Callable[[], bool]): Callback that returns ``True`` to stop
            iteration.
        non_blocking (bool, optional): If ``True`` and no frame is available,
            ``None`` is yielded instead of blocking. Defaults to ``False``.

    Yields:
        Waveform of int16 samples with 50% overlap, or None when no data is available and non_blocking is True.
    """

    chunk_length = stream.frames_per_buffer
    data = np.zeros(2 * chunk_length, dtype=np.int16)
    try:
        while not check_abort():
            if non_blocking:
                try:
                    block = stream.get(True)
                except queue.Empty:
                    block = None
            else:
                block = stream.get()

            if block is None:
                yield None
            else:
                data[:chunk_length] = data[chunk_length:]
                data[chunk_length:] = np.frombuffer(block, dtype=np.int16)
                yield data
    except StopIteration:
        pass
