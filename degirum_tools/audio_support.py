#
# audio_support.py: audio stream handling classes and functions
#
# Copyright DeGirum Corporation 2023
# All rights reserved
#
# Implements classes and functions to handle audio streams
#

import queue, numpy as np
from contextlib import contextmanager
from . import environment as env
from typing import Union, Callable, Generator, Optional, Any


@contextmanager
def open_audio_stream(
    sampling_rate_hz: int, buffer_size: int, audio_source: Union[int, str, None] = None
) -> Generator[Any, None, None]:
    """Open PyAudio audio stream

    Args:
        sampling_rate_hz - desired sample rate in Hz
        buffer_size - read buffer size
        audio_source - 0-based index for local microphones or local WAV file path
    Returns context manager yielding audio stream object and closing it on exit
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
                pass

            def __exit__(self, exc_type, exc_val, exc_tb):
                self._stream.stop_stream()  # stop audio streaming
                self._stream.close()  # close audio stream
                self._audio.terminate()  # terminate audio library

            def get(self, no_wait=False):
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
                pass

            def __exit__(self, exc_type, exc_val, exc_tb):
                self._wav.close()

            def get(self, no_wait=False):
                buf = self._wav.readframes(self.frames_per_buffer)
                if len(buf) < self.frames_per_buffer * self._sample_width:
                    raise StopIteration
                return buf

        yield WavStream(audio_source, sampling_rate_hz, buffer_size)


def audio_source(
    stream, check_abort: Callable[[], bool], non_blocking: bool = False
) -> Generator[Optional[np.ndarray], None, None]:
    """Generator function, which returns audio frames captured from given audio stream.
    Useful to pass to model batch_predict().

    stream - audio stream context manager object returned by open_audio_stream()
    check_abort - check-for-abort function or lambda; stream will be terminated when it returns True
    non_blocking - True for non-blocking mode (immediately yields None if a block is not captured yet)
        False for blocking mode (waits for the end of the block capture and always yields captured block)

    Yields audio waveform captured from given audio stream
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
    stream, check_abort: Callable[[], bool], non_blocking: bool = False
) -> Generator[Optional[np.ndarray], None, None]:
    """Generator function, which returns audio frames captured from given audio stream with half-length overlap.
    Useful to pass to model batch_predict().

    stream - audio stream context manager object returned by open_audio_stream()
    check_abort - check-for-abort function or lambda; stream will be terminated when it returns True
    non_blocking - True for non-blocking mode (immediately yields None if a block is not captured yet)
        False for blocking mode (waits for the end of the block capture and always yields captured block)

    Yields audio waveform captured from given audio stream with half-length overlap.
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
