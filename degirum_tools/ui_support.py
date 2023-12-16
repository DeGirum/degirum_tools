#
# ui_support.py: UI support classes and functions
#
# Copyright DeGirum Corporation 2023
# All rights reserved
#
# Implements classes and functions to handle image display, progress indication, etc.
#

import cv2, os, time, PIL.Image, numpy as np
from .environment import get_test_mode, in_colab, in_notebook
from dataclasses import dataclass
from typing import Optional, Any
from enum import Enum


def luminance(color: tuple) -> float:
    """Calculate luminance from RGB color"""
    return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]


def deduce_text_color(bg_color: tuple):
    """Deduce text color from background color"""
    return (0, 0, 0) if luminance(bg_color) > 180 else (255, 255, 255)


def color_complement(color):
    """Return color complement: 255 - color"""
    adj_color = (color[0] if isinstance(color, list) else color)[::-1]
    return tuple([255 - c for c in adj_color])


def crop(img, bbox: list):
    """Crop and return OpenCV image to given bbox"""
    return img[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]


class CornerPosition(Enum):
    """Corner position options"""

    TOP_LEFT = 1
    TOP_RIGHT = 2
    BOTTOM_LEFT = 3
    BOTTOM_RIGHT = 4


def put_text(
    image: np.ndarray,
    label: str,
    corner_xy: tuple,
    *,
    corner_position: CornerPosition = CornerPosition.TOP_LEFT,
    font_color: tuple,
    bg_color: Optional[tuple] = None,
    font_face: int = cv2.FONT_HERSHEY_PLAIN,
    font_scale: float = 1,
    font_thickness: int = 1,
    line_spacing: float = 1,
):
    """Draw given text on given OpenCV image at given point with given color

    Args:
        image - numpy array with image
        label - text to draw
        corner_xy - text corner coordinate tuple (x,y); meaning depends on `corner_position` argument
        corner_position - where to place text relative to corner_xy
        font_color - text color (BGR)
        bg_color - background color (BGR) or None for transparent
        font_face - font to use
        font_scale - font scale factor to use
        font_thickness - font thickness to use
        line_spacing - line spacing factor
    """

    im_h, im_w = image.shape[:2]
    margin = 4

    @dataclass
    class LineInfo:
        line: str = ""
        x: int = 0
        y: int = 0
        line_height: int = 0
        line_height_no_baseline: int = 0

    top_left_xy = corner_xy
    lines: list[LineInfo] = []
    max_width = 0
    for line in label.splitlines():
        li = LineInfo()
        li.line = line
        (line_width, li.line_height_no_baseline), baseline = cv2.getTextSize(
            line,
            font_face,
            font_scale,
            font_thickness,
        )
        li.x = max(0, top_left_xy[0])
        li.y = max(0, top_left_xy[1])

        li.line_height = li.line_height_no_baseline + baseline + margin
        top_left_xy = (li.x, li.y + int(li.line_height * line_spacing))
        max_width = max(max_width, line_width)
        lines.append(li)

    max_width += margin

    # adjust coordinates according to corner_position option
    if corner_position != CornerPosition.TOP_LEFT:
        y_adjustment = (
            lines[-1].y + lines[-1].line_height - lines[0].y
            if corner_position != CornerPosition.TOP_RIGHT
            else 0
        )
        x_adjustment = max_width if corner_position != CornerPosition.BOTTOM_LEFT else 0
        for li in lines:
            li.x = max(0, li.x - x_adjustment)
            li.y = max(0, li.y - y_adjustment)

    for li in lines:
        if bg_color is not None:
            # get actual mask sizes with regard to image crop
            if im_h - (li.y + li.line_height) <= 0:
                sz_h = max(im_h - li.y, 0)
            else:
                sz_h = li.line_height

            if im_w - (li.x + max_width) <= 0:
                sz_w = max(im_w - li.x, 0)
            else:
                sz_w = max_width

            # add background mask to image
            if sz_h > 0 and sz_w > 0:
                bg_mask = np.zeros((sz_h, sz_w, 3), np.uint8)
                bg_mask[:, :] = np.array(bg_color)
                image[
                    li.y : li.y + sz_h,
                    li.x : li.x + sz_w,
                ] = bg_mask

        # add text to image
        image = cv2.putText(
            image,
            li.line,
            (
                li.x + margin // 2,
                li.y + li.line_height_no_baseline + margin // 2,
            ),  # putText start bottom-left
            font_face,
            font_scale,
            font_color,
            font_thickness,
            cv2.LINE_AA,
        )

    return image


class FPSMeter:
    """Simple FPS meter class"""

    def __init__(self, avg_len: int = 100):
        """Constructor

        avg_len - number of samples to average
        """
        self._avg_len = avg_len
        self.reset()

    def reset(self):
        """Reset accumulators"""
        self._timestamp_ns = -1
        self._duration_ns = -1
        self._count = 0

    def record(self) -> float:
        """Record timestamp and update average duration.

        Returns current average FPS"""
        t = time.time_ns()
        if self._timestamp_ns > 0:
            cur_dur_ns = t - self._timestamp_ns
            self._count = min(self._count + 1, self._avg_len)
            self._duration_ns = (
                self._duration_ns * (self._count - 1) + cur_dur_ns
            ) // self._count
        self._timestamp_ns = t
        return self.fps()

    def fps(self) -> float:
        """Return current average FPS"""
        return 1e9 / self._duration_ns if self._duration_ns > 0 else 0


class Display:
    """Class to handle OpenCV image display"""

    def __init__(
        self,
        capt: str = "<image>",
        show_fps: bool = True,
        w: Optional[int] = None,
        h: Optional[int] = None,
    ):
        """Constructor

        capt - window title
        show_fps - True to show FPS
        show_embedded - True to show graph embedded into the notebook when possible
        w, h - initial window width/hight in pixels; None for autoscale
        """
        self._fps = FPSMeter()

        if not capt:
            raise Exception("Window title must be non-empty")

        self._capt = capt
        self._show_fps = show_fps
        self._window_created = False
        self._no_gui = not Display._check_gui() or get_test_mode()
        self._w = w
        self._h = h
        self._video_writer: Optional[Any] = None
        self._video_file: Optional[str] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # close OpenCV window in any
        if self._window_created:
            cv2.destroyWindow(self._capt)

        # close video writer if any, and show video in Colab
        if self._video_writer is not None:
            self._video_writer.release()
            if in_colab():
                import IPython

                IPython.display.display(
                    IPython.display.Video(self._video_file, embed=True)
                )

        return exc_type is KeyboardInterrupt  # ignore KeyboardInterrupt errors

    @property
    def window_name(self) -> str:
        """
        Returns window name
        """
        return self._capt

    @staticmethod
    def crop(img, bbox: list):
        """Crop and return OpenCV image to given bbox"""
        return crop(img, bbox)

    @staticmethod
    def _check_gui() -> bool:
        """Check if graphical display is supported

        Returns False if not supported
        """
        import platform

        if platform.system() == "Linux":
            return os.environ.get("DISPLAY") is not None
        return True

    @staticmethod
    def _display_fps(img: np.ndarray, fps: float):
        """Helper method to display FPS"""
        put_text(
            img,
            f"{fps:5.1f} FPS",
            (0, 0),
            font_color=(0, 0, 0),
            bg_color=(255, 255, 255),
        )

    def show(self, img: Any, waitkey_delay: int = 1):
        """Show image or model result

        img - numpy array with valid OpenCV image, or PIL image, or model result object
        waitkey_delay - delay in ms for waitKey() call; use 0 to show still images, use 1 for streaming video
        """

        import IPython.display

        # show image in notebook
        def show_in_notebook(img):
            IPython.display.display(PIL.Image.fromarray(img[..., ::-1]), clear=True)

        if hasattr(img, "image_overlay"):
            # special case for model results: call it recursively
            self.show(img.image_overlay, waitkey_delay)
            return

        if isinstance(img, PIL.Image.Image):
            # PIL image: convert to OpenCV format
            img = np.array(img)[:, :, ::-1]

        if isinstance(img, np.ndarray):
            fps = self._fps.record()
            if self._show_fps and fps > 0:
                Display._display_fps(img, fps)

            if in_colab():
                # special case for Colab environment
                if waitkey_delay == 0:
                    # show still image in notebook
                    show_in_notebook(img)
                else:
                    # save videos to file
                    from .video_support import create_video_writer

                    if self._video_writer is None:
                        self._video_file = f"{os.getcwd()}/{self._capt}.mp4"
                        self._video_writer = create_video_writer(
                            self._video_file, img.shape[1], img.shape[0]
                        )
                    self._video_writer.write(img)

                    class printer(str):
                        def __repr__(self):
                            return self

                    if self._video_writer.count % 10 == 0:
                        IPython.display.display(
                            printer(
                                f"{self._video_file}: frame {self._video_writer.count}, {fps:.1f} FPS"
                            ),
                            clear=True,
                        )

            elif self._no_gui and in_notebook():
                # show image in notebook when possible
                show_in_notebook(img)
            else:
                # show image in OpenCV window
                if not self._window_created:
                    cv2.namedWindow(self._capt, cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty(self._capt, cv2.WND_PROP_TOPMOST, 1)
                    if self._w is not None and self._h is not None:
                        cv2.resizeWindow(self._capt, self._w, self._h)
                    else:
                        cv2.resizeWindow(self._capt, img.shape[1], img.shape[0])

                cv2.imshow(self._capt, img)
                self._window_created = True
                key = cv2.waitKey(waitkey_delay) & 0xFF
                if key == ord("x") or key == ord("q"):
                    if self._fps:
                        self._fps.reset()
                    raise KeyboardInterrupt
                elif key == 43 or key == 45:  # +/-
                    _, _, w, h = cv2.getWindowImageRect(self._capt)
                    factor = 1.25 if key == 43 else 0.75
                    new_w = max(100, int(w * factor))
                    new_h = int(new_w * img.shape[0] / img.shape[1])
                    cv2.resizeWindow(self._capt, new_w, new_h)

        else:
            raise Exception("Unsupported image type")

    def show_image(self, img: Any):
        """Show still image or model result

        img - numpy array with valid OpenCV image, or PIL image, or model result object
        """
        self.show(img, 0)


class Timer:
    """Simple timer class"""

    def __init__(self):
        """Constructor. Records start time."""
        self._start_time = time.time_ns()

    def __call__(self) -> float:
        """Call method.

        Returns time elapsed (in seconds, since object construction)."""
        return (time.time_ns() - self._start_time) * 1e-9


class Progress:
    """Simple progress indicator"""

    def __init__(
        self,
        last_step: Optional[int] = None,
        *,
        start_step: int = 0,
        bar_len: int = 15,
        speed_units: str = "FPS",
    ):
        """Constructor
        last_step - last step
        start_step - starting step
        bar_len - progress bar length in symbols
        """
        self._display_id: Optional[str] = None
        self._len = bar_len
        self._last_step = last_step
        self._start_step = start_step
        self._time_to_refresh = lambda: time.time() - self._last_update_time > 0.5
        self._speed_units = speed_units
        self.reset()

    def reset(self):
        self._start_time = time.time()
        self._step = self._start_step
        self._percent = 0.0
        self._last_updated_percent = self._percent
        self._last_update_time = 0.0
        self._tip_phase = 0
        self._update()

    def step(self, steps: int = 1):
        """Update progress by given number of steps
        steps - number of steps to advance
        """
        assert (
            self._last_step is not None
        ), "Progress indicator: to do stepping last step must be assigned on construction"
        self._step += steps
        self._percent = (
            100 * (self._step - self._start_step) / (self._last_step - self._start_step)
        )
        if (
            self._percent - self._last_updated_percent >= 100 / self._len
            or self._percent >= 100
            or self._time_to_refresh()
        ):
            self._update()

    @property
    def step_range(self) -> Optional[tuple]:
        """Get start-end step range (if defined)"""
        if self._last_step is not None:
            return (self._start_step, self._last_step)
        else:
            return None

    @property
    def percent(self) -> float:
        return self._percent

    @percent.setter
    def percent(self, value: float):
        v = float(value)
        delta = abs(self._last_updated_percent - v)
        self._percent = v
        if self._last_step is not None:
            self._step = round(
                0.01 * self._percent * (self._last_step - self._start_step)
                + self._start_step
            )
        if delta >= 100 / self._len or self._time_to_refresh():
            self._update()

    def _update(self):
        """Update progress bar"""
        self._last_updated_percent = self._percent
        bars = int(self._percent / 100 * self._len)
        elapsed_s = time.time() - self._start_time

        tips = "−\\/"
        tip = tips[self._tip_phase] if bars < self._len else ""
        self._tip_phase = (self._tip_phase + 1) % len(tips)

        prog_str = f"{round(self._percent):4d}% |{'█' * bars}{tip}{'-' * (self._len - bars - 1)}|"
        if self._last_step is not None:
            prog_str += f" {self._step}/{self._last_step}"

        prog_str += f" [{elapsed_s:.1f}s elapsed"
        if self._percent > 0 and self._percent <= 100:
            remaining_est_s = elapsed_s * (100 - self._percent) / self._percent
            prog_str += f", {remaining_est_s:.1f}s remaining"
        if self._last_step is not None and elapsed_s > 0:
            prog_str += f", {(self._step - self._start_step) / elapsed_s:.1f} {self._speed_units}]"
        else:
            prog_str += "]"

        class printer(str):
            def __repr__(self):
                return self

        prog_str = printer(prog_str)

        if in_notebook():
            import IPython.display

            if self._display_id is None:
                self._display_id = "dg_progress_" + str(time.time_ns())
                IPython.display.display(prog_str, display_id=self._display_id)
            else:
                IPython.display.update_display(prog_str, display_id=self._display_id)
        else:
            print(prog_str, end="\r")
        self._last_update_time = time.time()
