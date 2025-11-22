#
# ui_support.py: UI support classes and functions
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements classes and functions to handle image display, progress indication, etc.
#

"""
UI Support Module Overview
=========================

This module provides a comprehensive set of utilities for user interface operations,
including image display, text rendering, progress tracking, and performance monitoring.
It supports both traditional GUI environments and Jupyter notebooks.

Key Features:
    - **Image Display**: Show images in GUI windows or Jupyter notebooks
    - **Text Rendering**: Draw text with customizable fonts, colors, and positions
    - **Progress Tracking**: Display progress bars with speed and percentage
    - **Performance Monitoring**: Measure and display FPS
    - **Environment Detection**: Auto-detect and adapt to different display environments
    - **Color Utilities**: Convert between color spaces and compute complementary colors

Typical Usage:
    1. Use `Display` class for showing images in any environment
    2. Draw text on images with `put_text()`
    3. Track progress with `Progress` class
    4. Monitor performance with `FPSMeter`
    5. Stack images with `stack_images()`

Integration Notes:
    - Works in both GUI and Jupyter notebook environments
    - Automatically detects and adapts to the display environment
    - Supports both OpenCV and PIL image formats
    - Handles video files in Jupyter notebooks
    - Provides consistent interface across different platforms

Key Classes:
    - `Display`: Main class for showing images and videos
    - `Progress`: Progress bar with speed and percentage display
    - `FPSMeter`: Frames per second measurement
    - `Timer`: Simple timing utility
    - `stdoutRedirector`: Context manager for redirecting stdout

Configuration Options:
    - Font settings (face, scale, thickness)
    - Color schemes (RGB/BGR)
    - Progress bar appearance
    - Display window properties
"""

import cv2, sys, os, time, PIL.Image, numpy as np, random, string
from .environment import get_test_mode, in_colab, in_notebook, to_valid_filename
from .image_tools import luminance
from dataclasses import dataclass
from typing import Optional, Union, Any, List, Callable
from enum import Enum
from pathlib import Path


@dataclass
class _LineInfo:
    line: str = ""
    x: int = 0
    y: int = 0
    line_height: int = 0
    line_height_no_baseline: int = 0


def deduce_text_color(bg_color: tuple):
    """Return a readable text color.

    Chooses black or white based on the luminance of ``bg_color`` so that text
    remains legible.

    Args:
        bg_color (tuple): Background color as an ``(R, G, B)`` tuple.

    Returns:
        (Tuple[int, int, int]): ``(R, G, B)`` value for black or white text.
    """
    return (0, 0, 0) if luminance(bg_color) > 180 else (255, 255, 255)


def color_complement(color):
    """Return the complement of an RGB color.

    Args:
        color (tuple | list): Color specified as ``(R, G, B)``.

    Returns:
        (Tuple[int, int, int]): Complementary color in ``(R, G, B)`` format.
    """
    adj_color = color[0] if isinstance(color, list) else color
    return tuple([255 - c for c in adj_color])


def rgb_to_bgr(color):
    """Convert an RGB color tuple to BGR.

    Args:
        color (tuple): Color in ``(R, G, B)`` format.

    Returns:
        (Tuple[int, int, int]): Color in ``(B, R, G)`` order for OpenCV functions.
    """
    return color[::-1]


def ipython_display(obj: Any, clear: bool = False, display_id: Optional[str] = None):
    """Display an object in IPython notebooks.

    Args:
        obj (Any): Object to display. Supported types are ``PIL.Image``,
            ``numpy.ndarray`` images, or a string path/URL to an image or video.
        clear (bool, optional): Whether to clear the previous output. Defaults to ``False``.
        display_id (Optional[str], optional): Custom display ID to update an existing output.

    Raises:
        Exception: If the object type is unsupported.
    """

    import IPython.display

    if isinstance(obj, PIL.Image.Image):
        # PIL image
        IPython.display.display(obj, clear=clear, display_id=display_id)
    elif isinstance(obj, np.ndarray):
        # OpenCV image
        IPython.display.display(
            PIL.Image.fromarray(obj[..., ::-1]), clear=clear, display_id=display_id
        )
    elif isinstance(obj, str):
        # filename or URL
        is_url = obj.startswith("http")
        if obj.endswith(".mp4") or obj.endswith(".avi"):
            # video
            IPython.display.display(
                IPython.display.Video(obj, embed=in_colab() and not is_url),
                clear=clear,
                display_id=display_id,
            )
        else:
            # assume image
            IPython.display.display(
                IPython.display.Image(obj), clear=clear, display_id=display_id
            )
    else:
        raise Exception(f"ipython_display: unsupported object type {type(obj)}")


class CornerPosition(Enum):
    """Enumeration of possible corner positions for text placement.

    This enum defines the possible positions where text can be placed relative to
    a reference point in an image. The AUTO option will automatically choose the
    best corner based on the reference point's position.

    Attributes:
        AUTO (int): Automatically choose the best corner position.
        TOP_LEFT (int): Place text at the top-left corner.
        TOP_RIGHT (int): Place text at the top-right corner.
        BOTTOM_LEFT (int): Place text at the bottom-left corner.
        BOTTOM_RIGHT (int): Place text at the bottom-right corner.
    """

    AUTO = 0
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
) -> np.ndarray:
    """Draw text on an image with customizable appearance and positioning.

    This function draws text on an OpenCV image with support for multi-line text,
    background colors, and automatic positioning. The text can be placed relative
    to any corner of the image, and will automatically adjust to stay within
    image boundaries.

    Args:
        image (np.ndarray): Input image in OpenCV format (BGR).
        label (str): Text to draw. Can contain newlines for multi-line text.
        corner_xy (tuple): Base coordinates (x, y) for text placement.
        corner_position (CornerPosition, optional): Position of text relative to
            corner_xy. Defaults to TOP_LEFT.
        font_color (tuple): Text color in RGB format.
        bg_color (Optional[tuple], optional): Background color in RGB format.
            If None, no background is drawn. Defaults to None.
        font_face (int, optional): OpenCV font face. Defaults to FONT_HERSHEY_PLAIN.
        font_scale (float, optional): Font size multiplier. Defaults to 1.
        font_thickness (int, optional): Font thickness in pixels. Defaults to 1.
        line_spacing (float, optional): Multiplier for line spacing. Defaults to 1.

    Returns:
        Image with text drawn on it.
    """

    if not label:
        return image

    font_color = rgb_to_bgr(font_color)
    bg_color = rgb_to_bgr(bg_color) if bg_color is not None else None

    im_h, im_w = image.shape[:2]
    margin = 6

    top_left_xy = corner_xy
    lines: List[_LineInfo] = []
    max_width = max_height = 0

    # Helper function that measures how much width and height to use for font
    def _measure_block(lines, font_face, font_scale, font_thickness, margin, line_spacing):
        max_w = 0
        total_h = 0
        for i, line in enumerate(lines):
            (w, h_no), baseline = cv2.getTextSize(line, font_face, font_scale, font_thickness)
            line_h = h_no + baseline + margin
            max_w = max(max_w, w + margin)  # add right margin (matches your drawing)
            total_h += line_h if i == 0 else int(line_h * line_spacing)
        return max_w, total_h

    lines_text = label.splitlines()

    # 1) initial measure at requested font_scale
    req_scale = float(font_scale)
    block_w, block_h = _measure_block(
        lines_text, font_face, req_scale, font_thickness, margin=6, line_spacing=line_spacing
    )

    # 2) compute available space (whole image; change if you want a smaller box)
    avail_w, avail_h = image.shape[1], image.shape[0]

    # 3) compute fit scale (≤ 1.0 means shrink)
    def _safe_ratio(num, den):
        return num / den if den > 0 else 1.0

    scale_w = _safe_ratio(avail_w, block_w) if block_w > 0 else 1.0
    scale_h = _safe_ratio(avail_h, block_h) if block_h > 0 else 1.0
    fit_scale = min(1.0, scale_w, scale_h)

    # 4) apply (with a tiny safety margin to avoid off-by-one overflows)
    font_scale = max(0.1, req_scale * fit_scale * 0.98)

    # (optional) adjust thickness proportionally, keep ≥1
    if font_thickness > 0:
        font_thickness = max(1, int(round(font_thickness * (font_scale / max(req_scale, 1e-6)))))

    for line in lines_text:
        li = _LineInfo()
        li.line = line
        (line_width, li.line_height_no_baseline), baseline = cv2.getTextSize(
            line,
            font_face,
            font_scale,
            font_thickness,
        )
        li.x = top_left_xy[0]
        li.y = top_left_xy[1]
        li.line_height = li.line_height_no_baseline + baseline + margin
        top_left_xy = (li.x, li.y + int(li.line_height * line_spacing))
        max_width = max(max_width, line_width)
        max_height = max(max_height, li.line_height)
        lines.append(li)

    max_width += margin

    # deduce corner position if AUTO
    if corner_position == CornerPosition.AUTO:
        if corner_xy[0] < im_w / 2:
            if corner_xy[1] < im_h / 2:
                corner_position = CornerPosition.TOP_LEFT
            else:
                corner_position = CornerPosition.BOTTOM_LEFT
        else:
            if corner_xy[1] < im_h / 2:
                corner_position = CornerPosition.TOP_RIGHT
            else:
                corner_position = CornerPosition.BOTTOM_RIGHT

    # adjust coordinates according to corner_position option
    if corner_position != CornerPosition.TOP_LEFT:
        y_adjustment = (
            lines[-1].y + lines[-1].line_height - lines[0].y
            if corner_position != CornerPosition.TOP_RIGHT
            else 0
        )
        x_adjustment = max_width if corner_position != CornerPosition.BOTTOM_LEFT else 0
        for li in lines:
            li.x -= x_adjustment
            li.y -= y_adjustment

    # fit to image
    if (min_x := min(li.x for li in lines)) < 0:
        for li in lines:
            li.x -= min_x
    elif (max_x := max(li.x for li in lines) + max_width - im_w) > 0:
        for li in lines:
            li.x -= max_x
    if (min_y := min(li.y for li in lines)) < 0:
        for li in lines:
            li.y -= min_y
    elif (max_y := max(li.y for li in lines) + max_height - im_h) > 0:
        for li in lines:
            li.y -= max_y

    for li in lines:
        sz_h = int(max(0, li.line_height))
        sz_w = int(max(0, max_width))
        if bg_color is not None and sz_h > 0 and sz_w > 0:
            # Desired rectangle
            x0_req, y0_req = int(li.x), int(li.y)
            x1_req, y1_req = x0_req + sz_w, y0_req + sz_h

            # Clip to image bounds
            x0 = max(0, min(im_w, x0_req))
            y0 = max(0, min(im_h, y0_req))
            x1 = max(0, min(im_w, x1_req))
            y1 = max(0, min(im_h, y1_req))

            tw, th = x1 - x0, y1 - y0
            if tw > 0 and th > 0:
                patch = np.full((th, tw, 3), bg_color, dtype=image.dtype)
                image[y0:y1, x0:x1] = patch  # safe assignment

        # Draw text (OpenCV will clip text that overflows)
        text_x = int(li.x + margin // 2)
        text_y = int(li.y + li.line_height_no_baseline + margin // 2)  # baseline origin
        image = cv2.putText(
            image,
            li.line,
            (text_x, text_y),                 # bottom-left origin
            font_face,
            font_scale,
            font_color,
            font_thickness,
            cv2.LINE_AA,
        )

    return image


def stack_images(
    image1: Union[np.ndarray, PIL.Image.Image],
    image2: Union[np.ndarray, PIL.Image.Image],
    dimension: str = "horizontal",
    downscale: Optional[float] = None,
    labels: Optional[list] = None,
    font_color: tuple = (255, 255, 255),
) -> Union[np.ndarray, PIL.Image.Image]:
    """Stack two images either horizontally or vertically.

    Args:
        image1 (np.ndarray | PIL.Image.Image): First image.
        image2 (np.ndarray | PIL.Image.Image): Second image.
        dimension (str, optional): ``"horizontal"`` or ``"vertical"``. Defaults to
            ``"horizontal"``.
        downscale (Optional[float], optional): Scaling factor for both images if
            less than ``1.0``.
        labels (Optional[list], optional): Optional text labels for ``image1``
            and ``image2``.
        font_color (tuple, optional): RGB color for labels. Defaults to white.

    Returns:
        Combined image with optional resizing and labels.
    """
    ret_type: Callable = np.array

    if isinstance(image1, PIL.Image.Image):
        img1 = np.array(image1)
        img2 = np.array(image2)
        ret_type = PIL.Image.fromarray
    else:
        img1 = image1
        img2 = image2  # type: ignore[assignment]

    if downscale is not None:
        if downscale < 1.0:
            img1 = cv2.resize(
                img1,
                dsize=None,
                fx=downscale,
                fy=downscale,
                interpolation=cv2.INTER_AREA,
            )
            img2 = cv2.resize(
                img2,
                dsize=None,
                fx=downscale,
                fy=downscale,
                interpolation=cv2.INTER_AREA,
            )

    h, w, c = img1.shape
    h2, w2, c = img2.shape
    img_dtype = img1.dtype

    if dimension == "horizontal":
        if img1.shape[0] != img2.shape[0]:
            raise Exception("Image heights must match for horizontal stacking.")

        stacked_img = np.zeros((h, w + w2, c), dtype=img_dtype)
        stacked_img[:h, :w, :] = img1
        stacked_img[:h, w:, :] = img2
    elif dimension == "vertical":
        if img1.shape[1] != img2.shape[1]:
            raise Exception("Image widths must match for vertical stacking.")

        stacked_img = np.zeros((h + h2, w, c), dtype=img_dtype)
        stacked_img[:h, :w, :] = img1
        stacked_img[h:, :w, :] = img2
    else:
        raise Exception("Unsupported image stacking dimension.")

    if isinstance(labels, list) and isinstance(labels[0], str):
        if len(labels) != 2:
            raise Exception("Must have two labels for stacked images.")

        if dimension == "horizontal":
            stacked_img = put_text(
                stacked_img,
                labels[0],
                (0, 0),
                font_color=font_color,
                corner_position=CornerPosition.BOTTOM_LEFT,
            )
            stacked_img = put_text(
                stacked_img,
                labels[1],
                (w, 0),
                font_color=font_color,
                corner_position=CornerPosition.BOTTOM_LEFT,
            )
        else:
            stacked_img = put_text(
                stacked_img,
                labels[0],
                (0, h),
                font_color=font_color,
                corner_position=CornerPosition.BOTTOM_LEFT,
            )
            stacked_img = put_text(
                stacked_img,
                labels[1],
                (0, 0),
                font_color=font_color,
                corner_position=CornerPosition.BOTTOM_LEFT,
            )

    return ret_type(stacked_img)


class FPSMeter:
    """Frame rate measurement utility.

    This class provides functionality to measure and track frames per second (FPS)
    over a configurable window of time. It's useful for monitoring performance
    in video processing and real-time applications.

    Attributes:
        avg_len (int): Number of samples to use for FPS calculation.
    """

    def __init__(self, avg_len: int = 100):
        """Constructor.

        Args:
            avg_len (int): Number of samples to use for FPS calculation. Defaults to 100.
        """
        self._avg_len = avg_len
        self.reset()

    def reset(self):
        """Reset accumulators."""
        self._timestamp_ns = -1
        self._duration_ns = -1
        self._count = 0

    def record(self) -> float:
        """Record timestamp and update average duration.

        Returns current average FPS."""
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
        """Return current average FPS."""
        return 1e9 / self._duration_ns if self._duration_ns > 0 else 0


class Display:
    """Display manager for showing images and videos in various environments.

    This class provides a unified interface for displaying images and videos in
    both GUI windows and Jupyter notebooks. It automatically detects the display
    environment and adapts its behavior accordingly.

    Attributes:
        window_name (str): Name of the display window in GUI mode.
        show_fps (bool): Whether to show FPS counter on displayed images.
        width (Optional[int]): Target width for displayed images.
        height (Optional[int]): Target height for displayed images.
    """

    def __init__(
        self,
        capt: str = "<image>",
        show_fps: bool = True,
        w: Optional[int] = None,
        h: Optional[int] = None,
    ):
        """Constructor.

        Args:
            capt (str): Window title. Defaults to "<image>".
            show_fps (bool): Whether to show FPS counter. Defaults to True.
            w (Optional[int]): Initial window width in pixels; None for autoscale. Defaults to None.
            h (Optional[int]): Initial window height in pixels; None for autoscale. Defaults to None.

        Raises:
            Exception: If window title is empty.
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
        self._display_id: Optional[str] = None

    def _update_notebook_display(self, obj: Any):
        """Update notebook display with given object.

        Args:
            obj (Any): Object to display in the notebook.
        """

        import IPython.display

        if self._display_id is None:
            self._display_id = "dg_show_" + "".join(random.choices(string.digits, k=10))
            IPython.display.display(obj, display_id=self._display_id)
        else:
            IPython.display.update_display(obj, display_id=self._display_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # close OpenCV window in any
        if self._window_created:
            try:
                cv2.destroyWindow(self._capt)
            except Exception:
                pass

        # close video writer if any, and show video in Colab
        if self._video_writer is not None:
            self._video_writer.release()
            if in_colab():
                import IPython.display

                self._update_notebook_display(
                    IPython.display.Video(self._video_file, embed=True)
                )

        return exc_type is KeyboardInterrupt  # ignore KeyboardInterrupt errors

    @property
    def window_name(self) -> str:
        """Get the window name.

        Returns:
            Name of the display window.
        """
        return self._capt

    @staticmethod
    def _check_gui() -> bool:
        """Check if graphical display is supported.

        Returns:
            True if graphical display is supported, False otherwise.
        """
        import platform

        if platform.system() == "Linux":
            return os.environ.get("DISPLAY") is not None
        return True

    @staticmethod
    def _display_fps(img: np.ndarray, fps: float):
        """Helper method to display FPS."""
        put_text(
            img,
            f"{fps:5.1f} FPS",
            (0, 0),
            font_color=(0, 0, 0),
            bg_color=(255, 255, 255),
        )

    def show(self, img: Any, waitkey_delay: int = 1):
        """Show image or model result.

        Args:
            img (Any): Image to display. Can be a numpy array with valid OpenCV image,
                PIL image, or model result object.
            waitkey_delay (int): Delay in ms for waitKey() call. Use 0 to show still images,
                use 1 for streaming video. Defaults to 1.
        """

        # show image in notebook
        def show_in_notebook(img):
            self._update_notebook_display(PIL.Image.fromarray(img[..., ::-1]))

        # show image in Colab
        def show_in_colab(img):
            if waitkey_delay == 0:
                # show still image in notebook
                show_in_notebook(img)
            else:
                # save videos to file
                from .video_support import create_video_writer

                if self._video_writer is None:
                    self._video_file = (
                        f"{os.getcwd()}/{Path(to_valid_filename(self._capt)).stem}.mp4"
                    )
                    self._video_writer = create_video_writer(
                        self._video_file, img.shape[1], img.shape[0]
                    )
                self._video_writer.write(img)

                class printer(str):
                    def __repr__(self):
                        return self

                if self._video_writer.count % 10 == 0:
                    self._update_notebook_display(
                        printer(
                            f"{self._video_file}: frame {self._video_writer.count}, {fps:.1f} FPS"
                        )
                    )

        def preprocess_img(img):
            if "image_overlay" in vars(type(img)):
                # special case for model results: use image_overlay property
                img = img.image_overlay
            if isinstance(img, PIL.Image.Image):
                # PIL image: convert to OpenCV format
                img = np.array(img)[:, :, ::-1]
            if not isinstance(img, np.ndarray):
                raise Exception(f"Display: unsupported image type {type(img)}")
            return img

        orig_img = img
        img = preprocess_img(orig_img)

        fps = self._fps.record()
        if self._show_fps and fps > 0:
            Display._display_fps(img, fps)

        if in_colab():
            # special case for Colab environment
            show_in_colab(img)
        elif self._no_gui:
            if in_notebook():
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
            # process pause
            if key == ord(" "):
                while True:
                    img = preprocess_img(orig_img)
                    cv2.imshow(self._capt, img)
                    key = cv2.waitKey(waitkey_delay) & 0xFF
                    if key == ord(" ") or key == ord("x") or key == ord("q"):
                        break
            # process exit keys
            if key == ord("x") or key == ord("q"):
                if self._fps:
                    self._fps.reset()
                raise KeyboardInterrupt
            # process resize keys
            elif key == 43 or key == 45:  # +/-
                _, _, w, _ = cv2.getWindowImageRect(self._capt)
                factor = 1.25 if key == 43 else 0.75
                new_w = max(100, int(w * factor))
                new_h = int(new_w * img.shape[0] / img.shape[1])
                cv2.resizeWindow(self._capt, new_w, new_h)

    def show_image(self, img: Any):
        """Show still image or model result.

        Args:
            img (Any): Image to display. Can be a numpy array with valid OpenCV image,
                PIL image, or model result object.
        """
        self.show(img, 0)


class Timer:
    """Simple timer class."""

    def __init__(self):
        """Constructor. Records start time."""
        self._start_time = time.time_ns()

    def __call__(self) -> float:
        """Get elapsed time since timer creation.

        Returns:
            Time elapsed in seconds since object construction.
        """
        return (time.time_ns() - self._start_time) * 1e-9


class Progress:
    """Progress bar with speed and percentage display.

    This class provides a progress bar that shows completion percentage, speed,
    and optional messages. It works in both GUI and Jupyter notebook environments.

    Attributes:
        last_step (Optional[int]): Total number of steps (None for indeterminate).
        start_step (int): Starting step number.
        bar_len (int): Length of the progress bar in characters.
        speed_units (str): Units to display for speed (e.g., "FPS", "items/s").
    """

    utf8_supported = sys.stdout.encoding.lower() == "utf-8"

    def __init__(
        self,
        last_step: Optional[int] = None,
        *,
        start_step: int = 0,
        bar_len: int = 15,
        speed_units: str = "FPS",
    ):
        """Constructor.

        Args:
            last_step (Optional[int]): Total number of steps (None for indeterminate). Defaults to None.
            start_step (int): Starting step number. Defaults to 0.
            bar_len (int): Progress bar length in symbols. Defaults to 15.
            speed_units (str): Units to display for speed (e.g., "FPS", "items/s"). Defaults to "FPS".
        """
        self._display_id: Optional[str] = None
        self._len = bar_len
        self._last_step = last_step
        self._start_step = start_step
        self._time_to_refresh = lambda: time.time() - self._last_update_time > 0.5
        self._speed_units = speed_units
        self._message = ""
        self.reset()

    def reset(self):
        """Reset the progress bar to its initial state.

        This method resets the current step, message, and timing information.
        """
        self._start_time = time.time()
        self._step = self._start_step
        self._percent = 0.0
        self._last_updated_percent = self._percent
        self._last_update_time = 0.0
        self._tip_phase = 0
        self._longest_line = 0
        self._update()

    def step(self, steps: int = 1, *, message: Optional[str] = None):
        """Update progress by given number of steps.

        Args:
            steps: Number of steps to advance.
            message: Optional message to display.
        """
        assert (
            self._last_step is not None
        ), "Progress indicator: to do stepping last step must be assigned on construction"
        assert (
            self._last_step > self._start_step
        ), f"Progress indicator: last step {self._last_step} must be greater than start step {self._start_step}"

        self._step += steps
        self._percent = (
            100 * (self._step - self._start_step) / (self._last_step - self._start_step)
        )
        if message is not None:
            self._message = message
        if (
            self._percent - self._last_updated_percent >= 100 / self._len
            or self._percent >= 100
            or self._time_to_refresh()
        ):
            self._update()

    @property
    def step_range(self) -> Optional[tuple]:
        """Get start-end step range (if defined)."""
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

    @property
    def message(self) -> str:
        return self._message

    @message.setter
    def message(self, value: str):
        self._message = value
        self._update()

    def _update(self):
        """Update progress bar."""
        self._last_updated_percent = self._percent
        bars = int(self._percent / 100 * self._len)
        elapsed_s = time.time() - self._start_time

        if Progress.utf8_supported:
            tips = "−\\/"
            bar_char = "█"
        else:
            tips = "-\\/"
            bar_char = "#"

        tip = tips[self._tip_phase] if bars < self._len else ""
        self._tip_phase = (self._tip_phase + 1) % len(tips)

        prog_str = f"{round(self._percent):4d}% |{bar_char * bars}{tip}{'-' * (self._len - bars - 1)}|"
        if self._last_step is not None:
            prog_str += f" {self._step}/{self._last_step}"

        prog_str += f" [{elapsed_s:.1f}s elapsed"
        if self._percent > 0 and self._percent <= 100:
            remaining_est_s = elapsed_s * (100 - self._percent) / self._percent
            prog_str += f", {remaining_est_s:.1f}s remaining"
        if self._last_step is not None and elapsed_s > 0:
            prog_str += f", {(self._step - self._start_step) / elapsed_s:.1f} {self._speed_units}] {self._message}"
        else:
            prog_str += "]"

        if in_notebook():

            class printer(str):
                def __repr__(self):
                    return self

            prog_str = printer(prog_str)

            import IPython.display

            if self._display_id is None:
                self._display_id = "dg_progress_" + "".join(
                    random.choices(string.digits, k=10)
                )
                IPython.display.display(prog_str, display_id=self._display_id)
            else:
                IPython.display.update_display(prog_str, display_id=self._display_id)
        else:
            if len(prog_str) < self._longest_line:
                prog_str += " " * (self._longest_line - len(prog_str))
            else:
                self._longest_line = len(prog_str)

            print(prog_str, end="\r")

        self._last_update_time = time.time()


class stdoutRedirector:
    """Redirect stdout to another stream."""

    def __init__(self, stream: Optional[str] = None):
        """Constructor.

        Args:
            stream (Optional[str]): Output stream to redirect to; None to redirect to null device. Defaults to None.
        """
        self._stdout = sys.stdout
        self._stream = stream

    def __enter__(self):
        sys.stdout = open(os.devnull if self._stream is None else self._stream, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout
