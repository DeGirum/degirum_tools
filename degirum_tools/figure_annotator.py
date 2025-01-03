#
# figure_annotator.py: geometric figure annotation command-line utility
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements classes for the geometric figure annotation command-line utility
# and the driver for this utility.
#

from PIL import Image
import json
import argparse
import cv2
import numpy as np
from enum import Enum
from typing import Tuple, List, Dict, Union, Optional
from copy import deepcopy
from pathlib import Path
from . import environment as env


help_message_line = """
    The DeGirum Line Annotator is a command-line utility for interactive
    annotation of an image. The user can draw lines on an image and save
    the line data to a JSON file.

    Open an image file

        To open an image file, click File > Open... or press Control-O.

    Annotate

        The user left-clicks the endpoints of a line segment. Once a line is complete,
        the user can modify the most recently added line or draw a new line.

    Undo

        To remove the most-recently clicked point, click Edit > Undo or press Control-Z.

    Save

        To save the current results, click File > Save or press Control-S. To save the
        current results to a specific file, click File > Save As... or press Control-Shift-S.

    Additional Controls

        Right-click  -  if cursor is next to a point of a line, the point is grabbed and can be
                        dragged to a new location. Right-click again to release the point at the
                        current cursor location.

        Control-right-click  -  if cursor position is next to a line, the line is grabbed and
                                can be dragged to a new location. Control-right-click again to
                                release the line at the current cursor location.

        Escape key  -  when drawing a line and line is incomplete, removes the remnants of the line;
                       when dragging a point, returns the point to original position and cancels drag;
                       when dragging a line, returns the line to original position and cancels drag

    Key Shortcuts Summary

        Control-O        -  open an image file
        Control-S        -  save changes
        Control-Shift-S  -  save line data to a JSON file
        Control-Z        -  undo last selected point
    """

help_message_grid = """
    The DeGirum Zone Annotator is a command-line utility for interactive
    annotation of an image. The user can draw zones on an image and save
    the zone data to a JSON file.

    Open an image file

        To open an image file, click File > Open... or press Control-O.

    Annotate

        There are two modes of annotation: non-grid and grid modes.

        In non-grid mode, the user left-clicks the vertices of a zone,
        clockwise or counter-clockwise from each other, to create a convex
        polygon. Once a zone is complete, the user can modify the most recently
        added zone or draw a new zone.

        In grid mode, the user can create a 1-by-N grid with N adjacent
        zones. Starting from the top-left corner of the grid and going
        counter-clockwise, the user left-clicks to select 4 corner points of the grid.
        At this point, this grid is minimally-complete and the user can save it as a
        zone. However, the user can continue to select points, point on the top edge of
        the grid followed by a point on the bottom edge of the grid, to define dividing
        segments that divide the main grid into zones. The user can add as many grids
        as needed, and can select the active grid to edit from the drop-down menu
        labeled "Active Grid" (see below).

    Undo

        To remove the most-recently clicked point, click Edit > Undo or press Control-Z.
        If in grid mode and no points are selected for this grid, this action deletes
        the grid.

    Modes

        Right below the main menu bar is a drop-down menu labeled "Active Grid".
        Initially, this drop-down menu has only one option, "Non-grid mode", meaning that
        the utility is in non-grid mode. Whenever non-grid mode is necessary, select this
        option from the drop-down menu. When grids are added (see below), their names appear
        in the menu. In order to edit a specific grid, e.g. "Grid 1", select "Grid 1" from
        the menu.

    Add grid

        To add a grid, click Edit > Add Grid or press Control-A. A new grid will be created
        and added to the "Active Grid" menu.

    Remove grid

        To remove a grid, select it from the "Active Grid" menu and click Edit > Remove Grid
        or press Control-D. The grid will be removed from the canvas and from the menu.

    Save

        To save the current results, click File > Save or press Control-S. To save the
        current results to a specific file, click File > Save As... or press Control-Shift-S.

    Additional Controls

        Right-click  -  if cursor is next to a point of a zone (if in non-grid mode) or of a grid
                        (if that grid is selected), the point is grabbed and can be dragged to a
                        new location. Right-click again to release the point at the current cursor
                        location.

        Control-right-click  -  if cursor position is enclosed by a zone (if in non-grid mode)
                                or a grid (if that grid is selected), the zone/grid is grabbed and
                                can be dragged to a new location. Control-right-click again to
                                release the zone/grid at the current cursor location.

        Escape key  -  when drawing a zone and zone is incomplete, removes the remnants of the zone;
                        when dragging a point, returns the point to original position and cancels drag;
                        when dragging a zone/grid, returns the zone/grid to original position and
                        cancels drag

    Key Shortcuts Summary

        Control-O        -  open an image file
        Control-S        -  save changes
        Control-Shift-S  -  save zone data to a JSON file
        Control-A        -  add grid
        Control-D        -  remove grid
        Control-Z        -  undo last selected point; if in grid mode and grid is empty, deletes grid
    """

help_message_polygon = """
    The DeGirum Zone Annotator is a command-line utility for interactive
    annotation of an image. The user can draw zones on an image and save
    the zone data to a JSON file.

    Open an image file

        To open an image file, click File > Open... or press Control-O.

    Annotate

        The user left-clicks the vertices of a zone, clockwise or counter-clockwise
        from each other, to create a convex polygon. Once a zone is complete, the user
        can modify the most recently added zone or draw a new zone.

    Undo

        To remove the most-recently clicked point, click Edit > Undo or press Control-Z.

    Save

        To save the current results, click File > Save or press Control-S. To save the
        current results to a specific file, click File > Save As... or press Control-Shift-S.

    Additional Controls

        Right-click  -  if cursor is next to a point of a zone, the point is grabbed and can be
                        dragged to a new location. Right-click again to release the point at the
                        current cursor location.

        Control-right-click  -  if cursor position is enclosed by a zone, the zone is grabbed and
                                can be dragged to a new location. Control-right-click again to
                                release the zone at the current cursor location.

        Escape key  -  when drawing a zone and zone is incomplete, removes the remnants of the zone;
                       when dragging a point, returns the point to original position and cancels drag;
                       when dragging a zone, returns the zone to original position and cancels drag

    Key Shortcuts Summary

        Control-O        -  open an image file
        Control-S        -  save changes
        Control-Shift-S  -  save zone data to a JSON file
        Control-Z        -  undo last selected point
    """


def _figure_annotator_run(args):
    """
    Launch interactive utility for geometric figure annotation in images.

    Args:
        args: argparse command line arguments
    """
    if args.num_vertices < 2:
        raise ValueError("Number of vertices must be at least 2")

    FigureAnnotator(
        num_vertices=args.num_vertices,
        image_path=args.img_path,
        results_file_name=args.save_path,
    )


def _zone_annotator_run(args):
    """
    Launch interactive utility for 4-sided zone annotation in images.

    Args:
        args: argparse command line arguments
    """
    FigureAnnotator(image_path=args.img_path, results_file_name=args.save_path)


def _line_annotator_run(args):
    """
    Launch interactive utility for line annotation in images.

    Args:
        args: argparse command line arguments
    """
    FigureAnnotator(
        num_vertices=FigureAnnotatorType.LINE.value,
        image_path=args.img_path,
        results_file_name=args.save_path,
    )


def _figure_annotator_args(parser):
    """
    Define figure_annotator subcommand arguments

    Args:
        parser: argparse parser object to be stuffed with args
    """
    parser.add_argument(
        "img_path",
        nargs="?",
        type=str,
        default="",
        help="path to image file for annotation",
    )
    parser.add_argument(
        "--num-vertices",
        type=int,
        default=FigureAnnotatorType.QUADRILATERAL.value,
        help="number of vertices in annotation figures",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="zones.json",
        help="JSON file path to save figures",
    )
    parser.set_defaults(func=_figure_annotator_run)


def _zone_annotator_args(parser):
    """
    Define zone_annotator subcommand arguments

    Args:
        parser: argparse parser object to be stuffed with args
    """
    parser.add_argument(
        "img_path",
        nargs="?",
        type=str,
        default="",
        help="path to image file for annotation",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="zones.json",
        help="JSON file path to save zones",
    )
    parser.set_defaults(func=_zone_annotator_run)


def _line_annotator_args(parser):
    """
    Define line_annotator subcommand arguments

    Args:
        parser: argparse parser object to be stuffed with args
    """
    parser.add_argument(
        "img_path",
        nargs="?",
        type=str,
        default="",
        help="path to image file for annotation",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="lines.json",
        help="JSON file path to save lines",
    )
    parser.set_defaults(func=_line_annotator_run)


class FigureAnnotatorType(Enum):
    """Figure annotator types"""

    LINE = 2
    QUADRILATERAL = 4


object_format_versions = {
    "line": 1,
    "zone": 1,
}


class Grid:
    def __init__(self, grid_id: str):
        self.grid_id = grid_id
        self.ids: List[int] = []
        self.points: List[Tuple[float, float]] = []
        self.displayed_points: List[Tuple[int, int]] = []
        self.top_m = 0.0
        self.top_b = 0.0
        self.bottom_m = 0.0
        self.bottom_b = 0.0

    def complete(self) -> bool:
        """
        Returns True if grid is complete, i. e.
        at least four points are defined and number of points is even.
        """
        points_len = len(self.points)
        return points_len >= 4 and points_len % 2 == 0

    @staticmethod
    def lin_func(m, b, x):
        """Return y-coordinate for a given x-coordinate based on line defined by slope and y-intercept."""
        return m * x + b

    def process_point_addition(self, point):
        """
        Add new point to grid.
        """
        points_len = len(self.points)
        if points_len >= 4:  # Scale intermediate point to grid
            self.points.insert(-2, self.scale_point(point, points_len))
        else:
            self.points.append(point)

        if len(self.points) == 4:
            self.update_grid_parameters()

    def scale_point(self, point, idx):
        if self.mostly_horizontal():
            scaled_x = point[0]
            if idx % 2 == 0:
                lims = sorted([self.points[0][0], self.points[-1][0]])
                scaled_x = np.clip(scaled_x, lims[0], lims[1])
                scaled_y = Grid.lin_func(self.top_m, self.top_b, scaled_x)
            else:
                lims = sorted([self.points[1][0], self.points[-2][0]])
                scaled_x = np.clip(scaled_x, lims[0], lims[1])
                scaled_y = Grid.lin_func(self.bottom_m, self.bottom_b, scaled_x)
        else:
            scaled_y = point[1]
            if idx % 2 == 0:
                lims = sorted([self.points[0][1], self.points[-1][1]])
                scaled_y = np.clip(scaled_y, lims[0], lims[1])
                scaled_x = Grid.lin_func(self.top_m, self.top_b, scaled_y)
            else:
                lims = sorted([self.points[1][1], self.points[-2][1]])
                scaled_y = np.clip(scaled_y, lims[0], lims[1])
                scaled_x = Grid.lin_func(self.bottom_m, self.bottom_b, scaled_y)
        return (scaled_x, scaled_y)

    def mostly_horizontal(self):
        if len(self.points) >= 4:
            guide_top_r = self.points[-1]
            guide_top_l = self.points[0]
            return abs(guide_top_r[1] - guide_top_l[1]) < abs(
                guide_top_r[0] - guide_top_l[0]
            )
        return True

    def update_grid_parameters(self):
        """
        Calculates key defining parameters of grid:
        the slopes and y-intercepts of the top and bottom lines of the grid.
        """
        if len(self.points) >= 4:
            guide_top_r = self.points[-1]
            guide_top_l = self.points[0]
            guide_bot_r = self.points[-2]
            guide_bot_l = self.points[1]

            if self.mostly_horizontal():
                self.top_m = (guide_top_r[1] - guide_top_l[1]) / (
                    guide_top_r[0] - guide_top_l[0]
                )
                self.top_b = guide_top_r[1] - self.top_m * guide_top_r[0]
                self.bottom_m = (guide_bot_r[1] - guide_bot_l[1]) / (
                    guide_bot_r[0] - guide_bot_l[0]
                )
                self.bottom_b = guide_bot_r[1] - self.bottom_m * guide_bot_r[0]
            else:
                self.top_m = (guide_top_r[0] - guide_top_l[0]) / (
                    guide_top_r[1] - guide_top_l[1]
                )
                self.top_b = guide_top_r[0] - self.top_m * guide_top_r[1]
                self.bottom_m = (guide_bot_r[0] - guide_bot_l[0]) / (
                    guide_bot_r[1] - guide_bot_l[1]
                )
                self.bottom_b = guide_bot_r[0] - self.bottom_m * guide_bot_r[1]

    def update_displayed_points(
        self, current_width, current_height, original_width, original_height
    ):
        """Scale the original points to the current image size."""
        self.displayed_points.clear()
        for x, y in self.points:
            self.displayed_points.append(
                (
                    int(x * current_width / original_width),
                    int(y * current_height / original_height),
                )
            )

    def get_temp_polygon(self) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        Returns part of latest incomplete polygon if such exists and indices of points comprising
        this part, empty lists otherwise.
        """
        points_len = len(self.displayed_points)
        if points_len < 4:
            return self.displayed_points, list(range(points_len))
        elif points_len % 2 == 1:
            return self.displayed_points[-3:-2], [points_len - 3]
        else:
            return [], []

    def get_grid_polygons(self, display: bool = True) -> List[List[Tuple[int]]]:
        """Returns list of polygons defined by grid."""
        points: Union[List[Tuple[float, float]], List[Tuple[int, int]]] = []
        if display:
            points = self.displayed_points
        else:
            points = self.points
        polygons = []
        if len(points) >= 4:
            polygons.append(
                np.array([points[0], points[1], points[-2], points[-1]])
                .astype(int)
                .tolist()
            )
        if len(points) > 5:
            for i in range(0, len(points) - 2, 2):
                if len(points[i:]) == 5:
                    break
                if i == len(points) - 4:
                    polygons.append(
                        np.array(
                            [points[i], points[i + 1], points[i + 2], points[i + 3]]
                        )
                        .astype(int)
                        .tolist()
                    )
                else:
                    polygons.append(
                        np.array(
                            [points[i], points[i + 1], points[i + 3], points[i + 2]]
                        )
                        .astype(int)
                        .tolist()
                    )
        return polygons


class FigureAnnotator:
    def __init__(
        self,
        num_vertices: int = FigureAnnotatorType.QUADRILATERAL.value,
        image_path: str = "",
        results_file_name: str = "zones.json",
        test_mode: bool = False,
    ):
        self.num_vertices = num_vertices  # Number of vertices for each polygon
        self.figure_type: str = (
            "line" if num_vertices == FigureAnnotatorType.LINE.value else "zone"
        )
        self.with_grid: bool = (
            self.num_vertices == FigureAnnotatorType.QUADRILATERAL.value
        )
        self.image_path: str = image_path
        self.results_file_name = results_file_name
        self.save_path = ""

        # Constants
        self.line_width = 1
        self.darker_theme_color = "#89CFF0"
        self.lighter_theme_color = "lightblue"

        if not test_mode:
            self.tk = env.import_optional_package(
                "tkinter",
                custom_message="'tkinter' is not available. Hint: install tkinter with "
                + "'sudo apt install python3-tk' (Linux) or 'brew install tcl-tk' (macOS)",
            )
            self.tkFont = env.import_optional_package(
                "tkinter.font",
                custom_message="'tkinter' is not available. Hint: install tkinter with "
                + "'sudo apt install python3-tk' (Linux) or 'brew install tcl-tk' (macOS)",
            )
            self.ttk = env.import_optional_package(
                "tkinter.ttk",
                custom_message="'tkinter' is not available. Hint: install tkinter with "
                + "'sudo apt install python3-tk' (Linux) or 'brew install tcl-tk' (macOS)",
            )
            self.tkFiledialog = env.import_optional_package(
                "tkinter.filedialog",
                custom_message="'tkinter' is not available. Hint: install tkinter with "
                + "'sudo apt install python3-tk' (Linux) or 'brew install tcl-tk' (macOS)",
            )
            self.imageTk = env.import_optional_package(
                "PIL.ImageTk",
                custom_message="'tkinter' is not available. Hint: install tkinter with "
                + "'sudo apt install python3-tk' (Linux) or 'brew install tcl-tk' (macOS)",
            )
            self.root = self.tk.Tk()
            self.root.title(f"{self.figure_type.capitalize()} Annotator")
            self.root.geometry("700x410" if self.with_grid else "700x375")
            self.root.resizable(False, False)

            # Override the close button event
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)

            # Set window icon
            self.icon_image = self.imageTk.PhotoImage(
                file=str(Path(__file__).parent) + "/assets/logo.ico"
            )
            self.root.iconphoto(True, self.icon_image)  # type: ignore

            # Set font
            self.font = self.tkFont.Font(family="courier 10 pitch", size=12)

            # Create the main frame to hold the menu and canvas
            self.main_frame = self.tk.Frame(self.root, bg=self.lighter_theme_color)
            self.main_frame.pack(fill=self.tk.BOTH, expand=True)

            self.menu_bar = self.tk.Menu(self.root, bg=self.darker_theme_color)
            self.root.config(menu=self.menu_bar)

            self.file_menu = self.tk.Menu(self.menu_bar, tearoff=0)
            self.file_menu.add_command(
                label="Open Image...",
                font=self.font,
                command=self.open_image,
                accelerator="Ctrl-O",
            )
            self.file_menu.add_command(
                label="Save JSON",
                font=self.font,
                command=self.save,
                accelerator="Ctrl-S",
                state=self.tk.DISABLED,
            )
            self.file_menu.add_command(
                label="Save JSON As...",
                font=self.font,
                command=self.save_to_file,
                accelerator="Ctrl-Shift-S",
                state=self.tk.DISABLED,
            )
            self.menu_bar.add_cascade(label="File", font=self.font, menu=self.file_menu)

            self.edit_menu = self.tk.Menu(self.menu_bar, tearoff=0)
            if self.with_grid:
                self.edit_menu.add_command(
                    label="Add Grid",
                    font=self.font,
                    command=self.add_grid,
                    accelerator="Ctrl-A",
                    state=self.tk.DISABLED,
                )
                self.edit_menu.add_command(
                    label="Remove Grid",
                    font=self.font,
                    command=self.remove_grid,
                    accelerator="Ctrl-D",
                    state=self.tk.DISABLED,
                )
            self.edit_menu.add_command(
                label="Undo",
                font=self.font,
                command=self.undo,
                accelerator="Ctrl-Z",
                state=self.tk.DISABLED,
            )
            self.menu_bar.add_cascade(label="Edit", font=self.font, menu=self.edit_menu)

            self.help_menu = self.tk.Menu(self.menu_bar, tearoff=0)
            self.help_menu.add_command(
                label="Help", font=self.font, command=self.show_help
            )
            self.menu_bar.add_cascade(label="Help", font=self.font, menu=self.help_menu)

            if self.with_grid:
                # Create a frame for the second menu and "Current Selection" OptionMenu
                self.grid_selection_frame = self.tk.Frame(
                    self.main_frame, bg=self.lighter_theme_color
                )
                self.grid_selection_frame.pack(fill=self.tk.X, pady=5)

                # Add "Active Grid" ComboBox to the grid_selection_frame
                self.grid_selection_menu_label = self.tk.Label(
                    self.grid_selection_frame,
                    text="Active Grid",
                    font=self.font,
                    bg=self.lighter_theme_color,
                )
                self.grid_selection_menu_label.grid(row=0, column=0)
                self.grid_selection_default_value = "Non-grid mode"
                self.added_grid_id = ""
                self.grid_selection_var = self.tk.StringVar(self.grid_selection_frame)
                self.grid_selection_var.set(
                    self.grid_selection_default_value
                )  # Default value
                self.grid_selection_options = [self.grid_selection_default_value]

                self.grid_selection_menu = self.ttk.Combobox(
                    self.grid_selection_frame,
                    textvariable=self.grid_selection_var,
                    values=self.grid_selection_options,
                    font=self.font,
                    state="readonly",
                )
                self.grid_selection_menu.grid(row=0, column=1, padx=10)

            self.open_image_frame = None
            if not self.image_path:
                self.open_image_frame = self.tk.Frame(self.main_frame)
                self.open_image_frame.pack(fill=self.tk.NONE, padx=150, pady=150)
                self.open_button = self.tk.Button(
                    self.open_image_frame,
                    text="Open Image",
                    command=self.open_image,
                    font=self.tk.font.Font(family="courier 10 pitch", size=28),
                    bg=self.darker_theme_color,
                    fg="black",
                )
                self.open_button.grid(row=0, column=3)

            self.canvas = self.tk.Canvas(self.main_frame, cursor="cross")
            self.canvas.pack(fill=self.tk.BOTH, expand=True)

        from PIL import ImageTk

        self.original_image: Optional[Image.Image] = None  # Store the original image
        self.image_tk: Optional[ImageTk.PhotoImage] = None
        self.points: List[Tuple] = []  # Points relative to the original image
        self.displayed_points: List[Tuple] = []  # Points relative to the resized image
        self.polygon_ids: List[int] = []  # Store polygon IDs for undo
        self.saved_points: List = []

        if self.with_grid:
            self.grids: Dict[int, Grid] = {}  # Grids
            self.saved_grids: Dict[int, Grid] = {}

        self.original_width = 0  # Store original image width
        self.original_height = 0  # Store original image height
        self.current_width = 0  # Current width of the displayed image
        self.current_height = 0  # Current height of the displayed image
        self.aspect_ratio = 1.0  # Aspect ratio of the original image

        # Dragging state
        self.dragging_polygon_id: int = -1  # The polygon ID being dragged
        self.dragging_polygon_offset: Tuple[int, int] = (
            0,
            0,
        )  # The offset from the click position to the vertices
        self.dragging_polygon_offset_start: Tuple[int, int] = (
            0,
            0,
        )  # The starting offset from the click position to the vertices
        self.dragging_polygon_point_index: int = (
            -1
        )  # The index of the points in `self.points` being dragged
        self.dragging_point_start = (
            None  # The starting coordinates of the dragged point
        )
        self.dragging_point_idx: int = -1  # The index of the point being dragged

        if not test_mode:
            # Load image if path is provided
            if self.image_path:
                self.load_image()

            # Bind mouse-controlled actions
            self.canvas.bind("<Button-1>", self.on_click)
            self.canvas.bind("<Motion>", self.on_motion)
            self.canvas.bind(
                "<Button-3>", self.on_right_click
            )  # Right-click to initiate drag or move point
            self.canvas.bind(
                "<Control-Button-3>", self.on_right_click_and_ctrl
            )  # Right-click to initiate drag or move point
            self.canvas.bind("<Configure>", self.on_resize)  # Bind to window resize

            # Bind keyboard shortcuts
            self.root.bind_all("<Control-o>", self.open_image)
            self.root.bind_all("<Control-a>", self.add_grid)
            self.root.bind_all("<Control-d>", self.remove_grid)
            self.root.bind_all("<Control-s>", self.save)
            self.root.bind_all("<Control-S>", self.save_to_file)
            self.root.bind_all("<Control-z>", self.undo)
            self.root.bind_all("<Escape>", self.process_esc)

            # Run application
            self.root.mainloop()

    def open_image(self, event=None):
        """Opens image."""
        self.image_path = self.tkFiledialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if self.image_path:
            self.load_image()

    def load_image(self):
        """Loads image."""
        # Remove "Open Image" button if necessary, and make window resizable
        if self.open_image_frame:
            self.open_image_frame.destroy()
            del self.open_button
            self.open_image_frame = None

        self.root.resizable(True, True)

        # Clear the canvas before loading a new image
        self.canvas.delete("all")
        self.points.clear()
        self.displayed_points.clear()
        self.polygon_ids.clear()
        if self.with_grid:
            self.grids.clear()

        # Load the original image and store it
        self.original_image = Image.open(self.image_path)
        self.original_width, self.original_height = (
            self.original_image.size
        )  # Store original dimensions
        self.aspect_ratio = (
            self.original_width / self.original_height
        )  # Calculate aspect ratio

        # Update the window size to fit the original image dimensions
        self.root.geometry(f"{self.original_width}x{self.original_height}")

        self.update_image(self.original_image)
        self.file_menu.entryconfig("Save JSON", state=self.tk.NORMAL)
        self.file_menu.entryconfig("Save JSON As...", state=self.tk.NORMAL)
        if self.with_grid:
            self.edit_menu.entryconfig("Add Grid", state=self.tk.NORMAL)

    def update_image(self, image):
        """Helper function to update the image on the canvas."""
        self.current_width, self.current_height = image.size
        self.image_tk = self.imageTk.PhotoImage(image)
        self.canvas.config(width=self.current_width, height=self.current_height)
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.image_tk)

        # After updating the image, redraw the points and lines with scaled coordinates
        self.update_displayed_points()
        if self.with_grid:
            for grid in self.grids.values():
                grid.update_displayed_points(
                    self.current_width,
                    self.current_height,
                    self.original_width,
                    self.original_height,
                )
        self.redraw_polygons()

    def update_displayed_points(self):
        """Scale the original points to the current image size."""
        self.displayed_points = [
            (
                int(x * self.current_width / self.original_width),
                int(y * self.current_height / self.original_height),
            )
            for x, y in self.points
        ]

    def update_grid_menu(self):
        """Update the dropdown menu listing all of the present grids."""
        self.grid_selection_options = [self.grid_selection_options[0]] + [
            self.grid_idx_to_key(v) for v in sorted(self.grids.keys())
        ]
        self.grid_selection_menu["values"] = self.grid_selection_options
        if self.added_grid_id:
            self.grid_selection_var.set(self.added_grid_id)
            self.added_grid_id = ""
        else:
            self.grid_selection_var.set(self.grid_selection_options[-1])

    def on_resize(self, event):
        """Resize the image while maintaining its original aspect ratio."""
        if self.original_image:
            new_width = event.width
            new_height = event.height

            # Maintain the aspect ratio based on the new size
            if new_width / new_height > self.aspect_ratio:
                new_width = int(new_height * self.aspect_ratio)
            else:
                new_height = int(new_width / self.aspect_ratio)

            # Resize the image to the new width and height
            resized_image = self.original_image.resize(
                (new_width, new_height), Image.BILINEAR  # type: ignore
            )
            self.update_image(resized_image)

    def on_click(self, event):
        """Processes point addition."""
        if (
            self.image_tk
            and self.dragging_polygon_id < 0
            and self.dragging_point_idx < 0
            and event.x <= self.current_width
            and event.y <= self.current_height
        ):
            scaled_x = int(event.x * self.original_width / self.current_width)
            scaled_y = int(event.y * self.original_height / self.current_height)
            cur_sel = self.get_cur_sel()

            if cur_sel is not None:
                grid = self.grids[cur_sel]
                points_len = len(grid.points)

                if (
                    points_len > 4 and points_len % 2 == 1 and len(grid.ids) > 1
                ):  # Remove most recent intermediate polygon (if such exists)
                    self.canvas.delete(grid.ids.pop())

                # Add new point to grid
                grid.process_point_addition((scaled_x, scaled_y))
                grid.update_displayed_points(
                    self.current_width,
                    self.current_height,
                    self.original_width,
                    self.original_height,
                )

                # Draw added point
                points_len = len(grid.points)
                self.draw_point(
                    grid.displayed_points[-3 if points_len > 4 else -1], cur_sel
                )

                # Draw main grid and label (if it was just created)
                if points_len == 4:
                    self.draw_polygon(grid.get_grid_polygons()[-1], cur_sel)
                    self.draw_grid_label(grid)

                # Draw new intermediate zones (if such are created)
                if points_len > 4 and points_len % 2 == 0:
                    polygons = grid.get_grid_polygons()
                    self.draw_polygon(polygons[-2], cur_sel)
                    self.draw_polygon(polygons[-1], cur_sel)
            else:
                self.points.append((scaled_x, scaled_y))

                self.update_displayed_points()
                self.draw_point(self.displayed_points[-1])

                if len(self.displayed_points) % self.num_vertices == 0:
                    self.draw_polygon(self.displayed_points[-self.num_vertices :])

            if self.points or self.with_grid and self.grids:
                self.edit_menu.entryconfig("Undo", state=self.tk.NORMAL)

    def on_motion(self, event):
        """Processes actions related to mouse movement."""
        cur_sel = self.get_cur_sel()
        grid_chosen = cur_sel is not None
        if self.image_tk:
            # Update incomplete polygon drawing, if applicable
            self.canvas.delete("temp_polygon")
            if grid_chosen:
                grid = self.grids[cur_sel]
                temp_polygon = grid.get_temp_polygon()[0] + [(event.x, event.y)]
                if len(temp_polygon) > 1:
                    self.canvas.create_line(
                        temp_polygon,
                        fill="yellow",
                        tags="temp_polygon",
                        width=self.line_width,
                    )
            else:
                if len(self.displayed_points) % self.num_vertices != 0:
                    current_polygon = self.displayed_points[
                        -(len(self.displayed_points) % self.num_vertices) :
                    ] + [(event.x, event.y)]
                    self.canvas.create_line(
                        current_polygon,
                        fill="yellow",
                        tags="temp_polygon",
                        width=self.line_width,
                    )

        if self.dragging_polygon_id >= 0:
            # Move entire polygon
            offset = (event.x, event.y)
            self.update_dragging_polygon(offset, cur_sel)
            self.dragging_polygon_offset = offset

        if self.dragging_point_idx >= 0:
            # Move an individual point
            self.update_dragging_point((event.x, event.y), cur_sel)

    def on_right_click(self, event):
        """Processes movement of selected point."""
        if self.figures_complete() and self.dragging_polygon_id < 0:
            if self.dragging_point_idx >= 0:
                # Complete the drag operation
                self.dragging_point_idx = -1
                self.redraw_polygons()
                return

            # Right Click: Move a specific point
            cur_sel = self.get_cur_sel()
            points = (
                self.grids[cur_sel].displayed_points
                if cur_sel is not None
                else self.displayed_points
            )
            clicked_point, point_index = self.find_point(event.x, event.y, points)

            if clicked_point is not None:
                self.dragging_point_idx = point_index
                self.dragging_point_start = clicked_point

    def on_right_click_and_ctrl(self, event):
        """Processes movement of selected polygon."""
        if self.figures_complete() and self.dragging_point_idx < 0:
            if self.dragging_polygon_id >= 0:
                # Complete the drag operation
                self.dragging_polygon_id = -1
                self.redraw_polygons()
                return

            # Ctrl + Right Click: Drag entire polygon
            cur_sel = self.get_cur_sel()
            ids = self.grids[cur_sel].ids if cur_sel is not None else self.polygon_ids
            clicked_polygon_id, polygon_point_index = self.find_polygon(
                event.x, event.y, ids
            )
            if clicked_polygon_id is not None:
                self.dragging_polygon_id = clicked_polygon_id
                self.dragging_polygon_offset_start = self.dragging_polygon_offset = (
                    event.x,
                    event.y,
                )
                self.dragging_polygon_point_index = polygon_point_index

    def update_dragging_point(self, point, cur_sel):
        if cur_sel is not None:
            grid = self.grids[cur_sel]
            dpi = self.dragging_point_idx

            # Scale dragged point to original dimensions
            grid.points[dpi] = (
                int(point[0] * self.original_width / self.current_width),
                int(point[1] * self.original_height / self.current_height),
            )

            # Modify grid points based on the change to the dragged point
            if dpi < 2 or dpi >= len(grid.displayed_points) - 2:
                # corner anchor point: modify grid parameters, scale intermediate points
                grid.update_grid_parameters()
                for i in range(2, len(grid.points) - 2):
                    grid.points[i] = grid.scale_point(grid.points[i], i)
            else:
                # intermediate anchor point: scale it based on grid parameters
                grid.points[dpi] = grid.scale_point(grid.points[dpi], dpi)

            # Update displayed grid points
            grid.update_displayed_points(
                self.current_width,
                self.current_height,
                self.original_width,
                self.original_height,
            )
        else:
            # Update displayed dragged point
            self.displayed_points[self.dragging_point_idx] = point

            # Scale dragged point to original dimensions
            self.points[self.dragging_point_idx] = (
                int(point[0] * self.original_width / self.current_width),
                int(point[1] * self.original_height / self.current_height),
            )

        # Re-draw all figures
        self.redraw_polygons()

    def update_dragging_polygon(self, offset, cur_sel):
        grid_chosen = cur_sel is not None

        # Calculate the offset for moving the polygon
        dx = offset[0] - self.dragging_polygon_offset[0]
        dy = offset[1] - self.dragging_polygon_offset[1]

        # Move the displayed points of the selected polygon
        if grid_chosen:
            grid = self.grids[cur_sel]
            for i in range(len(grid.displayed_points)):
                grid.displayed_points[i] = (
                    grid.displayed_points[i][0] + dx,
                    grid.displayed_points[i][1] + dy,
                )
                grid.points[i] = (
                    int(
                        grid.displayed_points[i][0]
                        * self.original_width
                        / self.current_width
                    ),
                    int(
                        grid.displayed_points[i][1]
                        * self.original_height
                        / self.current_height
                    ),
                )
        else:
            for i in range(
                self.dragging_polygon_point_index,
                self.dragging_polygon_point_index + self.num_vertices,
            ):
                self.displayed_points[i] = (
                    self.displayed_points[i][0] + dx,
                    self.displayed_points[i][1] + dy,
                )
                self.points[i] = (
                    int(
                        self.displayed_points[i][0]
                        * self.original_width
                        / self.current_width
                    ),
                    int(
                        self.displayed_points[i][1]
                        * self.original_height
                        / self.current_height
                    ),
                )

        # Update the polygon's position on the canvas
        if grid_chosen:
            for disp_polygon_id, disp_polygon in zip(
                grid.ids, grid.get_grid_polygons()
            ):
                self.canvas.coords(disp_polygon_id, *sum(disp_polygon, []))
            self.remove_grid_label(grid)
            self.draw_grid_label(grid)
        else:
            self.canvas.coords(
                self.dragging_polygon_id,
                *sum(
                    self.displayed_points[
                        self.dragging_polygon_point_index : self.dragging_polygon_point_index
                        + self.num_vertices
                    ],
                    (),
                ),
            )

    def find_polygon(self, x, y, ids):
        """Find a polygon or line near the given (x, y) position."""
        for idx, polygon_id in enumerate(ids):
            points = self.canvas.coords(polygon_id)
            if self.num_vertices == 2:  # Handle line
                if self.is_near_line(x, y, points):
                    return polygon_id, idx * self.num_vertices
            else:  # Handle polygon
                if (
                    cv2.pointPolygonTest(
                        np.array(points, dtype=np.int32).reshape(-1, 2), (x, y), False
                    )
                    > 0
                ):
                    return polygon_id, idx * self.num_vertices
        return None, None

    def find_point(self, x, y, points):
        """Find a specific point near the given (x, y) position."""
        for i, (px, py) in enumerate(points):
            if abs(px - x) < 5 and abs(py - y) < 5:
                return (px, py), i
        return None, None

    def is_near_line(self, x, y, line_points):
        """Check if (x, y) is near the line defined by line_points."""
        x1, y1, x2, y2 = line_points
        dist = (
            abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
            / ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        )
        return dist < 5  # Threshold distance to be "near" the line

    def get_cur_sel(self):
        """Returns current selection for editing."""
        if self.with_grid:
            cur_sel = self.grid_selection_var.get()
            if cur_sel != self.grid_selection_default_value:
                return self.grid_key_to_idx(cur_sel)
            else:
                return None
        else:
            return None

    def get_tag(self, prefix: str = "", grid_idx=None) -> str:
        """Returns selection-specific tag for a given canvas element type"""
        tag = prefix
        if grid_idx is not None:
            tag += f"_grid_{str(grid_idx)}"
        return tag

    def draw_point(self, point, grid_id=None):
        """Draws point on canvas."""
        self.canvas.create_oval(
            point[0] - 2,
            point[1] - 2,
            point[0] + 2,
            point[1] + 2,
            fill="red",
            outline="red",
            width=2,
            tags=self.get_tag("point", grid_id),
        )

    def draw_polygon(self, points, grid_id=None):
        """Draws polygon on canvas."""
        polygon_id = self.canvas.create_polygon(
            points,
            outline="yellow",
            fill="",
            width=self.line_width,
            tags=self.get_tag("polygon", grid_id),
        )
        if grid_id is not None:
            self.grids[grid_id].ids.append(polygon_id)
        else:
            self.polygon_ids.append(polygon_id)

    def redraw_polygons(self):
        """Redraws polygons on canvas."""
        self.canvas.delete(self.get_tag("polygon"))
        self.canvas.delete(self.get_tag("point"))
        self.polygon_ids.clear()
        for i, point in enumerate(self.displayed_points):
            self.draw_point(point)
            if (i + 1) % self.num_vertices == 0:
                self.draw_polygon(
                    self.displayed_points[i - self.num_vertices + 1 : i + 1]
                )
        if self.with_grid:
            for grid_id in self.grid_selection_options:
                if grid_id == self.grid_selection_default_value:
                    continue
                idx = self.grid_key_to_idx(grid_id)
                self.canvas.delete(self.get_tag("polygon", idx))
                self.canvas.delete(self.get_tag("point", idx))
                grid = self.grids[idx]
                self.remove_grid_label(grid)
                grid.ids.clear()
                grid.update_grid_parameters()
                if len(grid.displayed_points) >= 4:
                    for i in [0, 1, -2, -1]:
                        self.draw_point(grid.displayed_points[i], idx)
                    for point in grid.displayed_points[2:-2]:
                        self.draw_point(point, idx)
                    for poly in grid.get_grid_polygons():
                        self.draw_polygon(poly, idx)
                    self.draw_grid_label(grid)

    def grid_key_to_idx(self, key: str):
        return int(key.lstrip("Grid "))

    def grid_idx_to_key(self, idx: int):
        return "Grid " + str(idx)

    def add_grid(self, event=None):
        """Add grid."""
        if self.image_tk:
            grid_ids = sorted(list(self.grids.keys()))
            new_grid_id = next(
                (i for i, num in enumerate(grid_ids) if i != num), len(grid_ids)
            )
            self.grids[new_grid_id] = Grid(str(new_grid_id))
            self.added_grid_id = self.grid_idx_to_key(new_grid_id)
            self.update_grid_menu()
            self.edit_menu.entryconfig("Remove Grid", state=self.tk.NORMAL)

    def remove_grid(self, event=None):
        """Remove grid."""
        if (
            self.image_tk
            and all(
                [
                    grid.complete() or not len(grid.points)
                    for grid in self.grids.values()
                ]
            )
            and self.dragging_point_idx < 0
            and self.dragging_polygon_id < 0
        ):
            cur_sel = self.get_cur_sel()
            if cur_sel is not None:
                self.canvas.delete(self.get_tag("polygon", cur_sel))
                self.canvas.delete(self.get_tag("point", cur_sel))
                grid = self.grids[cur_sel]
                if grid.complete():
                    self.remove_grid_label(grid)
                self.grids.pop(cur_sel)
                self.update_grid_menu()
                if len(self.grids.values()) == 0:
                    self.edit_menu.entryconfig("Remove Grid", state=self.tk.DISABLED)
                    if self.figures_empty():
                        self.edit_menu.entryconfig("Undo", state=self.tk.DISABLED)

    def draw_grid_label(self, grid: Grid):
        ref_point = grid.displayed_points[0]
        center_offset = 10
        if grid.top_m >= 0:
            if ref_point[0] < grid.displayed_points[-1][0]:
                circle_center = (
                    ref_point[0] - center_offset,
                    ref_point[1] - center_offset,
                )
            else:
                circle_center = (
                    ref_point[0] + center_offset,
                    ref_point[1] + center_offset,
                )
        else:
            if ref_point[0] < grid.displayed_points[-1][0]:
                circle_center = (
                    ref_point[0] - center_offset,
                    ref_point[1] + center_offset,
                )
            else:
                circle_center = (
                    ref_point[0] + center_offset,
                    ref_point[1] - center_offset,
                )
        label = grid.grid_id
        tag = self.get_tag("label", label)
        self.canvas.create_oval(
            circle_center[0] - 7,
            circle_center[1] - 7,
            circle_center[0] + 7,
            circle_center[1] + 7,
            fill="yellow",
            outline="yellow",
            tags=tag,
        )
        self.canvas.create_text(
            circle_center[0],
            circle_center[1],
            text=label,
            fill="red",
            font=("Arial", 10, "bold"),
            tags=tag,
        )

    def remove_grid_label(self, grid: Grid):
        self.canvas.delete(self.get_tag("label", grid.grid_id))

    def process_esc(self, event=None):
        cur_sel = self.get_cur_sel()
        if self.dragging_point_idx >= 0:
            self.update_dragging_point(self.dragging_point_start, cur_sel)
            self.dragging_point_idx = -1
        elif self.dragging_polygon_id >= 0:
            self.update_dragging_polygon(self.dragging_polygon_offset_start, cur_sel)
            self.dragging_polygon_id = -1
        elif self.image_tk:
            self.canvas.delete("temp_polygon")
            if cur_sel is not None:
                grid = self.grids[cur_sel]
                temp_polygon_point_indices = grid.get_temp_polygon()[1]
                if temp_polygon_point_indices:
                    del grid.points[
                        temp_polygon_point_indices[0] : temp_polygon_point_indices[-1]
                        + 1
                    ]
                    grid.update_displayed_points(
                        self.current_width,
                        self.current_height,
                        self.original_width,
                        self.original_height,
                    )
            else:
                if len(self.points) % self.num_vertices != 0:
                    del self.points[-(len(self.points) % self.num_vertices) :]
                    self.update_displayed_points()
            self.redraw_polygons()
            if self.figures_empty():
                self.edit_menu.entryconfig("Undo", state=self.tk.DISABLED)

    def undo(self, event=None):
        """Processes point deletion."""
        if self.dragging_polygon_id < 0 and self.dragging_point_idx < 0:
            cur_sel = self.get_cur_sel()
            if cur_sel is not None:
                grid = self.grids[cur_sel]
                points = grid.points
                points_len = len(points)
                if not points_len:  # If grid has no points, delete it
                    self.grids.pop(cur_sel)
                    self.update_grid_menu()
                    if len(self.grids.values()) == 0:
                        self.edit_menu.entryconfig(
                            "Remove Grid", state=self.tk.DISABLED
                        )
                else:
                    if grid.complete():
                        self.canvas.delete(
                            grid.ids.pop()
                        )  # Remove the last polygon of the grid from canvas
                        if grid.ids:
                            self.canvas.delete(
                                grid.ids.pop()
                            )  # Remove second-to-last polygon if such exists

                    self.canvas.delete(
                        self.canvas.find_withtag(self.get_tag("point", cur_sel))[-1]
                    )  # Remove the last point from canvas

                    points.pop(-1 if points_len <= 4 else -3)
                    grid.update_displayed_points(
                        self.current_width,
                        self.current_height,
                        self.original_width,
                        self.original_height,
                    )

                    points_len = len(points)
                    if points_len < 4:
                        self.remove_grid_label(grid)
                    if points_len > 4 and points_len % 2 == 0:
                        polygons = grid.get_grid_polygons()
                        self.draw_polygon(polygons[-1], cur_sel)

                    self.canvas.delete("temp_polygon")
                    temp_polygon = grid.get_temp_polygon()[0]
                    if len(temp_polygon) > 1:
                        self.canvas.create_line(
                            temp_polygon,
                            fill="yellow",
                            tags="temp_polygon",
                            width=self.line_width,
                        )
            elif self.points:
                if len(self.points) % self.num_vertices == 0 and self.polygon_ids:
                    self.canvas.delete(
                        self.polygon_ids.pop()
                    )  # Remove the last polygon from canvas
                self.canvas.delete(
                    self.canvas.find_withtag(self.get_tag("point"))[-1]
                )  # Remove the last point from canvas
                self.points.pop()  # Remove the last point
                self.update_displayed_points()

                self.canvas.delete("temp_polygon")
                if len(self.displayed_points) % self.num_vertices != 0:
                    current_polygon = self.displayed_points[
                        -(len(self.displayed_points) % self.num_vertices) :
                    ]
                    if len(current_polygon) > 1:
                        self.canvas.create_line(
                            current_polygon,
                            fill="yellow",
                            tags="temp_polygon",
                            width=self.line_width,
                        )

            if self.figures_empty():
                self.edit_menu.entryconfig("Undo", state=self.tk.DISABLED)

    def figures_complete(self):
        return (
            all(
                [grid.complete() for grid in self.grids.values()]
                if self.with_grid
                else [True]
            )
            and len(self.points) % self.num_vertices == 0
        )

    def figures_empty(self):
        return not (self.points or self.with_grid and self.grids)

    def check_completeness_on_save(self):
        if (
            self.points
            and len(self.points) % self.num_vertices != 0
            or self.with_grid
            and (
                not self.grids
                and not self.points
                or self.grids
                and any([not grid.complete() for grid in self.grids.values()])
            )
        ):
            self.tk.messagebox.showerror(
                "Error", "No points or insufficient points to save."
            )
            return False
        return True

    def data_updated(self):
        return (
            self.saved_points != self.points
            or self.with_grid
            and self.grids
            and (
                len(self.saved_grids.values()) != len(self.grids.values())
                or any(
                    [
                        saved_grid.points != grid.points
                        for saved_grid, grid in zip(
                            self.saved_grids.values(), self.grids.values()
                        )
                    ]
                )
            )
        )

    def save_data(self):
        self.saved_points = deepcopy(self.points)
        if self.with_grid:
            self.saved_grids = deepcopy(self.grids)
        out_json = {
            "version": object_format_versions[self.figure_type],
            "type": self.figure_type,
        }
        data = [
            (
                [*sum(self.points[i : i + self.num_vertices], ())]
                if self.num_vertices == FigureAnnotatorType.LINE.value
                else self.points[i : i + self.num_vertices]
            )
            for i in range(
                0, len(self.points) - self.num_vertices + 1, self.num_vertices
            )
        ]
        if self.with_grid:
            for grid in self.grids.values():
                polygons = grid.get_grid_polygons(display=False)
                data.extend(polygons[1:] if len(polygons) > 1 else polygons)
        out_json["objects"] = data
        with open(self.save_path, "w") as f:
            json.dump(out_json, f, indent=4)

    def save(self, event=None):
        if self.image_tk:
            if not self.check_completeness_on_save():
                return
            if self.data_updated():
                if self.save_path:
                    # Data already saved previously.
                    self.save_data()
                else:
                    # Data has not been saved to a file yet.
                    self.save_to_file()

    def save_to_file(self, event=None):
        """Saves selected figures to JSON file."""
        if self.image_tk:
            if not self.check_completeness_on_save():
                return

            self.save_path = self.tkFiledialog.asksaveasfilename(
                initialfile=self.results_file_name,
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")],
            )
            if self.save_path:
                self.save_data()

    def on_close(self):
        """Prompt the user before closing the window if there are unsaved changes."""
        if not self.figures_empty() and self.data_updated():
            response = self.tk.messagebox.askyesnocancel(
                "Unsaved Changes", "You have unsaved changes. Save before exiting?"
            )
            if response:  # Yes: Save the changes
                self.save()
                self.root.destroy()  # Then close the window
            elif response is False:  # No: Discard changes and exit
                self.root.destroy()
            else:  # Cancel: Do nothing, stay in the application
                return
        else:
            # No unsaved changes, close the window
            self.root.destroy()

    def get_help_message(self):
        figure_type = self.figure_type
        if figure_type == "line":
            return help_message_line
        elif figure_type == "zone":
            if self.num_vertices == FigureAnnotatorType.QUADRILATERAL.value:
                return help_message_grid
            else:
                return help_message_polygon

    def show_help(self):
        help_window = self.tk.Toplevel(self.root)
        help_window.title(f"About {self.figure_type.capitalize()} Annotator")
        help_window.geometry("1100x500")  # Set the size of the window

        scrollbar = self.tk.Scrollbar(help_window)
        scrollbar.pack(side=self.tk.RIGHT, fill=self.tk.Y)

        help_text = self.tk.Text(
            help_window, wrap=self.tk.WORD, yscrollcommand=scrollbar.set, font=self.font
        )

        help_text.insert(self.tk.END, self.get_help_message())
        help_text.config(state=self.tk.DISABLED)
        help_text.pack(expand=True, fill=self.tk.BOTH)
        scrollbar.config(command=help_text.yview)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=f"{__file__}",
        description="Launch interactive utility for geometric figure annotation in images",
    )

    _figure_annotator_args(parser)
    _figure_annotator_run(parser.parse_args())
