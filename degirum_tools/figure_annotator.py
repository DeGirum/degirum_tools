#
# figure_annotator.py: geometric figure annotation command-line utility
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements classes for the geometric figure annotation command-line utility
# and the driver for this utility.
#

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import json
import argparse
import cv2
import numpy as np
from enum import Enum
from typing import Tuple, List, Dict, Optional


def _figure_annotator_run(args):
    """
    Launch interactive utility for geometric figure annotation in images.

    Args:
        args: argparse command line arguments
    """
    if args.num_vertices < 2:
        raise ValueError("Number of vertices must be at least 2")

    FigureAnnotator(num_vertices=args.num_vertices, results_file_name=args.save_path)


def _zone_annotator_run(args):
    """
    Launch interactive utility for 4-sided zone annotation in images.

    Args:
        args: argparse command line arguments
    """
    FigureAnnotator(results_file_name=args.save_path)


def _line_annotator_run(args):
    """
    Launch interactive utility for line annotation in images.

    Args:
        args: argparse command line arguments
    """
    FigureAnnotator(
        num_vertices=FigureAnnotatorType.LINE.value, results_file_name=args.save_path
    )


def _figure_annotator_args(parser):
    """
    Define figure_annotator subcommand arguments

    Args:
        parser: argparse parser object to be stuffed with args
    """
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
    def __init__(self):
        self.ids = []
        self.points = []
        self.displayed_points = []
        self.top_m = 0
        self.top_b = 0
        self.bottom_m = 0
        self.bottom_b = 0

    def complete(self) -> bool:
        """
        Returns True if grid is complete, i. e.
        at least four points are defined and number of points is even.
        """
        points_len = len(self.points)
        return points_len >= 4 and points_len % 2 == 0

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
            self.top_m = (guide_top_r[1] - guide_top_l[1]) / (
                guide_top_r[0] - guide_top_l[0]
            )
            self.top_b = guide_top_r[1] - self.top_m * guide_top_r[0]
            self.bottom_m = (guide_bot_r[1] - guide_bot_l[1]) / (
                guide_bot_r[0] - guide_bot_l[0]
            )
            self.bottom_b = guide_bot_r[1] - self.bottom_m * guide_bot_r[0]

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

    def get_temp_polygon(self) -> Tuple[List[Tuple[float]], List[int]]:
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
        points = self.displayed_points if display else self.points
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
        results_file_name: str = "zones.json",
    ):
        self.num_vertices = num_vertices  # Number of vertices for each polygon
        self.figure_type: str = (
            "line" if num_vertices == FigureAnnotatorType.LINE.value else "zone"
        )
        self.with_grid: bool = (
            self.num_vertices == FigureAnnotatorType.QUADRILATERAL.value
        )
        self.results_file_name = results_file_name

        self.root = tk.Tk()
        self.root.title(f"{self.figure_type.capitalize()} Annotator")

        # Create the main frame to hold the menu and canvas
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)

        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(
            label="Open...", command=self.open_image, accelerator="Ctrl-O"
        )
        file_menu.add_command(
            label="Save As...", command=self.save_as_json, accelerator="Ctrl-S"
        )
        self.menu_bar.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(self.menu_bar, tearoff=0)
        if self.with_grid:
            edit_menu.add_command(
                label="Add grid", command=self.add_grid, accelerator="Ctrl-A"
            )
            edit_menu.add_command(
                label="Remove grid", command=self.remove_grid, accelerator="Ctrl-D"
            )
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Ctrl-Z")
        self.menu_bar.add_cascade(label="Edit", menu=edit_menu)

        if self.with_grid:
            # Create a frame for the second menu and "Current Selection" OptionMenu
            self.grid_selection_frame = tk.Frame(self.main_frame)
            self.grid_selection_frame.pack(fill=tk.X, pady=5)

            # Add "Active Grid" ComboBox to the grid_selection_frame
            self.grid_selection_menu_label = tk.Label(
                self.grid_selection_frame, text="Active Grid"
            )
            self.grid_selection_menu_label.grid(row=0, column=0)
            self.grid_selection_default_value = "Non-grid mode"
            self.added_grid_id = ""
            self.grid_selection_var = tk.StringVar(self.grid_selection_frame)
            self.grid_selection_var.set(
                self.grid_selection_default_value
            )  # Default value
            self.grid_selection_options = [self.grid_selection_default_value]

            self.grid_selection_menu = ttk.Combobox(
                self.grid_selection_frame,
                textvariable=self.grid_selection_var,
                values=self.grid_selection_options,
            )
            self.grid_selection_menu.grid(row=0, column=1, padx=10)

        self.canvas = tk.Canvas(self.main_frame, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.image_path: str = ""
        self.original_image: Optional[Image.Image] = None  # Store the original image
        self.image_tk: Optional[ImageTk.PhotoImage] = None
        self.points: List[Tuple] = []  # Points relative to the original image
        self.displayed_points: List[Tuple] = []  # Points relative to the resized image
        self.polygon_ids: List[int] = []  # Store polygon IDs for undo

        if self.with_grid:
            self.grids: Dict[int, Grid] = {}  # Grids

        self.original_width = 0  # Store original image width
        self.original_height = 0  # Store original image height
        self.current_width = 0  # Current width of the displayed image
        self.current_height = 0  # Current height of the displayed image
        self.aspect_ratio = 1.0  # Aspect ratio of the original image

        # Dragging state
        self.dragging_polygon_id = None  # The polygon ID being dragged
        self.dragging_polygon_offset: Optional[Tuple] = (
            None  # The offset from the click position to the vertices
        )
        self.dragging_polygon_offset_start: Optional[Tuple] = (
            None  # The starting offset from the click position to the vertices
        )
        self.dragging_polygon_point_index = (
            None  # The index of the points in `self.points` being dragged
        )
        self.dragging_point_start = (
            None  # The starting coordinates of the dragged point
        )
        self.dragging_point_idx = None  # The index of the point being dragged

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
        self.root.bind_all("<Control-s>", self.save_as_json)
        self.root.bind_all("<Control-z>", self.undo)
        self.root.bind_all("<Escape>", self.process_esc)

        # Run application
        self.root.mainloop()

    def open_image(self, event=None):
        """Opens image."""
        self.image_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if self.image_path:
            self.load_image()

    def load_image(self):
        """Loads image."""
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

    def update_image(self, image):
        """Helper function to update the image on the canvas."""
        self.current_width, self.current_height = image.size
        self.image_tk = ImageTk.PhotoImage(image)
        self.canvas.config(width=self.current_width, height=self.current_height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)

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
        self.grid_selection_options = [self.grid_selection_options[0]] + sorted(
            [self.grid_idx_to_key(v) for v in self.grids.keys()]
        )
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
                (new_width, new_height), Image.BILINEAR
            )
            self.update_image(resized_image)

    def on_click(self, event):
        """Processes point addition."""
        if (
            self.image_tk
            and self.dragging_polygon_id is None
            and self.dragging_point_idx is None
            and event.x <= self.current_width
            and event.y <= self.current_height
        ):
            scaled_x = int(event.x * self.original_width / self.current_width)
            scaled_y = int(event.y * self.original_height / self.current_height)
            cur_sel = self.get_cur_sel()

            if cur_sel is not None:
                grid = self.grids[cur_sel]

                points_len = len(grid.points)
                if points_len >= 4:  # Add intermediate point to grid
                    if points_len % 2 == 0:
                        scaled_y = FigureAnnotator.lin_func(
                            grid.top_m, grid.top_b, scaled_x
                        )
                    else:
                        scaled_y = FigureAnnotator.lin_func(
                            grid.bottom_m, grid.bottom_b, scaled_x
                        )

                if (
                    points_len > 4 and points_len % 2 == 1
                ):  # Remove most recent intermediate polygon (if such exists)
                    if len(grid.ids) > 1:
                        self.canvas.delete(grid.ids.pop())

                # Add new point
                (
                    grid.points.insert(-2, (scaled_x, scaled_y))
                    if points_len >= 4
                    else grid.points.append((scaled_x, scaled_y))
                )
                grid.update_displayed_points(
                    self.current_width,
                    self.current_height,
                    self.original_width,
                    self.original_height,
                )

                points_len = len(grid.points)
                self.draw_point(
                    grid.displayed_points[-3 if points_len > 4 else -1], cur_sel
                )

                if (
                    points_len == 4
                ):  # Determine parameters of minimally-complete grid and draw on canvas
                    grid.update_grid_parameters()
                    self.draw_polygon(grid.get_grid_polygons()[-1], cur_sel)
                if (
                    points_len > 4 and points_len % 2 == 0
                ):  # Add new intermediate polygons (if such are created)
                    polygons = grid.get_grid_polygons()
                    self.draw_polygon(polygons[-2], cur_sel)
                    self.draw_polygon(polygons[-1], cur_sel)
            else:
                self.points.append((scaled_x, scaled_y))

                self.update_displayed_points()
                self.draw_point(self.displayed_points[-1])

                if len(self.displayed_points) % self.num_vertices == 0:
                    self.draw_polygon(self.displayed_points[-self.num_vertices :])

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
                        temp_polygon, fill="yellow", tags="temp_polygon", width=2
                    )
            else:
                if len(self.displayed_points) % self.num_vertices != 0:
                    current_polygon = self.displayed_points[
                        -(len(self.displayed_points) % self.num_vertices) :
                    ] + [(event.x, event.y)]
                    self.canvas.create_line(
                        current_polygon, fill="yellow", tags="temp_polygon", width=2
                    )

        if self.dragging_polygon_id is not None:
            # Move entire polygon
            offset = (event.x, event.y)
            self.update_dragging_polygon(offset, cur_sel)
            self.dragging_polygon_offset = offset

        if self.dragging_point_idx is not None:
            # Move an individual point
            if not grid_chosen:
                self.update_dragging_point((event.x, event.y))

    def on_right_click(self, event):
        """Processes movement of selected point."""
        if self.dragging_polygon_id is None:
            if self.dragging_point_idx is not None:
                # Complete the drag operation
                self.dragging_point_idx = None
                self.redraw_polygons()
                return

            # Right Click: Move a specific point
            cur_sel = self.get_cur_sel()
            if cur_sel is not None:
                clicked_point = None
            else:
                clicked_point, point_index = self.find_point(
                    event.x, event.y, self.displayed_points
                )

            if clicked_point is not None:
                self.dragging_point_idx = point_index
                self.dragging_point_start = clicked_point

    def on_right_click_and_ctrl(self, event):
        """Processes movement of selected polygon."""
        if self.dragging_point_idx is None:
            if self.dragging_polygon_id is not None:
                # Complete the drag operation
                self.dragging_polygon_id = None
                self.redraw_polygons()
                return

            # Ctrl + Right Click: Drag entire polygon
            cur_sel = self.get_cur_sel()
            ids = self.grids[cur_sel].ids if cur_sel is not None else self.polygon_ids
            clicked_polygon_id, polygon_index = self.find_polygon(event.x, event.y, ids)
            if clicked_polygon_id is not None:
                self.dragging_polygon_id = clicked_polygon_id
                self.dragging_polygon_offset_start = self.dragging_polygon_offset = (
                    event.x,
                    event.y,
                )
                self.dragging_polygon_point_index = polygon_index

    def update_dragging_point(self, point):
        self.displayed_points[self.dragging_point_idx] = point
        self.points[self.dragging_point_idx] = (
            int(point[0] * self.original_width / self.current_width),
            int(point[1] * self.original_height / self.current_height),
        )
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

    @staticmethod
    def lin_func(m, b, x):
        """Return y-coordinate for a given x-coordinate based on line defined by slope and y-intercept."""
        return m * x + b

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

    def get_tag(self, prefix="", grid_idx=None):
        """Returns selection-specific tag for a given canvas element type"""
        tag = prefix
        if grid_idx is not None:
            tag += f"_grid_{str(grid_idx)}"
        return tag

    def draw_point(self, point, grid_id=None):
        """Draws point on canvas."""
        self.canvas.create_oval(
            point[0] - 3,
            point[1] - 3,
            point[0] + 3,
            point[1] + 3,
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
            width=2,
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
                grid.ids.clear()
                grid.update_grid_parameters()
                if len(grid.displayed_points) >= 4:
                    for i in [0, 1, -2, -1]:
                        self.draw_point(grid.displayed_points[i], idx)
                    for point in grid.displayed_points[2:-2]:
                        self.draw_point(point, idx)
                    for poly in grid.get_grid_polygons():
                        self.draw_polygon(poly, idx)

    def grid_key_to_idx(self, key: str):
        return int(key.lstrip("Grid "))

    def grid_idx_to_key(self, idx: int):
        return "Grid " + str(idx)

    def add_grid(self, event=None):
        """Add grid."""
        if self.image_tk:
            grid_ids = list(self.grids.keys())
            new_grid_id = next(
                (i for i, num in enumerate(grid_ids) if i != num), len(grid_ids)
            )
            self.grids[new_grid_id] = Grid()
            self.added_grid_id = self.grid_idx_to_key(new_grid_id)
            self.update_grid_menu()

    def remove_grid(self, event=None):
        """Remove grid."""
        if self.image_tk:
            cur_sel = self.get_cur_sel()
            if cur_sel is not None:
                self.canvas.delete(self.get_tag("polygon", cur_sel))
                self.canvas.delete(self.get_tag("point", cur_sel))
                self.grids.pop(cur_sel)
                self.update_grid_menu()

    def process_esc(self, event=None):
        cur_sel = self.get_cur_sel()
        if self.dragging_point_idx is not None:
            self.update_dragging_point(self.dragging_point_start)
            self.dragging_point_idx = None
        elif self.dragging_polygon_id is not None:
            self.update_dragging_polygon(self.dragging_polygon_offset_start, cur_sel)
            self.dragging_polygon_id = None
        elif self.image_tk:
            self.canvas.delete("temp_polygon")
            if cur_sel is not None:
                grid = self.grids[cur_sel]
                temp_polygon_point_indices = grid.get_temp_polygon()[1]
                if temp_polygon_point_indices:
                    for idx in temp_polygon_point_indices:
                        grid.points.pop(idx)
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

    def undo(self, event=None):
        """Processes point deletion."""
        if self.dragging_polygon_id is None and self.dragging_point_idx is None:
            cur_sel = self.get_cur_sel()
            if cur_sel is not None:
                grid = self.grids[cur_sel]
                points = grid.points
                points_len = len(points)
                if not points_len:  # If grid has no points, delete it
                    self.grids.pop(cur_sel)
                    self.update_grid_menu()
                else:
                    if points_len >= 4 and points_len % 2 == 0:
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
                    if points_len > 4 and points_len % 2 == 0:
                        polygons = grid.get_grid_polygons()
                        self.draw_polygon(polygons[-1], cur_sel)

                    self.canvas.delete("temp_polygon")
                    temp_polygon = grid.get_temp_polygon()[0]
                    if len(temp_polygon) > 1:
                        self.canvas.create_line(
                            temp_polygon, fill="yellow", tags="temp_polygon", width=2
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
                            current_polygon, fill="yellow", tags="temp_polygon", width=2
                        )

    def save_as_json(self, event=None):
        """Saves selected figures to JSON file."""
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
            messagebox.showerror("Error", "No points or insufficient points to save.")
            return

        save_path = filedialog.asksaveasfilename(
            initialfile=self.results_file_name,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
        )
        if save_path:
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
            with open(save_path, "w") as f:
                json.dump(out_json, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=f"{__file__}",
        description="Launch interactive utility for geometric figure annotation in images",
    )

    _figure_annotator_args(parser)
    _figure_annotator_run(parser.parse_args())
