#
# figure_annotator.py: geometric figure annotation command-line utility
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements classes for the geometric figure annotation command-line utility
# and the driver for this utility.
#

import argparse
import json
from copy import deepcopy
from typing import List, Dict, Tuple, Union, Optional
from enum import Enum
from tkinter import filedialog, messagebox, OptionMenu
from PIL import Image, ImageTk


def _figure_annotator_run(args):
    """
    Launch interactive utility for geometric figure annotation in images.

    Args:
        args: argparse command line arguments
    """
    FigureAnnotator(num_vertices=args.num_vertices, results_file_name=args.save_path)


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


class FigureAnnotatorType(Enum):
    """Figure annotator types"""

    LINE = 2
    QUADRILATERAL = 4


class PointSelState(Enum):
    """Point selection states for a region in automatic mode"""

    RightAnchors = 1  # Right anchor points are being selected
    LeftAnchors = 2  # Left anchor points are being selected
    IntermediateAnchors = 3  # Intermediate anchor points are being selected


class Region:
    """
    A class for defining a region of figures.

    Attributes:
        auto_mode (bool): when True, zones in region are generated using annotated
            anchor points; when False, all figure corners are annotated by the user
        current_selection (List[tuple]): list of tuples, each tuple is a point (x,y);
            when auto_mode is True, holds at most 2 point tuples (anchor pair) before
            this list is appended to either `anchor_points` or `intermediate_points`;
            when auto_mode is False, holds at most n point tuples (figure) before this
            list is appended to `figures`; n is number of vertices in figure
        anchor_points (List[List[tuple]]): list of 2 lists of point (x,y) tuples -
            rightmost anchor points, leftmost anchor points; only present when
            `auto_mode` is True
        intermediate_points (List[List[tuple]]): list of lists of point (x,y) tuples -
            intermediate anchor points; only present when `auto_mode` is True
        point_selection_state (PointSelState): annotation state of region when
            `auto_mode` is True
        figures (List[List[tuple]]): list of lists, where each list represents a figure,
            containing n point (x,y) tuples
    """

    def __init__(self, auto_mode: bool = False):
        self.auto_mode = auto_mode
        self.current_selection: List[tuple] = []
        if auto_mode:
            self.anchor_points: List[List[tuple]] = []
            self.intermediate_points: List[List[tuple]] = []
            self.point_selection_state = PointSelState.RightAnchors
        else:
            self.figures: List[List[tuple]] = []

    def is_empty(self) -> bool:
        """Returns True if self is a newly-instantiated object, False otherwise"""
        return self.__class__(self.auto_mode).__dict__ == self.__dict__


class FigureAnnotator:
    def __init__(
        self,
        num_vertices: int = 4,
        results_file_name: str = "zones.json",
        debug: bool = False,
    ):
        """Class to initialize and drive the UI for selecting geometric figures in a tkinter window.

        Attributes:
            results_file_name (str, optional): JSON file path to save figures
        """

        # Initialize properties
        self.image_path: Optional[str] = None
        self.image: Optional[Image.Image] = None
        self.canvas_image: Optional[ImageTk.PhotoImage] = None
        self.regions_json: List[Dict[str, List[Dict[str, List[tuple]]]]] = []
        self.img_width: int = 0
        self.img_height: int = 0
        self.regions = [Region()]
        self.results_file_name = results_file_name
        self.num_vertices = num_vertices
        self.figure_type = (
            "line" if num_vertices == FigureAnnotatorType.LINE.value else "zone"
        )

        if not debug:
            # Setup master window
            import tkinter as tk

            self._tk = tk
            self._master = tk.Tk()
            self._master.title(f"{self.figure_type.capitalize()} Selection")

            # Disable window resizing
            self._master.resizable(False, False)

            # Setup canvas for image display
            self._canvas = self._tk.Canvas(self._master, bg="white")

            # Define variables for entries
            self._offset_var = self._tk.IntVar(value=1)
            self._spacer_var = self._tk.IntVar(value=1)
            self._num_zones_between_barriers = self._tk.StringVar(value="1")
            self._num_zones_between_barriers_options = ["1", "2"]
            self._current_region_idx = self._tk.StringVar(value="0")
            self._current_region_idx_options = ["0"]
            self._save_format = self._tk.StringVar(value="DeGirum-compatible")
            self._save_format_options = ["DeGirum-compatible", "Ultralytics-compatible"]

            # Setup buttons
            _button_frame = self._tk.Frame(self._master)
            _button_frame.pack(side=self._tk.TOP)

            self._tk.Button(
                _button_frame, text="Load Image", command=self._load_image
            ).grid(row=0, column=0)
            self._tk.Button(
                _button_frame,
                text="Remove Last Selection",
                command=self._remove_last_selection,
            ).grid(row=0, column=1)
            self._tk.Button(
                _button_frame, text="Compute", command=self._compute_and_draw_figures
            ).grid(row=0, column=2)
            self._tk.Button(
                _button_frame, text="Add Region", command=self._add_region
            ).grid(row=0, column=3)
            self._tk.Button(
                _button_frame, text="Save", command=self._save_to_json
            ).grid(row=0, column=4)
            self._tk.Label(_button_frame, text="Format").grid(row=0, column=5)
            self._save_format_drop = OptionMenu(
                _button_frame,
                self._save_format,
                *self._save_format_options,
            )
            self._save_format_drop.grid(row=0, column=6)

            if num_vertices == FigureAnnotatorType.QUADRILATERAL.value:
                self._tk.Label(
                    _button_frame, text="Number of zones between barriers"
                ).grid(row=1, column=0)
                self._num_zones_between_barriers_drop = OptionMenu(
                    _button_frame,
                    self._num_zones_between_barriers,
                    *self._num_zones_between_barriers_options,
                )
                self._num_zones_between_barriers_drop.grid(row=1, column=1)
                self._tk.Label(_button_frame, text="Offset from boundary point").grid(
                    row=1, column=2
                )
                self._tk.Entry(_button_frame, textvariable=self._offset_var).grid(
                    row=1, column=3
                )
                self._tk.Label(_button_frame, text="Spacer between bounded zones").grid(
                    row=1, column=4
                )
                self._tk.Entry(_button_frame, textvariable=self._spacer_var).grid(
                    row=1, column=5
                )

            region_sel_row_ind = (
                2 if num_vertices == FigureAnnotatorType.QUADRILATERAL.value else 1
            )
            self._tk.Label(_button_frame, text="Current Region").grid(
                row=region_sel_row_ind, column=0
            )
            self._regions_drop = OptionMenu(
                _button_frame,
                self._current_region_idx,
                *self._current_region_idx_options,
            )
            self._regions_drop.grid(row=region_sel_row_ind, column=1)

            self._master.mainloop()

    def _load_image(self):
        """
        Function bound to 'Load Image' button.
        Load an image, and intialize first region for annotation.
        """
        self.image_path = filedialog.askopenfilename(
            filetypes=[("Image Files", ["*.png", "*.jpg", "*.jpeg"])]
        )
        if not self.image_path:
            return

        self.image = Image.open(self.image_path)
        self.img_width, self.img_height = self.image.size  # type: ignore[attr-defined]

        # Check if canvas is already initialized
        if self._canvas:
            self._canvas.destroy()  # Destroy previous canvas

        # Create canvas
        self._canvas = self._tk.Canvas(
            self._master, bg="white", width=self.img_width, height=self.img_height
        )
        self.canvas_image = ImageTk.PhotoImage(self.image)
        self._canvas.create_image(0, 0, anchor=self._tk.NW, image=self.canvas_image)

        self._canvas.pack(side=self._tk.BOTTOM)
        self._canvas.bind("<Button-1>", self.on_canvas_click)

        # Reset regions and regions JSON
        auto_mode = (
            messagebox.askyesno(
                "Selection Mode", "Add zones to region in automatic mode?"
            )
            if self.num_vertices == FigureAnnotatorType.QUADRILATERAL.value
            else False
        )
        self.regions = [Region(auto_mode)]
        self._update_region_menu()
        self.regions_json = []

        # Notify user to start by selecting appropriate points, depending on the mode for the region.
        if auto_mode:
            message = "Please select two right anchor points."
        else:
            message = (
                f"Please select {self.num_vertices} points of a {self.figure_type}."
            )
        messagebox.showinfo("Info", message)

    def _update_region_menu(self):
        """Update the dropdown menu listing all of the present regions."""
        self._current_region_idx_options = [str(v) for v in range(len(self.regions))]
        self._regions_drop["menu"].delete(0, len(self._current_region_idx_options))
        for opt in self._current_region_idx_options:
            self._regions_drop["menu"].add_command(
                label=opt, command=self._tk._setit(self._current_region_idx, opt)
            )
        self._current_region_idx.set(self._current_region_idx_options[-1])

    def _add_region(self):
        """
        Function bound to 'Add Region' button.
        Add new figure region to canvas.
        """
        if len(self.regions) == 1 and self.regions[0].is_empty():
            if self.regions[0].auto_mode:
                message = "Please select two right anchor points."
            else:
                message = (
                    f"Please select {self.num_vertices} points of a {self.figure_type}."
                )
            messagebox.showwarning(
                "Warning",
                "No regions exist yet to add to. " + message,
            )
        else:
            auto_mode = (
                messagebox.askyesno(
                    "Selection Mode", "Add zones to region in automatic mode?"
                )
                if self.num_vertices == FigureAnnotatorType.QUADRILATERAL.value
                else False
            )
            self.regions.append(Region(auto_mode))
            self._update_region_menu()
            if auto_mode:
                message = (
                    "New zone region added. Please select two right anchor points."
                )
            else:
                message = (
                    f"New {self.figure_type} region added. "
                    + f"Please select {self.num_vertices} points of a {self.figure_type}."
                )
            messagebox.showinfo(
                "Success",
                message,
            )

    def draw_point(self, point):
        """Draw point on canvas."""
        x0, y0 = point[0] - 3, point[1] - 3
        x1, y1 = point[0] + 3, point[1] + 3
        self._canvas.create_oval(x0, y0, x1, y1, fill="red")

    def _process_point_addition(self, current_region: Region) -> Tuple[str, str]:
        """
        Add user-selected point to appropriate region field.

        Args:
            current_region (Region): Region object to which the point was added

        Returns:
            Tuple[str, str]: title and message for dialog message box,
                empty strings if no message should be displayed
        """
        title = ""
        message = ""
        if current_region.auto_mode:
            if current_region.point_selection_state == PointSelState.RightAnchors:
                if len(current_region.current_selection) == 2:
                    current_region.anchor_points.append(
                        current_region.current_selection
                    )
                    current_region.current_selection = []
                    current_region.point_selection_state = PointSelState.LeftAnchors
                    title = "Info"
                    message = "Please select two left anchor points."
                elif (
                    len(current_region.anchor_points) == 1
                    and len(current_region.current_selection) == 1
                ):
                    current_region.anchor_points[0].append(
                        current_region.current_selection[0]
                    )
                    current_region.current_selection = []
                    current_region.point_selection_state = PointSelState.LeftAnchors
                    title = "Info"
                    message = "Please select two left anchor points."

            elif current_region.point_selection_state == PointSelState.LeftAnchors:
                if len(current_region.current_selection) == 2:
                    current_region.anchor_points.append(
                        current_region.current_selection
                    )
                    current_region.current_selection = []
                    current_region.point_selection_state = (
                        PointSelState.IntermediateAnchors
                    )
                    title = "Info"
                    message = "Please select intermediate boundary points, going from right to left."
                elif (
                    len(current_region.anchor_points) == 2
                    and len(current_region.current_selection) == 1
                ):
                    current_region.anchor_points[1].append(
                        current_region.current_selection[0]
                    )
                    current_region.current_selection = []
                    current_region.point_selection_state = (
                        PointSelState.IntermediateAnchors
                    )
                    title = "Info"
                    message = "Please select intermediate boundary points, going from right to left."

            elif (
                current_region.point_selection_state
                == PointSelState.IntermediateAnchors
            ):
                if len(current_region.current_selection) == 2:
                    current_region.intermediate_points.append(
                        current_region.current_selection
                    )
                    current_region.current_selection = []
                elif (
                    len(current_region.current_selection) == 1
                    and len(current_region.intermediate_points) > 0
                    and len(current_region.intermediate_points[-1]) == 1
                ):
                    current_region.intermediate_points[-1].append(
                        current_region.current_selection[0]
                    )
                    current_region.current_selection = []
        else:
            if len(current_region.current_selection) == self.num_vertices:
                current_region.figures.append(current_region.current_selection)
                current_region.current_selection = []
                title = "Success"
                message = f"{self.figure_type.capitalize()} added."

        return title, message

    def on_canvas_click(self, event):
        """Handle mouse clicks on canvas to create points for bounding boxes."""
        current_region = self.regions[self.get_current_region_idx()]
        current_region.current_selection.append((event.x, event.y))
        self._redraw_selections()
        title, message = self._process_point_addition(current_region)
        if title:
            messagebox.showinfo(title=title, message=message)

    def get_num_zones_between_barriers(self):
        """Return number of zones between barriers."""
        return int(self._num_zones_between_barriers.get())

    def get_current_region_idx(self):
        """Return index of current region."""
        return int(self._current_region_idx.get())

    def get_save_format(self):
        """Return string indicating format for saving figures."""
        return self._save_format.get()

    @staticmethod
    def region_parameters(region: Region) -> Tuple[float, float, float, float]:
        """
        Return calculated key defining parameters of region:
        the slopes and y-intercepts of the top and bottom lines of the regions.

        Args:
            region (Region): region object

        Returns:
            Tuple[float, float, float, float]: slope of top line, y-intercept of
                top line, slope of bottom line, y-intercept of bottom line
        """
        guide_top_1 = region.anchor_points[0][0]
        guide_top_2 = region.anchor_points[1][0]
        guide_bot_1 = region.anchor_points[0][1]
        guide_bot_2 = region.anchor_points[1][1]
        m_top = (guide_top_1[1] - guide_top_2[1]) / (guide_top_1[0] - guide_top_2[0])
        b_top = guide_top_1[1] - m_top * guide_top_1[0]
        m_bot = (guide_bot_1[1] - guide_bot_2[1]) / (guide_bot_1[0] - guide_bot_2[0])
        b_bot = guide_bot_1[1] - m_bot * guide_bot_1[0]
        return m_top, b_top, m_bot, b_bot

    @staticmethod
    def lin_func(m, b, x):
        """Return y-coordinate for a given x-coordinate based on line defined by slope and y-intercept."""
        return m * x + b

    def _check_regions_completeness(self) -> str:
        """
        Check the completeness of each region, and return a warning about incomplete regions if such
        regions are found.

        Returns:
            str: warning message about incomplete regions if such exist, empty string otherwise
        """
        message = ""
        incomplete_auto_region_idx = ""
        incomplete_manual_region_idx = ""
        for i in range(len(self.regions)):
            region = self.regions[i]
            if region.auto_mode and (
                region.is_empty()
                or len(region.current_selection) > 0
                or len(region.anchor_points) < 2
                or len(region.anchor_points[1]) < 2
                or (
                    len(region.intermediate_points) > 0
                    and len(region.intermediate_points[-1]) < 2
                )
            ):
                incomplete_auto_region_idx += str(i)
            elif not region.auto_mode and (
                region.is_empty() or len(region.current_selection) > 0
            ):
                incomplete_manual_region_idx += str(i)

        if len(incomplete_auto_region_idx) > 0:
            message = (
                "Cannot compute zones because not enough points are specified in\n\n"
                + "  - automatic-mode region(s) "
                + ", ".join(incomplete_auto_region_idx)
            )
        if len(incomplete_manual_region_idx) > 0:
            if not message:
                message = f"Cannot compute {self.figure_type}s because not enough points are specified in\n"
            message += (
                "\n  - "
                + (
                    "manual-mode "
                    if self.num_vertices == FigureAnnotatorType.QUADRILATERAL.value
                    else ""
                )
                + "region(s) "
                + ", ".join(incomplete_manual_region_idx)
            )

        return message

    @staticmethod
    def compute_figures(
        regions: List[Region],
        num_zones_between_barriers: int = 1,
        offset: int = 0,
        spacer: int = 0,
    ) -> List[Dict[str, List[Dict[str, List[tuple]]]]]:
        """
        Compute zones based on user-defined anchor points for regions in automatic
        mode, and return a list of dictionaries, each corresponding to a region.

        Args:
            regions (List[Region]): list of region objects
            num_zones_between_barriers (int): number of zones between two adjacent
                boundaries, possible values are 1 and 2; used for regions in auto mode
            offset (int): distance between a boundary line and parallel zone edge near
                this boundary line; used for regions in auto mode
            spacer (int): used when num_zones_between_barriers = 2; distance between
                two zones that are between adjacent boundary lines; used for regions
                in auto mode

        Returns:
            List[Dict[str, List[Dict[str, List[tuple]]]]]: list of dictionaries, each
                corresponding to a region
        """
        regions_json: List[Dict[str, List[Dict[str, List[tuple]]]]] = []
        for region in regions:
            regions_json.append({"figures": []})
            figures_list = regions_json[-1]["figures"]
            if region.auto_mode:
                ref_points = region.intermediate_points.copy()

                m_top, b_top, m_bot, b_bot = FigureAnnotator.region_parameters(region)

                ref_points.insert(0, region.anchor_points[0])
                ref_points.append(region.anchor_points[1])

                for i in range(len(ref_points) - 1):
                    ref_top = ref_points[i][0]
                    ref_bot = ref_points[i][1]
                    next_ref_top = ref_points[i + 1][0]
                    next_ref_bot = ref_points[i + 1][1]

                    if num_zones_between_barriers == 1:
                        tr_x = ref_top[0] - offset
                        tr_y = FigureAnnotator.lin_func(m_top, b_top, tr_x)
                        br_x = ref_bot[0] - offset
                        br_y = FigureAnnotator.lin_func(m_bot, b_bot, br_x)
                        tl_x = next_ref_top[0] + offset
                        tl_y = FigureAnnotator.lin_func(m_top, b_top, tl_x)
                        bl_x = next_ref_bot[0] + offset
                        bl_y = FigureAnnotator.lin_func(m_bot, b_bot, bl_x)
                        figures_list.append(
                            {
                                "points": [
                                    (tr_x, tr_y),
                                    (br_x, br_y),
                                    (bl_x, bl_y),
                                    (tl_x, tl_y),
                                ]
                            }
                        )
                    elif num_zones_between_barriers == 2:
                        top_width = (
                            ref_top[0] - next_ref_top[0] - spacer
                        ) // 2 - offset
                        bot_width = (
                            ref_bot[0] - next_ref_bot[0] - spacer
                        ) // 2 - offset

                        tr_x = ref_top[0] - offset
                        tr_y = FigureAnnotator.lin_func(m_top, b_top, tr_x)
                        tl_x = tr_x - top_width
                        tl_y = FigureAnnotator.lin_func(m_top, b_top, tl_x)

                        br_x = ref_bot[0] - offset
                        br_y = FigureAnnotator.lin_func(m_bot, b_bot, br_x)
                        bl_x = br_x - bot_width
                        bl_y = FigureAnnotator.lin_func(m_bot, b_bot, bl_x)

                        figures_list.append(
                            {
                                "points": [
                                    (tr_x, tr_y),
                                    (br_x, br_y),
                                    (bl_x, bl_y),
                                    (tl_x, tl_y),
                                ]
                            }
                        )  # right zone

                        next_tl_x = next_ref_top[0] + offset
                        next_tl_y = FigureAnnotator.lin_func(m_top, b_top, next_tl_x)
                        next_tr_x = next_tl_x + top_width
                        next_tr_y = FigureAnnotator.lin_func(m_top, b_top, next_tr_x)

                        next_bl_x = next_ref_bot[0] + offset
                        next_bl_y = FigureAnnotator.lin_func(m_bot, b_bot, next_bl_x)
                        next_br_x = next_bl_x + bot_width
                        next_br_y = FigureAnnotator.lin_func(m_bot, b_bot, next_br_x)

                        figures_list.append(
                            {
                                "points": [
                                    (next_tr_x, next_tr_y),
                                    (next_br_x, next_br_y),
                                    (next_bl_x, next_bl_y),
                                    (next_tl_x, next_tl_y),
                                ]
                            }
                        )  # left zone
            else:
                for figure in region.figures:
                    figures_list.append({"points": figure})

        return regions_json

    def _compute_and_draw_figures(self):
        """
        Function bound to 'Compute' button.
        Compute zones for all regions in auto mode based on anchor points of each region,
        and include manually-defined region figures, for the output figure JSON. Draw all
        figures on canvas.
        """
        message = self._check_regions_completeness()
        if message:
            messagebox.showwarning(title="Warning", message=message)
            return

        spacer = self._spacer_var.get()
        offset = self._offset_var.get()
        num_zones_between_barriers = self.get_num_zones_between_barriers()

        self.regions_json = FigureAnnotator.compute_figures(
            self.regions, num_zones_between_barriers, offset, spacer
        )

        self._draw_figures()

    def _draw_figures(self):
        """Draw all figures for all regions on canvas."""

        # Draw points
        self._redraw_selections()

        # Draw lines connecting the points
        for region in self.regions_json:
            for figure in region["figures"]:
                points = figure["points"]
                for i in range(self.num_vertices):
                    x1, y1 = points[i]
                    x2, y2 = points[(i + 1) % self.num_vertices]
                    self._canvas.create_line(x1, y1, x2, y2, fill="blue", width=2)

    def _redraw_selections(self):
        """Re-draw all selected points for all regions on canvas."""

        # Clear the canvas
        self._canvas.delete("all")

        # Redraw the image
        self._canvas.create_image(0, 0, anchor=self._tk.NW, image=self.canvas_image)

        # Redraw all points
        for region in self.regions:
            if region.auto_mode:
                for edge in region.anchor_points:
                    for p in edge:
                        self.draw_point(p)
                for edge in region.intermediate_points:
                    for p in edge:
                        self.draw_point(p)
            else:
                for figure in region.figures:
                    for p in figure:
                        self.draw_point(p)
            for p in region.current_selection:
                self.draw_point(p)

    def _process_point_deletion(self, current_region_idx: int) -> Optional[tuple]:
        """
        Remove last user-selected point from appropriate region field.

        Args:
            current_region_idx (int): index of region in region list

        Returns:
            Optional[tuple]: removed point tuple
        """
        current_region = self.regions[current_region_idx]
        removed_point: Optional[tuple] = None

        if current_region.auto_mode:
            if current_region.point_selection_state == PointSelState.RightAnchors:
                if len(current_region.current_selection) > 0:
                    removed_point = current_region.current_selection.pop()
                    # messagebox.showinfo("Success", "Last right anchor point removed.")
                elif len(current_region.anchor_points) > 0:
                    removed_point = current_region.anchor_points[0].pop()
                    current_region.anchor_points.pop()
                    # messagebox.showinfo("Success", "Last right anchor point removed.")
            elif current_region.point_selection_state == PointSelState.LeftAnchors:
                if len(current_region.current_selection) > 0:
                    removed_point = current_region.current_selection.pop()
                    # message = "Last left anchor point removed."
                elif len(current_region.anchor_points) == 2:
                    removed_point = current_region.anchor_points[1].pop()
                    current_region.anchor_points.pop()
                    # message = "Last left anchor point removed."
                else:
                    removed_point = current_region.anchor_points[0].pop()
                    current_region.point_selection_state = PointSelState.RightAnchors
                    # message = "Last right anchor point removed."
                # messagebox.showinfo("Success", message)
            else:
                if len(current_region.current_selection) > 0:
                    removed_point = current_region.current_selection.pop()
                    # message = "Last intermediate anchor point removed."
                elif len(current_region.intermediate_points) > 0:
                    removed_point = current_region.intermediate_points[-1].pop()
                    if len(current_region.intermediate_points[-1]) == 0:
                        current_region.intermediate_points.pop()
                    # message = "Last intermediate anchor point removed."
                else:
                    removed_point = current_region.anchor_points[1].pop()
                    current_region.point_selection_state = PointSelState.LeftAnchors
                    # message = "Last left anchor point removed."
                # messagebox.showinfo("Success", message)
        else:
            if len(current_region.current_selection) > 0:
                removed_point = current_region.current_selection.pop()
                # messagebox.showinfo("Success", "Last corner point removed.")
            elif len(current_region.figures) > 0:
                removed_point = current_region.figures[-1].pop()
                if len(current_region.figures[-1]) > 0:
                    current_region.current_selection.extend(
                        deepcopy(current_region.figures[-1])
                    )
                current_region.figures.pop()
                # messagebox.showinfo("Success", "Last corner point removed.")

        return removed_point

    def _remove_last_selection(self):
        """
        Function bound to 'Remove Last Selection' button.
        Remove the last drawn selection for a region from canvas.
        """

        # Remove last selected point.
        current_region_idx = self.get_current_region_idx()
        removed_point = self._process_point_deletion(current_region_idx)
        if removed_point is None:
            messagebox.showwarning("Warning", "No points to remove.")
            return

        # Re-draw points
        self._redraw_selections()

        # Update regions if necessary
        if self.regions[current_region_idx].is_empty():
            if len(self.regions) > 1:
                self.regions.pop(current_region_idx)
                self._update_region_menu()
                messagebox.showinfo("Success", "Last region removed.")
            else:
                self.regions_json = []

    @staticmethod
    def convert_regions_to_json(
        regions_json: List[Dict[str, List[Dict[str, List[tuple]]]]],
        width_scaling_factor: float,
        height_scaling_factor: float,
        save_format: str,
    ) -> List[Union[List[tuple], Dict[str, List[tuple]]]]:
        """
        Convert internal list of regions into a JSON-serializable list
        of figures in a specified format.

        Args:
            regions_json (List[Dict[str, List[Dict[str, List[tuple]]]]]):
                list of dictionaries, each corresponding to a region
            width_scaling_factor (float): scale factor for x-coordinates
            height_scaling_factor (float): scale factor for y-coordinates
            save_format (str): saving format type; supported values are
                'DeGirum-compatible', 'Ultralytics-compatible'

        Returns:
            List[Union[List[tuple], Dict[str, List[tuple]]]]:
                JSON-serializable list of figures in specified format
        """
        figures_data: List[Union[List[tuple], Dict[str, List[tuple]]]] = []
        for region in regions_json:
            figures_list = region["figures"]
            for figure in figures_list:
                box = figure["points"]
                rescaled_box = []
                for x, y in box:
                    rescaled_x = int(x * width_scaling_factor)
                    rescaled_y = int(y * height_scaling_factor)
                    rescaled_box.append((rescaled_x, rescaled_y))
                figures_data.append(
                    rescaled_box
                    if save_format == "DeGirum-compatible"
                    else {"points": rescaled_box}
                )

        return figures_data

    def _save_to_json(self):
        """
        Function bound to 'Save' button.
        Saves rescaled figures based on image-to-canvas size ratio to a file.
        """
        if len(self.regions_json) == 0:
            messagebox.showwarning("Warning", f"No {self.figure_type}s to save.")
        else:
            save_format = self.get_save_format()
            canvas_width, canvas_height = (
                self._canvas.winfo_width(),
                self._canvas.winfo_height(),
            )
            width_scaling_factor = self.img_width / canvas_width
            height_scaling_factor = self.img_height / canvas_height
            results_file_path = filedialog.asksaveasfilename(
                initialfile=self.results_file_name, filetypes=[("JSON file", "*.json")]
            )
            figures_data = FigureAnnotator.convert_regions_to_json(
                self.regions_json,
                width_scaling_factor,
                height_scaling_factor,
                save_format,
            )
            if len(results_file_path) > 0:
                with open(results_file_path, "w") as json_file:
                    json.dump(figures_data, json_file, indent=4)

                messagebox.showinfo(
                    "Success",
                    f"{self.figure_type.capitalize()}s saved as {results_file_path.split('/')[-1]}",
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=f"{__file__}",
        description="Launch interactive utility for geometric figure annotation in images",
    )
    _figure_annotator_args(parser)
    _figure_annotator_run(parser.parse_args())
