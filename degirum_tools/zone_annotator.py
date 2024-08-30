import argparse
import json
from enum import Enum
from tkinter import filedialog, messagebox, OptionMenu

from PIL import Image, ImageTk
from copy import deepcopy
from typing import List


def _zone_annotator_run(results_file_name):
    ZoneAnnotator(results_file_name=results_file_name)


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


class _PointSelState(Enum):
    RightAnchors = 1
    LeftAnchors = 2
    IntermediateAnchors = 3


class _Region:
    def __init__(self, auto_mode: bool = True):
        self.auto_mode = auto_mode
        self.current_selection: List[tuple] = []
        if auto_mode:
            self.anchor_points: List[List[tuple]] = []
            self.intermediate_points: List[List[tuple]] = []
            self.point_selection_state = _PointSelState.RightAnchors
        else:
            self.zones: List[List[tuple]] = []


class ZoneAnnotator:
    def __init__(self, results_file_name: str = "zones.json"):
        """Initializes the UI for selecting zones in a tkinter window."""

        import tkinter as tk

        self.tk = tk
        self.master = tk.Tk()
        self.master.title("Zones Selection")

        # Disable window resizing
        self.master.resizable(False, False)

        # Setup canvas for image display
        self.canvas = self.tk.Canvas(self.master, bg="white")

        # Define variables for entries
        self.offset_var = self.tk.IntVar(value=1)
        self.spacer_var = self.tk.IntVar(value=1)
        self.num_zones_between_barriers = self.tk.StringVar(value="1")
        self.num_zones_between_barriers_options = ["1", "2"]
        self.current_region_idx = self.tk.StringVar(value="0")
        self.current_region_idx_options = ["0"]

        # Setup buttons
        button_frame = self.tk.Frame(self.master)
        button_frame.pack(side=self.tk.TOP)

        self.tk.Button(
            button_frame, text="Upload Image", command=self.upload_image
        ).grid(row=0, column=0)
        self.tk.Button(
            button_frame,
            text="Remove Last Selection",
            command=self.remove_last_selection,
        ).grid(row=0, column=1)
        self.tk.Button(
            button_frame, text="Compute Zones", command=self.compute_zones
        ).grid(row=0, column=2)
        self.tk.Button(
            button_frame, text="Add Zone Region", command=self.add_zone_region
        ).grid(row=0, column=3)
        self.tk.Button(button_frame, text="Save", command=self.save_to_json).grid(
            row=0, column=4
        )
        self.tk.Label(button_frame, text="Offset from boundary point").grid(
            row=1, column=0
        )
        self.tk.Entry(button_frame, textvariable=self.offset_var).grid(row=1, column=1)
        self.tk.Label(button_frame, text="Spacer between bounded zones").grid(
            row=1, column=2
        )
        self.tk.Entry(button_frame, textvariable=self.spacer_var).grid(row=1, column=3)
        self.tk.Label(button_frame, text="Number of zones between barriers").grid(
            row=1, column=4
        )
        self.num_zones_between_barriers_drop = OptionMenu(
            button_frame,
            self.num_zones_between_barriers,
            *self.num_zones_between_barriers_options,
        )
        self.num_zones_between_barriers_drop.grid(row=1, column=5)
        self.tk.Label(button_frame, text="Current Region").grid(row=2, column=0)
        self.regions_drop = OptionMenu(
            button_frame, self.current_region_idx, *self.current_region_idx_options
        )
        self.regions_drop.grid(row=2, column=1)

        # Initialize properties
        self.image_path = ""
        self.image = None
        self.canvas_image = None
        self.regions_json: List[dict] = []
        self.img_width = 0
        self.img_height = 0
        self.regions = [_Region()]
        self.results_file_name = results_file_name

        # Constants
        self.canvas_max_width = 1280
        self.canvas_max_height = 720

        self.master.mainloop()

    def upload_image(self):
        """Upload an image and resize it to fit canvas."""
        self.image_path = filedialog.askopenfilename(
            filetypes=[("Image Files", ["*.png", "*.jpg", "*.jpeg"])]
        )
        if not self.image_path:
            return

        self.image = Image.open(self.image_path)
        self.img_width, self.img_height = self.image.size  # type: ignore[attr-defined]

        # Calculate the aspect ratio and resize image
        aspect_ratio = self.img_width / self.img_height
        if aspect_ratio > 1:
            # Landscape orientation
            canvas_width = min(self.canvas_max_width, self.img_width)
            canvas_height = int(canvas_width / aspect_ratio)
        else:
            # Portrait orientation
            canvas_height = min(self.canvas_max_height, self.img_height)
            canvas_width = int(canvas_height * aspect_ratio)

        # Check if canvas is already initialized
        if self.canvas:
            self.canvas.destroy()  # Destroy previous canvas

        self.canvas = self.tk.Canvas(
            self.master, bg="white", width=canvas_width, height=canvas_height
        )
        resized_image = self.image.resize((canvas_width, canvas_height), Image.LANCZOS)  # type: ignore[attr-defined]
        self.canvas_image = ImageTk.PhotoImage(resized_image)
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)

        self.canvas.pack(side=self.tk.BOTTOM)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Reset regions and zones JSON
        auto_mode = messagebox.askyesno(
            "Selection Mode", "Add zones to region in automatic mode?"
        )
        self.regions = [_Region(auto_mode)]
        self.update_regions()
        self.regions_json = []

        # Notify user to start by selecting appropriate points, depending on the mode for the region.
        if auto_mode:
            messagebox.showinfo("Info", "Please select two right anchor points.")
        else:
            messagebox.showinfo("Info", "Please select four corner points of a zone.")

    def update_regions(self):
        """Update the dropdown menu listing all of the present regions."""
        self.current_region_idx_options = [str(v) for v in range(len(self.regions))]
        self.regions_drop["menu"].delete(0, len(self.current_region_idx_options))
        for opt in self.current_region_idx_options:
            self.regions_drop["menu"].add_command(
                label=opt, command=self.tk._setit(self.current_region_idx, opt)
            )
        self.current_region_idx.set(self.current_region_idx_options[-1])

    def add_zone_region(self):
        """Add new zone region to canvas."""
        if (
            len(self.regions) == 1
            and self.regions[0].auto_mode
            and len(self.regions[0].anchor_points) == 0
        ):
            messagebox.showwarning(
                "Warning",
                "No regions exist yet to add to. Please select two right anchor points.",
            )
        else:
            auto_mode = messagebox.askyesno(
                "Selection Mode", "Add zones to region in automatic mode?"
            )
            self.regions.append(_Region(auto_mode))
            self.update_regions()
            if auto_mode:
                messagebox.showinfo(
                    "Success",
                    "New zone region added. Please select two right anchor points.",
                )
            else:
                messagebox.showinfo(
                    "Success",
                    "New zone region added. Please select four corner points of a zone.",
                )

    def draw_point(self, point):
        """Draw point on canvas."""
        x0, y0 = point[0] - 3, point[1] - 3
        x1, y1 = point[0] + 3, point[1] + 3
        self.canvas.create_oval(x0, y0, x1, y1, fill="red")

    def on_canvas_click(self, event):
        """Handle mouse clicks on canvas to create points for bounding boxes."""
        current_region = self.regions[self.get_current_region_idx()]
        current_region.current_selection.append((event.x, event.y))
        self.redraw_selections()
        region_in_auto_mode = current_region.auto_mode

        if region_in_auto_mode:
            if current_region.point_selection_state == _PointSelState.RightAnchors:
                if len(current_region.current_selection) == 2:
                    current_region.anchor_points.append(
                        current_region.current_selection
                    )
                    current_region.current_selection = []
                    current_region.point_selection_state = _PointSelState.LeftAnchors
                    messagebox.showinfo("Info", "Please select two left anchor points.")
                elif (
                    len(current_region.anchor_points) > 0
                    and len(current_region.current_selection) == 1
                ):
                    current_region.anchor_points[0].append(
                        current_region.current_selection[0]
                    )
                    current_region.current_selection = []
                    current_region.point_selection_state = _PointSelState.LeftAnchors
                    messagebox.showinfo("Info", "Please select two left anchor points.")

            elif current_region.point_selection_state == _PointSelState.LeftAnchors:
                if len(current_region.current_selection) == 2:
                    current_region.anchor_points.append(
                        current_region.current_selection
                    )
                    current_region.current_selection = []
                    current_region.point_selection_state = (
                        _PointSelState.IntermediateAnchors
                    )
                    messagebox.showinfo(
                        "Info",
                        "Please select the intermediate boundary points, going from right to left.",
                    )
                elif (
                    len(current_region.anchor_points) == 2
                    and len(current_region.current_selection) == 1
                ):
                    current_region.anchor_points[1].append(
                        current_region.current_selection[0]
                    )
                    current_region.current_selection = []
                    current_region.point_selection_state = (
                        _PointSelState.IntermediateAnchors
                    )
                    messagebox.showinfo(
                        "Info",
                        "Please select the intermediate boundary points, going from right to left.",
                    )

            elif (
                current_region.point_selection_state
                == _PointSelState.IntermediateAnchors
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
            if len(current_region.current_selection) == 4:
                current_region.zones.append(current_region.current_selection)
                current_region.current_selection = []
                messagebox.showinfo("Success", "Zone added.")

    def get_num_zones_between_barriers(self):
        """Return number of zones between barriers."""
        return int(self.num_zones_between_barriers.get())

    def get_current_region_idx(self):
        """Return index of current region."""
        return int(self.current_region_idx.get())

    def region_parameters(self, region):
        """
        Return calculated key defining parameters of region:
        the slopes and y-intercepts of the top and bottom lines of the regions.
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

    def lin_func(self, m, b, x):
        """Return y-coordinate for a given x-coordinate based on line defined by slope and y-intercept."""
        return m * x + b

    def compute_zones(self):
        """
        Compute zones for all automatic regions based on anchor points of each region,
        and add manually-defined region zones, for the output zone JSON.
        """
        incomplete_auto_region_idx = ""
        incomplete_manual_region_idx = ""
        for i in range(len(self.regions)):
            region = self.regions[i]
            if region.auto_mode and (
                len(region.anchor_points) < 2
                or len(region.anchor_points[1]) < 2
                or (
                    len(region.intermediate_points) > 0
                    and len(region.intermediate_points[-1]) < 2
                )
                or len(region.current_selection) > 0
            ):
                incomplete_auto_region_idx += str(i)
            elif not region.auto_mode and (
                (len(region.zones) > 0 and len(region.zones[-1]) != 4)
                or len(region.current_selection) > 0
            ):
                incomplete_manual_region_idx += str(i)
        incomplete_auto_regions_present = len(incomplete_auto_region_idx) > 0
        incomplete_manual_regions_present = len(incomplete_manual_region_idx) > 0
        if incomplete_auto_regions_present or incomplete_manual_regions_present:
            if incomplete_auto_regions_present:
                messagebox.showwarning(
                    "Warning",
                    f"Cannot compute zones because not enough anchor points are specified in automatic-mode region(s) {', '.join(incomplete_auto_region_idx)}.",
                )
            if incomplete_manual_regions_present:
                messagebox.showwarning(
                    "Warning",
                    f"Cannot compute zones because not enough corner points for zones are specified in manual-mode region(s) {', '.join(incomplete_manual_region_idx)}.",
                )
            return

        self.regions_json = []
        spacer = self.spacer_var.get()
        offset = self.offset_var.get()
        num_zones_between_barriers = self.get_num_zones_between_barriers()

        for region in self.regions:
            self.regions_json.append({"zones": []})
            zones_list = self.regions_json[-1]["zones"]
            if region.auto_mode:
                ref_points = region.intermediate_points.copy()

                m_top, b_top, m_bot, b_bot = self.region_parameters(region)

                ref_points.insert(0, region.anchor_points[0])
                ref_points.append(region.anchor_points[1])

                for i in range(len(ref_points) - 1):
                    ref_top = ref_points[i][0]
                    ref_bot = ref_points[i][1]
                    next_ref_top = ref_points[i + 1][0]
                    next_ref_bot = ref_points[i + 1][1]

                    if num_zones_between_barriers == 1:
                        tr_x = ref_top[0] - offset
                        tr_y = self.lin_func(m_top, b_top, tr_x)
                        br_x = ref_bot[0] - offset
                        br_y = self.lin_func(m_bot, b_bot, br_x)
                        tl_x = next_ref_top[0] + offset
                        tl_y = self.lin_func(m_top, b_top, tl_x)
                        bl_x = next_ref_bot[0] + offset
                        bl_y = self.lin_func(m_bot, b_bot, bl_x)
                        zones_list.append(
                            {
                                "points": [
                                    [tr_x, tr_y],
                                    [br_x, br_y],
                                    [bl_x, bl_y],
                                    [tl_x, tl_y],
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
                        tr_y = self.lin_func(m_top, b_top, tr_x)
                        tl_x = tr_x - top_width
                        tl_y = self.lin_func(m_top, b_top, tl_x)

                        br_x = ref_bot[0] - offset
                        br_y = self.lin_func(m_bot, b_bot, br_x)
                        bl_x = br_x - bot_width
                        bl_y = self.lin_func(m_bot, b_bot, bl_x)

                        zones_list.append(
                            {
                                "points": [
                                    [tr_x, tr_y],
                                    [br_x, br_y],
                                    [bl_x, bl_y],
                                    [tl_x, tl_y],
                                ]
                            }
                        )  # right zone

                        next_tl_x = next_ref_top[0] + offset
                        next_tl_y = self.lin_func(m_top, b_top, next_tl_x)
                        next_tr_x = next_tl_x + top_width
                        next_tr_y = self.lin_func(m_top, b_top, next_tr_x)

                        next_bl_x = next_ref_bot[0] + offset
                        next_bl_y = self.lin_func(m_bot, b_bot, next_bl_x)
                        next_br_x = next_bl_x + bot_width
                        next_br_y = self.lin_func(m_bot, b_bot, next_br_x)

                        zones_list.append(
                            {
                                "points": [
                                    [next_tr_x, next_tr_y],
                                    [next_br_x, next_br_y],
                                    [next_bl_x, next_bl_y],
                                    [next_tl_x, next_tl_y],
                                ]
                            }
                        )  # left zone
            else:
                for zone in region.zones:
                    zones_list.append({"points": zone})
        self.draw_zones()

    def draw_zones(self):
        """Draw all zones for all regions on canvas."""
        self.redraw_selections()
        for region in self.regions_json:
            for zone in region["zones"]:
                points = zone["points"]
                for i in range(4):
                    x1, y1 = points[i]
                    x2, y2 = points[(i + 1) % 4]
                    self.canvas.create_line(x1, y1, x2, y2, fill="blue", width=2)

    def redraw_selections(self):
        """Re-draw all selected points for all regions on canvas."""
        self.canvas.delete("all")  # Clear the canvas
        self.canvas.create_image(
            0, 0, anchor=self.tk.NW, image=self.canvas_image
        )  # Redraw the image

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
                for zone in region.zones:
                    for p in zone:
                        self.draw_point(p)
            for p in region.current_selection:
                self.draw_point(p)

    def remove_last_selection(self):
        """Remove the last drawn selection for a region from canvas."""
        current_region_idx = self.get_current_region_idx()
        current_region = self.regions[current_region_idx]
        if current_region.auto_mode:
            if current_region.point_selection_state == _PointSelState.RightAnchors:
                if len(current_region.current_selection) > 0:
                    current_region.current_selection.pop()
                    self.redraw_selections()
                    # messagebox.showinfo("Success", "Last right anchor point removed.")
                elif len(current_region.anchor_points) > 0:
                    current_region.anchor_points.pop()
                    self.redraw_selections()
                    # messagebox.showinfo("Success", "Last right anchor point removed.")
                    if len(self.regions) > 1:
                        self.regions.pop(current_region_idx)
                        self.update_regions()
                        messagebox.showinfo("Success", "Last region removed.")
                else:
                    messagebox.showwarning("Warning", "No points to remove.")
            elif current_region.point_selection_state == _PointSelState.LeftAnchors:
                if len(current_region.current_selection) > 0:
                    current_region.current_selection.pop()
                    # message = "Last left anchor point removed."
                elif len(current_region.anchor_points) == 2:
                    current_region.anchor_points.pop()
                    # message = "Last left anchor point removed."
                else:
                    current_region.anchor_points[0].pop()
                    current_region.point_selection_state = _PointSelState.RightAnchors
                    # message = "Last right anchor point removed."
                self.redraw_selections()
                # messagebox.showinfo("Success", message)
            else:
                if len(current_region.current_selection) > 0:
                    current_region.current_selection.pop()
                    # message = "Last intermediate anchor point removed."
                elif len(current_region.intermediate_points) > 0:
                    current_region.intermediate_points[-1].pop()
                    if len(current_region.intermediate_points[-1]) == 0:
                        current_region.intermediate_points.pop()
                    # message = "Last intermediate anchor point removed."
                else:
                    current_region.anchor_points[1].pop()
                    current_region.point_selection_state = _PointSelState.LeftAnchors
                    # message = "Last left anchor point removed."
                self.redraw_selections()
                # messagebox.showinfo("Success", message)
        else:
            if len(current_region.current_selection) > 0:
                current_region.current_selection.pop()
                self.redraw_selections()
                # messagebox.showinfo("Success", "Last corner point removed.")
                if (
                    len(current_region.current_selection) == 0
                    and len(current_region.zones) == 0
                    and len(self.regions) > 1
                ):
                    self.regions.pop(current_region_idx)
                    self.update_regions()
                    messagebox.showinfo("Success", "Last region removed.")
            elif len(current_region.zones) > 0:
                current_region.zones[-1].pop()
                if len(current_region.zones[-1]) > 0:
                    current_region.current_selection.extend(
                        deepcopy(current_region.zones[-1])
                    )
                current_region.zones.pop()
                self.redraw_selections()
                # messagebox.showinfo("Success", "Last corner point removed.")
            else:
                messagebox.showwarning("Warning", "No points to remove.")

    def save_to_json(self):
        """Saves rescaled zones based on image-to-canvas size ratio to a file."""
        if len(self.regions_json) == 0:
            messagebox.showwarning("Warning", "No zones to save.")
        else:
            save_dg_format = messagebox.askyesno(
                "Saving Format", "Save zones in Degirum format?"
            )
            canvas_width, canvas_height = (
                self.canvas.winfo_width(),
                self.canvas.winfo_height(),
            )
            width_scaling_factor = self.img_width / canvas_width
            height_scaling_factor = self.img_height / canvas_height
            zones_data = []
            for region in self.regions_json:
                zones_list = region["zones"]
                for box in zones_list:
                    box = box["points"]
                    rescaled_box = []
                    for x, y in box:
                        rescaled_x = int(x * width_scaling_factor)
                        rescaled_y = int(y * height_scaling_factor)
                        rescaled_box.append((rescaled_x, rescaled_y))
                    zones_data.append(
                        rescaled_box if save_dg_format else {"points": rescaled_box}
                    )
            results_file_path = filedialog.asksaveasfilename(
                initialfile=self.results_file_name, filetypes=[("JSON file", "*.json")]
            )
            if len(results_file_path) > 0:
                with open(results_file_path, "w") as json_file:
                    json.dump(zones_data, json_file, indent=4)

                messagebox.showinfo(
                    "Success",
                    "Zones saved as {}".format(results_file_path.split("/")[-1]),
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=f"{__file__}",
        description="Launch interactive utility for zone annotation in images",
    )
    _zone_annotator_args(parser)
    _zone_annotator_run(parser.parse_args())
