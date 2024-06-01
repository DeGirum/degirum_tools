import importlib
import importlib.util

import numpy as np
import degirum as dg
from matplotlib import colormaps


# Define a class for depth estimation results
class DepthEstimationResults(dg.postprocessor.InferenceResults):
    color_map = "viridis"  # Default color map for depth visualization

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # call base class constructor first

        # Retrieve backend
        # If the image is a 3D numpy array and OpenCV (cv2) is available, use OpenCV as the backend
        if (
            isinstance(self.image, np.ndarray)
            and len(self.image.shape) == 3
            and importlib.util.find_spec("cv2")
        ):
            self._backend = importlib.import_module("cv2")
            self._backend_name = "cv2"
            # Define a map for resizing methods in OpenCV
            _resize_map = {
                "nearest": self._backend.INTER_NEAREST,
                "bilinear": self._backend.INTER_LINEAR,
                "area": self._backend.INTER_AREA,
                "bicubic": self._backend.INTER_CUBIC,
                "lanczos": self._backend.INTER_LANCZOS4,
            }

        # If PIL is available, use PIL as the backend
        elif importlib.util.find_spec("PIL"):
            self._backend = importlib.import_module("PIL")
            # If the image is a PIL Image, set the backend name to "pil"
            if self._backend and isinstance(self.image, self._backend.Image.Image):
                self._backend_name = "pil"
                # Define a map for resizing methods in PIL
                _resize_map = {
                    "nearest": self._backend.Image.Resampling.NEAREST,
                    "bilinear": self._backend.Image.Resampling.BILINEAR,
                    "area": self._backend.Image.Resampling.BOX,
                    "bicubic": self._backend.Image.Resampling.BICUBIC,
                    "lanczos": self._backend.Image.Resampling.LANCZOS,
                }

        # Resize the depth map back to the original image size if the original image exists.
        if self.image is not None:
            # Get the first inference result data
            data = self._inference_results[0]["data"]
            # Transpose the data to match the image dimensions
            data = np.transpose(data, (1, 2, 0))
            # Get the resize mode from the model parameters
            resize_mode = _resize_map[self._model_params.InputResizeMethod[0]]

            # If the backend is OpenCV, resize using OpenCV's resize function
            if self._backend_name == "cv2":
                image_size = self.image.shape[:2][::-1]
                data = self._backend.resize(data, image_size, interpolation=resize_mode)
            else:  # If the backend is PIL, resize using PIL's resize function
                data_surrogate = self._backend.Image.fromarray(data.squeeze())
                data_surrogate = data_surrogate.resize(
                    self.image.size, resample=resize_mode
                )
                data = np.array(data_surrogate)

            # Expand the dimensions of the data and update the inference results
            data = np.expand_dims(data, axis=0)
            self._inference_results[0][
                "data"
            ] = data  # TODO: Should the returned depth map be normalized already?

    # Normalize the depth map
    def _normalize_depth_map(self, depth_map):
        return (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    # Convert the depth map to an image
    def _convert_depth_to_image(self, depth_map):
        # Get the color map
        c_map = colormaps[DepthResults.color_map]
        # Squeeze the depth map to remove single-dimensional entries from the shape of an array.
        depth_map = depth_map.squeeze(0)
        # Normalize the depth map
        depth_map = self._normalize_depth_map(depth_map)
        # Apply the color map to the depth map
        depth_map = c_map(depth_map)[:, :, :3] * 255
        # Convert the depth map to 8-bit unsigned integer format
        depth_map = depth_map.astype(np.uint8)

        return depth_map

    # Get the image overlay
    @property
    def image_overlay(self):
        # Convert the depth map to an image
        image = self._convert_depth_to_image(self._inference_results[0]["data"])

        # If the backend is OpenCV, return the image as is
        if self._backend_name == "cv2":
            return image

        # If the backend is PIL, convert the numpy array to a PIL Image
        return self._backend.Image.fromarray(image)

    # Return a string representation of the object for debugging
    def __repr__(self):
        return self._inference_results.__repr__()

    # Return a string representation of the object for printing
    def __str__(self):
        return self._inference_results.__str__()
