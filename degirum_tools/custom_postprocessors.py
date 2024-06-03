import importlib
import importlib.util

import numpy as np
import degirum as dg
from matplotlib import colormaps

import cv2
from PIL.Image import Image as PILImage
import PIL


def get_image_backend(image):
    """
    This function sets the image processing backend based on the type of the image and the available libraries.
    It supports OpenCV (cv2) and PIL.

    Args:
        image: The image to be processed. It can be a 3D numpy array or a PIL Image.

    Returns:
        backend: The backend module for image processing.
        backend_name: The name of the backend module.
        resize_map: A dictionary mapping resize methods to their corresponding functions in the backend module.
    """
    backend = None
    backend_name = None
    resize_map = None

    if (
        isinstance(image, np.ndarray)
        and len(image.shape) == 3
        and importlib.util.find_spec("cv2")
    ):
        backend = importlib.import_module("cv2")
        backend_name = "cv2"
        # Define a map for resizing methods in OpenCV
        resize_map = {
            "nearest": backend.INTER_NEAREST,
            "bilinear": backend.INTER_LINEAR,
            "area": backend.INTER_AREA,
            "bicubic": backend.INTER_CUBIC,
            "lanczos": backend.INTER_LANCZOS4,
        }

    # If PIL is available, use PIL as the backend
    elif importlib.util.find_spec("PIL"):
        backend = importlib.import_module("PIL")
        # If the image is a PIL Image, set the backend name to "pil"
        if backend and isinstance(image, backend.Image.Image):
            backend_name = "pil"
            # Define a map for resizing methods in PIL
            resize_map = {
                "nearest": backend.Image.Resampling.NEAREST,
                "bilinear": backend.Image.Resampling.BILINEAR,
                "area": backend.Image.Resampling.BOX,
                "bicubic": backend.Image.Resampling.BICUBIC,
                "lanczos": backend.Image.Resampling.LANCZOS,
            }

    return backend, backend_name, resize_map


def resize_to_original_size(image, original_image, backend, interpolation):
    if backend == "cv2":
        h, w = original_image.shape[:2]
    else:  # If the backend is PIL, find original dimensions using image.size
        w, h = original_image.size
    resized_image = cv2.resize(src=image, dsize=(w, h), interpolation=interpolation)
    resized_image = np.clip(resized_image, 0, 255).astype(np.uint8)
    return resized_image


def adjust_image(image, backend):
    if backend == "cv2":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image = PIL.Image.fromarray(image)
    return image


# Normalize a numpy array
def normalize_array(np_array):
    """
    Normalizes a numpy array.

    Parameters
    ----------
    np_array : numpy.ndarray
        the array to be normalized

    Returns
    -------
    numpy.ndarray
        the normalized array
    """
    return (np_array - np_array.min()) / (np_array.max() - np_array.min())


class StyleTransfer(dg.postprocessor.InferenceResults):
    def __init__(self, *args, **kwargs):
        """
        Initializes the StyleTransfer object with the provided arguments.
        """
        super().__init__(*args, **kwargs)  # call base class constructor first
        # Retrieve backend
        self._backend, self._backend_name, _resize_map = get_image_backend(self.image)

        # Resize the depth map back to the original image size if the original image exists.
        if self.image is not None:
            # Get the first inference result data
            data = self._inference_results[0]["data"]
            # Transpose the data from BCHW to BHWC
            # TODO: Remove this line when the model output is in BHWC format. This is possible when tflite models are used. Maybe we need to add a flag to the model configuration to indicate the output format.
            data = np.transpose(data, (0, 2, 3, 1))
            stylized_image = resize_to_original_size(
                data.squeeze(), self.image, self._backend_name, cv2.INTER_CUBIC
            )
            self._inference_results[0]["data"] = stylized_image

    @property
    def image_overlay(self):
        """
        Returns the image overlay.

        Returns
        -------
        numpy.ndarray or PIL.Image.Image
            the image overlay
        """
        return adjust_image(self._inference_results[0]["data"], self._backend_name)


# Define a class for depth estimation results
class DepthEstimation(dg.postprocessor.InferenceResults):
    """
    A class used to represent the results of depth estimation.

    Attributes
    ----------
    color_map : str
        the default color map for depth visualization
    _backend : module
        the backend module for image processing
    _backend_name : str
        the name of the backend module
    _resize_map : dict
        a dictionary mapping resize methods to their corresponding functions in the backend module

    Methods
    -------
    _normalize_depth_map(depth_map)
        Normalizes the depth map
    _convert_depth_to_image(depth_map)
        Converts the depth map to an image
    image_overlay
        Returns the image overlay
    """

    color_map = "viridis"  # Default color map for depth visualization

    def __init__(self, *args, **kwargs):
        """
        Initializes the DepthEstimation object and sets the image processing backend.
        """
        super().__init__(*args, **kwargs)  # call base class constructor first

        # Retrieve backend
        self._backend, self._backend_name, _resize_map = get_image_backend(self.image)

        # Resize the depth map back to the original image size if the original image exists.
        if self.image is not None:
            # Get the first inference result data
            data = self._inference_results[0]["data"]
            # Transpose the data to match the image dimensions
            # data = np.transpose(data, (1, 2, 0))
            # Get the resize mode from the model parameters
            resized_image = resize_to_original_size(
                data.squeeze(), self.image, self._backend_name, cv2.INTER_CUBIC
            )
            self._inference_results[0]["data"] = resized_image

    # Convert the depth map to an image
    def _convert_depth_to_image(self, depth_map):
        """
        Converts the depth map to an image.

        Parameters
        ----------
        depth_map : numpy.ndarray
            the depth map to be converted

        Returns
        -------
        numpy.ndarray
            the depth map converted to an image
        """
        # Get the color map
        c_map = colormaps[DepthEstimation.color_map]
        # Normalize the depth map
        depth_map = normalize_array(depth_map)
        # Apply the color map to the depth map
        depth_map = c_map(depth_map)[:, :, :3] * 255
        # Convert the depth map to 8-bit unsigned integer format
        depth_map = depth_map.astype(np.uint8)
        return depth_map

    # Get the image overlay
    @property
    def image_overlay(self):
        """
        Returns the image overlay.

        Returns
        -------
        numpy.ndarray or PIL.Image.Image
            the image overlay
        """
        # Convert the depth map to an image
        image = self._convert_depth_to_image(self._inference_results[0]["data"])
        return adjust_image(image, self._backend_name)


class SuperResolution(dg.postprocessor.InferenceResults):
    """
    A class used to represent the results of super resolution.

    Attributes
    ----------
    resize_factor : int
        the factor by which the image size is increased
    _backend : module
        the backend module for image processing
    _backend_name : str
        the name of the backend module

    Methods
    -------
    image_overlay
        Returns the image overlay
    """

    resize_factor = 4  # The factor by which the image size is increased

    def __init__(self, *args, **kwargs):
        """
        Initializes the SuperResolution object and sets the image processing backend.
        """
        super().__init__(*args, **kwargs)  # call base class constructor first

        # Retrieve backend
        self._backend, self._backend_name, _resize_map = get_image_backend(self.image)

        r_factor = SuperResolution.resize_factor

        # Get the first inference result data
        data = self._inference_results[0]["data"]

        # If the data is not 4-dimensional, reshape it
        if len(data.shape) != 4:
            if self._model_params.InputType[0] == "Image":
                data = np.reshape(
                    data,
                    (
                        1,
                        3,
                        self._model_params.InputH[0] * r_factor,
                        self._model_params.InputW[0] * r_factor,
                    ),
                )
            else:
                data = np.reshape(
                    data,
                    (
                        1,
                        3,
                        self._model_params.InputW[0] * r_factor,
                        self._model_params.InputC[0] * r_factor,
                    ),
                )

        # Normalize the data and transpose it to match the image dimensions
        data = (
            (data.squeeze() * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
        )  # Currently in RGB
        nh, nw, _ = data.shape

        # Get the original image size
        try:
            if self._backend_name == "cv2":
                h, w, _ = self.image.shape
            else:
                w, h = self.image.size
        except Exception as e:
            pass

        # Set the resize method based on the size of the data
        if nh >= h * r_factor or nw >= w * r_factor:
            resize_method = cv2.INTER_AREA
        else:
            resize_method = cv2.INTER_CUBIC

        # If the data size does not match the image size, resize the data
        if not (nh == h and nw == w):
            data = cv2.resize(data, (w * r_factor, h * r_factor), resize_method)

        # Update the inference results with the resized data
        self._inference_results[0]["data"] = data

    @property
    def image_overlay(self):
        """
        Returns the image overlay.

        Returns
        -------
        numpy.ndarray or PIL.Image.Image
            the image overlay
        """
        return adjust_image(self._inference_results[0]["data"], self._backend_name)
