import importlib
import importlib.util
from typing import Optional, Union

import numpy as np
import degirum as dg
from matplotlib import colormaps

import cv2
from PIL.Image import Image as PILImage
import PIL


cv2_resize_map = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


pil_resize_map = {
    "nearest": PIL.Image.Resampling.NEAREST,
    "bilinear": PIL.Image.Resampling.BILINEAR,
    "area": PIL.Image.Resampling.BOX,
    "bicubic": PIL.Image.Resampling.BICUBIC,
    "lanczos": PIL.Image.Resampling.LANCZOS,
}


def get_image_backend(image: Union[np.ndarray, PILImage]) -> tuple:
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
    resize_map: Optional[dict] = None

    if (
        isinstance(image, np.ndarray)
        and len(image.shape) == 3
        and importlib.util.find_spec("cv2")
    ):
        backend = importlib.import_module("cv2")
        backend_name = "cv2"
        # Define a map for resizing methods in OpenCV
        resize_map = cv2_resize_map

    # If PIL is available, use PIL as the backend
    elif importlib.util.find_spec("PIL"):
        backend = importlib.import_module("PIL")
        # If the image is a PIL Image, set the backend name to "pil"
        if backend and isinstance(image, backend.Image.Image):
            backend_name = "pil"
            # Define a map for resizing methods in PIL
            resize_map = pil_resize_map

    return backend, backend_name, resize_map


def get_image_shape(image: Union[np.ndarray, PILImage]) -> tuple[int, int, int]:
    if isinstance(image, np.ndarray):
        h, w, c = image.shape
    else:
        w, h = image.size
        c = len(image.getbands())

    return h, w, c


def resize_to_original_size(
    image: np.ndarray,
    original_image: Union[np.ndarray, PILImage],
    backend: str,
    interpolation: str
) -> np.ndarray:
    h, w, _ = get_image_shape(original_image)

    resized_image = cv2.resize(src=image,
                               dsize=(w, h),
                               interpolation=cv2_resize_map[interpolation])

    return resized_image


def adjust_image(image: np.ndarray, backend: str, colorspace: str) -> np.ndarray:
    if backend == "cv2":
        if colorspace == 'rgb':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        if colorspace == 'bgr':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = PIL.Image.fromarray(image)  # type: ignore[assignment]

    return image


def sanitize_image(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


# Normalize a numpy array
def normalize_array(np_array: np.ndarray) -> np.ndarray:
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


def transpose_to_hwc(data: np.ndarray, output_channel_first: bool) -> np.ndarray:
    """
    Transposes data to NHWC format if the channel comes before HW channe"""
    if len(data.shape) == 4:
        if output_channel_first:
            data = np.transpose(data, (0, 2, 3, 1))

    return data


# This class does not work, but should theoretically work if we implement some changes in PySDK preprocessor and server-side.
class Colorization(dg.postprocessor.InferenceResults):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # call base class constructor first
        # Retrieve backend
        self._backend, self._backend_name, _ = get_image_backend(self.image)

        assert self._model_params.SaveModelInput[0] is True  # New model parameter needed to set model.save_model_input

        if self.image is not None:
            # Get the first inference result data
            data = self._inference_results[0]["data"]
            # Transpose if the data is in the BCHW format.
            data = transpose_to_hwc(data, self._model_params.OutputChannelFirst)
            # TODO: When batch support is added to framework, remove data.squeeze() and handle accordingly.
            ab_channel = resize_to_original_size(
                data.squeeze(), self.image, self._backend_name, self._model_params.InputResizeMethod[0]
            )
            L_channel = resize_to_original_size(self._model_image, self.image, self._backend_name, self._model_params.InputResizeMethod[0])

            colorized_image = np.concatenate((L_channel[:, :, None], ab_channel), axis=2) * 255
            colorized_image = sanitize_image(colorized_image)

            self._inference_results[0]["data"] = colorized_image

    @property
    def image_overlay(self):
        """
        Returns the image overlay.

        Returns
        -------
        numpy.ndarray or PIL.Image.Image
            the image overlay
        """
        return adjust_image(self._inference_results[0]["data"],
                            self._backend_name,
                            self._model_params.OutputColorSpace)


class BackgroundRemoval(dg.postprocessor.InferenceResults):
    background_color = (255, 255, 255)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # call base class constructor first
        # Retrieve backend
        self._backend, self._backend_name, _ = get_image_backend(self.image)

        if self.image is not None:
            # Get the first inference result data
            data = self._inference_results[0]["data"]
            # Transpose if the data is in the BCHW format.
            data = transpose_to_hwc(data, self._model_params.OutputChannelFirst)
            # TODO: When batch support is added to framework, remove data.squeeze() and handle accordingly.
            mask = resize_to_original_size(
                data.squeeze(), self.image, self._backend_name, self._model_params.InputResizeMethod[0]
            )
            self._inference_results[0]["data"] = mask

    def _mask_image(self):
        h, w, c = get_image_shape(self.image)

        bg_color = (
            BackgroundRemoval.background_color
            if self._model_params.OutputColorSpace == 'rgb'
            else BackgroundRemoval.background_color[::-1]
        )

        mask = self._inference_results[0]["data"]
        black_masked_image = np.array(self.image) * mask
        background = np.full((h, w, c), bg_color) * (1 - mask)
        composite = sanitize_image(black_masked_image + background)

        return composite

    @property
    def image_overlay(self):
        """
        Returns the image overlay.

        Returns
        -------
        numpy.ndarray or PIL.Image.Image
            the image overlay
        """

        masked_image = self._mask_image()
        return adjust_image(masked_image,
                            self._backend_name,
                            self._model_params.OutputColorSpace)


class StyleTransfer(dg.postprocessor.InferenceResults):
    def __init__(self, *args, **kwargs):
        """
        Initializes the StyleTransfer object with the provided arguments.
        """
        super().__init__(*args, **kwargs)
        # Retrieve backend
        self._backend, self._backend_name, _ = get_image_backend(self.image)

        if self.image is not None:
            # Get the first inference result data
            data = self._inference_results[0]["data"]
            # Transpose if the data is in the BCHW format.
            data = transpose_to_hwc(data, self._model_params.OutputChannelFirst)
            # TODO: When batch support is added to framework, remove data.squeeze() and handle accordingly.
            stylized_image = resize_to_original_size(
                data.squeeze(), self.image, self._backend_name, self._model_params.InputResizeMethod[0]
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
        return adjust_image(self._inference_results[0]["data"],
                            self._backend_name,
                            self._model_params.OutputColorSpace)


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
        self._backend, self._backend_name, _ = get_image_backend(self.image)

        # Resize the depth map back to the original image size if the original image exists.
        if self.image is not None:
            # Get the first inference result data, this is in NHW format.
            data = self._inference_results[0]["data"]
            # Transpose the data to match the image dimensions
            # data = np.transpose(data, (1, 2, 0))
            # Get the resize mode from the model parameters
            resized_image = resize_to_original_size(
                data.squeeze(), self.image, self._backend_name, self._model_params.InputResizeMethod[0]
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
        return adjust_image(image,
                            self._backend_name,
                            self._model_params.OutputColorSpace)


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
            else:  # InputType is "Tensor", so preprocessing input params are ordered differently
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
        data = transpose_to_hwc(data, self._model_params.OutputChannelFirst)
        data = sanitize_image(data.squeeze() * 255)

        nh, nw, _ = data.shape

        # Get the original image size
        h, w, _ = get_image_shape(self.image)

        # If the data size does not match the image size scaled, resize the data
        if not (nh == h * r_factor and nw == w * r_factor):
            # Set the resize method, downscaling is better with AREA
            if nh >= h * r_factor or nw >= w * r_factor:
                resize_method = cv2.INTER_AREA
            else:
                resize_method = cv2.INTER_CUBIC

            data = cv2.resize(data, dsize=(w * r_factor, h * r_factor), interpolation=resize_method)

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
        return adjust_image(self._inference_results[0]["data"],
                            self._backend_name,
                            self._model_params.OutputColorSpace)
