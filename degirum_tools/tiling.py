import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator, Iterable, Sequence, Set, Union

if sys.version_info == (3, 9): # TypeAlias dissapears in version 3.9 but reappears in version 3.10
    from typing_extensions import TypeAlias
else:
    from typing import TypeAlias

import cv2
import numpy as np
from PIL.Image import Image as PILImageClass
from PIL import Image as PILImage

from degirum_tools.tile_strategy import BaseTileStrategy
from degirum_tools.math_support import nms
from degirum.exceptions import DegirumException
from degirum.postprocessor import DetectionResults, InferenceResults, _inference_result_type
from degirum.model import Model


_ImageType: TypeAlias = Union[str, PILImageClass, np.ndarray, bytes]

class TileModel():
    # This set potentially needs to be updated if the public interface to the model class changes.
    _model_attr_blacklist = { "__call__", "predict", "predict_batch", "predict_dir" }

    # These two variables must remain outside the constructor. They also must be the first variables
    # set in the constructor in order for the pseudo-subclassing trick to work.
    _model: Union[Model, None] = None
    _model_attrs: Set[str] = set()

    def __init__(self, model: Model, tile_strategy: BaseTileStrategy):
        self._model = model
        self._model_attrs = set(model.__dir__()).difference(self._model_attr_blacklist)

        self._model_params = model.model_info

        if _inference_result_type(self._model_params)() != DetectionResults:
            raise DegirumException('Tiling not supported with non-detection type models.')
        
        self._tile_strategy = tile_strategy
        self._tile_strategy._set_model_parameters(model.model_info)
        self._tile_strategy._set_label_dict(self._model.label_dictionary)

    # Override necessary to access dg.model.Model interface.
    def __getattr__(self, attr):
        try: 
            attr_value = self._model.__getattribute__(attr)
        except AttributeError as e: 
            if sys.version_info >= (3, 10):
                raise AttributeError("'{}' object has no attribute '{}'", name=e.name, obj=type(self).__name__)
            else:
                raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, e.args[0].split("'")[3]))

        return attr_value
    
    # Override necessary to access dg.model.Model interface.
    def __setattr__(self, attr, obj):
        if attr in self._model_attrs:
            return self._model.__setattr__(attr, obj)
        return super().__setattr__(attr, obj)
    
    # Override to make some attributes pseudo private.        
    def __dir__(self):
        return self._model.__dir__()
    
    def __enter__(self):
        self._model._in_context = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._model._in_context = False
        self._model._release_runtime()

    def __call__(self, input):
        return self.predict(input)

    def predict(self, input: _ImageType, frame_info: Any=None) -> DetectionResults:
        def _identity_conversion(x,y):
            return (x, y)
        
        if isinstance(input, str):
            original_image = cv2.imread(input)
        elif isinstance(input, np.ndarray):
            original_image = input
        elif isinstance(input, PILImageClass):
            original_image = np.array(input)
        elif isinstance(input, bytes):
            original_image = np.array(PILImage.open(BytesIO(input)))
        else:
            raise DegirumException('Image format not accepted.')

        for result in self._model.predict_batch(self._tile_strategy._generate_tiles(original_image)):
            self._tile_strategy._accumulate_results(result.results, result.info)

        results = self._tile_strategy._get_results()
                 
        results = DetectionResults(model_params=self._model._model_parameters,
                                input_image=original_image,
                                model_image=None if self._model.save_model_image else None, # TODO IMPLEMENT
                                inference_results=results,
                                draw_color=self._model.overlay_color,
                                line_width=self._model.overlay_line_width,
                                show_labels=self._model.overlay_show_labels,
                                show_probabilities=self._model.overlay_show_probabilities,
                                alpha=self._model.overlay_alpha,
                                font_scale=self._model.overlay_font_scale,
                                fill_color=self._model.input_letterbox_fill_color,
                                frame_info=frame_info,
                                conversion=_identity_conversion,
                                label_dictionary=self._model.label_dictionary)
        
        nms(results, iou_threshold=self._model.output_nms_threshold)

        return results
    
    # predict_batch copied and modified from degirum.model.Model
    def predict_batch(self, input: Iterable)-> Iterator[InferenceResults]:
        def source():
            for d in input:
                if isinstance(d, tuple):
                    # if data is tuple, we treat first element as frame data and second element as frame info
                    yield d[1], d[0]
                else:
                    # otherwise we treat data as frame data and if it is string, we set frame info equal to frame data
                    # (data is string when it is a filename)
                    yield d if isinstance(d, str) else "", d
        
        for frame_info, data in source():
            yield self.predict(data, frame_info)

    # predict_dir copied and modified from degirum.model.Model
    def predict_dir(
            self,
            path: str,
            *,
            recursive: bool= False,
            extensions: Union[Sequence[str], str]=[".jpg", ".jpeg", ".png", ".bmp"],
        ) -> Iterator[InferenceResults]:

            if len(self._model_params.InputType) > 1:
                raise DegirumException(
                    "'predict_dir' method is not supported for models with number of inputs greater than one."
                )

            mask = "**/*.*" if recursive else "*.*"
            ext = extensions if isinstance(extensions, list) else [extensions]

            def source():
                for e in Path(path).glob(mask):
                    if e.is_file() and e.suffix.lower() in ext:
                        yield str(e), str(e)

            for frame_info, data in source():
                yield self.predict(data, frame_info)