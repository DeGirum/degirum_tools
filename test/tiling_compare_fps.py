from dataclasses import dataclass

import degirum as dg
import degirum_tools as dgtools
import numpy as np

from degirum_tools.compound_models import CompoundModelBase
from degirum_tools.tile_compound_models import TileExtractorPseudoModel, TileModel, BoxFusionTileModel, LocalGlobalTileModel, BoxFusionLocalGlobalTileModel
from degirum_tools.ui_support import Timer  # perf_counter_ns() probably better than time_ns() but keep it the same for comparisons sake.


@dataclass
class ModelTimeProfile:
    """Class to hold model time profiling results"""

    elapsed: float  # elapsed time in seconds
    iterations: int  # number of iterations made
    observed_fps: float  # observed inference performance, frames per second
    max_possible_fps: float  # maximum possible inference performance, frames per second
    parameters: dict  # copy of model parameters
    time_stats: dict  # model time statistics dictionary


def compound_time_profile(model, tiling=(1, 1), iterations=100) -> ModelTimeProfile:
    # tiling (cols, rows)
    if isinstance(model, CompoundModelBase):
        # skip non-image type models
        if model.model2.model_info.InputType[0] != "Image":
            raise NotImplementedError

        saved_params = {
            "input_image_format": model.model2.input_image_format,
            "measure_time": model.model2.measure_time,
            "image_backend": model.model2.image_backend,
        }
    else:
        raise Exception('Model type not supported.')

    elapsed = 0.0
    try:
        # configure model
        if isinstance(model, CompoundModelBase):
            model.model2.input_image_format = "JPEG"
            model.model2.measure_time = True
            model.model2.image_backend = "opencv"

            imgsz = model.model2.model_info.InputH + model.model2.model_info.InputW

        imgsz[0] *= tiling[1]
        imgsz[1] *= tiling[0]
        imgsz.append(3)

        frame = np.zeros(imgsz, dtype=np.uint8)

        # define source of frames
        def source():
            for fi in range(iterations):
                yield frame

        # Another thing that should be implemented in ModelLike or in CompoundModelBase?
        # with model:
        model(frame)  # run model once to warm up the system

        # run batch prediction
        t = Timer()
        for res in model.predict_batch(source()):
            pass
        elapsed = t()

    finally:
        # restore model parameters
        for k, v in saved_params.items():
            if isinstance(model, CompoundModelBase):
                setattr(model.model2, k, v)
                stats = model.model2.time_stats()
            else:
                setattr(model, k, v)
                stats = model.time_stats()

    return ModelTimeProfile(
        elapsed=elapsed,
        iterations=iterations,
        observed_fps=iterations / elapsed,
        max_possible_fps=1e3 / stats["CoreInferenceDuration_ms"].avg,
        parameters=model.model2.model_info if isinstance(model, CompoundModelBase) else model.model_info,
        time_stats=stats,
    )


token = "dg_DuH5LpfrcPmqkeq6uX84QQyC15hLZGUo7sZc7"
zoo_name = "visdrone_detection"
model_name = 'yolov8s_relu6_visdrone--640x384_quant_n2x_orca1_1'
# (col, row)
tiling = (2, 2)

zoo = dg.connect(dg.CLOUD, "https://cs.degirum.com/degirum/" + zoo_name, token)
model = zoo.load_model(model_name)

model_size = model.model_info.InputW + model.model_info.InputH
image_size = model_size[0] * tiling[0], model_size[1] * tiling[1]

tiles = dgtools.generate_tiles_fixed_ratio(model_size, tiling, image_size, (0.0, 0.0))
current_tile_extractor = dgtools.RegionExtractionPseudoModel(tiles, model)
nms_options = dgtools.NmsOptions(threshold=0.3, use_iou=False, box_select=dgtools.NmsBoxSelectionPolicy.LARGEST_AREA)
compound_model = dgtools.CroppingAndDetectingCompoundModel(current_tile_extractor, model, nms_options=nms_options)

time_results = compound_time_profile(compound_model, tiling)

print(f"Current dg_tools tiling FPS: {time_results.observed_fps}")

new_tile_extractor = TileExtractorPseudoModel(2, 2, 0.0, model)
tile_model = TileModel(new_tile_extractor, model, nms_options=nms_options)
time_results = compound_time_profile(tile_model, tiling)
print(f"SimpleTiling compound model equivalent FPS: {time_results.observed_fps}")

new_tile_extractor = TileExtractorPseudoModel(2, 2, 0.0, model, global_tile=True)
tile_model = LocalGlobalTileModel(new_tile_extractor, model, 0.01, nms_options=nms_options)
time_results = compound_time_profile(tile_model, tiling)
print(f"LocalGlobalTiling compound model equivalent FPS: {time_results.observed_fps}")

new_tile_extractor = TileExtractorPseudoModel(2, 2, 0.0, model, global_tile=True)
tile_model = BoxFusionTileModel(new_tile_extractor, model, 0.02, 0.8, nms_options=nms_options)
time_results = compound_time_profile(tile_model, tiling)
print(f"WBFSimpleTiling compound model equivalent FPS: {time_results.observed_fps}")

new_tile_extractor = TileExtractorPseudoModel(2, 2, 0.0, model, global_tile=True)
tile_model = BoxFusionLocalGlobalTileModel(new_tile_extractor, model, 0.01, 0.02, 0.8, nms_options=nms_options)
time_results = compound_time_profile(tile_model, tiling)
print(f"WBFLocalGlobalTiling compound model equivalent FPS: {time_results.observed_fps}")
