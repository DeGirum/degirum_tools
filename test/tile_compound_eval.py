import degirum as dg
from degirum_tools import NmsOptions, NmsBoxSelectionPolicy, get_token
from degirum_tools.tile_compound_models import TileExtractorPseudoModel, BoxFusionLocalGlobalTileModel, BoxFusionTileModel, LocalGlobalTileModel, TileModel
from degirum_tools.detection_eval import ObjectDetectionModelEvaluator

hw_location = dg.CLOUD
model_name = "yolov8s_relu6_visdrone--640x384_quant_n2x_orca1_1"
model_zoo_url = "https://cs.degirum.com/degirum/visdrone_scales"

img_folder_path = "VisDrone2019-DET-val/images"
anno_json = "annotations_VisDrone_val.json"

class_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

tiling_params = (3, 2, 0.1)

# connect to AI inference engine
zoo = dg.connect(hw_location, model_zoo_url, get_token())
# load object detection model
model = zoo.load_model(model_name)

model.output_confidence_threshold = 0.001
model.output_nms_threshold = 0.7
model.output_max_detections = 300
model.output_max_detections_per_class = (100)
model.output_max_classes_per_detection = (1)
model.output_use_regular_nms = True
model.input_resize_method = "bilinear"
model.input_pad_method = "letterbox"
model.image_backend = "opencv"
model.input_image_format = "JPEG"
model.input_numpy_colorspace = "auto"
model.input_letterbox_fill_color = (114, 114, 114)

nms_options = NmsOptions(
    threshold=model.output_nms_threshold,
    use_iou=True,
    box_select=NmsBoxSelectionPolicy.MOST_PROBABLE,
)

compound_models = []

# SimpleTiling equivalent
tile_extractor = TileExtractorPseudoModel(*tiling_params, model, global_tile=False)
tile_model = TileModel(tile_extractor, model, nms_options=nms_options)
compound_models.append(tile_model)

# # LocalGlobalTiling equivalent
tile_extractor = TileExtractorPseudoModel(*tiling_params, model, global_tile=True)
tile_model = LocalGlobalTileModel(tile_extractor, model, 0.01, nms_options=nms_options)
compound_models.append(tile_model)

# # WBFSimpleTiling equivalent
tile_extractor = TileExtractorPseudoModel(*tiling_params, model, global_tile=False)
tile_model = BoxFusionTileModel(tile_extractor, model, 0.02, 0.8, nms_options=nms_options)
compound_models.append(tile_model)

# WBFLocalGlobalTiling equivalent
tile_extractor = TileExtractorPseudoModel(*tiling_params, model, global_tile=True)
tile_model = BoxFusionLocalGlobalTileModel(tile_extractor, model, 0.01, 0.02, 0.8, nms_options=nms_options)
compound_models.append(tile_model)

for cmodel in compound_models:
    print(cmodel.__class__.__name__)
    setattr(cmodel, 'output_postprocess_type', 'DetectionYoloV8')
    map_evaluator = ObjectDetectionModelEvaluator(cmodel, classmap=class_map, pred_path="compoundsimpletile.json")

    map_results = map_evaluator.evaluate(
        img_folder_path,
        ground_truth_annotations_path=anno_json,
        max_images=0,
    )
