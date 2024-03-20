import degirum as dg
from degirum_tools.detection_eval import ObjectDetectionModelEvaluator

from degirum_tools.tiling import TileModel
from degirum_tools.tile_strategy import WBFLocalGlobalTiling #SimpleTiling, LocalGlobalTiling, WBFSimpleTiling,

access_token = 'YOUR_TOKEN'

classmap = [0,1,2,3,4,5,6,7,8,9]

zoo = dg.connect(dg.CLOUD, "https://cs.degirum.com/degirum/visdrone_detection", access_token)

model_name = 'yolov8s_relu6_visdrone--640x384_quant_n2x_orca1_1'

# You can find these ptahs in /data/ml-data/VisDrone on mercury
img_folder_path = "../VisDrone2019-DET-val/images" 
anno_json = "../annotations_VisDrone_val.json"

model = zoo.load_model(model_name)
#tile_model = TileModel(model, SimpleTiling(2, 2, 0.1))
#tile_model = TileModel(model, LocalGlobalTiling(2, 2, 0.1, 0.01))
#tile_model = TileModel(model, WBFSimpleTiling(3, 2, 0.1, 0.02, 0.8))
tile_model = TileModel(model, WBFLocalGlobalTiling(3, 2, 0.1, 0.01, 0.02, 0.8))

map_evaluator = ObjectDetectionModelEvaluator(tile_model, classmap=classmap)

map_results = map_evaluator.evaluate(
    img_folder_path,
    ground_truth_annotations_path=anno_json,
    num_val_images=0,
    print_frequency=100,
)