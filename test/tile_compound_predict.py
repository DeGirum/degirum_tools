import cv2
import degirum as dg
import matplotlib.pyplot as plt

# from degirum_tools.tiling import TileModel as OldTileModel

from degirum_tools.tile_compound_models import TileExtractorPseudoModel, TileModel, LocalGlobalTileModel, BoxFusionTileModel, BoxFusionLocalGlobalTileModel
from degirum_tools import NmsBoxSelectionPolicy, NmsOptions

zoo_name = "visdrone"
model_name = 'yolov8s_relu6_visdrone--640x640_quant_n2x_orca1_1'
token = "dg_UMD61uvuaZQGK9J4g271V7cy2z4BKdT3kXmD3"
file = '0000009_01723_d_0000006.jpg'

color_rolodex = [(255, 000, 000),
                 (255, 160, 160),
                 (255, 125, 000),
                 (255, 255, 000),
                 (255, 255, 160),
                 (000, 255, 000),
                 (000, 000, 255),
                 (100, 100, 255),
                 (130, 000, 255),
                 (255, 000, 255)]

zoo = dg.connect(dg.CLOUD, "https://cs.degirum.com/degirum/" + zoo_name, token)
model = zoo.load_model(model_name)

model.overlay_color = color_rolodex

nms_options = NmsOptions(
    threshold=0.6,
    use_iou=True,
    box_select=NmsBoxSelectionPolicy.MOST_PROBABLE,
)

# SimpleTiling equivalent
tile_extractor = TileExtractorPseudoModel(3, 2, 0.1, model, global_tile=False)
tile_model = TileModel(tile_extractor, model, nms_options=nms_options)
results = tile_model(file)
plt.imshow(cv2.cvtColor(results.image_overlay, cv2.COLOR_BGR2RGB))
plt.title('SimpleTiling equivalent')
plt.show(block=True)

# LocalGlobalTiling equivalent
tile_extractor = TileExtractorPseudoModel(3, 2, 0.1, model, global_tile=True)
tile_model = LocalGlobalTileModel(tile_extractor, model, 0.01)
results = tile_model(file)
plt.imshow(cv2.cvtColor(results.image_overlay, cv2.COLOR_BGR2RGB))
plt.title('LocalGlobalTiling equivalent')
plt.show(block=True)

# WBFSimpleTiling equivalent
tile_extractor = TileExtractorPseudoModel(3, 2, 0.1, model, global_tile=False)
tile_model = BoxFusionTileModel(tile_extractor, model, 0.02, 0.8)
results = tile_model(file)
plt.imshow(cv2.cvtColor(results.image_overlay, cv2.COLOR_BGR2RGB))
plt.title('WBFSimpleTiling equivalent')
plt.show(block=True)

# WBFLocalGlobalTiling equivalent
tile_extractor = TileExtractorPseudoModel(3, 2, 0.1, model, global_tile=True)
tile_model = BoxFusionLocalGlobalTileModel(tile_extractor, model, 0.01, 0.02, 0.8, nms_options=nms_options)
results = tile_model(file)
plt.imshow(cv2.cvtColor(results.image_overlay, cv2.COLOR_BGR2RGB))
plt.title('WBFLocalGlobalTiling equivalent')
plt.show(block=True)

# tile_model = OldTileModel(model, WBFLocalGlobalTiling(3,2,0.1, 0.01))
# results = tile_model(file)
# plt.imshow(cv2.cvtColor(results.image_overlay, cv2.COLOR_BGR2RGB))
# plt.show(block=True)
