# just an example, will remove from the actual final pull request

import time

import cv2
import degirum as dg
import matplotlib.pyplot as plt

from degirum_tools.tiling import TileModel
from degirum_tools.tile_strategy import SimpleTiling, WBFSimpleTiling


model_name = 'yolov8s_relu6_visdrone--640x640_quant_n2x_orca1_1'

zoo = dg.connect(dg.CLOUD, "https://cs.degirum.com/degirum/visdrone", "TOKEN")
model = zoo.load_model(model_name)
tile_model = TileModel(model, SimpleTiling(3, 2, 0.1))


tile_model.overlay_show_probabilities = True
tile_model.overlay_line_width = 2

start = time.time()
results = tile_model('0000009_01723_d_0000006.jpg')
print('Total time: {}'.format(time.time()-start))


plt.imshow(cv2.cvtColor(results.image_overlay, cv2.COLOR_BGR2RGB))
plt.show(block=True)


tile_model = TileModel(model, WBFSimpleTiling(3, 2, 0.1, wbf_thr=0.8))
results = tile_model('0000009_01723_d_0000006.jpg')
plt.imshow(cv2.cvtColor(results.image_overlay, cv2.COLOR_BGR2RGB))
plt.show(block=True)