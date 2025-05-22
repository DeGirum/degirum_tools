import degirum as dg
import degirum_tools

gst_pipeline = "filesrc location=Traffic.mp4 ! decodebin ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink"
model = dg.load_model(
    model_name='yolov8n_relu6_coco--640x640_quant_hailort_hailo8_1',
    inference_host_address='@cloud',
    zoo_url='degirum/models_hailo_dg',
    token=degirum_tools.get_token(),       
)
counter = 0

#with degirum_tools.Display("AI Camera") as display:
for inference_result in degirum_tools.predict_stream(model, gst_pipeline):
    print(inference_result)
    #display.show(inference_result)  