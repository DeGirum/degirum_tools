import degirum as dg
import degirum_tools


# gst_pipeline = "filesrc location=Traffic.mp4 ! decodebin ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink"
# gst_pipeline= "v4l2src device=/dev/video0 ! videoconvert ! videoscale ! video/x-raw,width=960,height=540 ! appsink name=sink"
# gst_pipeline="v4l2src device=/dev/video0 ! videoconvert ! videoscale ! video/x-raw,width=640,height=640,format=RGB ! appsink name=sink"
gst_pipeline='rtsp://admin:admin123@192.168.0.194:554'
#gst_pipeline = 19
#gst_pipeline='Traffic.mp4'
source_type="gstream"

#         f"framerate={int(fps)}/1 ! videoconvert ! appsink name=sink"
model = dg.load_model(
    model_name='yolov8n_relu6_coco--640x640_quant_openvino_multidevice_1',
    #model_name='yolov8n_relu6_lp_ocr--256x128_quant_openvino_multidevice_1',
    inference_host_address='@cloud',
    zoo_url='degirum/intel',
    token="",       
)
counter = 0

with degirum_tools.Display("AI Camera") as display:
    for inference_result in degirum_tools.predict_stream(model, gst_pipeline, source_type):
        print(inference_result)
        #display.show(inference_result)