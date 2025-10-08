#!/usr/bin/env python3

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

if not Gst.is_initialized():
    Gst.init(None)

# Create the same pipeline as in the test
pipeline_str = "fakesrc num-buffers=2 ! video/x-raw,width=640,height=480,format=I420,framerate=30/1 ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink"
pipeline = Gst.parse_launch(pipeline_str)
appsink = pipeline.get_by_name("sink")

print("Available methods on GstAppSink:")
for attr in dir(appsink):
    if not attr.startswith('_'):
        print(f"  {attr}")

print("\nTrying different sample pulling methods:")

# Set appsink properties like in the actual implementation
appsink.set_property("emit-signals", True)
appsink.set_property("sync", False)
appsink.set_property("drop", True)

# Set pipeline to PLAYING
ret = pipeline.set_state(Gst.State.PLAYING)
print(f"Pipeline state set to PLAYING, return: {ret}")

# Wait for pipeline to start
if ret == Gst.StateChangeReturn.ASYNC:
    state_ret = pipeline.get_state(Gst.CLOCK_TIME_NONE)
    print(f"Pipeline state after waiting: {state_ret}")

import time
time.sleep(0.5)

# Try different methods
methods_to_try = [
    'pull_sample',
    'try_pull_sample', 
    'emit',
    'get_sample',
    'try_get_sample'
]

for method in methods_to_try:
    if hasattr(appsink, method):
        print(f"\n{method} exists, trying...")
        try:
            if method == 'emit':
                result = appsink.emit("pull-sample")
            else:
                result = getattr(appsink, method)()
            print(f"  Result: {result}")
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print(f"\n{method} does not exist")

pipeline.set_state(Gst.State.NULL)
