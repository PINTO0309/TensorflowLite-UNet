#!/usr/bin/env python

import sys
import cv2
import numpy as np
from PIL import Image
import time
from openvino.inference_engine import IENetwork, IEPlugin

model_xml='/opt/intel/computer_vision_sdk_2018.4.420/deployment_tools/model_optimizer/10_lrmodels/UNet/FP16/semanticsegmentation_frozen_person_32.xml'
model_bin='/opt/intel/computer_vision_sdk_2018.4.420/deployment_tools/model_optimizer/10_lrmodels/UNet/FP16/semanticsegmentation_frozen_person_32.bin'
net = IENetwork.from_ir(model=model_xml, weights=model_bin)
seg_image = Image.open("data/input/009649.png")
palette = seg_image.getpalette() # Get a color palette
index_void = 2 # Define index_void Back Ground
camera_width = 320
camera_height = 240
fps = ""
elapsedTime = 0

plugin = IEPlugin(device="HETERO:MYRIAD,CPU")
plugin.set_config({"TARGET_FALLBACK": "HETERO:MYRIAD,CPU"})
plugin.set_initial_affinity(net)
#plugin = IEPlugin(device="CPU")
exec_net = plugin.load(network=net)

input_blob = next(iter(net.inputs))        #input_blob = 'input'
out_blob   = next(iter(net.outputs))       #out_blob   = 'output/BiasAdd'
n, c, h, w = net.inputs[input_blob].shape  #n, c, h, w = 1, 3, 256, 256

del net

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
time.sleep(1)

while cap.isOpened():
    t1 = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    #frame = cv2.imread('data/input/000003.jpg')
    prepimg = frame[:, :, ::-1].copy()
    #prepimg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    prepimg = Image.fromarray(prepimg)
    prepimg = prepimg.resize((256, 256), Image.ANTIALIAS)
    prepimg = np.asarray(prepimg) / 255.0
    prepimg = prepimg.transpose((2, 0, 1)).reshape((1, c, h, w))

    exec_net.start_async(request_id=0, inputs={input_blob: prepimg})

    if exec_net.requests[0].wait(-1) == 0:
        outputs = exec_net.requests[0].outputs[out_blob] # (1, 3, 256, 256)
        outputs = outputs.transpose((2, 3, 1, 0)).reshape((h, w, c)) # (256, 256 3)
        outputs = cv2.resize(outputs, (camera_width, camera_height)) # (240, 320, 3)

        # View
        res = np.argmax(outputs, axis=2)
        if index_void is not None:
            res = np.where(res == index_void, 0, res)
        image = Image.fromarray(np.uint8(res), mode="P")
        image.putpalette(palette)
        image = image.convert("RGB")

        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.addWeighted(frame, 1, image, 0.9, 0)

    cv2.putText(image, fps, (camera_width-180,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
    cv2.imshow("Result", image)

    if cv2.waitKey(1)&0xFF == ord('q'):
        break
    elapsedTime = time.time() - t1
    fps = "(Playback) {:.1f} FPS".format(1/elapsedTime)

cv2.destroyAllWindows()
del exec_net
del plugin
