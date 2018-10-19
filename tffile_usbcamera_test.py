import cv2, sys, time
import numpy as np
import tensorflow as tf
from PIL import Image

width  = 320
height = 240
fps = ""
framecount = 0
elapsedTime = 0
index_void = 2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cv2.namedWindow('FPS', cv2.WINDOW_AUTOSIZE)

seg_image = Image.open("data_set/VOCdevkit/person/SegmentationClass/009649.png")
interpreter = tf.contrib.lite.Interpreter(model_path="model/semanticsegmentation_frozen_person_quantized_08.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

while True:
    t1 = time.perf_counter()
    ret, frame = cap.read()

    # Resize image
    frame = frame.resize((128, 128), Image.ANTIALIAS)
    # Normalization
    frame = np.asarray(frame)
    prepimg = frame / 255.0
    prepimg = prepimg[np.newaxis, :, :, :]

    # Segmentation
    interpreter.set_tensor(input_details[0]['index'], np.array(prepimg, dtype=np.float32))
    t1 = time.time()
    interpreter.invoke()
    outputs = interpreter.get_tensor(output_details[0]['index'])

    # Get a color palette
    palette = seg_image.getpalette()

    # View
    output = outputs[0]
    res = np.argmax(output, axis=2)
    if index_void is not None:
        res = np.where(res == index_void, 0, res)
    image = Image.fromarray(np.uint8(res), mode="P")
    image.putpalette(palette)
    image = image.convert("RGB")
    image = image.resize((width, height))

    cv2.putText(image, fps, (width-180,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
    cv2.imshow('FPS', image)

    if cv2.waitKey(1)&0xFF == ord('q'):
        break
    t2 = time.perf_counter()
    framecount += 1
    if framecount >= 5:
        fps = "(Playback) {:.1f} FPS".format(framecount/elapsedTime)
        print("fps = ", str(fps))
        framecount = 0
        elapsedTime = 0
    elapsedTime += t2-t1

cap.release()
cv2.destroyAllWindows()

