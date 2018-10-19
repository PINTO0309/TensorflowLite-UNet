import sys, time
import numpy as np
import tensorflow as tf
from PIL import Image


if __name__ == '__main__':

    # Read image
    image     = Image.open("data_set/VOCdevkit/person/18418693150_c40831b00a_o.jpg")
    seg_image = Image.open("data_set/VOCdevkit/person/SegmentationClass/009649.png")
    print("image.size = ", image.size)

    base_width  = image.size[0]
    base_height = image.size[1]
    image.save("3.jpg")

    # Resize image
    image = image.resize((128, 128), Image.ANTIALIAS)

    # Delete alpha channel
    print("image.mode ==", image.mode)
    if image.mode == "RGBA":
        image = image.convert("RGB")

    # Normalization
    image = np.asarray(image)
    prepimg = image / 255.0
    prepimg = prepimg[np.newaxis, :, :, :]

    # Segmentation
    interpreter = tf.contrib.lite.Interpreter(model_path="model/semanticsegmentation_frozen_person_quantized_32.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    interpreter.set_tensor(input_details[0]['index'], np.array(prepimg, dtype=np.float32))
    t1 = time.time()
    interpreter.invoke()
    print("elapsedtime =", time.time() - t1)
    outputs = interpreter.get_tensor(output_details[0]['index'])

    # Get a color palette
    palette = seg_image.getpalette()

    # Define index_void Back Ground
    index_void = 2

    # View
    output = outputs[0]
    res = np.argmax(output, axis=2)
    if index_void is not None:
        res = np.where(res == index_void, 0, res)
    image = Image.fromarray(np.uint8(res), mode="P")
    image.putpalette(palette)
    image = image.convert("RGB")
    image = image.resize((base_width, base_height))

    image.save("4.jpg")


