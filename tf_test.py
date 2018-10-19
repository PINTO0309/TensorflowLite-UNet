import sys, time
import numpy as np
import tensorflow as tf
from PIL import Image

if __name__ == '__main__':

    image     = Image.open("data_set/VOCdevkit/person/18418693150_c40831b00a_o.jpg")
    seg_image = Image.open("data_set/VOCdevkit/person/SegmentationClass/009649.png")
    print("image.size = ", image.size)

    base_width  = image.size[0]
    base_height = image.size[1]
    image.save("1.jpg")

    # resize image
    image = image.resize((256, 256), Image.ANTIALIAS)

    # delete alpha channel
    print("image.mode ==", image.mode)
    if image.mode == "RGBA":
        image = image.convert("RGB")

    # normalization
    image = np.asarray(image)
    prepimg = image / 255.0

    # 1 Channel -> 3 Channels convert
    if prepimg.ndim < 3:
        prepimg = prepimg[:, :, np.newaxis]
        prepimg = np.insert(prepimg, 1, prepimg[:,:,0], axis=2)
        prepimg = np.insert(prepimg, 2, prepimg[:,:,0], axis=2)

    # Read .pb file
    with tf.gfile.FastGFile("model/semanticsegmentation_frozen_person_32.pb", "rb") as f:
        graphdef = tf.GraphDef()
        graphdef.ParseFromString(f.read())
        _ = tf.import_graph_def(graphdef, name="")
    sess = tf.Session()

    # Segmentation
    t1 = time.time()
    outputs = sess.run("output/BiasAdd:0", {"input:0":[prepimg]})
    print("elapsedtime =", time.time() - t1)

    # Get a color palette
    palette = seg_image.getpalette()

    # Define index_void (len(DataSet.CATEGORY)-1)
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

    image.save("2.jpg")


