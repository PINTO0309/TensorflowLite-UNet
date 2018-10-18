import random
import tensorflow as tf
import numpy as np
from util import loader as ld


class ImageAugmenter:
    NONE = 0
    FLIP = 1
    BRIGHTNESS = 2
    HUE = 3
    SATURATION = 4

    NUMBER_OF_AUGMENT = 5

    def __init__(self, size, class_count):
        self._sess = tf.Session()
        self._class_count = class_count
        self._width, self._height = size[0], size[1]
        self._ph_original = tf.placeholder(tf.float32, [size[0], size[1], 3])
        self._ph_segmented = tf.placeholder(tf.float32, [size[0], size[1], class_count])
        self._operation = {}
        self.init_graph()

    def augment_dataset(self, dataset, method=None):
        input_processed = []
        output_processed = []
        for ori, seg in zip(dataset.images_original, dataset.images_segmented):
            ori_processed, seg_processed = self.augment(ori, seg, method)
            input_processed.append(ori_processed)
            output_processed.append(seg_processed)

        return ld.DataSet(np.asarray(input_processed), np.asarray(output_processed), dataset.palette)

    def augment(self, image_in, image_out, method=None):
        if method is None:
            idx = random.randrange(ImageAugmenter.NUMBER_OF_AUGMENT)
        else:
            assert len(method) <= ImageAugmenter.NUMBER_OF_AUGMENT, "method is too many."
            if ImageAugmenter.NONE not in method:
                method.append(ImageAugmenter.NONE)
            idx = random.choice(method)

        op = self._operation[idx]
        return self._sess.run([op["original"], op["segmented"]], feed_dict={self._ph_original: image_in,
                                                                            self._ph_segmented: image_out})

    def init_graph(self):
        self._operation[ImageAugmenter.NONE] = {"original": self._ph_original, "segmented": self._ph_segmented}
        self._operation[ImageAugmenter.FLIP] = self.flip()
        self._operation[ImageAugmenter.BRIGHTNESS] = self.brightness()
        self._operation[ImageAugmenter.HUE] = self.hue()
        self._operation[ImageAugmenter.SATURATION] = self.saturation()

    def flip(self):
        image_out_index = tf.argmax(self._ph_segmented, axis=2)
        image_out_index = tf.reshape(image_out_index, (self._width, self._height, 1))
        image_in_processed = tf.image.flip_left_right(self._ph_original)
        image_out_processed = tf.image.flip_left_right(image_out_index)
        image_out_processed = tf.one_hot(image_out_processed, depth=len(ld.DataSet.CATEGORY), dtype=tf.float32)
        image_out_processed = tf.reshape(image_out_processed, (self._width, self._height, len(ld.DataSet.CATEGORY)))
        return {"original": image_in_processed, "segmented": image_out_processed}

    def brightness(self):
        max_delta = 0.3
        image_in_processed = tf.image.random_brightness(self._ph_original, max_delta)
        return {"original": image_in_processed, "segmented": self._ph_segmented}

    def hue(self):
        max_delta = 0.5
        image_in_processed = tf.image.random_hue(self._ph_original, max_delta)
        return {"original": image_in_processed, "segmented": self._ph_segmented}

    def saturation(self):
        lower, upper = 0.0, 1.2
        image_in_processed = tf.image.random_saturation(self._ph_original, lower, upper)
        return {"original": image_in_processed, "segmented": self._ph_segmented}


if __name__ == "__main__":
    pass
