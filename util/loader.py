from PIL import Image
import numpy as np
import glob
import os
from util import image_augmenter as ia


class Loader(object):
    def __init__(self, dir_original, dir_segmented, init_size=(256, 256), one_hot=True):
        self._data = Loader.import_data(dir_original, dir_segmented, init_size, one_hot)

    def get_all_dataset(self):
        return self._data

    def load_train_test(self, train_rate=0.85, shuffle=True, transpose_by_color=False):
        """
        `Load datasets splited into training set and test set.
        Args:
            train_rate (float): Training rate.
            shuffle (bool): If true, shuffle dataset.
            transpose_by_color (bool): If True, transpose images for chainer. [channel][width][height]
        Returns:
            Training Set (Dataset), Test Set (Dataset)
        """
        if train_rate < 0.0 or train_rate > 1.0:
            raise ValueError("train_rate must be from 0.0 to 1.0.")
        if transpose_by_color:
            self._data.transpose_by_color()
        if shuffle:
            self._data.shuffle()

        train_size = int(self._data.images_original.shape[0] * train_rate)
        data_size = int(len(self._data.images_original))
        train_set = self._data.perm(0, train_size)
        test_set = self._data.perm(train_size, data_size)

        return train_set, test_set

    @staticmethod
    def import_data(dir_original, dir_segmented, init_size=None, one_hot=True):
        # Generate paths of images to load
        paths_original, paths_segmented = Loader.generate_paths(dir_original, dir_segmented)

        # Extract images to ndarray using paths
        images_original, images_segmented = Loader.extract_images(paths_original, paths_segmented, init_size, one_hot)

        # Get a color palette
        image_sample_palette = Image.open(paths_segmented[0])
        palette = image_sample_palette.getpalette()

        return DataSet(images_original, images_segmented, palette,
                       augmenter=ia.ImageAugmenter(size=init_size, class_count=len(DataSet.CATEGORY)))

    @staticmethod
    def generate_paths(dir_original, dir_segmented):
        paths_original = glob.glob(dir_original + "/*")
        paths_segmented = glob.glob(dir_segmented + "/*")
        if len(paths_original) == 0 or len(paths_segmented) == 0:
            raise FileNotFoundError("Could not load images.")
        filenames = list(map(lambda path: path.split(os.sep)[-1].split(".")[0], paths_segmented))
        paths_original = list(map(lambda filename: dir_original + "/" + filename + ".jpg", filenames))

        return paths_original, paths_segmented

    @staticmethod
    def extract_images(paths_original, paths_segmented, init_size, one_hot):
        images_original, images_segmented = [], []

        # Load images from directory_path using generator
        print("Loading original images", end="", flush=True)
        for image in Loader.image_generator(paths_original, init_size, antialias=True):
            images_original.append(image)
            if len(images_original) % 200 == 0:
                print(".", end="", flush=True)
        print(" Completed", flush=True)
        print("Loading segmented images", end="", flush=True)
        for image in Loader.image_generator(paths_segmented, init_size, normalization=False):
            images_segmented.append(image)
            if len(images_segmented) % 200 == 0:
                print(".", end="", flush="True")
        print(" Completed")
        assert len(images_original) == len(images_segmented)

        # Cast to ndarray
        images_original = np.asarray(images_original, dtype=np.float32)
        images_segmented = np.asarray(images_segmented, dtype=np.uint8)

        # Change indices which correspond to "void" from 255
        images_segmented = np.where((images_segmented != 15) & (images_segmented != 255), 0, images_segmented)
        images_segmented = np.where(images_segmented == 15, 1, images_segmented)
        images_segmented = np.where(images_segmented == 255, len(DataSet.CATEGORY)-1, images_segmented)

        # One hot encoding using identity matrix.
        if one_hot:
            print("Casting to one-hot encoding... ", end="", flush=True)
            identity = np.identity(len(DataSet.CATEGORY), dtype=np.uint8)
            images_segmented = identity[images_segmented]
            print("Done")
        else:
            pass

        return images_original, images_segmented

    @staticmethod
    def cast_to_index(ndarray):
        return np.argmax(ndarray, axis=2)

    @staticmethod
    def cast_to_onehot(ndarray):
        identity = np.identity(len(DataSet.CATEGORY), dtype=np.uint8)
        return identity[ndarray]

    @staticmethod
    def image_generator(file_paths, init_size=None, normalization=True, antialias=False):
        """
        `A generator which yields images deleted an alpha channel and resized.
        Args:
            file_paths (list[string]): File paths you want load.
            init_size (tuple(int, int)): If having a value, images are resized by init_size.
            normalization (bool): If true, normalize images.
            antialias (bool): Antialias.
        Yields:
            image (ndarray[width][height][channel]): Processed image
        """
        for file_path in file_paths:
            if file_path.endswith(".png") or file_path.endswith(".jpg"):
                # open a image
                image = Image.open(file_path)
                # to square
                image = Loader.crop_to_square(image)
                # resize by init_size
                if init_size is not None and init_size != image.size:
                    if antialias:
                        image = image.resize(init_size, Image.ANTIALIAS)
                    else:
                        image = image.resize(init_size)
                # delete alpha channel
                if image.mode == "RGBA":
                    image = image.convert("RGB")
                image = np.asarray(image)
                if normalization:
                    image = image / 255.0
                yield image

    @staticmethod
    def crop_to_square(image):
        size = min(image.size)
        left, upper = (image.width - size) // 2, (image.height - size) // 2
        right, bottom = (image.width + size) // 2, (image.height + size) // 2
        return image.crop((left, upper, right, bottom))


class DataSet(object):
    CATEGORY = (
        "ground",
        "person",
        "void"
    )

    def __init__(self, images_original, images_segmented, image_palette, augmenter=None):
        assert len(images_original) == len(images_segmented), "images and labels must have same length."
        self._images_original = images_original
        self._images_segmented = images_segmented
        self._image_palette = image_palette
        self._augmenter = augmenter

    @property
    def images_original(self):
        return self._images_original

    @property
    def images_segmented(self):
        return self._images_segmented

    @property
    def palette(self):
        return self._image_palette

    @property
    def length(self):
        return len(self._images_original)

    @staticmethod
    def length_category():
        return len(DataSet.CATEGORY)

    def print_information(self):
        print("****** Dataset Information ******")
        print("[Number of Images]", len(self._images_original))

    def __add__(self, other):
        images_original = np.concatenate([self.images_original, other.images_original])
        images_segmented = np.concatenate([self.images_segmented, other.images_segmented])
        return DataSet(images_original, images_segmented, self._image_palette, self._augmenter)

    def shuffle(self):
        idx = np.arange(self._images_original.shape[0])
        np.random.shuffle(idx)
        self._images_original, self._images_segmented = self._images_original[idx], self._images_segmented[idx]

    def transpose_by_color(self):
        self._images_original = self._images_original.transpose(0, 3, 1, 2)
        self._images_segmented = self._images_segmented.transpose(0, 3, 1, 2)

    def perm(self, start, end):
        end = min(end, len(self._images_original))
        return DataSet(self._images_original[start:end], self._images_segmented[start:end], self._image_palette,
                       self._augmenter)

    def __call__(self, batch_size=20, shuffle=True, augment=True):
        """
        `A generator which yields a batch. The batch is shuffled as default.
        Args:
            batch_size (int): batch size.
            shuffle (bool): If True, randomize batch datas.
        Yields:
            batch (ndarray[][][]): A batch data.
        """

        if batch_size < 1:
            raise ValueError("batch_size must be more than 1.")
        if shuffle:
            self.shuffle()

        for start in range(0, self.length, batch_size):
            batch = self.perm(start, start+batch_size)
            if augment:
                assert self._augmenter is not None, "you have to set an augmenter."
                yield self._augmenter.augment_dataset(batch, method=[ia.ImageAugmenter.NONE, ia.ImageAugmenter.FLIP])
            else:
                yield batch


if __name__ == "__main__":
    dataset_loader = Loader(dir_original="../data_set/VOCdevkit/person/JPEGImages",
                            dir_segmented="../data_set/VOCdevkit/person/SegmentationClass")
    train, test = dataset_loader.load_train_test()
    train.print_information()
    test.print_information()