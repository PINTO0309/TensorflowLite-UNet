import tensorflow as tf
from util import loader as ld


class UNet:
    def __init__(self, size=(256, 256), l2_reg=None):
        self.model = self.create_model(size, l2_reg)

    @staticmethod
    def create_model(size, l2_reg):
        inputs = tf.placeholder(tf.float32, [None, size[0], size[1], 3], name="input")
        teacher = tf.placeholder(tf.float32, [None, size[0], size[1], len(ld.DataSet.CATEGORY)])
        is_training = tf.placeholder(tf.bool)

        # 1, 1, 3
        conv1_1 = UNet.conv(inputs, filters=32, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        conv1_2 = UNet.conv(conv1_1, filters=32, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        pool1 = UNet.pool(conv1_2)

        # 1/2, 1/2, 64
        conv2_1 = UNet.conv(pool1, filters=64, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        conv2_2 = UNet.conv(conv2_1, filters=64, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        pool2 = UNet.pool(conv2_2)

        # 1/4, 1/4, 128
        conv3_1 = UNet.conv(pool2, filters=128, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        conv3_2 = UNet.conv(conv3_1, filters=128, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        pool3 = UNet.pool(conv3_2)

        # 1/8, 1/8, 256
        conv4_1 = UNet.conv(pool3, filters=256, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        conv4_2 = UNet.conv(conv4_1, filters=256, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        pool4 = UNet.pool(conv4_2)

        # 1/16, 1/16, 512
        conv5_1 = UNet.conv(pool4, filters=512, l2_reg_scale=l2_reg)
        conv5_2 = UNet.conv(conv5_1, filters=512, l2_reg_scale=l2_reg)
        concated1 = tf.concat([UNet.conv_transpose(conv5_2, filters=256, l2_reg_scale=l2_reg), conv4_2], axis=3)

        conv_up1_1 = UNet.conv(concated1, filters=256, l2_reg_scale=l2_reg)
        conv_up1_2 = UNet.conv(conv_up1_1, filters=256, l2_reg_scale=l2_reg)
        concated2 = tf.concat([UNet.conv_transpose(conv_up1_2, filters=128, l2_reg_scale=l2_reg), conv3_2], axis=3)

        conv_up2_1 = UNet.conv(concated2, filters=128, l2_reg_scale=l2_reg)
        conv_up2_2 = UNet.conv(conv_up2_1, filters=128, l2_reg_scale=l2_reg)
        concated3 = tf.concat([UNet.conv_transpose(conv_up2_2, filters=64, l2_reg_scale=l2_reg), conv2_2], axis=3)

        conv_up3_1 = UNet.conv(concated3, filters=64, l2_reg_scale=l2_reg)
        conv_up3_2 = UNet.conv(conv_up3_1, filters=64, l2_reg_scale=l2_reg)
        concated4 = tf.concat([UNet.conv_transpose(conv_up3_2, filters=32, l2_reg_scale=l2_reg), conv1_2], axis=3)

        conv_up4_1 = UNet.conv(concated4, filters=32, l2_reg_scale=l2_reg)
        conv_up4_2 = UNet.conv(conv_up4_1, filters=32, l2_reg_scale=l2_reg)
        outputs = UNet.conv(conv_up4_2, filters=ld.DataSet.length_category(), kernel_size=[1, 1], activation=None, name="output")

        return Model(inputs, outputs, teacher, is_training)

    @staticmethod
    def conv(inputs, filters, kernel_size=[3, 3], activation=tf.nn.relu, l2_reg_scale=None, batchnorm_istraining=None, name=None):
        if l2_reg_scale is None:
            regularizer = None
        else:
            regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale)
        conved = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
            name=name
        )
        if batchnorm_istraining is not None:
            conved = UNet.bn(conved, batchnorm_istraining)

        return conved

    @staticmethod
    def bn(inputs, is_training):
        normalized = tf.layers.batch_normalization(
            inputs=inputs,
            axis=-1,
            momentum=0.9,
            epsilon=0.001,
            center=True,
            scale=True,
            training=is_training,
        )
        return normalized

    @staticmethod
    def pool(inputs):
        pooled = tf.layers.max_pooling2d(inputs=inputs, pool_size=[2, 2], strides=2)
        return pooled

    @staticmethod
    def conv_transpose(inputs, filters, l2_reg_scale=None):
        if l2_reg_scale is None:
            regularizer = None
        else:
            regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale)
        conved = tf.layers.conv2d_transpose(
            inputs=inputs,
            filters=filters,
            strides=[2, 2],
            kernel_size=[2, 2],
            padding='same',
            activation=tf.nn.relu,
            kernel_regularizer=regularizer
        )
        return conved


class Model:
    def __init__(self, inputs, outputs, teacher, is_training):
        self.inputs = inputs
        self.outputs = outputs
        self.teacher = teacher
        self.is_training = is_training
