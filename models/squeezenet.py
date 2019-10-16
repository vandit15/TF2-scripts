import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import Model


class Fire(tf.keras.Model):
    def __init__(self, squeeze_planes, expand1x1_planes,
                expand3x3_planes):
        super(Fire, self).__init__()
        self.squeeze = layers.Conv2D(squeeze_planes, (1,1))
        self.squeeze_activation = layers.ReLU()
        self.expand1x1 = layers.Conv2D(expand1x1_planes, (1,1))
        self.expand1x1_activation = layers.ReLU()
        self.expand3x3 = layers.Conv2D(expand3x3_planes,(3,3), padding = 'same')
        self.expand3x3_activation = layers.ReLU()
        self.concatenate = layers.Concatenate(axis = 1)

    def call(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return self.concatenate([self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))])


class SqueezeNet(tf.keras.Model):
    def __init__(self, version = '1_0', num_classes = 1000):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        if version == '1_0':
            self.features = tf.keras.Sequential([
                layers.Conv2D(96, (7,7), strides = 2),
                layers.ReLU(),
                layers.MaxPool2D(pool_size = (3,3), strides = 2),
                Fire(16,64,64),
                Fire(16,64,64),
                Fire(32,128,128),
                layers.MaxPool2D(pool_size = (3,3), strides = 2),
                Fire(32, 128,128),
                Fire(48, 192, 192),
                Fire(48, 192, 192),
                layers.MaxPool2D(pool_size = (3,3), strides = 2),
                Fire(64, 256, 256)
                ])
        elif version == '1_1':
            self.features = tf.keras.Sequential([
                layers.Conv2D(64, (3,3), strides = 2),
                layers.ReLU(),
                layers.MaxPool2D(pool_size = (3,3), strides = 2),
                Fire(16,64,64),
                Fire(16,64,64),
                layers.MaxPool2D(pool_size = (3,3), strides = 2),
                Fire(32,128,128),
                Fire(32, 128,128),
                layers.MaxPool2D(pool_size = (3,3), strides = 2),
                Fire(48, 192, 192),
                Fire(48, 192, 192),
                Fire(64, 256, 256),
                Fire(64, 256, 256)
                ])

        final_conv = layers.Conv2D(self.num_classes, (1,1))
        self.classifier = tf.keras.Sequential([
            layers.Dropout(rate = 0.5),
            final_conv,
            layers.ReLU(),
            layers.GlobalAveragePooling2D()
            ])
        self.flatten = layers.Flatten()

    def call(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = self.flatten(x)
        return x


def _squeezenet(version, **kwargs):
    model = SqueezeNet(version, **kwargs)
    return model


def squeezenet1_0(**kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    """
    return _squeezenet('1_0',**kwargs)


def squeezenet1_1(**kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    """
    return _squeezenet('1_1', **kwargs)


if __name__ == "__main__":
    model = squeezenet1_0()
    out = model(tf.ones([10,224,224,3]))
    print(out)