import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import Model


class Bottleneck(tf.keras.Model):
    expansion = 4
    def __init__(self, filters, strides=1, downsample=None,
                dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__(name='')

        self.conv2a = layers.Conv2D(filters, (1, 1), padding = 'same')
        self.bn2a = layers.BatchNormalization()

        self.conv2b = layers.Conv2D(filters, (3,3), padding='same', 
            dilation_rate = dilation)
        self.bn2b = layers.BatchNormalization()

        self.conv2c = layers.Conv2D(filters, (1, 1), padding = 'same')
        self.bn2c = layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        # print(input_tensor.shape)
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)
        if self.downsample is not None:
            input_tensor = self.downsample(x)
             
        x += input_tensor
        return tf.nn.relu(x)


class BasicBlock(tf.keras.Model):
    expansion = 1
    def __init__(self, filters, strides=1, downsample=None,
                dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = layers.BatchNormalization
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = layers.Conv2D(filters, kernel_size = (3,3), strides = 1,
            padding = 'same')
        self.bn1 = norm_layer()
        self.relu1 = layers.ReLU()
        self.conv2 = layers.Conv2D(filters, kernel_size = (3,3), strides = 1,
            padding = 'same')
        self.bn2 = norm_layer()
        self.relu2 = layers.ReLU()
        self.downsample = downsample

    def call(self, x, training=False):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu2(out)

        return out



class ResNet(Model):
    def __init__(self, block, layer_list, num_classes=1000, zero_init_residual=False,
                replace_stride_with_dilation=None,
                norm_layer=None):
        super(ResNet, self).__init__()
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]        
        self._norm_layer = norm_layer
        self.dilation = 1
        self._norm_layer = layers.BatchNormalization

        self.conv1 = layers.Conv2D(64, kernel_size = 7, strides = 2, padding = 'same',
            use_bias = False)
        self.relu = layers.ReLU()
        self.bn1 = layers.BatchNormalization()
        self.maxpool = layers.MaxPooling2D(pool_size = (3,3), strides = 2, padding = 'same')
        self.layer1 = self._make_layer(block, 64, layer_list[0])
        self.layer2 = self._make_layer(block, 64, layer_list[1], strides = 2, 
            dilate = replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layer_list[2], strides = 2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 64, layer_list[3], strides = 2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = layers.GlobalAveragePooling2D()
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(num_classes)



    def _make_layer(self, block, filters, blocks, strides=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= strides
            strides = 1

        if strides != 1:
            downsample = tf.keras.Sequential(
                    [layers.Conv2D(filters*block.expansion,strides, padding = "same"),
                    layers.BatchNormalization()]
                )

        layer = []
        layer.append(block(filters, strides, downsample, previous_dilation,
            norm_layer))
        for _ in range(1,blocks):
            layer.append(block(filters, strides, downsample, previous_dilation,
                norm_layer))

        return tf.keras.Sequential(layer)


    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x    


def _resnet(block, layers_list, **kwargs):
    model = ResNet(block, layers_list, **kwargs)
    return model


def resnet18(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet50(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet(Bottleneck, [3, 8, 36, 3], **kwargs)


if __name__ == "__main__":
    model = resnet18()
    out = model(tf.ones([10,224,224,3]))
    print(out)