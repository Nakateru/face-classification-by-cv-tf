"""
ResNet (Residual Network) 残差网络
unit 一般取2-3层
残差块 res_block ── ... ── res_block
Deeper residual module
x
↓ ─────────────────┐
convolution layer1 │ convolution layer: conv + bn(BatchNormalization) 批量标准化:提高梯度的收敛程度，加快训练速度
↓ ReLU             │x :Identity Function : down sample(下采样)
convolution layer2 │
↓ F(x)             │  F(x) : ResNet Function 残差函数
⊕←─────────────────┘
H(x) = F(x) + x
随着网络的加深，优化效果反而越差，测试数据和训练数据的准确率反而降低。(20层以上)
这是由于网络的加深会造成梯度爆炸和梯度消失。
对输入数据和中间层的数据进行归一化操作，这种方法可以保证网络在反向传播中采用随机梯度下降（SGD），
从而让网络达到收敛。

inputs  outputs  kernels   element numbers
256     256         3*3     ~600K
To reduce element numbers
256     64          1*1     ~16K
64      64          3*3     ~36K
64      256         1*1     ~16K
total:~70K (about 1/9 of 600K)
参数量少了 => 堆叠层数可能


ResNet-18:
image 32*32
↓
3*3 conv,64  # first conv(预处理层)  + 16 conv + (avg.pool + fc) = 1 + 16 + 1 = 18
3*3 conv,64┐
3*3 conv,64┘ 32*32
3*3 conv,64┐
3*3 conv,64┘
3*3 conv,128┐
3*3 conv,128┘ 16*16
3*3 conv,128┐
3*3 conv,128┘
3*3 conv,256┐
3*3 conv,256┘ 8*8
3*3 conv,256┐
3*3 conv,256┘
3*3 conv,512┐
3*3 conv,512┘ 4*4
3*3 conv,512┐
3*3 conv,512┘
↓
avg.pool       1*1
↓
fc 10

"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, Model


class BasicBlock(layers.Layer):
    def __init__(self, filter_num, strides=1):
        super(BasicBlock, self).__init__()
        # convolution layer1
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=strides, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        # convolution layer2
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        if strides != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=strides))
        else:
            self.downsample = lambda x: x

    @tf.function
    def call(self, inputs, training=None):
        # [b, h, w, c]
        # convolution layer1
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        # convolution layer2
        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(inputs)

        output = layers.add([out, identity])
        output = tf.nn.relu(output)

        return output


class ResNet(Model):
    def __init__(self, layer_dims, num_classes=100):
        # [2, 2, 2, 2] 4个resblock,每个包含2个BasicBlock,每个2层convolution (共16层convolution)
        super(ResNet, self).__init__()

        # first conv
        self.stem = Sequential([layers.Conv2D(64, (3, 3), strides=(1, 1)),
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')])

        # 16 conv
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], strides=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], strides=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], strides=2)

        # output: [b, 512, h, w]
        self.avgpool = layers.GlobalAveragePooling2D()

        self.fc = layers.Dense(num_classes)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.stem(inputs)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # [b, c]
        x = self.avgpool(x)

        # [b, 100]
        x = self.fc(x)

        return x

    # 当 blocks=2， 1 个 build_resblock 包含2个BasicBlock（共4层conv 前2层有下采样）
    def build_resblock(self, filter_num, blocks, strides=1):
        res_blocks = Sequential()

        # may down sample
        res_blocks.add(BasicBlock(filter_num, strides))  # 包含2层conv

        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, strides=1))  # 包含2层conv

        return res_blocks


def resnet18():  # 1+(2+2+2+2)*2+1=18
    return ResNet([2, 2, 2, 2])


def resnet34():  # 1+(3+4+6+3)*2+1=34
    return ResNet([3, 4, 6, 3])
