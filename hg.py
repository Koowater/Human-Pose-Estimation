import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import Model as model
from tensorflow.keras import Input
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2D


'''
    HourGlass module에서 사용되는 convolution layer block입니다.
'''
class ConvBlock(model):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        
        if in_planes != out_planes:
            self.downsample = 1
            self.dsbn = BatchNormalization(momentum=0.9, epsilon=1e-5)
            self.dsrl = Activation('relu')
            self.dsconv = Conv2D(out_planes, (1, 1), padding='same', strides=(1, 1), use_bias=False)
        else:
            self.downsample = 0
            
        self.bn1 = BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.conv1 = Conv2D(int(out_planes / 2), (3, 3), padding='same', strides=(1, 1), use_bias=False)
        self.bn2 = BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.conv2 = Conv2D(int(out_planes / 4), (3, 3), padding='same', strides=(1, 1), use_bias=False)
        self.bn3 = BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.conv3 = Conv2D(int(out_planes / 4), (3, 3), padding='same', strides=(1, 1), use_bias=False)
        
    def call(self, input_tensor):
        residual = input_tensor
        
        out1 = self.bn1(input_tensor)
        out1 = tf.nn.relu(out1)
        out1 = self.conv1(out1)
        
        out2 = self.bn2(out1)
        out2 = tf.nn.relu(out2)
        out2 = self.conv2(out2)
        
        out3 = self.bn3(out2)
        out3 = tf.nn.relu(out3)
        out3 = self.conv3(out3)
        
        # residual
        out3 = tf.concat([out1, out2, out3], 3)        

        if self.downsample == 1:
            residual = self.dsbn(residual)
            residual = self.dsrl(residual)
            residual = self.dsconv(residual)

        #out3 += residual
        
        return tf.math.add(out3, residual)    

