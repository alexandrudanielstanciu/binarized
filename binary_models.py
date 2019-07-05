import numpy as np
import keras.backend as K
from tensorflow.keras import Sequential
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, MaxPooling2D,GlobalAveragePooling2D,Reshape,Dropout,ZeroPadding2D,Conv2D
from keras.layers import Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler,ModelCheckpoint,TensorBoard
from keras.utils import np_utils
from tensorflow.python.keras import backend as K
from binary_ops import binary_tanh as binary_tanh_op
from binary_layers import BinaryDense, BinaryConv2D,SeparableBinaryConv2D

# Variables
epsilon = 1e-6
momentum = 0.9
kernel_size = (3, 3)
kernel_size2 = (1, 1)
pool_size = (2, 2)
classes = 10
use_bias = False
H = 1.
kernel_lr_multiplier = 'Glorot'


def binary_tanh(x):
    return binary_tanh_op(x)

def BinaryVGG16():
  model = Sequential()
  # 64
  model.add(BinaryConv2D(64, kernel_size=kernel_size, input_shape=(32, 32, 3),
                         data_format='channels_last',
                         H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                         padding='same', use_bias=use_bias, name='conv1'))
  model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn1'))
  model.add(Activation(binary_tanh, name='act1'))
  model.add(BinaryConv2D(64, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                         data_format='channels_last',
                         padding='same', use_bias=use_bias, name='conv2'))
  model.add(MaxPooling2D(pool_size=pool_size, name='pool2', data_format='channels_last'))
  model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn2'))
  model.add(Activation(binary_tanh, name='act2'))
  # 128
  model.add(BinaryConv2D(128, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                         data_format='channels_last',
                         padding='same', use_bias=use_bias, name='conv3'))
  model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn3'))
  model.add(Activation(binary_tanh, name='act3'))
  model.add(BinaryConv2D(128, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                         data_format='channels_last',
                         padding='same', use_bias=use_bias, name='conv4'))
  model.add(MaxPooling2D(pool_size=pool_size, name='pool4', data_format='channels_last'))
  model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn4'))
  model.add(Activation(binary_tanh, name='act4'))
  # 256
  model.add(BinaryConv2D(256, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                         data_format='channels_last',
                         padding='same', use_bias=use_bias, name='conv5'))
  model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn5'))
  model.add(Activation(binary_tanh, name='act5'))
  model.add(BinaryConv2D(256, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                         data_format='channels_last',
                         padding='same', use_bias=use_bias, name='conv6'))
  model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn6'))
  model.add(Activation(binary_tanh, name='act6'))
  model.add(BinaryConv2D(256, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                         data_format='channels_last',
                         padding='same', use_bias=use_bias, name='conv7'))
  model.add(MaxPooling2D(pool_size=pool_size, name='pool7', data_format='channels_last'))
  model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn7'))
  model.add(Activation(binary_tanh, name='act7'))
  # 512
  model.add(BinaryConv2D(512, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                         data_format='channels_last',
                         padding='same', use_bias=use_bias, name='conv8'))
  model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn8'))
  model.add(Activation(binary_tanh, name='act8'))
  model.add(BinaryConv2D(512, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                         data_format='channels_last',
                         padding='same', use_bias=use_bias, name='conv9'))
  model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn9'))
  model.add(Activation(binary_tanh, name='act9'))
  model.add(BinaryConv2D(512, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                         data_format='channels_last',
                         padding='same', use_bias=use_bias, name='conv10'))
  model.add(MaxPooling2D(pool_size=pool_size, name='pool10', data_format='channels_last'))
  model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn10'))
  model.add(Activation(binary_tanh, name='act10'))
  #512
  model.add(BinaryConv2D(512, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                         data_format='channels_last',
                         padding='same', use_bias=use_bias, name='conv11'))
  model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn11'))
  model.add(Activation(binary_tanh, name='act11'))
  model.add(BinaryConv2D(512, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                         data_format='channels_last',
                         padding='same', use_bias=use_bias, name='conv12'))
  model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn12'))
  model.add(Activation(binary_tanh, name='act12'))
  model.add(BinaryConv2D(512, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                         data_format='channels_last',
                         padding='same', use_bias=use_bias, name='conv13'))
  model.add(MaxPooling2D(pool_size=pool_size, name='pool13', data_format='channels_last'))
  model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn13'))
  model.add(Activation(binary_tanh, name='act13'))
  #final
  model.add(Flatten())
  # dense1
  model.add(BinaryDense(4096, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, name='dense1'))
  model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn14'))
  model.add(Activation(binary_tanh, name='act14'))
  # dense2
  model.add(BinaryDense(4096, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, name='dense2'))
  model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn15'))
  model.add(Activation(binary_tanh, name='act15'))
  #dense3
  model.add(BinaryDense(classes, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, name='dense3'))
  model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn16'))

  return model

def addBottleneck(model_input, filters, kernel, t, s):
  model = model_input
  channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
  tchannel = K.int_shape(model_input.layers[-1].output)[channel_axis] *t
  model.add(BinaryConv2D(tchannel, kernel_size=(1,1), H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                     data_format='channels_last',
                     padding='same', use_bias=use_bias))
  model.add(BatchNormalization(axis=1))
  model.add(Activation(binary_tanh))

  model.add(SeparableBinaryConv2D(filters,kernel_size=kernel_size,strides=(s,s), H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                     data_format='channels_last',
                     padding='same', use_bias=use_bias))
  model.add(BatchNormalization(axis=1))
  #model.add(Activation(binary_tanh))

  return model

def InvertedResidualBlock(inputs, filters, kernel, t, strides, n):
    x = addBottleneck(inputs, filters, kernel, t, strides)
    for i in range(1, n):
        x = addBottleneck(x, filters, kernel, t, 1)
    return x

def BinaryMobilenetV2():
  model = Sequential() 
  model.add(BinaryConv2D(32, kernel_size=kernel_size, strides=(2, 2), H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                     data_format='channels_last',
                     padding='same', use_bias=use_bias,input_shape=(32, 32, 3)))
  model.add(BatchNormalization(axis=1))
  model.add(Activation(binary_tanh))
  model = InvertedResidualBlock(model, 16, (3, 3), t=1, strides=1, n=1)
  model = InvertedResidualBlock(model, 24, (3, 3), t=6, strides=2, n=2)
  model = InvertedResidualBlock(model, 32, (3, 3), t=6, strides=2, n=3)
  model = InvertedResidualBlock(model, 64, (3, 3), t=6, strides=2, n=4)
  model = InvertedResidualBlock(model, 96, (3, 3), t=6, strides=1, n=3)
  model = InvertedResidualBlock(model, 160, (3, 3), t=6, strides=2, n=3)
  model = InvertedResidualBlock(model, 320, (3, 3), t=6, strides=1, n=1)

  model.add(BinaryConv2D(1280, H=H,kernel_size=(1,1), kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias,strides=(1, 1)))
  model.add(BatchNormalization(axis=1))
  model.add(Activation(binary_tanh))

  model.add(Flatten()) 
  model.add(BinaryDense(classes, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias))
  model.add(BatchNormalization(axis=1))

  return model
