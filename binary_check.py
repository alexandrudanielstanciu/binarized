import numpy as np
import keras
import keras.backend as K
import pandas as pd
from keras.datasets import cifar10
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler,ModelCheckpoint,TensorBoard
from keras.utils import np_utils
from binary_models import BinaryVGG16,BinaryMobilenetV2
from tensorflow.python.keras import backend as K
K.clear_session()

classes = 10
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = keras.utils.to_categorical(Y_train, classes) * 2 -1
Y_test = keras.utils.to_categorical(Y_test, classes) * 2 -1

# model = BinaryVGG16()
# model.load_weights("./vgg.hdf5")

model = BinaryMobilenetV2()
model.load_weights("./mobilenet.hdf5")


opt = Adam() 
model.compile(loss='squared_hinge', optimizer=opt, metrics=['acc'])
#model.summary()

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

for layer in model.layers:
    print(layer.name)
    # if("activation" in layer.name):
    #     print(layer.name)
    #     print(layer.get_weights())
    #layerDense = model.layers[-2]
    if("conv2d" in layer.name):
        #print(layer.name)
        layerWeights2 = layer.get_weights()
        #print(layerWeights2)
        for layerWeights in layerWeights2:
            layerWeights[layerWeights < 0] = -1
            layerWeights[layerWeights > 0] = 1
        #print(layerWeights2)
        layer.set_weights(layerWeights2)

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score after binarization:', score[0])
print('Test accuracy after binarization:', score[1])
