import numpy as np
import pickle
import keras
import keras.backend as K
import pandas as pd
from keras.datasets import cifar10
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler,ModelCheckpoint,TensorBoard
from keras.utils import np_utils
from binary_ops import binary_tanh
from binary_models import BinaryVGG16,BinaryMobilenetV2

# Variabile
batch_size = 50
epochs = 30 
classes = 10
img_rows, img_cols = 32, 32
lr_start = 1e-3
lr_end = 1e-4
lr_decay = (lr_end / lr_start)**(1. / epochs)

# Incarcarea bazei de date 
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = keras.utils.to_categorical(Y_train, classes) * 2 -1
Y_test = keras.utils.to_categorical(Y_test, classes) * 2 -1

#model = BinaryVGG16()
model = BinaryMobilenetV2()
#model.load_weights("./weights.hdf5")

opt = Adam() 
model.compile(loss='squared_hinge', optimizer=opt, metrics=['acc'])
model.summary()

#Callbacks
tensorboard = TensorBoard(log_dir='./logs')
filepath="./checkpoint-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
callbacks_list = [lr_scheduler,checkpoint,tensorboard]

hist = model.fit(X_train, Y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(X_test, Y_test),
                    callbacks=callbacks_list)

model.save_weights('./Model.h5')
print("Done")

df = pd.DataFrame.from_dict(hist.history)
df.to_csv('./hist.csv', encoding='utf-8', index=False)

fpkl= open("./modelweights.pkl", 'wb')      
pickle.dump(model.get_weights(), fpkl, protocol= pickle.HIGHEST_PROTOCOL)
fpkl.close()