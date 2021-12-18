#!/usr/bin/python3.9
import keras
import os
import tensorflow as tf
import scipy
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

epochs=10
xsize = 224
ysize=224
trdata = ImageDataGenerator(rescale=1.0/255.0)
traindata = trdata.flow_from_directory(directory="cats_and_dogs/train",target_size=(xsize,ysize),shuffle=True,batch_size=64,class_mode="binary")
#traindata = tf.keras.preprocessing.image_dataset_from_directory("cats_and_dogs/train/", labels="inferred",image_size=(xsize,ysize),label_mode="categorical")
tsdata = ImageDataGenerator(rescale=1.0/255.0)
testdata = tsdata.flow_from_directory(directory="cats_and_dogs/validation", target_size=(xsize,ysize),shuffle=False,batch_size=10,class_mode="binary")
#testdata = tf.keras.preprocessing.image_dataset_from_directory("cats_and_dogs/validation/", labels="inferred",image_size=(xsize,ysize),label_mode="categorical")

model = Sequential() # Sequential model since we are just passing the image input through layers

# First convolution layer
model.add(Conv2D(input_shape=(xsize,ysize,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# Second convolution layer
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# Third convolution layer
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# Fourth convolution layer
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=4096,activation="relu",kernel_initializer="he_uniform"))
model.add(Dense(units=4096,activation="relu",kernel_initializer="he_uniform"))
model.add(Dense(units=1, activation="sigmoid"))

opt = tf.keras.optimizers.Adam(learning_rate=0.001,decay=(0.001/epochs))
model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
#model.load_weights('vgg16_1.h5')
from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("vgg16_2.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='accuracy', min_delta=0, patience=20, verbose=1, mode='auto')
hist = model.fit(steps_per_epoch=30,x=traindata, validation_data= testdata, validation_steps=100,epochs=epochs,callbacks=[checkpoint,early])
import matplotlib.pyplot as plt
plt.plot(hist.history["accuracy"])
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])

plt.savefig('Training_hist.png')

