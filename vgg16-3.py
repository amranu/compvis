#!/usr/bin/python3.9
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

xsize = 224
ysize = 224
 
# define cnn model
# load model
model = VGG16(include_top=False, input_shape=(xsize, ysize, 3))
# mark loaded layers as not trainable
for layer in model.layers:
    layer.trainable = False
# add new classifier layers
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
output = Dense(1, activation='sigmoid')(class1)
# define new model
model = Model(inputs=model.inputs, outputs=output)
model.summary()
# compile model
opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
 
generator = ImageDataGenerator(featurewise_center=True)
generator.mean = [123.68, 116.779, 103.939]
training = generator.flow_from_directory('cats_and_dogs/train',class_mode='binary', batch_size=64, target_size=(xsize, ysize))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="cats_and_dogs/validation/", target_size=(224,224),batch_size=10,class_mode="binary")

# fit model
checkpoint = ModelCheckpoint("vgg16_3.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
hist= model.fit(validation_data=testdata, x=training, steps_per_epoch=3, epochs=10, verbose=1,validation_steps=10,callbacks=[checkpoint])
# save model
model.save('vgg16_3.h5')

import matplotlib.pyplot as plt
plt.plot(hist.history["accuracy"])
plt.plot(hist.history['val_accuracy'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy"])
plt.savefig('Accuracy_hist.png')
plt.clf()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["loss","Validation Loss"])
 
plt.savefig('Training_hist3.png')
