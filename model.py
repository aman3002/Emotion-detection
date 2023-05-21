import pandas as pd
import tensorflow
import numpy as np
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import keras.layers
import PIL
a=ImageDataGenerator(rescale=1./255)
train=a.flow_from_directory('c:/users/aman/documents/images/images/train',target_size=(48,48),batch_size=64,color_mode="grayscale",class_mode="categorical")
test=a.flow_from_directory('c:/users/aman/documents/images/images/validation',target_size=(48,48),batch_size=64,color_mode="grayscale",class_mode="categorical")
model=tensorflow.keras.models.Sequential()
model.add(keras.layers.Conv2D(64,kernel_size=(5,5),activation="relu",input_shape=(48,48,1)))
model.add(keras.layers.Conv2D(64,kernel_size=(5,5),activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Conv2D(128,kernel_size=(3,3),activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Conv2D(256,kernel_size=(1,1),activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(300,activation="relu"))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(7,activation="softmax"))
model.compile(loss="categorical_crossentropy",optimizer=Adam(learning_rate=0.001),metrics=["accuracy"])
model.fit(train,epochs=32,validation_data=test)
model.save("c:/users/aman/documents/model.h5")
