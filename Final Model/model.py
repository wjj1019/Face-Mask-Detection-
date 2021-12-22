from keras.preprocessing.image import ImageDataGenerator
import cv2
from keras.models import Sequential
from keras.layers import AveragePooling2D,Input,Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense,Dropout
from keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import imutils
import numpy as np
# Get Train and Test Data (into array form), using glob library makes the work easier
import glob
import random
import pandas as pd


#Train Dataset 
train_mon = [cv2.imread(file) for file in glob.glob("./Dataset/train/with_mask/*.jpg")]
train_moff = [cv2.imread(file) for file in glob.glob("./Dataset/train/without_mask/*.jpg")]
#Test Dataset
test_mon = [cv2.imread(file) for file in glob.glob("./Dataset/test/with_mask/*.jpg")]
test_moff = [cv2.imread(file) for file in glob.glob("./Dataset/test/without_mask/*.jpg")]

# Resizing every image and assign label
train_1 = []
for i in range(len(train_mon)):
    resized = cv2.resize(train_mon[i], (224,224), interpolation = cv2.INTER_LINEAR)
    train_1.append((resized, 1))
    
train_0 = []
for i in range(len(train_moff)):
    resized = cv2.resize(train_moff[i], (224,224), interpolation = cv2.INTER_LINEAR)
    train_0.append((resized, 0))

test_1 = []
for i in range(len(test_mon)):
    resized = cv2.resize(test_mon[i], (224,224), interpolation = cv2.INTER_LINEAR)
    test_1.append((resized, 1))
    
test_0 = []
for i in range(len(test_moff)):
    resized = cv2.resize(test_moff[i], (224,224), interpolation = cv2.INTER_LINEAR)
    test_0.append((resized, 0))


#Combine Binary class Train and Test dataset and shuffle them 
train = train_1 + train_0
test = test_1 + test_0

random.shuffle(train)
random.shuffle(test)

X_train = np.array([train[i][:-1][0] for i in range(len(train))])
y_train = pd.get_dummies([train[i][-1] for i in range(len(train))])

X_test = np.array([test[i][:-1][0] for i in range(len(test))])
y_test = pd.get_dummies([test[i][-1] for i in range(len(test))])

#Normalizing the data - To have mean of 0 
X_train = X_train/255
X_test = X_test/255

print(X_train.shape, y_train.shape)


from tensorflow.keras.applications import MobileNetV2

#Include_top = False allows removal of Fully Connected layer so that we can replace it wiht out FC layer
baseModel = MobileNetV2(weights='imagenet', include_top = False, input_shape = (224,224,3))

top = baseModel.output
top = AveragePooling2D(pool_size = (7,7), padding="same")(top)
top = Flatten(name = 'flatten')(top)
top = Dense(128, activation = 'relu')(top)
top = Dropout(0.5)(top)
top = Dense(2, activation = 'softmax')(top)

model = Model(inputs = baseModel.input, outputs = top)

#Freezing the Convolutional Layers (pretrained)
for layer in baseModel.layers:
    layer.trainable = False

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(X_train, y_train, batch_size = 10, validation_data=(X_test, y_test),
         epochs = 3)

model.save('model.h5')