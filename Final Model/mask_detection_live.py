from json import load
import cv2
import pickle
import numpy as np
from keras.models import load_model

###########
width = 640
height = 480
threshold = 0.65
#############

#Webcam Initializing
cap = cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


model = load_model('model.h5')

while True:
    success, frame = cap.read()
    img = np.asarray(frame)
    img = cv2.resize(img, (224,224))
    cv2.imshow('Processed Image',  frame)

    img = img.reshape(1,224,224,3)
    img = img/255

    #Prediction
    predictions = model.predict(img) #Prediction provides probabilities of two different classes (mask on and off)
    class_pred = np.argmax(predictions) #get the index of the highest probabilty 
    prob = np.amax(predictions) #get the value with the higher value 

    #Conditiona
    if class_pred == 1:
        print('Maks On', prob)

        cv2.putText(frame, 'Mask On', (round(w/2)-80,70),
        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2, cv2.LINE_4)
    
    else:
        print('Mask Off', prob)
        cv2.putText(frame, 'Mask Off', (round(w/2)-104,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_4)
    
    cv2.imshow('Processed Image', frame)

    #Webcam Exit method
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break