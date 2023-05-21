import os
import cv2
from keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
model=load_model("c:/users/aman/documents/model.h5")
face=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
while True:
    t,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face.detectMultiScale(gray,1.32,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi=gray[x:x+w,y:y+h]
        roi=cv2.resize(roi,(48,48),interpolation=cv2.INTER_AREA)
        imp=img_to_array(roi)
        imp=np.expand_dims(imp,axis=0)
        imp/=255
        p=model.predict(imp)
        ma=np.argmax(p[0])
        clas=['angry','disgust','fear','happy','neutral','sad','suprise']
        pe=clas[ma]
        cv2.putText(img,pe,(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.imshow('img',img)
    if(cv2.waitKey(1) & 0xff==ord('q')):
       break
cap.release()

