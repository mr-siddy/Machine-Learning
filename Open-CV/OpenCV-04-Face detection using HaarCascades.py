# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 11:25:06 2020

@author: mrsid
"""


import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
    
    ret,frame = cap.read()
     
    gray_frame =  cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    if ret == False:
        continue
    
    # give readed image to model 
    
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    
    # cv2.imshow("Gray video frame", gray_frame)
    
    # iterate over faces to draw rectangle
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h),(255,0,0),2)

    cv2.imshow("Video frame",frame)
 
    # wait for user input - q, then we'll stop the loop
    
    key_pressed = cv2.waitKey(1) & 0xFF
    
    if key_pressed == ord('q'):        #ord is used to get ascii value of letter
        break
    
cap.release()

cv2.destroyAllWindows()