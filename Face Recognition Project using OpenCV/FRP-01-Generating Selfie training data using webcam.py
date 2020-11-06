# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 12:14:43 2020

@author: mrsid
"""

# Write a Python Script that captures images from your webcam video stream
# Extracts all Faces from the image frame (using haarcascades)
# Stores the Face information into numpy arrays

# 1. Read and show video stream, capture images
# 2. Detect Faces and show bounding box (haarcascade)
# 3. Flatten the largest face image(gray scale) and save in a numpy array
# 4. Repeat the above for multiple people to generate training data


import cv2
import numpy as np

# Init camera

cap = cv2.VideoCapture(0)

# Face Detection

face_cascade =  cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

# counter

skip=0

# to store 10th face in a array in a data folder

face_data = []
dataset_path = './data/'
file_name = input("enter the name of person: ") 
while True:
    ret,frame = cap.read()
    
    if ret == False:
        continue
    
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #gray frame to save memory
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    #print(faces)
    
    # sorting faces based on largest frame using lambda function, reverse so that largest face comes to front
    
    faces = sorted(faces, key= lambda f:f[2]*f[3], reverse=True)
    
    # bounding box
    
    for face in faces:  #if we do normal sorting we can do faces[-1:] ie largest comes first acc to area
        x,y,w,h = face
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)
        
        # extract(crop out required face): region of intrest
        
        offset = 10 #pixels
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        
        # store every 10th face
         
        skip += 1
        if(skip%10==0):
            face_data.append(face_section)
            print(len(face_data))
        
        cv2.imshow("face section", face_section) 
        
    cv2.imshow("frame", frame)
    
     
    
    
    
    
    
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

# convert facelist array into a numpy array

face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))         
print(face_data.shape) 

# save this data into file system

np.save(dataset_path+file_name+'.npy', face_data)
print("data successfully saved at "+dataset_path+file_name+'.npy')    

cap.release()

cv2.destroyAllWindows()