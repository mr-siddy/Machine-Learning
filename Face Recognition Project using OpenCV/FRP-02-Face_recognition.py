# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 15:24:58 2020

@author: mrsid
"""


# Recognise Faces using some classification algorithm - like Logistic, KNN, SVM etc.


# 1. load the training data (numpy arrays of all the persons)
		# x- values are stored in the numpy arrays
		# y-values we need to assign for each person
# 2. Read a video stream using opencv
# 3. extract faces out of it --> Testing purposes
# 4. use knn to find the prediction of face (int)
# 5. map the predicted id to name of the user 
# 6. Display the predictions on the screen - bounding box and name


import numpy as np
import cv2
import os

########## KNN CODE ############
def distance(v1, v2):
	# Eucledian 
	return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
	dist = []
	
	for i in range(train.shape[0]):
		# Get the vector and label
		ix = train[i, :-1]
		iy = train[i, -1]
		# Compute the distance from test point
		d = distance(test, ix)
		dist.append([d, iy])
	# Sort based on distance and get top k
	dk = sorted(dist, key=lambda x: x[0])[:k]
	# Retrieve only the labels
	labels = np.array(dk)[:, -1]
	
	# Get frequencies of each label
	output = np.unique(labels, return_counts=True)
	# Find max frequency and corresponding label
	index = np.argmax(output[1])
	return output[0][index]
################################
    
# Init camera

cap = cv2.VideoCapture(0)

# Face Detection

face_cascade =  cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

# counter

skip=0


dataset_path = './data/'
face_data = []
label = []

class_id=0 # labels for the given file

names = {} # mapping b/w id-name

#####      Data preparation   #####

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        
        # create a mapping b/w class_id and name
        names[class_id] = fx[:-4] #cut last 4 chaars ie .npy
        print("loaded"+fx)
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)
        
        #create labels of class 
        
        target = class_id*np.ones((data_item.shape[0],))
        class_id += 1
        label.append(target)
        
face_dataset = np.concatenate(face_data, axis =0)
face_label = np.concatenate(label, axis = 0 ).reshape((-1,1))

print(face_dataset.shape)
print(face_label.shape)

# concatenate x and y to single train

trainset = np.concatenate((face_dataset,face_label), axis = 1)
print(trainset.shape)
 

######     Testing     ######

while True:
    ret,frame = cap.read()
    if ret == False:
        continue
    
    faces = face_cascade.detectMultiScale(frame,1.3,5)
    
    for face in faces:
        x,y,w,h = face
        
        # get the face region of interest
        
        offset = 10 #pixels
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        
        # predicted  output
        
        out = knn (trainset, face_section.flatten())
        
        # display on screen the name and a rectangle around it
        pred_name = names[int(out)]
        cv2.putText(frame, pred_name,(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,255), 2)
        
        
    cv2.imshow("faces", frame)
    
    
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
    
    
cap.release()

cv2.destroyAllWindows()