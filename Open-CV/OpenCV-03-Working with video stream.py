# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 16:40:33 2020

@author: mrsid
"""


import cv2

cap = cv2.VideoCapture(0)

while True:
    
    ret,frame = cap.read()
    
    gray_frame =  cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    if ret == False:
        continue
    
    cv2.imshow("Video frame",frame)
    cv2.
    
    # wait for user input - q, then we'll stop the loop
    
    key_pressed = cv2.waitKey(1) & 0xFF
    
    if key_pressed == ord('q'):        #ord is used to get ascii value of letter
        break
    
cap.release()

cv2.destroyAllWindows()
    