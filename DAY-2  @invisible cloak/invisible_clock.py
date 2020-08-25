import cv2
import numpy as np

cap = cv2.VideoCapture(0)
back = cv2.imread('./image.jpg')

while cap.isOpened():
    #take each frame
    ret, frame = cap.read()
    # we will use HSV model coz it describes how human eye tends to see colors
    if ret:
        # to convert rgb to hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # to convert rgb to hsv
        #cv2.imshow("hsv",hsv)
        #to get the hsv value lower:hue-10,100,100 heigher:h+10,100,100
        red = np.uint8([[[0,0,255]]]) #bgr value
        hsv_red = cv2.cvtColor(red,cv2.COLOR_BGR2HSV)
        #get hsv of red from bgr
        #print(hsv_red)

        #threshold the hsv to get  only red colors
        l_red = np.array([0,100,100])
        u_red = np.array([0,255,255])
        
        mask = cv2.inRange(hsv, l_red, u_red)
        #cv2.imshow("mask",mask )

        #all things red
        part1 = cv2.bitwise_and(back,back,mask = mask)
        #cv2.imshow("part1",part1)

        mask = cv2.bitwise_not(mask)

        #all things not red
        part2 = cv2.bitwise_and(frame, frame, mask = mask)
        #cv2.imshow("mask", part2)
        
        cv2.imshow("cloak", part1+part2)


        if cv2.waitKey(5) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()