import cv2
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, back = cap.read() #back is what the camera is reading and ret is saying that camera is reading
    if ret== True :
        cv2.imshow("image",back)
        if cv2.waitKey(5) == ord('q'):
            #saving the image
            cv2.imwrite('image.jpg',back)
            break

cap.release()
cv2.destroyAllWindows