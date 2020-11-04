import cv2

img = cv2.imread('dog.png')
gray = cv2.imread('dog.png', cv2.IMREAD_GRAYSCALE)
                  
cv2.imshow('Dog Image', img)
cv2.imshow('Gray Dog image', gray)

cv2.waitKey(0)
cv2.destroyAllWindows()