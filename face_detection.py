import cv2
import os
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('img4.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #transform to gray
faces = face_cascade.detectMultiScale(gray,1.1,10)#detects faces in the image and draws rectangle around the faces

for (x,y,w,h) in faces :
    cv2.rectangle(img , (x,y) , (x+w,y+h) , (0,0,255), 4)

imgCropped = img[y+3:y+h-2,x+3:x+w-2]
grayImage = cv2.cvtColor(imgCropped, cv2.COLOR_BGR2GRAY)
(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 100, 255, cv2.THRESH_BINARY)
#produce the output
cv2.imshow('face detected', img)
cv2.imshow('gray', grayImage)
cv2.imshow('cropped face resized', imgCropped)
cv2.imshow('Black white image', blackAndWhiteImage)
cv2.waitKey()

