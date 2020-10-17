import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('input1.JPG')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #transform to gray
faces = face_cascade.detectMultiScale(gray,1.1,4)#detects faces in the image and draws rectangle around the faces

for (x,y,w,h) in faces :
    cv2.rectangle(img , (x,y) , (x+w,y+h) , (0,0,255), 3)


#produce the output
cv2.imshow('img', img)
cv2.waitKey()