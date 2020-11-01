import cv2
import numpy as np
import dlib

from isolated_features import IsolatedFeatures

url = 'input3.jpg'

img = cv2.imread(url)
img = cv2.resize(img, (320, 400),  interpolation = cv2.INTER_AREA)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector()
predictor =  dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

face = detector(gray)[0]

landmarks = predictor(gray, face)

features = IsolatedFeatures(landmarks)

poly = features.LeftCheekNoBeard
facePoly = features.FaceNoBeard

mask = np.zeros((img.shape[0], img.shape[1]))
cv2.fillConvexPoly(mask, poly, 1)
mask = mask.astype(np.bool)

faceMask = np.zeros((img.shape[0], img.shape[1]))
cv2.fillConvexPoly(faceMask, facePoly, 1)
faceMask = faceMask.astype(np.bool)

SUM_pix_gray = np.sum(gray[faceMask])
AVG_pix_gray = SUM_pix_gray / np.sum(faceMask == 1)

(thresh, blackAndWhiteImage) = cv2.threshold(gray, AVG_pix_gray - 15, 255, cv2.THRESH_BINARY)

#get pixels
white_pix = int((np.sum(blackAndWhiteImage[mask] == 255) / np.sum(mask == 1)) * 100)
black_pix = int((np.sum(blackAndWhiteImage[mask] == 0) / np.sum(mask == 1)) * 100)
print('URL:', url)
print('AVG:', AVG_pix_gray)
print('white:', white_pix, '%')
print('black:', black_pix, '%')

outBW = np.zeros_like(blackAndWhiteImage)
outBW[mask] = blackAndWhiteImage[mask]

outG = np.zeros_like(gray)
outG[mask] = gray[mask]

#cv2.polylines(img, [np.int32(poly)], True, (255, 0, 0), 3)

cv2.polylines(img, [np.int32(features.CheeksBeard)], True, (255, 0, 0), 3)

for i in range(68):
    cv2.circle(img, (landmarks.part(i).x, landmarks.part(i).y), 3, (0, 255, 0), -1)
    
cv2.imshow("Mask1", outG)
cv2.imshow("Mask2", outBW)  
cv2.imshow("Face", img)
