import cv2
import numpy as np
import dlib

from isolated_features import IsolatedFeatures

r = 11
path = 'age_25_29'

url = 'cropped\\128\\male\\'+path+'\\pic_'+ str(r)..zfill(4) +'.png'

img = cv2.imread(url)
img = cv2.resize(img, (320, 400),  interpolation = cv2.INTER_AREA)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector()
predictor =  dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

face = detector(gray)[0]

landmarks = predictor(gray, face)
features = IsolatedFeatures(landmarks)

#Average Face Brightness
fullMask = np.zeros((img.shape[0], img.shape[1]))
cv2.fillConvexPoly(fullMask, features.FullFace, 1)
fullMask = fullMask.astype(np.bool)
AVG_pix_gray = np.sum(gray[fullMask]) / np.sum(fullMask == 1)
(thresh, blackAndWhiteImage) = cv2.threshold(gray, AVG_pix_gray, 255, cv2.THRESH_BINARY)

LeftFeature = features.LeftCheek
LeftMask = np.zeros((img.shape[0], img.shape[1]))
cv2.fillConvexPoly(LeftMask, LeftFeature, 1)
LeftMask = LeftMask.astype(np.bool)
LeftBlackPix = int((np.sum(blackAndWhiteImage[LeftMask] == 0) / np.sum(LeftMask == 1)) * 100)

RightFeature = features.RightCheek
RightMask = np.zeros((img.shape[0], img.shape[1]))
cv2.fillConvexPoly(RightMask, RightFeature, 1)
RightMask = RightMask.astype(np.bool)
RightBlackPix = int((np.sum(blackAndWhiteImage[RightMask] == 0) / np.sum(RightMask == 1)) * 100)

if LeftBlackPix < RightBlackPix:
    mask = LeftMask
    poly = LeftFeature
else:
    mask = RightMask
    poly = RightFeature

var = int(np.var(gray[mask]) / 1000)
if var <= 1:
    var = 2
if var % 2 != 1:
    var = var + 1

blackAndWhiteImage = cv2.adaptiveThreshold(gray, 255,
                                           cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                           21, 6)

black_pix = int((np.sum(blackAndWhiteImage[mask] == 0) / np.sum(mask == 1)) * 100)
print('Y', str(r),'black:', black_pix, '%')

outBW = np.zeros_like(blackAndWhiteImage)
outBW[mask] = blackAndWhiteImage[mask]

outG = np.zeros_like(gray)
outG[mask] = gray[mask]

cv2.polylines(img, [np.int32(poly)], True, (255, 0, 0), 2)
cv2.polylines(outBW, [np.int32(poly)], True, (255, 0, 0), 1)
cv2.polylines(outG, [np.int32(poly)], True, (255, 0, 0), 1)

##for i in range(68):
##    cv2.circle(img, (landmarks.part(i).x, landmarks.part(i).y), 3, (0, 255, 0), -1)

##cv2.imshow("Mask1", outG)
##cv2.imshow("Mask2", outBW)  
##cv2.imshow("Face", img)
