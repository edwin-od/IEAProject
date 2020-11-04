import cv2
import numpy as np
import dlib
import random

from IsolatedFeatures import IsolatedFeatures
from KNN import Iteration
from KNN import KNNTable

detector = dlib.get_frontal_face_detector()
predictor =  dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

urlYoung = 'cropped\\128\\male\\age_20_24\\pic_'
urlOld = 'cropped\\128\\male\\age_55_59\\pic_'
scans = 100
YoungOffset = 50
OldOffset = 30

def calculateFacePercentage(url, feature, AdaptiveBlockSize, AdaptiveC):
    img = cv2.imread(url)
    img = cv2.resize(img, (320, 400),  interpolation = cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    if len(faces) == 0 or len(faces) > 1:
        return -1

    face = faces[0]
    
    landmarks = predictor(gray, face)
    features = IsolatedFeatures(landmarks)

    FullPoly = features.FullFace

    LeftPoly = None
    RightPoly = None

    if feature == 0:
        LeftPoly = features.LeftCheek
        RightPoly = features.RightCheek
    elif feature == 1:
        LeftPoly = features.LeftUnderEye
        RightPoly = features.RightUnderEye
    else:
        FullPoly = features.FullCheeks
        LeftPoly = features.LeftLips
        RightPoly = features.RightLips

    #Average Face Brightness
    fullMask = np.zeros((img.shape[0], img.shape[1]))
    cv2.fillConvexPoly(fullMask, FullPoly, 1)
    fullMask = fullMask.astype(np.bool)
    AVG_pix_gray = np.sum(gray[fullMask]) / np.sum(fullMask == 1)
    (thresh, blackAndWhiteImage) = cv2.threshold(gray, AVG_pix_gray, 255, cv2.THRESH_BINARY)

    LeftMask = np.zeros((img.shape[0], img.shape[1]))
    cv2.fillConvexPoly(LeftMask, LeftPoly, 1)
    LeftMask = LeftMask.astype(np.bool)
    LeftBlackPix = int((np.sum(blackAndWhiteImage[LeftMask] == 0) / np.sum(LeftMask == 1)) * 100)

    RightMask = np.zeros((img.shape[0], img.shape[1]))
    cv2.fillConvexPoly(RightMask, RightPoly, 1)
    RightMask = RightMask.astype(np.bool)
    RightBlackPix = int((np.sum(blackAndWhiteImage[RightMask] == 0) / np.sum(RightMask == 1)) * 100)

    if LeftBlackPix < RightBlackPix:
        mask = LeftMask
    else:
        mask = RightMask

    blackAndWhiteImage = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                               cv2.THRESH_BINARY, AdaptiveBlockSize, AdaptiveC)
    return int((np.sum(blackAndWhiteImage[mask] == 0) / np.sum(mask == 1)) * 100)


youngCheeksY = 0
oldCheeksY = 0
youngCheeksO = 0
oldCheeksO = 0
sumCheeks = 0

youngUnderEyeY = 0
oldUnderEyeY = 0
youngUnderEyeO = 0
oldUnderEyeO = 0
sumUnderEye = 0

youngLipsY = 0
oldLipsY = 0
youngLipsO = 0
oldLipsO = 0
sumLips = 0

arrYoungCheeks = []
arrOldCheeks = []

arrYoungUnderEye = []
arrOldUnderEye = []

arrYoungLips = []
arrOldLips = []

YoungScans = int(str(int(scans / 2)))

OldScans = int(str(int(scans / 2)))

randomset=[]
for i in range(1, scans+1, 1):
    randomset.append(i)
random.shuffle(randomset)  

CheeksBlockSize = 11
CheeksC = 4
UnderEyeBlockSize = 9 #11|7 or 9|7/6/5 -> 9|5 best
UnderEyeC = 5
LipsBlockSize = 7 #9|5/6/7 or 7|6 -> 9|6 or 7|5/6/7 -> 7|6 best small set - and 7|7 best big set
LipsC = 7
  
halfset=int(scans/2)

for r in range(0, YoungScans):
    #Training Young
    url = urlYoung + str( randomset[r] + YoungOffset).zfill(4) + '.png'
    cheek = calculateFacePercentage(url, 0, CheeksBlockSize, CheeksC)
    underEye = calculateFacePercentage(url, 1, UnderEyeBlockSize, UnderEyeC)
    lips = calculateFacePercentage(url, 2, LipsBlockSize, LipsC)
    if min(cheek, underEye, lips) == -1:
        YoungScans -= 1
        print('ERROR: Young Scans decreased from', YoungScans + 1, 'to', YoungScans)
        continue
    pr = 'Y'+str(r + YoungOffset)+' -> '+'cheeks: '+str(cheek)+'% | under eye: '+str(underEye)+'% | lips: '+str(lips)+'%'
    print(pr)
    sumCheeks += cheek
    sumUnderEye += underEye
    sumLips += lips
    arrYoungCheeks.append(cheek)
    arrYoungUnderEye.append(underEye)
    arrYoungLips.append(lips)

for r in range(0, OldScans):
    #Training Old
    url = urlOld + str(randomset[r] + OldOffset).zfill(4) + '.png'
    cheek = calculateFacePercentage(url, 0, CheeksBlockSize, CheeksC)
    underEye = calculateFacePercentage(url, 1, UnderEyeBlockSize, UnderEyeC)
    lips = calculateFacePercentage(url, 2, LipsBlockSize, LipsC)
    if min(cheek, underEye, lips) == -1:
        OldScans -= 1
        print('ERROR: Old Scans decreased from', OldScans + 1, 'to', OldScans)
        continue
    pr = 'O'+str(r + OldOffset)+' -> '+'cheeks: '+str(cheek)+'% | under eye: '+str(underEye)+'% | lips: '+str(lips)+'%'
    print(pr)
    sumCheeks += cheek
    sumUnderEye += underEye
    sumLips += lips
    arrOldCheeks.append(cheek)
    arrOldUnderEye.append(underEye)
    arrOldLips.append(lips)

#Calculating Efficiencies

AverageCheeks = sumCheeks / (YoungScans + OldScans)
AverageUnderEyes = sumUnderEye / (YoungScans + OldScans)
AverageLips = sumLips / (YoungScans + OldScans)

print(YoungScans)
print(halfset)


for i in range(0, YoungScans):
    if arrYoungCheeks[i] <= AverageCheeks:
        youngCheeksY += 1
    else:
        oldCheeksY += 1
    if arrYoungUnderEye[i] <= AverageUnderEyes:
        youngUnderEyeY += 1
    else:
        oldUnderEyeY += 1
    if arrYoungLips[i] <= AverageLips:
        youngLipsY += 1
    else:
        oldLipsY += 1
        
for i in range(0, OldScans):
    if arrOldCheeks[i] <= AverageCheeks:
        youngCheeksO += 1
    else:
        oldCheeksO += 1
    if arrOldUnderEye[i] <= AverageUnderEyes:
        youngUnderEyeO += 1
    else:
        oldUnderEyeO += 1
    if arrOldLips[i] <= AverageLips:
        youngLipsO += 1
    else:
        oldLipsO += 1

CheekYoungEfficiency = ((YoungScans-oldCheeksY)/YoungScans)
CheekOldEfficiency = ((OldScans-youngCheeksO)/OldScans)

UnderEyeYoungEfficiency = ((YoungScans-oldUnderEyeY)/YoungScans)
UnderEyeOldEfficiency = ((OldScans -youngUnderEyeO)/OldScans)

LipsYoungEfficiency = ((YoungScans-oldLipsY)/YoungScans)
LipsOldEfficiency = ((OldScans-youngLipsO)/OldScans)

print('Cheeks Efficiency -> Young',int(CheekYoungEfficiency*100),'% Old', int(CheekOldEfficiency*100),'%')

print('Under Eye Efficiency -> Young',int(UnderEyeYoungEfficiency*100),'% Old', int(UnderEyeOldEfficiency*100),'%')

print('Lips Efficiency -> Young',int(LipsYoungEfficiency*100),'% Old', int(LipsOldEfficiency*100),'%')

#KNN

iterations = []

for i in range(0, YoungScans):
    iterations.append(Iteration([arrYoungCheeks[i],
                                arrYoungUnderEye[i],
                                arrYoungLips[i]],
                                'young'))

for i in range(0, OldScans):    
    iterations.append(Iteration([arrOldCheeks[i],
                                arrOldUnderEye[i],
                                arrOldLips[i]],
                                'old'))

YoungScans = int(str(int(scans / 2)))

OldScans = int(str(int(scans / 2)))

table = KNNTable(iterations, ['young', 'old'])

##testing young images
successyoung=0
failold=0

for r in range(halfset, halfset + YoungScans):
    url = urlYoung + str(randomset[r] + YoungOffset).zfill(4) + '.png'
    img1 = cv2.imread(url)
    img1 = cv2.resize(img1, (320, 400),  interpolation = cv2.INTER_AREA)
    cheek = calculateFacePercentage(url, 0, CheeksBlockSize, CheeksC)
    underEye = calculateFacePercentage(url, 1, UnderEyeBlockSize, UnderEyeC)
    lips = calculateFacePercentage(url, 2, LipsBlockSize, LipsC)
    if min(cheek, underEye, lips) != -1:
        pr = 'I'+str(randomset[r]+YoungOffset)+' -> '+'cheeks: '+str(cheek)+'% | under eye: '+str(underEye)+'% | lips: '+str(lips)+'%'
        print(pr)
    
        new = Iteration([
            cheek,
            underEye,
            lips],
                None)
    
        k = 8
    
        neighbors = table.findNeighbors(new, k)
        for n in neighbors:
            s = ''
            for f in n.featureValues:
                s += ' ' + str(f)
            s += ' ' + str(n.target)
            print(s)
        print('_________________________')
    
        print(table.process(new, k))
        if table.process(new, k)=='old':
            failold+=1
        else:
            successyoung+=1
    else:
        YoungScans -= 1
        print('ERROR: Young Scans decreased from', YoungScans + 1, 'to', YoungScans)
        
##testing old images
successold=0
failyoung=0
for r in range(halfset, halfset + OldScans):
    url = urlOld + str(randomset[r] + OldOffset).zfill(4) + '.png'
    img1 = cv2.imread(url)
    img1 = cv2.resize(img1, (320, 400),  interpolation = cv2.INTER_AREA)
    cheek = calculateFacePercentage(url, 0, CheeksBlockSize, CheeksC)
    underEye = calculateFacePercentage(url, 1, UnderEyeBlockSize, UnderEyeC)
    lips = calculateFacePercentage(url, 2, LipsBlockSize, LipsC)
    if min(cheek, underEye, lips) != -1:
        pr = 'I'+str(randomset[r]+OldOffset)+'-> '+'cheeks: '+str(cheek)+'% | under eye: '+str(underEye)+'% | lips: '+str(lips)+'%'
        print(pr)
    
        new = Iteration([
            cheek,
            underEye,
            lips],
                None)
    
        k = 8
    
        neighbors = table.findNeighbors(new, k)
        for n in neighbors:
            s = ''
            for f in n.featureValues:
                s += ' ' + str(f)
            s += ' ' + str(n.target)
            print(s)
        print('_________________________')
    
        print(table.process(new, k))
        if table.process(new, k)=='old':
            successold+=1
        else:
            failyoung+=1
    else:
        OldScans -= 1
        print('ERROR: Old Scans decreased from', OldScans + 1, 'to', OldScans)


print('##################### VALIDATION STATISTICS #####################')      
print('YOUNG:')
TPy = successyoung
FPy = failyoung
TNy = successold
FNy = failold

PRy = TPy / (TPy + FPy)
Ry = TPy / (TPy + FNy)
Fy = (2 * PRy * Ry) / (PRy + Ry)
AccY = (TPy + TNy) / (TPy + FPy + TNy + FNy)

print('PR', PRy)
print('R', Ry)
print('F', Fy)
print('Accuracy', int(AccY * 100))

print('OLD:')
TPo = successold
FPo = failold
TNo = successyoung
FNo = failyoung

PRo = TPo / (TPo + FPo)
Ro = TPo / (TPo + FNo)
Fo = (2 * PRo * Ro) / (PRo + Ro)
AccO = (TPo + TNo) / (TPo + FPo + TNo + FNo)

print('PR', PRo)
print('R', Ro)
print('F', Fo)
print('Accuracy', int(AccO * 100))
