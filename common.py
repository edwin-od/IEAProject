import cv2
import numpy as np
import dlib

from isolated_features import IsolatedFeatures

detector = dlib.get_frontal_face_detector()
predictor =  dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def calculateFacePercentage(url, feature, AdaptiveBlockSize, AdaptiveC): #feature = 0=>Cheeks|1=>UnderEyes|2=>Lips
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
    elif feature == 2:
        LeftPoly = features.LeftLips
        RightPoly = features.RightLips
    else:
        raise Exception("Feature integers can only be {0, 1, 2}")

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

YoungScans = 31
OldScans = 31

for r in range(1, YoungScans + 1):
    zeros = ''
    cheek = -1
    underEye = -1
    lips = -1
    if len(str(r)) == 1:
        zeros = '000'
    elif len(str(r)) == 2:
        zeros = '00'
    elif len(str(r)) == 3:
        zeros = '0'
    url = 'cropped\\128\\male\\age_25_29\\pic_' + zeros + str(r) + '.png'
    cheek = calculateFacePercentage(url, 0, 11, 4)
    underEye = calculateFacePercentage(url, 1, 11, 4)
    lips = calculateFacePercentage(url, 2, 11, 4)
    if min(cheek, underEye, lips) == -1:
        YoungScans -= 1
        print('ERROR: Young Scans decreased from', YoungScans + 1, 'to', YoungScans)
        continue
    pr = 'Y'+str(r)+' -> '+'cheeks: '+str(cheek)+'% | under eye: '+str(underEye)+'% | lips: '+str(lips)
    print(pr)
    sumCheeks += cheek
    sumUnderEye += underEye
    sumLips += lips
    arrYoungCheeks.append(cheek)
    arrYoungUnderEye.append(underEye)
    arrYoungLips.append(lips)
    
for r in range(1, OldScans + 1):
    zeros = ''
    cheek = -1
    underEye = -1
    lips = -1
    if len(str(r)) == 1:
        zeros = '000'
    elif len(str(r)) == 2:
        zeros = '00'
    elif len(str(r)) == 3:
        zeros = '0'
    else:
        zeros = ''
    url = 'cropped\\128\\male\\age_55_59\\pic_' + zeros + str(r) + '.png'
    cheek = calculateFacePercentage(url, 0, 11, 4)
    underEye = calculateFacePercentage(url, 1, 11, 4)
    lips = calculateFacePercentage(url, 2, 11, 4)
    if min(cheek, underEye, lips) == -1:
        OldScans -= 1
        print('ERROR: Old Scans decreased from', OldScans + 1, 'to', OldScans)
        continue
    pr = 'O'+str(r)+' -> '+'cheeks: '+str(cheek)+'% | under eye: '+str(underEye)+'% | lips: '+str(lips)
    print(pr)
    sumCheeks += cheek
    sumUnderEye += underEye
    sumLips += lips
    arrOldCheeks.append(cheek)
    arrOldUnderEye.append(underEye)
    arrOldLips.append(lips)

AverageCheeks = sumCheeks / (YoungScans + OldScans)
AverageUnderEyes = sumUnderEye / (YoungScans + OldScans)
AverageLips = sumLips / (YoungScans + OldScans)

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
        
print('Cheeks Efficiency -> Young',int(((YoungScans-oldCheeksY)/YoungScans)*100),'% Old',
       int(((OldScans-youngCheeksO)/OldScans)*100),'%')

print('Under Eye Efficiency -> Young',int(((YoungScans-oldUnderEyeY)/YoungScans)*100),'% Old',
       int(((OldScans-youngUnderEyeO)/OldScans)*100),'%')

print('Lips Efficiency -> Young',int(((YoungScans-oldLipsY)/YoungScans)*100),'% Old',
       int(((OldScans-oldLipsO)/OldScans)*100),'%')
