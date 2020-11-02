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
sumYoungCheeks = 0
sumOldCheeks = 0

youngUnderEyeY = 0
oldUnderEyeY = 0
youngUnderEyeO = 0
oldUnderEyeO = 0
sumYoungUnderEye = 0
sumOldUnderEye = 0

youngLipsY = 0
oldLipsY = 0
youngLipsO = 0
oldLipsO = 0
sumYoungLips = 0
sumOldLips = 0

arrCheeks = []
arrUnderEye = []
arrLips = []

YoungScans = 30
OldScans = 30

for r in range(1, YoungScans + OldScans + 1):
    zeros = ''
    cheek = -1
    underEye = -1
    lips = -1
    if r <= YoungScans:
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
            print('Young Scans', YoungScans)
            continue
        print('Y', r)
        sumYoungCheeks += cheek
        sumYoungUnderEye += underEye
        sumYoungLips += lips
    elif r  <= YoungScans + OldScans:
        if len(str(r - YoungScans)) == 1:
            zeros = '000'
        elif len(str(r - YoungScans)) == 2:
            zeros = '00'
        elif len(str(r - YoungScans)) == 3:
            zeros = '0'
        else:
            zeros = ''
        url = 'cropped\\128\\male\\age_55_59\\pic_' + zeros + str(r - YoungScans) + '.png'
        cheek = calculateFacePercentage(url, 0, 11, 4)
        underEye = calculateFacePercentage(url, 1, 11, 4)
        lips = calculateFacePercentage(url, 2, 11, 4)
        if min(cheek, underEye, lips) == -1:
            OldScans -= 1
            print('Old Scans', OldScans)
            continue
        print('O', r - YoungScans)
        sumOldCheeks += cheek
        sumOldUnderEye += underEye
        sumOldLips += lips
    else:
        break
    arrCheeks.append(cheek)
    arrUnderEye.append(underEye)
    arrLips.append(lips)

print('cheeks array', len(arrCheeks))
print('under eye array', len(arrUnderEye))
print('lips array', len(arrLips))

AverageYoungCheeks = sumYoungCheeks / YoungScans
AverageOldCheeks = sumOldCheeks / OldScans

AverageYoungUnderEyes = sumYoungUnderEye / YoungScans
AverageOldUnderEyes = sumOldUnderEye / OldScans

AverageYoungLips = sumYoungLips / YoungScans
AverageOldLips = sumOldLips / OldScans

for i in range(0, YoungScans + OldScans):

    if i <= YoungScans:
        if arrCheeks[i] <= AverageYoungCheeks:
            youngCheeksY += 1
        else:
            oldCheeksY += 1
        if arrUnderEye[i] <= AverageYoungUnderEyes:
            youngUnderEyeY += 1
        else:
            oldUnderEyeY += 1
        if arrLips[i] <= AverageYoungLips:
            youngLipsY += 1
        else:
            oldLipsY += 1
    elif i  <= YoungScans + OldScans:
        if arrCheeks[i] <= AverageOldCheeks:
            youngCheeksO += 1
        else:
            oldCheeksO += 1
        if arrUnderEye[i] <= AverageOldUnderEyes:
            youngUnderEyeO += 1
        else:
            oldUnderEyeO += 1
        if arrLips[i] <= AverageOldLips:
            youngLipsO += 1
        else:
            oldLipsO += 1
    else:
        break
        
print( 'Cheeks Efficiency -> Young',int(((youngCheeksY-oldCheeksY)/youngCheeksY)*100),'% Old',
       int(((youngCheeksO-oldCheeksO)/youngCheeksO)*100))

print( 'Under Eye Efficiency -> Young',int(((youngUnderEyeY-oldUnderEyeY)/youngUnderEyeY)*100),'% Old',
       int(((youngUnderEyeO-oldUnderEyeO)/youngUnderEyeO)*100))

print( 'Cheeks Efficiency -> Young',int(((youngLipsY-oldLipsY)/youngLipsY)*100),'% Old',
       int(((youngLipsO-oldLipsO)/youngLipsO)*100))
