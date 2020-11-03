import cv2
import numpy as np
import dlib
import random

from IsolatedFeatures import IsolatedFeatures
#from DecisionTree import Feature as DTFeature
#from DecisionTree import DecisionTable
from KNN import Iteration
from KNN import KNNTable

detector = dlib.get_frontal_face_detector()
predictor =  dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

urlYoung = 'cropped\\128\\male\\age_20_24\\pic_'
urlOld = 'cropped\\128\\male\\age_55_59\\pic_'
scans = 100

urlNew = 'cropped\\128\\male\\age_60_94\\pic_'
newIndex = 90

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
YoungOffset = 50

OldScans = int(str(int(scans / 2)))
OldOffset = 30

setsize=int(scans/2)
randomset= []
for i in range(1, setsize+1, 1):
    randomset.append(i)
random.shuffle(randomset)  
for r in range(0,len(randomset)):
    url = urlYoung + str( randomset[r]).zfill(4) + '.png'
CheeksBlockSize = 11
CheeksC = 4
UnderEyeBlockSize = 9 #11|7 or 9|7/6/5 -> 9|5 best
UnderEyeC = 5
LipsBlockSize = 7 #9|5/6/7 or 7|6 -> 9|6 or 7|5/6/7 -> 7|6 best small set - and 7|7 best big set
LipsC = 7
setsize=int(scans/2)
randomset= []
for i in range(1, setsize+1, 1):
    randomset.append(i)
random.shuffle(randomset)  
halfset=int(len(randomset)/2)

for r in range(0,halfset):
    print('picture number',randomset[r])
    url = urlYoung + str( randomset[r]).zfill(4) + '.png'
    cheek = calculateFacePercentage(url, 0, CheeksBlockSize, CheeksC)
    underEye = calculateFacePercentage(url, 1, UnderEyeBlockSize, UnderEyeC)
    lips = calculateFacePercentage(url, 2, LipsBlockSize, LipsC)
    if min(cheek, underEye, lips) == -1:
        YoungScans -= 1
        print('ERROR: Young Scans decreased from', YoungScans + 1, 'to', YoungScans)
        continue
    pr = 'Y'+str(r)+' -> '+'cheeks: '+str(cheek)+'% | under eye: '+str(underEye)+'% | lips: '+str(lips)+'%'
    print(pr)
    sumCheeks += cheek
    sumUnderEye += underEye
    sumLips += lips
    arrYoungCheeks.append(cheek)
    arrYoungUnderEye.append(underEye)
    arrYoungLips.append(lips)
    
for r in range(0, halfset):
    print('picture number',randomset[r])
    url = urlOld + str(randomset[r]).zfill(4) + '.png'
    cheek = calculateFacePercentage(url, 0, CheeksBlockSize, CheeksC)
    underEye = calculateFacePercentage(url, 1, UnderEyeBlockSize, UnderEyeC)
    lips = calculateFacePercentage(url, 2, LipsBlockSize, LipsC)
    if min(cheek, underEye, lips) == -1:
        OldScans -= 1
        print('ERROR: Old Scans decreased from', OldScans + 1, 'to', OldScans)
        continue
    pr = 'O'+str(r)+' -> '+'cheeks: '+str(cheek)+'% | under eye: '+str(underEye)+'% | lips: '+str(lips)+'%'
    print(pr)
    sumCheeks += cheek
    sumUnderEye += underEye
    sumLips += lips
    arrOldCheeks.append(cheek)
    arrOldUnderEye.append(underEye)
    arrOldLips.append(lips)


#Calculating Efficiencies
youngscansupdated=int(YoungScans/2)
oldscansupdated=int(OldScans/2)

AverageCheeks = sumCheeks / (YoungScans + OldScans)
AverageUnderEyes = sumUnderEye / (YoungScans + OldScans)
AverageLips = sumLips / (YoungScans + OldScans)


for i in range(0, youngscansupdated-1):
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
        
for i in range(0, oldscansupdated-1):
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

CheekYoungEfficiency = ((youngscansupdated-oldCheeksY)/youngscansupdated)
CheekOldEfficiency = ((oldscansupdated-youngCheeksO)/oldscansupdated)

UnderEyeYoungEfficiency = ((youngscansupdated-oldUnderEyeY)/youngscansupdated)
UnderEyeOldEfficiency = ((oldscansupdated -youngUnderEyeO)/oldscansupdated)

LipsYoungEfficiency = ((youngscansupdated-oldLipsY)/youngscansupdated)
LipsOldEfficiency = ((oldscansupdated-youngLipsO)/oldscansupdated)

print('Cheeks Efficiency -> Young',int(CheekYoungEfficiency*100),'% Old', int(CheekOldEfficiency*100),'%')

print('Under Eye Efficiency -> Young',int(UnderEyeYoungEfficiency*100),'% Old', int(UnderEyeOldEfficiency*100),'%')

print('Lips Efficiency -> Young',int(LipsYoungEfficiency*100),'% Old', int(LipsOldEfficiency*100),'%')

#KNN

iterations = []

for i in range(0, youngscansupdated-1):
    iterations.append(Iteration([arrYoungCheeks[i],
                                arrYoungUnderEye[i],
                                arrYoungLips[i]],
                                'young'))

for i in range(0, oldscansupdated-1):    
    iterations.append(Iteration([arrOldCheeks[i],
                                arrOldUnderEye[i],
                                arrOldLips[i]],
                                'old'))

table = KNNTable(iterations, ['young', 'old'])
##testing young images
successyoung=0
failold=0

for r in range(halfset,len(randomset)-(int(YoungScans/2)-youngscansupdated)):
    print('picture number',randomset[r])
    url = urlYoung + str( randomset[r]).zfill(4) + '.png'
    img1 = cv2.imread(url)
    img1 = cv2.resize(img1, (320, 400),  interpolation = cv2.INTER_AREA)
    cv2.imshow("Face", img1)
    cheek = calculateFacePercentage(url, 0, CheeksBlockSize, CheeksC)
    underEye = calculateFacePercentage(url, 1, UnderEyeBlockSize, UnderEyeC)
    lips = calculateFacePercentage(url, 2, LipsBlockSize, LipsC)
    if min(cheek, underEye, lips) != -1:
        pr = 'I'+str(newIndex)+' -> '+'cheeks: '+str(cheek)+'% | under eye: '+str(underEye)+'% | lips: '+str(lips)+'%'
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
        print('ERROR: Image could not be processed')
##testing old images
successold=0
failyoung=0
for r in range(halfset,len(randomset)-(int(YoungScans/2)-youngscansupdated)):
    print('picture number',randomset[r])
    url = urlOld + str(randomset[r]).zfill(4) + '.png'
    img1 = cv2.imread(url)
    img1 = cv2.resize(img1, (320, 400),  interpolation = cv2.INTER_AREA)
    cv2.imshow("Face", img1)
    cheek = calculateFacePercentage(url, 0, CheeksBlockSize, CheeksC)
    underEye = calculateFacePercentage(url, 1, UnderEyeBlockSize, UnderEyeC)
    lips = calculateFacePercentage(url, 2, LipsBlockSize, LipsC)
    if min(cheek, underEye, lips) != -1:
        pr = 'I'+str(newIndex)+' -> '+'cheeks: '+str(cheek)+'% | under eye: '+str(underEye)+'% | lips: '+str(lips)+'%'
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
        print('ERROR: Image could not be processed')


print('############################################\n''############################################\n')
print('############################################\n''############################################\n')
print('####################SWAPING THE DATASETS##################\n')
print('############################################\n''############################################\n')
print('############################################\n''############################################\n')

for r in range(halfset,len(randomset)-(int(YoungScans/2)-youngscansupdated)):
    print('picture number',randomset[r])
    url = urlYoung + str( randomset[r]).zfill(4) + '.png'
    cheek = calculateFacePercentage(url, 0, CheeksBlockSize, CheeksC)
    underEye = calculateFacePercentage(url, 1, UnderEyeBlockSize, UnderEyeC)
    lips = calculateFacePercentage(url, 2, LipsBlockSize, LipsC)
    if min(cheek, underEye, lips) == -1:
        YoungScans -= 1
        print('ERROR: Young Scans decreased from', YoungScans + 1, 'to', YoungScans)
        continue
    pr = 'Y'+str(r)+' -> '+'cheeks: '+str(cheek)+'% | under eye: '+str(underEye)+'% | lips: '+str(lips)+'%'
    print(pr)
    sumCheeks += cheek
    sumUnderEye += underEye
    sumLips += lips
    arrYoungCheeks.append(cheek)
    arrYoungUnderEye.append(underEye)
    arrYoungLips.append(lips)
    
for r in range(halfset,len(randomset)-(int(YoungScans/2)-youngscansupdated)):
    print('picture number',randomset[r])
    url = urlOld + str(randomset[r]).zfill(4) + '.png'
    cheek = calculateFacePercentage(url, 0, CheeksBlockSize, CheeksC)
    underEye = calculateFacePercentage(url, 1, UnderEyeBlockSize, UnderEyeC)
    lips = calculateFacePercentage(url, 2, LipsBlockSize, LipsC)
    if min(cheek, underEye, lips) == -1:
        OldScans -= 1
        print('ERROR: Old Scans decreased from', OldScans + 1, 'to', OldScans)
        continue
    pr = 'O'+str(r)+' -> '+'cheeks: '+str(cheek)+'% | under eye: '+str(underEye)+'% | lips: '+str(lips)+'%'
    print(pr)
    sumCheeks += cheek
    sumUnderEye += underEye
    sumLips += lips
    arrOldCheeks.append(cheek)
    arrOldUnderEye.append(underEye)
    arrOldLips.append(lips)


#Calculating Efficiencies
youngscansupdated1=int(YoungScans/2)
oldscansupdated1=int(OldScans/2)

AverageCheeks = sumCheeks / (YoungScans + OldScans)
AverageUnderEyes = sumUnderEye / (YoungScans + OldScans)
AverageLips = sumLips / (YoungScans + OldScans)


for i in range(0, youngscansupdated1-1):
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
        
for i in range(0, oldscansupdated1-1):
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

CheekYoungEfficiency = ((youngscansupdated1-oldCheeksY)/youngscansupdated1)
CheekOldEfficiency = ((oldscansupdated1-youngCheeksO)/oldscansupdated1)

UnderEyeYoungEfficiency = ((youngscansupdated1-oldUnderEyeY)/youngscansupdated1)
UnderEyeOldEfficiency = ((oldscansupdated1 -youngUnderEyeO)/oldscansupdated1)

LipsYoungEfficiency = ((youngscansupdated1-oldLipsY)/youngscansupdated1)
LipsOldEfficiency = ((oldscansupdated1-youngLipsO)/oldscansupdated1)

print('Cheeks Efficiency -> Young',int(CheekYoungEfficiency*100),'% Old', int(CheekOldEfficiency*100),'%')

print('Under Eye Efficiency -> Young',int(UnderEyeYoungEfficiency*100),'% Old', int(UnderEyeOldEfficiency*100),'%')

print('Lips Efficiency -> Young',int(LipsYoungEfficiency*100),'% Old', int(LipsOldEfficiency*100),'%')

#KNN

iterations = []

for i in range(0, youngscansupdated1-1):
    iterations.append(Iteration([arrYoungCheeks[i],
                                arrYoungUnderEye[i],
                                arrYoungLips[i]],
                                'young'))

for i in range(0, oldscansupdated1-1):    
    iterations.append(Iteration([arrOldCheeks[i],
                                arrOldUnderEye[i],
                                arrOldLips[i]],
                                'old'))

table = KNNTable(iterations, ['young', 'old'])
##testing young images
successyoung1=0
failold1=0

for r in range(0,len(randomset)):
    print('picture number',randomset[r])
    url = urlYoung + str( randomset[r]).zfill(4) + '.png'
    img1 = cv2.imread(url)
    img1 = cv2.resize(img1, (320, 400),  interpolation = cv2.INTER_AREA)
    cv2.imshow("Face", img1)
    cheek = calculateFacePercentage(url, 0, CheeksBlockSize, CheeksC)
    underEye = calculateFacePercentage(url, 1, UnderEyeBlockSize, UnderEyeC)
    lips = calculateFacePercentage(url, 2, LipsBlockSize, LipsC)
    if min(cheek, underEye, lips) != -1:
        pr = 'I'+str(newIndex)+' -> '+'cheeks: '+str(cheek)+'% | under eye: '+str(underEye)+'% | lips: '+str(lips)+'%'
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
            failold1+=1
        else:
            successyoung1+=1
    else:
        print('ERROR: Image could not be processed')
##testing old images
successold1=0
failyoung1=0
for r in range(0,len(randomset)):
    print('picture number',randomset[r])
    url = urlOld + str(randomset[r]).zfill(4) + '.png'
    img1 = cv2.imread(url)
    img1 = cv2.resize(img1, (320, 400),  interpolation = cv2.INTER_AREA)
    cv2.imshow("Face", img1)
    cheek = calculateFacePercentage(url, 0, CheeksBlockSize, CheeksC)
    underEye = calculateFacePercentage(url, 1, UnderEyeBlockSize, UnderEyeC)
    lips = calculateFacePercentage(url, 2, LipsBlockSize, LipsC)
    if min(cheek, underEye, lips) != -1:
        pr = 'I'+str(newIndex)+' -> '+'cheeks: '+str(cheek)+'% | under eye: '+str(underEye)+'% | lips: '+str(lips)+'%'
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
            successold1+=1
        else:
            failyoung1+=1
    else:
        print('ERROR: Image could not be processed')

print('FIRST VALIDATION RESULT#####################')      
print('success young:',successyoung,'\n fail old:',failold,'\nsuccess old:',successold,'\n fail young:',failyoung)
print('SECOND VALIDATION RESULT####################')
print('success young:',successyoung1,'\n fail old:',failold1,'\nsuccess old:',successold1,'\n fail young:',failyoung1)






#Decision Tree

##AverageCheeks = sumCheeks / (YoungScans + OldScans)
##AverageUnderEyes = sumUnderEye / (YoungScans + OldScans)
##AverageLips = sumLips / (YoungScans + OldScans)
##
##arrCheeks = []
##arrUnderEye = []
##arrLips = []
##
##arr = []
##
##for i in range(0, YoungScans):
##    arr.append('young')
##    if arrYoungCheeks[i] <= AverageCheeks:
##        youngCheeksY += 1
##        arrCheeks.append('clear')
##    else:
##        oldCheeksY += 1
##        arrCheeks.append('wrinkled')
##    if arrYoungUnderEye[i] <= AverageUnderEyes:
##        youngUnderEyeY += 1
##        arrUnderEye.append('clear')
##    else:
##        oldUnderEyeY += 1
##        arrUnderEye.append('wrinkled')
##    if arrYoungLips[i] <= AverageLips:
##        youngLipsY += 1
##        arrLips.append('clear')
##    else:
##        oldLipsY += 1
##        arrLips.append('wrinkled')
##        
##for i in range(0, OldScans):
##    arr.append('old')
##    if arrOldCheeks[i] <= AverageCheeks:
##        youngCheeksO += 1
##        arrCheeks.append('clear')
##    else:
##        oldCheeksO += 1
##        arrCheeks.append('wrinkled')
##    if arrOldUnderEye[i] <= AverageUnderEyes:
##        youngUnderEyeO += 1
##        arrUnderEye.append('clear')
##    else:
##        oldUnderEyeO += 1
##        arrUnderEye.append('wrinkled')
##    if arrOldLips[i] <= AverageLips:
##        youngLipsO += 1
##        arrLips.append('clear')
##    else:
##        oldLipsO += 1
##        arrLips.append('wrinkled')
##
##print('Cheeks Efficiency -> Young',int(((YoungScans-oldCheeksY)/YoungScans)*100),'% Old',
##       int(((OldScans-youngCheeksO)/OldScans)*100),'%')
##
##print('Under Eye Efficiency -> Young',int(((YoungScans-oldUnderEyeY)/YoungScans)*100),'% Old',
##       int(((OldScans-youngUnderEyeO)/OldScans)*100),'%')
##
##print('Lips Efficiency -> Young',int(((YoungScans-oldLipsY)/YoungScans)*100),'% Old',
##       int(((OldScans-youngLipsO)/OldScans)*100),'%')
##
##Age = DTFeature('Age', ['young', 'old'], arr, None)
##Cheeks = DTFeature('Cheeks', ['clear', 'wrinkled'], arrCheeks, Age)
##UnderEye = DTFeature('Under Eye', ['clear', 'wrinkled'], arrUnderEye, Age)
##Lips = DTFeature('Lips', ['clear', 'wrinkled'], arrLips, Age)
##table = DecisionTable([Cheeks, UnderEye, Lips], Age)
##tree = table.generateDecisionTree()
##tree.printTree()
