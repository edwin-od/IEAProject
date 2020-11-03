cv2.polylines(img, [np.int32(poly)], True, (255, 0, 0), 2)

for i in range(68):
        cv2.circle(img, (landmarks.part(i).x, landmarks.part(i).y), 3, (0, 255, 0), -1)
