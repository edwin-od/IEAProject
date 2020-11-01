FaceMask = np.zeros((img.shape[0], img.shape[1]))
cv2.fillConvexPoly(FaceMask, features.FullFace, 1)
FaceMask = FaceMask.astype(np.bool)

avgFace = (np.sum(gray[FaceMask]) / np.sum(FaceMask == 1))
avgPoly = (np.sum(gray[mask]) / np.sum(mask == 1))
ratio = avgPoly / avgFace
minRatio = 1.1
maxRatio = 1.15
ratio = minRatio if ratio < minRatio else maxRatio if ratio > maxRatio else ratio
threshold = (avgFace * 1.12 + avgPoly) / 2


this.FullFace = np.array([
            [landmarks.part(0).x, landmarks.part(0).y],
            [landmarks.part(1).x, landmarks.part(1).y],
            [landmarks.part(2).x, landmarks.part(2).y],
            [landmarks.part(3).x, landmarks.part(3).y],
            [landmarks.part(4).x, landmarks.part(4).y],
            [landmarks.part(5).x, landmarks.part(5).y],
            [landmarks.part(6).x, landmarks.part(6).y],
            [landmarks.part(7).x, landmarks.part(7).y],
            [landmarks.part(8).x, landmarks.part(8).y],
            [landmarks.part(9).x, landmarks.part(9).y],
            [landmarks.part(10).x, landmarks.part(10).y],
            [landmarks.part(11).x, landmarks.part(11).y],
            [landmarks.part(12).x, landmarks.part(12).y],
            [landmarks.part(13).x, landmarks.part(13).y],
            [landmarks.part(14).x, landmarks.part(14).y],
            [landmarks.part(15).x, landmarks.part(15).y],
            [landmarks.part(16).x, landmarks.part(16).y],
            [landmarks.part(26).x, landmarks.part(26).y],
            [landmarks.part(25).x, landmarks.part(25).y],
            [landmarks.part(24).x, landmarks.part(24).y],
            [landmarks.part(23).x, landmarks.part(23).y],
            [landmarks.part(22).x, landmarks.part(22).y],
            [landmarks.part(21).x, landmarks.part(21).y],
            [landmarks.part(20).x, landmarks.part(20).y],
            [landmarks.part(19).x, landmarks.part(19).y],
            [landmarks.part(18).x, landmarks.part(18).y],
            [landmarks.part(17).x, landmarks.part(17).y]
        ])
