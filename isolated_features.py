import numpy as np

class IsolatedFeatures:
    def __init__(this, landmarks):
        this.FullCheeks = np.array([
            [landmarks.part(28).x, landmarks.part(28).y],
            [landmarks.part(17).x, landmarks.part(28).y],
            [landmarks.part(17).x, landmarks.part(29).y],
            [landmarks.part(18).x, landmarks.part(31).y],
            [landmarks.part(40).x, landmarks.part(31).y],
            [landmarks.part(30).x, landmarks.part(30).y],
            [landmarks.part(47).x, landmarks.part(35).y],
            [landmarks.part(25).x, landmarks.part(35).y],
            [landmarks.part(26).x, landmarks.part(29).y],
            [landmarks.part(26).x, landmarks.part(28).y]
        ])
        this.LeftCheek = np.array([
            [landmarks.part(40).x, landmarks.part(31).y],
            [landmarks.part(48).x, landmarks.part(48).y],
            [landmarks.part(3).x, landmarks.part(3).y],
            [landmarks.part(1).x, landmarks.part(1).y],
            [landmarks.part(17).x, landmarks.part(28).y],
            [landmarks.part(21).x, landmarks.part(28).y],
            [landmarks.part(21).x, landmarks.part(29).y]
        ])
        this.RightCheek = np.array([
            [landmarks.part(47).x, landmarks.part(35).y],
            [landmarks.part(54).x, landmarks.part(54).y],
            [landmarks.part(13).x, landmarks.part(13).y],
            [landmarks.part(14).x, landmarks.part(14).y],
            [landmarks.part(26).x, landmarks.part(28).y],
            [landmarks.part(22).x, landmarks.part(28).y],
            [landmarks.part(22).x, landmarks.part(29).y]
        ])
        this.FullLips = np.array([
            [landmarks.part(61).x, landmarks.part(61).y],
            [landmarks.part(50).x, landmarks.part(50).y],
            [landmarks.part(49).x, landmarks.part(49).y],
            [landmarks.part(48).x, landmarks.part(48).y],
            [landmarks.part(59).x, landmarks.part(59).y],
            [landmarks.part(58).x, landmarks.part(58).y],
            [landmarks.part(57).x, landmarks.part(57).y],
            [landmarks.part(56).x, landmarks.part(56).y],
            [landmarks.part(55).x, landmarks.part(55).y],
            [landmarks.part(54).x, landmarks.part(54).y],
            [landmarks.part(53).x, landmarks.part(53).y],
            [landmarks.part(52).x, landmarks.part(52).y],
            [landmarks.part(63).x, landmarks.part(63).y],
            [landmarks.part(64).x, landmarks.part(64).y],
            [landmarks.part(65).x, landmarks.part(65).y],
            [landmarks.part(66).x, landmarks.part(66).y],
            [landmarks.part(67).x, landmarks.part(67).y],
            [landmarks.part(60).x, landmarks.part(60).y]
        ])   
        this.LeftLips = np.array([
            [landmarks.part(50).x, landmarks.part(50).y],
            [landmarks.part(51).x, landmarks.part(51).y],
            [landmarks.part(62).x, landmarks.part(62).y],
            [landmarks.part(61).x, landmarks.part(61).y],
            [landmarks.part(49).x, landmarks.part(60).y],
            [landmarks.part(67).x, landmarks.part(67).y],
            [landmarks.part(66).x, landmarks.part(66).y],
            [landmarks.part(57).x, landmarks.part(57).y],
            [landmarks.part(58).x, landmarks.part(58).y],
            [landmarks.part(59).x, landmarks.part(59).y],
            [landmarks.part(49).x, landmarks.part(49).y]
        ])
        this.RightLips = np.array([
            [landmarks.part(52).x, landmarks.part(52).y],
            [landmarks.part(51).x, landmarks.part(51).y],
            [landmarks.part(62).x, landmarks.part(62).y],
            [landmarks.part(63).x, landmarks.part(63).y],
            [landmarks.part(53).x, landmarks.part(63).y],
            [landmarks.part(65).x, landmarks.part(65).y],
            [landmarks.part(66).x, landmarks.part(66).y],
            [landmarks.part(57).x, landmarks.part(57).y],
            [landmarks.part(56).x, landmarks.part(56).y],
            [landmarks.part(55).x, landmarks.part(55).y],
            [landmarks.part(53).x, landmarks.part(53).y]
        ])
