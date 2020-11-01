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
        this.LeftUnderEye = np.array([
            [landmarks.part(36).x, landmarks.part(36).y],
            [landmarks.part(41).x, landmarks.part(41).y],
            [landmarks.part(40).x, landmarks.part(40).y],
            [landmarks.part(39).x, landmarks.part(39).y],
            [landmarks.part(39).x, landmarks.part(29).y],
            [landmarks.part(36).x, landmarks.part(29).y]
        ])
        this.RightUnderEye = np.array([
            [landmarks.part(42).x, landmarks.part(42).y],
            [landmarks.part(47).x, landmarks.part(47).y],
            [landmarks.part(46).x, landmarks.part(46).y],
            [landmarks.part(45).x, landmarks.part(45).y],
            [landmarks.part(45).x, landmarks.part(29).y],
            [landmarks.part(42).x, landmarks.part(29).y]
        ])
