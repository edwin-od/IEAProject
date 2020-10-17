import math

class TargetFeature:
    def __init__(this, name, options, values):
        this.name = name
        this.options = options
        this.values = values

    #Calculating the target feature's entropy
    def H(this):
        h = 0
        for option in this.options:
            count = 0
            for value in this.values:
                if value == option:
                    count = count + 1
            if count == 0:
                continue
            f = count / len(this.values)
            h -= f * math.log2(f)
        return h

        
class Feature:
    def __init__(this, name, options, values, target):
        this.name = name
        this.options = options
        this.values = values
        this.target = target

    #Calculating the feature's gain according to the specified target feature
    def G(this):
        g = this.target.H()
        for option in this.options:
            count = 0
            index = 0
            tempTarget = TargetFeature(this.target.name + "-" + str(option), this.target.options, [])
            for value in this.values:
                if value == option:
                    count = count + 1
                    tempTarget.values.append(this.target.values[index])
                index = index + 1
            if count == 0:
                continue
            f = count / len(this.values)
            g -= f * tempTarget.H()
        return g


Accept = TargetFeature("Acceptable", ["Yes", "No"], ["Yes", "No", "Yes", "No", "Yes"])

furn = Feature("Furniture", ["Yes", "No"], ["No", "Yes", "No", "No", "Yes"], Accept)
rooms = Feature("Nr Rooms", [3, 4], [3, 3, 4, 3, 4], Accept)
newKit = Feature("New Kitchen", ["Yes", "No"], ["Yes", "No", "No", "No", "No"], Accept)

print(furn.G())
