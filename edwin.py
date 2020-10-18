import math

class Feature:
    def __init__(this, name, options, values, target):
        this.name = name
        this.options = options
        this.values = values
        this.target = target

    #Calculating the target feature's entropy
    def H(this):
        if this.target != None:
            return 1
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
    
    #Calculating the feature's gain according to the specified target feature
    def G(this):
        if this.target == None:
            return 0
        g = this.target.H()
        for option in this.options:
            count = 0
            index = 0
            tempTarget = Feature(this.target.name, this.target.options, [], None)
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

class Input:
    def __init__(this, features):
        this.features = features

class TreeNode:
    def __init__(this, name, parent, parentLink, children):
        this.name = name
        this.parent = parent
        this.parentLink = parentLink
        this.children = children

    def traverse(this, In):
        if this.children == None:
            return this.name
        
        #Find the feature
        f = None
        for feature in In.features:
            if feature == this.name:
                f = feature
        if f == None:
            raise Exception("Error: The feature '" + this.name + "' is missing")
        for child in this.children:
            if child.parentLink == In.features[f]:
                return child.traverse(In)
        return "Unidentified"

    def printTree(this):
        print(this.name)
        p = this.parent
        s = "  ";
        while p != None:
            s += "   "
            p = p.parent
        if(this.children == None):
            return
        for child in this.children:
            print(s + "|__(" + str(child.parentLink) + ")__", end = "")
            child.printTree()


class DecisionTable:
    def __init__(this, features, target):
        this.features = features
        this.target = target

    def generateDecisionTree(this):
        #Find the root
        maxGainFeature = None
        for feature in this.features:
            if maxGainFeature == None or feature.G() >= maxGainFeature.G():
                maxGainFeature = feature
        tree = TreeNode(maxGainFeature.name, None, None, [])

        #Create new table (and find new root + child tables, etc. - repeat until pure sets achieved) for each option of root
        for option in maxGainFeature.options:
            tempTarget = Feature(this.target.name, this.target.options, [], None)
            tempFeatures = []
            for feature in this.features:
                if feature.name == maxGainFeature.name:
                    continue
                tempFeatures.append(Feature(feature.name, feature.options, [], tempTarget))
            index = 0
            for value in maxGainFeature.values:
                if value == option:
                    tempTarget.values.append(this.target.values[index])
                    for tempFeature in tempFeatures:
                        for feature in this.features:
                            if feature.name == tempFeature.name:
                                tempFeature.values.append(feature.values[index])
                                break
                index = index + 1
            if tempTarget.H() == 0:
                tree.children.append(TreeNode(tempTarget.values[0], tree, option, None))
                continue
            tempTable = DecisionTable(tempFeatures, tempTarget)
            tree.children.append(tempTable.generateDecisionTree())
            tree.children[len(tree.children) - 1].parent = tree
            tree.children[len(tree.children) - 1].parentLink = option
        return tree


#Test Example Chapter 7 Exercise 7.3

##Accept = Feature("Acceptable", ["Yes", "No"], ["Yes", "No", "Yes", "No", "Yes"], None)
##furn = Feature("Furniture", ["Yes", "No"], ["No", "Yes", "No", "No", "Yes"], Accept)
##rooms = Feature("Nr Rooms", [3, 4], [3, 3, 4, 3, 4], Accept)
##newKit = Feature("New Kitchen", ["Yes", "No"], ["Yes", "No", "No", "No", "No"], Accept)
##table = DecisionTable([furn, rooms, newKit], Accept)
##tree = table.generateDecisionTree()
##tree.printTree()

#Test Example Chapter 7 Exercise 7.4

Sunburn = Feature("Sunburn", ["Yes", "No"], ["Yes", "No", "No", "Yes", "No", "No"], None)
Hair = Feature("Hair", ["Blond", "Brown"], ["Blond", "Blond", "Brown", "Blond", "Blond", "Brown"], Sunburn)
Height = Feature("Height", ["Average", "Tall", "Short"], ["Average", "Tall", "Short", "Short", "Tall", "Tall"], Sunburn)
Lotion = Feature("Lotion", ["Yes", "No"], ["No", "Yes", "Yes", "No", "No", "No"], Sunburn)
table = DecisionTable([Hair, Height, Lotion], Sunburn)
tree = table.generateDecisionTree()
tree.printTree()

newInput = Input({"Hair": "Brown", "Height": "Average", "Lotion": "No"})
print(tree.traverse(newInput))
