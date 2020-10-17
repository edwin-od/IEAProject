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
            tempTarget = TargetFeature(this.target.name, this.target.options, [])
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


class TreeNode:
    def __init__(this, name, parent, parentLink, children):
        this.name = name
        this.parent = parent
        this.parentLink = parentLink
        this.children = children

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

        #Create new table (and find root + child tables, etc.) for each option of root
        for option in maxGainFeature.options:
            count = 0
            index = 0
            tempTarget = TargetFeature(this.target.name, this.target.options, [])
            tempFeatures = []
            for feature in this.features:
                if feature.name == maxGainFeature.name:
                    continue
                tempFeatures.append(Feature(feature.name, feature.options, [], tempTarget))
            for value in maxGainFeature.values:
                if value == option:
                    count = count + 1
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

Accept = TargetFeature("Acceptable", ["Yes", "No"], ["Yes", "No", "Yes", "No", "Yes"])

furn = Feature("Furniture", ["Yes", "No"], ["No", "Yes", "No", "No", "Yes"], Accept)
rooms = Feature("Nr Rooms", [3, 4], [3, 3, 4, 3, 4], Accept)
newKit = Feature("New Kitchen", ["Yes", "No"], ["Yes", "No", "No", "No", "No"], Accept)

table = DecisionTable([furn, rooms, newKit], Accept)

tree = table.generateDecisionTree()
tree.printTree()
