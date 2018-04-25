import csv
import math
import random
import matplotlib.pyplot as plt

# Class used for learning and building the Decision Tree using the given Training Set
class DecisionTree:
    tree = {}

    def learn(self, training_set, attributes, target):
        self.tree = build_tree(training_set, attributes, target)


# Class Node which will be used while classify a test-instance using the tree which was built earlier
class Node:
    value = ""
    children = []

    def __init__(self, val, dictionary):
        self.value = val
        if isinstance(dictionary, dict):
            self.children = dictionary.keys()


# Majority Function which tells which class has more entries in given data-set
def majorClass(attributes, data, target):

    freq = {}
    index = attributes.index(target)

    for tuple in data:
        if freq.has_key(tuple[index]):
            freq[tuple[index]] += 1 
        else:
            freq[tuple[index]] = 1

    max = 0
    major = ""

    for key in freq.keys():
        if freq[key] > max:
            max = freq[key]
            major = key

    return major


# Calculates the entropy of the data given the target attribute
def entropy(attributes, data, targetAttr):
    freq = {}
    entropy = 0.0

    i = 0
    for entry in attributes:
        if targetAttr == entry:
            break
        i = i + 1

    i = i - 1

    for entry in data:
        if freq.has_key(entry[i]):
            freq[entry[i]] += 1.0
        else:
            freq[entry[i]] = 1.0

    for freq in freq.values():
        entropy += (-freq / len(data)) * math.log(freq / len(data), 2)
        
    return entropy


def info_gain(attributes, data, attr, targetAttr):
    """Compute the information gain"""
    freq = {}
    data_entropy = 0.0
    i = attributes.index(attr)

    for entry in data:
        if freq.has_key(entry[i]):
            freq[entry[i]] += 1.0
        else:
            freq[entry[i]] = 1.0

    for val in freq.keys():
        valProb = freq[val] / sum(freq.values())
        dataSubset = [entry for entry in data if entry[i] == val]
        data_entropy += valProb * entropy(attributes, dataSubset, targetAttr)

    return entropy(attributes, data, targetAttr) - data_entropy


def attr_choose(data, attributes, target):
    """Choosing Attribute with Maximum Gain"""
    best = attributes[0]
    maxGain = 0

    for attr in attributes:
        newGain = info_gain(attributes, data, attr, target) 
        if newGain > maxGain:
            maxGain = newGain
            best = attr

    return best


def get_values(data, attributes, attr):
    """Getting a unique attribute from the data"""
    index = attributes.index(attr)
    values = []

    for entry in data:
        if entry[index] not in values:
            values.append(entry[index])

    return values


def get_data(data, attributes, best, val):
    """This function get all the rows of the data where the chosen best attribute has it's value"""
    new_data = [[]]
    index = attributes.index(best)

    for entry in data:
        # find entries with the give value
        if entry[index] == val:
            newEntry = []
            # add value if it is not in best column
            for i in range(0, len(entry)):
                if i != index:
                    newEntry.append(entry[i])
            new_data.append(newEntry)
    new_data.remove([])

    return new_data



def build_tree(data, attributes, target):
    """This function is used to build the decision tree"""
    data = data[:]
    vals = [record[attributes.index(target)] for record in data]
    default = majorClass(attributes, data, target)

    if not data or (len(attributes) - 1) <= 0:
        return default
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    else:
        best = attr_choose(data, attributes, target)
        tree = {
            best: {}
        }
    
        for val in get_values(data, attributes, best):
            new_data = get_data(data, attributes, best, val)
            newAttr = attributes[:]
            newAttr.remove(best)
            subtree = build_tree(new_data, newAttr, target)

            # Add the new subtree to the empty dictionary object in our new
            # tree/node we just created.

            tree[best][val] = subtree
    
    return tree


def run_decision_tree():
    """Function that runs the decision tree algorithm"""
    data = []

    with open("dataset.tsv") as tsv:
        for line in csv.reader(tsv, delimiter="\t"):

            if line[0] > '37':
                line[0] = '1'
            else:
                line[0] = '0'

            if line[2] > '178302':
                line[2] = '1'
            else:
                line[2] = '0'

            data.append(tuple(line))

        print("Number of records: %d" % len(data))

        # Using discrete Discrete Splitting for attributes "age" and "fnlwght"
        attributes = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-info_gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary']
        target = attributes[-1]

        K = 10
        acc = []
        for k in range(K):
            random.shuffle(data)
            training_set = [x for i, x in enumerate(data) if i % K != k]
            test_set = [x for i, x in enumerate(data) if i % K == k]
            tree = DecisionTree()
            tree.learn(training_set, attributes, target)
            results = []

            for entry in test_set:
                tempDict = tree.tree.copy()
                result = ""

                while isinstance(tempDict, dict):
                    root = Node(tempDict.keys()[0], tempDict[tempDict.keys()[0]])
                    tempDict = tempDict[tempDict.keys()[0]]
                    index = attributes.index(root.value)
                    value = entry[index]

                    if value in tempDict.keys():
                        child = Node(value, tempDict[value])
                        result = tempDict[value]
                        tempDict = tempDict[value]
                    else:
                        result = "Null"
                        break

                if result != "Null":
                    results.append(result == entry[-1])

            accuracy = float(results.count(True)) / float(len(results))
            print("Accuracy is ", accuracy)
            acc.append(accuracy)

        avg_acc = sum(acc)/len(acc)
        print("Average accuracy: %.4f" % avg_acc)

        plt.boxplot(acc)
        plt.title('Average Accuracy Plot')
        plt.savefig('Accuracy')
        plt.show()

        # Writing results to a file (D)
        f = open("result.txt", "w")
        f.write("accuracy: %.5f" % avg_acc)
        f.write("\nPercentage accuracy: %.5f" % avg_acc*100)
        f.close()


if __name__ == "__main__":
    run_decision_tree()
