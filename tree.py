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


# This Function which tells which class has more entries in given dataset
def rootclass(attributes, data, target):

    freq = {}
    index = attributes.index(target)

    for row_data in data:
        if freq.has_key(row_data[index]):
            freq[row_data[index]] += 1
        else:
            freq[row_data[index]] = 1

    max_val = 0
    major = ""

    for key in freq.keys():
        if freq[key] > max_val:
            max_val = freq[key]
            major = key

    return major


# Calculates the entropy of the data given the target attribute
def entropy(attributes, data, targetattr):
    freq = {}
    entropy = 0.0

    i = 0
    for entry in attributes:
        if targetattr == entry:
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


def info_gain(attributes, data, attr, targetattr):
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
        val_prob = freq[val] / sum(freq.values())
        subset_data = [entry for entry in data if entry[i] == val]
        data_entropy += val_prob * entropy(attributes, subset_data, targetattr)

    return entropy(attributes, data, targetattr) - data_entropy


def attr_choose(data, attributes, target):
    """Choosing Attribute with Maximum Gain"""
    best = attributes[0]
    max_gain = 0

    for attr in attributes:
        new_gain = info_gain(attributes, data, attr, target)
        if new_gain > max_gain:
            max_gain = new_gain
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
            new_entry = []
            # add value if it is not in best column
            for i in range(0, len(entry)):
                if i != index:
                    new_entry.append(entry[i])
            new_data.append(new_entry)
    new_data.remove([])

    return new_data


def build_tree(data, attributes, target):
    """This function is used to build the decision tree"""
    data = data[:]
    vals = [record[attributes.index(target)] for record in data]
    default = rootclass(attributes, data, target)

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
            new_attr = attributes[:]
            new_attr.remove(best)
            subtree = build_tree(new_data, new_attr, target)

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
                temp_dict = tree.tree.copy()
                result = ""

                while isinstance(temp_dict, dict):
                    root = Node(temp_dict.keys()[0], temp_dict[temp_dict.keys()[0]])
                    temp_dict = temp_dict[temp_dict.keys()[0]]
                    index = attributes.index(root.value)
                    value = entry[index]

                    if value in temp_dict.keys():
                        child = Node(value, temp_dict[value])
                        result = temp_dict[value]
                        temp_dict = temp_dict[value]
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
        f.write("\nPercentage accuracy: %.5f" % (avg_acc*100))
        f.close()


if __name__ == "__main__":
    run_decision_tree()
