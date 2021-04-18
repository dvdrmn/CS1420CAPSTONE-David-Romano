import csv
from collections import defaultdict, Counter
import pandas as pd
import random
import string
import numpy
import numpy as np
import math
import itertools


input_variables = ['Date', 'VH', 'Open', 'Close', 'Final']


def partition_loss(subsets):
    num_obs = sum(len(subset) for subset in subsets)

    loss = 0
    for subset in subsets:
        counter = Counter(label for _, label in subset)
        prediction = counter.most_common(1)
        h = (1 - prediction[0][1]/float(len(subset)))*(len(subset)/float(num_obs))
        loss = loss + h

    return loss

def partition_by(inputs, attribute):
    groups = defaultdict(list)
    for input in inputs:
        key = input[0][attribute]
        groups[key].append(input)
    return groups

def partition_loss_by(inputs, attribute):
    partitions = partition_by(inputs, attribute)
    return partition_loss(partitions.values())


def classify(tree, to_classify):
    if isinstance(tree, int):
        return tree
    if isinstance(tree, float):
        return tree

    attribute, subtree_dict = tree
    subtree_key = to_classify.get(attribute)

    try:
        subtree = subtree_dict[subtree_key]
    except KeyError:
        subtree = subtree_dict[list(subtree_dict.keys())[0]]
    return classify(subtree, to_classify)

def bootstrap_sample(inputs, length):
    a = len(inputs)
    indices = range(a)
    sampled_indices = numpy.random.choice(indices, size=length)
    return [inputs[i] for i in sampled_indices]

def build_forest_tree(inputs, num_levels, num_split_candidates, split_candidates = None):

    if split_candidates == None:
        split_candidates = inputs[0][0].keys()
        split_candidates = list(split_candidates)


    if len(split_candidates) <= num_split_candidates:
        sampled_split_candidates = split_candidates
    else:
        sampled_split_candidates = random.sample(split_candidates, num_split_candidates)

    print(split_candidates)

    if num_levels == 0:
        return sum([x[1] for x in inputs])/float(len(inputs))

    if not split_candidates:
        return sum([x[1] for x in inputs])/float(len(inputs))

    best_attribute = random.choice(sampled_split_candidates)
    best_loss = partition_loss_by(inputs, best_attribute)
    for candidate in sampled_split_candidates:
        e = partition_loss_by(inputs, candidate)
        if e < best_loss:
            best_attribute = candidate
            best_loss = e

    partitions = partition_by(inputs, best_attribute)
    new_candidates = [a for a in split_candidates if a != best_attribute]

    subtrees = {attribute_value : build_forest_tree(subset, num_levels - 1,num_split_candidates, new_candidates) for attribute_value, subset in partitions.items()}

    #subtrees[None] = num_trues > num_falses

    return(best_attribute, subtrees)

def build_tree(inputs, num_levels, split_candidates = None):

    if split_candidates == None:
        split_candidates = list(inputs[0][0].keys())

    # print(split_candidates)

    day_vals = [tuple[1] for tuple in inputs]
    if(len(day_vals) == 0):
        return 0
    all_same = all(x == day_vals[0] for x in day_vals)

    if all_same == True:
        return sum(day_vals)/len(day_vals)

    total_length = len(split_candidates)
    if (total_length == 0) or (num_levels == 0):
        average = sum(day_vals)/len(day_vals)
        return average

    loss_array = []
    for index in range(len(split_candidates)):
        loss_array.append(partition_loss_by(inputs, split_candidates[index]))

    index_of_best_attribute = loss_array.index(min(loss_array))
    best_attribute = split_candidates[index_of_best_attribute]
    # print(best_attribute)
    best_attribute_partitions = partition_by(inputs, best_attribute)
    # print(len(inputs))
    # print(len(best_attribute_partitions[0]))
    # print(len(best_attribute_partitions[1]))
    split_candidates.remove(best_attribute)
    return (best_attribute, {1: build_tree(best_attribute_partitions[1], num_levels-1, split_candidates), 0: build_tree(best_attribute_partitions[0], num_levels-1, split_candidates)})




def forest_classify(trees, loan):
    votes = [classify(tree, loan[0]) for tree in trees]
    return sum(votes) / float(len(votes))


def load_data():
    data = []
    f = open('NBADATA/nfl21.csv', "rt")
    reader = csv.DictReader(f)
    trainingarr = []
    testingarr = []
    data = []
    for row in reader:
        data.append(row)
    for i in range(0, len(data), 2):
        inputs = {}
        first_or_second = random.randint(0, 1)
        if (data[i+first_or_second]['VH'] == 'H'):
            inputs['VH'] = 1
        else:
            inputs['VH'] = 0
        if (str(data[i+first_or_second]['Open']) == 'pk'):
            inputs['Open'] = 0
        elif (float(data[i+first_or_second]['Open']) < 30):
            inputs['Open'] = -1 * float(data[i+first_or_second]['Open'])
        else :
            if (str(data[i+(1-first_or_second)]['Open']) == 'pk'):
                inputs['Open'] = 0
            else:
                inputs['Open'] = float(data[i + (1 - first_or_second)]['Open'])
        if (str(data[i+first_or_second]['Close']) == 'pk'):
            inputs['Close'] = 0
        elif (float(data[i+first_or_second]['Close']) < 30):
            inputs['Close'] = -1 * float(data[i+first_or_second]['Close'])
        else :
            if (str(data[i+(1-first_or_second)]['Close']) == 'pk'):
                inputs['Close'] = 0
            else:
                inputs['Close'] = float(data[i + (1 - first_or_second)]['Close'])
        final_diff = float(data[i + (first_or_second)]['Final']) - float(data[i + (1 - first_or_second)]['Final'])

        if (final_diff > ((-1) * inputs['Close'])):
            label = 1
        else:
            label = 0
        r = random.random()
        if r <= 0.8:
            trainingarr.append((inputs, label))
        else:
            testingarr.append((inputs,label))
    return trainingarr, testingarr


def main():
    # print("here")
    trainingarr, testingarr = load_data()

    tree = build_forest_tree(trainingarr, 3,4)
    print('bighere')

    print(len(testingarr))
    print(tree)

    total_error = 0
    for i in range(len(testingarr)):
    	total_error += ((classify(tree, testingarr[i][0]) - testingarr[i][1])**2)/len(testingarr)

    print(total_error)




    # error_list = []
    # max = 5
    # x_vals = np.arange(1,max+1)
    # for x in range(max):
    # 	print(x)
    # 	tree = build_tree(loans,x+1)
    # 	total_error = 0
    # 	for i in range(len(loans)):
    # 		total_error += ((classify(tree, loans[i][0]) - loans[i][1])**2)/len(loans)
    #
    # 	error_list.append(total_error)
    #
    # print(x_vals)
    # print(error_list)
    # plt.plot(x_vals, error_list)
    # plt.ylabel("error")
    # plt.xlabel("tree depth")

    #plt.show()

main()
