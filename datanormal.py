import csv
from collections import defaultdict, Counter
import pandas as pd
import random
import matplotlib.pyplot as plt
import string
import numpy
import numpy as np
import math
import itertools
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

input_variables = ["\"Final\"", "\"Home\"", "\"Open\"", "\"Close\"", "\"Change\""]
input_variables2 = ["\"Home\"", "\"Open\"", "\"Close\"", "\"Change\""]
def load_data():
    data = []
    f = open('NBADATA/nba21.csv', "rt")
    reader = csv.DictReader(f)
    data = []
    final_d = []
    tree_x = []
    tree_y = []
    final_score = []
    line_shift = []
    for row in reader:
        data.append(row)
    for i in range(0, len(data)-2, 2):
        holder = np.zeros(5)
        first_or_second = random.randint(0, 1)
        if (data[i+first_or_second]['VH'] == 'H'):
            holder[1] = 1
        else:
            holder[1] = 0
        if (str(data[i+first_or_second]['Open']) == 'pk'):
            holder[2] = 0
        elif (float(data[i+first_or_second]['Open']) < 30):
            holder[2] = -1 * float(data[i+first_or_second]['Open'])
        else :
            if (str(data[i+(1-first_or_second)]['Open']) == 'pk'):
                holder[2] = 0
            else:
                holder[2] = float(data[i + (1 - first_or_second)]['Open'])
        print(data[i+first_or_second]['Close'])
        print(i)
        if (str(data[i+first_or_second]['Close']) == 'pk'):
            holder[3] = 0
        elif (float(data[i+first_or_second]['Close']) < 30):
            holder[3] = -1 * float(data[i+first_or_second]['Close'])
        else :
            if (str(data[i+(1-first_or_second)]['Close']) == 'pk'):
                holder[3] = 0
            else:
                holder[3] = float(data[i + (1 - first_or_second)]['Close'])
        holder[4] = holder[2] - holder[3]
        if (holder[4] >= 4):
            continue
        final_diff = float(data[i + (first_or_second)]['Final']) - float(data[i + (1 - first_or_second)]['Final'])
        holder[0] = final_diff
        line_shift.append(holder[2] - holder[3])
        final_score.append(final_diff)
        strings = ["%.2f" % number for number in holder]
        final_d.append(strings)


        new_x = np.zeros(len(holder) -1 )

        for j in range(len(holder)):
            if (j != 0):
                new_x[j-1] = holder[j]

        tree_x.append(new_x)
        tree_y.append(holder[0])

    with open("NBADATA/nba21data.txt", "w") as txt_file:
        txt_file.write(" ".join(input_variables) + "\n")
        for line in final_d:
            txt_file.write(" ".join(line) + "\n")

    plt.scatter(line_shift, final_score)
    plt.title("Relationship Between Line Movement and Margin of Victory")
    plt.xlabel("Shift in Point Spread")
    plt.ylabel("Margin of Victory")
    plt.show()

load_data()
