import sys
import numpy as np
from os import listdir
from os.path import isfile, join
from collections import defaultdict

import matplotlib.pyplot as plt

onlyfiles = [f for f in listdir('/home/darumuga/') if isfile(join('/home/darumuga/', f))]
onlyfiles = [f for f in onlyfiles if 'run_figure1.o' in f]

num_trials = 1000
samples = [2, 5, 10, 20]
gammas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

with open('figure1_results.csv', 'wb') as out:
    for f in onlyfiles:
        with open(f, 'rb') as inp:
            for line in inp:
                out.write(line)

training_means = defaultdict(lambda: defaultdict(float))
testing_means = defaultdict(lambda: defaultdict(float))

with open('figure1_results.csv', 'rb') as inp:
    for line in inp:
        split = line.split(',')
        if int(split[2]) == 0:
            training_means[int(split[0])][float(split[1])] += float(split[3])
        else:
            testing_means[int(split[0])][float(split[1])] += float(split[3])

for i in samples:
    for j in gammas:
        training_means[i][j] = training_means[i][j] / num_trials
        testing_means[i][j] = testing_means[i][j] / num_trials

style = ['r--', 'b--']

for i in samples:
    data = sorted(list(training_means[i].iteritems()), key=lambda x: x[0])
    trl, = plt.plot([x[0] for x in data], [x[1] for x in data], style[0], label='Training Loss')

    data = sorted(list(testing_means[i].iteritems()), key=lambda x: x[0])
    tel, = plt.plot([x[0] for x in data], [x[1] for x in data], style[1], label='Testing Loss')
    plt.legend(handles=[trl, tel], loc=2)
    plt.show()
