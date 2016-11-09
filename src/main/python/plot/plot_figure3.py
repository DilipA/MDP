import sys
import numpy as np
from os import listdir
from os.path import isfile, join
from collections import defaultdict

import matplotlib.pyplot as plt

onlyfiles = [f for f in listdir('/home/darumuga/') if isfile(join('/home/darumuga/', f))]
onlyfiles = [f for f in onlyfiles if 'run_figure3.o' in f]

trajecs = [5, 10, 20, 50]
gammas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

with open('figure3_results.csv', 'wb') as out:
    for f in onlyfiles:
        with open(f, 'rb') as inp:
            for line in inp:
                out.write(line)

means = defaultdict(lambda: defaultdict(float))

with open('figure3_results.csv', 'rb') as inp:
    for line in inp:
        split = line.split(',')
        means[int(split[0])][float(split[1])] += float(split[2])

for i in trajecs:
    for j in gammas:
        means[i][j] = means[i][j] / 1000

style = ['r--', 'b--', 'g--', 'm--']
legend_handles = []
for index, i in enumerate(trajecs):
    data = sorted(list(means[i].iteritems()), key=lambda x: x[0])
    l, = plt.plot([x[0] for x in data], [x[1] for x in data], style[index], label='{0} trajectories'.format(i))
    legend_handles.append(l)

plt.legend(handles=legend_handles, loc=2)
plt.show()
