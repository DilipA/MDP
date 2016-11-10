import sys
import argparse
import numpy as np
from os import listdir
from os.path import isfile, join
from collections import defaultdict
import scipy.stats as ss
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--no_parse', action='store_true', help='Skip parsing Grid data files in the current directory')

    args = parser.parse_args()

    num_samples = 1000
    trajecs = [5, 10, 20, 50]
    gammas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

    if not args.no_parse:
        onlyfiles = [f for f in listdir('/home/darumuga/') if isfile(join('/home/darumuga/', f))]
        onlyfiles = [f for f in onlyfiles if 'run_figure3.o' in f]
    
        with open('figure3_results.csv', 'wb') as out:
            for f in onlyfiles:
                with open(f, 'rb') as inp:
                    for line in inp:
                        out.write(line)

    data = defaultdict(lambda: defaultdict(list))
    
    with open('figure3_results.csv', 'rb') as inp:
        for line in inp:
            split = line.split(',')
            data[int(split[0])][float(split[1])] += [float(split[2])]

    style = ['r--', 'b--', 'g--', 'm--']
    legend_handles = []
    for index, i in enumerate(trajecs):
        tuples = sorted(list(data[i].iteritems()), key=lambda x: x[0])
        l, = plt.plot([x[0] for x in tuples], [np.mean(x[1]) for x in tuples], style[index], label='{0} trajectories'.format(i))
        #plt.errorbar([x[0] for x in tuples], [np.mean(x[1]) for x in tuples], yerr=ss.t.ppf(0.95, np.array([num_samples-1]*len(gammas)))*np.array([np.std(x[1]) for x in tuples]), color=style[index][0])
        legend_handles.append(l)
    
    plt.legend(handles=legend_handles, loc=2)
    plt.title('Planning loss vs. Discount factor', loc='center')
    plt.xlabel('Gamma')
    plt.ylabel('Planning Loss')
    plt.show()
