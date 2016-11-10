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

    num_trials = 1000
    samples = [2, 5, 10, 20]
    gammas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

    if not args.no_parse:
        onlyfiles = [f for f in listdir('/home/darumuga/') if isfile(join('/home/darumuga/', f))]
        onlyfiles = [f for f in onlyfiles if 'run_figure1.o' in f]

        with open('figure1_results.csv', 'wb') as out:
            for f in onlyfiles:
                with open(f, 'rb') as inp:
                    for line in inp:
                        out.write(line)

    training_data = defaultdict(lambda: defaultdict(list))
    testing_data = defaultdict(lambda: defaultdict(list))

    with open('figure1_results.csv', 'rb') as inp:
        for line in inp:
            split = line.split(',')
            if int(split[2]) == 0:
                training_data[int(split[0])][float(split[1])] += [float(split[3])]
            else:
                testing_data[int(split[0])][float(split[1])] += [float(split[3])]
                
style = ['r--', 'b--']
for i in samples:
    tuples = sorted(list(training_data[i].iteritems()), key=lambda x: x[0])
    trl, = plt.plot([x[0] for x in tuples], [np.mean(x[1]) for x in tuples], style[0], label='Training Loss')
    print ss.t.ppf(0.95, np.array([num_trials-1]*len(gammas)))*np.array([np.std(x[1]) for x in tuples])
    #plt.errorbar([x[0] for x in tuples], [np.mean(x[1]) for x in tuples], yerr=ss.t.ppf(0.95, np.array([num_trials-1]*len(gammas)))*np.array([np.std(x[1]) for x in tuples]), color=style[0][0])


    tuples = sorted(list(testing_data[i].iteritems()), key=lambda x: x[0])
    tel, = plt.plot([x[0] for x in tuples], [np.mean(x[1]) for x in tuples], style[1], label='Testing Loss')
    #plt.errorbar([x[0] for x in tuples], [np.mean(x[1]) for x in tuples], yerr=ss.t.ppf(0.95, np.array([num_trials-1]*len(gammas)))*np.array([np.std(x[1]) for x in tuples]), color=style[1][0])

    plt.legend(handles=[trl, tel], loc=9)
    plt.title('{0} samples per (s,a) pair'.format(i), loc='center')
    plt.xlabel('Gamma')
    plt.ylabel('Loss')
    plt.show()
