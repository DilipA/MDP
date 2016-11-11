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
    parser.add_argument('--epsilon', action='store_true', help='Generate results for epsilon greedy experiments')

    args = parser.parse_args()

    num_trials = 1000
    samples = [2, 5, 10, 20]
    
    if not args.epsilon:
        output_file = 'figure1_results.csv'
    else:
        output_file = 'figure1_epsilon_results.csv'
        
    if not args.no_parse:
        onlyfiles = [f for f in listdir('/home/darumuga/') if isfile(join('/home/darumuga/', f))]
        if not args.epsilon:
            onlyfiles = [f for f in onlyfiles if 'run_figure1.o' in f]
        else:
            onlyfiles = [f for f in onlyfiles if 'run_figure1_eps.o' in f]

        with open(output_file, 'wb') as out:
            for f in onlyfiles:
                with open(f, 'rb') as inp:
                    for line in inp:
                        out.write(line)

    training_data = defaultdict(lambda: defaultdict(list))
    testing_data = defaultdict(lambda: defaultdict(list))

    with open(output_file, 'rb') as inp:
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
    plt.errorbar([x[0] for x in tuples], [np.mean(x[1]) for x in tuples], yerr=1.96*np.array([np.std(x[1]) for x in tuples])*(1/np.sqrt(num_trials)), color=style[0][0])


    tuples = sorted(list(testing_data[i].iteritems()), key=lambda x: x[0])
    tel, = plt.plot([x[0] for x in tuples], [np.mean(x[1]) for x in tuples], style[1], label='Testing Loss')
    plt.errorbar([x[0] for x in tuples], [np.mean(x[1]) for x in tuples], yerr=1.96*np.array([np.std(x[1]) for x in tuples])*(1/np.sqrt(num_trials)), color=style[1][0])

    plt.legend(handles=[trl, tel], loc=9)
    plt.title('{0} samples per (s,a) pair'.format(i), loc='center')
    if not args.epsilon:
        plt.xlabel('Gamma')
    else:
        plt.xlabel('Epsilon')
    plt.ylabel('Loss')
    plt.xlim(-0.1,1.1)
    plt.ylim(-82.0, -70.0)
    plt.show()
    #plt.save('figure1_{0}_eps.png'.format(i))
