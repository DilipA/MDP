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
    
    if not args.epsilon:
        output_file = 'policy_counts.csv'
    else:
        output_file = 'policy_counts_epsilon.csv'
        
    if not args.no_parse:
        onlyfiles = [f for f in listdir('/home/darumuga/') if isfile(join('/home/darumuga/', f))]
        if not args.epsilon:
            onlyfiles = [f for f in onlyfiles if 'run_policy_count.o' in f]
        else:
            onlyfiles = [f for f in onlyfiles if 'run_policy_count_eps.o' in f]

        with open(output_file, 'wb') as out:
            for f in onlyfiles:
                with open(f, 'rb') as inp:
                    for line in inp:
                        out.write(line)

    count_data = defaultdict(list)

    with open(output_file, 'rb') as inp:
        for line in inp:
            split = line.split(',')
            count_data[float(split[0])] += [int(split[1])]
                
    tuples = sorted(list(count_data.iteritems()), key=lambda x: x[0])
    plt.plot([x[0] for x in tuples], [np.mean(x[1]) for x in tuples], 'g--')
    plt.errorbar([x[0] for x in tuples], [np.mean(x[1]) for x in tuples], yerr=1.96*np.array([np.std(x[1]) for x in tuples])*(1/np.sqrt(num_trials)), color='g')

    plt.title('Number of optimal policies vs. Discount Factor', loc='center')
    if not args.epsilon:
        plt.xlabel('Gamma')
    else:
        plt.xlabel('Epsilon')
    plt.ylabel('Number of optimal policies')
    plt.xlim(-0.1,1.1)
    plt.show()
