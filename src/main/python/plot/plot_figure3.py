import sys
import argparse
import numpy as np
from os import listdir
from os.path import isfile, join
from collections import defaultdict
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--no_parse', action='store_true', help='Skip parsing Grid data files in the current directory')
    parser.add_argument('--epsilon', action='store_true', help='Generate results for epsilon greedy experiments')
    parser.add_argument('--boltzmann', action='store_true', help='Generate results for Boltzmann action selection experiments')

    args = parser.parse_args()

    num_samples = 1000
    trajecs = [5, 10, 20, 50]

    if args.boltzmann:
        output_file = 'figure3_boltzmann_results.csv'
    elif args.epsilon:
        output_file = 'figure3_epsilon_results.csv'
    else:
        output_file = 'figure3_gamma_results.csv'

    if not args.no_parse:
        onlyfiles = [f for f in listdir('/home/darumuga/') if isfile(join('/home/darumuga/', f))]
        if args.boltzmann:
            onlyfiles = [f for f in onlyfiles if 'run_figure3_beta.o' in f]
        elif args.epsilon:
            onlyfiles = [f for f in onlyfiles if 'run_figure3_eps.o' in f]
        else:
            onlyfiles = [f for f in onlyfiles if 'run_figure3.o' in f]

        empty_count = 0
        with open(output_file, 'wb') as out:
            for f in onlyfiles:
                if len(open(f, 'rb').readline()) == 0:
                    empty_count += 1
                with open(f, 'rb') as inp:
                    for line in inp:
                        out.write(line)
        print 'Failure rate: {0}/{1} = {2}'.format(empty_count, num_samples, float(empty_count) / float(num_samples))

    data = defaultdict(lambda: defaultdict(list))
    
    with open(output_file, 'rb') as inp:
        for line in inp:
            split = line.split(',')
            data[int(split[0])][float(split[1])] += [float(split[2])]

    style = ['r--', 'b--', 'g--', 'm--']
    legend_handles = []
    for index, i in enumerate(trajecs):
        tuples = sorted(list(data[i].iteritems()), key=lambda x: x[0])
        l, = plt.plot([x[0] for x in tuples], [np.mean(x[1]) for x in tuples], style[index], label='{0} trajectories'.format(i))
        plt.errorbar([x[0] for x in tuples], [np.mean(x[1]) for x in tuples], yerr=1.96*np.array([np.std(x[1]) for x in tuples])*(1.0/np.sqrt(num_samples)), color=style[index][0])
        legend_handles.append(l)
    
    plt.legend(handles=legend_handles, loc=1)
    if args.boltzmann:
        plt.title('Planning loss vs. Boltzmann temperature', loc='center')
        plt.xlabel('Beta')
    elif args.epsilon:
        plt.title('Planning loss vs. Exploration', loc='center')
        plt.xlabel('Epsilon')
    else:
        plt.title('Planning loss vs. Discount factor', loc='center')
        plt.xlabel('Gamma')
    plt.ylabel('Planning Loss')
    
    if args.boltzmann:
        plt.xlim(-0.1, 1.0)
    else:
        plt.lim(-0.1, 1.1)
    plt.show()
