package simple.experiment.full;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Lists;
import com.google.common.collect.Table;
import simple.MDP.MDP;
import simple.MDP.State;
import simple.MDP.Trajectory;
import simple.MDP.exceptions.MDPException;
import simple.algorithms.PolicyEvaluation;
import simple.algorithms.ValueIteration;
import simple.experiment.data.DataGenerator;
import simple.experiment.model_based.MDPEstimator;
import simple.sample.RandomMDP;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Created by dilip on 11/8/16.
 */
public class ExperimentRunner {

    public static void runFigure3() throws MDPException {

        long startTime = System.currentTimeMillis();

        //Draw a single MDP from RandomMDP
        MDP randomMDP = RandomMDP.sample();

        Table<Integer, Double, Double> results = HashBasedTable.create();

        List<Integer> nVals = new ArrayList<>();
        nVals.add(5);
        nVals.add(10);
        nVals.add(20);
        nVals.add(50);

        List<Double> gammas = new ArrayList<>();
        gammas.add(0.0);
        gammas.add(0.1);
        gammas.add(0.2);
        gammas.add(0.3);
        gammas.add(0.4);
        gammas.add(0.5);
        gammas.add(0.6);
        gammas.add(0.7);
        gammas.add(0.8);
        gammas.add(0.9);
        gammas.add(0.99);

        int numDatasets = 1000;

        for(Integer n : nVals){
            System.out.println("Running for trajectories of length " + n);
            for(int i=0;i < numDatasets;i++) {
                List<Trajectory> dataset = DataGenerator.generateNTrajectories(n, 5, randomMDP);
                MDPEstimator estimator = new MDPEstimator(randomMDP.getStates(), randomMDP.getActions(), dataset);
                MDP estimatedMDP = estimator.getMdp();

                for (Double gamma : gammas) {
                    System.out.println("Running for gamma = " + gamma);
                    double gammaEval = 0.99;

                    ValueIteration vi1 = new ValueIteration(randomMDP, gammaEval);
                    vi1.run();
                    vi1.computePolicy();

                    PolicyEvaluation pe1 = new PolicyEvaluation(randomMDP, gammaEval, vi1.getPolicy());
                    pe1.run();
                    Map<State, Double> v1 = pe1.getValueFunction();

                    ValueIteration vi2 = new ValueIteration(estimatedMDP, gamma);
                    vi2.run();
                    vi2.computePolicy();

                    PolicyEvaluation pe2 = new PolicyEvaluation(randomMDP, gammaEval, vi2.getPolicy());
                    pe2.run();
                    Map<State, Double> v2 = pe2.getValueFunction();

                    double sumDiff = 0.0;
                    for (State s : randomMDP.getStates()) {
                        sumDiff += v1.get(s) - v2.get(s);
                    }

                    double empiricalLoss = sumDiff / randomMDP.getStates().size();
                    if(results.get(n, gamma) == null){
                        results.put(n, gamma, 0.0);
                    }
                    results.put(n, gamma, empiricalLoss + results.get(n, gamma));
                }
            }
        }

        for(Integer n : nVals){
            for(Double gamma : gammas){
                results.put(n, gamma, results.get(n, gamma) / numDatasets);
                System.out.println(n + "\t" + gamma + "\t" + results.get(n, gamma) / numDatasets);
            }
        }
        System.out.println(results);

        long stopTime = System.currentTimeMillis();
        long elapsedTime = stopTime - startTime;
        System.out.println(elapsedTime);

    }

    public static void main(String[] args) throws MDPException {
        runFigure3();
    }
}
