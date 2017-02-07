package simple.experiment.full;

import com.google.common.collect.HashBasedTable;
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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by dilip on 2/1/17.
 */
public class BoltzmannExperimentRunner {

    public static void runFigure1Grid() throws MDPException {
        //Draw a single MDP from RandomMDP
        MDP randomMDP = RandomMDP.sample();

        // Number of samples, Gamma value, 0-1 train-test, loss value
        Table<Integer, Double, Map<Integer, Double>> results = HashBasedTable.create();

        List<Integer> nVals = new ArrayList<>();
        nVals.add(2);
        nVals.add(5);
        nVals.add(10);
        nVals.add(20);

        double gamma = 0.99;

        List<Double> temps = new ArrayList<>();
        temps.add(0.0);
        temps.add(0.5);
        temps.add(0.75);
        temps.add(1.0);
        temps.add(1.5);
        temps.add(2.0);
        temps.add(3.0);
        temps.add(4.0);
        temps.add(5.0);
        temps.add(7.0);
        temps.add(10.0);

        for (Integer n : nVals) {
            List<Trajectory> dataset = DataGenerator.generateNSATrajectories(n, randomMDP);
            for (Double temp : temps) {
                MDPEstimator estimator = new MDPEstimator(randomMDP.getStates(), randomMDP.getActions(), dataset);
                MDP estimatedMDP = estimator.getMdp();

                ValueIteration train_policy = new ValueIteration(estimatedMDP, gamma, temp);
                train_policy.runQ();
                train_policy.computePolicy();

                PolicyEvaluation train_eval = new PolicyEvaluation(estimatedMDP, gamma, train_policy.getPolicy());
                train_eval.run();
                Map<State, Double> v1 = train_eval.getValueFunction();

                double trainingLoss = 0.0;
                for(State s : randomMDP.getStates()) {
                    trainingLoss += v1.get(s);
                }

                trainingLoss = - trainingLoss / randomMDP.getStates().size();
                if(results.get(n, temp) == null) {
                    results.put(n, temp, new HashMap<>());
                }
                results.get(n, temp).put(0, trainingLoss);

                ValueIteration test_policy = new ValueIteration(estimatedMDP, gamma, temp);
                test_policy.runQ();
                test_policy.computePolicy();

                PolicyEvaluation test_eval = new PolicyEvaluation(randomMDP, gamma, test_policy.getPolicy());
                test_eval.run();
                Map<State, Double> v2 = test_eval.getValueFunction();

                double testingLoss = 0.0;
                for(State s : randomMDP.getStates()) {
                    testingLoss += v2.get(s);
                }

                testingLoss = - testingLoss / randomMDP.getStates().size();
                if(results.get(n, temp) == null) {
                    results.put(n, temp, new HashMap<>());
                }
                results.get(n, temp).put(1, testingLoss);
            }
        }

        for(Integer n : nVals){
            for(Double temp : temps){
                for(int i=0;i <= 1;i++){
                    System.out.println(n + "," + temp + "," + i + "," + results.get(n, temp).get(i));
                }
            }
        }
    }

    public static void runFigure3Grid() throws MDPException {
        //Draw a single MDP from RandomMDP
        MDP randomMDP = RandomMDP.sample();

        Table<Integer, Double, Double> results = HashBasedTable.create();

        List<Integer> nVals = new ArrayList<>();
        nVals.add(1);
        nVals.add(3);
        nVals.add(5);
        nVals.add(10);
        nVals.add(20);
        nVals.add(50);

        double gamma = 0.99;

        List<Double> temps = new ArrayList<>();
        temps.add(0.0);
        temps.add(2.0);
        temps.add(5.0);
        temps.add(8.0);
        temps.add(10.0);
        temps.add(12.0);
        temps.add(14.0);
        temps.add(16.0);
        temps.add(18.0);
        temps.add(20.0);
        temps.add(30.0);
        temps.add(40.0);
        temps.add(50.0);
        temps.add(60.0);
        temps.add(70.0);
        temps.add(80.0);
        temps.add(90.0);
        temps.add(100.0);

        for(Integer n : nVals) {
            System.err.println("Running with " + n + " trajectories");
            List<Trajectory> dataset = DataGenerator.generateNTrajectories(10, n, randomMDP);
            MDPEstimator estimator = new MDPEstimator(randomMDP.getStates(), randomMDP.getActions(), dataset);
            MDP estimatedMDP = estimator.getMdp();

            for (Double temp : temps) {
                System.err.println("Running with Boltzmann temperature " + temp);
                double betaEval = 100.0;

                ValueIteration vi1 = new ValueIteration(randomMDP, gamma);
                vi1.run();
                vi1.computePolicy();
                System.err.println(vi1.getQ());

                PolicyEvaluation pe1 = new PolicyEvaluation(randomMDP, gamma, vi1.getPolicy());
//                PolicyEvaluation pe1 = new PolicyEvaluation(randomMDP, vi1.getStochasticPolicy());
                pe1.run();
                Map<State, Double> v1 = pe1.getValueFunction();

                ValueIteration vi2 = new ValueIteration(estimatedMDP, gamma, temp);
                vi2.runQ();
                vi2.computePolicy();
                System.err.println(vi2.getQ());

                PolicyEvaluation pe2 = new PolicyEvaluation(randomMDP, gamma, vi2.getPolicy());
//                PolicyEvaluation pe2 = new PolicyEvaluation(randomMDP, vi2.getStochasticPolicy());
                pe2.run();
                Map<State, Double> v2 = pe2.getValueFunction();

                double sumDiff = 0.0;
                for (State s : randomMDP.getStates()) {
                    sumDiff += v1.get(s) - v2.get(s);
                }

                double empiricalLoss = sumDiff / randomMDP.getStates().size();
                results.put(n, temp, empiricalLoss);
            }
        }

        for(Integer n : nVals){
            for(Double temp : temps){
                System.out.println(n + "," + temp + "," + results.get(n, temp));
            }
        }
    }

    public static void main(String[] args) throws MDPException {
        runFigure3Grid();
    }
}
