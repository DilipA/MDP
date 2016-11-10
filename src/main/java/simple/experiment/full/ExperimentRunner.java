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

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * Created by dilip on 11/8/16.
 */
public class ExperimentRunner {

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

        for (Integer n : nVals) {
            List<Trajectory> dataset = DataGenerator.generateNSATrajectories(n, randomMDP);
            MDPEstimator estimator = new MDPEstimator(randomMDP.getStates(), randomMDP.getActions(), dataset);
            MDP estimatedMDP = estimator.getMdp();

            for (Double gamma : gammas) {
                double gammaEval = 0.99;

                ValueIteration train_policy = new ValueIteration(estimatedMDP, gamma);
                train_policy.run();
                train_policy.computePolicy();

                PolicyEvaluation train_eval = new PolicyEvaluation(estimatedMDP, gammaEval, train_policy.getPolicy());
                train_eval.run();
                Map<State, Double> v1 = train_eval.getValueFunction();

                double trainingLoss = 0.0;
                for(State s : randomMDP.getStates()) {
                    trainingLoss += v1.get(s);
                }

                trainingLoss = - trainingLoss / randomMDP.getStates().size();
                if(results.get(n, gamma) == null) {
                    results.put(n, gamma, new HashMap<>());
                }
                results.get(n, gamma).put(0, trainingLoss);

                ValueIteration test_policy = new ValueIteration(estimatedMDP, gamma);
                test_policy.run();
                test_policy.computePolicy();

                PolicyEvaluation test_eval = new PolicyEvaluation(randomMDP, gammaEval, test_policy.getPolicy());
                test_eval.run();
                Map<State, Double> v2 = test_eval.getValueFunction();

                double testingLoss = 0.0;
                for(State s : randomMDP.getStates()) {
                    testingLoss += v2.get(s);
                }

                testingLoss = - testingLoss / randomMDP.getStates().size();
                if(results.get(n, gamma) == null) {
                    results.put(n, gamma, new HashMap<>());
                }
                results.get(n, gamma).put(1, testingLoss);
            }
        }

        for(Integer n : nVals){
            for(Double gamma : gammas){
                for(int i=0;i <= 1;i++){
                    System.out.println(n + "," + gamma + "," + i + "," + results.get(n, gamma).get(i));
                }
            }
        }
    }

    public static void runFigure3() throws MDPException {

        long startTime = System.currentTimeMillis();

        //Draw a single MDP from RandomMDP
        MDP randomMDP = RandomMDP.sample();

        Table<Integer, Double, List<Double>> results = HashBasedTable.create();

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
            System.out.println("Running on " + n + " trajectories of length 10");
            IntStream.range(0, numDatasets).parallel().forEach(i -> {
                try {
                    System.out.println("Runing on dataset " + (i+1) + " of " + numDatasets);
                    List<Trajectory> dataset = DataGenerator.generateNTrajectories(10, n, randomMDP);
                    MDPEstimator estimator = new MDPEstimator(randomMDP.getStates(), randomMDP.getActions(), dataset);
                    MDP estimatedMDP = estimator.getMdp();

                    for (Double gamma : gammas) {
                        //System.out.println("Running for gamma = " + gamma);
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
                        if (results.get(n, gamma) == null) {
                            results.put(n, gamma, new ArrayList<>());
                        }
                        results.get(n, gamma).add(empiricalLoss);
                    }
                } catch (MDPException e) {
                    e.printStackTrace();
                }
            });
        }

        try(BufferedWriter bw = new BufferedWriter(new FileWriter(new File("figure3_results.csv")))){
            for(Integer n : nVals){
                for(Double gamma : gammas){
                    for(Double loss : results.get(n, gamma)){
                        bw.write(n + "," + gamma + "," + loss + "\n");
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        long stopTime = System.currentTimeMillis();
        long elapsedTime = stopTime - startTime;
        System.out.println(elapsedTime);

    }

    public static void runFigure3Grid() throws MDPException {
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

        for(Integer n : nVals) {
            List<Trajectory> dataset = DataGenerator.generateNTrajectories(10, n, randomMDP);
            MDPEstimator estimator = new MDPEstimator(randomMDP.getStates(), randomMDP.getActions(), dataset);
            MDP estimatedMDP = estimator.getMdp();

            for (Double gamma : gammas) {
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
                results.put(n, gamma, empiricalLoss);
            }
        }

        for(Integer n : nVals){
            for(Double gamma : gammas){
                System.out.println(n + "," + gamma + "," + results.get(n, gamma));
            }
        }
    }

    public static void main(String[] args) throws MDPException {
        runFigure1Grid();
    }
}
