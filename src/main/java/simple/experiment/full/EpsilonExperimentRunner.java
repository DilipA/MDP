package simple.experiment.full;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Lists;
import com.google.common.collect.Table;
import simple.MDP.Action;
import simple.MDP.MDP;
import simple.MDP.State;
import simple.MDP.Trajectory;
import simple.MDP.exceptions.MDPException;
import simple.algorithms.PolicyEvaluation;
import simple.algorithms.ValueIteration;
import simple.experiment.data.DataGenerator;
import simple.experiment.model_based.MDPEstimator;
import simple.sample.RandomMDP;

import java.util.*;

/**
 * Created by dilip on 11/8/16.
 */
public class EpsilonExperimentRunner {

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

        List<Double> epsilons = new ArrayList<>();
        epsilons.add(0.0);
        epsilons.add(0.1);
        epsilons.add(0.2);
        epsilons.add(0.3);
        epsilons.add(0.4);
        epsilons.add(0.5);
        epsilons.add(0.6);
        epsilons.add(0.7);
        epsilons.add(0.8);
        epsilons.add(0.9);
        epsilons.add(1.0);

        for (Integer n : nVals) {
            List<Trajectory> dataset = DataGenerator.generateNSATrajectories(n, randomMDP);
            for (Double epsilon : epsilons) {
                MDPEstimator estimator = new MDPEstimator(epsilon, randomMDP.getStates(), randomMDP.getActions(), dataset);
                MDP estimatedMDP = estimator.getMdp();

                ValueIteration train_policy = new ValueIteration(estimatedMDP, gamma);
                train_policy.run();
                train_policy.computePolicy();

                PolicyEvaluation train_eval = new PolicyEvaluation(estimatedMDP, gamma, 0.0, train_policy.getPolicy());
                train_eval.run();
                Map<State, Double> v1 = train_eval.getValueFunction();

                double trainingLoss = 0.0;
                for(State s : randomMDP.getStates()) {
                    trainingLoss += v1.get(s);
                }

                trainingLoss = - trainingLoss / randomMDP.getStates().size();
                if(results.get(n, epsilon) == null) {
                    results.put(n, epsilon, new HashMap<>());
                }
                results.get(n, epsilon).put(0, trainingLoss);

                ValueIteration test_policy = new ValueIteration(estimatedMDP, gamma);
                test_policy.run();
                test_policy.computePolicy();

                PolicyEvaluation test_eval = new PolicyEvaluation(randomMDP, gamma, 0.0, test_policy.getPolicy());
                test_eval.run();
                Map<State, Double> v2 = test_eval.getValueFunction();

                double testingLoss = 0.0;
                for(State s : randomMDP.getStates()) {
                    testingLoss += v2.get(s);
                }

                testingLoss = - testingLoss / randomMDP.getStates().size();
                if(results.get(n, epsilon) == null) {
                    results.put(n, epsilon, new HashMap<>());
                }
                results.get(n, epsilon).put(1, testingLoss);
            }
        }

        for(Integer n : nVals){
            for(Double epsilon : epsilons){
                for(int i=0;i <= 1;i++){
                    System.out.println(n + "," + epsilon + "," + i + "," + results.get(n, epsilon).get(i));
                }
            }
        }
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

        double gamma = 0.99;

        List<Double> epsilons = new ArrayList<>();
        epsilons.add(0.0);
        epsilons.add(0.1);
        epsilons.add(0.2);
        epsilons.add(0.3);
        epsilons.add(0.4);
        epsilons.add(0.5);
        epsilons.add(0.6);
        epsilons.add(0.7);
        epsilons.add(0.8);
        epsilons.add(0.9);
        epsilons.add(1.0);

        for(Integer n : nVals) {
            List<Trajectory> dataset = DataGenerator.generateNTrajectories(10, n, randomMDP);

            for (Double epsilon : epsilons) {
                MDPEstimator estimator = new MDPEstimator(epsilon, randomMDP.getStates(), randomMDP.getActions(), dataset);
                MDP estimatedMDP = estimator.getMdp();

                ValueIteration vi1 = new ValueIteration(randomMDP, gamma);
                vi1.run();
                vi1.computePolicy();

                PolicyEvaluation pe1 = new PolicyEvaluation(randomMDP, gamma, vi1.getPolicy());
                pe1.run();
                Map<State, Double> v1 = pe1.getValueFunction();

                ValueIteration vi2 = new ValueIteration(estimatedMDP, gamma);
                vi2.run();
                vi2.computePolicy();

                PolicyEvaluation pe2 = new PolicyEvaluation(randomMDP, gamma, vi2.getPolicy());
                pe2.run();
                Map<State, Double> v2 = pe2.getValueFunction();

                double sumDiff = 0.0;
                for (State s : randomMDP.getStates()) {
                    sumDiff += v1.get(s) - v2.get(s);
                }

                double empiricalLoss = sumDiff / randomMDP.getStates().size();
                results.put(n, epsilon, empiricalLoss);
            }
        }

        for(Integer n : nVals){
            for(Double epsilon : epsilons){
                System.out.println(n + "," + epsilon + "," + results.get(n, epsilon));
            }
        }
    }

    public static void runPolicyCountGrid() throws MDPException{
        List<Double> epsilons = new ArrayList<>();
        epsilons.add(0.0);
        epsilons.add(0.1);
        epsilons.add(0.2);
        epsilons.add(0.3);
        epsilons.add(0.4);
        epsilons.add(0.5);
        epsilons.add(0.6);
        epsilons.add(0.7);
        epsilons.add(0.8);
        epsilons.add(0.9);
        epsilons.add(1.0);

        double gamma = 0.99;

        Map<Double, Integer> results = new HashMap<>();

        Random rndg = new Random();

        for(Double epsilon : epsilons){
            MDP randomMDP = RandomMDP.sample();

            List<State> states = Lists.newArrayList(randomMDP.getStates());
            Set<String> policies = new HashSet<>();
            double totalPolicies = Math.pow(randomMDP.getActions().size(), randomMDP.getStates().size());
            boolean running = true;
            int noChanges = 0;
            while(running){
                Table<State, Action, Map<State, Double>> transition = HashBasedTable.create();
                for(State s : randomMDP.getStates()){
                    for(Action a : randomMDP.getActions()){
                        Map<State, Double> transitionMap = new HashMap<>();
                        for(State sprime : randomMDP.getStates()){
                            transitionMap.put(sprime, rndg.nextDouble());
                        }
                        double norm = transitionMap.values().stream().mapToDouble(i -> i).sum();
                        for(State sprime : transitionMap.keySet()){
                            transitionMap.put(sprime, transitionMap.get(sprime) / norm);
                        }
                        transition.put(s, a, transitionMap);
                    }
                }

                Table<State, Action, Map<State, Double>> transitionsMean = HashBasedTable.create();

                for(State s : randomMDP.getStates()){
                    for(Action a : randomMDP.getActions()){
                        for(State sprime : randomMDP.getStates()){
                            if(transitionsMean.get(s, a) == null){
                                transitionsMean.put(s, a, new HashMap<>());
                            }
                            double mean = 0.0;
                            for(Action aprime : randomMDP.getActions()){
                                mean += transition.get(s, aprime).get(sprime);
                            }
                            transitionsMean.get(s, a).put(sprime, mean);
                        }
                    }
                }

                for(State s : randomMDP.getStates()){
                    for(Action a : randomMDP.getActions()){
                        double norm = transitionsMean.get(s, a).values().stream().mapToDouble(i -> i).sum();
                        for(State sprime : randomMDP.getStates()){
                            transitionsMean.get(s, a).put(sprime, transitionsMean.get(s, a).get(sprime) / norm);
                        }
                    }
                }

                Table<State, Action, Map<State, Double>> transitionsReg = HashBasedTable.create();

                for(State s : randomMDP.getStates()){
                    for(Action a : randomMDP.getActions()){
                        Map<State, Double> transitionMap = new HashMap<>();
                        for(State sprime : randomMDP.getStates()){
                            transitionMap.put(sprime, (1 - epsilon) * transition.get(s, a).get(sprime) + epsilon * transitionsMean.get(s, a).get(sprime));
                        }
                        transitionsReg.put(s, a, transitionMap);
                    }
                }

                randomMDP.setTransition(transitionsReg);

                ValueIteration vi= new ValueIteration(randomMDP, gamma);
                vi.run();
                vi.computePolicy();
                Map<State, Action> policy = vi.getPolicy();
                StringBuilder sb = new StringBuilder();
                for(State s : states){
                    sb.append(policy.get(s).getId());
                }
                int before = policies.size();
                policies.add(sb.toString());

                if(before == policies.size()){
                    noChanges += 1;
                }
                else{
                    noChanges = 0;
                }

                if(policies.size() == totalPolicies || noChanges >= 5000){
                    running = false;
                }
            }
            results.put(epsilon, policies.size());
        }

        for(Double epsilon : epsilons){
            System.out.println(epsilon + "," + results.get(epsilon));
        }
    }

    public static void main(String[] args) throws MDPException {
        runPolicyCountGrid();
    }
}
