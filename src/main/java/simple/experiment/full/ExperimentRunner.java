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
import java.util.List;
import java.util.Map;

/**
 * Created by dilip on 11/8/16.
 */
public class ExperimentRunner {

    public static void runFigure1() throws MDPException {

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
            for(int i=0;i < numDatasets;i++) {
                System.out.println("Runing on dataset " + (i+1) + " of " + numDatasets);
                List<Trajectory> dataset = DataGenerator.generateNTrajectories(10, n, randomMDP);
                MDPEstimator estimator = new MDPEstimator(randomMDP.getStates(), randomMDP.getActions(), dataset);
                MDP estimatedMDP = estimator.getMdp();

                for (Double gamma : gammas) {
//                    System.out.println("Running for gamma = " + gamma);
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
                        results.put(n, gamma, new ArrayList<>());
                    }
                    results.get(n, gamma).add(empiricalLoss);
                }
            }
        }

//        for(Integer n : nVals){
//            for(Double gamma : gammas){
//                results.put(n, gamma, results.get(n, gamma));
//                System.out.println(n + "\t" + gamma + "\t" + results.get(n, gamma));
//            }
//        }
//        System.out.println(results);

        try(BufferedWriter bw = new BufferedWriter(new FileWriter(new File("out/figure3_results.csv")))){
            bw.write("Trajectories, gamma, loss\n");
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

    public static void main(String[] args) throws MDPException {
        runFigure3();
    }
}
