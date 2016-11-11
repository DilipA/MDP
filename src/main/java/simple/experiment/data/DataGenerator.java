package simple.experiment.data;

import com.google.common.collect.Lists;
import simple.MDP.Action;
import simple.MDP.MDP;
import simple.MDP.State;
import simple.MDP.Trajectory;
import simple.MDP.exceptions.MDPException;
import simple.sample.RandomMDP;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * Created by dilip on 11/7/16.
 */
public class DataGenerator {

    public static Trajectory generateTrajectory(int length, MDP mdp){
        Trajectory ret = new Trajectory(mdp.getStates(), mdp.getActions());
        State current = mdp.getRandomState();
        ret.intialize(current);
        while(ret.getLength() < length){
            Action a = mdp.getRandomAction();
            State next = mdp.sampleTransition(current, a);
            double reward = mdp.getReward(current, next, a);
            ret.step(a, reward, next);
            current = next;
        }

        return ret;
    }

    public static List<Trajectory> generateNTrajectories(int length, int N, MDP mdp){
        List<Trajectory> ret = new ArrayList<>();
        IntStream.range(0, N).forEach(i -> ret.add(generateTrajectory(length, mdp)));
        return ret;
    }

    public static Trajectory generateSATrajectory(State s, Action a, MDP mdp){
        Trajectory ret = new Trajectory(mdp.getStates(), mdp.getActions());
        ret.intialize(s);
        State next = mdp.sampleTransition(s, a);
        double reward = mdp.getReward(s, next, a);
        ret.step(a, reward, next);
        return ret;
    }

    public static List<Trajectory> generateNSATrajectories(int n, MDP mdp){
        List<Trajectory> ret = new ArrayList<>();
        for(State s : mdp.getStates()){
            for(Action a : mdp.getActions()){
                for(int i=0;i < n;i++){
                    ret.add(generateSATrajectory(s, a, mdp));
                }
            }
        }
        return ret;
    }

    public static void main(String[] args) throws MDPException {
        MDP randomMDP = RandomMDP.sample();

        //System.out.println(generateTrajectory(10, randomMDP));

        List<Trajectory> trajectories = generateNTrajectories(10, 10, randomMDP);
//        for(Trajectory t : trajectories){
//            System.out.println(t);
//            System.out.println("\n\n");
//        }
    }
}
