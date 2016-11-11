package simple.experiment.model_based;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import simple.MDP.Action;
import simple.MDP.MDP;
import simple.MDP.State;
import simple.MDP.Trajectory;
import simple.MDP.exceptions.MDPException;
import simple.experiment.data.DataGenerator;
import simple.sample.RandomMDP;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Class for producing an MDP where the model (transitions and rewards) is constructed from maximum likelihood estimates
 * of observed data
 * Created by dilip on 11/7/16.
 */
public class MDPEstimator {

    /**
     * An simple.MDP is composed of States.
     */
    private final Set<State> states;

    /**
     * In a State we can take an Action.
     */
    private final Set<Action> actions;

    /**
     * There is a probability of transition from one state to another by taking an action.
     */
    private final Table<State, Action, Map<State, Double>> transitionsEst;

    private final Table<State, Action, Map<State, Double>> transitionsMean;

    private final Table<State, Action, Map<State, Double>> transitionsReg;



    /**
     * Transition from one state to another yields a reward.
     */
    private final Table<State, Action, Map<State, Double>> rewards;

    /**
     * The resulting MDP from computing ML estimates over transitions and rewards
     */
    private final MDP mdp;

    public MDPEstimator(Set<State> states, Set<Action> actions, List<Trajectory> data) throws MDPException {
        this.states = states;
        this.actions = actions;
        this.transitionsEst = HashBasedTable.create();
        this.transitionsMean = HashBasedTable.create();
        this.transitionsReg = HashBasedTable.create();
        this.rewards = HashBasedTable.create();
        this.estimateModel(data, 0.0);
        this.mdp = new MDP(this.states, this.actions, this.transitionsEst, this.rewards);
    }

    public MDPEstimator(double epsilon, Set<State> states, Set<Action> actions, List<Trajectory> data) throws MDPException {
        this.states = states;
        this.actions = actions;
        this.transitionsEst = HashBasedTable.create();
        this.transitionsMean = HashBasedTable.create();
        this.transitionsReg = HashBasedTable.create();
        this.rewards = HashBasedTable.create();
        this.estimateModel(data, epsilon);
        this.mdp = new MDP(this.states, this.actions, this.transitionsReg, this.rewards);
    }

    private void estimateModel(List<Trajectory> data, double epsilon){
        for(State s : this.states){
            for(Action a : this.actions){
                Map<State, Double> rewardMap = new HashMap<>();
                Map<State, Double> transitionDataMap = new HashMap<>();
                int observedSA = data.stream().mapToInt(t -> t.getStateActionCounter().get(s, a)).sum();
                if(observedSA > 0){
                    double rewardSum = data.stream().mapToDouble(t -> t.getStateActionReward().get(s, a)).sum();
                    for (State sprime : this.states) {
                        int observedSASP = data.stream().mapToInt(t -> t.getStateActionTransition().get(s, a).get(sprime)).sum();
                        transitionDataMap.put(sprime, ((double) observedSASP) / observedSA);
                        rewardMap.put(sprime, rewardSum / observedSA);
                    }
                }
                else {
                    for(State sprime : this.states){
                        rewardMap.put(sprime, 0.5);
                        transitionDataMap.put(sprime, 1.0 / this.states.size());
                    }
                }
                this.rewards.put(s, a, rewardMap);
                this.transitionsEst.put(s, a, transitionDataMap);
            }
        }

        for(State s : this.states){
            for(Action a : this.actions){
                for(State sprime : this.states){
                    if(this.transitionsMean.get(s, a) == null){
                        this.transitionsMean.put(s, a, new HashMap<>());
                    }
                    double mean = 0.0;
                    for(Action aprime : this.actions){
                        mean += this.transitionsEst.get(s, aprime).get(sprime);
                    }
                    this.transitionsMean.get(s, a).put(sprime, mean);
                }
            }
        }

        for(State s : this.states){
            for(Action a : this.actions){
                double norm = this.transitionsMean.get(s, a).values().stream().mapToDouble(i -> i).sum();
                for(State sprime : this.states){
                    this.transitionsMean.get(s, a).put(sprime, this.transitionsMean.get(s, a).get(sprime) / norm);
                }
            }
        }

        for(State s : this.states){
            for(Action a : this.actions){
                Map<State, Double> transitionMap = new HashMap<>();
                for(State sprime : this.states){
                    transitionMap.put(sprime, (1 - epsilon) * this.transitionsEst.get(s, a).get(sprime) + epsilon * this.transitionsMean.get(s, a).get(sprime));
                }
                this.transitionsReg.put(s, a, transitionMap);
            }
        }
    }

    public MDP getMdp() {
        return this.mdp;
    }

    public static void main(String[] args) throws MDPException {
        MDP randomMDP = RandomMDP.sample();

        System.out.println(randomMDP);
        //System.out.println(generateTrajectory(10, randomMDP));

        List<Trajectory> trajectories = DataGenerator.generateNTrajectories(10, 10, randomMDP);
//        for(Trajectory t : trajectories){
//            System.out.println(t);
//            System.out.println("\n\n");
//        }

        MDPEstimator estimator = new MDPEstimator(randomMDP.getStates(), randomMDP.getActions(), trajectories);
        System.out.println(estimator.getMdp());
    }
}
