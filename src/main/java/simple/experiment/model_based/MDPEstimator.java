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
    private final Table<State, Action, Map<State, Double>> transitions;

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
        this.transitions = HashBasedTable.create();
        this.rewards = HashBasedTable.create();
        this.estimateModel(data);
        this.mdp = new MDP(this.states, this.actions, this.transitions, this.rewards);
    }

    private void estimateModel(List<Trajectory> data){
        for(State s : this.states){
            for(Action a : this.actions){
                Map<State, Double> rewardMap = new HashMap<>();
                Map<State, Double> transitionMap = new HashMap<>();
                int observedSA = data.stream().mapToInt(t -> t.getStateActionCounter().get(s, a)).sum();
                if(observedSA > 0){
                    double rewardSum = data.stream().mapToDouble(t -> t.getStateActionReward().get(s, a)).sum();
                    for (State sprime : this.states) {
                        int observedSASP = data.stream().mapToInt(t -> t.getStateActionTransition().get(s, a).get(sprime)).sum();
                        transitionMap.put(sprime, ((double) observedSASP) / observedSA);
                        rewardMap.put(sprime, rewardSum / observedSA);
                    }
                }
                else {
                    for(State sprime : this.states){
                        rewardMap.put(sprime, 0.5);
                        transitionMap.put(sprime, 1.0 / this.states.size());
                    }
                }
                this.rewards.put(s, a, rewardMap);
                this.transitions.put(s, a, transitionMap);
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
