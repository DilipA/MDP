package simple.MDP;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;

import java.util.*;

/**
 * Created by dilip on 11/7/16.
 */
public class Trajectory {
    /**
     * The list of states observed during the trajectory
     */
    private final List<State> states;

    /**
     * The list of actions taken during the trajectory
     */
    private final List<Action> actions;

    /**
     * The list of rewards observed during the trajectory
     */
    private final List<Double> rewards;

    /**
     * Counter for the number of times a given state-action pair has been seen during the trajectory
     */
    private final Table<State, Action, Integer> stateActionCounter;

    /**
     * Counter for the total reward observed for executing a given state-action pair during the trajectory
     */
    private final Table<State, Action, Double> stateActionReward;

    /**
     * Counter for the number of times a given state-action pair has transitioned to a particular next state
     * during the trajectory
     */
    private final Table<State, Action, Map<State, Integer>> stateActionTransition;

    /**
     * The length of this trajectory
     */
    protected int length;

    public Trajectory(Set<State> mdpStates, Set<Action> mdpActions){
        this.states = new ArrayList<>();
        this.actions = new ArrayList<>();
        this.rewards = new ArrayList<>();
        this.stateActionCounter = HashBasedTable.create();
        this.stateActionReward = HashBasedTable.create();
        this.stateActionTransition = HashBasedTable.create();
        for(State s : mdpStates){
            for(Action a : mdpActions){
                this.stateActionCounter.put(s, a, 0);
                this.stateActionReward.put(s, a, 0.0);
                this.stateActionTransition.put(s, a, new HashMap<>());
                for(State sprime : mdpStates){
                    this.stateActionTransition.get(s, a).put(sprime, 0);
                }
            }
        }
        this.length = 0;
    }

    public void intialize(State s){
        this.states.add(s);
    }

    public void step(Action a, double r, State sprime){
        //Update bookkeeping for state-action tracker
        State current = this.states.get(this.states.size()-1);
        int currentCount = this.stateActionCounter.get(current, a);
        this.stateActionCounter.put(current, a, currentCount + 1);

        //Update bookkeeping for reward tracker
        double currentReward = this.stateActionReward.get(current, a);
        this.stateActionReward.put(current, a, currentReward + r);

        //Update bookkeeping for state-action transition tracker
        Map<State, Integer> currentTransCount = this.stateActionTransition.get(current, a);
        currentTransCount.put(sprime, currentTransCount.get(sprime) + 1);
        this.stateActionTransition.put(current, a, currentTransCount);

        this.states.add(sprime);
        this.actions.add(a);
        this.rewards.add(r);
        this.length += 1;
    }


    public List<State> getStates() {
        return this.states;
    }

    public List<Action> getActions() {
        return this.actions;
    }

    public List<Double> getRewards() {
        return this.rewards;
    }

    public Table<State, Action, Integer> getStateActionCounter() {
        return this.stateActionCounter;
    }

    public Table<State, Action, Double> getStateActionReward() {
        return this.stateActionReward;
    }

    public Table<State, Action, Map<State, Integer>> getStateActionTransition() {
        return this.stateActionTransition;
    }

    public int getLength() {
        return this.length;
    }

    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder();
        sb.append("States:");
        for(State s : this.states){
            sb.append(s + ",");
        }
        sb.deleteCharAt(sb.length()-1);

        sb.append("\nActions:");
        for(Action a : this.actions){
            sb.append(a + ",");
        }
        sb.deleteCharAt(sb.length()-1);

        sb.append("\nRewards:");
        for(Double d : this.rewards){
            sb.append(d + ",");
        }
        sb.deleteCharAt(sb.length()-1);

        return sb.toString();
    }
}
