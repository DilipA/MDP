package simple.MDP;

import java.util.ArrayList;
import java.util.List;

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
     * The length of this trajectory
     */
    protected int length;

    public Trajectory(){
        this.states = new ArrayList<>();
        this.actions = new ArrayList<>();
        this.rewards = new ArrayList<>();
        this.length = 0;
    }

    public void intialize(State s){
        this.states.add(s);
    }

    public void step(Action a, double r, State sprime){
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
