package simple.MDP;

import java.util.*;
import java.util.stream.Collectors;

import com.google.common.collect.Lists;
import com.google.common.collect.Table;

import org.apache.commons.math3.distribution.EnumeratedDistribution;
import org.apache.commons.math3.util.Pair;
import simple.MDP.exceptions.MDPException;

/**
 * This class represents an simple.MDP.
 * 
 * @author Enrique Areyan Viqueira
 */
public class MDP {

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
	private final Table<State, Action, Map<State, Double>> transition;

	/**
	 * Transition from one state to another yields a reward.
	 */
	private final Table<State, Action, Map<State, Double>> reward;

	/**
	 * Amount of noise to add to true reward signal
	 */
	private final double rewardNoise;

	/**
	 * Random generator
	 */
	private final Random rndg = new Random();

	/**
	 * Constructor. 
	 * 
	 * @param states
	 * @param actions
	 * @param transition
	 * @param reward
	 * @throws MDPException
	 */
	public MDP(Set<State> states, Set<Action> actions, 
			Table<State, Action, Map<State, Double>> transition,
			Table<State, Action, Map<State, Double>> reward) throws MDPException {
		this.states = states;
		this.actions = actions;
		this.transition = transition;
		// Check that we get a valid probability distribution.
		for(State state : this.states){
			for(Action action : this.actions){
				double p = 0.0;
				for(Double nextStateProb : this.transition.get(state, action).values()){
					p += nextStateProb;
					if(nextStateProb < 0 ){
						throw new MDPException("The transition probability from state " + state + " is not well-defined, a transition is negative. ");
					}
				}
				// We have a tolerance to the probability adding to 1.
				if(Math.abs(p - 1.0) > 0.0001) {
					throw new MDPException("The transition probability from state " + state + " is not well-defined, it add to " + p);
				}
			}
		}
		this.reward = reward;
		this.rewardNoise = 0.0;
	}
	
	/**
	 * Getter.
	 * 
	 * @param sFrom
	 * @param sTo
	 * @param a
	 * @return the transition probability of landing in sTo starting from sFrom and taking action a.
	 */
	public double getTransition(State sFrom, State sTo, Action a) {
		return this.transition.get(sFrom, a).get(sTo);
	}

	/**
	 * Getter.
	 * 
	 * @param sFrom
	 * @param sTo
	 * @param a
	 * @return the reward obtained from being in state sFrom, taking action a and landing in sTo.
	 */
	public double getReward(State sFrom, State sTo, Action a) {
		return this.reward.get(sFrom, a).get(sTo) + this.rndg.nextGaussian()*this.rewardNoise;
	}
	
	/**
	 * Getter.
	 * 
	 * @return the set of states.
	 */
	public Set<State> getStates() {
		return this.states;
	}
	
	/**
	 * Getter.
	 * 
	 * @return the set of actions.
	 */
	public Set<Action> getActions() {
		return this.actions;
	}

	/**
	 * Produce a state uniformly at random from all the states of this MDP
	 * @return A state sampled uniformly at random from the state space
	 */
	public State getRandomState(){
		List<State> allStates = Lists.newArrayList(this.states);
		return allStates.get(new Random().nextInt(allStates.size()));
	}

	/**
	 * Produce an action uniformly at random from all the actions of this MDP
	 * @return An action sampled uniformly at random from the action space
	 */
	public Action getRandomAction(){
		List<Action> allActions = Lists.newArrayList(this.actions);
		return allActions.get(new Random().nextInt(allActions.size()));
	}

	public State sampleTransition(State s, Action a){
		List<Pair<State, Double>> pmf = this.transition.get(s, a).entrySet().stream()
				.map(e -> new Pair<>(e.getKey(), e.getValue())).collect(Collectors.toList());
		EnumeratedDistribution<State> transition = new EnumeratedDistribution<>(pmf);
		return transition.sample();
	}

	public State sampleRandomTransition(State s, Action a){
		long numReachable = this.transition.get(s, a).entrySet().stream().filter(e -> e.getValue() > 0.0).count();
		List<Pair<State, Double>> pmf = this.transition.get(s, a).entrySet().stream()
				.filter(e -> e.getValue() > 0.0)
				.map(e -> new Pair<>(e.getKey(), 1.0 / numReachable)).collect(Collectors.toList());
		EnumeratedDistribution<State> transition = new EnumeratedDistribution<>(pmf);
		return transition.sample();
	}

	@Override
	public String toString() {
		String ret = "\nStates: \n";
		for (State state : this.states) {
			ret += "\t" + state + "\n";
		}
		ret += "Actions:\n";
		for (Action action : this.actions) {
			ret += "\t" + action + "\n";
		}

		ret += "\nTransitions:";
		for (State state : this.states) {
			ret += "\nFrom state " + state;
			for (Action action : this.actions) {
				ret += "\n\t taking action " + action;
				for (Map.Entry<State, Double> entry : this.transition.get(state, action).entrySet()) {
					ret += "\n\t\t to " + entry.getKey() + ",  " + entry.getValue();
				}
			}
		}

		ret += "\nRewards:";
		for (State state : this.states) {
			ret += "\nFrom state " + state;
			for (Action action : this.actions) {
				ret += "\n\t taking action " + action;
				for (Map.Entry<State, Double> entry : this.reward.get(state, action).entrySet()) {
					ret += "\n\t\t to " + entry.getKey() + ",  " + entry.getValue();
				}
			}
		}
		return ret;
	}

}
