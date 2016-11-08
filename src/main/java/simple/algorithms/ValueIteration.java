package simple.algorithms;

import java.util.*;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import simple.MDP.Action;
import simple.MDP.MDP;
import simple.MDP.State;

/**
 * This class implements Value Iteration in an simple.MDP and a Gamma.
 * 
 * @author Enrique Areyan Viqueira
 */
public class ValueIteration {

	/**
	 * Value iteration runs in an simple.MDP.
	 */
	private final MDP mdp;
	
	/**
	 * There is a value of gamma.
	 */
	private final double gamma;

	/**
	 * This map stores the V function.
	 */
	private Map<State, Double> V;

	/**
	 * Map storing the Q-function for easily computing the policy
	 */
	private Table<State, Action, Double> Q;

	/**
	 * Map storing the policy induced by the learned value function
	 */
	private Map<State, Action> P;
	
	/**
	 * Tolerance parameter.
	 */
	private static double tolerance = 0.0001;
	
	/**
	 * Max iteration parameter.
	 */
	private static int maxIter = 10000;

	/**
	 * Constructor. 
	 * 
	 * @param mdp
	 * @param gamma
	 */
	public ValueIteration(MDP mdp, double gamma) {
		this.mdp = mdp;
		this.gamma = gamma;
		this.V = new HashMap<>();
		this.Q = HashBasedTable.create();
		this.P = new HashMap<>();
		// Initialize the value function to 0.0
		for (State s : this.mdp.getStates()) {
			this.V.put(s, 0.0);
		}
	}

	/**
	 * Implements VI in this simple.MDP.
	 */
	public void run() {
		int i = 0;
		boolean convergence = false;
		// Convergence criteria are: (1) number of iterations and (2) difference of absolute values.
		while (i < maxIter && !convergence) {
			convergence = true;
			// For each state of the simple.MDP.
			for (State state : this.mdp.getStates()) {
				double maxSum = Double.NEGATIVE_INFINITY;
				// Compute the value of the action with the highest expected reward.
				for (Action action : this.mdp.getActions()) {
					double sum = 0.0;
					for(State sprime : this.mdp.getStates()) {
						sum += this.mdp.getTransition(state, sprime, action) * (this.mdp.getReward(state, sprime, action) + this.gamma * this.V.get(state));
					}
					this.Q.put(state, action, sum);
					if(sum > maxSum) {
						maxSum = sum;
					}
				}
				// Current Value
				double currentV = this.V.get(state);
				// Update the V value.
				this.V.put(state, maxSum);
				// If the difference between V values is greater than tolerance, then we have not converged.
				if(convergence && (Math.abs(currentV - this.V.get(state)) > ValueIteration.tolerance)){
					convergence = false;
				}
			}
			i++;
		}
		System.out.println("Number of iters = " + i);
	}

	/**
	 * Compute policy induced by the current value function
	 */
	public void computePolicy(){
		for(State s : this.mdp.getStates()){
			List<Action> maxActions = new ArrayList<>();
			double maxQ = Double.NEGATIVE_INFINITY;
			for(Action a : this.mdp.getActions()){
				double q = this.Q.get(s, a);
				if(q > maxQ){
					maxActions.clear();
					maxActions.add(a);
					maxQ = q;
				}
				if(q == maxQ){
					maxActions.add(a);
				}
			}
			int tie = new Random().nextInt(maxActions.size());
			this.P.put(s, maxActions.get(tie));
		}
	}

	@Override
	public String toString() {
		String ret = "\n V function:";
		for (State s : this.mdp.getStates()) {
			ret += "\n\t V(" + s.getId() + ") = " + this.V.get(s);
		}
		ret += "\n\n Q function:";
		for (State s : this.mdp.getStates()) {
			for (Action a : this.mdp.getActions()){
				ret += "\n\t Q(" + s.getId() + "," + a.getId() + ") = " + this.Q.get(s,a);
			}

		}

		ret += "\n\n Policy:";
		for (State s : this.mdp.getStates()) {
			ret += "\n\t Pi(" + s.getId() + ") = " + this.P.get(s);
		}
		return ret;
	}
}
