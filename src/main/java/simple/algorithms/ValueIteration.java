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
	 * The value of beta to use for the Boltzmann operator in GVI
	 */
	private final double beta;

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
	public ValueIteration(MDP mdp, double gamma, double beta) {
		this.mdp = mdp;
		this.gamma = gamma;
		this.beta = beta;
		this.V = new HashMap<>();
		this.Q = HashBasedTable.create();
		this.P = new HashMap<>();
		// Initialize the value function to 0.0
		for (State s : this.mdp.getStates()) {
			this.V.put(s, 0.0);
		}
		for(State s : this.mdp.getStates()){
			for(Action a : this.mdp.getActions()){
				this.Q.put(s, a, 0.0);
			}
		}
	}

	public ValueIteration(MDP mdp, double gamma, double beta, double qInit) {
		this.mdp = mdp;
		this.gamma = gamma;
		this.beta = beta;
		this.V = new HashMap<>();
		this.Q = HashBasedTable.create();
		this.P = new HashMap<>();
		// Initialize the value function to 0.0
		for (State s : this.mdp.getStates()) {
			this.V.put(s, 0.0);
		}
		for(State s : this.mdp.getStates()){
			for(Action a : this.mdp.getActions()){
				this.Q.put(s, a, qInit);
			}
		}
	}

	public ValueIteration(MDP mdp, double gamma){
		this.mdp = mdp;
		this.gamma = gamma;
		this.beta = 0.0;
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
						sum += this.mdp.getTransition(state, sprime, action) * (this.mdp.getReward(state, sprime, action) + this.gamma * this.V.get(sprime));
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
//		System.out.println("Number of iters = " + i);
	}

	public Table<State, Action, Double> cloneTable(Table<State, Action, Double> toClone){
		Table<State, Action, Double> ret = HashBasedTable.create();
		for(State s : this.mdp.getStates()){
			for(Action a : this.mdp.getActions()){
				ret.put(s, a, toClone.get(s, a));
			}
		}
		return ret;
	}

	public double boltzmannPolicy(State s, Action a, Table<State, Action, Double> qVals){
		double maxQ = qVals.row(s).values().stream().max(Double::compareTo).get();
		double numerator = Math.exp(this.beta * (qVals.get(s, a) - maxQ));

		double lnorm = Math.log(this.mdp.getActions().stream().mapToDouble(aprime -> Math.exp(this.beta * (qVals.get(s, aprime) - maxQ))).sum());

		double ret = Math.exp(Math.log(numerator) - lnorm);
		return ret;
	}

	public void updateTransitionDynamics(Table<State, Action, Double> qVals){
		for(State s : this.mdp.getStates()){
			for(Action a : this.mdp.getActions()){
				Map<State, Double> newDistro = this.mdp.getDynamics().get(s, a);
				for(State sprime : this.mdp.getStates()){
					double boltzProb = this.boltzmannPolicy(s, a, qVals);
					newDistro.put(sprime, newDistro.get(sprime) * boltzProb);
				}

				double norm = newDistro.values().stream().mapToDouble(i -> i).sum();
				if(norm != 1.0){
					for(State sprime : this.mdp.getStates()){
						newDistro.put(sprime, newDistro.get(sprime) / norm);
					}
				}
				this.mdp.getDynamics().put(s, a, newDistro);
			}
		}
	}

	public void runQ(){
		int i = 0;
		boolean convergence = false;
		// Convergence criteria are: (1) number of iterations and (2) difference of absolute values.
		while (i < maxIter && !convergence) {
			convergence = true;
			// For each state of the simple.MDP.
			this.updateTransitionDynamics(this.Q);
			for (State state : this.mdp.getStates()) {
				double maxDiff = Double.NEGATIVE_INFINITY;
				// Compute the value of the action with the highest expected reward.
				for (Action action : this.mdp.getActions()) {
					double oldQ = this.Q.get(state, action);
					Table<State, Action, Double> oldQs = cloneTable(this.Q);
					this.Q.put(state, action, 0.0);
					for(State sprime : this.mdp.getStates()) {
						this.Q.put(state, action, this.Q.get(state, action)
//								+ (this.mdp.getTransition(state, sprime, action) * (this.mdp.getReward(state, sprime, action) + this.gamma * this.boltzmann(sprime, oldQs))));
								+ (this.mdp.getTransition(state, sprime, action) * (this.mdp.getReward(state, sprime, action) + this.gamma * oldQs.row(sprime).values().stream().max(Double::compareTo).get())));

					}
					if(Math.abs(this.Q.get(state, action) - oldQ) > maxDiff) {
						maxDiff = this.Q.get(state, action);
					}
				}
				// If the difference between V values is greater than tolerance, then we have not converged.
				if(convergence && maxDiff > ValueIteration.tolerance){
					convergence = false;
				}
			}
			i++;
		}
//		System.out.println("Finished in " + i + " iterations");
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

//			System.err.println("Q-values: " + this.Q.toString());
			int tie = new Random().nextInt(maxActions.size());
			this.P.put(s, maxActions.get(tie));
		}
	}

	public Map<State, Action> getPolicy(){
	    return this.P;
    }

    public Table<State, Action, Double> getQ(){
		return this.Q;
	}

    public Map<State, Map<Action, Double>> getStochasticPolicy(){
		Map<State, Map<Action, Double>> ret = new HashMap<>();
		for(State s : this.mdp.getStates()){
			ret.put(s, new HashMap<>());
			double maxQ = this.Q.row(s).values().stream().max(Double::compareTo).get();
			for(Action a : this.mdp.getActions()){
				ret.get(s).put(a, Math.exp(this.beta * (this.Q.get(s, a) - maxQ)));
			}
		}
//		System.err.println(ret);

		for(State s : this.mdp.getStates()){
			double norm = Math.log(ret.get(s).values().stream().mapToDouble(i -> i).sum());
			for(Action a : this.mdp.getActions()){
				ret.get(s).put(a, Math.exp(Math.log(ret.get(s).get(a)) - norm));
			}
		}
//		System.err.println(ret);

		return ret;
	}

	public double boltzmann(State s, Table<State, Action, Double> qVals){
    	List<Double> boltzed = new ArrayList<>();
		double maxQ = qVals.row(s).values().stream().max(Double::compareTo).get();

		for(Action a : this.mdp.getActions()){
    		boltzed.add((qVals.get(s, a) - maxQ) * Math.exp(this.beta * (qVals.get(s, a) - maxQ)));
		}

		double lnorm = Math.log(this.mdp.getActions().stream().mapToDouble(a -> Math.exp(this.beta * (qVals.get(s, a) - maxQ))).sum());

		double ret = Math.exp(Math.log(boltzed.stream().mapToDouble(i -> i).sum() + maxQ*Math.exp(lnorm)) - lnorm);

//		System.out.println("Boltzmann operator output: " + ret);
		return ret;
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
