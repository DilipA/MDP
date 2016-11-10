package simple.algorithms;

import com.google.common.collect.Lists;
import simple.MDP.Action;
import simple.MDP.MDP;
import simple.MDP.State;
import simple.MDP.Trajectory;
import simple.MDP.exceptions.MDPException;
import simple.experiment.data.DataGenerator;
import simple.experiment.model_based.MDPEstimator;
import simple.sample.RandomMDP;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Created by dilip on 11/7/16.
 */
public class PolicyEvaluation {

    /**
     * The policy to be evaluated
     */
    private final Map<State, Action> policy;

    /**
     * This map stores the V function.
     */
    private Map<State, Double> V;

    /**
     * The MDP on which the policy should be evaluated
     */
    private final MDP mdp;

    /**
     * The value of gamma that should be used for evaluation
     */
    private final double gamma;

    /**
     * The value of epsilon to use in the epsilon-greedy policy
     */
    private final double epsilon;

    /**
     * Tolerance parameter.
     */
    private static double tolerance = 0.0001;

    /**
     * Max iteration parameter.
     */
    private static int maxIter = 10000;

    /**
     * Random generator
     */
    private final Random rndg = new Random();

    public PolicyEvaluation(MDP mdp, double gamma, Map<State, Action> policy){
        this.mdp = mdp;
        this.gamma = gamma;
        this.policy = policy;
        this.epsilon = 0.0;
        this.V = new HashMap<>();
        // Initialize the value function to 0.0
        for (State s : this.mdp.getStates()) {
            this.V.put(s, 0.0);
        }
    }

    public PolicyEvaluation(MDP mdp, double gamma, double epsilon, Map<State, Action> policy){
        this.mdp = mdp;
        this.gamma = gamma;
        this.policy = policy;
        this.epsilon = epsilon;
        this.V = new HashMap<>();
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
                // Compute the value of the action with the highest expected reward.
                Action action;
                if(this.rndg.nextDouble() < this.epsilon){
                    action = Lists.newArrayList(this.mdp.getActions()).get(this.rndg.nextInt(this.mdp.getActions().size()));
                }
                else{
                    action = this.policy.get(state);
                }

                double sum = 0.0;
                for(State sprime : this.mdp.getStates()) {
                    sum += this.mdp.getTransition(state, sprime, action) * (this.mdp.getReward(state, sprime, action) + this.gamma * this.V.get(sprime));
                }

                // Current Value
                double currentV = this.V.get(state);
                // Update the V value.
                this.V.put(state, sum);
                // If the difference between V values is greater than tolerance, then we have not converged.
                if(convergence && (Math.abs(currentV - this.V.get(state)) > this.tolerance)){
                    convergence = false;
                }
            }
            i++;
        }
        //System.out.println("Number of iters = " + i);
    }

    public Map<State, Double> getValueFunction(){
        return this.V;
    }

    public static void main(String[] args) throws MDPException {
        int numTrajectories = 50;
        MDP randomMDP = RandomMDP.sample();

        List<Trajectory> trajectories = DataGenerator.generateNTrajectories(numTrajectories, 5, randomMDP);

        MDPEstimator estimator = new MDPEstimator(randomMDP.getStates(), randomMDP.getActions(), trajectories);
        MDP estimatedMDP = estimator.getMdp();

        double gamma = 1.0;
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
        for(State s : randomMDP.getStates()){
            sumDiff += v1.get(s) - v2.get(s);
        }
        System.out.println("Empirical Loss = " + sumDiff / randomMDP.getStates().size());
    }

}
