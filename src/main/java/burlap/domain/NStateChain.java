package burlap.domain;

import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.domain.singleagent.graphdefined.GraphDefinedDomain;
import burlap.domain.singleagent.graphdefined.GraphStateNode;
import burlap.mdp.auxiliary.DomainGenerator;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.statehashing.simple.SimpleHashableStateFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Created by dilip on 11/7/16.
 */
public class NStateChain implements DomainGenerator {

    public final GraphDefinedDomain domain;

    NStateChain(int numStates, int numActions, int numNextStates, double rewardNoise){
        this.domain = new GraphDefinedDomain(numStates);
        for(int i=0;i < numStates;i++){
            for(int j=0;j < numActions;j++) {
                Map<Integer, Double> transition = new HashMap<>();
                ThreadLocalRandom rnd = ThreadLocalRandom.current();
                while(transition.size() < numNextStates){
                    transition.put(rnd.nextInt(0, numStates), new Random().nextDouble());
                }
                double sum = transition.values().stream().mapToDouble(d -> d).sum();
                for(int k=0;k < numStates;k++){
                    if(!transition.keySet().contains(k)){
                        this.domain.setTransition(i, j, k, 0.0);
                    }
                    else{
                        this.domain.setTransition(i, j, k, transition.get(k) / sum);
                    }
                }
            }
        }

        this.domain.setRf(new NStateChainRF(numStates, numActions, rewardNoise));
    }

    @Override
    public SADomain generateDomain() {
        return this.domain.generateDomain();
    }

    public State getRandomStartState() {
        return new GraphStateNode(ThreadLocalRandom.current().nextInt(0, this.domain.getNumNodes()));
    }

    public GraphDefinedDomain getDomain(){
        return this.domain;
    }

    public static void main(String[] args) {
        NStateChain tenChain = new NStateChain(10, 2, 5, 0.1);
        SADomain domain = tenChain.generateDomain();

        //Pick initial state uniformly at random
        State start = tenChain.getRandomStartState();

        System.out.println(new SimpleHashableStateFactory().hashState(start));
        System.exit(0);

        //setup vi with 0.99 discount factor, a value
        //function initialization that initializes all states to value 0, and which will
        //run for 30 iterations over the state space
        Planner vi = new ValueIteration(domain, 0.99, new SimpleHashableStateFactory(),
                0.00001, 30);

        //run planning from our initial state
        Policy p = vi.planFromState(start);

        //evaluate the policy with one roll out visualize the trajectory
        Episode ea = PolicyUtils.rollout(p, start, domain.getModel());

    }
}

