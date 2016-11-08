//package burlap.domain;
//
//import burlap.domain.singleagent.graphdefined.GraphRF;
//import com.google.common.collect.HashBasedTable;
//import com.google.common.collect.Table;
//
//import java.util.Map;
//import java.util.Random;
//
///**
// * Created by dilip on 11/7/16.
// */
//public class NStateChainRF extends GraphRF {
//
//    public Table<Integer, Integer, Double> trueRewards;
//    public double noise;
//
//    public NStateChainRF(int numStates, int numActions, double noise){
//        this.trueRewards = HashBasedTable.create();
//        for(int i=0;i < numStates;i++){
//            for(int j=0;j < numActions;j++){
//                trueRewards.put(i, j, new Random().nextDouble());
//            }
//        }
//        this.noise = noise;
//    }
//
//    @Override
//    public double reward(int s, int a, int sprime) {
//        return this.trueRewards.get(s, a) + new Random().nextGaussian()*this.noise;
//    }
//}
