/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rank;

import index.IndexInterface;
import java.util.ArrayList;
import java.util.HashSet;
import steinerTree.SteinerTreeLimited;

/**
 *
 * @author edotony
 */
public class ConnectionLimited {

    public static HashSet<Integer> heuristicRankWithST(IndexInterface index, ArrayList<Integer> queryNodes, int minimumCoreIndex, int maximumDepth) {

        HashSet<Integer> heuristicNodes = SteinerTreeLimited.buildSteinerTree(index, minimumCoreIndex, queryNodes, maximumDepth);
        
        return Greedy.heuristicRank(index, new ArrayList<Integer>(heuristicNodes), minimumCoreIndex);
    }

}
