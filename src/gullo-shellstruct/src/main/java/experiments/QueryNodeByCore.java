/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package experiments;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;
import java.util.Set;

/**
 *
 * @author edotony
 */
public class QueryNodeByCore {

    private final ArrayList<ArrayList<Integer>> nodesOfCore;
    private final int maximumCore;
    private static final Random randomGenerator = new Random();

    public QueryNodeByCore(Set<Integer> connectedComponent, HashMap<Integer, Integer> cores) {
        this.nodesOfCore = new ArrayList<ArrayList<Integer>>();
        this.maximumCore = Collections.max(cores.values());

        for (int i = 0; i <= this.maximumCore; i++) {
            this.nodesOfCore.add(new ArrayList<Integer>());
        }
        
        for (int node : connectedComponent) {
            for (int i = cores.get(node); i >= 0; i--) {
                this.nodesOfCore.get(i).add(node);
            }
        }
    }

    public ArrayList<Integer> getQueryNodesOfCore(int core) {
        int node = this.nodesOfCore.get(core).get(randomGenerator.nextInt(this.nodesOfCore.get(core).size()));

        ArrayList<Integer> queryNode = new ArrayList<Integer>();
        queryNode.add(node);

        return queryNode;
    }
}
