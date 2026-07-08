/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rank;

import index.IndexInterface;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import main.Graph;
import steinerTree.SteinerTree;
import steinerTree.SteinerTree_Mehlhorn;

/**
 *
 * @author edotony
 */
public class GreedyConnection {

    public static HashSet<Integer> heuristicRankWithST(IndexInterface index, ArrayList<Integer> queryNodes, int minimumCoreIndex) {
        
        Map<Integer,HashSet<Integer>> neighborCache = new HashMap<Integer,HashSet<Integer>>();
        

        HashSet<Integer> subcore = Greedy.heuristicRank(index, queryNodes, index.getMinimumCoreIndex(queryNodes));
        
        //Graph heuristicGraph = SteinerTree.buildSteinerTree(index, minimumCoreIndex, queryNodes, subcore);
        Graph heuristicGraph = SteinerTree_Mehlhorn.buildSteinerTree(index, minimumCoreIndex, queryNodes, subcore);
        HashMap<Integer, Integer> nodesScore = new HashMap<Integer, Integer>(); // The score of each node in the euristicGraph

        TreeSet<NodeRank> queue = new TreeSet<NodeRank>(); // The queue form which the best node is popped and added to the euristicGraph
        HashMap<Integer, NodeRank> queueNodes = new HashMap<Integer, NodeRank>();

        // Compute the score for each node in the steiner tree and add its neighbors to newQueueNodes
        HashSet<Integer> newQueueNodes = new HashSet<Integer>();
        Set<Integer> nodes = heuristicGraph.getNodes();
        for (int node : nodes) {
            nodesScore.put(node, computeScore(node, index.getCoreMinimumDegree(minimumCoreIndex), heuristicGraph));
            if(!neighborCache.containsKey(node))
            {
                neighborCache.put(node, index.getNeighbors(node, minimumCoreIndex,subcore));
            }
            for (int neighbor : neighborCache.get(node)) {
                if (!heuristicGraph.doesNodeExist(neighbor)) {
                    newQueueNodes.add(neighbor);
                }
            }
        }

        // Check if the Steiner Tree is a possible solution
        boolean hasAllScoresZeroFlag = false;
        for (Map.Entry<Integer, Integer> entry : nodesScore.entrySet()) {
            if (entry.getValue() == 0) {
                hasAllScoresZeroFlag = true;
            } else {
                hasAllScoresZeroFlag = false;
                break;
            }
        }

        // Add the newQueueNodes to the queue
        for (int newNode : newQueueNodes) {
            //NodeRank queueNode = new NodeRank(newNode, index, minimumCoreIndex, nodesScore);
            if(!neighborCache.containsKey(newNode))
            {
                neighborCache.put(newNode, index.getNeighbors(newNode, minimumCoreIndex,subcore));
            }
            NodeRank queueNode = new NodeRank(newNode, index, minimumCoreIndex, nodesScore, neighborCache.get(newNode));
            queue.add(queueNode);
            queueNodes.put(newNode, queueNode);
        }

        while (!queue.isEmpty()) {

            // Poll the best node
            NodeRank nodeRank = queue.pollFirst();
            //System.out.println(nodeRank.getNode());
            queueNodes.remove(nodeRank.getNode());

            // Compute its score
            nodesScore.put(nodeRank.getNode(), computeScore(nodeRank, index.getCoreMinimumDegree(minimumCoreIndex)));

            // For all its neighbors
            if(!neighborCache.containsKey(nodeRank.getNode()))
            {
                neighborCache.put(nodeRank.getNode(), index.getNeighbors(nodeRank.getNode(), minimumCoreIndex, subcore));
            }
            HashSet<Integer> neighbors = neighborCache.get(nodeRank.getNode());
            //for (int neighbor : index.getNeighbors(nodeRank.getNode(), minimumCoreIndex)) {
            for (int neighbor : neighbors) {
                // If the neighbor exists in the heuristicGraph
                if (nodesScore.containsKey(neighbor)) {
                    // If its score is > 0
                    if (nodesScore.get(neighbor) > 0) {
                        // Update its score
                        int newScore = updateScore(nodesScore.get(neighbor));
                        nodesScore.put(neighbor, newScore);
                        // If the new score is zero, update the rank of its neighbors if they are in queue
                        if (newScore == 0) {
                            if(!neighborCache.containsKey(neighbor))
                            {
                                neighborCache.put(neighbor, index.getNeighbors(neighbor, minimumCoreIndex, subcore));
                            }
                            HashSet<Integer> neighborsNeighbors = neighborCache.get(neighbor);
                            
                            //for (int neighborNeighbor : index.getNeighbors(neighbor, minimumCoreIndex)) {
                            for (int neighborNeighbor : neighborsNeighbors) {
                                if (queueNodes.containsKey(neighborNeighbor)) {
                                    queue.remove(queueNodes.get(neighborNeighbor));
                                    queueNodes.get(neighborNeighbor).updateRankForSaturation();
                                    queue.add(queueNodes.get(neighborNeighbor));
                                }
                            }
                        }
                    }
                    // If the neighbor isn't in the heuristicGraph
                } else {
                    // If it is in queue, update its rank
                    if (queueNodes.containsKey(neighbor)) {
                        queue.remove(queueNodes.get(neighbor));
                        queueNodes.get(neighbor).updateRanksForAddition(nodeRank.getNode(), index.getCoreMinimumDegree(minimumCoreIndex), nodesScore);
                        queue.add(queueNodes.get(neighbor));
                        // Else, add it in queue
                    } else {
                        if(!neighborCache.containsKey(neighbor))
                        {
                            neighborCache.put(neighbor, index.getNeighbors(neighbor, minimumCoreIndex, subcore));
                        }
                        HashSet<Integer> neighborsNeighbors = neighborCache.get(neighbor);
                        NodeRank queueNode = new NodeRank(neighbor, index, minimumCoreIndex, nodesScore, neighborsNeighbors);
                        queue.add(queueNode);
                        queueNodes.put(neighbor, queueNode);
                    }
                }
            }

            // Check that the score of the euristicGraph are all zero
            for (Map.Entry<Integer, Integer> entry : nodesScore.entrySet()) {
                if (entry.getValue() == 0) {
                    hasAllScoresZeroFlag = true;
                } else {
                    hasAllScoresZeroFlag = false;
                    break;
                }
            }

            if (hasAllScoresZeroFlag) {
                break;
            }
        }

        return new HashSet<Integer>(nodesScore.keySet());
    }

    // Used on the Steiner Tree nodes
    private static int computeScore(int node, int cocktailPartyMinimumDegree, Graph heuristicGraph) {
        return Math.max(0, cocktailPartyMinimumDegree - heuristicGraph.getDegree(node));
    }

    // Used on the added nodes from the queue
    private static int computeScore(NodeRank nodeRank, int cocktailPartyMinimumDegree) {
        return Math.max(0, cocktailPartyMinimumDegree - nodeRank.getSolutionDegree());
    }
    
    // Used on the nodes already in the solution
    private static int updateScore(int score) {
        return Math.max(0, score - 1);
    }

}
