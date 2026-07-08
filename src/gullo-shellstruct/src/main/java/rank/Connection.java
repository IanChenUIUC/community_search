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

/**
 *
 * @author edotony
 */
public class Connection {

    public static HashSet<Integer> heuristicRankWithST(IndexInterface index, ArrayList<Integer> queryNodes, int minimumCoreIndex) {

        Graph heuristicGraph = SteinerTree.buildSteinerTree(index, minimumCoreIndex, queryNodes);
        HashMap<Integer, Integer> nodesScore = new HashMap<Integer, Integer>(); // The score of each node in the euristicGraph

        TreeSet<NodeRank> queue = new TreeSet<NodeRank>(); // The queue form which the best node is popped and added to the euristicGraph
        HashMap<Integer, NodeRank> queueNodes = new HashMap<Integer, NodeRank>();

        // Compute the score for each node in the strainer tree and add its neighbors to newQueueNodes
        HashSet<Integer> newQueueNodes = new HashSet<Integer>();
        Set<Integer> nodes = heuristicGraph.getNodes();
        for (int node : nodes) {
            nodesScore.put(node, computeScore(node, index.getCoreMinimumDegree(minimumCoreIndex), heuristicGraph));
            for (int neighbor : index.getNeighbors(node, minimumCoreIndex)) {
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
            NodeRank queueNode = new NodeRank(newNode, index, minimumCoreIndex, nodesScore);
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
            for (int neighbor : index.getNeighbors(nodeRank.getNode(), minimumCoreIndex)) {
                // If the neighbor exists in the heuristicGraph
                if (nodesScore.containsKey(neighbor)) {
                    // If its score is > 0
                    if (nodesScore.get(neighbor) > 0) {
                        // Update its score
                        int newScore = updateScore(nodesScore.get(neighbor));
                        nodesScore.put(neighbor, newScore);
                        // If the new score is zero, update the rank of its neighbors if they are in queue
                        if (newScore == 0) {
                            for (int neighborNeighbor : index.getNeighbors(neighbor, minimumCoreIndex)) {
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
                        NodeRank queueNode = new NodeRank(neighbor, index, minimumCoreIndex, nodesScore);
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
