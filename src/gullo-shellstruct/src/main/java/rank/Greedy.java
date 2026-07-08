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
import java.util.TreeSet;

/**
 *
 * @author edotony
 */
public class Greedy {

    public static HashSet<Integer> heuristicRank(IndexInterface index, ArrayList<Integer> queryNodes, int minimumCoreIndex) {
        
        Map<Integer,HashSet<Integer>> neighborCache = new HashMap<Integer,HashSet<Integer>>();

        ArrayList<HashSet<Integer>> groupsOfNodes = new ArrayList<HashSet<Integer>>(); // Groups of nodes that represent the connected components in the euristicGraph
        HashMap<Integer, Integer> nodeGroup = new HashMap<Integer, Integer>(); // The group of each node in the euristicGraph

        HashMap<Integer, Integer> nodesScore = new HashMap<Integer, Integer>(); // The score of each node in the euristicGraph

        TreeSet<NodeRankWithConnection> queue = new TreeSet<NodeRankWithConnection>(); // The queue form which the best node is popped and added to the euristicGraph
        HashMap<Integer, NodeRankWithConnection> queueNodes = new HashMap<Integer, NodeRankWithConnection>();

        // Add every query node to the queue
        for (int queryNode : queryNodes) {
            NodeRankWithConnection queueNode = new NodeRankWithConnection(queryNode);
            queue.add(queueNode);
            queueNodes.put(queryNode, queueNode);
        }

        int numberOfGroups = 0;
        boolean hasOneGroupFlag = false;
        boolean hasAllScoresZeroFlag = false;

        while (!queue.isEmpty()) {

            // Poll the best node
            NodeRankWithConnection nodeRank = queue.pollFirst();
            //System.out.println(nodeRank.getNode());
            queueNodes.remove(nodeRank.getNode());

            // Compute its score
            nodesScore.put(nodeRank.getNode(), computeScore(nodeRank, index.getCoreMinimumDegree(minimumCoreIndex)));

            // The group in which all nodes are merged by the added node
            int newGroup = -1;

            // For all its neighbors
            if(!neighborCache.containsKey(nodeRank.getNode()))
            {
                neighborCache.put(nodeRank.getNode(), index.getNeighbors(nodeRank.getNode(), minimumCoreIndex));
            }
            for (int neighbor : neighborCache.get(nodeRank.getNode())) {
                // If the neighbor exists in the solution
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
                                neighborCache.put(neighbor, index.getNeighbors(neighbor, minimumCoreIndex));
                            }
                            for (int neighborNeighbor : neighborCache.get(neighbor)) {
                                if (queueNodes.containsKey(neighborNeighbor)) {
                                    queue.remove(queueNodes.get(neighborNeighbor));
                                    queueNodes.get(neighborNeighbor).updateRankForSaturation();
                                    queue.add(queueNodes.get(neighborNeighbor));
                                }
                            }
                        }
                    }

                    // If the query node doesn't have a group yet
                    if (!nodeGroup.containsKey(nodeRank.getNode()) && !hasOneGroupFlag) {
                        // Set its group the same as its neighbor
                        nodeGroup.put(nodeRank.getNode(), nodeGroup.get(neighbor));
                        groupsOfNodes.get(nodeGroup.get(nodeRank.getNode())).add(nodeRank.getNode());
                    } // Else, if it already has a group
                    else if (!hasOneGroupFlag && !nodeGroup.get(nodeRank.getNode()).equals(nodeGroup.get(neighbor))) {
                        // Merge the two groups (the smallest in the biggest)
                        numberOfGroups--;
                        if (groupsOfNodes.get(nodeGroup.get(neighbor)).size() > groupsOfNodes.get(nodeGroup.get(nodeRank.getNode())).size()) {
                            newGroup = nodeGroup.get(neighbor);
                            int oldGroup = nodeGroup.get(nodeRank.getNode());
                            for (int nodeWithNewGroup : groupsOfNodes.get(oldGroup)) {
                                nodeGroup.put(nodeWithNewGroup, nodeGroup.get(neighbor));
                            }
                            groupsOfNodes.get(nodeGroup.get(neighbor)).addAll(groupsOfNodes.get(oldGroup));
                        } else {
                            newGroup = nodeGroup.get(nodeRank.getNode());
                            int oldGroup = nodeGroup.get(neighbor);
                            for (int nodeWithNewGroup : groupsOfNodes.get(oldGroup)) {
                                nodeGroup.put(nodeWithNewGroup, nodeGroup.get(nodeRank.getNode()));
                            }
                            groupsOfNodes.get(nodeGroup.get(nodeRank.getNode())).addAll(groupsOfNodes.get(oldGroup));
                        }
                    }
                }
            }
            // If it has no neighbors
            if (!nodeGroup.containsKey(nodeRank.getNode()) && !hasOneGroupFlag) {
                // Create a new group
                numberOfGroups++;
                groupsOfNodes.add(new HashSet<Integer>());
                groupsOfNodes.get(groupsOfNodes.size() - 1).add(nodeRank.getNode());
                nodeGroup.put(nodeRank.getNode(), groupsOfNodes.size() - 1);
            }

            // Update the connections for each node in one of the merged groups
            if (newGroup != -1) {
                for (int queueNode : queueNodes.keySet()) {
                    if (queueNodes.get(queueNode).hasGroups(nodeRank.getConnectedComponentIDs())) {
                        queue.remove(queueNodes.get(queueNode));
                        queueNodes.get(queueNode).updateConnectionsForMerging(nodeRank.getConnectedComponentIDs(), newGroup);
                        queue.add(queueNodes.get(queueNode));
                    }
                }
            }

            // For all its neighbors
            if(!neighborCache.containsKey(nodeRank.getNode()))
            {
                neighborCache.put(nodeRank.getNode(), index.getNeighbors(nodeRank.getNode(), minimumCoreIndex));
            }
            for (int neighborOutSolution : neighborCache.get(nodeRank.getNode())) {
                // If the neighbor isn't in the solution
                if (!nodesScore.containsKey(neighborOutSolution) && !queryNodes.contains(neighborOutSolution)) {
                    // If it is in queue, update its rank
                    if (queueNodes.containsKey(neighborOutSolution)) {
                        queue.remove(queueNodes.get(neighborOutSolution));
                        queueNodes.get(neighborOutSolution).updateRanksForAddition(nodeRank.getNode(), index.getCoreMinimumDegree(minimumCoreIndex), nodesScore);
                        if (!hasOneGroupFlag) {
                            queueNodes.get(neighborOutSolution).updateConnectionsForAddiction(nodeGroup.get(nodeRank.getNode()));
                        }
                        queue.add(queueNodes.get(neighborOutSolution));
                        // Else, add it in queue
                    } else {
                        //NodeRankWithConnection queueNode = new NodeRankWithConnection(neighborOutSolution, index, minimumCoreIndex, nodesScore, nodeGroup);
                        if(!neighborCache.containsKey(neighborOutSolution))
                        {
                            neighborCache.put(neighborOutSolution, index.getNeighbors(neighborOutSolution, minimumCoreIndex));
                        }
                        NodeRankWithConnection queueNode = new NodeRankWithConnection(neighborOutSolution, index, minimumCoreIndex, nodesScore, nodeGroup, neighborCache.get(neighborOutSolution));
                        queue.add(queueNode);
                        queueNodes.put(neighborOutSolution, queueNode);
                    }
                }
            }

            if (numberOfGroups == 1 && !hasOneGroupFlag && nodesScore.size() >= queryNodes.size()) {
                hasOneGroupFlag = true;
            }
            if (!hasOneGroupFlag) {
                continue;
            }

            // Check that the scores are all zero
            for (Map.Entry<Integer, Integer> entry : nodesScore.entrySet()) {
                if (entry.getValue() == 0) {
                    hasAllScoresZeroFlag = true;
                } else {
                    hasAllScoresZeroFlag = false;
                    break;
                }
            }

            if (hasOneGroupFlag && hasAllScoresZeroFlag) {
                break;
            }
        }

        return new HashSet<Integer>(nodesScore.keySet());
    }

    private static int computeScore(NodeRankWithConnection nodeRank, int cocktailPartyMinimumDegree) {
        return Math.max(0, cocktailPartyMinimumDegree - nodeRank.getSolutionDegree());
    }
    
    private static int updateScore(int score) {
        return Math.max(0, score - 1);
    }

}
