package steinerTree;

import index.IndexInterface;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;

public class SteinerTreeLimited {

    public static HashSet<Integer> buildSteinerTree(IndexInterface index, int minimumCoreIndex, ArrayList<Integer> queryNodes, int maximumDepth) {

        if (queryNodes.size() == 1) {
            HashSet<Integer> mstNodes = new HashSet<Integer>();
            mstNodes.add(queryNodes.get(0));
            return mstNodes;
        }

        HashMap<Integer, BFPathsLimited> paths = new HashMap<Integer, BFPathsLimited>(); // HashMap with the paths from each query node to the others

        // For each query node (less one) compute the shortest paths with the other query nodes
        HashSet<Integer> leftQueryNodes = new HashSet<Integer>(queryNodes);
        for (int node : queryNodes) {
            if (paths.size() == queryNodes.size() - 1) {
                continue;
            } else {
                paths.put(node, new BFPathsLimited(index, minimumCoreIndex, node, leftQueryNodes, maximumDepth));
            }
            leftQueryNodes.remove(node);
        }

        // Create the weighted graph
        WeightedGraph weightedGraph = new WeightedGraph();
        // Add the query nodes
        for (int node : queryNodes) {
            weightedGraph.addNode(node);
        }
        // Add edges between query nodes
        for (Entry<Integer, BFPathsLimited> entry : paths.entrySet()) {
            for (int node : queryNodes) {
                if (paths.get(entry.getKey()).hasPathTo(node)) {
                    weightedGraph.addEdge(entry.getKey(), node, paths.get(entry.getKey()).distTo(node));
                    weightedGraph.addEdge(node, entry.getKey(), paths.get(entry.getKey()).distTo(node));
                }
            }
        }

        // Computes the minimum spanning tree
        PrimMST mst = new PrimMST(weightedGraph);

        // Builds the graph on which apply heuristics
        HashSet<Integer> mstNodes = new HashSet<Integer>();
        // Add nodes
        Iterable<Integer> nodes = null;
        for (Edge edge : mst.edges()) {
            if (paths.containsKey(edge.getFrom()) && paths.get(edge.getFrom()).hasPathTo(edge.getTo())) {
                nodes = paths.get(edge.getFrom()).pathTo(edge.getTo());
            } else if (paths.containsKey(edge.getTo()) && paths.get(edge.getTo()).hasPathTo(edge.getFrom())) {
                nodes = paths.get(edge.getTo()).pathTo(edge.getFrom());
            }

            if (nodes != null) {
                for (int node : nodes) {
                    mstNodes.add(node);
                }
            }
        }
        // Add query nodes
        for (Integer queryNode : queryNodes) {
            mstNodes.add(queryNode);
        }

        return mstNodes;
    }
}