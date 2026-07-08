package steinerTree;

import index.IndexInterface;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;

import main.Graph;

public class SteinerTree {
	
	public static Graph buildSteinerTree(IndexInterface index, int minimumCoreIndex, ArrayList<Integer> queryNodes) {
		
		if (queryNodes.size() == 1) {
			Graph mstGraph = new Graph(index.getNumberOfNodes(minimumCoreIndex));
			mstGraph.addNode(queryNodes.get(0));
			return mstGraph;
		}
		
		HashMap<Integer, BFPaths> paths = new HashMap<Integer, BFPaths>(); // HashMap with the paths from each query node to the others
		
		// For each query node (less one) compute the shortest paths with the other query nodes
		HashSet<Integer> leftQueryNodes = new HashSet<Integer>(queryNodes);
		for (int node : queryNodes) {
			if (paths.size() == queryNodes.size() - 1) {
				continue;
			} else {
				paths.put(node, new BFPaths(index, minimumCoreIndex, node, leftQueryNodes));
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
		for (Entry<Integer, BFPaths> entry : paths.entrySet()) {
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
		Graph mstGraph = new Graph(index.getNumberOfNodes(minimumCoreIndex));
		// Add nodes
		Iterable<Integer> nodes;
		for (Edge edge : mst.edges()) {
			if (paths.containsKey(edge.getFrom()) && paths.get(edge.getFrom()).hasPathTo(edge.getTo())) {
				nodes = paths.get(edge.getFrom()).pathTo(edge.getTo());
			} else {
				nodes = paths.get(edge.getTo()).pathTo(edge.getFrom());
			}
			
			for (int node : nodes) {
				if (!mstGraph.doesNodeExist(node)) {
						mstGraph.addNode(node);
						for (int neighbor : index.getNeighbors(node, minimumCoreIndex)) {
							if (mstGraph.doesNodeExist(neighbor)) {
								mstGraph.addEdge(node, neighbor);
								mstGraph.addEdge(neighbor, node);
							}
						}
					}
			}
		}
		
		return mstGraph;
	}
        
	public static Graph buildSteinerTree(IndexInterface index, int minimumCoreIndex, ArrayList<Integer> queryNodes, HashSet<Integer> subcore) {
		
		if (queryNodes.size() == 1) {
			Graph mstGraph = new Graph(subcore.size());
			mstGraph.addNode(queryNodes.get(0));
			return mstGraph;
		}
		
		HashMap<Integer, BFPaths> paths = new HashMap<Integer, BFPaths>(); // HashMap with the paths from each query node to the others
		
		// For each query node (less one) compute the shortest paths with the other query nodes
		HashSet<Integer> leftQueryNodes = new HashSet<Integer>(queryNodes);
		for (int node : queryNodes) {
			if (paths.size() == queryNodes.size() - 1) {
				continue;
			} else {
				paths.put(node, new BFPaths(index, minimumCoreIndex, node, leftQueryNodes, subcore));
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
		for (Entry<Integer, BFPaths> entry : paths.entrySet()) {
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
		//Graph mstGraph = new Graph(index.getNumberOfNodes(minimumCoreIndex));
                Graph mstGraph = new Graph(subcore.size());
		// Add nodes
		Iterable<Integer> nodes;
		for (Edge edge : mst.edges()) {
			if (paths.containsKey(edge.getFrom()) && paths.get(edge.getFrom()).hasPathTo(edge.getTo())) {
				nodes = paths.get(edge.getFrom()).pathTo(edge.getTo());
			} else {
				nodes = paths.get(edge.getTo()).pathTo(edge.getFrom());
			}
			
			for (int node : nodes) {
				if (!mstGraph.doesNodeExist(node)) {
						mstGraph.addNode(node);
						for (int neighbor : index.getNeighbors(node, minimumCoreIndex,subcore)) {
							if (mstGraph.doesNodeExist(neighbor)) {
								mstGraph.addEdge(node, neighbor);
								mstGraph.addEdge(neighbor, node);
							}
						}
					}
			}
		}
		
		return mstGraph;
	}
}
