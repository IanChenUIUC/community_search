package steinerTree;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class WeightedGraph {
	
	private HashMap<Integer, HashSet<Edge>> adj;
	
	// Initializes an empty edge-weighted graph 
	public WeightedGraph() {
		this.adj = new HashMap<Integer, HashSet<Edge>>();
	}
	
	// Add a node without edges
	public void addNode(int node) {
		if (!this.doesNodeExist(node)) {
			this.adj.put(node, new HashSet<Edge>());
		}
	}
	
	// Add a directed edge
	public void addEdge(int from, int to, int weight) {
		Edge edge = new Edge(from, to, weight);
		if (from != to && !this.adj.get(from).contains(edge)) {
			this.adj.get(from).add(edge);
		}
	}
	
	// Return true if the node exists, else false
	public Boolean doesNodeExist(int node) {
		if (!this.adj.containsKey(node)) {
			return false;
		} else {
			return true;
		}
	}
	
	// Gets the incident edges of the specified node
	public HashSet<Edge> getIncidentEdges(int node) {
		return this.adj.get(node);
	}
	
	public int getNumberOfNodes() {
		return this.adj.size();
	}
	
	public int getNumberOfEdges() {
		int numberOfEdges = 0;
		for (Map.Entry<Integer, HashSet<Edge>> entry : this.adj.entrySet()) {
			numberOfEdges += entry.getValue().size();
		}
		
		return numberOfEdges / 2;
	}
	
	public Set<Integer> getNodes() {
		return this.adj.keySet();
	}
	
}
