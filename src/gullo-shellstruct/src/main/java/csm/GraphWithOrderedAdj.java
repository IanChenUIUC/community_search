package csm;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

import utilities.DFS;
import main.Graph;

public class GraphWithOrderedAdj extends Graph{
	
	private HashMap<Integer, ArrayList<Integer>> orderedAdj;
	
	public GraphWithOrderedAdj(String path) {
		super(path);
		
		this.orderedAdj = new HashMap<Integer, ArrayList<Integer>>();
		
		for (Map.Entry<Integer, HashSet<Integer>> entry : this.adj.entrySet()) {
			HashMap<Integer, Integer> neighborsDegree = new HashMap<Integer, Integer>();
			for (int neighbor : entry.getValue()) {
				neighborsDegree.put(neighbor, this.getDegree(neighbor));
			}
			this.orderedAdj.put(entry.getKey(), DFS.sortByValue(neighborsDegree));
		}
	}
	
	public GraphWithOrderedAdj(Graph graph) {
		super(graph, new HashSet<Integer>(graph.getNodes()));
		
		this.orderedAdj = new HashMap<Integer, ArrayList<Integer>>();
		for (Map.Entry<Integer, HashSet<Integer>> entry : this.adj.entrySet()) {
			HashMap<Integer, Integer> neighborsDegree = new HashMap<Integer, Integer>();
			for (int neighbor : entry.getValue()) {
				neighborsDegree.put(neighbor, this.getDegree(neighbor));
			}
			this.orderedAdj.put(entry.getKey(), DFS.sortByValue(neighborsDegree));
		}
		
	}
	
	public ArrayList<Integer> getOrderedNeighbors(int node) {
		return this.orderedAdj.get(node);
	}

}
