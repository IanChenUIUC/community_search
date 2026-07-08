package csm;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import main.Graph;

public class LIHeuristic {

	private HashSet<Integer>[] groupsOfNodes;
	private HashMap<Integer, Integer> nodeGroup;
	
	private int bestIndex;
	
	private Graph graph;
	
	@SuppressWarnings("unchecked")
	public LIHeuristic(Graph graph, int queryNode, HashSet<Integer> neighbors) {
		this.groupsOfNodes = (HashSet<Integer>[]) new HashSet[graph.getNumberOfNodes()];
		for (int i = 0; i < graph.getNumberOfNodes(); i++) {
			this.groupsOfNodes[i] = new HashSet<Integer>();
		}
		this.nodeGroup = new HashMap<Integer, Integer>();
		
		this.bestIndex = 1;
		
		this.graph = graph;
		
		for (int neighbor : neighbors) {
			this.groupsOfNodes[this.bestIndex].add(neighbor);
			this.nodeGroup.put(neighbor, this.bestIndex);
		}
	}
	
	@SuppressWarnings("unchecked")
	public LIHeuristic(Graph graph, int queryNode) {
		this.groupsOfNodes = (HashSet<Integer>[]) new HashSet[graph.getNumberOfNodes()];
		for (int i = 0; i < graph.getNumberOfNodes(); i++) {
			this.groupsOfNodes[i] = new HashSet<Integer>();
		}
		this.nodeGroup = new HashMap<Integer, Integer>();
		
		this.bestIndex = 1;
		
		this.graph = graph;
		
		this.groupsOfNodes[this.bestIndex].add(queryNode);
		this.nodeGroup.put(queryNode, this.bestIndex);
	}
	
	public void addNode(int node, Graph inLinkGraph) {
		HashSet<Integer> neighbors = this.graph.getNeighbors(node);
		
		int numberOfNeighbors = 0;
		Set<Integer> nodes = inLinkGraph.getNodes();
		if (nodes.size() < neighbors.size()) {
			for (int neighbor : nodes) {
				if (neighbors.contains(neighbor)) {
					numberOfNeighbors++;
				}
			}
		} else {
			for (int neighbor : neighbors) {
				if (nodes.contains(neighbor)) {
					numberOfNeighbors++;
				}
			}
		}
		
		this.groupsOfNodes[numberOfNeighbors].add(node);
		this.nodeGroup.put(node, numberOfNeighbors);
		
		if (numberOfNeighbors > this.bestIndex) {
			this.bestIndex = numberOfNeighbors;
		}
		
	}
	
	public int getBestNode() {
		if (this.bestIndex == 0) {
			return -1;
		} else {
			int node = this.groupsOfNodes[this.bestIndex].iterator().next();
			this.groupsOfNodes[this.bestIndex].remove(node);
			this.nodeGroup.remove(node);
			
			if (this.groupsOfNodes[this.bestIndex].isEmpty()) {
				int oldBestIndex = this.bestIndex;
				this.bestIndex = 0;
				for (int i = oldBestIndex - 1; i >= 0 ; i--) {
					if (!this.groupsOfNodes[i].isEmpty()) {
						this.bestIndex = i;
						break;
					}
				}
			}
			
			// Updates the other nodes in the queue that are its neighbors
			HashSet<Integer> neighbors = this.graph.getNeighbors(node);
			for (int neighbor :neighbors) {
				if (this.nodeGroup.containsKey(neighbor)) {
					this.groupsOfNodes[this.nodeGroup.get(neighbor)].remove(neighbor);
					int newGroup = this.nodeGroup.get(neighbor) + 1;
					this.nodeGroup.put(neighbor, newGroup);
					
					this.groupsOfNodes[this.nodeGroup.get(neighbor)].add(neighbor);
					
					if (this.nodeGroup.get(neighbor) > this.bestIndex) {
						this.bestIndex = this.nodeGroup.get(neighbor);
					}
				}
			}
			
			return node;
		}
	}
	
}
