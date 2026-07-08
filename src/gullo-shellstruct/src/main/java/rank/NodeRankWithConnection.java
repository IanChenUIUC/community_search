package rank;

import index.IndexInterface;

import java.util.HashMap;
import java.util.HashSet;

public class NodeRankWithConnection implements Comparable<NodeRankWithConnection>{
	
	private int node;
	
	private int solutionDegree;
	
	private int connections;
	private HashSet<Integer> connectedComponents;
	
	private int rank1;
	private int rank2;
	
	// Create the node given the core
	public NodeRankWithConnection(int node, IndexInterface index, int cocktailPartyMinimumIndex, HashMap<Integer, Integer> nodeCoefficient, HashMap<Integer, Integer> nodeGroup) {
		this.node = node;
		
		HashSet<Integer> neighbors = new HashSet<Integer>(index.getNeighbors(node, cocktailPartyMinimumIndex));
		neighbors.retainAll(nodeCoefficient.keySet());
		
		this.solutionDegree = neighbors.size();
		
		this.connectedComponents = new HashSet<Integer>();
		this.rank1 = 0;
		for (int neighbor : neighbors) {
			// Compute its connected components
			this.connectedComponents.add(nodeGroup.get(neighbor));
			
			// Compute rank 1
			if (nodeCoefficient.get(neighbor) > 0) {
				this.rank1++;
			}
		}
		this.connections = this.connectedComponents.size();
		
		this.rank2 = Math.max(0, index.getCoreMinimumDegree(cocktailPartyMinimumIndex) - this.solutionDegree);
	}
        
	// Create the node given an explicit list of neighbors
	public NodeRankWithConnection(int node, IndexInterface index, int cocktailPartyMinimumIndex, HashMap<Integer, Integer> nodeCoefficient, HashMap<Integer, Integer> nodeGroup, HashSet<Integer> explicitNeighbors) {
		this.node = node;
		
		HashSet<Integer> neighbors = new HashSet<Integer>(explicitNeighbors);
		neighbors.retainAll(nodeCoefficient.keySet());
		
		this.solutionDegree = neighbors.size();
		
		this.connectedComponents = new HashSet<Integer>();
		this.rank1 = 0;
		for (int neighbor : neighbors) {
			// Compute its connected components
			this.connectedComponents.add(nodeGroup.get(neighbor));
			
			// Compute rank 1
			if (nodeCoefficient.get(neighbor) > 0) {
				this.rank1++;
			}
		}
		this.connections = this.connectedComponents.size();
		
		this.rank2 = Math.max(0, index.getCoreMinimumDegree(cocktailPartyMinimumIndex) - this.solutionDegree);
	}
	
	// Create the node, which is a query node
	public NodeRankWithConnection(int node) {
		this.node = node;
		
		this.solutionDegree = 0;
		
		this.connectedComponents = new HashSet<Integer>();
		this.connections = Integer.MAX_VALUE;
		
		this.rank1 = Integer.MAX_VALUE;
		this.rank2 = 0;
	}
	
	// Get the node
	public int getNode() {
		return this.node;
	}
	
	// Get connected component IDs
	public HashSet<Integer> getConnectedComponentIDs() {
		return this.connectedComponents;
	}
	
	// Get the number of connections
	public int getConnections() {
		return this.connections;
	}
	
	// Get the rank
	public int getRank() {
		return this.rank1 - this.rank2;
	}
	
	// Get solutionDegree
	public int getSolutionDegree() {
		return this.solutionDegree;
	}
	
	public boolean hasGroups(HashSet<Integer> connectedComponentIDs) {
		for (int connectedComponentID : connectedComponentIDs) {
			if(this.connectedComponents.contains(connectedComponentID)) {
				return true;
			}
		}
		return false;
	}
	
	// Update the rank when one of its neighbors is saturated in the solution
	public void updateRankForSaturation() {
		this.rank1--;
	}
	
	// Update the rank when one of its neighbors is added to the solution
	public void updateRanksForAddition(int addedNode, int cocktailPartyMinimumDegree, HashMap<Integer, Integer> nodeCoefficient) {
		if (nodeCoefficient.get(addedNode) > 0) {
			this.rank1++;
		}
		
		this.solutionDegree++;
		this.rank2 = Math.max(0, cocktailPartyMinimumDegree - this.solutionDegree);
	}
	
	public void updateConnectionsForAddiction(int newConnectedComponentID) {
		this.connectedComponents.add(newConnectedComponentID);
		this.connections = this.connectedComponents.size();
	}
	
	// Update the connection when two connected components are merged by another node
	public void updateConnectionsForMerging(HashSet<Integer> deletedConnectedComponentIDs, int newConnectedComponentID) {
		this.connectedComponents.removeAll(deletedConnectedComponentIDs);
		this.connectedComponents.add(newConnectedComponentID);
		this.connections = this.connectedComponents.size();
	}

	@Override
	public int compareTo(NodeRankWithConnection o) {
		if (this.node == o.node) {
			return 0;
		} else if (this.getConnections() > o.getConnections()) {
			return -1;
		} else if (this.getConnections() < o.getConnections()) {
			return 1;
		} else if (this.getRank() > o.getRank()) {
			return -1;
		} else if (this.getRank() < o.getRank()) {
			return 1;
		} else if(this.node < o.node) {
			return 1;
		} else if(this.node > o.node) {
			return -1;
		} else {
			return 0;
		}	
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + this.node;
		return result;
	}
	
	@Override
	public boolean equals(Object obj) {
		if (this == obj) {
			return true;
		} else if (obj == null) {
			return false;
		} else if (getClass() != obj.getClass()) {
			return false;
		} 
		
		NodeRankWithConnection other = (NodeRankWithConnection) obj;
		if (this.node != other.node) {
			return false;
		}
		return true;
	}
	
	
}
