package rank;

import index.IndexInterface;

import java.util.HashMap;
import java.util.HashSet;

public class NodeRank implements Comparable<NodeRank>{
	
	private int node;
	
	private int solutionDegree;
	
	private int rank1;
	private int rank2;
	
	// Create the node (given a core)
	public NodeRank(int node, IndexInterface index, int cocktailPartyMinimumIndex, HashMap<Integer, Integer> nodeCoefficient) {
		this.node = node;
		
		HashSet<Integer> neighbors = new HashSet<Integer>(index.getNeighbors(node, cocktailPartyMinimumIndex));
		neighbors.retainAll(nodeCoefficient.keySet());
		
		this.solutionDegree = neighbors.size();
		
		this.rank1 = 0;
		for (int neighbor : neighbors) {
			if (nodeCoefficient.get(neighbor) > 0) {
				this.rank1++;
			}
		}
		
		this.rank2 = Math.max(0, index.getCoreMinimumDegree(cocktailPartyMinimumIndex) - this.solutionDegree);
	}
        
	// Create the node (given an explicit list of neighbors)
	public NodeRank(int node, IndexInterface index, int cocktailPartyMinimumIndex, HashMap<Integer, Integer> nodeCoefficient, HashSet<Integer> explicitNeighbors) {
		this.node = node;
		
                HashSet<Integer> neighbors = new HashSet<Integer>(explicitNeighbors);
		neighbors.retainAll(nodeCoefficient.keySet());
		
		this.solutionDegree = neighbors.size();
		
		this.rank1 = 0;
		for (int neighbor : neighbors) {
			if (nodeCoefficient.get(neighbor) > 0) {
				this.rank1++;
			}
		}
		
		this.rank2 = Math.max(0, index.getCoreMinimumDegree(cocktailPartyMinimumIndex) - this.solutionDegree);
	}
	
	// Create the node, which is a query node
	public NodeRank(int node) {
		this.node = node;
		
		this.solutionDegree = 0;
		
		this.rank1 = Integer.MAX_VALUE;
		this.rank2 = 0;
	}
	
	// Get the node
	public int getNode() {
		return this.node;
	}
	
	// Get the rank
	public int getRank() {
		return this.rank1 - this.rank2;
	}
	
	// Get solutionDegree
	public int getSolutionDegree() {
		return this.solutionDegree;
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

	@Override
	public int compareTo(NodeRank o) {
		if (this.node == o.node) {
			return 0;
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
		
		NodeRank other = (NodeRank) obj;
		if (this.node != other.node) {
			return false;
		}
		return true;
	}
	
	
}
