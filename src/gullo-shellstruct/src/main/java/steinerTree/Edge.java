package steinerTree;

public class Edge implements Comparable<Edge>{
	
	private int from;
	private int to;
	
	private int weight;

	public Edge(int from, int to, int weight) {
		this.from = from;
		this.to = to;
		
		this.weight = weight;
	}
	
	public int getFrom() {
		return this.from;
	}
	
	public int getTo() {
		return this.to;
	}
	
	public int otherNode(int node) {
		if (this.from == node) {
			return this.to;
		} else if (this.to == node) {
			return this.from;
		}
		return -1;
	}
	
	public int getWeight() {
		return this.weight;
	}
	
	@Override
	public int compareTo(Edge o) {
		if (this.weight < o.weight) {
			return -1;
		} else if (this.weight > o.weight) {
			return 1;
		}
		return 0;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + from;
		result = prime * result + to;
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj) {
			return true;
		}
		if (obj == null) {
			return false;
		}
		if (getClass() != obj.getClass()) {
			return false;
		}
		Edge other = (Edge) obj;
		if (from != other.from) {
			return false;
		}
		if (to != other.to) {
			return false;
		}
		return true;
	}

}
