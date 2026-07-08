package main;

import index.IndexInterface;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.UnsafeMemoryOutput;

public class Graph {
	
	// Variables
	protected HashMap<Integer, HashSet<Integer>> adj;
	
	protected HashMap<Integer, Integer> degrees;
	protected ArrayList<HashSet<Integer>> orderedNodes;
	
	protected int minimumDegree;
	
	// Creator from file
	public Graph(String path) {
		this.readFromFile(path);
		
		this.computeDegrees();
		this.computeMinimumDegree();
	}
	
	// Creator for an empty graph
	public Graph(int numberOfNodes) {
		this.adj = new HashMap<Integer, HashSet<Integer>>();
		
		this.degrees = new HashMap<Integer, Integer>();
		this.orderedNodes = new ArrayList<HashSet<Integer>>();
		for (int i = 0; i < numberOfNodes; i++) {
			this.orderedNodes.add(new HashSet<Integer>());
		}
		
		this.minimumDegree = 0;
	}
	
	// Creator for a connected component subgraph
	public Graph(Graph graph, HashSet<Integer> connectedComponent) {
		this.adj = new HashMap<Integer, HashSet<Integer>>();
		
		for (int node : connectedComponent) {
			// Add it to the graph
			this.addNode(node);
			// For all its neighbors
			for (int neighbor : graph.getNeighbors(node)) {
				// If the neighbor exists in the reconstructed graph
				if (this.doesNodeExist(neighbor)) {
					// Add the edges between them
					this.addEdge(node, neighbor);
					this.addEdge(neighbor, node);
				}
			}
		}
		
		this.computeDegrees();
		this.computeMinimumDegree();
	}
	
	// Creator for a connected component subgraph
	public Graph(IndexInterface index, HashSet<Integer> connectedComponent) {
		this.adj = new HashMap<Integer, HashSet<Integer>>();
		
		for (int node : connectedComponent) {
			// Add it to the graph
			this.addNode(node);
			// For all its neighbors
			for (int neighbor : index.getNeighbors(node, 0)) {
				// If the neighbor exists in the reconstructed graph
				if (this.doesNodeExist(neighbor)) {
					// Add the edges between them
					this.addEdge(node, neighbor);
					this.addEdge(neighbor, node);
				}
			}
		}
		
		this.computeDegrees();
		this.computeMinimumDegree();
	}
	
	// *** Build a graph *** \\
	// Read graph from file
	protected void readFromFile(String fileName) {
		System.out.println("Reading graph...");
		this.adj = new HashMap<Integer, HashSet<Integer>>();
		String line;
		String[] parts;

		try {
			FileReader fileReader = new FileReader(fileName);
			BufferedReader bufferedReader = new BufferedReader(fileReader);

			bufferedReader.readLine();
			while ((line = bufferedReader.readLine()) != null) {
				parts = line.split(",");
				int from = Integer.parseInt(parts[0].replace(" ", ""));
				int to = Integer.parseInt(parts[1].replace(" ", ""));

				this.addNode(from);
				this.addNode(to);
				this.addEdge(from, to);
				this.addEdge(to, from);
			}
			
			bufferedReader.close();
			fileReader.close();

		} catch (FileNotFoundException e) {
			System.out.println("File not found.");
			System.exit(0);
		} catch (IOException e) {
			System.out.println("Error in reading file.");
			System.exit(0);
		}

		System.out.println("Graph read.");
	}

	
	// Add a directed edge
	public void addEdge(int from, int to) {
		if (from != to && !this.adj.get(from).contains(to))
			this.adj.get(from).add(to);
	}

	// Add a node without edges
	public void addNode(int node) {
		if (!this.doesNodeExist(node)) {
			this.adj.put(node, new HashSet<Integer>());
		}
	}

	// *** Degree *** \\
	// Compute degrees for each node
	public HashMap<Integer, Integer> computeDegrees() {
		this.degrees = new HashMap<Integer, Integer>();
		this.orderedNodes = new ArrayList<HashSet<Integer>>();
		for (int i = 0; i < this.adj.size(); i++) {
			this.orderedNodes.add(new HashSet<Integer>());
		}
		
		for (Map.Entry<Integer, HashSet<Integer>> entry : this.adj.entrySet()) {
			this.degrees.put(entry.getKey(), entry.getValue().size());
			this.orderedNodes.get(entry.getValue().size()).add(entry.getKey());
		}

		return this.degrees;
	}
	
	// Compute the minimum degree
	public int computeMinimumDegree() {
		for (int i = 0; i < this.orderedNodes.size(); i++) {
			if (this.orderedNodes.get(i).size() > 0) {
				this.minimumDegree = i;
				return i;
			}
		}
		
		return this.minimumDegree;
	}
	
	// Compute the minimum degree after a node has been added
	public int computeMinimumDegree(int node) {
		this.updateDegrees(node);
		
		if (this.degrees.get(node) < this.minimumDegree) {
			this.minimumDegree = this.getDegree(node);
			return this.minimumDegree;
		} else if (this.degrees.get(node) == this.minimumDegree) {
			return this.minimumDegree;
		} else {
			for (int i = this.minimumDegree; i < this.orderedNodes.size(); i++) {
				if (this.orderedNodes.get(i).size() > 0) {
					this.minimumDegree = i;
					return i;
				}
			}
		}
		
		return this.minimumDegree;
	}
	
	// Compute degrees after a node has been added
	private void updateDegrees(int node) {
		int degree = this.adj.get(node).size();
		this.degrees.put(node, degree);
		this.orderedNodes.get(degree).add(node);
		
		for (int neighbor : this.adj.get(node)) {
			int neighborDegree = this.degrees.get(neighbor);
			this.orderedNodes.get(neighborDegree).remove(neighbor);
			this.degrees.put(neighbor, neighborDegree + 1);
			this.orderedNodes.get(neighborDegree + 1).add(neighbor);
		}
	}

	// *** Getters *** \\
	// Return the edges
	public HashSet<Integer[]> getEdges() {
		HashSet<Integer[]> edges = new HashSet<Integer[]>();
		HashSet<Integer> printedNodes = new HashSet<Integer>();
		for (Map.Entry<Integer, HashSet<Integer>> entry : this.adj.entrySet()) {
			for (int neighbor : entry.getValue()) {
				if (!printedNodes.contains(neighbor)) {
					Integer[] edge = new Integer[2];
					edge[0] = entry.getKey();
					edge[1] = neighbor;
					edges.add(edge);
				}
			}
			printedNodes.add(entry.getKey());
		}

		return edges;
	}
	
	// Return true if the node exists, else false
	public Boolean doesNodeExist(int node) {
		if (!this.adj.containsKey(node)) {
			return false;
		} else {
			return true;
		}
	}
	
	public HashMap<Integer, Integer> getDegrees() {
		return this.degrees;
	}
	
	public int getDegree(int node) {
		return this.adj.get(node).size();
	}

	public int getMinimumDegree() {
		return this.minimumDegree;
	}

	public HashSet<Integer> getNeighbors(int node) {
		return this.adj.get(node);
	}

	public int getNumberOfNodes() {
		return this.adj.size();
	}
	
	public int getNumberOfEdges() {
		int numberOfEdges = 0;
		for (Map.Entry<Integer, HashSet<Integer>> entry : this.adj.entrySet()) {
			numberOfEdges += entry.getValue().size();
		}
		
		return numberOfEdges / 2;
	}
	
	public int getNumberOfEdgesOfConnectedComponent(HashSet<Integer> connectedComponent) {
		int numberOfEdges = 0;
		for (Map.Entry<Integer, HashSet<Integer>> entry : this.adj.entrySet()) {
			if (connectedComponent.contains(entry.getKey())) {
				numberOfEdges += entry.getValue().size();
			}	
		}
		
		return numberOfEdges / 2;
	}

	public Set<Integer> getNodes() {
		return this.adj.keySet();
	}
	
	// *** Serialize *** \\
	public void serializeToFile(String datasetName) {
		try {
			Kryo kryo = new Kryo();
			FileOutputStream fileOutputStream = new FileOutputStream(datasetName + "Dataset.bin");
			UnsafeMemoryOutput output = new UnsafeMemoryOutput(fileOutputStream);
			kryo.writeObject(output, this.adj);
			
			fileOutputStream.close();
		} catch (FileNotFoundException e) {
			System.out.println("Error in writing file.");
			System.exit(0);
		} catch (IOException e) {
			System.out.println("Error in writing file.");
			System.exit(0);
		}
	}
	
	
}
