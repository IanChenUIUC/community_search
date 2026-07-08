package coreGroups;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

import main.Graph;

public class CoreGroups {

	public static HashMap<Integer, Integer> coreGroupsAlgorithm(Graph graph) {
		
		// ** Variables initialization ** \\
		HashMap<Integer, Integer> cores = new HashMap<Integer, Integer>(); // The core of each node
		
		ArrayList<HashSet<Integer>> orderedNodes = new ArrayList<HashSet<Integer>>(graph.getNumberOfNodes());
		for (int i = 0; i < graph.getNumberOfNodes(); i++) {
			orderedNodes.add(new HashSet<Integer>());
		}
		
		// Nodes added to groups
		HashMap<Integer, Integer> degrees = new HashMap<Integer, Integer>(graph.getDegrees());
		for (Map.Entry<Integer, Integer> entry : degrees.entrySet()) {
			orderedNodes.get(entry.getValue()).add(entry.getKey());
		}

		// ** Algorithm ** \\
		int node; // Current node
		int lowestDegree = 0;
		int neighborDegree;
		
		while (lowestDegree < graph.getNumberOfNodes()) {
			// If there aren't nodes with that degree, increase it
			if (orderedNodes.get(lowestDegree).size() == 0) {
				lowestDegree++;
			} else {
				// Else: take one node with the lowest degree, remove it from orderedNodes, set its core and set its degree to -1
				node = orderedNodes.get(lowestDegree).iterator().next();
				orderedNodes.get(lowestDegree).remove(node);
				cores.put(node, lowestDegree);
				degrees.put(node, -1);

				// Decreases the degree of its neighbors and change their sets
				for (int neighbor : graph.getNeighbors(node)) {
					neighborDegree = degrees.get(neighbor);
					if (neighborDegree > lowestDegree) {
						orderedNodes.get(neighborDegree).remove(neighbor);
						orderedNodes.get(neighborDegree - 1).add(neighbor);

						degrees.put(neighbor, neighborDegree - 1);
					}
				}
			}
		}
		
		return cores;
	}
	
	public static HashSet<Integer> coreOfNode(HashMap<Integer, Integer> cores, int node) {
		HashSet<Integer> core = new HashSet<Integer>();
		for (int i : cores.keySet()) {
			if (cores.get(i) >= cores.get(node)) {
				core.add(i);
			}
		}
		
		return core;
	}
	
	public static HashSet<Integer> coreOfIndex(HashMap<Integer, Integer> cores, int index) {
		HashSet<Integer> core = new HashSet<Integer>();
		for (int i : cores.keySet()) {
			if (cores.get(i) >= index) {
				core.add(i);
			}
		}
		
		return core;
	}
	
	public static Graph graphOfCore(Graph graph, int node, HashMap<Integer, Integer> cores) {
		HashSet<Integer> core = coreOfNode(cores, node);
		
		Graph coreGraph = new Graph(graph, core);
		
		return coreGraph;
	}
	
	public static Graph graphOfCore(Graph graph, ArrayList<Integer> queryNodes, HashMap<Integer, Integer> cores) {
		// Select the nodes with the lowest core
		HashSet<Integer> coreIndexes = new HashSet<Integer>();
		for (int node : queryNodes) {
			coreIndexes.add(cores.get(node));
		}
		
		HashSet<Integer> core = coreOfIndex(cores, Collections.min(coreIndexes));
		
		Graph coreGraph = new Graph(graph, core);
		
		return coreGraph;
	}
	
}
