package utilities;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Queue;

import main.Graph;

public class DFS {
	
	// DFS of the specified graph starting from a given node
	public static HashSet<Integer> DFSCall(Graph graph, int node) {
		HashSet<Integer> visitedNodes = new HashSet<Integer>();
		DFSRoutine(graph, node, visitedNodes);

		if (visitedNodes.size() == graph.getNumberOfNodes()) {
			System.out.println("The graph is connected.");
		} else {
			System.out.println("The graph is not connected.");
		}

		return visitedNodes;
	}

	// DFS routine
	private static void DFSRoutine(Graph graph, int node, HashSet<Integer> visitedNodes) {
		visitedNodes.add(node);
		for (int neighbor : graph.getNeighbors(node)) {
			if (!visitedNodes.contains(neighbor)) {
				DFSRoutine(graph, neighbor, visitedNodes);
			}
		}
	}

	// Return the nodes that form the largest connected component
	public static HashSet<Integer> largestConnectedComponent(Graph graph) {
		HashSet<HashSet<Integer>> connectedComponents = connectedComponents(graph);
		HashSet<Integer> largestConnectedComponent = new HashSet<Integer>();

		for (HashSet<Integer> connectedComponent : connectedComponents) {
			if (connectedComponent.size() > largestConnectedComponent.size()) {
				largestConnectedComponent = connectedComponent;
			}
		}

		if (largestConnectedComponent.size() == graph.getNumberOfNodes()) {
			System.out.println("\nThe graph is connected.");
		} else {
			System.out.println("\nThe graph is not connected.");
			System.out.println("The graph has " + connectedComponents.size() + " connected components.");
			System.out.println("The largest connected component has " + largestConnectedComponent.size() + " nodes and " + graph.getNumberOfEdgesOfConnectedComponent(largestConnectedComponent) + " edges.");
		}

		return largestConnectedComponent;

	}

	// Return the connected components of the given graph
	private static HashSet<HashSet<Integer>> connectedComponents(Graph graph) {
		HashSet<HashSet<Integer>> connectedComponents = new HashSet<HashSet<Integer>>();
		HashSet<Integer> notVisitedNodes = new HashSet<Integer>();
		notVisitedNodes.addAll(graph.getNodes());

		while (!notVisitedNodes.isEmpty()) {
			HashSet<Integer> visitedNodes = new HashSet<Integer>();
			//DFSRoutine(graph, notVisitedNodes.iterator().next(), visitedNodes, notVisitedNodes);
                        DFSRoutine_iterative(graph, notVisitedNodes.iterator().next(), visitedNodes, notVisitedNodes);

			HashSet<Integer> connectedComponent = new HashSet<Integer>(visitedNodes);

			connectedComponents.add(connectedComponent);
		}

		return connectedComponents;
	}

	// DFS routine for connectedComponents
	private static void DFSRoutine(Graph graph, int node, HashSet<Integer> visitedNodes, HashSet<Integer> notVisitedNodes) {
		notVisitedNodes.remove(node);
		visitedNodes.add(node);
		for (int neighbor : graph.getNeighbors(node)) {
			if (!visitedNodes.contains(neighbor)) {
				DFSRoutine(graph, neighbor, visitedNodes, notVisitedNodes);
			}
		}
	}
        
	private static void DFSRoutine_iterative(Graph graph, int node, HashSet<Integer> visitedNodes, HashSet<Integer> notVisitedNodes) {
                Queue<Integer> q = new LinkedList<Integer>();
                q.add(node);
                notVisitedNodes.remove(node);
                visitedNodes.add(node);
                while(!q.isEmpty())
                {
                    int x = q.poll();
                    for(int y:graph.getNeighbors(x))
                    {
                        if(!visitedNodes.contains(y))
                        {
                            q.add(y);
                            visitedNodes.add(y);
                            notVisitedNodes.remove(y);
                        }
                    }
                }
	}

	// Sort an HashMap<Integer, Integer> by its values
	@SuppressWarnings("unchecked")
	public static ArrayList<Integer> sortByValue(final HashMap<Integer, Integer> hashMap) {
		ArrayList<Integer> keys = new ArrayList<Integer>();
		keys.addAll(hashMap.keySet());
		Collections.sort(keys, new Comparator<Integer>() {
			@Override
			public int compare(Integer o1, Integer o2) {
				Object v1 = hashMap.get(o1);
				Object v2 = hashMap.get(o2);
				if (v1 == null) {
					return (v1 == null) ? 0 : 1;
				} else if (v1 instanceof Comparable) {
					return ((Comparable<Integer>) v2).compareTo((Integer) v1);
				} else {
					return 0;
				}
			}
		});
		return keys;
	}

}
