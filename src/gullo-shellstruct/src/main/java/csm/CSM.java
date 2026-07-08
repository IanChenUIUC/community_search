package csm;

import java.util.HashSet;

import coreGroups.CoreGroups;
import main.Graph;

public class CSM {
	
	// Complete algorithm
	public static HashSet<Integer> CSMAlgorithm(GraphWithOrderedAdj graph, int numberOfEdges, int queryNode, int parameter) {
		
		// *** Step 1 *** \\
		HashSet<Integer> H = new HashSet<Integer>(); // The best solution found so far
		double maximumMinimumDegree = 0; // The maximum minimum degree found so far
		
		Graph A = new Graph(graph.getNumberOfNodes()); // Graph with the vertices we have visited
		A.addNode(queryNode);  // queryNode has been visited yet
		A.computeMinimumDegree(queryNode);
		
		HashSet<Integer> differenceAH = new HashSet<Integer>(); // The difference between A and H and number of vertices added to H
		differenceAH.add(queryNode);
		
		int s = 0; // Number of vertices added to H
		
		// Vertices we need to visit
		LIHeuristic heuristic = new LIHeuristic(graph, queryNode, graph.getNeighbors(queryNode));
		
		double threshold = 0; // Threshold for s value
		double thresholdForMin = (1.0 + Math.sqrt(9 + 8 * (numberOfEdges - graph.getNumberOfNodes()))) / 2;
		double thresholdForBest = Math.min(graph.getDegree(queryNode), (long) thresholdForMin);
		
		int node = heuristic.getBestNode();
		
		while (node != -1) {

			// Check s with the threshold only if the parameter is different then -2147483648
			if (parameter != Integer.MIN_VALUE && s > threshold) {
				break;
			}
			
			// Add the node and its edges to the graph A
			A.addNode(node);
			differenceAH.add(node);
			for (int neighbor : graph.getNeighbors(node)) {
				if (A.doesNodeExist(neighbor)) {
					A.addEdge(node, neighbor);
					A.addEdge(neighbor, node);
				}
			}
			s++;
			
			if (A.computeMinimumDegree(node) > maximumMinimumDegree) {
				// Update the best solution
				H.addAll(differenceAH);
				differenceAH = new HashSet<Integer>();
				s = 0;
				maximumMinimumDegree = A.getMinimumDegree();
				
				// If it's the best solution return it
				if (maximumMinimumDegree == thresholdForBest) {
					return H;
				}
			}
			
			for (int neighbor : graph.getNeighbors(node)) {
				if (graph.getDegree(neighbor) > maximumMinimumDegree && !A.doesNodeExist(neighbor)) {
					heuristic.addNode(neighbor, A);
				}
			}
			
			// Compute the threshold
			long expCoefficient = (long) ((double) ((numberOfEdges - graph.getNumberOfNodes())) / ((maximumMinimumDegree + 1) / 2 - 1));
			threshold = Math.exp(-parameter) * ((expCoefficient - H.size()));
			
			// Select the best node
			node = heuristic.getBestNode();
		}
		
		// *** Step 3 *** \\
		HashSet<Integer> result = CoreGroups.coreOfNode(CoreGroups.coreGroupsAlgorithm(A), queryNode);
		
		return result;
	}
	
	public static HashSet<Integer> CSMNaiveAlgorithm(GraphWithOrderedAdj graph, int queryNode, int parameter, HashSet<Integer> connectedComponent, int numberOfEdgesOfConnectedComponent) {
		
		// *** Step 1 *** \\
		HashSet<Integer> H = new HashSet<Integer>(); // The best solution found so far
		double maximumMinimumDegree = 0; // The maximum minimum degree found so far
		
		Graph A = new Graph(graph.getNumberOfNodes()); // Graph with the vertices we have visited
		A.addNode(queryNode);  // queryNode has been visited yet
		A.computeMinimumDegree(queryNode);
		
		HashSet<Integer> differenceAH = new HashSet<Integer>(); // The difference between A and H and number of vertices added to H
		differenceAH.add(queryNode);
		
		int s = 0; // Number of vertices added to H
		
		// Vertices we need to visit
		LIHeuristic heuristic = new LIHeuristic(graph, queryNode, graph.getNeighbors(queryNode));
		
		double threshold = 0; // Threshold for s value
		double thresholdForMin = (1.0 + Math.sqrt(9 + 8 * (numberOfEdgesOfConnectedComponent - connectedComponent.size()))) / 2;
		double thresholdForBest = Math.min(graph.getDegree(queryNode), (long) thresholdForMin);
		
		int node = heuristic.getBestNode();
		
		while (node != -1) {

			// Check s with the threshold only if the parameter is different then -2147483648
			if (parameter != Integer.MIN_VALUE && s > threshold) {
				break;
			}
			
			// Add the node and its edges to the graph A
			A.addNode(node);
			differenceAH.add(node);
			for (int neighbor : graph.getNeighbors(node)) {
				if (A.doesNodeExist(neighbor)) {
					A.addEdge(node, neighbor);
					A.addEdge(neighbor, node);
				}
			}
			s++;
			
			if (A.computeMinimumDegree(node) > maximumMinimumDegree) {
				// Update the best solution
				H.addAll(differenceAH);
				differenceAH = new HashSet<Integer>();
				s = 0;
				maximumMinimumDegree = A.getMinimumDegree();
				
				// If it's the best solution return it
				if (maximumMinimumDegree == thresholdForBest) {
					return H;
				}
			}
			
			for (int neighbor : graph.getNeighbors(node)) {
				if (graph.getDegree(neighbor) > maximumMinimumDegree && !A.doesNodeExist(neighbor)) {
					heuristic.addNode(neighbor, A);
				}
			}
			
			// Compute the threshold
			long expCoefficient = (long) ((double) ((numberOfEdgesOfConnectedComponent - connectedComponent.size())) / ((maximumMinimumDegree + 1) / 2 - 1));
			threshold = Math.exp(-parameter) * ((expCoefficient - H.size()));
			
			// Select the best node
			node = heuristic.getBestNode();
		}
		
		//System.out.println(maximumMinimumDegree);
		// *** Step 2 *** \\ Executed only if we want to run naive() on the complete graph
		A = naive(graph, queryNode, (int) maximumMinimumDegree);
		//A = naive(new GraphWithOrderedAdj(A), queryNode, (int) maximumMinimumDegree);
		
		// *** Step 3 *** \\
		HashSet<Integer> result = CoreGroups.coreOfNode(CoreGroups.coreGroupsAlgorithm(A), queryNode);
		
		return result;
	}
	
	private static Graph naive(GraphWithOrderedAdj graph, int queryNode, int k) {
		
		Graph C = new Graph(graph.getNumberOfNodes()); // The solution graph
		LIHeuristic euristic = new LIHeuristic(graph, queryNode);
		
		int node = euristic.getBestNode();
		
		while (node != -1) {
			
			C.addNode(node);
			for (int neighbor : graph.getNeighbors(node)) {
				if (C.doesNodeExist(neighbor)) {
					C.addEdge(node, neighbor);
					C.addEdge(neighbor, node);
				}
			}
			
			C.computeMinimumDegree(node);
			// DELETED CONDITION ************ ************
			if (C.getMinimumDegree() >= k) {
				return C;
			}
			
			for (int neighbor : graph.getOrderedNeighbors(node)) {
				if (graph.getDegree(neighbor) >= k && !C.doesNodeExist(neighbor)) {
					euristic.addNode(neighbor, C);
				} else if (graph.getDegrees().get(neighbor) < k) {
					break;
				}
			}
			
			node = euristic.getBestNode();
		}
		
		return C;
	}

}
