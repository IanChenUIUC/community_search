package cocktailParty;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.Stack;

import main.Graph;
import steinerTree.BFPaths;

public class CocktailParty {

	public static HashSet<Integer> cocktailParty(Graph graph, ArrayList<Integer> queryNodes) {
		
		// ** Variables initialization ** \\
		Stack<Integer> deletedNodes = new Stack<Integer>(); // Nodes from the last deleted to the first deleted
		Stack<Integer> nodeMinimumDegree = new Stack<Integer>(); // For each node the minimum degree of the graph when it is removed

		ArrayList<HashSet<Integer>> orderedNodes = new ArrayList<HashSet<Integer>>(graph.getNumberOfNodes());
		for (int i = 0; i < graph.getNumberOfNodes(); i++) {
			orderedNodes.add(new HashSet<Integer>());
		}
				
		// Nodes added to groups
		HashMap<Integer, Integer> degrees = new HashMap<Integer, Integer>(graph.getDegrees());
		for (Map.Entry<Integer, Integer> entry : degrees.entrySet()) {
			orderedNodes.get(entry.getValue()).add(entry.getKey());
		}

		// ** Deletes nodes ** \\
		int node; // Current node
		int lowestDegree = 0;
		int neighborDegree;
		
		while (lowestDegree < graph.getNumberOfNodes()) {
			// If there aren't nodes with that degree, increase it
			if (orderedNodes.get(lowestDegree).size() == 0) {
				lowestDegree++;
			} else {
				// Else: take one node with the lowest degree, remove it from the orderedNodes, add it to deletedNodes, its minimum degree as well and set its degree to -1
				node = orderedNodes.get(lowestDegree).iterator().next();
				orderedNodes.get(lowestDegree).remove(node);
				deletedNodes.push(node);
				nodeMinimumDegree.push(lowestDegree);
				degrees.put(node, -1);

				// Decrease the degree of its neighbors and change their set
				for (int neighbor : graph.getNeighbors(node)) {
					neighborDegree = degrees.get(neighbor);
					if (neighborDegree != -1) {
						orderedNodes.get(neighborDegree).remove(neighbor);
						orderedNodes.get(neighborDegree - 1).add(neighbor);

						degrees.put(neighbor, neighborDegree - 1);

						if (neighborDegree - 1 <= lowestDegree) {
							lowestDegree = neighborDegree - 1;
						}
					}
				}
			}
		}

		// ** Create the possible solutions ** \\
		int cocktailPartyMinimumDegree = 0;
		int cocktailPartyIndex = 0;
		int cocktailPartySet = 0;
		
		HashSet<Integer> processedNodes = new HashSet<Integer>();
		
		ArrayList<HashSet<Integer>> groupsOfNodes = new ArrayList<HashSet<Integer>>();
		HashMap<Integer, Integer> nodeGroup = new HashMap<Integer, Integer>();
		
		int numberOfQueryNodes = queryNodes.size();
		
		boolean hasOneGroupFlag = false;
		
		// For every deleted node
		for (int i = deletedNodes.size() - 1; i >= 0; i--) {
			
			// Picks a new node
			node = deletedNodes.elementAt(i);
			
			// If it's a query nodes, decrease its number
			if (queryNodes.contains(node)) {
				numberOfQueryNodes--;
			}
			// Add it to processed nodes
			processedNodes.add(node);
			// For all its neighbors
			for (int neighbor : graph.getNeighbors(node)) {
				// If the neighbor exists in the processed nodes
				if (processedNodes.contains(neighbor)) {
					// If the node doesn't have a group yet
					if (!nodeGroup.containsKey(node)) {
						// Set its group the same as its neighbor
						nodeGroup.put(node, nodeGroup.get(neighbor));
						groupsOfNodes.get(nodeGroup.get(neighbor)).add(node);
					}
					// Else, if it already has a group
					else if (nodeGroup.get(node) != nodeGroup.get(neighbor)) {
						// Merge the two groups (the smallest in the biggest)
						if (groupsOfNodes.get(nodeGroup.get(neighbor)).size() > groupsOfNodes.get(nodeGroup.get(node)).size()) {
							// If the deleted group is the cocktailParty group, change even its value
							if (nodeGroup.get(node) == cocktailPartySet) {
								cocktailPartySet = nodeGroup.get(neighbor);
							}
							
							int oldGroup = nodeGroup.get(node);
							for (int nodeWithNewGroup : groupsOfNodes.get(oldGroup)) {
								nodeGroup.put(nodeWithNewGroup, nodeGroup.get(neighbor));
							}
							groupsOfNodes.get(nodeGroup.get(neighbor)).addAll(groupsOfNodes.get(oldGroup));
						} else {
							// If the deleted group is the cocktailParty group, change even its value
							if (nodeGroup.get(neighbor) == cocktailPartySet) {
								cocktailPartySet = nodeGroup.get(node);
							}
							
							int oldGroup = nodeGroup.get(neighbor);
							for (int nodeWithNewGroup : groupsOfNodes.get(oldGroup)) {
								nodeGroup.put(nodeWithNewGroup, nodeGroup.get(node));
							}
							groupsOfNodes.get(nodeGroup.get(node)).addAll(groupsOfNodes.get(oldGroup));
						}
					}
				}
			}
			// If it has no neighbors
			if (!nodeGroup.containsKey(node)) {
				// Create a new group
				groupsOfNodes.add(new HashSet<Integer>());
				groupsOfNodes.get(groupsOfNodes.size() - 1).add(node);
				nodeGroup.put(node, groupsOfNodes.size() - 1);
			}

			// If the reconstructed graph hasn't all the query nodes
			if (numberOfQueryNodes != 0) {
				continue;
			}
			
			// If the reconstructed graph hasn't the maximum minimum degree (=)
			if (cocktailPartyMinimumDegree >= nodeMinimumDegree.elementAt(i)) {
				continue;
			}
			
			// If at the previous iteration query nodes aren't in the same group
			if (!hasOneGroupFlag) {
				// If query nodes are all in the same group
				for (HashSet<Integer> groupOfNode : groupsOfNodes) {
					for (int queryNode : queryNodes) {
						if (groupOfNode.contains(queryNode)) {
							// It's a possible solution
							hasOneGroupFlag = true;
						} else {
							hasOneGroupFlag = false;
							break;
						}
					}
					if (hasOneGroupFlag) {
						cocktailPartySet = groupsOfNodes.indexOf(groupOfNode);
						break;
					}
				}
				// If query nodes are in different groups go to the next iteration
				if (!hasOneGroupFlag) {
					continue;
				}
			}

			// It's a possible solution
			cocktailPartyIndex = i;
			cocktailPartyMinimumDegree = nodeMinimumDegree.elementAt(i);
		}
		
		// ** Construct the solution set ** \\
		HashSet<Integer> cocktailPartyNodes = new HashSet<Integer>();
		for (int i = deletedNodes.size() - 1; i >= cocktailPartyIndex; i--) {
			node = deletedNodes.elementAt(i);
			if (nodeGroup.containsKey(node) && nodeGroup.get(node) == cocktailPartySet) {
				cocktailPartyNodes.add(node);
			}
		}
		
		return cocktailPartyNodes;
	}
        
	public static HashSet<Integer> cocktailParty_AtLeastK(Graph graph, ArrayList<Integer> queryNodes, int k) {
		
		// ** Variables initialization ** \\
		Stack<Integer> deletedNodes = new Stack<Integer>(); // Nodes from the last deleted to the first deleted
		Stack<Integer> nodeMinimumDegree = new Stack<Integer>(); // For each node the minimum degree of the graph when it is removed

		ArrayList<HashSet<Integer>> orderedNodes = new ArrayList<HashSet<Integer>>(graph.getNumberOfNodes());
		for (int i = 0; i < graph.getNumberOfNodes(); i++) {
			orderedNodes.add(new HashSet<Integer>());
		}
				
		// Nodes added to groups
		HashMap<Integer, Integer> degrees = new HashMap<Integer, Integer>(graph.getDegrees());
		for (Map.Entry<Integer, Integer> entry : degrees.entrySet()) {
			orderedNodes.get(entry.getValue()).add(entry.getKey());
		}

		// ** Deletes nodes ** \\
		int node; // Current node
		int lowestDegree = 0;
		int neighborDegree;
		
		while (lowestDegree < graph.getNumberOfNodes()) {
			// If there aren't nodes with that degree, increase it
			if (orderedNodes.get(lowestDegree).size() == 0) {
				lowestDegree++;
			} else {
				// Else: take one node with the lowest degree, remove it from the orderedNodes, add it to deletedNodes, its minimum degree as well and set its degree to -1
				node = orderedNodes.get(lowestDegree).iterator().next();
				orderedNodes.get(lowestDegree).remove(node);
				deletedNodes.push(node);
				nodeMinimumDegree.push(lowestDegree);
				degrees.put(node, -1);

				// Decrease the degree of its neighbors and change their set
				for (int neighbor : graph.getNeighbors(node)) {
					neighborDegree = degrees.get(neighbor);
					if (neighborDegree != -1) {
						orderedNodes.get(neighborDegree).remove(neighbor);
						orderedNodes.get(neighborDegree - 1).add(neighbor);

						degrees.put(neighbor, neighborDegree - 1);

						if (neighborDegree - 1 <= lowestDegree) {
							lowestDegree = neighborDegree - 1;
						}
					}
				}
			}
		}

		// ** Create the possible solutions ** \\
		int cocktailPartyMinimumDegree = 0;
		int cocktailPartyIndex = 0;
		int cocktailPartySet = 0;
		
		HashSet<Integer> processedNodes = new HashSet<Integer>();
		
		ArrayList<HashSet<Integer>> groupsOfNodes = new ArrayList<HashSet<Integer>>();
		HashMap<Integer, Integer> nodeGroup = new HashMap<Integer, Integer>();
		
		int numberOfQueryNodes = queryNodes.size();
		
		boolean hasOneGroupFlag = false;
		
		// For every deleted node
		for (int i = deletedNodes.size() - 1; i >= k; i--) {
			
			// Picks a new node
			node = deletedNodes.elementAt(i);
			
			// If it's a query nodes, decrease its number
			if (queryNodes.contains(node)) {
				numberOfQueryNodes--;
			}
			// Add it to processed nodes
			processedNodes.add(node);
			// For all its neighbors
			for (int neighbor : graph.getNeighbors(node)) {
				// If the neighbor exists in the processed nodes
				if (processedNodes.contains(neighbor)) {
					// If the node doesn't have a group yet
					if (!nodeGroup.containsKey(node)) {
						// Set its group the same as its neighbor
						nodeGroup.put(node, nodeGroup.get(neighbor));
						groupsOfNodes.get(nodeGroup.get(neighbor)).add(node);
					}
					// Else, if it already has a group
					else if (nodeGroup.get(node) != nodeGroup.get(neighbor)) {
						// Merge the two groups (the smallest in the biggest)
						if (groupsOfNodes.get(nodeGroup.get(neighbor)).size() > groupsOfNodes.get(nodeGroup.get(node)).size()) {
							// If the deleted group is the cocktailParty group, change even its value
							if (nodeGroup.get(node) == cocktailPartySet) {
								cocktailPartySet = nodeGroup.get(neighbor);
							}
							
							int oldGroup = nodeGroup.get(node);
							for (int nodeWithNewGroup : groupsOfNodes.get(oldGroup)) {
								nodeGroup.put(nodeWithNewGroup, nodeGroup.get(neighbor));
							}
							groupsOfNodes.get(nodeGroup.get(neighbor)).addAll(groupsOfNodes.get(oldGroup));
						} else {
							// If the deleted group is the cocktailParty group, change even its value
							if (nodeGroup.get(neighbor) == cocktailPartySet) {
								cocktailPartySet = nodeGroup.get(node);
							}
							
							int oldGroup = nodeGroup.get(neighbor);
							for (int nodeWithNewGroup : groupsOfNodes.get(oldGroup)) {
								nodeGroup.put(nodeWithNewGroup, nodeGroup.get(node));
							}
							groupsOfNodes.get(nodeGroup.get(node)).addAll(groupsOfNodes.get(oldGroup));
						}
					}
				}
			}
			// If it has no neighbors
			if (!nodeGroup.containsKey(node)) {
				// Create a new group
				groupsOfNodes.add(new HashSet<Integer>());
				groupsOfNodes.get(groupsOfNodes.size() - 1).add(node);
				nodeGroup.put(node, groupsOfNodes.size() - 1);
			}

			// If the reconstructed graph hasn't all the query nodes
			if (numberOfQueryNodes != 0) {
				continue;
			}
			
			// If the reconstructed graph hasn't the maximum minimum degree (=)
			if (cocktailPartyMinimumDegree >= nodeMinimumDegree.elementAt(i)) {
				continue;
			}
			
			// If at the previous iteration query nodes aren't in the same group
			if (!hasOneGroupFlag) {
				// If query nodes are all in the same group
				for (HashSet<Integer> groupOfNode : groupsOfNodes) {
					for (int queryNode : queryNodes) {
						if (groupOfNode.contains(queryNode)) {
							// It's a possible solution
							hasOneGroupFlag = true;
						} else {
							hasOneGroupFlag = false;
							break;
						}
					}
					if (hasOneGroupFlag) {
						cocktailPartySet = groupsOfNodes.indexOf(groupOfNode);
						break;
					}
				}
				// If query nodes are in different groups go to the next iteration
				if (!hasOneGroupFlag) {
					continue;
				}
			}

			// It's a possible solution
			cocktailPartyIndex = i;
			cocktailPartyMinimumDegree = nodeMinimumDegree.elementAt(i);
		}
		
		// ** Construct the solution set ** \\
		HashSet<Integer> cocktailPartyNodes = new HashSet<Integer>();
		for (int i = deletedNodes.size() - 1; i >= cocktailPartyIndex; i--) {
			node = deletedNodes.elementAt(i);
			if (nodeGroup.containsKey(node) && nodeGroup.get(node) == cocktailPartySet) {
				cocktailPartyNodes.add(node);
			}
		}
		
		return cocktailPartyNodes;
	}
        
        private static Graph preprocessForGreedyFast(Graph graph, ArrayList<Integer> queryNodes, int k) 
        {
            //for each node, compute distance fromt the query nodes
            HashSet<Integer> nodes = new HashSet<Integer>();
            for(int x:graph.getNodes())
            {
                nodes.add(x);
            }

            Map<Integer,Integer> distances = new HashMap<Integer,Integer>();
            for(int x:nodes)
            {
                distances.put(x, 0);
            }
            for(int q:queryNodes)
            {
                distances.remove(q);
            }

            
            for(int q:queryNodes)
            {
                BFPaths bfs = new BFPaths(graph, q, nodes);
                for(int x:nodes)
                {
                    if(distances.containsKey(x))
                    {
                        int d = bfs.distTo(x);
                        if(d != Integer.MAX_VALUE)
                        {
                            int dold = distances.get(x);
                            distances.put(x, dold+d*d);
                        }
                    }
                }
            }
            
            //sort nodes based on their distance from the query nodes
            Pair[] pairs = new Pair[distances.size()];
            int i = 0;
            for(int x:distances.keySet())
            {
                pairs[i] = new Pair(x,distances.get(x));
                i++;
            }
            Arrays.sort(pairs);
            
            
            //add nodes (following the order above) until we get a connected solution
            Set<Integer> ccs = new HashSet<Integer>(); //set of current connected components
            Map<Integer,Set<Integer>> cc_content = new HashMap<Integer,Set<Integer>>(); //for each connected component, its nodes
            Map<Integer,Integer> node2cc = new HashMap<Integer,Integer>(); //for each node, its connected component
            for(int q:queryNodes)
            {
                ccs.add(q);
                Set<Integer> cc = new HashSet<Integer>();
                cc.add(q);
                cc_content.put(q, cc);
                node2cc.put(q, q);
            }
            
            int j=0;
            HashSet<Integer> toBeRetained = new HashSet<Integer>();
            for(int q:queryNodes)
            {
                toBeRetained.add(q);
            }
            while(ccs.size() > 1 && j<pairs.length)
            {
                int x = pairs[j].getId();
                toBeRetained.add(x);
                
                Set<Integer> toBeMerged = new HashSet<Integer>();
                for(int y:graph.getNeighbors(x))
                {
                    if(node2cc.containsKey(y))
                    {
                        toBeMerged.add(node2cc.get(y));
                    }
                }
                
                if(!toBeMerged.isEmpty())
                {
                    Iterator<Integer> it = toBeMerged.iterator();
                    int newCC = it.next();
                    Set<Integer> newCC_content = cc_content.get(newCC);
                    newCC_content.add(x);
                    node2cc.put(x, newCC);
                    
                    while(it.hasNext())
                    {
                        int cc = it.next();
                        for(int y:cc_content.get(cc))
                        {
                            newCC_content.add(y);
                            node2cc.put(y, newCC);
                        }
                        ccs.remove(cc);
                        cc_content.remove(cc);
                    }
                    
                }
                else
                {
                    ccs.add(x);
                    Set<Integer> ccc = new HashSet<Integer>();
                    ccc.add(x);
                    cc_content.put(x, ccc);
                    node2cc.put(x, x);
                }
                
                j++;
            }
            
            while(toBeRetained.size() < k)
            {
                toBeRetained.add(pairs[j].getId());
                j++;
            }
            
            Graph newGraph = new Graph(graph,toBeRetained);
            
            return newGraph;
            
        }
        
        
        public static HashSet<Integer> cocktailParty_constrained_GreedyFast(Graph graph, ArrayList<Integer> queryNodes, int k) 
        {
            Graph preprocessedGraph = preprocessForGreedyFast(graph, queryNodes, k);
            //HashSet<Integer> solution = cocktailParty_AtLeastK(preprocessedGraph, queryNodes, k);
            HashSet<Integer> solution = cocktailParty(preprocessedGraph, queryNodes);
            
            return solution;
        }
	

}
