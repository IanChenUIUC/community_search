package steinerTree;

import index.IndexInterface;
import java.util.ArrayList;
import java.util.Collection;

import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Map;
import java.util.Stack;

import main.Graph;

public class BFPaths {
	
	private HashMap<Integer, Integer> edgeTo; // edgeTo[v] = previous edge on shortest s-v path
	private HashMap<Integer, Integer> distTo; // distTo[v] = number of edges shortest s-v path
        private HashMap<Integer, HashSet<Integer>> nexts;

	// Computes the shortest path between the source vertex s and every other vertex in the core identified by minimumCoreIndex
	public BFPaths(IndexInterface index, int minimumCoreIndex, int s, HashSet<Integer> queryNodes) {
		this.distTo = new HashMap<Integer, Integer>();
		this.edgeTo = new HashMap<Integer, Integer>();
		this.bfs(index, minimumCoreIndex, s, queryNodes);
	 }
	 
	 // Breadth-first search from a single source
	 private void bfs(IndexInterface index, int minimumCoreIndex, int s, HashSet<Integer> queryNodes) {
		int numberOfQueryNodes = queryNodes.size();
		LinkedList<Integer> q = new LinkedList<Integer>();
		
		this.distTo.put(s, 0);
	  	q.add(s);
	  	numberOfQueryNodes--;
	  	
	  	while (!q.isEmpty()) {
	  		int v = q.poll();
	  		for (int w : index.getNeighbors(v, minimumCoreIndex)) {
	  			if (!this.distTo.containsKey(w)) {
	  				this.edgeTo.put(w, v);
	  				this.distTo.put(w, this.distTo.get(v) + 1);
	  				q.add(w);
	  				
	  				if (queryNodes.contains(w)) {
	  					numberOfQueryNodes--;
	  				}
	  				if (numberOfQueryNodes == 0) {
						break;
					}
	  			}
	  		}
	  		if (numberOfQueryNodes == 0) {
				break;
			}
        }
	 }
         
        // Computes the shortest path between the source vertex s and every other vertex in the core identified by minimumCoreIndex,
         // but not considering the whole core, rather only the portion identified by the nodes in subcore (
	public BFPaths(IndexInterface index, int minimumCoreIndex, int s, HashSet<Integer> queryNodes, HashSet<Integer> subcore) {
		this.distTo = new HashMap<Integer, Integer>();
		this.edgeTo = new HashMap<Integer, Integer>();
		this.bfs(index, minimumCoreIndex, s, queryNodes, subcore);
	 }
        
	 // Breadth-first search from a single source
	 private void bfs(IndexInterface index, int minimumCoreIndex, int s, HashSet<Integer> queryNodes, HashSet<Integer> subcore) {
		int numberOfQueryNodes = queryNodes.size();
		LinkedList<Integer> q = new LinkedList<Integer>();
		
		this.distTo.put(s, 0);
	  	q.add(s);
	  	numberOfQueryNodes--;
	  	
	  	while (!q.isEmpty()) {
	  		int v = q.poll();
	  		for (int w : index.getNeighbors(v, minimumCoreIndex, subcore)) {
	  			if (!this.distTo.containsKey(w)) {
	  				this.edgeTo.put(w, v);
	  				this.distTo.put(w, this.distTo.get(v) + 1);
	  				q.add(w);
	  				
	  				if (queryNodes.contains(w)) {
	  					numberOfQueryNodes--;
	  				}
	  				if (numberOfQueryNodes == 0) {
						break;
					}
	  			}
	  		}
	  		if (numberOfQueryNodes == 0) {
				break;
			}
        }
	 }
         
         
        // Computes the shortest path between the a dummy source vertex (connected to all query nodes with an edge of length 0)
         // and every other vertex in the core identified by minimumCoreIndex,
         // but not considering the whole core, rather only the portion identified by the nodes in subcore (
	public BFPaths(IndexInterface index, int minimumCoreIndex, ArrayList<Integer> queryNodes, HashSet<Integer> subcore, Map<Integer,HashSet<Integer>> neighborCache) {
		this.distTo = new HashMap<Integer, Integer>();
		this.edgeTo = new HashMap<Integer, Integer>();
                this.nexts = new HashMap<Integer,HashSet<Integer>>();
		this.bfs_dummy(index, minimumCoreIndex, queryNodes, subcore, neighborCache);
	 }
        
	 // Breadth-first search from a single source
	 private void bfs_dummy(IndexInterface index, int minimumCoreIndex, ArrayList<Integer> queryNodes, HashSet<Integer> subcore, Map<Integer, HashSet<Integer>> neighborCache) {
		LinkedList<Integer> q = new LinkedList<Integer>();
		
                int s = -1;//dummy source node
		this.distTo.put(s, -1);
	  	q.add(s);
	  	
	  	while (!q.isEmpty()) {
	  		int v = q.poll();
                        
                        Collection<Integer> v_neighbors = queryNodes;
                        if(v != s)
                        {
                            if(!neighborCache.containsKey(v))
                            {
                                neighborCache.put(v, index.getNeighbors(v, minimumCoreIndex, subcore));
                            }
                            v_neighbors = neighborCache.get(v);
                        }
	  		for (int w : v_neighbors) {
	  			if (!this.distTo.containsKey(w)) {
	  				this.edgeTo.put(w, v);
	  				this.distTo.put(w, this.distTo.get(v) + 1);
	  				q.add(w);
                                        if(!nexts.containsKey(v))
                                        {
                                            nexts.put(v, new HashSet<Integer>());
                                        }
                                        nexts.get(v).add(w);
	  			}
	  		}
        }
	 }
         
         public Map<Integer,Integer> getVoronoiInfo_Mehlhorn()
         {
             Map<Integer,Integer> node2closestQueryNode = new HashMap<Integer,Integer>();
             
             ArrayList<Integer> currentLayer = new ArrayList<Integer>();
             for(int x:nexts.get(-1))//nexts of the dummy source, i.e., the query nodes
             {
                 currentLayer.add(x);
                 node2closestQueryNode.put(x, x);
             }
             while(!currentLayer.isEmpty())
             {
                 ArrayList<Integer> newLayer = new ArrayList<Integer>();
                 for(int x:currentLayer)
                 {
                     if(nexts.containsKey(x))
                     {
                         int label = node2closestQueryNode.get(x);
                         for(int y:nexts.get(x))
                         {
                             node2closestQueryNode.put(y, label);
                             newLayer.add(y);
                         }
                     }
                 }
                 currentLayer = newLayer;
             }
             
             
             return node2closestQueryNode;
         }
         
         
	 
	// Computes the shortest path between the source vertex s and every other vertex in the graph G
	public BFPaths(Graph graph, int s, HashSet<Integer> nodes) {
		this.distTo = new HashMap<Integer, Integer>();
		this.edgeTo = new HashMap<Integer, Integer>();
		this.bfs(graph, s, nodes);
	 }
		 
	 // Breadth-first search from a single source
	 private void bfs(Graph graph, int s, HashSet<Integer> nodes) {
		int numberOfNodes = nodes.size();
		LinkedList<Integer> q = new LinkedList<Integer>();
			
		this.distTo.put(s, 0);
	  	q.add(s);
	  	numberOfNodes--;
		  	
	  	while (!q.isEmpty()) {
	  		int v = q.poll();
	  		for (int w : graph.getNeighbors(v)) {
	  			if (!this.distTo.containsKey(w)) {
	  				this.edgeTo.put(w, v);
		  			this.distTo.put(w, this.distTo.get(v) + 1);
		  			q.add(w);
		  			
		  			if (nodes.contains(w)) {
		  				numberOfNodes--;
		  			}
	 				if (numberOfNodes == 0) {
						break;
						}
		  		}
		  	}
		  	if (numberOfNodes == 0) {
				break;
			}
        }
	  }
	 
	 // Returns the number of nodes that have been visited
	 public int numbertOfVisitedNodes() {
		 return this.distTo.size();
	 }
	 
	 // Is there a path between the source vertex s (or sources) and vertex v?
	 public boolean hasPathTo(int v) {
	        return this.distTo.containsKey(v);
	 }
	 
	 // Returns the number of edges in a shortest path between the source vertex s (or sources) and vertex v
	 public int distTo(int v) {
		 if (this.distTo.containsKey(v)) {
			 return this.distTo.get(v);
		} else {
			return Integer.MAX_VALUE;
		}
	 }
	 
	 // Returns a shortest path between the source vertex s (or sources) and v or null if no such path exists
	 public Iterable<Integer> pathTo(int v) {
	        if (!this.hasPathTo(v)) {
	        	return null;
	        }
	        Stack<Integer> path = new Stack<Integer>();
	        int x;
	        for (x = v; this.distTo.get(x) != 0; x = this.edgeTo.get(x)) {
	        	path.push(x);
	        }
	        path.push(x);
	        return path;
	 }

}
