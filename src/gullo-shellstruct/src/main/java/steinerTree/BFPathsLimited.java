package steinerTree;

import index.IndexInterface;

import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Stack;

import main.Graph;

public class BFPathsLimited {

    private final HashMap<Integer, Integer> edgeTo; // edgeTo[v] = previous edge on shortest s-v path
    private final HashMap<Integer, Integer> distTo; // distTo[v] = number of edges shortest s-v path

    // Computes the shortest path between the source vertex s and every other vertex in the graph G
    public BFPathsLimited(IndexInterface index, int minimumCoreIndex, int s, HashSet<Integer> queryNodes, int maximumDepth) {
        this.distTo = new HashMap<Integer, Integer>();
        this.edgeTo = new HashMap<Integer, Integer>();
        this.bfs(index, minimumCoreIndex, s, queryNodes, maximumDepth);
    }

    // Breadth-first search from a single source
    private void bfs(IndexInterface index, int minimumCoreIndex, int s, HashSet<Integer> queryNodes, int maximumDepth) {
        int numberOfQueryNodes = queryNodes.size();
        int depth = 0;
        LinkedList<Integer> q = new LinkedList<Integer>();

        this.distTo.put(s, 0);
        q.add(s);
        numberOfQueryNodes--;

        int currentLevel = 1;
        int nextLevel = 0;
        while (!q.isEmpty()) {
            int v = q.poll();
            currentLevel--;
            
            for (int w : index.getNeighbors(v, minimumCoreIndex)) {
                if (!this.distTo.containsKey(w)) {
                    nextLevel++;
                    
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
            
            if (currentLevel == 0) {
                currentLevel = nextLevel;
                nextLevel = 0;
                depth++;
            }
            
            if (numberOfQueryNodes == 0 || depth >= maximumDepth) {
                break;
            }
        }
    }

    // Computes the shortest path between the source vertex s and every other vertex in the graph G
    public BFPathsLimited(Graph graph, int s, HashSet<Integer> nodes) {
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