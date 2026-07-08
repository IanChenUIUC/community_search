package steinerTree;

import java.util.HashMap;
import java.util.HashSet;

public class PrimMST {

    private final HashMap<Integer, Edge> edgeTo;        // edgeTo[v] = shortest edge from tree vertex to non-tree vertex
    private final HashMap<Integer, Double> distTo;      // distTo[v] = weight of shortest such edge
    private final HashSet<Integer> marked;     // marked[v] = true if v on tree, false otherwise
    private final IndexMinPQ<Double> pq;

    public PrimMST(WeightedGraph G) {
        this.edgeTo = new HashMap<Integer, Edge>();
        this.distTo = new HashMap<Integer, Double>();
        this.marked = new HashSet<Integer>();
        this.pq = new IndexMinPQ<Double>();

        for (Integer node : G.getNodes()) {
            if (!this.marked.contains(node)) {
                prim(G, node);
            }

        }

    }

    // run Prim's algorithm in graph G, starting from vertex s
    private void prim(WeightedGraph G, int s) {
        this.distTo.put(s, 0.0);
        this.pq.insert(s, this.distTo.get(s));
        while (!this.pq.isEmpty()) {
            int v = this.pq.delMin();
            scan(G, v);
        }
    }

    // scan vertex v
    private void scan(WeightedGraph G, int v) {
        this.marked.add(v);
        for (Edge e : G.getIncidentEdges(v)) {
            int w = e.otherNode(v);
            if (this.marked.contains(w)) {
                continue;         // v-w is obsolete edge
            }
            if (!this.distTo.containsKey(w) || e.getWeight() < this.distTo.get(w)) {
                this.distTo.put(w, (double) e.getWeight());
                this.edgeTo.put(w, e);
                if (this.pq.contains(w)) {
                    this.pq.decreaseKey(w, this.distTo.get(w));
                } else {
                    this.pq.insert(w, this.distTo.get(w));
                }
            }
        }
    }

    public Iterable<Edge> edges() {
        return this.edgeTo.values();
    }

    public double weight() {
        double weight = 0.0;
        for (Edge e : edges()) {
            weight += e.getWeight();
        }
        return weight;
    }

}
