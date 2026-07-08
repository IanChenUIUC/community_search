/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package fromResults;

import java.util.Collections;
import java.util.HashMap;
import main.Graph;
import steinerTree.IndexMinPQ;

/**
 *
 * @author edotony
 */
public class DijkstraSP {
    private final HashMap<Integer, Double> distTo;          // distTo[v] = distance  of shortest s->v path
    private final HashMap<Integer, Integer> edgeTo;    // edgeTo[v] = last edge on shortest s->v path
    private final IndexMinPQ<Double> pq;    // priority queue of vertices

    /**
     * Computes a shortest paths tree from <tt>s</tt> to every other vertex in
     * the edge-weighted digraph <tt>G</tt>.
     * @param G
     */
    public DijkstraSP(Graph G, int s) {
        
        this.distTo = new HashMap<Integer, Double>();
        this.edgeTo = new HashMap<Integer, Integer>();
        this.distTo.put(s, 0.0);

        // relax vertices in order of distance from s
        this.pq = new IndexMinPQ<Double>();
        this.pq.insert(s, this.distTo.get(s));
        while (!this.pq.isEmpty()) {
            int v = this.pq.delMin();
            for (int e : G.getNeighbors(v))
                relax(v, e);
        }
    }

    // relax edge e and update pq if changed
    private void relax(int v, int w) {
        if (!this.distTo.containsKey(w) || this.distTo.get(w) > this.distTo.get(v) + 1) {
            this.distTo.put(w, this.distTo.get(v) + 1);
            this.edgeTo.put(w, v);
            if (this.pq.contains(w)) {
                this.pq.decreaseKey(w, this.distTo.get(w));
            } else {
                this.pq.insert(w, this.distTo.get(w));
            }
        }
    }

    /**
     * Returns the length of a shortest path from the source vertex <tt>s</tt> to vertex <tt>v</tt>.
     * @param v the destination vertex
     * @return the length of a shortest path from the source vertex <tt>s</tt> to vertex <tt>v</tt>;
     *    <tt>Double.POSITIVE_INFINITY</tt> if no such path
     */
    public double distTo(int v) {
        if (this.distTo.containsKey(v)) {
            return this.distTo.get(v);
        } else {
            return Double.POSITIVE_INFINITY;
        }
    }

    /**
     * Is there a path from the source vertex <tt>s</tt> to vertex <tt>v</tt>?
     * @param v the destination vertex
     * @return <tt>true</tt> if there is a path from the source vertex
     *    <tt>s</tt> to vertex <tt>v</tt>, and <tt>false</tt> otherwise
     */
    public boolean hasPathTo(int v) {
        return this.distTo.containsKey(v);
    }

    /**
     * Returns the distance of the most distance vertex from the source vertex
     * @return the distance of the most distance vertex from the source vertex
     */
    public double maxDistTo() {
        return Collections.max(this.distTo.values());
    }
}
