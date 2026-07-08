package steinerTree;

import java.util.HashMap;
import java.util.NoSuchElementException;

public class IndexMinPQ<Key extends Comparable<Key>>{
    private int N;           						// number of elements on PQ
    private HashMap<Integer, Integer> pq;        	// binary heap using 1-based indexing
    private HashMap<Integer, Integer> qp;        	// inverse of pq - qp[pq[i]] = pq[qp[i]] = i
    private HashMap<Integer, Key> keys;      		// keys[i] = priority of i

   public IndexMinPQ() {
        this.keys = new HashMap<Integer, Key>();    
        this.pq   = new HashMap<Integer, Integer>();
        this.qp   = new HashMap<Integer, Integer>();
    }

    public boolean isEmpty() {
        return N == 0;
    }

   public boolean contains(int i) {
        return this.qp.containsKey(i);
    }

   public void insert(int i, Key key) {
        if (this.contains(i)) throw new IllegalArgumentException("index is already in the priority queue");
        this.N++;
        this.qp.put(i, this.N);
        this.pq.put(N, i);
        this.keys.put(i, key);
        swim(N);
    }

    public int delMin() { 
        if (N == 0) throw new NoSuchElementException("Priority queue underflow");
        int min = this.pq.get(1);        
        exch(1, N--); 
        sink(1);
        this.qp.remove(min);
        this.keys.remove(this.pq.get(this.N + 1));
        this.pq.remove(this.N + 1);
        return min; 
    }

   public void decreaseKey(int i, Key key) {
        if (!this.contains(i)) throw new NoSuchElementException("index is not in the priority queue");
        if (this.keys.get(i).compareTo(key) <= 0) throw new IllegalArgumentException("Calling decreaseKey() with given argument would not strictly decrease the key");
        this.keys.put(i, key);
        swim(this.qp.get(i));
    }

  private boolean greater(int i, int j) {
        return this.keys.get(this.pq.get(i)).compareTo(this.keys.get(this.pq.get(j))) > 0;
    }

    private void exch(int i, int j) {
        int swap = this.pq.get(i); 
        this.pq.put(i, this.pq.get(j)); 
        this.pq.put(j, swap);
        this.qp.put(this.pq.get(i), i); 
        this.qp.put(this.pq.get(j), j);
    }


   private void swim(int k)  {
        while (k > 1 && greater(k/2, k)) {
            exch(k, k/2);
            k = k/2;
        }
    }

    private void sink(int k) {
        while (2*k <= this.N) {
            int j = 2*k;
            if (j < this.N && greater(j, j+1)) j++;
            if (!greater(k, j)) break;
            exch(k, j);
            k = j;
        }
    }
    
}