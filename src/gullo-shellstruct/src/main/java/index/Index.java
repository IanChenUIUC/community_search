package index;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.Stack;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.UnsafeMemoryInput;
import com.esotericsoftware.kryo.io.UnsafeMemoryOutput;

import coreGroups.CoreGroups;
import main.Graph;

public class Index implements IndexInterface {

    // The core index for each node in the graph
    HashMap<Integer, Integer> coreIndex;

    // For each core, its minimum degree
    private HashMap<Integer, Integer> coreMinimumDegree = new HashMap<Integer, Integer>();

    // For each core, for each node in the core its connected component
    private HashMap<Integer, HashMap<Integer, Integer>> coreComposition;
    // For each core, for each connected component its nodes
    private HashMap<Integer, HashMap<Integer, HashSet<Integer>>> coreConnectedComponents;

    // For each node, for each shell the neighbors of the node
    private HashMap<Integer, HashMap<Integer, HashSet<Integer>>> nodeNeighbors;

    public Index(Graph graph, String datasetName) {
        long startTime = System.currentTimeMillis();
        this.computeCoreIndex(graph);

        this.computeCoreComposition(graph);

        this.computeNodeNeighbors(graph);
        long endTime = System.currentTimeMillis();
        System.out.println("Index execution time: " + (endTime - startTime));
        
        this.serializeToFile(datasetName + "CoreStruct.bin");
    }

    @SuppressWarnings("unchecked")
    public Index(String datasetName) {
        this.coreIndex = new HashMap<Integer, Integer>();
        this.coreMinimumDegree = new HashMap<Integer, Integer>();
        this.coreComposition = new HashMap<Integer, HashMap<Integer, Integer>>();
        this.coreConnectedComponents = new HashMap<Integer, HashMap<Integer, HashSet<Integer>>>();
        this.nodeNeighbors = new HashMap<Integer, HashMap<Integer, HashSet<Integer>>>();

        try {
            Kryo kryo = new Kryo();
            FileInputStream fileInputStream = new FileInputStream(datasetName + ".bin");
            UnsafeMemoryInput input = new UnsafeMemoryInput(fileInputStream);

            this.coreIndex = kryo.readObject(input, this.coreIndex.getClass());
            this.coreMinimumDegree = kryo.readObject(input, this.coreMinimumDegree.getClass());
            this.coreComposition = kryo.readObject(input, this.coreComposition.getClass());
            this.coreConnectedComponents = kryo.readObject(input, this.coreConnectedComponents.getClass());
            this.nodeNeighbors = kryo.readObject(input, this.nodeNeighbors.getClass());

            input.close();
        } catch (FileNotFoundException e) {
            System.out.println("Error in reading file.");
            System.exit(0);
        }
    }

    private void computeCoreIndex(Graph graph) {
        // Compute the core of each node
        this.coreIndex = CoreGroups.coreGroupsAlgorithm(graph);

        // Add each core number in the set
        HashSet<Integer> coreNumber = new HashSet<Integer>();
        for (int i : this.coreIndex.keySet()) {
            coreNumber.add(this.coreIndex.get(i));
        }

        // Sort the set of the core numbers in an array
        ArrayList<Integer> sortedCoreNumber = new ArrayList<Integer>(coreNumber);
        Collections.sort(sortedCoreNumber);

        // Map the sorted core number with a consecutive index and save its minimum degree
        HashMap<Integer, Integer> mappedCoreNumber = new HashMap<Integer, Integer>();
        this.coreMinimumDegree = new HashMap<Integer, Integer>();
        int index = 0;
        for (int core : sortedCoreNumber) {
            mappedCoreNumber.put(core, index);
            this.coreMinimumDegree.put(index, core);
            index++;
        }

        // Replace the core number with the index
        for (int i : this.coreIndex.keySet()) {
            this.coreIndex.put(i, mappedCoreNumber.get(this.coreIndex.get(i)));
        }
        
        System.out.println("The graph has " + this.coreMinimumDegree.size() + " cores.");
    }

    private void computeCoreComposition(Graph graph) {
        this.coreComposition = new HashMap<Integer, HashMap<Integer, Integer>>();
        this.coreConnectedComponents = new HashMap<Integer, HashMap<Integer, HashSet<Integer>>>();

        // Add each node to the stack representing its core
        ArrayList<Stack<Integer>> coreGroup = new ArrayList<Stack<Integer>>();
        for (int i = 0; i < graph.getNumberOfNodes(); i++) {
            coreGroup.add(new Stack<Integer>());
        }
        for (int i : this.coreIndex.keySet()) {
            coreGroup.get(coreGroup.size() - this.coreIndex.get(i) - 1).push(i);
        }

        ArrayList<HashSet<Integer>> groupsOfNodes = new ArrayList<HashSet<Integer>>(); // Groups of nodes that represent the connected components
        HashMap<Integer, Integer> nodeGroup = new HashMap<Integer, Integer>(); // The group of each node

        // For each group
        int coreIndex = coreGroup.size() - 1;
        for (Stack<Integer> group : coreGroup) {
            boolean emptyGroup = group.isEmpty();

            // For each node in the group
            while (!group.isEmpty()) {
                int node = group.pop();

                // For all its neighbors
                for (int neighbor : graph.getNeighbors(node)) {
                    // If the node doesn't have a group yet and its neighbor does
                    if (!nodeGroup.containsKey(node) && nodeGroup.containsKey(neighbor)) {
                        // Set its group the same as its neighbor
                        nodeGroup.put(node, nodeGroup.get(neighbor));
                        groupsOfNodes.get(nodeGroup.get(node)).add(node);
                    } // Else, if it already has a group which is different from the neighbor one
                    else if (nodeGroup.get(node) != nodeGroup.get(neighbor) && nodeGroup.containsKey(node) && nodeGroup.containsKey(neighbor)) {
                        // Merge the two groups (the smallest in the biggest)
                        if (groupsOfNodes.get(nodeGroup.get(neighbor)).size() > groupsOfNodes.get(nodeGroup.get(node)).size()) {
                            int oldGroup = nodeGroup.get(node);
                            for (int nodeWithNewGroup : groupsOfNodes.get(oldGroup)) {
                                nodeGroup.put(nodeWithNewGroup, nodeGroup.get(neighbor));;
                            }
                            groupsOfNodes.get(nodeGroup.get(neighbor)).addAll(groupsOfNodes.get(oldGroup));
                            groupsOfNodes.get(oldGroup).clear();
                        } else {
                            int oldGroup = nodeGroup.get(neighbor);
                            for (int nodeWithNewGroup : groupsOfNodes.get(oldGroup)) {
                                nodeGroup.put(nodeWithNewGroup, nodeGroup.get(node));
                            }
                            groupsOfNodes.get(nodeGroup.get(node)).addAll(groupsOfNodes.get(oldGroup));
                            groupsOfNodes.get(oldGroup).clear();
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
            }

            if (!emptyGroup) {
                this.coreComposition.put(coreIndex, new HashMap<Integer, Integer>());
                this.coreConnectedComponents.put(coreIndex, new HashMap<Integer, HashSet<Integer>>());

                // Store the groups
                int connectedComponentIndex = 0;
                for (HashSet<Integer> connectedComponent : groupsOfNodes) {
                    if (!connectedComponent.isEmpty()) {
                        for (int node : connectedComponent) {
                            this.coreComposition.get(coreIndex).put(node, connectedComponentIndex);
                        }
                        this.coreConnectedComponents.get(coreIndex).put(connectedComponentIndex, new HashSet<Integer>(connectedComponent));
                        connectedComponentIndex++;
                    }
                }
            }
            coreIndex--;
        }
    }

    private void computeNodeNeighbors(Graph graph) {
        this.nodeNeighbors = new HashMap<Integer, HashMap<Integer, HashSet<Integer>>>();

        Set<Integer> nodes = graph.getNodes();
        for (int node : nodes) {
            this.nodeNeighbors.put(node, new HashMap<Integer, HashSet<Integer>>());

            for (int neighbor : graph.getNeighbors(node)) {
                if (this.nodeNeighbors.get(node).containsKey(this.coreIndex.get(neighbor))) {
                    this.nodeNeighbors.get(node).get(this.coreIndex.get(neighbor)).add(neighbor);
                } else {
                    this.nodeNeighbors.get(node).put(this.coreIndex.get(neighbor), new HashSet<Integer>());
                    this.nodeNeighbors.get(node).get(this.coreIndex.get(neighbor)).add(neighbor);
                }
            }
        }
    }

    private void serializeToFile(String path) {
        try {
            Kryo kryo = new Kryo();
            FileOutputStream fileOutputStream = new FileOutputStream(path);
            UnsafeMemoryOutput output = new UnsafeMemoryOutput(fileOutputStream);
            kryo.writeObject(output, this.coreIndex);
            output.flush();
            kryo.writeObject(output, this.coreMinimumDegree);
            output.flush();
            kryo.writeObject(output, this.coreComposition);
            output.flush();
            kryo.writeObject(output, this.coreConnectedComponents);
            output.flush();
            kryo.writeObject(output, this.nodeNeighbors);

            output.close();
            fileOutputStream.close();
        } catch (FileNotFoundException e) {
            System.out.println("Error in writing file.");
            System.exit(0);
        } catch (IOException e) {
            System.out.println("Error in writing file.");
            System.exit(0);
        }
    }

    public int getMinimumCoreIndex(ArrayList<Integer> queryNodes) {
        HashSet<Integer> queryIndexes = new HashSet<Integer>();
        for (int queryNode : queryNodes) {
            queryIndexes.add(this.coreIndex.get(queryNode));
        }
        int k = Collections.min(queryIndexes);

        if (sameConnectedComponent(k, queryNodes)) {
            return k;
        }

        // Array with the values about connected components
        HashMap<Integer, Boolean> sameConnectedComponent = new HashMap<Integer, Boolean>();
        int middle = k / 2;

        while (true) {
            boolean sameConnectedComponentMiddle;
            boolean sameConnectedComponentMiddleLess;
            boolean sameConnectedComponentMiddleMore;
            try {
                sameConnectedComponentMiddle = sameConnectedComponent.get(middle);
            } catch (Exception e) {
                sameConnectedComponent.put(middle, sameConnectedComponent(middle, queryNodes));
                sameConnectedComponentMiddle = sameConnectedComponent.get(middle);
            }
            try {
                sameConnectedComponentMiddleMore = sameConnectedComponent.get(middle + 1);
            } catch (Exception e) {
                sameConnectedComponent.put(middle + 1, sameConnectedComponent(middle + 1, queryNodes));
                sameConnectedComponentMiddleMore = sameConnectedComponent.get(middle + 1);
            }
            try {
                sameConnectedComponentMiddleLess = sameConnectedComponent.get(middle - 1);
            } catch (Exception e) {
                sameConnectedComponent.put(middle - 1, sameConnectedComponent(middle - 1, queryNodes));
                sameConnectedComponentMiddleLess = sameConnectedComponent.get(middle - 1);
            }

            if (!sameConnectedComponentMiddle && !sameConnectedComponentMiddleLess) {
                middle = middle / 2;
            } else if (!sameConnectedComponentMiddle && sameConnectedComponentMiddleLess) {
                return middle - 1;
            } else if (sameConnectedComponentMiddle && (middle == (k - 1) || !sameConnectedComponentMiddleMore)) {
                return middle;
            } else if (sameConnectedComponentMiddle && middle < (k - 1) && sameConnectedComponentMiddleMore) {
                middle = middle + middle / 2;
            }
        }
    }

    private boolean sameConnectedComponent(int k, ArrayList<Integer> queryNodes) {
        int connectedComponentIndex = this.coreComposition.get(k).get(queryNodes.get(0));
        boolean sameConnectedComponent = false;
        for (int queryNode : queryNodes) {
            if (connectedComponentIndex == this.coreComposition.get(k).get(queryNode)) {
                sameConnectedComponent = true;
            } else {
                sameConnectedComponent = false;
                break;
            }
        }

        if (sameConnectedComponent) {
            return true;
        } else {
            return false;
        }
    }

    public HashSet<Integer> getNeighbors(int node, int coreIndex) {
        HashSet<Integer> neighbors = new HashSet<Integer>();

        for (int shellIndex : this.nodeNeighbors.get(node).keySet()) {
            if (shellIndex >= coreIndex) {
                neighbors.addAll(this.nodeNeighbors.get(node).get(shellIndex));
            }
        }

        return neighbors;
    }

    public HashSet<Integer> getNeighbors(int node, int coreIndex, HashSet<Integer> subcore) {
        HashSet<Integer> neighbors = new HashSet<Integer>();

        for (int shellIndex : this.nodeNeighbors.get(node).keySet()) {
            if (shellIndex >= coreIndex) {
                for(int x:this.nodeNeighbors.get(node).get(shellIndex))
                {
                    if(subcore.contains(x))
                    {
                        neighbors.add(x);
                    }
                }
            }
        }

        return neighbors;
    }
        
    public int getCoreMinimumDegree(int coreIndex) {
        return this.coreMinimumDegree.get(coreIndex);
    }

    public int getNumberOfNodes(int coreIndex) {
        return this.coreComposition.get(coreIndex).size();
    }
    
    public Set<Integer> getNodes() {
        return this.coreIndex.keySet();
    }
    
    public HashSet<Integer> getCore(ArrayList<Integer> queryNodes) {
        int minCoreIndex = getMinimumCoreIndex(queryNodes);
        HashSet<Integer> core = new HashSet<Integer>();
        int cc = this.coreComposition.get(minCoreIndex).get(queryNodes.get(0));
        core.addAll(this.coreConnectedComponents.get(minCoreIndex).get(cc));
        return core;
    }
    
}
