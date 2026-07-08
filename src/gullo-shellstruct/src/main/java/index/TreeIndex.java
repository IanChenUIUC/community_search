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
import java.util.LinkedList;
import java.util.Queue;
import main.Graph;

public class TreeIndex implements IndexInterface {

    // The core index for each node in the graph
    HashMap<Integer, Integer> coreIndex;

    // For each core, its minimum degree
    private HashMap<Integer, Integer> coreMinimumDegree = new HashMap<Integer, Integer>();

    // For each core, for each connected component its parent
    private HashMap<Integer, HashMap<Integer, Integer>> connectedComponentParent;
    // For each core, for each connected component its children
    private HashMap<Integer, HashMap<Integer, HashSet<Integer>>> connectedComponentChildren;
    // For each core, for each connected component its new nodes
    private HashMap<Integer, HashMap<Integer, HashSet<Integer>>> connectedComponentNodes;

    // For each node, its connected component in its shell
    private HashMap<Integer, Integer> nodeGroup;
    // For each node, for each shell the neighbors of the node
    private HashMap<Integer, HashMap<Integer, HashSet<Integer>>> nodeNeighbors;

    public TreeIndex(Graph graph, String datasetName, String outputFile) {
        long startTime = System.currentTimeMillis();
        this.computeCoreIndex(graph);

        this.computeCoreComposition(graph);

        this.computeNodeNeighbors(graph);
        long endTime = System.currentTimeMillis();
        System.out.println("TreeIndex execution time: " + (endTime - startTime));
        
        this.serializeToFile(outputFile);
    }

    @SuppressWarnings("unchecked")
    public TreeIndex(String path) {
        this.coreIndex = new HashMap<Integer, Integer>();
        this.coreMinimumDegree = new HashMap<Integer, Integer>();
        this.connectedComponentChildren = new HashMap<Integer, HashMap<Integer, HashSet<Integer>>>();
        this.connectedComponentNodes = new HashMap<Integer, HashMap<Integer, HashSet<Integer>>>();
        this.connectedComponentParent = new HashMap<Integer, HashMap<Integer, Integer>>();
        this.nodeGroup = new HashMap<Integer, Integer>();
        this.nodeNeighbors = new HashMap<Integer, HashMap<Integer, HashSet<Integer>>>();

        try {
            Kryo kryo = new Kryo();
            FileInputStream fileInputStream = new FileInputStream(path + ".bin");
            UnsafeMemoryInput input = new UnsafeMemoryInput(fileInputStream);

            this.coreIndex = kryo.readObject(input, this.coreIndex.getClass());
            this.coreMinimumDegree = kryo.readObject(input, this.coreMinimumDegree.getClass());
            this.connectedComponentChildren = kryo.readObject(input, this.connectedComponentChildren.getClass());
            this.connectedComponentNodes = kryo.readObject(input, this.connectedComponentNodes.getClass());
            this.connectedComponentParent = kryo.readObject(input, this.connectedComponentParent.getClass());
            this.nodeGroup = kryo.readObject(input, this.nodeGroup.getClass());
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
    }

    private void computeCoreComposition(Graph graph) {
        this.connectedComponentChildren = new HashMap<Integer, HashMap<Integer, HashSet<Integer>>>();
        this.connectedComponentNodes = new HashMap<Integer, HashMap<Integer, HashSet<Integer>>>();
        this.connectedComponentParent = new HashMap<Integer, HashMap<Integer, Integer>>();

        this.nodeGroup = new HashMap<Integer, Integer>();
        HashMap<Integer, Integer> highestGroup = new HashMap<Integer, Integer>();

        // Add each node to the stack representing its core
        ArrayList<Stack<Integer>> coreGroup = new ArrayList<Stack<Integer>>();
        for (int i = 0; i < graph.getNumberOfNodes(); i++) {
            coreGroup.add(new Stack<Integer>());
        }
        for (int i : this.coreIndex.keySet()) {
            coreGroup.get(coreGroup.size() - this.coreIndex.get(i) - 1).push(i);
        }

        int connectedComponentIndex = 0; // The incremental ids for connected components

        // For each group
        int coreIndex = coreGroup.size() - 1;
        for (Stack<Integer> group : coreGroup) {
            if (!group.isEmpty()) {
                this.connectedComponentChildren.put(coreIndex, new HashMap<Integer, HashSet<Integer>>());
                this.connectedComponentNodes.put(coreIndex, new HashMap<Integer, HashSet<Integer>>());
                this.connectedComponentParent.put(coreIndex, new HashMap<Integer, Integer>());
            }

            // For each node in the group
            while (!group.isEmpty()) {
                int node = group.pop();

                // For all its neighbors
                for (int neighbor : graph.getNeighbors(node)) {
                    // If they are in the same shell, continue
                    if (this.coreIndex.get(node).intValue() == this.coreIndex.get(neighbor).intValue()) {
                        // If the neighbor isn't in the solution
                        if (!this.nodeGroup.containsKey(neighbor)) {
                            continue;
                        } // If the node doesn't have a group, add the node to the same group of the neighbor
                        else if (!this.nodeGroup.containsKey(node)) {
                            this.connectedComponentNodes.get(coreIndex).get(this.nodeGroup.get(neighbor)).add(node);

                            this.nodeGroup.put(node, this.nodeGroup.get(neighbor));
                            highestGroup.put(node, this.nodeGroup.get(neighbor));
                        } // If the node has a group different from the group of the neighbor, merge the two groups
                        else if (this.nodeGroup.get(node).intValue() != this.nodeGroup.get(neighbor).intValue()) {
                            int oldGroup = this.nodeGroup.get(neighbor);

                            this.connectedComponentNodes.get(coreIndex).get(this.nodeGroup.get(node)).addAll(this.connectedComponentNodes.get(coreIndex).get(oldGroup));
                            this.connectedComponentChildren.get(coreIndex).get(this.nodeGroup.get(node)).addAll(this.connectedComponentChildren.get(coreIndex).get(oldGroup));

                            for (int oldGroupNode : this.connectedComponentNodes.get(coreIndex).get(oldGroup)) {
                                this.nodeGroup.put(oldGroupNode, this.nodeGroup.get(node));
                            }

                            for (int oldHighestNode : highestGroup.keySet()) {
                                if (highestGroup.get(oldHighestNode) == oldGroup) {
                                    highestGroup.put(oldHighestNode, this.nodeGroup.get(node));
                                }
                            }

                            for (int newParentGroup : this.connectedComponentChildren.get(coreIndex).get(oldGroup)) {
                                this.connectedComponentParent.get(coreIndex + 1).put(newParentGroup, this.nodeGroup.get(node));
                            }

                            this.connectedComponentChildren.get(coreIndex).remove(oldGroup);
                            this.connectedComponentNodes.get(coreIndex).remove(oldGroup);
                            this.connectedComponentParent.get(coreIndex).remove(oldGroup);
                        }
                    } // Else, if they are in different shells
                    else {
                        // If the neighbor isn't in the solution continue
                        if (!this.nodeGroup.containsKey(neighbor)) {
                            continue;
                        } // If the node doesn't have a group
                        else if (!this.nodeGroup.containsKey(node)) {
                            // If the highest of the neighbor is in the last level
                            if (this.connectedComponentNodes.get(coreIndex).containsKey(highestGroup.get(neighbor))) {
                                this.connectedComponentNodes.get(coreIndex).get(highestGroup.get(neighbor)).add(node);

                                this.nodeGroup.put(node, highestGroup.get(neighbor));
                                highestGroup.put(node, highestGroup.get(neighbor));
                            } // If the highest of the neighbor is in the previous level
                            else if (this.connectedComponentNodes.get(coreIndex + 1).containsKey(highestGroup.get(neighbor))) {
                                this.connectedComponentNodes.get(coreIndex).put(connectedComponentIndex, new HashSet<Integer>());
                                this.connectedComponentNodes.get(coreIndex).get(connectedComponentIndex).add(node);
                                this.connectedComponentChildren.get(coreIndex).put(connectedComponentIndex, new HashSet<Integer>());
                                this.connectedComponentChildren.get(coreIndex).get(connectedComponentIndex).add(highestGroup.get(neighbor));

                                this.nodeGroup.put(node, connectedComponentIndex);
                                highestGroup.put(node, connectedComponentIndex);

                                this.connectedComponentParent.get(coreIndex + 1).put(highestGroup.get(neighbor), connectedComponentIndex);

                                int oldHighest = highestGroup.get(neighbor);
                                for (int oldHighestNode : highestGroup.keySet()) {
                                    if (highestGroup.get(oldHighestNode) == oldHighest) {
                                        highestGroup.put(oldHighestNode, connectedComponentIndex);
                                    }
                                }

                                connectedComponentIndex++;
                            } // If the highest of the neighbor isn't in the previous level neither
                            else {
                                int highestLevel = coreIndex + 2;
                                while (!this.connectedComponentNodes.get(highestLevel).containsKey(highestGroup.get(neighbor))) {
                                    highestLevel++;
                                }
                                while (!this.connectedComponentNodes.get(coreIndex + 1).containsKey(highestGroup.get(neighbor))) {
                                    this.connectedComponentNodes.get(highestLevel - 1).put(connectedComponentIndex, new HashSet<Integer>());
                                    this.connectedComponentChildren.get(coreIndex).put(connectedComponentIndex, new HashSet<Integer>());
                                    this.connectedComponentChildren.get(coreIndex).get(connectedComponentIndex).add(highestGroup.get(neighbor));

                                    this.connectedComponentParent.get(highestLevel).put(highestGroup.get(neighbor), connectedComponentIndex);

                                    int oldHighest = highestGroup.get(neighbor);
                                    for (int oldHighestNode : highestGroup.keySet()) {
                                        if (highestGroup.get(oldHighestNode) == oldHighest) {
                                            highestGroup.put(oldHighestNode, connectedComponentIndex);
                                        }
                                    }

                                    highestLevel--;
                                    connectedComponentIndex++;
                                }

                                this.connectedComponentNodes.get(coreIndex).put(connectedComponentIndex, new HashSet<Integer>());
                                this.connectedComponentNodes.get(coreIndex).get(connectedComponentIndex).add(node);
                                this.connectedComponentChildren.get(coreIndex).put(connectedComponentIndex, new HashSet<Integer>());
                                this.connectedComponentChildren.get(coreIndex).get(connectedComponentIndex).add(highestGroup.get(neighbor));

                                this.nodeGroup.put(node, connectedComponentIndex);
                                highestGroup.put(node, connectedComponentIndex);

                                this.connectedComponentParent.get(coreIndex + 1).put(highestGroup.get(neighbor), connectedComponentIndex);

                                int oldHighest = highestGroup.get(neighbor);
                                for (int oldHighestNode : highestGroup.keySet()) {
                                    if (highestGroup.get(oldHighestNode) == oldHighest) {
                                        highestGroup.put(oldHighestNode, connectedComponentIndex);
                                    }
                                }

                                connectedComponentIndex++;
                            }
                        } // If the node has a group different from the group of the neighbor
                        else if (this.nodeGroup.get(node).intValue() != highestGroup.get(neighbor).intValue()) {
                            // If the highest of the neighbor is in the last level and it's different from the group of the node
                            if (this.connectedComponentNodes.get(coreIndex).containsKey(highestGroup.get(neighbor)) && this.nodeGroup.get(node).intValue() != highestGroup.get(neighbor).intValue()) {
                                int oldGroup = highestGroup.get(neighbor);

                                this.connectedComponentNodes.get(coreIndex).get(this.nodeGroup.get(node)).addAll(this.connectedComponentNodes.get(coreIndex).get(oldGroup));
                                this.connectedComponentChildren.get(coreIndex).get(this.nodeGroup.get(node)).addAll(this.connectedComponentChildren.get(coreIndex).get(oldGroup));

                                for (int oldGroupNode : this.connectedComponentNodes.get(coreIndex).get(oldGroup)) {
                                    this.nodeGroup.put(oldGroupNode, this.nodeGroup.get(node));
                                }

                                for (int oldHighestNode : highestGroup.keySet()) {
                                    if (highestGroup.get(oldHighestNode) == oldGroup) {
                                        highestGroup.put(oldHighestNode, this.nodeGroup.get(node));
                                    }
                                }

                                for (int newParentGroup : this.connectedComponentChildren.get(coreIndex).get(oldGroup)) {
                                    this.connectedComponentParent.get(coreIndex + 1).put(newParentGroup, this.nodeGroup.get(node));
                                }

                                this.connectedComponentNodes.get(coreIndex).remove(oldGroup);
                                this.connectedComponentChildren.get(coreIndex).remove(oldGroup);
                                this.connectedComponentParent.get(coreIndex).remove(oldGroup);
                            } // If the highest of the neighbor is in the previous level
                            else if (this.connectedComponentNodes.get(coreIndex + 1).containsKey(highestGroup.get(neighbor))) {
                                this.connectedComponentChildren.get(coreIndex).get(this.nodeGroup.get(node)).add(highestGroup.get(neighbor));
                                this.connectedComponentParent.get(coreIndex + 1).put(highestGroup.get(neighbor), this.nodeGroup.get(node));

                                int oldHighest = highestGroup.get(neighbor);
                                for (int oldHighestNode : highestGroup.keySet()) {
                                    if (highestGroup.get(oldHighestNode) == oldHighest) {
                                        highestGroup.put(oldHighestNode, this.nodeGroup.get(node));
                                    }
                                }
                            } // If the highest of the neighbor isn't in the previous level neither
                            else {
                                int highestLevel = coreIndex + 2;
                                while (!this.connectedComponentNodes.get(highestLevel).containsKey(highestGroup.get(neighbor))) {
                                    highestLevel++;
                                }
                                while (!this.connectedComponentNodes.get(coreIndex + 1).containsKey(highestGroup.get(neighbor))) {
                                    this.connectedComponentNodes.get(highestLevel - 1).put(connectedComponentIndex, new HashSet<Integer>());
                                    this.connectedComponentChildren.get(coreIndex).put(connectedComponentIndex, new HashSet<Integer>());
                                    this.connectedComponentChildren.get(coreIndex).get(connectedComponentIndex).add(highestGroup.get(neighbor));

                                    this.connectedComponentParent.get(highestLevel).put(highestGroup.get(neighbor), connectedComponentIndex);

                                    int oldHighest = highestGroup.get(neighbor);
                                    for (int oldHighestNode : highestGroup.keySet()) {
                                        if (highestGroup.get(oldHighestNode) == oldHighest) {
                                            highestGroup.put(oldHighestNode, connectedComponentIndex);
                                        }
                                    }

                                    highestLevel--;
                                    connectedComponentIndex++;
                                }

                                this.connectedComponentChildren.get(coreIndex).get(this.nodeGroup.get(node)).add(highestGroup.get(neighbor));
                                this.connectedComponentParent.get(coreIndex + 1).put(highestGroup.get(neighbor), this.nodeGroup.get(node));

                                int oldHighest = highestGroup.get(neighbor);
                                for (int oldHighestNode : highestGroup.keySet()) {
                                    if (highestGroup.get(oldHighestNode) == oldHighest) {
                                        highestGroup.put(oldHighestNode, this.nodeGroup.get(node));
                                    }
                                }
                            }
                        }
                    }
                }
                // If it has no neighbors
                if (!this.nodeGroup.containsKey(node)) {
                    // Create a new group
                    this.connectedComponentNodes.get(coreIndex).put(connectedComponentIndex, new HashSet<Integer>());
                    this.connectedComponentNodes.get(coreIndex).get(connectedComponentIndex).add(node);
                    this.connectedComponentChildren.get(coreIndex).put(connectedComponentIndex, new HashSet<Integer>());

                    this.nodeGroup.put(node, connectedComponentIndex);
                    highestGroup.put(node, connectedComponentIndex);

                    connectedComponentIndex++;
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
            kryo.writeObject(output, this.connectedComponentChildren);
            output.flush();
            kryo.writeObject(output, this.connectedComponentNodes);
            output.flush();
            kryo.writeObject(output, this.connectedComponentParent);
            output.flush();
            kryo.writeObject(output, this.nodeGroup);
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
        int k = Collections.max(queryIndexes);

        HashSet<Integer> Q = new HashSet<Integer>(queryNodes);
        HashSet<Integer> C = new HashSet<Integer>();
        for (int queryNode : queryNodes) {
            if (this.coreIndex.get(queryNode) == k) {
                C.add(this.nodeGroup.get(queryNode));
                Q.remove(queryNode);
            }
        }

        while (Q.size() != 0 || C.size() != 1) {
            HashSet<Integer> cFirst = new HashSet<Integer>();
            for (int connectedComponentID : C) {
                cFirst.add(this.connectedComponentParent.get(k).get(connectedComponentID));
            }

            for (int queryNode : queryNodes) {
                if (this.coreIndex.get(queryNode) == k - 1) {
                    cFirst.add(this.nodeGroup.get(queryNode));
                    Q.remove(queryNode);
                }
            }

            C = new HashSet<Integer>(cFirst);
            k--;
        }

        return k;
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
        int currentCoreIndex = coreIndex;
        HashSet<Integer> currentConnectedComponents = new HashSet<Integer>(this.connectedComponentNodes.get(currentCoreIndex).keySet());

        int numberOfNodes = 0;
        while (!currentConnectedComponents.isEmpty()) {
            HashSet<Integer> childrenConnectedComponents = new HashSet<Integer>();
            for (int connectedComponent : currentConnectedComponents) {
                numberOfNodes = numberOfNodes + this.connectedComponentNodes.get(currentCoreIndex).get(connectedComponent).size();
                if (this.connectedComponentChildren.get(currentCoreIndex).containsKey(connectedComponent)) {
                    childrenConnectedComponents.addAll(this.connectedComponentChildren.get(currentCoreIndex).get(connectedComponent));
                }
            }

            currentConnectedComponents = childrenConnectedComponents;
            currentCoreIndex++;
        }

        return numberOfNodes;
    }
    
    /*
    public HashSet<Integer> getCore(ArrayList<Integer> queryNodes)
    {
        int minCoreIndex = getMinimumCoreIndex(queryNodes);
        
        HashSet<Integer> core = new HashSet<Integer>();
        int currentCoreIndex = minCoreIndex;
        int cc = -1;
        for(int q:queryNodes)
        {
            if(this.nodeGroup.containsKey(q))
            {
                cc = this.nodeGroup.get(q);
            }
        }
        HashSet<Integer> currentConnectedComponents = new HashSet<Integer>();
        currentConnectedComponents.add(cc);

        //int numberOfNodes = 0;
        while (!currentConnectedComponents.isEmpty()) {
            HashSet<Integer> childrenConnectedComponents = new HashSet<Integer>();
            for (int connectedComponent : currentConnectedComponents) {
                //numberOfNodes = numberOfNodes + this.connectedComponentNodes.get(currentCoreIndex).get(connectedComponent).size();
                if(this.connectedComponentNodes.containsKey(currentCoreIndex))
                {
                    if(this.connectedComponentNodes.get(currentCoreIndex).containsKey(connectedComponent))
                    {
                        core.addAll(this.connectedComponentNodes.get(currentCoreIndex).get(connectedComponent));
                    }
                    //else
                    //{
                        childrenConnectedComponents.add(connectedComponent);
                    //}
                    if (this.connectedComponentChildren.get(currentCoreIndex).containsKey(connectedComponent)) {
                        childrenConnectedComponents.addAll(this.connectedComponentChildren.get(currentCoreIndex).get(connectedComponent));
                    }
                }
            }

            currentConnectedComponents = childrenConnectedComponents;
            currentCoreIndex++;
        }

        return core;
        
    }
    */
    
    public HashSet<Integer> getCore(ArrayList<Integer> queryNodes)
    {
        HashSet<Integer> core = new HashSet<Integer>();
        HashSet<Integer> queryIndexes = new HashSet<Integer>();
        for (int queryNode : queryNodes) {
            queryIndexes.add(this.coreIndex.get(queryNode));
        }
        int k = Collections.max(queryIndexes);

        HashSet<Integer> Q = new HashSet<Integer>(queryNodes);
        HashSet<Integer> C = new HashSet<Integer>();
        for (int queryNode : queryNodes) {
            if (this.coreIndex.get(queryNode) == k) {
                int cc = this.nodeGroup.get(queryNode);
                C.add(cc);
                Q.remove(queryNode);
            }
        }

        while (Q.size() != 0 || C.size() != 1) {
            HashSet<Integer> cFirst = new HashSet<Integer>();
            for (int connectedComponentID : C) {
                int parentID = this.connectedComponentParent.get(k).get(connectedComponentID);
                cFirst.add(parentID);
            }

            for (int queryNode : queryNodes) {
                if (this.coreIndex.get(queryNode) == k - 1) {
                    int cc = this.nodeGroup.get(queryNode);
                    cFirst.add(cc);
                    Q.remove(queryNode);
                }
            }

            C = new HashSet<Integer>(cFirst);
            k--;
        }
        

        while(this.connectedComponentChildren.containsKey(k))
        {
            HashSet<Integer> newC = new HashSet<Integer>();
            for(int cc:C)
            {
                newC.add(cc);
                if(this.connectedComponentChildren.get(k).containsKey(cc))
                {
                    for(int ccc:this.connectedComponentChildren.get(k).get(cc))
                    {
                        newC.add(ccc);
                    }
                }
                if(this.connectedComponentNodes.get(k).containsKey(cc))
                {
                    core.addAll(this.connectedComponentNodes.get(k).get(cc));
                }
            }
            C = newC;
            k++;
        }
        
        core.addAll(queryNodes);
        
        return core;
    }
    
    
    /*
    public HashSet<Integer> getCore(ArrayList<Integer> queryNodes)
    {
        int minCoreIndex = getMinimumCoreIndex(queryNodes);
        
        HashSet<Integer> core = new HashSet<Integer>();
        Queue<Integer> q = new LinkedList<Integer>();
        for(int x:queryNodes)
        {
            q.add(x);
        }
        while (!q.isEmpty())
        {
            int x = q.poll();
            core.add(x);
            HashSet<Integer> neighbors = this.getNeighbors(x, minCoreIndex);
            for(int y:neighbors)
            {
                if(!core.contains(y))
                {
                    q.add(y);
                }
            }
        }

        return core;
        
    }
    */

    public Set<Integer> getNodes() {
        return this.coreIndex.keySet();
    }
    
}
