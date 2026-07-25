# About

We are analyzing the community structure and evolution of a simulated network, dubbed CANTATAS.
The details about the simulation can be found on [zenodo](https://zenodo.org/records/21513973).

## Community Evolution

The simulator starts from a seed network, which is sampled from the Dimensions database as of 2026.
Each node in this seed is a part of one or multiple fields of mathematics, physics, chemistry, biology, or scientometrics, with a special "founder" node for each field.
These founder nodes are our five distinct queries.

The simulator then evolves the network from the initial 2026, to 2146 (adding 120 new years).
For each of these years, we collect statistics for 5 communities, seeded at each of the founder nodes.
Namely,

- we analyze the distribution of fields of the communities
- we study the quality of each community, namely the size, minimum core degree, and separation (conductance).

---

## Pre-processing

Our inputs are:

- `input/abm272_multifield` network, as an edgelist as a parquet format, with a corresponding nodelist
- `input/tc_combined_nodelist.csv`, which describes in detail the properties of the seeds

Each of these are additionally symlinked to their location in the UIUC campus cluster filesystem.

The main preprocessing steps are to:

1. for each year of 2026, 2027, ..., 2146, create the induced subgraph of nodes present in that year
2. for each agent node, identify the fields that it is a part of

To do the former, we make use of the [format-conversion](https://github.com/IanChenUIUC/format-conversion) package, as sourced in the `pyproject.toml`.
Each subnetwork stores the same integer ids as in the final network.

To identify the fields of the agent nodes, we simply inherit the fields of its generator, and a simple DP works.
We give the following codes for each field:

0. math
1. physics
2. chemistry
3. bio
4. scientometrics
5. math|physics
6. math|chemistry
7. math|bio
8. math|scientometrics
9. physics|chemistry
10. physics|bio
11. physics|scientometrics
12. chemistry|bio
13. chemistry|scientometrics
14. bio|scientometrics

A node with a single field gets that field's code (0–4); an interdisciplinary node with two
fields gets a pair code (5–14). No node has more than two fields.
