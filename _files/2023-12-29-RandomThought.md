---
title: Some random thoughts on graph
layout: category
permalink: /categories/Random-thoughts/
taxonomy: Some random thoughts on graph
---

# Some random thoughts on graph

In the next coming year, what will be the future for the graph domain and why we need graph domain?



**What is the advantage of a graph?**

Graph can 

- represents irregular data other than NLP and sequence. It can help to represent the molecule and social network
- represents complicated and abstract knowledge



For the graph domain, the essential thing is how to find the graph structure, for instance, Neural Network can be a graph. Nonetheless, how can we define the graph results in different properties. The Neural Network can be viewed as a graph, e.g., the DAG, computation graph can be one, nonetheless, it is less informative. How to define a better graph as the network structure can be a good direction (recent ICLR has two good submissions)



The graph can have different different usage, especially for system II intellegence, here are a few examples:

- In CV, the graph can be utilized as the scen graph and 
- In Neural network explanation, it can be described as the basic unit for the 
- Graph is also connected with the comibinational optimization.
- In NLP, graph can be utilized to model the theory of mind, the compositional generalization (tree structure)



**Why GNN can work?**

- Why we use GNN?
  In one specific domain, there are particular heuristic methods can work well. Most heuristics (inductive bias) is super strong. Then it becomes unnecessary to have new design on GNN, we just aim to build a differentiable Neural Network (can be called GNN)
  **Instead of using GNN as an independent research direction, you can view GNN as a tool to make heuristic differentiable.** For instance, CN in link prediction to NeoGNN. 
- Why GNN is better than other graph modeling methods? 
  The key advantage is that it can adaptive model the feature and structure information together. For heuristic methods, most of them can only model the structure information, while the feature information is ignored.   (some people will also use position encoding)



**Why we need graph?**

- The generalization on structure is different with the generalization on the feature. If we know the reasoning rule, it is easily to generalize to the higher order case with recursively. Feature is easily to be greedy and get stuck into the spurious feaure correlation. Nonetheless, structure is not easy to learn. 
- You can always meet OOD in the graph scenario, as you consider 1, 2, 3, 4, 5, 6 order, the number of the samples will grow exponentially. Simple training on the dataset can not cover all of them.  
- We can easily find the required geometric information on structure and then develop corresponding inductive bias with MPNN on the structure data.   (also more explanable) 



**How to build a meaningful graph from a non-graph data?**



**How to remove the structure from the graph data?**











