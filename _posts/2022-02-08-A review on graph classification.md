---
title: A review on graph classification
layout: category
permalink: /categories/A review on graph classification/
taxonomy: A review on graph classification



---
# A review on Graph Classification

Graph Classification is a traditional task in Graph Domain. However, it is hard to tell which method is the state-of-the-art in this domain. Meanwhile, reviewing on Graph Classification task is also a wonderful journey to see how GNN grows from CNN with many meaningful attempts. In this blog, we are glad to give an introduction on this familar but strange topic: graph classification after a literature review on more than 30 papers.

In this review, we aim to answer the following questions:

- How does graph classification borrows ideas from Image Classification？
- How to add pooling operation for a good graph representation? (A cluster perpective borrowed from CV) 
- How to find good aggregation function for a good graph representation? (A graph isomorphism perspective from WL-test)
- Occam's Razor: Rethinking on the neccerarities on advance operations

Additional we will talk about:

- Interesting traditional Graph Kernel methods
- OOD (Out of Distribution) problem on Graph Classification (Yiqi will be invited for this part)





## How does graph classification borrows ideas from Image Classification?

It is not hard to think about the relationship between graph classification and Image classification. Image can be viewed as a specific graph which each node is a pixel which has three features on RGB, and the edge structure is more like grid. It seems natural to generalize the idea from image to graph. As most graphs have the similar properties like image which are locality, stationarity, and composionality. 

CNN has followining advantages:

- Comparing with methods from spectral domain which rely on fixed spectrum of graph Laplacian for single structure, CNN can **handle graphs with varying size and connectivity**
- Comparing with differentiable Neural Graph Fingerprint (another family of method inspired by Fingerprint), feature are not directed sum up localized vertex features, weights in CNN ensure the power to filter the unimportant features.
- Comparing with traditional Graph Kernel methods, the time complexity comes down from quadratic to linear on the number of nodes



However, the key challenge for applyiong CNN on graph data is **Fix order**. Images and sentences have their order, which is of great significance. For example,  image with or without order has great differences! 

![](https://pic2.zhimg.com/80/v2-183ec0fb1bfb8dd25e4a71be0425eb30_1440w.png)

So what needed to be solve is that: how to find the order in graph structure. More specific, how to find local receptive field and node order in the local receptive.

An example to give a closer look on what's connection. 

![](https://pica.zhimg.com/80/v2-98fcc4929f25bfcccb9272f80b9d22d8_1440w.png)

An image can be represented as a square grid graph whose nodes represent pixels. A CNN can be seen as traversing a node sequence (The red node) and **generating fixed-size neighborhood graphs with certain order**. What needed to be solve is how to transvere and generate the neighbor with order.

To solve this problem, we introduce three methods with genius design:

- PatchSAN: node label for extracting locally connected regions from graph, normalized neighborhood for feature compress
- ECC: 
- DGCNN: global graph topology for node sorting

We will detailed these algorithms in the following part.



### PATCHY-SAN (PSCN): Learning Convolutional Neural Networks for Graphs

The key idea in PATCHY-SAN is how to extract locally connected regions from graphs which serves as the receptive fields of a convolutional architecture. Revolving on this goal, two steps are constructed:

- Determine the node sequences for which $k$ nodes are selected into neighbor graph with fixed order.
- Normalize a graph: turn a graph representation into a vectotar representation.

The corresponding challenge are:

- How to determine who is neighbor node similarly with the physic position in image
- How to make unique mapping that nodes with similar structural roles are positioned similar in the vector representation



**Node labeling** aims to learn a funciton $\mathcal{l}: V \to S$ from vertices to **ordered** set. WL-test is one of the typical labeling technique, which is also injective (A unique adjacency matrix is given). Each node is mapped into a label, nodes with the same label are in the same set.

**Neighbor Determining：** First select the top $w$ elements as candidate, select neighbor (determined by the node labeling) from a sequence which generate from the center node (red node in CNN).  **Here has no order**

**Graph Normalize:** normalizing the neighborhood assembled. The target is that graph distance in the origin space can be reconstructed as much as possible in the feature space.

$$
\hat{\ell}=\underset{\ell}{\arg \min } \mathbb{E}_{\mathcal{G}}\left[\left|\mathbf{d}_{\mathbf{A}}\left(\mathbf{A}^{\ell}(G), \mathbf{A}^{\ell}\left(G^{\prime}\right)\right)-\mathbf{d}_{\mathbf{G}}\left(G, G^{\prime}\right)\right|\right] 
$$

where $A^l$ is the labeling procedure.

So the label is just the inverse of the rank for $d(u,v) < d(w,v) \to r(u) < r(w) \to \mathcal{l}(u) > \mathcal{l}(v)$. NAUTY is used as the labeling method, which accepts prior node partitions as input and breaks remaining ties by choosing the lexicographically maximal adjacency matrix.

**In a nutshell, just labeling, select node and order according to labeling**



### DGCNN: An End-to-End Deep Learning Architecture for Graph Classification

Yet, the above method still has much merits:

- Lack of the understanding graph from the node feature.
- Still use the traditional node labeling technique for sorting which is not only time-cosuming, but also lack of global topology view.

Then our question is: how to take advantage of the great deep learning power to find the receptive field (**non-local with global topology**) and determine the order.  The key idea is a deep learning version **WL test**. The steps are as following:

**To extract node feature and structure information**,  a diffusion based graph convolution layer:
$$
Z = f(\tilde{D}^{-1}\tilde{A}XW)
$$
where $\tilde{A} = A + I$, `$\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$`. <span>$f()$</span> is the tanh activation function Notice that view $Y = XW$，the procedure is similar to 1-WL test. Then $Z$ can be viewed as a WL signature vector. The non-linear function can be viewed as mapping to the new color.

**To find the order of nodes**: sortpooling is proposed. Notice that the above GNN can be viewed as a WL-test procedure. **It is also a labeling procedure!** The continuous WL color $Z^t$ is used for sorting. 

Sort principle: using the last channel of $Z^h$ in a descending order, if the first channel is the same, then compare the second one.

Then, unimportant node is dropped directly, and only $k$ nodes are remained.

**After sorting,** we have node representation with sorted. 1-D conv and MLP is applied on it.

**The key difference between DGCNN and PATCHYSAN** is that PATCHYSAN only use the traditional graph label, but DGCNN use the multi-dimension GNN. Additionally, sortpooling can drop some nodes which are less informative.



Overall speaking, the above methods focus on how to sort and find the order in graph, in order to apply CNN on top of it. However, the order in graph is somehow difficult to find correctly, it may not be a good idea to reused the CNN components in Graph domain. 

Then an interesting topic has been proposed: How to change the CNN component in CV into a graph version. CNN is easy to correspond to GNN. Then what is the corresponding to the popular pooling component?

## How to add pooling operation for a good graph representation? 

As mentioned above, it seems nature to design pooling specfic for graph for 

- General GNNs are inherently flat, which do not learn representation for a group of nodes. 
- The common pooling (readout function) is always global view, i.e. sum, mean all node features, which lost much structure information. The complex topological structure of graphs precludes any straightforward

Can we have local pooling like CV, following a structure: CNN-pooling-CNN-pooling-(flatten)-MLP classifier like.

![](https://pica.zhimg.com/80/v2-b10e63ea110d4971d3d87ad9e76021d8_720w.png)

To design **local** graph pooling, following chanlleges should be solved:

- What is the local patch that should be pooled?  clustering or downsampling.
- What will the graph be like after pooling for the node number becomes smaller than before? What is the node feature and new graph structure?  selection or aggregation.

To solve them challenge, various methods is proposed which can be roughly categorized into:

- parameter-free clustering pooling (chanllege 1)  including cliquePooling, GRACLUS Pooling
- model-based pooling (chanllege 1)
  - Selection pooling (challenge2)
    - TopKpooling, SAGPooling, HGP
  - Cluster pooling (challenge2)
    - spatial persepective including diffpooling
    - spectral perspective including Eigenpooling, mincut pooling

Paremeter-free clustering pooling uses the traditional machine learning method to pre-compute the cluster matrix for pooling based on graph-theoretical property. However, it neither consider the node feature nor adapts to specific task. We will not discuss in details here.   

Here we focus on the end-to-end method model-based pooling and the second question: What will the graph like after pooling for the node number becomes smaller than before?

### Cluster pooling 

Cluster pooling cluster similar nodes into one super node node by exploiting their hierarchical structure. 

It shares the following steps:

- Predefine the cluster number $n^{l+1}$ (the node number after pooling) 
- Learn an assign matrix $S \in \mathbb{R}^{n_l\times n_{l+1}}$ ）Softly divide origin graph into subgraphs. **However, in this case,the assign matrix is a computational heavy dense matrix**
- Coarsen graph: Update node feature in the same cluster into one supernode $X^{(l+1)} = S^{(l)^T} Z^{(l)}$  
- Update adjacent matrix $A^{(l+1)} = S^{(l)^T} A^{(l)}S^{(l)}$

The key challenge in cluster pooling is how to learn an informative cluster matrix. To achieve this goal, understanding from spatial and spectral (spectral cluster) perspective are proposed.

We will first introduce the spatial ones. 

#### Diffpool: Hierarchical Graph Representation Learning with Differentiable Pooling

Diffpool is the first paper on hierarchical graph pooling and define the above learning scheme.  

How to learn $S$ is simple and parallel with GNN model.

The hidden representation is learnt by 
$$
Z^{(l)}=\operatorname{GNN}_{l, \text { embed }}\left(A^{(l)}, X^{(l)}\right)
$$
and the assign matrix $S$ is learnt by
$$
S^{(l)}=softmax \left (\operatorname{GNN}_{l, \text { embed }}\left(A^{(l)}, X^{(l)}\right) \right)
$$
The output dimension is the predefine cluster numbers.  **Notice that Z is not the node hidden representation for classification, but parallel with the original GNN model with different learnable parameter.**

However,  $S^{(l)}$ is a large dense matrix with quadratic node size. where the former selection pooling will solve this problem.



#### STRUCTPOOL: STRUCTURED GRAPH POOLING VIA CONDITIONAL RANDOM FIELDS

It is hard to write this part for too many prior knowledge on CRF is needed, I will write about it later~

Then we will introduce from spectral perspective inspired from the spectral pooling.

#### EigenPooling: Graph Convolutional Networks with EigenPooling

In DiffPooling, $S$ is a global soft assignment matrix with global structural information. However, pooling should also taken local structural information (subgraph) into consideration.  It is hard to extract without the help of local spectral graph signal: 

- the subgraphs may contain different numbers of nodes, thus a fixed size pooling operator cannot work for all subgraphs
- the subgraphs could have very different structures, which may require different approaches to summarize the information for the supernode representation.

To extract this information, graph fourier transformation with Laplacian matrix is introduced with understanding in spectral domain. EigenPooling focuses on how to preserve the graph signal in pooling. 

**Assign matrix** is learnt from spectral cluster (we will introduce on the fundmental machine learning blog).  Unlike the soft assignment, the graph nodes are divided into different subgraphs. For each subgraph(cluster) $G^k$, it has an indicate matrix $C^k\in \mathbb{R}^{N\times N_k}$ where $N_k$ is the node number in this cluster.  $C^k[i,j]=1$ means node $i$ is the $j_{th}$ node on the $k$ cluster. 

**Update adjacent matrix**

The adjacent matrix is update by the following four steps to remain only inter-cluster connection

- Intra adjacent matrix by individual subgraphs. Only remain connection within subgraph $G^k$  
  $$
  A^k = (C^k)^TAC^k
  $$

- Then we concate intra adjacent matrix for each graph into a whole adjacent matrix as
  $$
  A_{int} = \sum_{k=1}^KC^kA^k(C^k)^T
  $$

- Then the inter adjacent matrix is the complement one as
  $$
  A_{ext} = A - A_{int}
  $$

- Finally it will generate the coarsen graph
  $$
  A_{coar} = S^TA_{ext}S
  $$

**Feature Update: ** Unlike directly update the feature, it first updates the graph signal. then the feature will update according to the new signal. It has following steps:

- extract subgraph signal (eigenvector) in spectral domain by matrix decomposition as $u_1^k, \cdots, u_{N_k}^k$ 
- Upsample the subgraph signal into whole graph signal **after clustering**: $\bar{u}_1^k = C^ku_l^k$. 
- Therefore, for each subgraph, we have signal $\Theta_l = [\bar{u}_1^k, \cdots, \bar{u}_1^k ] \in \mathbb{R}^{N\times K}$
- Trasfer feature according to the signal: $X_l = \Theta^T_lX$   **No need to transform all the graph signal here.** Just the important low signal is enough.



#### MinCutPooling: Spectral Clustering with Graph Neural Networks for Graph Pooling

Though eigenpooling is good to preserve the graph signal, it somehow suffers from the computational expensive for large graph comes from two ways:

- Use spectral clustering which needs Eigenvalue decomposition on Laplacian matrix
- Use Eigenvalue decomposition to extract the subgraph spectral signal

To avoid expensive eigendecomposition on spectral clustering, Mincut pooling design a continous relaxation of the normalized minCUT problem which trained end to end with GNN. The contributions are:

- formulate a continuous relaxation of the normalized minCUT problem can be optimized by GNN
- learns the solution taken node features into consideration

**Assignment matrix **is similar with diffpool with a parallel GNN learn for assign matrix
$$
\begin{aligned}
\overline{\mathbf{X}} &=\operatorname{GNN}\left(\mathbf{X}, \tilde{\mathbf{A}} ; \boldsymbol{\Theta}_{\mathrm{GNN}}\right) \\
\mathbf{S} &=\operatorname{SOFTMAX}\left(\operatorname{MLP}\left(\overline{\mathbf{X}} ; \boldsymbol{\Theta}_{\mathrm{MLP}}\right)\right)
\end{aligned}
$$
**The optimization object for minCUT ** is:
$$
\mathcal{L}_{u}=\mathcal{L}_{c}+\mathcal{L}_{o}=\underbrace{-\frac{\operatorname{Tr}\left(\mathbf{S}^{T} \tilde{\mathbf{A}} \mathbf{S}\right)}{\operatorname{Tr}\left(\mathbf{S}^{T} \tilde{\mathbf{D}} \mathbf{S}\right)}}_{\mathcal{L}_{c}}+\underbrace{\left\|\frac{\mathbf{S}^{T} \mathbf{S}}{\left\|\mathbf{S}^{T} \mathbf{S}\right\|_{F}}-\frac{\mathbf{I}_{K}}{\sqrt{K}}\right\|_{F}}_{\mathcal{L}_{o}}
$$
where $|| \cdot ||_F$ is the Frobenius norm $\tilde{D}$ is the normalized degree matrix.

$\mathcal{L}_{c}$ is a mincut solution, which will reach optimal when it just fits the $K$ component in graph. However, it may falls into trivial solution for assign all matrix into one cluster. 

Then $\mathcal{L}_{o}$ the regularization term is used for regularization, balance the number of nodes in one cluster.

![](https://pica.zhimg.com/80/v2-3f5ee8cabf7696706e09adcdd34a4e80_720w.png)

**Feature and adjacent matrix update** is similar with the main framework. Additionally, to make adjacent matrix with too much self-connection and encourage the internal connections. Self loop is removed as following.
$$
\hat{\mathbf{A}}=\mathbf{A}^{\text {pool }}-\mathbf{I}_{K} \operatorname{diag}\left(\mathbf{A}^{\text {pool }}\right) ; \quad \tilde{\mathbf{A}}^{\text {pool }}=\hat{\mathbf{D}}^{-\frac{1}{2}} \hat{\mathbf{A}} \hat{\mathbf{D}}^{-\frac{1}{2}}
$$


### Selection pooling

Selection pooling obtains a score of each node using information from graph convolutional layers, and then drop unnecessary nodes with lower scores at each pooling step.

It shares the following steps:

- Give each node a score,  score list $y\in \mathbb{R^{n \times 1}}$ 
- Select the top $k$ nodes.   $idx = rank(y,k)$
- New node feature is the selected node feature.  $X^{l+1} = X^l(idx, :)$
- New adjacent matrix is the edge between selected nodes.   $A^{l+1} = A^l(idx, idx)$ (If the adjacent matrix is too sparse, you can use $A^2$ to include two-hop neighborhood)

Pooling just selection the important nodes for a smaller graph which is not only easy to locate the node origin position on origin graph but also not computationally expensive. The assumption is that the selected nodes already have the neccerary information. 

The challenge in selection pooling is that:

- In step 1 and 2, it generates a discrete index list, which not be trainable.
- Selection is too simple which may lose essential feature and structure information

Revolving on these chanllenge, we will introduce the following papers.

#### TopKPooling: Graph U-Nets

This paper is not exactly about pooling, but learn from a famous architecture in CV: U-Net. In order to following U-Net, it is neccerary to design graph pooling and graph unpooling operatation.

**For graph pooling,** it follows the selection procedure.  To give each node a score, it projected nodes on a learnable vector.
$$
y = X^l\mathbf{p}^l/||\mathbf{p}^l||
$$
 To solve the trainable challenge, $y$ also serves as a gate for features.
$$
\tilde{y} = sigmoid(y(idx))
$$

$$
\tilde{X}^{l+1} = X^{l+1} \odot (\tilde{y}1_C^T)
$$

However, I think only use a linear projection for score is too simple, which will lead to select many nodes with similar feature.

**For graph unpooling**, it utilize the advantage of selection pooling: easy to track where the node originally is. 

The unpooling is just put the learned node feature on the original graph and other node feature set to 0
$$
X^{l+1} = distribute(0_{N\times C}, X^l,idx)
$$
The zero node feature will be fill by message passing in the following GCN layers.

The model architecture is as follows. Notice that a residual link on the same pooling and unpooling layer.

![](https://pic3.zhimg.com/80/v2-0ad61bb43c5a9ae7c1d6cb11ea0b7aac_720w.png)

#### SAGPooling: Self-Attention Graph Pooling

SAGPooling is quite similar with TopKPooling,  the differences are (1) model architecture: similar with diffpooling(2) scoring function. 

**For scoring function:**

The difference is that it puts the projection matrix before activation

Sortpooling layer can be written as 
$$
Y=\sigma\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} X \Theta_{a t t}\right)
$$
where $\Theta_{a t t}\in \mathbb{R}^{F\times 1}$ , the activation function is tanh

TopKpooling can be rewritten as:
$$
Y=\sigma(\sigma\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} X W\right) \Theta_{a t t} / ||\Theta_{a t t}||_2)
$$
where the activation function is sigmoid.

Other operation are same as TopKPool. 

Maybe this scoring is better than TopKPool, or the achitecture is better. 

#### HGP: Hierarchical Graph Pooling with Structure Learning

This paper focus on the problem that the second challenge selection is too simple which may lose essential feature and structure information. 

**For feature, HGP adaptively select node with both node features and graph topological.**

The scoring function is designed by that if node information contains by neighborhood. If it can be reconstructed by neighbor, it can be deleted with no information loss.
$$
Y =  ||(I_i^k-(D_i^k)^{-1}A_i^k)H_i^k||_1
$$
The more information remain after remove the information from neighbor, the more importance to preserve the node

**For structure, HGP refine the graph structure to preserve the key substructure**

With the selected feature and selected adjacant matrix, $H^{l+1} = H^l(idx, :)$  $A^{l+1} = A^l(idx, idx)$ 

The structural refine is a feature attention graph similar to GAT:
$$
\mathbf{E}_{i}^{k}(p, q)=\sigma\left(\overrightarrow{\mathbf{a}}\left[\mathbf{H}_{i}^{k}(p,:) \| \mathbf{H}_{i}^{k}(q,:)\right]^{\top}\right)+\lambda \cdot \mathbf{A}_{i}^{k}(p, q)
$$
where $\overrightarrow{\mathbf{a}}\in \mathbb{R}^{2d\times 1}$.

 Then it is sent to a **sparse softmax function**, to learn a sparse graph structure.

### Go beyond pooling: GMN with clustering (MEMORY-BASED GRAPH NETWORKS)

Similar with the graph pooling method, GMN also keeps in mind with the graph cluster. However, it joint  learn node representation and coarsen the graph without **new graph structure update but a fully connnected graph** which may lose much information. It uses the update query for find new structure and memory to maintain the refined graph information.

It mainly has the following procedure:

- Generate the original query $Q^0$ with MPNN (GAT) or simple MLP

- Generate the Key matrix $K \in \mathbb{R}^{n_{l+1} \times d_l}$  which is just the key cluster center. The cluster is computed by the student t-distribution for a particular query.
  $$
  C_{i, j}=\frac{\left(1+\left\|q_{i}-k_{j}\right\|^{2} / \tau\right)^{-\frac{\tau+1}{2}}}{\sum_{j^{\prime}}\left(1+\left\|q_{i}-k_{j^{\prime}}\right\|^{2} / \tau\right)^{-\frac{\tau+1}{2}}}
  $$
  Then the representation is combine and aggregated as  
  $$
  \mathbf{C}^{(l)}=\operatorname{softmax}\left(\Gamma_{\phi}\left(\|_{k=0}^{|h|} \mathbf{C}_{k}^{(l)}\right)\right) \in \mathbb{R}^{n_{l} \times n_{l+1}}
  $$
  where $\Gamma_{\phi}$ is a $[1\times 1]$ convolutional operator for reduction. 

- Value updated (new node feature)
  $$
  \mathbf{V}^{(l)}=\mathbf{C}^{(l) \top} \mathbf{Q}^{(l)} \in \mathbb{R}^{n_{l+1} \times d_{l}}
  $$

- Update the new query set (cluster center) 
  $$
  \mathbf{Q}^{(l+1)}=\sigma\left(\mathbf{V}^{(l)} \mathbf{W}\right) \in \mathbb{R}^{n_{l+1} \times d_{l+1}}
  $$



The graph structure is only utilized on the encoder, other is just like an iterative node feature clustering.

 



## How to find good readout function for a good graph representation?

In the last part, we mainly focus on how pooling can coarse the graph. However, pooling may also induce much noise and may not be so effectiveness.  

Another family methods focus on another key compoent in GNN, the readout function, for example Set2Set.  

One direction is how to extract graph feature effectively once (without refining node feature and adjacent matrix.

Another direction is how to go beyond the expression bound of 1-WL test. With greater expressive, it becomes possible to distuiguish more graphs.

We will not introduce the second one in this blog for it is another big problem in GNN with heavy math. I will write another blog later about it. 



### CapsGNN: CAPSULE GRAPH NEURAL NETWORK

The major readout function often takes the input in a scalar method (compute the numerical maximum element.) which lose much properties. It is better to encode in a vector method like Capsule. It can generate multiple embedding to capture properties from different perspectives,  routing preserves all the information from low-level capsules and routes them to the closest high-level capsules.

CapsGNN contains three key blocks:

- Basic node capsule extraction block: GNN
- High level graph capsule with: attention and dynamic routing
- generate class capsule for graphb classification.

The paper is about how to apply capsule network on graph structure data. 

The attention procedure is proposed for normalized the feature in different channels from different layers

![](https://pic1.zhimg.com/80/v2-102b990ff605386a1db9de17570a6c25_720w.png)

The rescale attention is like: 
$$
\operatorname{scaled}\left(\boldsymbol{s}_{(n, i)}\right)=\frac{F_{a t t n}\left(\tilde{\boldsymbol{s}_{n}}\right)_{i}}{\sum_{n} F_{a t t n}\left(\tilde{\boldsymbol{s}_{n}}\right)_{i}} \boldsymbol{s}_{(n, i)}
$$
And $F_{attn}$ is just two-layer MLP. 

**Calculate votes:**capsules of different nodes from the same channel share the transform matrix, which results in a set of votes. $V\in \mathbb{R}^{N\times C_{all}\times P \times d}$

Then dynamic routing mechaism for the final classification result. We will not detail the capsule network here for it is too complicated and is just about another topic.



### GMN: ACCURATE LEARNING OF GRAPH REPRESENTATIONS WITH GRAPH MULTISET POOLING

In this paper, it first formulates the graph pooling problem as a multiset encoding problem with auxiliary information about the graph structure.  Instead of using pooling for layers, it performs a single but strong single pooling both injectiveness and permutation invariance. Also, the method shows very good result on multiple tasks.

The method is self-attention style with an multi dimension query set.
$$
att(Q,K,V)=f(QK^T)V
$$
$Q\in \mathbb{R}^{n_1\times d_k}$ is not generated by the feature, but a learnable parameter matrix by default.

To utilize GNN into this framework, Graph Multi-head Attention is defined as:
$$
\operatorname{GMH}(\boldsymbol{Q}, \boldsymbol{H}, \boldsymbol{A})=\left[O_{1}, \ldots, O_{h}\right] \boldsymbol{W}^{O} ; \quad O_{i}=\operatorname{Att}\left(\boldsymbol{Q} \boldsymbol{W}_{i}^{Q}, \operatorname{GNN}_{i}^{K}(\boldsymbol{H}, \boldsymbol{A}), \operatorname{GNN}_{i}^{V}(\boldsymbol{H}, \boldsymbol{A})\right)
$$
$Q$ is a learnable parameter matrix, $K$ and $V$ are generated by two-layer GNN. 

Then the key component **Graph Multiset Pooling with Graph Multi-head Attention** is defined as: 
$$
\operatorname{GMPool}_{k}(\boldsymbol{H}, \boldsymbol{A})=\mathrm{LN}(\boldsymbol{Z}+\operatorname{rFF}(\boldsymbol{Z})) ; \quad \boldsymbol{Z}=\mathrm{LN}(\boldsymbol{S}+\operatorname{GMH}(\boldsymbol{S}, \boldsymbol{H}, \boldsymbol{A}))
$$
$k$ is the dimension of query size

Then remove the query for a simple self attention for inter-node relation:
$$
\operatorname{SelfAtt}(\boldsymbol{H})=\mathrm{LN}(\boldsymbol{Z}+\operatorname{rFF}(\boldsymbol{Z})) ; \quad \boldsymbol{Z}=\mathrm{LN}(\boldsymbol{H}+\mathrm{MH}(\boldsymbol{H}, \boldsymbol{H}, \boldsymbol{H}))
$$
The whole structure is:
$$
\text { Pooling }(\boldsymbol{H}, \boldsymbol{A})=\operatorname{GMPool}_{1}\left(\operatorname{SelfAtt}\left(\operatorname{GMPool}_{k}(\boldsymbol{H}, \boldsymbol{A})\right), \boldsymbol{A}^{\prime}\right)
$$
Here $A'$ is also a coarsening adjacent matrix. So somehow, it is still hierarchical.

**Theorem part to be added**

This paper somehow astonish me on the ability of attention. However, there are some query hyperparemater to turn. I am wondering, if this can also be applied to the selection pooling, will the performance be better also? What the power of the learnable query instead of self attention.







## Occam's Razor: Rethinking on the neccerarities on advance operations

With the development of graph classification for many years, advanced operation has been proposed. However, how they can really work well is still unknown. We have reason to doubt the effectiveness of these operation as in most case, the feature is only the one-hot node label or node degree. Not so much information is available.  it is important to understand which components of the increasingly complex methods are necessary or most effective.



### A simple yet effective baseline for non-attribute graph classification

In this paper, it proposes a really simple baseline with feature augumentation by neighborhood degree. 
The degree of neighborhood is defined as <span>$DN(v) = \{ degree(u|(u,v) \in E \}$</span>

The procedure is as follow

- Five feature types are conducted by (degree($v$), min(DN($v$)), max(DN($v$)), mean(DN($v$)), std(DN($v$))) 
- Performing either a histogram or an empirical distribution function(edf) operation, i.e, mapping all node feature into a histogram or an empirical distribution.
- SVM is used for classification

Since $DN(v)$ is the one-hop neighbor information. min max can be viewed as different readout function in GNN. This feature can be easily viewed as a simplified version for GNN. 

The remaining question are:

- Does capture complex graph structure really help the performance?
- How to utilize the exisiting features on graph?



### Are Powerful Graph Neural Nets Necessary? A Dissection on Graph Classification

This paper challenge two key component in GNN (1) non-linear graph filter (**Notice that **)  (2) Readout function. 

It aims to answer the following questions:

- Do we need a sophisticated graph filtering function for a particular task or dataset?
- And if we have a powerful set function, is it enough to use a simple graph filtering function?

**Model without non-linear graph filter** GFN

Instead of non-linear aggregation, the aggregated feature is used as data argumentation.
$$
X^G = [d, X, \tilde{A}^1X, \tilde{A}^2X, \cdots, \tilde{A}^KX]
$$
Then the readout function is:
$$
\operatorname{GFN}(G, X)=\rho\left(\sum_{v \in \mathcal{V}} \phi\left(X_{v}^{G}\right)\right)
$$
where $\rho$ and $\phi$ are all MLP where $\phi$ is always with more layers. 

**In fair comparison, similar baseline has been proposed for social data**, which even with no need on aggregation feature, it can also beat the stoa GNN model.  

Additionally, first sum aggregate all features then applies an MLP with relu for classification can also result in good result in biochemical dataset. 

GFN can shows good generalization ability while with less overfitting on the training data.

**Model without both non-linear graph filter and non-linear readout function**

However, experiment is lack here with less experiment on only with non-linear graph filter. There is no wonder that a linear GNN cannot perform a good performance.



### Rethinking pooling in graph neural networks

This paper is among the most superising one for its check the essential property for hierarchical pooling: **Is cluster a must-have property before measuring how well the assign matrix is?**  (Somehow, this paper ignore the select pooling and it hard to say what does his local mean in this paper).

The surprising finding are: 

- Even with the complementary for the assign matrix, comparable result can also be achieved.
- GNN can learn a smooth and homophilious node representation which makes pooling with hierarchical be less important.
- Simple GraphSAGE can also achieve good results.

The main experiment perspectives are as follows:  For off-the-shelf graph clustering method, replace the adjacent matrix by its complementary set for clustering. For pooling methods, the main focus is on cluster assignment matrix $S$. The clustering assignment matrix can be replaced by normalized random matrix. If the clustering method is distance-based, assign node to the farthest distance instead of the closest. Surprisingly, these variants can also show comparable performance on some datasets.

After analysis, the main reason comes from: nodes display similar activation patterns before pooling for GNN can extract low-frequency information across the graph. Therefore, what pooling do is just build a more similar representation.

Moreover, challenges are still for homogeneous node representations even before the first pooling layer. (Does it depends on different situation?) This naturally poses a challenge for the upcoming pooling layers to learn meaningful local structures.





## Graph Kernel Methods

Kernel trick is usually used in dual SVM. SVM can design different kernels for different specific domains. It is a specific topic for feature engineering. 

### Brief introduction on SVM and kernel method.

SVM is a minimal risk algorithm which wants to make the distance between nodes and decision boundary as large as possible, which wants to
$$
\begin{array}{cl}
\max _{\mathbf{w}} & \text { margin }(\mathbf{w}) \\
\text { subject to } & \text { every } y_{n} \mathbf{w}^{T} \mathbf{x}_{n}>0 \\
& \text { margin }(\mathbf{w})=\min _{n=1, \ldots, N} \operatorname{distance}\left(\mathbf{x}_{n}, \mathbf{w}\right)
\end{array}
$$
after simplify, the form can be rewritten as:
$$
\begin{array}{cl}
\min _{\mathbf{w}} & \frac{1}{2}w^Tw \\
\text { subject to } & \min_{n=1, \cdots, n} y_{n} (\mathbf{w}^{T} \mathbf{x}_{n}+b)>0 
\end{array}
$$
However, the time complexility of solving this problem is linear with the data dimension which grows exponentially with the faeture engineering for higher order interactions. 

 With strong dual condition, the problem can be converted into:
$$
\min_{b,w} \to \min_{b,w}\left(\max_{\alpha \ge 0}\mathcal{L}(b,w,\alpha) \right) \to \max_{b,w}\left(\min_{\alpha \ge 0}\mathcal{L}(b,w,\alpha) \right)
$$
With solve this problem, we can get:
$$
w = \sum_{n=1}^N\alpha_n y_n z_n
$$

$$
\sum_{n=1}^N\alpha_ny_ = 0
$$

where $z_n$ is $x_n$ after feature engineering.

Then the problem canbe solved as:
$$
\begin{aligned} \min _{\alpha} & \frac{1}{2} \sum_{n=1}^{N} \sum_{m=1}^{N} \alpha_{n} \alpha_{m} y_{n} y_{m} \mathbf{z}_{n}^{T} \mathbf{z}_{m}-\sum_{n=1}^{N} \alpha_{n} \\ \text { subject to } & \sum_{n=1}^{N} y_{n} \alpha_{n}=0 \\ & \alpha_{n} \geq 0, \text { for } n=1,2, \ldots, N \end{aligned}
$$
Then the time complexity is connected to the number of training samples. $\mathbf{z}_{n}^{T} \mathbf{z}_{m}$ is the design space for the kernel function！

The SVM function can be writtern as
$$
\mathbf{y} = \left (\sum \alpha_ny_nz_n \right)\mathbf{z} + b
$$
Therefore, each new sample will compare the similarity with the kernel function. 

**For kernel method it measures the similarity which corresponds to the inner product**

**For graph kernel, it is design to measure the graph structure similarity**

### Traditional Graph Kernel

Graph kernel learns the structural latent representation by predefined sub-structure for graphs. 

Most of the graph kernel methods belongs to the R-convolution, which the key idea is

- find the atom subgraph pattern (recursively decompose into subgraph)
- $\phi(\mathcal{G})$ denotes the vector which contains counts of atomic sub-structures. A count vector with normalization.
- The similarity between graphs is computed as $K(\mathcal{G},\mathcal{G}') = \phi(\mathcal{G}) \cdot \phi(\mathcal{G}')$

Traditional graph kernel focuses on the design of atomic graph sub-structure with limited sub-graph (Graphlet), subtree pattern (WL), walk (random walk) and path (Shorest Path).

**subgraph (graphlet)**

![](https://pic3.zhimg.com/80/v2-27230fbb1c89d6f4b1628ad351001200_1440w.png)

 A graphlet $G$ is an induced and **non-isomorphic subgraphs**. It is generated by add a node, add an edge, remove an edge. The graph kernel with $k \le 5$ will have a count vector with 52 features.

**subtree (WL)**

WL is to iterate over each vertex of a labeled graph and its neighbors in order to create a multiset label. The resultant multiset is given a new label, which is then used for the next iteration until the label set does not change anymore. To compute the similarity, just count the co-occurrences of labels in both graphs. The dimension is based on the number of iteration.

**Path (Shortest path)** 

The shortest path pattern is defined as the triplet $(l_s^i, l_e^i, n_k)$. The kernel similarity just computes the co-occurance of the shortest path pattern.

### Deep Graph Kernel

The intuition is that we count different subgraphs patterns independently. However, it is easy to see some of the subgraph patterns are dependent with each other. A similarity matrix (positive semidefinite) $M$ should be computed.  
$$
K(\mathcal{G},\mathcal{G}') = \phi(\mathcal{G})^T \mathcal{M} \phi(\mathcal{G}')
$$
It is learnt by the skip gram inspired by Word2Vec, and the key is to find the co-occurance between subgraph patterns.

**SP:** the shortert path with the same source is viewed as the context

**WL:** The multilabel set on different node but the same iteration can be viewed as the context.



**some weakness**

- graphlet kernel builds kernels based on fixed-sized subgraphs. These subgraphs, which are often called motifs or graphlets, reflect functional network properties. 
  However, due to the combinatorial complexity of subgraph enumeration, graphlet kernels are restricted to subgraphs with few nodes
- WL kernel
  only support discrete features and use memory linear in the number of training examples at test time.
- Deep graph kernels and graph invariant kernels compare graphs based on the existence or count of small substructures such as shortest paths, graphlets, subtrees, and other graph invariants?
- All graph kernels have a training complexity at least quadratic in the number of graphs, which is prohibitive for large-scale problems



### DDGK:  Learning Graph Representations for Deep Divergence Graph Kernels

DDGK is a new expressive kernel with deep learning which encodes a relaxed notion of graph isomorphism. It breaks the heuristics constraints in the traditional method.

The major perspective are three-fold:

- How to represent a graph (capture the graph information)
- How to align graphs and find the similarity? cross-graph alignment.
- How to measure the divergence score(the similarity kernel representation)

Notice that, it is more like a graph embedding method without the supervision signal.

**Graph encoder** should be able to reconstruct structure on some given nodes. It aims to distinguish the label of the neighborhood node with a single linear layer (embedding lookup). 
$$
J(\theta)=\sum_{i} \sum_{j \atop e_{i j} \in E} \log \operatorname{Pr}\left(v_{j} \mid v_{i}, \theta\right)
$$
**Cross-Graph Attention** aims to measure how much the target graph diverges from the source graph, it use the source graph encoder to predict the structure of the target graph. If the pair is similar, we expect the source graph encoder to correctly predict the target graph’s structure.

A bidirection projection is proposed acrossing pair of nodes. It will assign the source node to the target graph with certain probability.
$$
\operatorname{Pr}\left(v_{j} \mid u_{i}\right)=\frac{e^{\mathcal{M}_{T \rightarrow S}\left(v_{j}, u_{i}\right)}}{\sum_{v_{k} \in V_{S}} e^{\mathcal{M}_{T \rightarrow S}\left(v_{k}, u_{i}\right)}}
$$
A reverse projection is similar with it. $\mathcal{M}_{T \rightarrow S}\left(v_{j}, u_{i}\right)$ is a simple multiclass classifier.

**Divergence Embedding**

Then each pair of nodes between two graphs are used for comparion. It measure how well the source node can predict the target node neighborhood and concate as the final embedding.
$$
\mathcal{D}^{\prime}(T \| S)=\sum_{v_{i} \in V_{T}} \sum_{j \atop e_{j i} \in E_{T}}-\log \operatorname{Pr}\left(v_{j} \mid v_{i}, H_{S}\right)
$$
 

## Paper Reading List

- Nested Graph Neural Networks (NIPS)
- StructPool Structured Graph Pooling via Conditional Random Fields (ICLR)
- Rethinking pooling in graph neural networks (NIPS)
- Principal Neighbourhood Aggregation for Graph Nets (NIPS)
- MEMORY-BASED GRAPH NETWORKS (ICLR)
- A FAIR COMPARISON OF GRAPH NEURAL NETWORKS FOR GRAPH CLASSIFICATION (ICLR)
- InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization (ICLR)
- Convolutional Kernel Networks for Graph-Structured Data (ICML)
- ACCURATE LEARNING OF GRAPH REPRESENTATIONS WITH GRAPH MULTISET POOLING (ICLR2020)
- Benchmarking Graph Neural Networks (Arxiv 2020)
- Bridging the Gap Between Spectral and Spatial Domains in Graph Neural Networks (Arxiv2020) 
- Open Graph Benchmark (NIPS2020)
- Tudataset A collection of benchmark datasets for learning with graphs (arxiv2020)
- Spectral Clustering with Graph Neural Networks for Graph Pooling (ICML2020) 
- Graph Convolutional Networks with EigenPooling (KDD)
- HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (ICLR)
- Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks (AAAI) 
- Self-Attention Graph Pooling (ICML)
- Graph U-Net (ICML)
- Are Powerful Graph Neural Nets Necessary? A Dissection on Graph Classification (Arxiv)
- CAPSULE GRAPH NEURAL NETWORK(ICLR)
- CLIQUE POOLING FOR GRAPH CLASSIFICATION (Arxiv)
- Hierarchical Graph Pooling with Structure Learning (AAAI19)
- DDGK: Learning Graph Representations for Deep Divergence Graph Kernels (WWW19) 
- An End-to-End Deep Learning architecture for graph classification (AAAI)
- Hierarchical graph representation learning with differentiable pooling (NIPS)
- Towards Sparse Hierarchical Graph Classifiers (Arxiv)
- A simple yet effective baseline for non-attribute graph classification (arxiv)
- Dynamic edge-conditioned filters in convolutional neural networks on graphs (CVPR)
- Learning Convolutional Neural Networks for Graphs



## Useful words

Several approaches have been proposed in recent GNN literature, ranging from model-free methods that precompute the pooled graphs by leveraging graph-theoretical properties (Bruna et al., 2013; Defferrard et al., 2016), to model-based methods that perform pooling trough a learnable function of the node features (Ying et al., 2018; Cangea et al., 2018; Hongyang Gao, 2019). However, existing model-free methods only consider the graph topology but ignore the node features, while model-based methods are mostly based on heuristics. As a consequence, the former cannot learn how to coarsen graphs adaptively for a specific downstream task, while the latter are unstable and prone to find degenerate solutions even on simple tasks. A pooling operation that is theoretically grounded and can adapt to the data and the task is still missing.