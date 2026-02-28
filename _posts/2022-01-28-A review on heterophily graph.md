---
title: A review on heterophily graphs
layout: category
permalink: /categories/A review on heterophily graph/
taxonomy: A review on heterophily graph



---

# A review on heterophily graph

It's good to see you here, with my recent reading on the heterophily graph.  In this blog, we aim to figure out on the recent massive paper on heterophily graph.

In this review, we aim to study the following research question.

- How does GNN work well on the homophily graph?
- What the heterophily graph datasite looks like? How to measure the heterphoily? 
- What is the current solution to the heterophily graph?
- What is the connection with other problems on Graph Neural Network



## How does GNN work well on the homophily graph

To ask this question, we first distuiguish GNN with other exisiting Euclidean-based methods. The key difference between GNN and other method is the two components in GNN.

- Message Passing mechanism: take use of rich information from **neighborhood** of an object can be captured.
- Aggregator: select and compress the information from ego node feature (The feature of node itself) and neighbor feature.

**So what is the key difference when adding them operators? The answer is smoothness.** 

Firstly, GNN procedure can be viewed as a denoising procedure, from a noise signal (original feature) $S \in \mathbb{R}^{N \times d}$ for a clean signal (feature after GNN) $F \in \mathbb{R}^{N \times d}$.
$$
arg \min_F \mathcal{L} = ||F-S||^2_F + c \cdot tr(F^TLF)
$$
where $L = D - A$ is the Laplacian matrix. 

It seems not so hard to understand the first term, which is common in learning good represention, for example, auto encoder. The optimization of the second term is more about message passing mechanism. Taking the second term into consideration solely the smootheness between neighbor is:
$$
tr(F^TLF) = \frac{1}{2}c\sum_{i\in V}\sum_{j\in N(i)} ||F_i - F_j||_2^2
$$
 which measures the difference between node $i$ and all its neighbors

The minimization is similar with the well known Courannt-Fischer problem.
$$
\lambda_1 = arg \min_F tr(F^TLF) = 0
$$
and  $F$ is the eigenvector the corresponding to $\lambda_1$, which is an all one vector, which is extremely smoothly.  



**Then how does this smoothness indeed help us to learn a discriminative model? With homophily assumption, it reduce the noise in the same class. In other words, reduce the intra-class variance for easier classification**


Smoothness can also be viewed as the noisy (variance) reduced. Assume that the noise power is the same, defined by its variance $\sigma^2$. Then after aggregation, the new variance is $\sum_{v_j \in N_{v_i}}a_{i,j}^2 \cdot \sigma^2$. where $a_{i,j}$ is the aggregator factor. For example, `$a_{i,j} = \frac{1}{||N_{v_i}||}$` for mean aggregetor 



Then if the label is also smooth, which means the aggregate label is still the same or much same as before. The label smoothness can be measured as 
$$
\lambda_l = \sum_{e_{v_i,v_j\in \epsilon}} (1- \mathbb{I}(v_i v_j)) / |\epsilon|
$$
The feature smooth can be measure as
$$
\lambda_{f}=\frac{\left\|\sum_{v \in \mathcal{V}}\left(\sum_{v^{\prime} \in \mathcal{N}_{v}}\left(x_{v}-x_{v^{\prime}}\right)\right)^{2}\right\|_{1}}{|\mathcal{E}| \cdot d}
$$


If $\lambda_l$ is large, and $\lambda_f$ is small， GNN can work well which reduce the intra-class variance a lot. 

Anyway, whenever $\lambda_l$ is the small, GNN can help to reduce the noise for smoothness and lower intra-class variance.    

(The explanation is somehow still needed to be proved here. Wait for your discussion)



## How to measure the heterphoily? 

In this section,  we want to ask how does the heterphoily looks like, and why the original GCN can not work well on some cases.

### The measurement for heterphoily

Basically, when we talk about measurement, it is the label measurement in most cases. Similarly, the measurement metrics on homophoily and heterphoily are just opposite to each other, high homophily means low heterphoily.

Then two basic measurement are 

- Edge homophily ratio: `$h=\frac{\left|\left\{(u, v):(u, v) \in \mathcal{E} \wedge y_{u}=y_{v}\right\}\right|}{|\mathcal{E}|}$`
- Node homophily ratio: `$h= \frac{1}{|\mathcal{V}|}  \sum_{v\in \mathcal{V}} \frac{\left|\left\{(u, v): v \in \mathcal{N}_v \wedge y_{u}=y_{v}\right\}\right|}{d_u}$`

Notice that, please be careful whether your graph is a directed one or indirected one. This setting will definitely influence you performance!

The above measurement is naive and intuitive, with also drawback existing. Some new measurement are proposed.

#### Compatibility matrix

**What problem to solve？**  The homophily level varies aming different pair of classes, measurement should e be class-wise.  

The compatibility matrix is defined as follows：
$$
\mathbf{H}=\left(\mathbf{Y}^{\top} \mathbf{A} \mathbf{Y}\right) \oslash\left(\mathbf{Y}^{\top} \mathbf{A} \mathbf{E}\right)
$$
$Y\in \mathbb{R}^{|\mathcal{V}|\times |\mathcal{Y}|}$   is a class indicator matrix.  $|\mathcal{V}|\times |\mathcal{Y}|$ is a $|\mathcal{V}|\times |\mathcal{Y}|$ all-ones matrix.  $\oslash$ is Hadamard (element-wise division), In the $H$ matrix, the diagonal elements measure the homophily.

#### Class-wise measurement

**What problem to solve?** 

- **Number of classes matters!** Heterophily means labels with different classes.  However, "difference" has different meaning in a dataset set with 6 classes and 2 classes. Heterophily means labels with other 5 classes and other 1 classes respectivelt.
- **class balance matters!** For instance, if 99% of nodes were of one class, then most edges would likely be within that same class, so the edge homophily would be high.

To overcome such weakness, the class-heterophily measurement is proposed as:
$$
\hat{h}=\frac{1}{|C|} \sum_{k=1}^{|C|}\left[h_{k}-\frac{\left|C_{k}\right|}{n}\right]_{+}
$$
where $[a]_+=max(a,0)$ and $h_k $is a class-wise metric， the second term is the average value for a randomly connected graph.
$$
h_k =  \frac{\sum_{v\in C_k}\left|\left\{(u, v): v \in \mathcal{N}_v \wedge y_{u}=y_{v}\right\}\right|}{\sum_{v\in C_k}d_u}
$$

#### Aggregation Similarity score

**A new perspective from back propogation: whether  homophily node or heterphoily contributed more on the back propogation direction.**. 

For a GNN without activation function, 
$$
Y = softmax(\hat{A}XW) = softmax(Y')
$$

$$
\bigtriangleup Y' = \mathbf{\hat{A} X X^{T} \hat{A}^{T}}(Z-Y)=S(\hat{A}, X)(Z-Y)
$$

$Z-Y$  is the prediction error matrix. $Z$ is the ground truth matrix  $S(\hat{A}, X)$ determines where to be updated. Then the aggregation similarity score is defined as: which is to determine whether the homophily node contribute more or heterophily node contribute more.
$$
S_{a g g}(S(\hat{A}, X))=\frac{\left|\left\{v \mid \operatorname{Mean}_{u}\left(\left\{S(\hat{A}, X)_{v, u} \mid Z_{u,:}=Z_{v,:}\right\}\right) \geq \operatorname{Mean}_{u}\left(\left\{S(\hat{A}, X)_{v, u} \mid Z_{u,:} \neq Z_{v,:}\right\}\right)\right\}\right|}{|\mathcal{V}|}
$$
The author thinks that his metric can identify the harmful heterophily and the metric wil not take useful heterophily into account.

#### Cross-class Neighborhood Similarity

Just the similarity between feature on different classes.
$$
s(c, c') = \frac{1}{\left|\mathcal{V}_{c}\right|\left|\mathcal{V}_{c^{\prime}}\right|} \sum_{i \in \mathcal{V}_{c}, j \in \mathcal{V}_{c^{\prime}}} \cos (d(i), d(j))
$$


### How to conduct a synthetic hetephoily: From homophoily dataset

#### The drawback of real world data

Commonly used data is proposed by GEOM-GCN: GEOMETRIC GRAPH CONVOLUTIONAL NETWORKS. However, it still has several problems on it. 

- WebKB benchmarks: relatively small sizes
- Unreliable label assignment:  Squirrel and Chameleon have class labels based on ranking of page traffic
- Unusual network structure: quirrel and Chameleon are dense, with many nodes sharing the same neighbors

Also, new good large benchmark dataset has been proposed recently. Please reference Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods 



#### The procedure to generate a dataset

- The number of class is presscribed
- start from a small initial graph, new nodes are added into the graph one by one.
- The probability puv for a newly added node $u$ in class $i$ to connect with an existing node $v$ in class $j$ is proportional to both the class compatibility $H_{ij}$ between class $i$ and $j$, and the degree $d_v$ of the existing node $v$.
- the degree distribution for the generated graphs follow a power law, and the heterophily can be controlled by class compatibility matrix $H$
- Node feature is sampled from the corresponding class in the dataset like Cora.

The detailed can be found in Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs

![](https://pic3.zhimg.com/80/v2-807d2a594a907e6a1c9e8efa0d2944f8_720w.png)



#### Modify a dataset from dataset of the existing homophily dataset

The key is that **how to generate the heterophilous edge?**

![img](https://pica.zhimg.com/80/v2-bc04fa5d9a83591190c3de90ed0cd5cd_720w.png)

The key is to add synthetic,cross label edge s that connect nodes with different labels. where $\gamma$ is the noise probability, $D_c, c \in \mathcal{C}$ is a discrete neighborhood distribution for each class $c \in \mathcal{C}$, which has been predefined.  For example, the neighborhood distribution for class $0$ is: $D_0 = catrgorical([0, 1/2, 0, 0, 0, 0, 1/2])$. Notice that this probability is predefined. This leads to quite interesting experiments result. Please refer to the original paper. 

The detailed procedure can be seen in Is Homophily a Necessity for Graph Neural Networks?

#### CSBM generated conduction

CSBM(contextual stochastic block model) is a graph generator, which is a generative model for random graphs. It has been widely used in graph clustering. It has such feature:

- node features are Gaussian random vectors, where the mean of the Gaussian depends on the community assignment.
- The difference of the means is controlled by a parameter $\mu$
- the difference of the edge densities in the communities and between the communities is controlled by a parameter $\lambda$.  $\lambda>0$ correspond to homophilc graphs.

We will have more detailed introduction about it later





## What is the current solution to the heterophily graph?

I think this will be the most exiciting part when summary recent progress. In this section, we will introduce from the mulitple perspectives.

Still all our perspective will focus on this  formulation: 
$$
arg \min_F \mathcal{L} = ||F-S||^2_F + c \cdot tr(F^TLF)
$$
The design space first focuses on the first term, which means to preserve the original feature. Correct, if the heterphoily neighbor may disturb it. **Adaptive self-loop is necceary**

Then for the second term  with smoothness assumption, the questions is raised: Is smoothness hurt the distuiguishbility?  From the frequency perspective, low signal may be good, however, if in the optimal case $\lambda=0$, much information is lose in this procedure.

Then the solution will be:

- Instead of node similarity, we want to find the neighbor difference, or can be called high frequency signal
- find more diverse neighbor to inlarge receptive field for more information
  - Conduct more adjacent matrixes than $L$ for more useful signal. For example KNN graph,, structure-graph.
  - build deeper GNN and select the useful neighbor, In other word $tr(F^TLF)$ becomes $tr(F^T (\alpha_1 L^1+\alpha_2 L^2+\alpha_3 L^3)F)$. May be the third order neighbor can help become a homophily graph. **In fact, oversmoothness and heterphily problem are just the two sides of a coin**



The following topic will focus on these designing space:

- Keep origin feature and find differences with neighborhood
- Find more useful adjacent matrix
- Deeper GNN for larger receptive field

Some of our introduced method will include more than one of these perspectives. Unfortunetely, some good model can not be concluded in this framework, which include LinkX and CPGCN. 

Our discussion is based on the paperlist mentioned at the end of this blog.



### Keep origin feature

First  we introduce a word called: **Node feature similarity**, which is opposite to the neighbor smoothness. Concretely speaking, the aggregation process of GNNs try to find preserve structural similarity, while tends to destroy node similarity in the original feature space. 

To keep the original feature, self loop is of great necceraity and other feature preserve method

#### SimP-GCN:Node Similarity Preserving Graph Convolutional Networks

The solution of SimP-GCN is the following three steps:

- Feature similarity adjacent matrix
- **adaptive self-loop**
- Feature-based SSL task

![](https://pic1.zhimg.com/80/v2-a4403766cb65b7092e226af70f3d0581_720w.png)

**The two channel propogation** is mainly based on two graphs, the adjacent matrix and feature KNN matrix.
$$
\mathbf{P}^{(l)}=\mathrm{s}^{(l)} * \tilde{\mathbf{D}}^{-1 / 2} \tilde{\mathrm{A}} \tilde{\mathrm{D}}^{-1 / 2}+\left(1-\mathrm{s}^{(l)}\right) * \mathrm{D}_{f}^{-1 / 2} \mathbf{A}_{f} \mathbf{D}_{f}^{-1 / 2}
$$
where $\mathbf{P}^{(l)}$ is the propogation matrix, KNN adjacent $A_f$ is defined by feature cosine similarity, select the top K similar nodes as neighborhood.

The adaptive combination score is learned as:
$$
\mathbf{s}^{(l)}=\sigma\left(\mathbf{H}^{(l-1)} \mathbf{W}_{s}^{(l)}+b_{s}^{(l)}\right)
$$
where $\mathbf{W}_{s}^{(l)} \in \mathbb{R}^{d^{(l-1)} \times 1}$

**Adaptive self-loop** as

$$
\tilde{\mathbf{P}}^{(l)}=\mathbf{P}^{(l)}+\gamma \mathbf{D}_{K}^{(l)}
$$
 
 $\mathbf{D}_{K}^{(l)} = diag(K_1^{(l)},K_2^{(l)}, K_n^{(l)} )$ , it is learned from 

$$
K_{i}^{(l)}=\mathbf{H}_{i}^{(l-1)} \mathbf{W}_{K}^{(l)}+b_{K}^{(l)}
$$

where $\mathbf{W}_{K}^{(l)} \in \mathbb{R}^{d^{(l-1)} \times 1}$

**SSL task** to preserve the original node feature similarity
$$
\mathcal{L}_{\text {self }}(\mathbf{A}, \mathbf{X})=\frac{1}{|\mathcal{T}|} \sum_{\left(v_{i}, v_{j}\right) \in \mathcal{T}}\left\|f_{w}\left(\mathbf{H}_{i}^{(l)}-\mathbf{H}_{j}^{(l)}\right)-\mathrm{S}_{i j}\right\|^{2}
$$
where $S_{ij}$ is the cosine node feature similarity between node i and j, SSL is a regression to learn original feature similarity and dissimilarity.

#### H2GCN: Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs

In this paper, they think maybe the self-loop is not a good solution.

They design a separate aggregation for ego embedding and neighbor embedding. . They use the combination operation for these two embeddings. The origin faeture is better preserved.
$$
\mathbf{r}_{v}^{(k)}=\operatorname{COMBINE}\left(\mathbf{r}_{v}^{(k-1)}, \operatorname{AGGR}\left(\left\{\mathbf{r}_{u}^{(k-1)}: u \in \bar{N}(v)\right\}\right)\right)
$$
The neighbor $\bar{N}(v)$ does not include $v$. the combine may follows by non-linear transformation.

They point that combine is better for generalization theorical.

Theoritical Jusfiction is that 


Consider a graph G without self-loops with node features `$x_v = onehot(y_v)$` for each node `$v$`, and an equal number of nodes per class `$y ∈ Y$` in the training set `$\mathcal{T}_V$` . Also assume that all nodes in `$T_V$` have degree d, and proportion h of their neighbors belong to the same class, while proportion `$\frac{1−h} {|Y|−1} $`of them belong to any other class (uniformly). Then for `$h < \frac{1−|Y|+2d}{ 2|Y|d}$` , a simple GCN layer formulated as `$(A + I)XW$` is less robust, i.e., misclassifies a node for smaller train/test data deviations, than a `$AXW $` layer that separates the ego- and neighbor-embeddings.

**TODO: give more intuition on the proof**



The above two papers are two papers focus on this original feature preserve. **Notice that, this componet has been widely used who utilize it as a channel**



### Find differences with neighborhood

In this part, there are major two methods which take idea from spatial and spectral perspectives.

The questions are similar:

- What is the role of low-frequency signal and high-frequency signal?
- How to use the high-filter reduce the hurt from bad heterophily?

#### FAGCN：Beyond Low-frequency Information in Graph Convolutional Network

This paper mainly talk about the low-signal is not enough for learning good representation, as low frequency igonore the difference. good GNN should be able to seperate and capture low-frequency, high-frequency. Then adaptively propogate low-frequency signals, high-frequency signal and raw features with self-gating mechasim.

The relation between heterphoily and frequency is as follows:

![](https://pic1.zhimg.com/80/v2-c1e0217e3a8a1879d8edeeb8d279dd3d_720w.png)

**The signal is extracted as:**
$$
\mathcal{F}_{L}=\varepsilon I+D^{-1 / 2} A D^{-1 / 2}=(\varepsilon+1) I-L
$$

$$
\mathcal{F}_{H}=\varepsilon I - D^{-1 / 2} A D^{-1 / 2}=(\varepsilon-1) I+L
$$

where $\epsilon$ is the hyperparameter to balance the low and high frequency: amplifies the low-frequency signals and restrains the high-frequency signals.(also the self loop, see below)

**multi channel by self-gate:**
$$
\tilde{\mathbf{h}}_{i}=\alpha_{i j}^{L}\left(\mathcal{F}_{L} \cdot \mathbf{H}\right)_{i}+\alpha_{i j}^{H}\left(\mathcal{F}_{H} \cdot \mathbf{H}\right)_{i}=\varepsilon \mathbf{h}_{i}+\sum_{j \in \mathcal{N}_{i}} \frac{\alpha_{i j}^{L}-\alpha_{i j}^{H}}{\sqrt{d_{i} d_{j}}} \mathbf{h}_{j}
$$
$\alpha_{i,j}^L+\alpha_{i,j}^H = 1$ as the two part of the signal. It is learned from 
$$
\alpha_{i j}^{G}=\tanh \left(\mathbf{g}^{\top}\left[\mathbf{h}_{i} \| \mathbf{h}_{j}\right]\right)
$$

#### ACM-GCN: Is Heterophily A Real Nightmare For Graph Neural Networks To Do Node Classification?

This paper starts from whether all heterophily is bad for graph and indicate the bad heterophily by analys the BP. The measurement can be found in the measurement part above. They address it with diversification operation, and propose the Adaptive Channel Mixing (ACM) framework.

The intuition is that they want to ignore the grey one to be small, high-pass filter (diversification operation) to extract the information of neighborhood differences and address harmful heterophily.

![](https://pic1.zhimg.com/80/v2-c4f53751a9a49783f79d8dae98e778a9_720w.png)

The learning is three step, similar to the former one: 

**Feature Extraction for each channel: ** （extract signal)
$$
H_{L}^{l}=\operatorname{ReLU}\left(H_{\mathrm{LP}} H^{l-1} W_{L}^{l-1}\right), H_{H}^{l}=\operatorname{ReLU}\left(H_{\mathrm{HP}} H^{l-1} W_{H}^{l-1}\right), H_{I}^{l}=\operatorname{ReLU}\left(I H^{l-1} W_{I}^{l-1}\right)
$$
where $H_{LP} = A$, $H_{LP} = L$ 

**Feature-based weight Learning:** (like self gate )

$$
\begin{array}{l}
\tilde{\alpha}_{L}^{l}=\sigma\left(H_{L}^{l} \tilde{W}_{L}^{l}\right), \tilde{\alpha}_{H}^{l}=\sigma\left(H_{H}^{l} \tilde{W}_{H}^{l}\right), \tilde{\alpha}_{I}^{l}=\sigma\left(H_{I}^{l} \tilde{W}_{I}^{l}\right), \tilde{W}_{L}^{l-1}, \tilde{W}_{H}^{l-1}, \tilde{W}_{I}^{l-1} \in \mathbb{R}^{F_{l} \times 1} \\
{\left[\alpha_{L}^{l}, \alpha_{H}^{l}, \alpha_{I}^{l}\right]=\operatorname{Softmax}\left(\left[\tilde{\alpha}_{L}^{l}, \tilde{\alpha}_{H}^{l}, \tilde{\alpha}_{I}^{l}\right] W_{\text {Mix }}^{l} / T,\right), W_{\text {Mix }}^{l} \in \mathbb{R}^{3 \times 3}, T \in \mathbb{R} \text { is the temperature; }}
\end{array}
$$

**multi channel aggregate:**

$$
H^{l}=\left(\operatorname{diag}\left(\alpha_{L}^{l}\right) H_{L}^{l}+\operatorname{diag}\left(\alpha_{H}^{l}\right) H_{H}^{l}+\operatorname{diag}\left(\alpha_{I}^{l}\right) H_{I}^{l}\right)
$$

**Notice that more than FAGCN,  this is a framework which can be applied to any GNN model**



**One remain question, what is the difference between  A L and L -L for these signal**

### Find more useful adjacent matrix

Like Geom-GCN says, the behind basic idea is the aggregation on a graph can benefit from a continuous space underlying the graph. where this useful continuous lies? It lies in:

- Graph embedding similarity adjacent matrix
- Structural similarity adjacent matrix (Struct2Vec)
- Origin Feature similarity adjacent matrix (SimP-GCN)

They can help to capture long-range information which allerviate the problem of deeper GNN:

- relevant messages from distant nodes are mixed indistinguishably with a large number of irrelevant messages from proximal nodes in multi-layer MPNNs, which cannot be extracted effectively
- representations of different nodes would become very similar in multi-layer MPNNs

#### Geom-GCN:  GEOMETRIC GRAPH CONVOLUTIONAL NETWORKS (Graph embedding similarity)

In this paper, they point out

- pooling and aggregate will compress the structural information of nodes in neighborhoods. Multi channel to preserve the neighbor structure is neccerary.
- Caputure long-rang on the geometric continuous latent space. (graph embedding technique to conduct new information)

**structural neighborhood** is conducted on different Graph Embedding measurement  like DeepWalk, IsoMap. The neighbor is defined as $\mathcal{N}(v)=(\{N_{g}(v), N_{s}(v)\}, \tau)$, each $\tau$ is a geometric graph embdding methods with $N_{s}(v)=\left\{u \mid u \in V, d\left(\boldsymbol{z}_{u}, \boldsymbol{z}_{v}\right)<\rho\right\}$ ,

$\rho$ is hyperparameter predefined.

**Bi-level (multi channel) aggregation** is a multi-channel aggregation 

- Low-level aggregation (aggregate from a kind of measurement, a channel)
  $$
  \boldsymbol{e}_{(i, r)}^{v, l+1}=p\left(\left\{\boldsymbol{h}_{u}^{l} \mid u \in N_{i}(v), \tau\left(\boldsymbol{z}_{v}, \boldsymbol{z}_{u}\right)=r\right\}\right), \forall i \in\{g, s\}, \forall r \in R
  $$
  $p$ is a permutation-invariant function like mean aggregation. 

- High-level aggregation (aggregate the multi-channel representation)
  $$
  \boldsymbol{m}_{v}^{l+1}=\underset{i \in\{g, s\}, r \in R}{q}\left(\left(\boldsymbol{e}_{(i, r)}^{v, l+1},(i, r)\right)\right)
  $$
  $q$ is a pooling operation

- Non-linear transform

#### WRGAT:Breaking the Limit of Graph Neural Networks by Improving the Assortativity of Graphs with Local Mixing Patterns (Structural similarity )

This paper first point out global assortativity can not learn the diversity on each node. They propose a new node-level measurement. Then they find the node with low local assortativity can not be learned well from GNN. So they include the structural proximity into consideration to help learning

**Node-level assortativity**:

- The origin measurement: random walk with stationary 
- New measurement: random walk with restart



**Structural neighborhood**: conduct similarly like Struct2Vec. It can generate multiple view of graphs when consideration different hop of papers. The intuition is that, two nodes are similar if they have same degree. They are more similar if there neighbor nodes also has same degree.

The structural distance can be computed as:
$$
f_{\tau}(g, h)=f_{\tau-1}(g, h)+\mathcal{D}\left(s\left(\mathcal{N}_{\tau}(g)\right), s\left(\mathcal{N}_{\tau}(h)\right)\right)
$$
$S_1$ and $S_2$  are two ordered degree sequences and the distance is computed by DTW.

The edge weight is computed as 
$$
w_{\tau}(g, h)=e^{-f_{\tau}(g, h)}, \quad \tau=0,1, \ldots T
$$
which $\tau$ means multi hop neighbor.

**Multi channel aggregation:**  
$$
\boldsymbol{m}_{u}=\sum_{\tau \in \mathcal{R}} \sum_{v \in \mathcal{N}_{1}^{\tau}(u)} \boldsymbol{W}_{\tau} \boldsymbol{h}_{v} w_{\tau}(u, v) \alpha_{\tau}(u, v)
$$
where $\alpha$ is a GAT like attention.
$$
\boldsymbol{e}_{u, v}^{\tau}=a_{\tau}\left(\boldsymbol{W}_{\tau} \boldsymbol{h}_{u}, \boldsymbol{W}_{\tau} \boldsymbol{h}_{v}\right)=\boldsymbol{a}^{T}\left[\boldsymbol{W}_{\tau} \boldsymbol{h}_{u} \| \boldsymbol{W}_{\tau} \boldsymbol{h}_{v}\right]
$$

$$
\alpha_{\tau}(u, v)=\frac{\exp \left(\operatorname{LeakyReLU}\left(e_{u, v}^{\tau}\right)\right)}{\sum_{x \in \mathcal{N}_{1}^{\tau}(u)} \exp \left(\operatorname{LeakyReLU}\left(e_{u, x}^{\tau}\right)\right)}
$$



#### Query Attention (NL-MLP):Non-Local Graph Neural Networks

This paper want  to go beyond the local aggegation is harmful for disassortative graphs. It is necceary to have 
**non-local aggregation to capture long-range dependencies(nodes with the same label are distant from each other) with attention query. **

![](https://pic1.zhimg.com/80/v2-60a6a17a0ea1d724c10d92e265159605_720w.png)

**Local Embedding:** a local embedding which can be a one-layer GNN or just MLP as a feature extractor.

**Attention-guide：** learn an attention score for each local embedding with a learnable query
$$
a_{v}=\operatorname{ATTEND}\left(c, z_{v}\right) \in \mathbb{R}, \forall v \in V
$$
where $c$ is a calibration vector that is randomly initialized and jointly learned during training. This procedure, make the distance but similar nodes together.

**Non-local aggregation:**

Then node can be sorted with the attention score, and form like a sequence.  A 1D conv is used for neighbor feature extraction.



### Deeper GNN for larger receptive field

In the former section, we says that relevant messages from distant nodes are mixed indistinguishably with a large number of irrelevant messages from proximal nodes in multi-layer MPNNs, which cannot be extracted effectively.

So is it any method can direct use the distant nodes without mixed up a lot? As the former section with more useful adjacent nodes is built on much assuption and will certain lose some information.  **The question is how to build deeper GNN without oversmooth and capture more important nodes?**

Here we will only talk about two typical methods: H2GCN and GPR-GCN. Methods like AAPNP and GCNII will be discussed in my next blog.



#### H2GCN: Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs

We have introduced this method first, here we introduce two key component in it.

**Higher-order Neighborhoods**

This design opinion is siimlary with GCN-Cheby and MixHop, which is:
$$
\mathbf{r}_{v}^{(k)}=\operatorname{COMBINE}\left(\mathbf{r}_{v}^{(k-1)}, \operatorname{AGGR}\left(\left\{\mathbf{r}_{u}^{(k-1)}: u \in N_{1}(v)\right\}\right), \operatorname{AGGR}\left(\left\{\mathbf{r}_{u}^{(k-1)}: u \in N_{2}(v)\right\}\right), \ldots\right)
$$
The key on this idea is that: maybe one-hop neighbor is homophily, but the two-hop neighbor is homophily. Just find them. The theoritical proof as follows

Theorem 2 Consider a graph  `$\mathcal{G}$`  without self-loops    with label set  `$\mathcal{Y}$` , where for each node  v , its neighbors' class labels  `$\left\{y_{u}: u \in N(v)\right\}$`  are conditionally independent given  `$y_{v}$` , and `$P\left(y_{u}=\right.   \left.y_{v} \mid y_{v}\right)=h$`, `$P\left(y_{u}=y \mid y_{v}\right)=\frac{1-h}{|\mathcal{Y}|-1}, \forall y \neq y_{v} $`.  Then, the 2 -hop neighborhood  `$N_{2}(v)$`  for a node  `$v$`  will always be homophily-dominant in expectation.

**Combination of Intermediate representations**

The design opinion is similar with the paper we will introduce later: GPR-GCN, also like JKNet. The key is to select the most informative layer representation, adapting to different neighborhood range structural properties. 
$$
\mathbf{r}_{v}^{(\text {final })}=\operatorname{COMBINE}\left(\mathbf{r}_{v}^{(1)}, \mathbf{r}_{v}^{(2)}, \ldots, \mathbf{r}_{v}^{(K)}\right)
$$
However, this combination is somehow naive, in the next paper, we will detail into how to do this layer selection.

#### GPRGNN: ADAPTIVE UNIVERSAL GENERALIZED PAGERANK GRAPH NEURAL NETWORK

This paper is to build a GNN with Generalized PageRank(GPR) which can adaptively learn different node label pattern, in this  way, they can not only deal with the naive homophily, but also the complex heterphily pattern. The main approach is that GPR can learn a weight for each layer like a step of random walk. 

The framework is just like a layer selection:

![](https://pic3.zhimg.com/80/v2-1c01e94fa819e027afbafa8574d341b9_720w.png)

The GPR at some natural number $K$,  $\sum_{k=0}^{K} \gamma_{k} \tilde{\mathbf{A}}_{\mathrm{sym}}^{k}$ is actual corresponds to lean optimial polynomial graph filter.

![](https://pica.zhimg.com/80/v2-4653fb3515c7a05c416130b11d0c4e50_720w.png)

**I will revise the pagerank algorithm then for a detailed discussion**

We can find the heterphoily needs more step to becom stable, which means large-step propagation is indeed of great importance for homophilic graphs.

The  model structure is like 
$$
\hat{\mathbf{P}}=\operatorname{softmax}(\mathbf{Z}), \mathbf{Z}=\sum_{k=0}^{K} \gamma_{k} \mathbf{H}^{(k)}, \mathbf{H}^{(k)}=\tilde{\mathbf{A}}_{\mathrm{sym}} \mathbf{H}^{(k-1)}, \mathbf{H}_{i:}^{(0)}=f_{\theta}\left(\mathbf{X}_{i:}\right)
$$
Notice that, it escapes from the non-linear activation function, where the each hidden state corresponds to the one-step pagerank. **GPR-GNN has the ability to adaptively control the contribution of each propagation step and adjust it to the node label pattern.**



## What is the correlation with other problems in graph

**TODO on a summary on is homophily neccerary and two coins** 

## Questions

- What is the relationship between mixup learning and heterphoily graph？
- It seems that still no method to judge a graph without its label to measure it is homophily or not. A unify method may be of great need？
- Is dissimilar feature really a drawback for heterphoily graph or for GNN?











# PaperList

- Powerful Graph Convolutioal Networks with Adaptive Propagation Mechanism for Homophily and Heterophily  （AAAI2022） 
- GBK-GNN: Gated Bi-Kernel Graph Neural Networks for Modeling Both Homophily and Heterophily (WWW2022)
- Graph Neural Networks Inspired by Classical Iterative Algorithms (ICML2021)
- Beyond Low-frequency Information in Graph Convolutional Networks (AAAI2021)
- ADAPTIVE UNIVERSAL GENERALIZED PAGERANK GRAPH NEURAL NETWORK (ICLR2021)
- Breaking the Limit of Graph Neural Networks by Improving the Assortativity of Graphs with Local Mixing Patterns (KDD2021)
- A Real Nightmare For Graph Neural Networks To Do Node Classification? (Preprint)
- Beyond Low-frequency oneIs Homophily a Necessity for Graph Neural Networks? (ICLR2022)
- TWO SIDES OF THE SAME COIN: HETEROPHILY AND OVERSMOOTHING IN GRAPH CONVOLUTIONAL NEURAL NETWORKS (Preprint)
- Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods (NeurIPS2021)
- New Benchmarks for Learning on Non-Homophilous Graphs (WWW2021)Node Similarity Preserving Graph Convolutional Networks (WSDM2021)
- Simple and Deep Graph Convolutional Networks (ICML2020)5Non-Local Graph Neural Network (TPAMI)
- Graph Neural Networks with Heterophily (AAAI2020)
- Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs (NeurIPS2020)
- GEOM-GCN: GEOMETRIC GRAPH CONVOLUTIONAL NETWORKS (ICLR2020)
- MEASURING AND IMPROVING THE USE OF GRAPH INFORMATION IN GRAPH NEURAL NETWORKS (ICLR2020)





# Useful sentences

However, the typical GCN design that mixes the embeddings through an average [17] or weighted average [36] as the COMBINE function results in final embeddings that are similar across neighboring nodes (especially within a community or cluster) for any set of original features [28]. While this may work well in the case of homophily, where neighbors likely belong to the same cluster and class, it poses severe challenges in the case of heterophily: it is not possible to distinguish neighbors from different classes based on the (similar) learned representations.

The aggregators lack the ability to capture long-range dependencies in disassortative graphs. In MPNNs, the neighborhood is defined as the set of all neighbors one hop away (e.g., GCN), or all neighbors up to r hops away (e.g., ChebNet). In other words, only messages from nearby nodes are aggregated. The MPNNs with such aggregation are inclined to learn similar representations for proximal nodes in a graph. This implies that they are probably desirable methods for assortative graphs (e.g., citation networks (Kipf & Welling, 2017) and community networks (Chen et al., 2019)), where node homophily holds (i.e., similar nodes are more likely to be proximal, and vice versa), but may be inappropriate to the disassortative graphs (Newman, 2002) where node homophily does not hold. For example, Ribeiro et al. (2017) shows disassortative graphs where nodes of the same class exhibit high structural similarity but are far apart from each other. In such cases, the representation ability of MPNNs may be limited significantly, since they cannot capture the important features from distant but informative nodes.

The homophily principle (McPherson et al., 2001) in the context of node classification asserts that nodes from the same class tend to form edges. Homophily is also a common assumption in graph clustering (Von Luxburg, 2007; Tsourakakis, 2015; Dau & Milenkovic, 2017) and in many GNNs design (Klicpera et al., 2018). Methods developed for homophilic graphs are nonuniversal in so far that they fail to properly solve learning problems on heterophilic (disassortative) graphs (Pei et al., 2019; Bojchevski et al., 2019; 2020). In heterophilic graphs, nodes with distinct labels are more likely to link together (For example, many people tend to preferentially connect with people of the opposite sex in dating graphs, different classes of amino acids are more likely to connect within many protein structures (Zhu et al., 2020) etc). GNNs model the homophily principle by aggregating node features within graph neighborhoods. For this purpose, they use different mechanisms such as averaging in each network layer. Neighborhood aggregation is problematic and significantly more difficult for heterophilic graphs (Jia & Benson, 2020)