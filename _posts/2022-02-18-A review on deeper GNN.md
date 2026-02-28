---
title: A review on deeper GNN
layout: category
permalink: /categories/A review on deeper GNN/
taxonomy: A review on deeper GNN



---

# A review on problematic deeper GNN

Node classification is the most well-known topic on graph domain which aims to distuiguish the type of each node on graph. In this field, people also study much fundmental limitation on GNN. The main challenge is that we believe that GNN will be more powerful with more layers and more parameters. For example, it is easy for build a CNN with more than 100 layers.  However, GNN always can not. To build deeper GNN with more parameters, people try to understand ane explain this problem and give some solutions. 

We aim to answer the folowing research questions in deeper GNN:

- How does GNN really help? Deep understanding on GNN
- Problems on GNN: overfitting? Gradient Explosion? gradient vanishment? oversmooth? oversquash?
- Why GNN fails because of oversmoothness? From both empirical and theoretical perspectives with some solution.



Word at the from, this part view is somehow more difficult than my early review in graph classification, heterphily graph and domain adaptation with many advanced topics on GNN. I will try my best to understand and write about it. This will not be the last version of this blog. I aim to go beyond above it.



## How does GNN really help? Deep understanding on GNN

**TODO**



## Problems on GNN: overfitting? gradient explosion? gradient vanishment? oversmooth? oversquash?

In the problematic deeper GNN, various problem has been proposed, various paper has proposed different problems. We will first introduce them quickly.

- Gradient explosion and vanishment are two common problems in DNN, which cause overfitting and failure of training. Welcome to see it in my another blog [[link]](https://huanhuqueyue.github.io/personal-page/categories/neuronCampaign/). little new understand on GNN from this perspective.
- Overfitting is also a really common problem in DNN. The main phenomenon is: high train accuracy and low test accuracy with large accuracy gap.  Since it is highly related with generalization, some new understanding on transductive setting are provided.
- Oversmooth is a new problem in GNN: node representations become indistinguishable when the number of layers increases. An intuition understanding is that with aggregate too much neighborhood nodes (entire the whole graph), each node representation is aggregated from whole graph which results in distuiguishable.
- Oversquash is a new problem reference from RNN. Information from the exponentially-growing receptive field is compressed into fixed-length node vectors. It will cause that GNN only focus on overfitting the local-neighbors without considering the new exponentially aggregated node features from further hop. From my perspective, I think it is on the oppsite to oversmoothness problem for it admits different aggregation with various level. 
- (Model degredation is a too confused phase, hard for me to understand it.)

Among them, oversmooth is the main focus recently. Based on this, the following contents will build on the following perspectives:

- The fundamental theoretical understanding on oversmooth
- Spatial: understanding and solution inspired by PageRank
- Spectral: understanding and solution inspired by graph signa
- Understand GNN as a recursive boosting procedure
- Dynamical system: understanding like a continous function with PDE solution
- Advanced operation
  - additional connection on model architecture
  - normalization regularization trick
- Advanced analysis and rethinking on oversmoothness



### The fundamental theory understanding on oversmooth

In this section, we mainly focus on two most widely used theory understanding on oversmooth problem with and without considering the non-linear activation function.

#### Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning

Suppose that a graph $\mathcal{G}$ has $k$ connected components `$\{C_i \}_{i=1}^k$`
and the indication vector for the `$i$`-th components is denoted by $1^{(i)}\in \mathbb{R}^n$, This vector indicates whether a vertex is in the component $C_i$
$$
\mathbf{1}_{j}^{(i)}=\left\{\begin{array}{l}
1, v_{j} \in C_{i} \\
0, v_{j} \notin C_{i}
\end{array}\right.
$$
**Theorem 1** If a graph has no bipartite components, then for any $w\in \mathbb{R}^n$ and $\alpha \in (0,1]$
$$
\begin{array}{l}
\lim _{m \rightarrow+\infty}\left(I-\alpha L_{r w}\right)^{m} \mathbf{w}=\left[\mathbf{1}^{(1)}, \mathbf{1}^{(2)}, \ldots, \mathbf{1}^{(k)}\right] \theta_{1} \\
\lim _{m \rightarrow+\infty}\left(I-\alpha L_{s y m}\right)^{m} \mathbf{w}=D^{-\frac{1}{2}}\left[\mathbf{1}^{(1)}, \mathbf{1}^{(2)}, \ldots, \mathbf{1}^{(k)}\right] \theta_{2},
\end{array}
$$
where $\theta_1 \in \mathbb{R}^k$, $$\theta_2 \in \mathbb{R}^k$$, i.e. they converge to a linear combination of $\{1^{(i)}\}^k_{i=1}$ and $\{D^{\frac{1}{2}}1^{(i)}\}^k_{i=1}$ respectively, which corresponds to the eigenspace to eigenvalue 0.

The proof understanding are as follows:

- The polynomial of a matrix can be written as $L^m = U \Lambda^m U^T$ which corresponds to the polynomial on the eigenvalue. 
- With no bipartite components, the eigenvalue of $\mathcal{L}$ falls in $[0,2)$  (如果有时间去看一看原因
- Then the eigenvalue of $I-\alpha\mathcal{L}$ falls in $(-2\alpha I, 1]$ 
- With $\alpha \in [0,1]$ (a balance between the current node and neighbor node), the eigenvalue is less than 1 except for the eigenvector with all $1$ correponsd to eigenvalue 1
- therefore, with the only remaining eigenvalue, the proof is easy to see.

We see that without consideration on the the transformation part, the information will lose until only degree and the components information.

Let $\lambda_2$ denote the second largest eigenvalue of transition matrix $\tilde{T} = D^{−1}A$ of a non-bipartite graph, $p(t)$ be the probability distribution vector and $\pi$ the stationary distribution. If walk starts from the vertex $i$, $p_i(0) = 1$, then after $t$ steps for every vertex, we have:
$$
\left|p_{j}(t)-\pi_{j}\right| \leq \sqrt{\frac{d_{j}}{d_{i}}} \lambda_{2}^{t}
$$
**TODO:** check this theory, read GDC (上面的证明是哪篇了，我咋给忘了)



#### GRAPH NEURAL NETWORKS EXPONENTIALLY LOSE EXPRESSIVE POWER FOR NODE CLASSIFICATION

This paper consider the expressivity of GNN, a fundamental topic on deep learning, as we all know that the two-layer MLP has the expressive ability of any non-linear functions.

With consideration on the non-linear transformation,  this paper find that as the layer size goes infinite, the output exponentially falls into the set of signal carrying information of connected component and node degree (A subspace that is invariant under the dynamics) (same with the upper one).

**The key assumption is that weights on the non-linear transformation satisfy the conditions determined by the spectra of the (augmented) normalized Laplacian**  The speed approximate the invariant spce is $O((s\lambda)^L)$ where $s$ is the largest singular value of the matrix $W$, $\lambda$ corresponds to the eigenvalue of the laplacian matrix.

##### Key notation 

For a linear operator `$P:\mathbb{R}^N\to \mathbb{R}^M$` and a subset $V \subset \mathbb{R}^N$, we denote the restriction of $P$ to $V$ by <span>$P|_V$</span>

Let $P \in \mathbb{R}^{N \times N}$ be the symmetric adjacent matrix

 For $M \le N$, let $U$ be a $M$-dimensional subspace of $\mathbb{R}^N$. If $U$ is the eigenvector subspace of GNN, it has the following assumption

- $U$ has an orthonormal basis $(e_m)_{m\in[M]}$ that consists of **non-negative** vectors.
- $U$ is invariant under $P$, i.e., if $u \in U$, then $Pu \in U$. Also the orthogonal complement of $U$ called $U^\perp$ has also the same property.

Then the linear mapping with constraint can be written <span>$P|_{U^\perp}: U^\perp \to U^\perp$</span>. The operator norm $\lambda$ of <span>$P|_{U^\perp}$</span> is equal to <span>$\lambda = sup_\mu|g(\mu)|$</span> where $g$ is the polynomial. 

The subspace $\mathcal{M}\in \mathbb{R}^{N\times C}$ by a basis and a vector as:
$$
\mathcal{M}:=U \otimes \mathbb{R}^{C}=\left\{\sum_{m=1}^{M} e_{m} \otimes w_{m} \mid w_{m} \in \mathbb{R}^{C}\right\}
$$
The distance between a vector representation and the subspace is:
$$
d_{\mathcal{M}}(X):=\inf \left\{\|X-Y\|_{\mathrm{F}} \mid Y \in \mathcal{M}\right\}
$$
which is the closest Frounbies norm to the subspace. 

The maximum singular value of non-linear transform $W_{lh}$ is denoted by $s_l = \prod_{h=1}^Hs_{lh}$



一个问题：为什么要在一个subspace上看这个问题呢

##### Theorem 

$$
d_{\mathcal{M}}\left(f_{l}(X)\right) \leq s_{l} \lambda d_{\mathcal{M}}(X)
$$

for any $X\in \mathbb{R}^{N\times C}$ which non-linear operation $\sigma$ decreases the distance $d_{\mathcal{M}}$

This theorem can be proved from three basic lemma

**Lemma1**
$$
d_{\mathcal{M}}\left(PX\right) \leq \lambda d_{\mathcal{M}}(X)
$$
Give a subspace $\mathcal{M} \in (e_m)_{m\in[M]}$, and any vector <span>$X \in \mathbb{R}^{N\times C}$</span> can be written as `$X=\sum_{m=1}^{N} e_{m} \otimes w_{m}$`. Given the distance to the subspace `$d^2_{\mathcal{M}}(X)=\sum^N_{m=M+1}||w_m||^2$`. 

Then $PX$ can be written as 
$$
\begin{aligned}
P X &=\sum_{m=1}^{N} P e_{m} \otimes w_{m} \\
&=\sum_{m=1}^{M} P e_{m} \otimes w_{m}+\sum_{m=M+1}^{N} P e_{m} \otimes w_{m} \\
&=\sum_{m=1}^{M} P e_{m} \otimes w_{m}+\sum_{m=M+1}^{N} e_{m} \otimes\left(\lambda_{m} w_{m}\right)
\end{aligned}
$$
(The first tem will becomes 0 after the minimal mapping) 

The second term can be rewritten as a linear combination of the eigenvectors. then the distance can be writtern as
$$
\begin{aligned}
d_{\mathcal{M}}^{2}(P X) &=\sum_{m=M+1}^{N}\left\|\lambda_{m} w_{m}\right\|^{2} \\
& \leq \lambda^{2} \sum_{m=M+1}^{N}\left\|w_{m}\right\|^{2} \\
& \leq \lambda^{2} \sum_{m=M+1}^{N}\left\|w_{m}\right\|^{2} \\
&=\lambda^{2} d_{\mathcal{M}}^{2}(X)
\end{aligned}
$$
where $\lambda$ is the supermum of the $\lambda$.

**Lemma 2** 
$$
d_{\mathcal{M}}\left(XW_{lh}\right) \leq s_{lh} d_{\mathcal{M}}(X)
$$
The prove is the same with the first lemma, whether the matrix $W$ or $P$ comes from right or left side not matters a lot for $P$ is a symmetric matrix.

**Lemma 3**
$$
d_{\mathcal{M}}\left(\sigma(X)\right) \leq  d_{\mathcal{M}}(X)
$$
The proof of lemma three is different than the first two, for the activation function is element-wise, not vector-wise. First, we need to change the expression of the basis from both the node number $n$ and the $d$ dimension size. Let $(e_c')_{c\in [C]}$ be the standard basis of $\mathbb{R}^C$. The norm is more like a matrix form: `$(e_n \otimes e_c')_{c\in [C], n\in [N]}$`. any $X$ can be decoupled into:

$$
X = \sum_{n=1}^N  \sum_{c=1}^C a_{nc}e_n \otimes e_c'
$$

Then
$$
\begin{aligned}
d_{\mathcal{M}}^{2}(X) &=\sum_{n=M+1}^{N}\left\|\sum_{c=1}^{C} a_{n c} e_{c}^{\prime}\right\|^2 \\
&=\sum_{n=M+1}^{N} \sum_{c=1}^{C} a_{n c}^{2} \\
&=\sum_{c=1}^{C}\left(\sum_{n=1}^{N} a_{n c}^{2}-\sum_{n=1}^{M} a_{n c}^{2}\right) \\
&=\sum_{c=1}^{C}\left(\left\|X_{\cdot}\right\|^{2}-\sum_{n=1}^{M}\left\langle X_{\cdot c}, e_{n}\right\rangle^{2}\right)
\end{aligned}
$$
The distance can be written as $d_{\mathcal{M}}^{2}(\sigma(X) = \sum_{c=1}^{C}\left(\left\|X_{\cdot c}^+\right\|^{2}-\sum_{n=1}^{M}\left\langle X_{\cdot c}^+, e_{n}\right\rangle^{2}\right)$

**TODO:** add the proof this part.

Then for GNN with $\mathcal{M}$ as the eigenvector to the largest eigenvalues, if will falls exponentially into the eigenspace when $s\lambda < 1$\



### Spatial: understanding and solution inspired by PageRank

#### Inspiration from ML method

Actually, the former theorem also can somehow be viewed as the extension of the pagerank (markov) problem. 

The fundmental theory is: any Markov process on finite states converges to a unique distribution (equilibrium) (stationary distribution) if it is irreducible and aperiodic. 

- A markov chain can be describe with the initial distribution $\pi_0$ corresponds to the state space $S$. Each step will transition according to the current step with the Probability transition matrix $P\in \mathbb{R}^{n\times n}$.

- Stationary distribution means reach a unchanged state 
  $$
  \tilde{\pi} = \tilde{\pi} P
  $$

- A markov chain can have: 0, 1, $\infty$ stationary distribution, to keep a unique one, the following properties should be satisified

  - irreducible: in any step, <span>$P(X_t=i|X_0=j) > 0$</span>. All nodes can be reached
  - aperiodic: no derminstic movement: after $t$ steps, particular node will return to its position
  - positive recurrent: all nodes can be reached no matter which node the process originally starts with.

PageRank with random walk is an algorithm with markov proptery on graph, we will detail it on different versions in other blog.



#### JKNet: Representation Learning on Graphs with Jumping Knowledge Networks

##### Analysis

**The range of “neighboring” nodes that a node’s representation draws from strongly depends on the graph structure, analogous to the spread of a random walk.**

The basic analysis tool is the sensitivity analysis (influence distribution) inspired by page rank.

The motivation is that the influence from different nodes will heavily affected by the graph structure. For example, 

![](https://pica.zhimg.com/80/v2-2cbe3f6c087bd630ebeb4273e402219d_1440w.png)

With the same step but node with different space position, the reachable node neighbor has significant differently.  

Differences makes us to think that whether large or small neighborhood is good. The answer is neither. 

- Too much neighbor with higher-order features where some of the information may be “washed out” via averaging
- Less neighbor is less informative with not much information

What we need is the changable locality to different nodes. 

To quantify how the neighbor influence the other nodes, influence distribution is proposed, which gives insight into how large a neighborhood a node is drawing information from.

The influence distribution is defined as:

For a simple graph $G = (V, E)$, let `$h^{(0)}_x$` be the input feature and `$h^{(k)}_x$` be the learned hidden feature of node $x \in V $at the k-th (last) layer of the model. The influence score $I(x, y)$ of node $x$ by any node $y \in V$ is the sum of the absolute values of the entries of the Jacobian matrix $\frac{\partial h_{x}^{(k)}}{\partial h_{y}^{(0)}}$ . We define the influence distribution $I_x$ of $x \in V$ by normalizing the influence scores: $I_x(y)=I(x,y)/ \sum_z I(x, z)$, or
$$
I_{x}(y)=e^{T}\left[\frac{\partial h_{x}^{(k)}}{\partial h_{y}^{(0)}}\right] e /\left(\sum_{z \in V} e^{T}\left[\frac{\partial h_{x}^{(k)}}{\partial h_{z}^{(0)}}\right] e\right)
$$
where $e$ is the all-ones vector.

**The finding is **

Influence distributions of common aggregation schemes are closely connected to random walk distribution, which has a limitation(stationary) distribution (graph is non-bipartite).

**Theorem **

Given a $k$-layer GCN with averaging aggregation, assume that all paths in the computation graph of the model are activated with the same probability of success $\rho$. Then the influence distribution $I_x$ for any node $x \in V$ is equivalent, in expectation, to the $k$-step random walk distribution on $\tilde{G}$ starting at node $x$.

It is proved by 

- The one-step differentiate step can be described with non-linear activation mark, degree, weight. Then
  $$
  \begin{aligned}
  \frac{\partial h_{x}^{(k)}}{\partial h_{y}^{(0)}} &=\sum_{p=1}^{\Psi}\left[\frac{\partial h_{x}^{(k)}}{\partial h_{y}^{(0)}}\right]_{p} \\
  &=\sum_{p=1}^{\Psi} \prod_{l=k}^{1} \frac{1}{\widetilde{\operatorname{deg}}\left(v_{p}^{l}\right)} \cdot \operatorname{diag}\left(1_{f_{v_{p}^{l}}^{(l)}>0}\right) \cdot W_{l}
  \end{aligned}
  $$
  where $\Psi$ is the total number of paths, and the $\prod$ computed on each node on the path.

- Then for a single node, it can be rewritten as:
  $$
  \left[\frac{\partial h_{x}^{(k)}}{\partial h_{y}^{(0)}}\right]_{p}^{(i, j)}=\prod_{l=k}^{1} \frac{1}{\tilde{\operatorname{deg}}\left(v_{p}^{l}\right)} \sum_{q=1}^{\Phi} Z_{q} \prod_{l=k}^{1} w_{q}^{(l)}
  $$
  $Z_q$ is the probablity of activation or not.  The simplification is made over here. The assumption is that **The activation is a probability with no relation to the weight and input, just a prob**. Then the non-linear can be easily through aways as
  $$
  \mathbb{E}\left [ \left[\frac{\partial h_{x}^{(k)}}{\partial h_{y}^{(0)}}\right]_{p} \right ] = \rho \cdot \prod^1_{l=k}W_l \cdot \left(\sum_{p=1}^{\Psi} \prod_{l=k}^{1} \frac{1}{\widetilde{\operatorname{deg}}\left(v_{p}^{l}\right)}\right)
  $$
  The random probablity is just the last term. Actually, aggregation is just a random walk form.

- The distribution will be a little change with GCN symmetric form, beening normalized by $(\widetilde{\operatorname{deg}}(x)\widetilde{\operatorname{deg}}(y))^{-\frac{1}{2}}$   

**W**在这里会起到什么作用

Then we can unify GCN with the random walk, both of them will share a same stationary distribution.

##### Method

**Adapt to different local neighborhood range to enable to better adapt structure-aware representations.**

The model is very simple:

- concate hidden representation from different layer with different range of neighbor
- readout: Maxpooling, LSTM attention

It is to determine the importance of different ranges after looking on all of them. Maxpooling will find the suitable layer with the maximum influences

#### APPNP: PREDICT THEN PROPAGATE: GRAPH NEURAL NETWORKS MEET PERSONALIZED PAGERANK

Inspired by the JKNet, which gives an unifying view of random walk (pagerank). The random walk will result in a stationary distribution (oversmooth) regardless of which node starts with.  (need to check what the stationary distribution is.)

Thus, to remain the connection to the original node, it is natural to use the personal pagerank which gives a chance to return to the root node. **To preserve locality and avoid oversmooth**. This allows network with more large range of neighborhoods.

The personal pagerank takes the form: 
`$\boldsymbol{\pi}_{\mathrm{ppr}}\left(\boldsymbol{i}_{x}\right)=(1-\alpha) \tilde{\boldsymbol{A}} \boldsymbol{\pi}_{\mathrm{ppr}}\left(\boldsymbol{i}_{x}\right)+\alpha \boldsymbol{i}_{x}$`. 
The solution will be:
$$
\pi_{ppr}(i_x) = \alpha(I_n - (1-\alpha)\hat{\tilde{A}})^{-1}i_x
$$
With the stationary distribution, the stationary hidden representation can be written as:
$$
Z_{APPNP} = \text{softmax}\left(\alpha(I_n - (1-\alpha)\hat{\tilde{A}})^{-1}H\right)
$$
where $H = f_\theta(X)$. Naturally, the transformation is seperate from the aggregation.   this allows us to achieve a much higher range without changing the neural network(possible benefit will be discussed in later paper)

However, $\pi_{ppr}$ is a quite dense matrix which is computational expensive. An approximate version: topic-sensitive PageRank via power iteration.
$$
\begin{aligned}
\boldsymbol{Z}^{(0)} &=\boldsymbol{H}=f_{\theta}(\boldsymbol{X}) \\
\boldsymbol{Z}^{(k+1)} &=(1-\alpha) \hat{\tilde{A}} \boldsymbol{Z}^{(k)}+\alpha \boldsymbol{H} \\
\boldsymbol{Z}^{(K)} &=\operatorname{softmax}\left((1-\alpha) \hat{\tilde{A}} \boldsymbol{Z}^{(K-1)}+\alpha \boldsymbol{H}\right)
\end{aligned}
$$

#### GPRGNN: ADAPTIVE UNIVERSAL GENERALIZED PAGERANK GRAPH NEURAL NETWORK

As JKNet propose the maxp pooling function to select on different layers, GPRGNN uses a learnable parameters on  generalized pagerank for the layer selection.  原来的GPRGNN是咋优化的，这个优化不是很好理解。

##### What advances in Generalized pagerank

It is first proposed for graph clustering. The GPR takes the form as: 
`$\sum_{k=0}^{\infty} \gamma_{k} \tilde{\mathbf{A}}_{\mathrm{sym}}^{k} \mathbf{H}^{(0)}=\sum_{k=0}^{\infty} \gamma_{k} \mathbf{H}^{(k)}$`.  

Clustering of the graph is performed locally by thresholding the GPR score. 

Other pagerank can be viewed as a specific choice of GPR. APPNP can be viewed as fixed $\gamma_k = \alpha(1-\alpha)^k$

The learnable $\gamma_{k} $ gives model ability to learn long or short range information adaptively. 

The final form is similar with APPNP:

$$
\begin{aligned}
\boldsymbol{H}^{(0)} &=\boldsymbol{H}=f_{\theta}(\boldsymbol{X}) \\
\boldsymbol{H}^{k} &=\hat{\tilde{A}} \boldsymbol{H}^{k-1} \\
\boldsymbol{H} &= \sum_{k=0}^K\gamma_k\boldsymbol{H}^k\\
\boldsymbol{\hat{P}} &=\operatorname{softmax}\left(\boldsymbol{Z}\right)
\end{aligned}
$$

![](https://pica.zhimg.com/80/v2-5c41016548fc64429fd69cfb6592806c_720w.png)

We can see that different graphs appears differently, while the heterophily graph requires more information from the further neighborhoods.

##### Theory properties:

The filter of GPRGNN is: $g_{\gamma, K}(\lambda)=\sum_{k=0}^{K} \gamma_{k} \lambda^{k}$

Assume $\sum \gamma = 1$ and $\gamma$ can be a minus number. if $\gamma > 0$, low-frequency filter, if $\gamma < 0$, high-frequency filter. 

lemma 1看了  

**TODO: lemma 2**

Assume the graph $G$ is connected and the training set contains nodes from each of the classes. Also assume that $k'$ is large enough so that the over-smoothing effect occurs for $H(k) , \forall k \ge k'$ which dominate the contribution to the final output Z. Then, the gradients of $\gamma$ and $\gamma$  are identical in sign for all $k \ge k'$ .

It means that when oversmooth happens, $\gamma_k$ will be 0

### Spectral: understanding and solution inspired by graph signal

#### Fundamental knowledge in graph signal process

A vector $x\in \mathbb{R}^n$ defined on the vertices of the graph is the graphs signal. The basic operations are: 

- variation $\Delta$: $\mathbb{R}^n\to \mathbb{R}$        $\Delta(x) = \sum_{(i,j)\in \mathcal{E}}(x(i)-x(j))^2 = x^tLx$   measure the smoothness
- $\tilde{D}$-inner product:        $(x,y)_{\tilde{D}} = \sum_{i\in \mathcal{V}}(d(i)+\gamma)x(i)y(i) = x^T\tilde{D}y$   measure the importance of the signal(put more weight on high degree nodes)  

The general form of graph signal as:
$$
\min{\Delta(u)}  \text{ subject to } (u, u)_{\tilde{D}}=1, (u, u_j)_{\tilde{D}}=1, j \in \{1, \cdots, n\}
$$
The solution will be:
$$
Lu = \lambda \tilde{D}u
$$
The generalized eigenvalue corresponds to the graph signal. 

The fourier base is defined as:

- fourier transform $Fx = \hat{x} = U^T\tilde{D}x$
- inverse fourier transform $F^{-1}\hat{x} = U\hat{x}$
- Graph filter: $\hat{y}(\lambda) = h(\lambda)\hat{x}(\lambda)$  which is equal to $y=h(\tilde{L}_{rw})x$ $h$ is the tylor expansion of $h$
- The noise-to-signal ratio is defined as <span>$\frac{||Z||_D}{||X||_D}$</span>

#### Revisiting Graph Neural Networks: All We Have is Low-Pass Filters

With these fundamental information, revisiting GNN from graphs signal process. The answer is with informative feature, GNN only perform low-pass filter for denoising without any non-linear propoerty. 

The motivation or this paper is： Why and when do graph neural networks work well for vertex classification? 

- Is there a condition GNN can work even without training? 
- Is there a condition GNN cannot work well? 

##### Assumption 1: Input features consist of low-frequency true features and noise. The true features have sufficient information for the machine learning task.

The experiment verify with different noise level with different frequency on MLP.

- compute Fourier basis $U$
- Add gaussian noise to input features 
- compute first $k$-component $\hat{X}_k=U[: k]^T\tilde{D}^{\frac{1}{2}}X$
- reconstruct feature $\hat{X}_k=\tilde{D}^{-\frac{1}{2}}U[: k]^T\hat{X}_k$

- Train MLP on the new feature with different frequencies

![](https://pic1.zhimg.com/80/v2-f9d7e67238cba133646be70c071bb2cd_1440w.png)

**TODO:** add more experimental explaination

GNN can provide low frequency smooth data.

##### Theory: bias-variance understanding on GNN

**TODO:** some review on the complexity and generalization

With the assumption that faeture is composed of the true feature $\bar{x}$ and noise $z(i)$, we have

**Lemma 5**    Suppose Assumption 4. For any $0 < \delta < 1/2$, with probability at least $1 − \delta$, we have
$$
\left\|\bar{X}-\tilde{A}_{r w}^{k} X\right\|_{D} \leq \sqrt{k \epsilon}\|\bar{X}\|_{D}+O(\sqrt{\log (1 / \delta) R(2 k)}) \mathbb{E}\left[\|Z\|_{D}\right]
$$
where $R(2k)$ is a probability that a random walk with a random initial vertex returns to the initial vertex after $2k$ steps.

The first and second terms are bias induced by filter and variance from the original noise. Bias will increase with more hops adjacent matrix with a speed of $O(\sqrt{\epsilon})$. where the variance will decrease like $O(1/deg^{k/2})$ 

Then the optimial $k$ is that: 

Suppose that `$\mathbb{E}[||Z||_D] \le \rho ||\bar{X}||_D$` for some $\rho = O(1)$. Let `$k^*$` be defined by `$k^*$` = $O(log(log(1/δ)\rho/\epsilon))$, and suppose that there exist constants $C_d$ and $\bar{d} > 1$ such that $R(2k) ≤ C_d/ \bar{d}^k$ for `$k \le k^*$` . Then, by choosing $k = k^*$ , the right-hand side of (6) is `$\tilde{O}( \sqrt{\epsilon})$`.

**TODO, find the prove**

understanding on GNN:

- GCN may falls on overfitting the intermediate representation
- SGC is similar to MLP with true feature

#### Scattering GCN: Overcoming Oversmoothness in Graph Convolutional Networks

Similar with the heterphoily jobs, it uses the band-pass filtering of graph signal for the low-pass signal only consider the local activation patterns. The neural pathway encode higher-order forms of regularity in graphs, with higher signal.

##### Geometric scattering 

is defined by the lazy random walk matrix:
$$
P=\frac{1}{2}(I_n+WD^{-1})
$$
where $x_t = P^Tx$ is the low frequency in the Geometric GNN.

The wavelet is then defined as:
$$
\left\{\begin{array}{l}
\boldsymbol{\Psi}_{0}:=\boldsymbol{I}_{n}-\boldsymbol{P} \\
\boldsymbol{\Psi}_{k}:=\boldsymbol{P}^{2^{k-1}}-\boldsymbol{P}^{2^{k}}=\boldsymbol{P}^{2^{k-1}}\left(\boldsymbol{I}_{n}-\boldsymbol{P}^{2^{k-1}}\right), \quad k \geq 1
\end{array}\right.
$$
The geometric is defined as:
$$
U_px=\boldsymbol{\Psi}_{k_m}|\boldsymbol{\Psi}_{k_{m-1}}|\boldsymbol{\Psi}_{k_1}x||
$$
which is stack of the element-wise absolute value non-linearity.

Then all features are combined together as:

![](https://pic1.zhimg.com/80/v2-b2ff9c810b4932180e01743a8a28e71a_1440w.png)

residual connection with a cutoff frequency.

Theory part only use some specific graph which GCN can not find but with scattering channels for better expressivity. like cyclic or bipartite.

#### S2GC: SIMPLE SPECTRAL GRAPH CONVOLUTION

This paper tries to extract the higher frequency (self loop and seletive layer) with the modified Markov Diffusion kernel, which tries to enlarge the receptive of GNN. similar with APPNP, another explanation and solution

##### Markov diffusion kernel

It is similar to the shortest path kernel we introduced before, which focuses on the co-occurrance on a markov chain.
$$
d_{i j}(K)=\left\|\mathbf{Z}(K)\left(\mathbf{x}_{i}(0)-\mathbf{x}_{j}(0)\right)\right\|_{2}^{2}
$$
where $Z(K)=\frac{1}{K}\sum_{k=1}^KT^k$, $T$ is the transition matrix (adjacent)   $T=A' = (D + I)^{-1/2}  ( A + I )  (D + I)^{-1/2}$

##### Method

It can be simply reduce to the form
$$
\hat{Y}=\text{softmax}(\frac{1}{K}\sum_{k=0}^K\tilde{T}^kXW)
$$
with the Laplacian regularization as:
$$
\min{ h^TLh +\frac{1}{2}||h_i - x_i||_2^2} = \min{\frac{1}{2}\left(\sum_{i, j=1}^{n} \widetilde{\mathbf{A}}_{i j}\left\|\frac{\mathbf{h}_{i}}{\sqrt{d_{i}}}-\frac{\mathbf{h}_{j}}{\sqrt{d_{j}}}\right\|_{2}^{2}\right)+\frac{1}{2}\left(\sum_{i=1}^{n}\left\|\mathbf{h}_{i}-\mathbf{x}_{i}\right\|_{2}^{2}\right)}
$$
Then add the self-loop as:
$$
\hat{Y}=\operatorname{softmax}\left(\frac{1}{K} \sum_{k=1}^{K}\left((1-\alpha) \widetilde{\mathbf{T}}^{k} \mathbf{X}+\alpha \mathbf{X}\right) \mathbf{W}\right)
$$

##### Theory analysis

**Theorem 1** $N(\tilde{T}^0)\subseteq N(\tilde{T}^0)\subseteq N(\tilde{T}^1)\subseteq N \cdots(\tilde{T}^0)$ **smaller neighbor belongs to the larger neighborhoods**

**Theorem 2** the energy of infinite-dimensional receptive field (largest k) will not dominate the sum energy of our filter. (different pespective from oversquash)

**TODO reading**



### Understand GNN as a recursive procedure

GNN stacks different orders of neighborhood sequentially which first aggregation the first order neighbor, then the second in an recursive way as following

![](https://pic1.zhimg.com/80/v2-00a7063891981d3a14cd8d8b13a44e16_1440w.png)

Then the focus on GNN is the drawback on this procedure, and what is the best way to learn from the multi-hop neighborhood.

#### Oversquash: ON THE BOTTLENECK OF GRAPH NEURAL NETWORKS AND ITS PRACTICAL IMPLICATIONS

Oversquash is also a problem in RNN, which increasing growth information is referred into a fixed-size representation space. GNN will have exponential propogation messgae which may fails from distant nodes and build the long-range dependence.

experiments shows that GNN always underfitting on the condition that fitting the tree-structure graph. Empicially, GNN will overfiting short-range signal rather than the long-range information squashed in the bottleneck.

The solution is that: add a direction between two nodes. An easy solution is to build a fully-connected GNN layer. (This is also the reason why graphormer can help) 

other ablation finds that: larger hidden dimension do not have significant improvement. 

Even half fully-connnected can help a lot.

All directed interaction is not neccerary needed without graph structure.

#### ADAGCN: ADABOOSTING GRAPH CONVOLUTIONAL NETWORKS INTO DEEP MODELS

On this sequencial behavior, Adaboost is a good solution for the sequential relationship between different orders. To use a RNN-like GCN with iterative updating of the node weights. 

our AdaGCN also follows this direction by choosing an appropriate f in each layer rather than directly deepen GCN layers

The base classifier is designed as: 
$$
Z^l = f_\theta(\hat{A}^lX)
$$
with only a linear transformation.

adaboost is defined as:
$$
\begin{aligned}
e r r^{(l)} &=\sum_{i=1}^{n} w_{i} \mathbb{I}\left(c_{i} \neq f_{\theta}^{(l)}\left(x_{i}\right)\right) / \sum_{i=1}^{n} w_{i} \\
\alpha^{(l)} &=\log \frac{1-e r r^{(l)}}{\operatorname{err}(l)}+\log (K-1)
\end{aligned}
$$
What does $K$ means
$$
w_{i} \leftarrow w_{i} \cdot \exp \left(\alpha^{(l)} \cdot \mathbb{I}\left(c_{i} \neq f_{\theta}^{(l)}\left(x_{i}\right)\right)\right), i=1, \ldots, n
$$
The different is that $f_\theta$ is shared by different layer but with different parameter.





### Advanced operation

#### architectural modifications: additional connection on model architecture

Inspired by the residual connection in CV, GNN also design specificed residual connection. JKNet and S2GC can be viewed as the Dense connection in GNN. residual connection in GNN, however, can only prevent fast performance degrade but not enhance performance. New perspective should be proposed.

##### GCNII: Simple and Deep Graph Convolutional Networks

It propose two simple yet effective techniques: Initial residual and Identity mapping. 

The final form:
$$
\mathbf{H}^{(\ell+1)}=\sigma\left(\left(\left(1-\alpha_{\ell}\right) \tilde{\mathbf{P}} \mathbf{H}^{(\ell)}+\alpha_{\ell} \mathbf{H}^{(0)}\right)\left(\left(1-\beta_{\ell}\right) \mathbf{I}_{n}+\beta_{\ell} \mathbf{W}^{(\ell)}\right)\right)
$$
Initial residual is somehow similar with the feature similarity preserve.  **A question: if large embedding size, use a MLP layer**.

This is somehow different with the APPNP with linear combination, it indeed makes deep with non-linear transformation. 

Identity mapping is similar with res connection, but with also the influence on the non-linearity and the initial residual. **Hard to find so much difference**

- It will increase the maximum singular value which will reduce $s\lambda < 1$
- small the norm of $W$, put strong regularization on $W^l$ to avoid overfiting

**Theory part**

**Theorem 1** Assume the self-looped graph $\tilde{G}$ is connected. Let $h^{(K)} = ( \frac{I_n+\tilde{D} {−1/2}\tilde{A}\tilde{ D}^ {−1/2}}{2})^K ·x$ denote the representation by applying a K-layer renormalized graph convolution with residual connection to a graph signal $x$. Let $\lambda \tilde{G}$ denote the spectral gap of the self-looped graph $\tilde{G}$, that is, the least nonzero eigenvalue of the normalized Laplacian $\tilde{L} = I_n − \tilde{D} ^{−1/2}\tilde{A} \tilde{D}^{ −1/2}$ . We have 

1) As K goes to infinity, $h (K)$ converges to $\pi = <\frac{\tilde{D}^{1/2}1,x>} {2m+n}\cdot \tilde{D}^{1/2}1$, where 1 denotes an all-one vector. 

2) The convergence rate is determined by
$$
\mathbf{h}^{(K)}=\pi \pm\left(\sum_{i=1}^{n} x_{i}\right) \cdot\left(1-\frac{\lambda_{\tilde{G}}^{2}}{2}\right)^{K} \cdot \mathbf{1}
$$
The prove keys are: 

- split the origin $h^K$ with linear combination of basis (which can represents the random walk)
  $$
  \tilde{\mathbf{D}}^{1 / 2} \mathbf{x}=\left(\mathbf{D}+\mathbf{I}_{n}\right)^{1 / 2} \mathbf{x}=\sum_{i=1}^{n}\left(\mathbf{x}(i) \sqrt{d_{i}+1}\right) \cdot \mathbf{e}_{\mathbf{i}}
  $$
  
- use the lemma Let $p^(K)_i =  (\frac{I_n+\tilde{A} \tilde{D}^{−1}}{2})^Ke_i$  is the K-th transition probability vector from node i on connected self-looped graph $\tilde{G}$. Let $\lambda\tilde{G}$ denote the spectral gap of $\tilde{G}$. The j-th entry of $p^{(K)}_i$ can be bounded by
  $$
  \left|\mathbf{p}_{i}^{(K)}(j)-\frac{d_{j}+1}{2 m+n}\right| \leq \sqrt{\frac{d_{j}+1}{d_{i}+1}}\left(1-\frac{\lambda_{\tilde{G}}^{2}}{2}\right)^{K} .
  $$

Theorem 2

Consider the self-looped graph $\tilde{G}$ and a graph signal $x$. A $K$-layer GCNII can express a $K$ order polynomial filter $\sum_{l=0}^K\theta_l\tilde{L}^l)x$ with arbitrary coefficients $\theta$

Too much assumption, not so good.



#### regularization & normalization trick

The most two popular regularization methods are batch norm and dropout. However, dropout can not work well on the graph architecture. Batchnorm can not capture the relation between nodes. 

Various methods has been proposed for both enhance generalization ability, reduce overfitting, reduce oversmooth. 

##### DROPEDGE: TOWARDS DEEP GRAPH CONVOLUTIONAL NETWORKS ON NODE CLASSIFICATION

Dropedge is a natural extension of dropedge. Two way to view it:

- A data augmenter: similar with dropout
- A mesage reducer: reduce some neighborhoods
  - slow down the convergence speed, the relaxed $\epsilon$-smooth will only increase. 
  - Smaller gap between the origin feature and the convergence subspace

Dropedge is two-step as:

- $A_{drop} = A - A'$
- Renormalization $A_{drop} \to \hat{A}_{drop}$



**proof 1 ** 证的不咋地

$$
\hat{l}(\mathcal{M}, \epsilon) \le \hat{l}(\mathcal{M}', \epsilon)
$$

$\epsilon$ smoothness is designed as the layer with `$l^{*}(\mathcal{M}, \epsilon):=\min _{l}\left\{d_{\mathcal{M}}\left(\boldsymbol{H}^{(l)}\right)<\epsilon\right\}$`

The relaxed $\epsilon$-smoothing is the upper bound of $\epsilon$-smooth as 
$$
\hat{l}(\mathcal{M}, \epsilon):=  \left\lceil\frac{\log \left(\epsilon / d_{\mathcal{M}}(\boldsymbol{X})\right)}{\log s \lambda}\right\rceil
$$
where $s$ is the largest eigenvalue, $\lambda$ is the second largest eigenvalue of $\hat{A}$

We also need to adopt some concepts from Lovász et al. (1993) in proving Theorem 1. Consider the graph $G$ as an electrical network, where each edge represents an unit resistance. Then the effective resistance, $R_{st}$ from node $s$ to node $t$ is defined as the total resistance between node $s$ and $t$. According to Corollary 3.3 and Theorem 4.1 (i) in Lovász et al. (1993), we can build the connection between $\lambda$ and $R_{st}$ for each connected component via commute time as the following inequality.

Remove any edge will cause $R_{st}$ increase, if remove into two bipatite, more dimension of information will be remained.

##### GRAND: Graph Random Neural Networks for Semi-Supervised Learning on Graph

Take the idea of Dropedge, more tricks based on dropout also with contrastive loss is proposed. Grand has the similar advantage,  a new advantage is enhance robustness. 

Grand is two-fold:

- Each node features can be randomly drop either partially (dropout) or entirely (dropnode)

- decouple propogation and transformation  $\bar{A}=\sum^K_{k=0}\frac{1}{K+1}\hat{A}^k$, then MLP

- An additional consistency regularization for different views.
  $$
  \mathcal{L}_{\text {con }}=\frac{1}{S} \sum_{s=1}^{S} \sum_{i=0}^{n-1}\left\|\overline{\mathbf{Z}}_{i}^{\prime}-\widetilde{\mathbf{Z}}_{i}^{(s)}\right\|_{2}^{2}
  $$
  where $\bar{Z}_i'$ is the mean of different views.

The theorem is somehow little trivial with understanding on how these loss as the regularization term.

##### PAIRNORM: TACKLING OVERSMOOTHING IN GNNS

From the perspective of Batchnorm, the regularization normalization on GNN is proposed, which provents all node embedding becoming too similar.

The idea is somehow similar with feature similarity preservation.

**Analysis**

understanding that most GNNs perform a special form of Laplacian smoothing, which makes node features more similar to one another. The key idea is to ensure that the total pairwise feature distances remains a constant across layers, which in turn leads to distant pairs having less similar features, preventing feature mixing across clusters.

Two measurements are proposed:
$$
\begin{array}{l}
\text { row-diff }\left(\mathbf{H}^{(k)}\right)=\frac{1}{n^{2}} \sum_{i, j \in[n]}\left\|\mathbf{h}_{i}^{(k)}-\mathbf{h}_{j}^{(k)}\right\|_{2} \\
\text { col-diff }\left(\mathbf{H}^{(k)}\right)=\frac{1}{d^{2}} \sum_{i, j \in[d]}\left\|\mathbf{h}_{\cdot i}^{(k)} /\right\| \mathbf{h}_{\cdot i}^{(k)}\left\|_{1}-\mathbf{h}_{\cdot j}^{(k)} /\right\| \mathbf{h}_{\cdot j}^{(k)}\left\|_{1}\right\|_{2}
\end{array}
$$
where row-diff measure is the average of all pairwise distances between node features, quantifies node-wise oversmoothing. col-diff quantifies feature-wise smoothness.  

其实感觉col diff会减小其实挺奇怪的。 Think of the reason on measurement here.

![](https://pic3.zhimg.com/80/v2-b3444000427aae847a13ba49b84e2642_1440w.png)

The reason why row-difference changes so sharply is still under discussion.

**A new interpertation: graph-regularized least squares**
$$
\min _{\overline{\mathbf{X}}} \sum_{i \in \mathcal{V}}\left\|\overline{\mathbf{x}}_{i}-\mathbf{x}_{i}\right\|_{\tilde{\mathbf{D}}}^{2}+\sum_{(i, j) \in \mathcal{E}}\left\|\overline{\mathbf{x}}_{i}-\overline{\mathbf{x}}_{j}\right\|_{2}^{2}
$$
where $\bar{X}\in \mathbb{R}^{N\times d}$,  $\left\|z_i\right\|_{\tilde{\mathbf{D}}}^{2} = z_i^T\tilde{\mathbf{D}}z_i$, with a closed form solution $\bar{X}=(2I - \tilde{A}_{rw})^{-1}X$

其实重要的就是保护对应的表达空间。 We should not only consider smooth the same cluster, but also distant disconnected pairs
$$
\min _{\overline{\mathbf{X}}} \sum_{i \in \mathcal{V}}\left\|\overline{\mathbf{x}}_{i}-\mathbf{x}_{i}\right\|_{\tilde{\mathbf{D}}}^{2}+\sum_{(i, j) \in \mathcal{E}}\left\|\overline{\mathbf{x}}_{i}-\overline{\mathbf{x}}_{j}\right\|_{2}^{2} - \lambda \sum_{(i, j) \notin \mathcal{E}}\left\|\overline{\mathbf{x}}_{i}-\overline{\mathbf{x}}_{j}\right\|_{2}^{2}
$$
**The distance should keep the same in total**
$$
\sum_{(i, j) \in \mathcal{E}}\left\|\dot{\mathbf{x}}_{i}-\dot{\mathbf{x}}_{j}\right\|_{2}^{2}+\sum_{(i, j) \notin \mathcal{E}}\left\|\dot{\mathbf{x}}_{i}-\dot{\mathbf{x}}_{j}\right\|_{2}^{2}=\sum_{(i, j) \in \mathcal{E}}\left\|\mathbf{x}_{i}-\mathbf{x}_{j}\right\|_{2}^{2}+\sum_{(i, j) \notin \mathcal{E}}\left\|\mathbf{x}_{i}-\mathbf{x}_{j}\right\|_{2}^{2}
$$
To avoid high computional cost, the computational step is:
$$
\operatorname{TPSD}(\tilde{\mathbf{X}})=\sum_{i, j \in[n]}\left\|\tilde{\mathbf{x}}_{i}-\tilde{\mathbf{x}}_{j}\right\|_{2}^{2}=2 n^{2}\left(\frac{1}{n} \sum_{i=1}^{n}\left\|\tilde{\mathbf{x}}_{i}\right\|_{2}^{2}-\left\|\frac{1}{n} \sum_{i=1}^{n} \tilde{\mathbf{x}}_{i}\right\|_{2}^{2}\right)
$$
Further simplify will be $\operatorname{TPSD}(\tilde{\mathbf{X}}) = \operatorname{TPSD}(\tilde{\mathbf{X}}^c) = 2n||\tilde{\mathbf{X}}^c||^2_F$ which $X^c = X - \bar{X}$, the center representation

The final procedure are center and rescale:
$$
\begin{array}{l}
\tilde{\mathbf{x}}_{i}^{c}=\tilde{\mathbf{x}}_{i}-\frac{1}{n} \sum_{i=1}^{n} \tilde{\mathbf{x}}_{i} \\
\dot{\mathbf{x}}_{i}=s \cdot \frac{\tilde{\mathbf{x}}_{i}^{c}}{\sqrt{\frac{1}{n} \sum_{i=1}^{n}\left\|\tilde{\mathbf{x}}_{i}^{c}\right\|_{2}^{2}}}=s \sqrt{n} \cdot \frac{\tilde{\mathbf{x}}_{i}^{c}}{\sqrt{\left\|\tilde{\mathbf{X}}^{c}\right\|_{F}^{2}}}
\end{array}
$$
![](https://pic2.zhimg.com/80/v2-5da52b1f8002e01f5c028bbecd93e142_1440w.png)

##### Towards Deeper Graph Neural Networks with Differentiable Group Normalization

GroupNorm gives us another interpretation on oversmooth, inter-class distance is smaller than intra-class distance. Moreover, it is somehow similar with the diffpooling.

**The main ignore of the pairnorm ignore the same group nodes without connection**. Nodes within the same community/class need be similar to facilitate the classification, while different classes are expected to be separated in embedding space from a global perspectives of view. 

The challenges are:

- oversmooth related to local node relation (pairnorm) but also global structures (group)
- group information is hard to get on training.

**Analysis measurement**

group ratio is defined as:
$$
R_{\mathrm{Group}}=\frac{\frac{1}{(C-1)^{2}} \sum_{i \neq j}\left(\frac{1}{\left|\boldsymbol{L}_{i} \| \boldsymbol{L}_{j}\right|} \sum_{h_{i v} \in \boldsymbol{L}_{i}} \sum_{h_{j v^{\prime}} \in \boldsymbol{L}_{j}}\left\|h_{i v}-h_{j v^{\prime}}\right\|_{2}\right)}{\frac{1}{C} \sum_{i}\left(\frac{1}{\left|\boldsymbol{L}_{i}\right|^{2}} \sum_{h_{i v}, h_{i v^{\prime}} \in \boldsymbol{L}_{i}}\left\|h_{i v}-h_{i v^{\prime}}\right\|_{2}\right)}
$$
Isntance information Gain as the mutual information:
$$
G_{\text {Ins }}=I(\mathcal{X} ; \mathcal{H})=\sum_{x_{v} \in \mathcal{X}, h_{v} \in \mathcal{H}} P_{\mathcal{X H}}\left(x_{v}, h_{v}\right) \log \frac{P_{\mathcal{X} \mathcal{H}}\left(x_{v}, h_{v}\right)}{P_{\mathcal{X}}\left(x_{v}\right) P_{\mathcal{H}}\left(h_{v}\right)}
$$
The representation

这套文章提出来的指标都是自己解决好的指标，而不是为了寻找什么insight。

**GroupNorm:** normalize the node embeddings group by group. each group to be rescale to be more similar. 

The two steps are:

- assign each node with learnable group $S= \text{softmax}\left (H^{(k)}U^{(k)} \right)$
- rescale on each group:  $H_i^k = S^k[:, i] \circ H^k$   $H_i^k=\gamma_i(\frac{H_i^k-\mu_i}{\sigma_i})+\beta_i$
- Strength the importance of origin information $\tilde{H}^k = H^k+\lambda\sum_{i=1}^G\tilde{H}_i^k$

#### Benchmark: Bag of Tricks for Training Deeper Graph Neural Networks: A Comprehensive Benchmark Study



The empirical study has the findings:

- As we empirically show, while initial connection and jumping connection are both “beneficial" training tricks when applied alone, combining them together deteriorates deep GNN performance. 
- Although dense connection brings considerable improvement on large-scale graphs with deep GNNs, it sacrifices the training stability to a severe extent. 
- As another example, the gain from NodeNorm become diminishing when applied to large-scale datasets or deeper GNN backbones. 
- Moreover, using random dropping techniques alone often yield unsatisfactory performance. 
- Lastly, we observe that adopting initial connection and group normalization is universally effective across tens of classical graph datasets. Those findings urge more synergistic rethinking of those seminal works



skip connection group:

- skip connneciton can accrelate the training
- shallow model is not suitable for skip connection (except on large dataset (maybe noise) )
- SGC benefits from skip conneciton

Normalization 

- NodeNorm  `$\left(\mathbf{x}_{i} ; p\right)=\frac{\mathbf{x}_{i}}{\operatorname{std}\left(\mathbf{x}_{i}\right)^{\frac{1}{p}}}$`
- MeanNorm `$\left(\mathbf{x}_{(k)}\right)=\mathbf{x}_{(k)}-\mathbb{E}\left[\mathbf{x}_{(k)}\right] $`
- BatchNorm `$\left(\mathbf{x}_{(k)}\right)=\gamma \cdot \frac{\mathbf{x}_{(k)}-\mathbb{E}\left[\mathbf{x}_{(k)}\right]}{s t d\left(\mathbf{x}_{(k)}\right)}+\beta $`

observation:

- training with norm is much stable
- node norm and pairnorm perform well on small dataset, while group norm on the larger dataset.



Drop observation

- dropout for shallow GNN 
- drop technique suitable for random dropping



### Tradeoff between Neighborhood size and neural network depth

The difference between GNN and basic MLP is the aggregation  (neighbor size). In the traditional GNN, more neighborhood means more parameter leads to overfiting. Many paper propose the decouple transformation and aggregation. So what is the key reason for oversmooth?

#### DAGNN: Towards Deeper Graph Neural Networks

The key factor compromising the performance is entanglement of representation transformation and propogation.Decouple is the key component.



##### Analysis: Quantitative metric for smoothness

$$
D(x_i, x_j)=\frac{1}{2}||\frac{x_i}{||x_i||} - \frac{x_j}{||x_j||} ||
$$

where `$||\cdot||$` denotes the Euclidean norm

The smothness score will decade quickly on well-trained GCN, however, disentangle do not comes down quickly with only linear propogation. The distangement architecture is:
$$
\begin{aligned}
Z &=\operatorname{MLP}(X) \\
X_{o u t} &=\operatorname{softmax}\left(\widehat{A}^{k} Z\right)
\end{aligned}
$$

##### Theoretically analysis

Nothing new that $D^{-1}A$ and $D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$ will converge into a vector.



##### Application

Design on DAGNN, it utilizes an adaptive adjustment mechanism that can adaptively balance the information from local and global neighborhoods for each node
$$
\begin{array}{ll}
Z=\operatorname{MLP}(X) & \in \mathbb{R}^{n \times c} \\
H_{\ell}=\widehat{A}^{\ell} Z, \ell=1,2, \cdots, k & \in \mathbb{R}^{n \times c} \\
H=\operatorname{stack}\left(Z, H_{1}, \cdots, H_{k}\right) & \in \mathbb{R}^{n \times(k+1) \times c} \\
S=\sigma(H s) & \in \mathbb{R}^{n \times(k+1) \times 1} \\
\widetilde{S}=\operatorname{reshape}(S) & \in \mathbb{R}^{n \times 1 \times(k+1)} \\
X_{\text {out }}=\operatorname{softmax}(\text { squeeze }(\widetilde{S} H)) & \in \mathbb{R}^{n \times c}
\end{array}
$$
where $s\in \mathbb{R}^{n \times c}$ is a projection function, $c$ is the number of classes. 

#### Revisiting Oversmoothing in Deep GCNs

However, another point of view is proposed (the solution is related with spectral). which the transformation layer is learn to anti-oversmooth during training.  The understanding is: untrained GCN indeed oversmooth, but the learning procedure will lean to distuiguish against it. However, the model is not well-trained against the oversmoothness.

This paper propose the understanding: 

- The forward procedure is optimized the smoothness   **Here has a learning rate analysis**
  $$
  \begin{aligned}
  \nabla_{X} &=\frac{\partial R(X)}{\partial X}=\frac{1}{2} \frac{\partial \frac{\operatorname{Tr}\left(X^{\top} \Delta X\right)}{\operatorname{Tr}\left(X^{\top} X\right)}}{\partial X}=\frac{\left(\Delta-I \frac{\operatorname{Tr}\left(X^{\top} \Delta X\right)}{\operatorname{Tr}\left(X^{\top} X\right)}\right) X}{\operatorname{Tr}\left(X^{\top} X\right)} \\
  X_{m i d} &=X-\eta \nabla_{X}=\frac{(2-\Delta) X}{2-\frac{\operatorname{Tr}\left(X^{\top} \Delta X\right)}{\operatorname{Tr}\left(X^{\top} X\right)}}=\frac{\left(I+D^{-\frac{1}{2}} A D^{-\frac{1}{2}}\right)}{2-\frac{\operatorname{Tr}\left(X^{\top} \Delta X\right)}{\operatorname{Tr}\left(X^{\top} X\right)}} X
  \end{aligned}
  $$

- The backward is for the classification loss which will reduce the oversmoothness

**TODO read the proof**

 the connected nodes share similar representations ("similar" means the scale of each feature channel is approximately proportional to the square root of its degree) **为什么这个位置没有高阶的度数**

**Solution**

meansubtract which will approach the second large eigenvalue.

#### Evaluating Deep Graph Neural Networks

The most interesting in this paper may reject the last paper perspective, which even single MLP will fall into the oversmooth problem.  This paper study:

- The root problem why deep model performance decay happens in deeper GNN (oversmoothness?)
- when and how to build deeper GNN?



##### Experiment setting

- smoothness measurement
  The stationary state
  $$
  \hat{\mathrm{A}}_{i, j}^{\infty}=\frac{\left(d_{i}+1\right)^{r}\left(d_{j}+1\right)^{1-r}}{2 M+N}
  $$
  node smoothness: the similarity with the initialization state
  $$
  \begin{array}{c}
  \alpha=\operatorname{Sim}\left(\mathbf{x}_{v}^{k}, \mathbf{x}_{v}^{0}\right) \\
  \beta=\operatorname{Sim}\left(\mathbf{x}_{v}^{k}, \mathbf{x}_{v}^{\infty}\right) \\
  N S L_{v}(k)=\alpha *(1-\beta)
  \end{array}
  $$

- The number of transformation is $D_t$, the number of propogation is $D_p$

##### Misconception

###### Oversmoothness is not the main contributor

![](https://pic1.zhimg.com/80/v2-17653087fc0bd461acfdd35bca3a0476_1440w.png)

The experiment is to have double propogation and transofrmation to test the performance.

When aggregation double, the performance does not matters too much in layer 8 with 16 propogations. Oversmooth is not the main concept. Also，with less smoothness, the performance does not change too much. But with more parameters, performance indeed drops. 

So does more parameter cause the overfiting.

###### Overfiting is not the main concept

<img src="https://pic2.zhimg.com/80/v2-9916726f79543d0e4162f4a67699f5e7_1440w.png" style="zoom:67%;" />

GCN on both train and test accuracy drop which is not overfiting, which reach the train acc: 100%. It is underfitting.

###### Entangle and disentangle

Entanglement with residual will have much slower performance drop while disentangle model performance drop more quickly. 

##### The key cause

MLP with no residual will drop on this state when stack more MLP.

<img src="https://pica.zhimg.com/80/v2-23dd5af704bb5fe1f7d0d7b63bdb9abf_1440w.png" style="zoom:67%;" />

The performance will decade without residual connection.

- **Why need deep EP?** sparse graph (and large diameter) **How?** combine features in different steps.
- **Why need deep ET?** large graph with more information. **How** residual, jump connection



### Others in oversmoothness

#### MADGap: Measuring and Relieving the Over-smoothing Problem for Graph Neural Networks from the Topological View

This paper provide two quantity measurement for analysis, MAD for smoothness (similarity in nodes), and MADGap for oversmoothness, which measure the informaion-noise ratio (inter-class and intra-class). 

With these findings, the paper proposed MADgap regularization and adaedge to remove the intra-class edges.

The smoothness is measured by:
$$
D_{i j}=1-\frac{\boldsymbol{H}_{i,:} \cdot \boldsymbol{H}_{j,:}}{\left|\boldsymbol{H}_{i,:}\right| \cdot\left|\boldsymbol{H}_{j,:}\right|} \quad i, j \in[1,2, \cdots, n]
$$
The observation is that in the low layer, the information to noise ratio is larger with local neighborhoods.

MAD value of high-layer GNNs gets close to 0

The MADGAP is defined by 
$$
\text{MADGap}=MAD^{rmt}-MAD^{neb}
$$
rmt is the MAD value remote nodes in graph topology

The regularization is defined as the MADGap



### Reading list

- Evaluating Deep Graph Neural Networks

- On Provable Benefits of Depth in Training Graph Convolutional Networks (Towrite)

- ADAPTIVE UNIVERSAL GENERALIZED PAGERANK GRAPH NEURAL NETWORK

- Lipschitz Normalization for Self-Attention Layers with Application to Graph Neural Networks (Towrite)

- SIMPLE SPECTRAL GRAPH CONVOLUTION

- ADAGCN: ADABOOSTING GRAPH CONVOLUTIONAL NETWORKS INTO DEEP MODELS 

- DIRECTIONAL GRAPH NETWORKS (graph classification) not so related

- Graph Neural Networks Inspired by Classical Iterative Algorithms (Towrite)

- Two Sides of the Same Coin: Heterophily and Oversmoothing in Graph Convolutional Neural Networks (Towrite)

- Bag of Tricks for Training Deeper Graph Neural Networks: A Comprehensive Benchmark Study

- Training Graph Neural Networks with 1000 Layers  (Toread, not so related)

- Optimization of Graph Neural Networks: Implicit Acceleration by Skip Connections and More Depth (toread)

- GRAND: Graph Neural Diffusion(toread)

- ON THE BOTTLENECK OF GRAPH NEURAL NETWORKS AND ITS PRACTICAL IMPLICATIONS

- Revisiting "Over-smoothing" in Deep GCNs

- Evaluating deep graph neural networks

- Simple and Deep Graph Convolutional Networks

- DROPEDGE: TOWARDS DEEP GRAPH CONVOLUTIONAL NETWORKS ON NODE CLASSIFICATION

- PAIRNORM: TACKLING OVERSMOOTHING IN GNNS

- Measuring and Relieving the Over-smoothing Problem for Graph Neural Networks from the Topological View

- Continuous Graph Neural Networks (toread)

- Towards Deeper Graph Neural Networks

- GRAPH NEURAL NETWORKS EXPONENTIALLY LOSE EXPRESSIVE POWER FOR NODE CLASSIFICATION 

- MEASURING AND IMPROVING THE USE OF GRAPH INFORMATION IN GRAPH NEURAL NETWORKS

- Optimization and Generalization Analysis of Transduction through Gradient Boosting and Application to Multi-scale Graph 

  Neural Networks  (toread)

- Graph Random Neural Networks for Semi-Supervised Learning on Graphs

- scattering GCN: Overcoming Oversmoothness in Graph Convolutional Networks

- Towards Deeper Graph Neural Networks with Differentiable Group Normalization 

- Bayesian Graph Neural Networks with Adaptive Connection Sampling (toread)

- Predict then Propagate: Graph Neural Networks meet Personalized PageRank

- Representation Learning on Graphs with Jumping Knowledge Networks

- DeepGCNs: Can GCNs Go as Deep as CNNs? (image, not so related)

- Revisiting Graph Neural Networks: All We Have is Low-Pass Filters 



### reading

Intuitively, the desirable representation of node features does not necessarily need too many nonlinear transformation f applied on them. This is simply due to the fact that the feature of each node is normally one-dimensional sparse vector rather than multi-dimensional data structures, e.g., images, that intuitively need deep convolution network to extract high-level representation for vision tasks. This insight has been empirically demonstrated in many recent works, showing that a two-layer fully-connected neural networks is a better choice in the implementation.

