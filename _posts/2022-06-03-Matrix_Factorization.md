---
title: Matrix Factorization
layout: category
permalink: /categories/MF/
taxonomy: Matrix Factorization
---

# Matrix Factorization  -- a graph perspective

The matrix factorization is one important techinique on learning representation. The main application of the matrix factorization are three-folds.

- Dimensiton reduction or fit the missing entrance (low-rank recovery)
- Data cluster using the principal components $Q$ find the low rank representation for data compression or find the hidden cluster in the Matirx Factorization

For the low rank approximation with missing data, the matrix factorization can be formed as:
$$
\min_{U,V} ||W \odot (M-UV)||
$$
where $W$ is the mask for the existing entrance. $M$ is the input matrix. However, this optimization will be ill-posed, which will definitely leads to overfitting. Addition assumption has to be made, the most importance is that low rank. It can be explain in an hard method as: $UV$. or it can be utilized with a soft constraint->nuclear norm. However, this constraint is somehow computational expensive. 



Why matrix factorization? the matrix can be represented with a linear reductive structure, this structure will not change much on the different conditions, which is more robust. How to find this kind of subspace becomes a key problem. 



**attention please what is the really $L$, normalized or not. **



## Basic model

### PCA

Let the input data feature matrix $X=(x_1, \cdots , x_n)\in \mathbb{R}^{p\times n}$, PCA aims to find the optimal low-dimensional subspace with the principal direction $U=(u_1, \cdots , u_k)\in \mathbb{R}^{p\times k}$ , and the projected data point $V=(v_1, \cdots , v_n)\in \mathbb{R}^{n\times k}$.  It aims to minimize the covariance with the following loss function:  
$$
\min _{U, V}\left\|X-U V^{T}\right\|_{F}^{2} \text { s.t. } V^{T} V=I
$$
(projection direction和projected data point是我搞混了)

**Notice that, the data is already centelized in the dataset.**



### LDA



### Spectral Cluster





## Graph based method

In the above discussion, the input data $X$ is the only avaible vector data for learning a data representation.  And for the manifold learning and graph embedding methods, only the graph adjacent matrix $W$ is taking into consideration. However, there lacks of methods which takes both graph structure $W$ and the node feature $X$ into consideration.  

We first answer the question that how can those methods benefit from each other.

In most case, the feature-based matrix factorization on $X$ can only considers on the linear case, while the manifold learning can consider the local information (local linear -> global nonlinear relationship). They can find the data lies in a nonlinear data manifold.   (In some cases, there is no $W$, graph is constructed based on the feature similarity.)



### Graph-Laplacian PCA: Closed-form Solution and Robustness

To take both the PCA and the Laplacian embedding into one framework, the target is as follow:
$$
\begin{array}{l}
\min _{U, Q} J=\left\|X-U Q^{T}\right\|_{F}^{2}+\alpha \operatorname{Tr}\left(Q^{T}(D-W) Q\right) \\
\text { s.t. } Q^{T} Q=I
\end{array}
$$
 The solution is that:
$$
\begin{array}{l}
Q^{*}=\left(v_{1}, v_{2}, \cdots, v_{k}\right) \\
U^{*}=X Q^{*}
\end{array}
$$
 where $\left(v_{1}, v_{2}, \cdots, v_{k}\right)$ is the eigenvectors corresponding to the first $k$ smallest eigenvalues of $G = -X^TX + \alpha L$

这里不太明白的是矩阵的求导的过程。

We first need to find the solution by fix $Q$, then we find the result for $U$.

Finally, in order to balance two terms easily, or keep them in the same scale, the data is normalized by $\lambda_n$, the largest eigenvalue of $X^TX$, and the $\epsilon_n$, the largest eigenvalue of Laplacian matrix $L$.  



**A question: how to connect is with the graph neural networ, I think  dual may be a solution **



However, the above one is not as robust, a robust version with a weaken norm is given then
$$
\begin{array}{l}
\min _{U, Q} J=\left\|X-U Q^{T}\right\|_{2,1}^{2}+\alpha \operatorname{Tr}\left(Q^{T}(D-W) Q\right) \\
\text { s.t. } Q^{T} Q=I
\end{array}
$$
In order to give into a Augmented Lagrange Multiplier feasible form, it can be rewritten as:
$$
\begin{array}{l}
\min _{U, Q, E}\|E\|_{2,1}+\alpha \operatorname{Tr} Q^{T}(D-W) Q \\
\text { s.t. } E=X-U Q^{T}, Q^{T} Q=I
\end{array}
$$
**I do not know much about the proximal optimizer and Augmented Lagrange Multipler, also need knowledge on the gradient on the nrom**



Then, it can be written as the two argumented term of $E-X+U Q^{T}$ as:
$$
\begin{array}{l}
\min _{U, Q, E}\|E\|_{2,1}+\operatorname{Tr} C^{T}\left(E-X+U Q^{T}\right) \\
\quad+\frac{\mu}{2}\left\|E-X+U Q^{T}\right\|_{F}^{2}+\alpha \operatorname{Tr} Q^{T} L Q \\
\text { s.t. } Q^{T} Q=I
\end{array}
$$
Fix $E$, the problem is the same as the original one, while fixed $U$ and $Q$, the problem change as follows:
$$
\min _{E}\|E\|_{2,1}+\frac{\mu}{2}\|E-A\|_{F}^{2}
$$
where $A = X-UQ^T-C/\mu$, which is viewed as a group, which can then be viewed as $n$ independent matrix as: 
$$
\min _{e_{i}}\left\|e_{i}\right\|+\frac{\mu}{2}\left\|e_{i}-a_{i}\right\|^{2}
$$
Then the constraint parameter can be update as 
$$
\begin{array}{l}
C=C+\mu\left(E-X+U Q^{T}\right) \\
\mu=\rho \mu
\end{array}
$$
这里有一个比较大的问题，是project的data尽量正交  和   选的坐标基底尽可能正交

L是否normalized 是否会有影响



### Connecting Graph Convolutional Networks and Graph-Regularized PCA

Similarly, there are another work on the PCA based network, however, the representation constraint is not the same with the original one. The target is that:
$$
\begin{array}{l}
\min _{U, Q} J=\left\|X-Z W^{T}\right\|_{F}^{2}+\alpha \operatorname{Tr}\left(Z^{T}\tilde{L} Z\right) \\
\text { s.t. } W^{T} W=I
\end{array}
$$
Notice that, the smooth term is different, it aims to smooth the representation, but not the coodinate.

The solution steps are similar with the above one, and the solution is that:
$$
\begin{aligned}
W^{*} &=\left(\mathbf{w}_{1}, \mathbf{w}_{2}, \ldots, \mathbf{w}_{k}\right) \\
Z^{*} &=(I+\alpha \tilde{L})^{-1} X W^{*}
\end{aligned}
$$
where $\mathbf{w}_{1}, \mathbf{w}_{2}, \ldots, \mathbf{w}_{k}$ are the eigenvectors corresponding to the largest $k$ eigenvalues of the matrix $X^T(I+\alpha \tilde{L})^{-1}X$



So the question is that:

- What is the different between find the graph-based projection and graph-based data-> correlation with the graph signal process
- How to set the centerize the feature into the real practice scenario.
- What is the connection between LDA, and the weight of neural network
- What is the correlation with our form
- How to transform this kind of duration form

貌似现在的分析只能对应的是中间矩阵是半正定的情况。

### Low-Rank Matrix Approximation with manifold regularization

矩阵分解和直接优化对应的目标，进行分解有什么本质的区别？感觉似乎是没有的



The optimization forms takes that:
$$
\begin{array}{l}
\min _{U, Y} J=\left\|A-UY\right\|_{F}^{2}+\alpha \operatorname{Tr}\left(Y\tilde{L} Y^T\right) \\
\text { s.t. } U^{T} U=I
\end{array}
$$
The solution is then find the optimized $(U, Y)$ pair, since results in the same result with  $(UQ, Q^TY)$. 

The formulation can be written without a factorization form.
$$
\min_{\text{rank(X)}\le r}\left\|A-X\right\|_{F}^{2}+\alpha \operatorname{Tr}\left(X\Phi X^T\right)
$$
where $X=UY$, does not  result in the final result.

As $\Phi$ can be the Laplacian matrix, which is semi-positive in most cases, the square root result, can be written as $\Phi+I = B^TB$, the $I$ is from the first term.

Then
$$
f(X)=||A||_F^2-2tr(XBB^{-1}A^T)+||XB||_F^2)=||A||_F^2+||XB-AB^{-1}||_F^2-||AB^{-1}||_F^2)
$$



The other part of the method with the latenate iteration algorithm is similar with the above one. Algorithm suppose the solution by the SVD method, which is not the key in our analysis





## Robust PCA algorithm with graph constraint

Different from the above methods,  the robust based method is more robust to the outliner, while the original PCA use the L2 norm may effect by this form since it is based on the gaussian noise assumption. L1 norm seems to be more robust, but it is non-convecx and hard to be optimized. Below methods are trying to solve this kind of situation.





### Practical Low-Rank Matrix Approximation under Robust L1-Norm

This paper majorly add a convex optimizer trace regularization to avoid oversmooth, and argumented Lagrange multiplier is utilized a new optimizer. 

The problem still with the matrix factorization framework can be found as:
$$
\min_{U,V} ||W \odot (M-UV)||_1 s.t. U^U = I
$$
The constraint is to avoid too many pairs appear in the final result. To give a low rank smooth optimizer, a small nuclear norm is given as a regualrization term as 
$$
\min_{U,V} ||W \odot (M-UV)||_1+\lambda ||V||_* s.t. U^U = I
$$
Then the problem is sent to a ALM problem solver for the final result. induce that $E=UV$. 
$$
\begin{aligned}
f(E, U, V, L, \mu)=&\|W \odot(M-E)\|_{1}+\lambda\|V\|_{*}+\\
&\langle L, E-U V\rangle+\frac{\mu}{2}\|E-U V\|_{F}^{2},
\end{aligned}
$$
Then $U$ is solved via Orthogonal Procrustes, $V$  is solved via Singular Value Shrinkage, and $E$  is solved via Absolute Value Shirnkage. 





### Robust Principal Component Analysis on Graphs

We need to notice that at first that the below method do not have additiona graph structure for learning, it is conducted based on the feature similarity.

Graph can improve the cluster property due to the graph smoothness assumption on the low-rank matrix. 

Different from the above work, this paper abandon the explict matrix factorization but an add framework. 

Thr proposed model is as follows:
$$
\begin{array}{l}
\min _{L, S}\|L\|_{*}+\lambda\|S\|_{1}+\gamma \operatorname{tr}\left(L \Phi L^{T}\right) \text {, } \\
\text { s.t. } X=L+S,
\end{array}
$$
where $S$ is the sparse error, while $L$ is the low-rank approximation of $X$. The final term are defined the smoothness on the graph structure.

![](https://pic1.zhimg.com/80/v2-9124fe3d6f563ccb4f1672ca10a7aeb1_720w.png)

Then the problem can be rewritten as the following function to let $L$ become a condition for $L$ in ALM solver.
$$
\begin{array}{l}
\min _{L, S}\|L\|_{*}+\lambda\|S\|_{1}+\gamma \operatorname{tr}\left(W \Phi W^{T}\right) \text {, } \\
\text { s.t. } X=L+S, L=W
\end{array}
$$
Then for each index, an lagrange multiplier is give as $Z_1\in \mathbb{R}^{p\times n}$ and $Z_2\in \mathbb{R}^{p\times n}$,

Then the problem can be transformed into: 
$$
\begin{aligned}
(L, S, W)^{k+1} &=\underset{L, S, W}{\operatorname{argmin}}\|L\|_{*}+\lambda\|S\|_{1}+\gamma \operatorname{tr}\left(W \Phi W^{T}\right) \\
&+\left\langle Z_{1}^{k}, X-L-S\right\rangle+\frac{r_{1}}{2}\|X-L-S\|_{F}^{2} \\
&+\left\langle Z_{2}^{k}, W-L\right\rangle+\frac{r_{2}}{2}\|W-L\|_{F}^{2}, \\
Z_{1}^{k+1} &=Z_{1}^{k}+r_{1}\left(X-L^{k+1}-S^{k+1}\right), \\
Z_{2}^{k+1} &=Z_{2}^{k}+r_{2}\left(W^{k+1}-L^{k+1}\right),
\end{aligned}
$$


Then the problem can be solved by:
$$
\begin{array}{l}
L^{k+1}=\operatorname{prox}_{\frac{1}{\left(r_{1}+r_{2}\right)}}\|L\|_{*}\left(\frac{r_{1} H_{1}^{k}+r_{2} H_{2}^{k}}{r_{1}+r_{2}}\right), \\
S^{k+1}=\operatorname{prox}_{\frac{\lambda}{r_{1}}}\|S\|_{1}\left(X-L^{k+1}+\frac{Z_{1}^{k}}{r_{1}}\right) \\
W^{k+1}=r_{2}\left(\gamma \Phi+r_{2} I\right)^{-1}\left(L^{k+1}-\frac{Z_{2}^{k}}{r_{2}}\right)
\end{array}
$$
Assuming that a p-nearest neighbors graph is available, there are several methods to construct neighborhoods are 

- binary
- heat kernel
- correlation distance







### Fast Robust PCA on Graphs

Similar with the above paper, this paper give mode detailed on how the graph based on feature similarity can enhance the performance. 

In this methods, it introduce the graph smoothness on both samples and features smoothness, also the method can show clear cluster under some theoritical condition.



The target is as follow
$$
\begin{array}{l}
\min _{U, S}\|S\|_{1}+\gamma_{1} \operatorname{tr}\left(U \mathcal{L}_{1} U^{\top}\right)+\gamma_{2} \operatorname{tr}\left(U^{\top} \mathcal{L}_{2} U\right), \\
\text { s.t. } X=U+S,
\end{array}
$$
where $U$ is not constaint as the low dimensional representation.

The optimzation procedure is used via two graph constraints with **Fast Iterative Soft Thresholding Algorithm**

![](https://pic2.zhimg.com/80/v2-d193d7db2548d565cfeccd658cfbb7dd_720w.png)



The graph is constructed with:
$$
A_{i j}=\left\{\begin{array}{ll}
\exp \left(-\frac{\left\|\left(x_{i}-x_{j}\right)\right\|_{2}^{2}}{\sigma^{2}}\right) & \text { if } x_{j} \text { is connected to } x_{i} \\
0 & \text { otherwise. }
\end{array}\right.
$$
Two graphs are based on the sample similarity and data similarity respectively, how can they give us more information. 

The graph of feature can provide a basis for data, which is well aligned with the corvariance matrix $C$.

The graph of samples provdethe embedding which has the similar interpretation as PCA. In a word, the Laplacian matrix has some similarity with the PCA based method. 

Therefore, the low rank matrix should be able to represent by a linear combination of the feature and samples vector.  The result is bounded by the gap between eigenvalues 
$$
\begin{array}{c}
\phi\left(U^{*}-X\right)+\gamma_{1}\left\|U^{*} \bar{Q}_{k_{1}}\right\|_{F}^{2}+\gamma_{2}\left\|\bar{P}_{k_{2}}^{\top} U^{*}\right\|_{F}^{2} \\
\leq \phi(E)+\gamma\left\|X^{*}\right\|_{F}^{2}\left(\frac{\lambda_{k_{1}}}{\lambda_{k_{1}+1}}+\frac{\omega_{k_{2}}}{\omega_{k_{2}+1}}\right)
\end{array}
$$


### Deep Matrix Factorization with Spectral Geometric Regularization

The Deep Matrix is similar with the DNN which that as the original matrix facotirxzaiton always give a binary factors like $X=X_1X_2$, it gives another form as $X=\prod_{i=1}^NX_i$. 

The product graph is give as the Cartesian product of $\mathcal{G}_1$ and $\mathcal{G}_2$, where the Laplacian matrix can be represetned as 
$$
L_{\mathcal{G}_1 \Box \mathcal{G}_2 } = L_1 \otimes I + I \otimes L_2
$$
And the function is defined by the eigenvectors from both individual Laplacian matrix: $\Phi , \Psi$,

$C$, the function map is defined as $C=\Phi^TX\Psi$, which map between the functional map between the function space of $\mathcal{G}_1$ and $\mathcal{G}_2$. It can also be called the signal on the product graph. The following property can be found.
$$
\alpha = \Phi^Tx=C\Psi^Ty=C\beta
$$
for $x=\Phi^T\alpha$ and $y=\Psi^T\beta$. 

The optimization object is as follows:
$$
\min_X E_{data}(X)+\mu E_{dir}(X)  s.t. rank(X) \lt r
$$
The dirichlet energy is
$$
E_{dir}(X)=tr(X^TL_rX)+tr(X^TL_cX)
$$
then we decompose $X$ as $X=AZB^T$, $Z$ is the signal lies in the latent product graph 

For those three factor can also be factorized as:
$$
\begin{array}{l}
\boldsymbol{Z}=\boldsymbol{\Phi}^{\prime} \boldsymbol{C} \boldsymbol{\Psi}^{\prime \top} \\
\boldsymbol{A}=\Phi \boldsymbol{P} \Phi^{\prime \top} \\
B=\boldsymbol{\Psi} Q \boldsymbol{\Psi}^{\prime \top}
\end{array}
$$
The objective can be transformed into 
$$
\min_{P,C,Q} ||(\Phi PCQ^T\Psi^T-M)||_F^2+tr(QC^TP^T\Lambda_rPCQ^T)+tr(PCQ^T\Lambda_cQC^TP^T)
$$


### Matrix Decomposition on Graphs: A Functional View

Once we have two graphs, it is natural to think about the correlation between those graph, which is the function on the product graph. 

It tries to give a unify view on the geometric matrix completion and graph regularized dimension reduction. 

We give the form $X=\Phi C\Psi^T$

The matrix factorization can establish for basis consistency as the low dimension representation of $X$ can be represented as the span of $\Psi$ and $\Phi$.

Then it requires the correspondance with each eigenvalue as:
$$
E_{reg}=||C\Lambda_r-\Lambda_cC||^2
$$
where $\Lambda$ is the eigenvalue of the graph. 



## Other Matrix Factorization Methods



### Local Low-Rank Matrix Approximation



## Understanding on graph with matrix factorization

### Simplication of Graph Convolutional Networks: A Matrix Factorization-based Perpective

The motivation of this paper is to connect the matrix factorization based graph embedding method with GNN. In this way, it does not need to load the whole graph at once, but can use the sample to get the embedding of each node.

However, this paper has a very important drawback which that, there is no discussion on the feature space. The only input is the graph structure.

This paqper aim to analysze the connection between GCN and MF, simply GCN with MF only, anduise unitization and cotrain to learn a node classification model.



Analysis is done in the last layer: As the original GCN can be written as 
$$
\mathbf{H}^{(-1)}=\sigma\left(\tilde{\mathbf{D}}^{-\frac{1}{2}} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-\frac{1}{2}} \mathbf{H}^{(-1)} \mathbf{W}^{(-1)}\right)
$$
where $\mathbf{H}^{(-1)}$ is the hidden representation on the last layer. 

To write it in a node-wise method. 
$$
h_{i}^{(-1)}=\sum_{j \in I} \frac{1}{\sqrt{\left(d_{i}+1\right)\left(d_{j}+1\right)}} \mathbf{A}_{i, j} h_{j}^{(-1)}+\frac{1}{d_{i}+1} h_{i}^{(-1)}
$$
Notice that, here something trival happens that for the last layer represnetation $h_{i}^{(-1)}$ should not be the same from the left hand and the right hand.

Then it can be rewritten as:
$$
h_{i}^{(-1)}=\sum_{j \in I} \frac{1}{d_i}\sqrt{\frac{\left(d_{j}+1\right)}{\left(d_{i}+1\right)}}A_{i,j}h_j^{(-1)}
$$
Then the distance function that GCN tries to optimize becomes:
$$
l_{s}=\sum_{i \in I} \text { Cosine }\left(h_{i}^{(-1)}, \sum_{j \in I} \frac{1}{d_{i}} \sqrt{\frac{d_{i}+1}{d_{j}+1}} \mathbf{A}_{i, j} h_{j}^{(-1)}\right)
$$
Then if we choose the negative sample randomly, the optimal representation canb written as
$$
VV^T = log(|G|D^{(-1)}AD^{(-1)})-log(k)
$$
I do not think this form is very meanful since the form is the same with the LINE, what is the difference then. Another question is that, for the node embedding method, does it matter to have only one embedding or two embedding? The answer is that it does not matter a lot.





Also, I think one thing important is that we need to clarify the difference and connection with the Graph Embedding method.



