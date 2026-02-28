---
title: Source Free Unsupervised Graph Domain Adaptation
layout: category
permalink: /categories/sourcefree/
taxonomy: Source Free Unsupervised Graph Domain Adaptation
---
# Source Free Unsupervised Graph Domain Adaptation

### Content

In this blog, we will not only talk about our paper, but also give a brief introduction on the domain adaptation for those who are not familiar with it.  If you are familar with DA, it's ok to start from the section 4

1. What is Domain Adaptation and its practice use?
2. The traditional methods in domain adaptation.
3. Brief introduction on domain adaptation methods in CV.
4. Why we need source free unsupervised graph domain adaptation? The key challenge and our solution.
5. Experiments
6. Conclusion and future work



### What is Domain Adaptation and its practice use?

To give you a more precise description, here we find the definition in Wikipedia.

> **Domain adaptation(DA)** is a field associated with machine learning and transfer learning. This scenario arises when we aim at learning from a source data distribution a well performing model on **a different (but related)** target data distribution. For instance, one of the tasks of the common span filter problem consists in adapting a model from one user (the source distribution) to a new user who receives significantly different emails (the target distribution). Domain adaptation has also been shown to be beneficial for learning unrelated sources.

So study the domain adaptation is just like the domain

So the most important characteristics in DA are as follows:

- Source domain are well-labeled and target domain are unlabeled
- **Difference**: There exists the domain gap between source and target domain.  
  - covariate shift(marginal distribution shift): the most common domain gap where `$ P(X_s) \ne P(X_t) $`and `$ P(Y|X_s) = P(Y|X_t) $`  which is the prior knowledge **we assume** in the proof of this paper.
  - target shift (conditional distribution shift):  `$ P(X_s) = P(X_t) $`  and `$ P(Y|X_s) \ne P(Y|X_t)$`
  - Joint distribution shift:  `$P(X_s) \ne P(X_t)$` and `$P(Y|X_s) \ne P(Y|X_t)$` which is the most difficult one with no assumption.
- **Related**: The task is related, which means labels in the source domain and the labels in the target domain **are the same**. Here we need to clarify its difference between the recent popular pretrain (specifically talking about graph). The key is related
  - Methods in graph pretrain first use unlabeled graph to learn a good representation for different tasks, then they use the labeled data for the specific downstream task to finetune the parameters. The key of pretrain is **enhances a model with labeled data by leveraging additional knowledge from unlabeled data**.
  - Methods in the proposed DA first use labeled source graph to learn a discriminative model and adapt it to the **unlabeled** target graph. With no label in the target domain, it is a harder problem than pretraining. The key of DA  is an unlabeled classification problem by leveraging the relation information between node features and labels learned from a labeled source data.  
  - The challenges and required techniques are entirely different for these two scenarios.

To concrete the difference, we will introduce some real-world dataset for you to fully understand what is the domain gap. One thing needs to be mentioned that,  we have not found any mathmatical metric to evaluate the domain gap, most of the domain gaps just come from the perceptual knowledge.

**Taking an example in CV (VisDA-C dataset)**

Our target is to recognize the plane in the real-world. However,it is hard to manually collected so many images (with different angels) and tried to label them.

![](https://pic3.zhimg.com/80/v2-f868b79f6189d56b6411ef7d635ee489_1440w.png)

But it is quite easy to form the synthestic plane with the technology of graphics. 

![](https://pic2.zhimg.com/80/v2-f47ddd83baad63d278775035b9b484ea_1440w.png)

Here comes the domain gap! The real image and the synthestic image are quite different but they are all planes. Oh, we can say that they are two different but related tasks.  We will not mentioned more example in CV, like the change of the angle or different backgrounds. This is a research topic of great practice value.

**Then we take some examples in graph to see this pheonmon in the graph domain to stress the scenario of practice use.**

The first scenario happens when a new plaform is built or the new incoming data.

We all know that ACM and DBLP are well-labeled database in citation network.  If we would like to develop some new databases that might not be labeled yet, such as Aminer, based on the resources in the existing databases. The domain gap exists because different platforms have their own interest. Methods in UGDA scenario could help the transferring in an unsupervised manner. 

The second scenario happens in the ACM platform, which is a large platform of papers. 

However, we found that the domain gap even exists in different subgraphs with different structures even though the features come from the same distribution. A dense graph with 1500 nodes and 4960 edges, and a sparse graph also with 1500 nodes but with 759 edges. We can found the model trained on one graph can hardly achieve a good performance on another one. Similar observation can also be found in [Investigating and Mitigating Degree-Related Biases in Graph Convolutional Networks](https://arxiv.org/pdf/2006.15643.pdf) Methods in UGDA can help to enhance the performance on label-scare subgraph.



### The traditional methods in domain adaptation.

Revolving on the above goal to mitagate the domain gap, we first give some traditional methods and definitions as they still have some treasures we can learn from.  And there are two major methods in our discussion:

- Instance-based transfer learning approaches
- Feature-based transfer learning approaches 

#### Instance-based transfer learning approaches

The intuition of the instance-based methods is that we can identify the importance of an source example is in the feature distribution of the target domain. 

**problem Setting:**

Given  `$D_S= \left \{ x_{S_i}, y_{S_i} \right \}^{n_S}_{i=1}$, $D_T=\left \{ x_{T_i} \right \}^{n_T}_{i=1}$`

The goal is to learn $f_T$, $s.t. \sum_i \epsilon (f_T(x_{T_i}),y_{T_i}) $ is small

where $y_{T_i}$  is unknown.

**The assumptions are:**


`$ \mathcal{Y}_{S}=\mathcal{Y}_{T},$`  and `$P(Y_S|X_S)=P(Y_T|X_T) $`

`$\mathcal{X}_{S}=\mathcal{X}_{T},$`

$P(X_S)\ne P(X_T)$

So the solution will be:
$$
\theta^* = arg \min{\mathbb{E_{(x,y)\sim P_T}}[l(x,y,\theta)]}
$$

$$
= arg \min{\mathbb{E}_{(x,y)\sim P_T} \left [ \frac{P_S(x,y)}{P_T(x,y)}  l(x,y,\theta) \right ]}
$$

$$
= arg \min{\int_{y}\int_{x}P_T(x,y) \left ( \frac{P_S(x,y)}{P_S(x,y)} l(x,y,\theta) \right )}dxdy
$$

$$
=arg \min{\int_{y}\int_{x}P_S(x,y) \left ( \frac{P_T(x,y)}{P_S(x,y)} l(x,y,\theta) \right )}dxdy
$$

$$
= arg \min{\mathbb{E}_{(x,y)\sim P_S} \left [ \frac{P_T(x,y)}{P_S(x,y)}  l(x,y,\theta) \right ]}
$$

Denote that $\beta(x)=\frac{P_T(x)}{P_S(x)}$ which can be viewed as the weight on each instance which describe the probability of the souce example appearing in the target domain to reduce the domain shift. 


#### Feature-based Transfer Learning Approaches

The intuition from the feature based domain is that the source and target domains have some overlapping features. (features only have support in either the source or the target domain). The feature-based methods aim to learn a mapping function $ \varphi$ to enhance the overlapping features. In other word, to stress the importance of the overlapping features and ignoring the distuiguished features. An example is the following image.

 ![](https://pica.zhimg.com/80/v2-842e558d15318e7a305d8805288534c8_720w.png)



Feature-based methods are the most common used in DA. So we will discuss some typicial methods:

- Feature Augmentation (FAM model)
- Transfer Component Analysis (TCA model)

##### FAM model

FAM is just a feature argumentation. The mathematic formulation of the model is:
$$
 \Phi_S(x_i^S)=\left \langle x_i^S, x_i^S, 0 \right \rangle ,  \Phi_T(x_i^T)=\left \langle x_i^T, 0, x_i^T \right \rangle 
$$
The key steps are:

- replicate the features for 3 times: General feature, Source features, Target features
- For the transformation of the source domain: target features set to 0
- For the transformation of the target domain:  source features set to 0

They hope the weight corresponding to the general feature in the first part can learn the overlapping features while others learn something distuiguishing.

However, I cannot totally agree with these methods for most of the features are low-rank, and there exists too many solutions.

##### Transfer Component Analysis

Its intuition is similar with the PCA. However, PCA aims to learn a low dimension representation to best preserve the information in the feature. And the PCA aims to learn a low dimension representation to best preserve the similar features on the two domains.  It seems a little hard for me to explain it well intuitively, if there is any problem, contact with me or read the whole paper. [link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5640675)

**Maximum Mean Discrepancy (MMD)**

As we know that $X^S= \{ x_i^s\}$ and $X^T= \{ x_i^t\}$ are from different feature distribution. Then how to use the sample to estimate the distance between the distribution. Unlike the KL divergence or other methods which nees some hyperparameter or the density estimation. MMD is based on the reproducing Hilbert space (RKHS) with no need on any parameter. 
$$
Dist(X^S,X^T)=MMD(X^S,X^T)=\left \| \frac{1}{n_s}\sum_{i=1}^{n_s}\phi (x^s_i) - \frac{1}{n_t}\sum_{i=1}^{n_t}\phi (x^t_i)   \right \| ^ 2 _ {\mathcal{H} }
$$
$\phi$ is the kernel function (similar with the kernel in SVM), the sample are projects into RKHS space and calculate the distance between their average values.  **Notice that, the kernel of TCA is chosen manually.**  The $\phi$ should:

- The distance between the marginal distributions $ P(\phi(X_S))$ and $P(\phi(X_T))$ is small. 
- $\phi(X_S)$ and $\phi(X_T)$ preserve important properties of $X_S$ and $X_T$

minimize the distance between the two distribution. 

**Hilbertt-Schmidt Independence Criterion** 

With the distance metric, I think we can find the dependence with the metric. It computes the Hilbert-Schmidt norm of a cross-covariance perator in the RKHS. The more depedent feature is, the more neccerary we should preserve it.
$$
HISC(X,Y) = \left (  \frac{1}{(n-1)^2 tr(HKHK_{tt})} \right )
$$
where $K$, $K_{tt}$ are the kernel matrix in the source and target domain. $H=I-(\frac{1}{n}11^T)$ is a centering matrix. If it is equal to 0, it means they are independent. The value is larger means the feature are more dependent.

Based on the good information preserving of the HISC property, it is often used as a metric in dimensionality reduction. It is often desirable to preserve the local data geometry while aligning with the side information. **In our case, it is the unlabeled data in target domain.** 

Mathematically, the problem will be:
$$
max_{\mathcal{K \succeq 0}} tr(HKHK_{tt}) s.t. K
$$
$k_{tt}$ is computed by the information from the target domain.

**MMDE**

We revisit a traditional method to generate embedding.  

Similar with the PCA, first we need to compute the similarity between each two instances. However, we will give different weight on nodes from different domains. The Gram matrices is defined as: 
$$
K = \begin{pmatrix}
  K_{S,S} & K_{S,T}\\
  K_{T,S} & K_{T,T}
\end{pmatrix} \in \mathbb{R}^{(n_1+n_2)\times(n_1+n_2)}
$$
where $K=\left [ \phi(x_i)^T \phi(x_j)^T \right ]$, $\phi$ is defined on all the data and the MMD distance can be writtern as: $tr(KL)$.

$L_{ij}=\frac{1}{n_s^2}$ if $x_i, x_j \in X_S$.   $L_{ij}=\frac{1}{n_t^2}$ if $x_i, x_j \in X_T$ or $L_{ij}=\frac{1}{n_sn_t}$ 

And the solution can be written with the constraint of $K$.  
$$
\min_{K \succeq 0} tr(KL)-\lambda tr(K)
$$
where thre first term is the object to minimize the distance between distribution, while the second maximizes the variance in the feature space.

**TCA**

To avoid solving this large and dense matrix and investigate more correlation in an inductive way, TCA is proposed. 

1. decompose the kernel matrix $K$ into $K=(KK^{-\frac{1}{2}})(K^{-\frac{1}{2}}K)$, a dual form
2. A matrix factorize is proposed then. $\tilde{K}=(KK^{-\frac{1}{2}}\tilde{W})(\tilde{W}^TK^{-\frac{1}{2}}K)=KWW^TK$, where $W=K^{-\frac{1}{2}\tilde{W}}$. The operation is simplified by the kernel trick. 
3. As the optimization object is the trace of the matrix, the result can be rewrite: $tr((KWW^TK)L)=tr(KWLW^TK)$

Then the optimization object is:
$$
\min_W tr(W^TKLKW) + \mu tr(W^TW)   s.t.W^TKHKW=I_m
$$
The first term is to minimize the discrepancy in the distributions,  and the second term is to maxmize the variance to preserve the diversity in data. The constraint is the variance of the projected samples. 

Further simplify this problem, the solution is just the top $m$ singular value of the matrix $(KLK+\mu I)^{-1}KHK$



##### A simple conclusion

From the empical perspective, I think the reason for $P(X_S) \ne P(X_T)$ come from two perspectives. 

- The difference of sample probabilities.  one class may appear many times in the source domain and seldom appear in the target domain.  In this case: instance-based method may help.
- The difference of feature distribution. like the above example of plane and plane model. In this case: feature-based method may help.

I think it's really worthy to learn these traditional methods. I am still thinking how it can help to our understanding today.



### Brief introduction on domain adaptation methods in CV.

In the above section, we introduce how to mitigate the domain gap mainly from the dimension reduction perspective. However, with the era of the deep learning incoming, rigorous mathmetic proof and careful design seems to be useless as an loss item can help us to do most of the things we want. (We do not introduce the graph domain adaptation methods in details here, for they share the spirit)

After reading a large number of papers, we conclude these methods from three perspectives:

- Feature-level transfer methods  (Feature-based transfer)
- Label-level transfer methods (instance-based transfer)
- Model-level transfer methods

To clarify again for our problem:

- Inputs are the souce features, the source labels and the target feature
- Ouput should be a model performs well on the target domain



**Feature-level transfer methods**

The feature level methods usually have joint training framework with a shared encoder and a classifier (maybe have some additional model component).

- A feature encoder is trained to align the feature distributions between the source domain and the target domain for mitigating the domain gap.  After encoding, the features (domain invariant representation) is so similar that we can not be distinguished that which domains it comes from.
- As the domain gap vanish, a classifier is trained on the encoded features with the cross entropy loss, supervised by source labels. The classifier can also generate well on the target domain.

 

With different training procedures, we can further divide these methods into two categories:

- Distance based methods
- Domain adversarial methods

**Distance based methods** is most close to the traditional feature-based transfer methods. They also incorporate maximum mean discrepancy (MMD) loss on the hidden representation of a particular layer or some layers. It acts as a domain distance loss to match the distribution sttisitcal momonts at different orders. 

Here we show an example of the classical method: DDC

![](https://pic2.zhimg.com/80/v2-776bf5c97595108a2cbde770408db7cc_720w.png)

Dotted line means the weights are shared. We can see that it has a shared seven-layer encoder with a one-layer classifier.

**Domain adversarial method**

Domain adversarial methods have an additional component called domain classifer. Also, they generate the domain label. The additional task of the domain classifer is to recognized which domain the sample comes from.

And the feature encoder will fight against it! It will try to generate feature looks like from the similar distribution to confuse the domain classifier. As the training moving on, we result in a good classifier and also a good encoder generated aligned feature.

Here we show an example of the classical method: DANN.

![](https://pica.zhimg.com/80/v2-950f65cf72dfc1a28bf125f316222591_720w.png)

Notice that the gradient will reverse comes out from the domain classifier, which means to oppose the sign of the gradient. It is easy to understand why design like this. oppose the sign of the gradient is the same with oppose the sign of the loss, where the domain classifier aims to learn to be more discriminative. To the opposite, the encoder wants to generate more similar feature, so it is natural to have such methods.

 



**Label transfer methods**

The label transfer method is some kinds of connected with the instance-based methods since they generate the pseudo label for each instance. These methods is to take advantage of the well-trained source model to generate pseudo-label for target samples based on the maximum posterior probability. The pseudo labels are then used to supervise the training of target model.

The key of these methods is to design some rules to generate the convincing label (most of them are true label).  The new pseudo label can be used as the true label to further train on the target domain. 

This method have some potential limitations like: 

- The priori knowledge of these methods are the model trained on the source data already has primary discriminative ability. In most case, the source model can have an accuracy of 60% without adaptation. It is somehow good but not well enough.
- If the rule is not designed well and generate many bad pseudo label (not correct), it may indeed lead to the failure of the training. The bad pseudo label will give bad guidance to generate even worse pseudo label.

Then we take asymmetric tri-training methods as an example

![](https://pic1.zhimg.com/80/v2-89eccfc0754298637e0df25f0b4b67c2_720w.png)

The rule of this model is voting.  F1 and F2 classifer are to generate the pseudo label and the $F_t$ is to learn to classify well on the target domain. 

The voting procedure is as follows:

- F1 and F2 are firstly trained on the source domain, A regularization term is added to ensure that the weight of F1 and F2 are orthogonal. This is to make sure that they learn some complementary knowledge.
- Then they will predict on the target domain. The rule to generate the psuedo label is 
  - Both F1 and F2 agree with the same prediction.
  - one of them is really confident with the result, which gives the 95% confidence
- Use the pseudo label to further train the network
- The model will be more confident on the target domain and do the aboves thing again.



**Model-level transfer methods**

Model-level transfer methods are to finetune the source model to conduct a target network. It is also used as an auxiliary technique in many models. The weight indeed matters a lot and the potential of model weights should been further studied. 





### Why we need source free unsupervised graph domain adaptation? The key challenge and our solution.

After so many efforts to introduce the domain adaptation, now it is the time to introduce the new scenario proposed in our paper.  I think without deep understanding on the related work. It is hard to have a good idea. 

#### Preliminary

We first give some mathematics definitions on graph and our task for a clear and elegant expression.

A graph is defined as: `$G=(V,E,X,Y)$`, `$V=(v_1,\cdots, v_n)$` is the node set with $n$ nodes and $E$ is the edge set. $X$ is the node feature where $Y$ is the node label.  

The model we used can be expressed as a conditional probablity `$ \mathcal{Q} (Y|G ; \theta)$`.

To better express the model for each node and considering about the design of the GNN, we decompose the model as: `$\mathcal{Q}(Y|G;\theta)={ \prod_{v_i \in V}q(y_i|x_i,\mathcal{N}_i;\theta)}$` where `$\mathcal{N}_i$` is the neighboorhood of the node $i$, $q$ is the conditional probability for each node.

Then the **unsupervised graph domain adaptation UGDA** can be expressed in a mathematic form.

So the inputs are a labeled source graph $G_s=(V_S,E_S,X_S,Y_s)$ and an unlabeled target graph $G_T=(V_T,E_T,X_T)$.

The output is `$\mathcal{Q}(Y|G;\theta_t)$` a model with good performance on the target domain.



#### Problems and challenge

The **key problem** of the existing UGDA methods is that they **heavily rely** on the source graph data $G_s=(V_S,E_S,X_S,Y_s)$. During the adaptation, UGDA needs:

- $X_S$ and $E_S$ to mitigate the domain gap with $X_T$ and $E_T$ and generate aligned features by a GNN encoder.
- $Y_S$ as the supervision signal to learn a discriminative classifier on the aligned features.

So **the key challenges** is that if we **cannot** access the source graph when doing adaptation, all the UGDA methods are not able to work any more.  To describe the key challenges correponding to the above bullet are:

- How can the model adapt well to the shifted target data distribution without accessing the source graph $X_S$ and $E_S$ for aligning the feature distributions. 
- How to enhance the discriminative ability of the source model without accessing source labels $Y_S$for supervision.

**The practical value:** This scenario is frequently appear in the domain adaptation because of **the privacy problems** which is becoming more and more important in recent years. For example, if the source and target graph are from two different platforms, and there are some sensitive attributes in the source graph. This may lead to the sereve data leakage problems.

To deal with the above issues, we propose a new scenario called **Source Free** Unsupervised Graph Domain Adaptation (SFUGDA) with no need to access the source graph in the adaptation procedure.

The entire training framework will be the following two stages:

- The **unaccessible** source training procedure
  - input: labeled source graph $G_S=(V_S,E_S,X_S,Y_s)$
  - output: a well-trained source model: <span>$\mathcal{Q}(Y|G;\theta_S)$</span>
  - **Notice that:** this procedure is totally **unaccessible** in the SFUGDA scenario. We can not determine neither the model architecture nor what optimizer to use. So it is hard to define what is a well-trained model. In the experiment of this paper, we think the well-trained model is the model with the best validation performance on the source domain.
- The adaptation procedure.
  - input: unlabeled target graph `$G_T=(V_T,E_T,X_T)$`  well-trained source model `$\mathcal{Q}(Y|G;\theta_S)$`
  - output: well-trained target model `$\mathcal{Q}(Y|G;\theta_T)$`
  - **Notice that:** as we cannot decide the model architecture, so the algorithm in the adaptation procedure cannot have any specific design to particular model component like BatchNorm or WeightNorm. In other word, the algorithm should be **model agnostic**



#### Solution

To deal the above challenges, we design a model agnostic algorithm for SFUGDA method called **SO**urce free domain **G**raph **A**daptation algorithm (**SOGA**). **にじげん!**

SOGA has two components:

- Structure Consistency (**SC**) optimization object: adapt on the shifted target data distribution by leavaging the structure information $E_T$ rather than align with $X_S$ and $E_S$.
- Information Maximization (**IM**) optimization object: further enhance the discriminative ability of the source model without access to $Y_S$. We theoretically prove that IM loss can enhance the lower bound of the target performance of the source model.



Then the whole training architecture is:

![](https://pic1.zhimg.com/80/v2-e0f30e058368e4c6b02ada5aaf5c30a2_1440w.png)

We will introduce those two optimization objects in detail next.

##### Information Maximzation (IM) Optimization Object

The target of designing this object is based on the following :

- The well-trained source model has primary discriminative ability

- The adaptation is an unsupervised learning without any directly supervised loss

As we do not have supervised signal, so the target should be:

- the unsupervised loss should first keep the origin performance of the source model，do not make the result worth
- then if possible, try to enhance the performance.

As a result, we design the IM optimization object to improve the lower bound of the performance.



The IM optimization object can be written as: 
$$
\mathcal{L}_{IM}=MI(V_t,\hat{Y}_t)=-H(\hat{Y}_t|V_t) + H(\hat{Y}_t)
$$
where $\hat{Y}_t$ is the prediction on the target domain, $\mathbf{V_t}$ is information of input nodes containing node feature $\mathbf{X_t}$ and information from node neighbor $\mathbf{\mathcal{N}_t}$. $MI(\cdot, \cdot)$ is the mutual information, and $H(\cdot)$ and $H(\cdot | \cdot)$ are entropy and conditional entropy, respectively.

So the objective can be divided into two parts, one is to minimize the conditional entropy and the other is to maximize the entropy of the marginal distribution of $\mathbf{\hat{Y}_t}$.

**The conditional entropy** is to enhance the confidence of the prediction. (one hot prediction is optimal). This term can theoretically enhance the discriminative ability. (**See the lemmas** in paper if you like). The final implement form of the conditional entropy is:
$$
H(\mathbf{\hat{Y}_t} | \mathbf{X_t}) =  \mathbb{E}_{x_i \sim p_t(x)} \left[- \sum_{y = 1}^k q(y|x_i, \mathbf{\mathcal{N}_i}; \Theta) \log q(y | x_i, \mathbf{\mathcal{N}_i}; \Theta)\right]
$$
**Entropy of Marginal Distribution** is to avoid to concentrating to predictions on on one category.
$$
H(\mathbf{\hat{Y}_t}) = - \sum_y q(y) \log q(y), \\
    \it{where}\; q(y) = \; \mathbb{E}_{v_i \sim p_t(v)} \left[q(y|x_i, \mathbf{\mathcal{N}_i}; \Theta) \right].
$$


##### Strcuture Consistency (SC) Optimization Object

This is also the key point of the domain adaptation with graph properties. Here we want to compare our methods with the Source free Unsupervised Domain Adaptation method in CV from the problem perspective to the experiment perspective.

**Different between SFUGDA and source free methods in CV**

**From the problem perspective**, the main reason is that, the instances in the image dataset are i.i.d.. However, the samples in the graph dataset are naturally structured by dependencies. This dependence (structure) contains much information, for example the homophily property. 

This gives additional challenge that even two graph have the exact same distributions, it will still suffer from the domain gaps of different graphs.

And it is also an advantage of the graph data. It seems natural for the graph data to do some unsupervised learning like the graph embedding methods, they just utilize the graph structure can learn an informative embedding.  Though we cannot directly align the feature distribution and find the good absolute position, the structure somehow reveal the relative position between node pairs, which may help to adaptation.

**From the experiment perspective**, we also found there are some source free domain adaptation methods in CV domain. We reimplement them (using GNN as the backbone instead of CNN). However, none of them achieve results. I think the main problem is that:

1. Most of them belong to the label transfer method which uses the pseudo label to guide the training.
2. However, when they generate the label, they do not concern much about the graph structure, for example, the homophily.
3. We test the label smoothness (how homophily the label is). The smoothness of the pseudo label is much smaller than the ground truth.
4. The wrong pseudo label lead to training into the wrong direction and become worth and worth at last.

As a result, we think it is quite neccerary for us to introduce the graph consistency constraint.



**SC design**

SC is based on two hypothesesn on graph structure

1. The probability of sharing the same label for local neighbors is relatively high.
2. the probability of sharing the same label for the nodes with the same structural role is relatively high.

So the intuition of SC is that if two nodes are local neighbor or with same structural role, their predicted vector $\hat{y}$ should be similar. SC can help to the structure consistency by enlarging similarity between nodes with connection, and distinguish nodes without connection. 

Then the problem is that how to define local neighbor and the node with smae structural role?

- For local neighbor nodes: it is just the node with direct edge connection with the node.  The node pair is defined as $(v_i, v_j)\in E_t$
- For nodes with same structural role: we following the similar construction with Struct2Vec. Two nodes are similar if they have similar degree. More similar if their neighbor nodes also have the same degree. For more details, please refer the origin paper. [link](https://arxiv.org/pdf/1704.03165.pdf)  The node pair is defined as $(v_i, v_j)\in S_t$. $S_t$ is the graph structure conducted by the structure role information.



The mathmatical formulation of the SC loss is :
`$$
\mathcal{L}_{SC} =  \lambda_1  \sum_{(v_i, v_j) \in \mathbf{E_t}} \log J_{ij}  -  \epsilon \cdot \mathbb{E}_{v_n \sim p_{n}} [\log J_{in} ]
        + \lambda_2 \sum_{(v_i, v_j) \in \mathbf{\mathcal{S}_t}} \log J_{ij} - \epsilon \cdot \mathbb{E}_{v_n \sim p_{n}} \left[\log J_{in}\right]
$$`
where `$J_{ij} = \sigma(\left\langle\!\mathbf{\hat{y}_t^{(i)}}, \mathbf{\hat{y}_t^{(j)}}\!\right\rangle)$`, `$p_{n}$ and $p_{n}'$` are the distributions for negative samples, and $\epsilon$ is the number of negative samples. We use uniform distributions for `$p_{n}$` and `$p_{n}'$` while they can be adjusted if needed. $\epsilon$ is set as 5 in our experiments. In all our experiments except for the hyperparameter sensitive analysis, `$\lambda_1$` and `$\lambda_2$` are set to the default value 1.0. 



### Experiments

Firstly, I want to clarify our fairness in conducting experiments. I believe that our experiments are really fair comparison. 

- We ensure all the reported test result is after the select of the validation set.
- Though there are some hyperparameter in our method SOGA, we set all of them to the default value without any hyperparameter tuning.
- For baseline method with tuning hyperparameter, we do carefully grid search in a large range. And we have shown the range of each hyperparameter and the best hyperparameter on each dataset in the appendix for the reproducibility of our experiment.
- Each result is run by five random seeds for fair comparison.



We aims to answer four research questions in this section. I will try to quicky go through it. For the detailed results, check our paper. Then I am going to some remain problems needs to be understanding in the experiments.

###### RQ1: How does the GCN-SOGA compare with other state-of-the-art node classification methods? (GCN-SOGA indicates SOGA applying on the default source domain model: GCN)?    

Even better result than other UGDA methods with access to the source data. 

###### RQ2: Can SOGA still achieve satisfactory results when being applied to different source domain GNN models other than GCN? 

Though some model GAT and GraphSAGE show very bad performance without adaptation, all of them show comparable result after applying SOGA for adaptation which indicate the model agnostic property.

###### RQ3: How do different components in SOGA contribute to its effectiveness?

- The weight of the source model is most important part, with only unsupervised loss. Model can learn nothing at all.
- IM loss can enhance the model performance. However, this improvement is not stable in the adapation procedure.
- SC loss can preseve the consistency and make the adaptation performance more stable

###### RQ4: How do different choices of hyperparameters λ1 and λ2 affect the performance of SOGA?

The performance is robust to these hyperparameter with little change.



#### Remain Question in experiment.

Here we will answer some open questions have not been well understood yet. We do not have a precise answer here.

###### Why SOGA can achieve better performance then other UGDA method with access to the source data?

I think it's mainly because the UGDA is far from their upper bound.

The two losses in UGDA (cross entropy and domain align loss) is somehow conflict.

Admittedly, the domain align loss like domain adversarial loss can make the domain look similar. However, as mentioned in the traditional method, what we really desire is.

1. The distance between the marginal distributions $ P(\phi(X_S))$ and $P(\phi(X_T))$ is small. 
2. $\phi(X_S)$ and $\phi(X_T)$ preserve important properties of $X_S$ and $X_T$

As the domain align loss may probably also abandon the important properties of $X_S$ and $X_T$, it may reduce the discriminative ability. As a result, the cross entropy loss will be larger. It may result in a fluctuating optimization procedure, and struck in a bad local minima. As in the below figure, the Macro-F1 score of the UDAGCN really fluctuating which may indicate our guess.

![](https://pic1.zhimg.com/80/v2-47c02f280c635b6f81f8aa07dca3cd88_1440w.png)

However, the source free methods do not have such limitation. And this may be the reason for it.



###### How can the model adapt on the target domain without explicit adaptation components?

The first thing is that we cannot build explicit adaptation component since we have no access to the source data. It is impossible for us to do the alignment.

The second thing is that even in the source free scenario, they will design some specific model component for example, batch norm, weight norm. They try to use the model to memory some information from the source domain for alignment and adaption.

however，the specific design is not practical in the real-world. Adding these component means that we need to retrain the new model on the source domain while we cannot use the existing well-trained source model directly.

So our algorithm may be not so fancy, but it indeed considers many real-world constraint which limits the fancy of our model. But it becomes more practical use.

**Back to the question**: The connection between the source and the target is built by setting the source model as the initialization of the target model. Since we cannot access the source data, we can only leverage the information stored in the source model. Also Lemma 2 shows the important of the primary discriminative ability of the source model. However, as I mentioned in section3, the model weight is important. But what the real effectiveness of it is still under discussion like the pretrain weight.



###### Why GraphSAGE and GAT do not performance well in some case when directly apply on the target domain while GCN can always have a good initial performance?

This is really a confusing question that we have tuned the hyperparameters but after all, no good result is achieved.  I am still wondering why they have this pheonomon. If you have any idea, discuss with me freely.



### Conclusion and future work

In this work, we articulate a new scenario called Source Free Unsupervised Graph Domain Adaptation (SFUGDA) with no access to the source graph because of practical reasons like privacy policies. Most existing methods cannot work well as it is impossible for feature alignment anymore. Facing the challenges in SFUGDA, we propose our algorithm SOGA, which could be applied to arbitrary GNN model by adapting to the shifted target domain distribution and enhancing the discriminative ability of the source model. Extensive experiments indicate the effectiveness of SOGA from multiple perspectives.

Talking about the future work, I think there are quite a lot things to do as we are the first to open this scenario！

- How to further take advantage of the unsupervised graph information is a great topic. I do believe that we do not find the optimal to use the structure information.
- Some deep understanding are lose as the above section mentioned. We only ensure the correctness of the result but do not have know exactly why these things happends



This is a quite new scenario and I am quite looking forward to your follow up next! If you have any problem, please feel free to contact with me. I will be more than glad to help you.