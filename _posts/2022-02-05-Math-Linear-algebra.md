---
title: Highlight of Linear Algebra
layout: category
permalink: /categories/Math-Linear-Algebra/
taxonomy: Highlight of Linear Algebra
---

$Ax=b$, $Ax=\lambda x$ $Av=\sigma u$, <span> $\min{ \frac{||Ax||^2}{||x||^2}}$  </span> 

- Multiplication Ax using columns of A   

  - What is column, row space, independent vectors, basis,  rank, CR decomposition
  - Matrix mutiplication, outer product

- Factorization

  - $A=LU$  

    - 不断消去上方单元   $Ax=b\to LUx=b \to Lc = b, c = Ux$

  - $A=QR$

  - 特征值分解

    - 对称矩阵，正定矩阵，半正定矩阵  (), the energy function
    - Rank的一些性质
    - $A+sI \to \lambda_1+s, \lambda_2+s, $

  - 奇异值分解   $Av_1 = \sigma_1 u_1 $

    - derive from $AA^T$ and $A^TA$

    - How to compute

    - The relation between left eigenvector and right,  more vectors with 0

    - If $A-xy^T$ has rank 1, <span>$\sigma_1 \ge  |\lambda  |$</span>

    - Reflect(affine), scale, reflect, the reduced form

    - The function understanding of SVD

    - Function form, polar decomposition 
      $rcos\theta+irsin\theta = re^{i\theta}$   A = QS, orthgonal, semi-positive definite, seperate rotation from strech

    - $$
      A = U\Sigma V^T=(UV^T)(V\Sigma V^T) = QS
      $$

      

- orthogonal matrix and 

  - Orthogonal basis: a and c  $c = b - \frac{a^Tb}{a^Ta}a$
  - Orthogonal projection：  $P = QQ^T$  Projection matrix $P^2=P$
  - co-efficient of orthogonal basis:   $c_1=q_1^Tv$
  - orthogonal matrix and vector will not change the matrix norm. Then the matrix norm is connected with SVD as  $A=U\Sigma V^T$, as U and V are all orthogonal vectors, matrix norm can connect with $\Sigma$ matrix

- norm

  - Eckart young  if B has rank k, <span>$||A-B|| \ge ||A - A_k||$</span>
  - Inner product
  - Matrix Norm: three norm only effected by $\sigma$, what about others.
    - l2 norm: <span>$\max \frac{||Ax||}{||x||} = \sigma_1$</span>
    - Frobenius: $\sqrt{\sigma_1^2+\sigma_2^2+\cdots + \sigma_r^2}$
    - Nuclear norm:   $\sigma_1+\sigma_2+\cdots + \sigma_r$      The minimum value of <span>$||U||_F||V||_F$  $||A^TA||_N=||A||_F^2$</span>
  - Vector Norm  $l_0, l_1, l_2, l_\infty$
  - The inituition or the matrix norm: The minimum of  <span>$||V||_p$</span>
  - Norm的性质：  Rescale， triangle
  - Function norm-> vector space to be completed  <span>$||v_n - v_\infty||<span> \to 0 $  
    - ending with all zero is not completed  p105 
    - $norm < \infty$
  - spectral radius: about the stationary of Markov chain

- Application

  - PCA  

    - The stastics behind 
      - variances are the diagonal entries of the matrix $AA^T$
      - covariances are the off-diagonal entries of matrix $AA^T$
    - The geometric behind
      The sum of squared distances from the data points to the $u_1$ line is a minimum.
    - linear algreba behind
      Total variance $T$ is the sum of sigma
    - The quick drop of $\sigma$ in hilebert matrix

  - Rayleigh Quotients, generalized eigenvalue   $Sx_1 = \lambda M x_1$    99页
    $$
    R(x) = \frac{x^TSx}{x^Tx}
    $$

    - Generalized Rayleigh Quotients  $R(x) = \frac{x^TSx}{X^TMx}$
      - M: covariance matrix, is positive definite, maximum of $R(x)$ is largest eigenvalue of $M^{-1}S$, $M^{-\frac{1}{2}}SM^{-\frac{1}{2}}$
    - Generalized Eigenvectors and M-orthogonal   $x_1^TMx_2 = 0$ , $x = M^{-\frac{1}{2}}y$
    - Semi-definite situation  $\alpha Sx = \beta Mx$ , $\alpha$ may equal to 0. number of samples smaller than features
    - Generalized SVD  $A=U_A\Sigma_AZ$   $B=U_B\Sigma_BZ$
    - Any two positive definite matrix can be decomposed by the same inverse matrix

  - LDA

    Seperated rate
    $$
    R = \frac{(x^Tm_1-x^Tm_2)^2}{x^T\Sigma_1x+x^T\Sigma_2x}
    $$
    $S = (m_1-m_2)(m_1-m_2)^T$

    **求解背后的物理意义没太看懂**

- 





$\lambda_1 \le \sigma_1$

AB and BA  

$ABx=\lambda x$, $BABx=\lambda Bx$ ,  $Bx$是BA的eigenvector   **eigenvector也有对应的关系**







# Low rank and Compressed Sensing

- Key insights
  - matrix are composed of small rank matrix: ($uv^T$) is extreme case with rank 1
  - Singular value: low effective rank.
  - Most matrices are completed by low rank matrix.



- How matrix change when add small rank matrix

  - Normal Perspective(use small matrix and exchange for the larger matrix)

    - $A^{-1}$ 
      $$
      (A-UV^T)^{-1} = A^{-1}+A^{-1}U(I-V^TA^{-1}U)^{-1}V^TA^{-1}
      $$

      $$
      (I-UV^T)^{-1} = I +U(I-V^TU)^{-1}V^T
      $$

    - eigenvalue and signal values (interlacing)    
      A graphical explanation: the solution is an inverse function which the last one can never go beyond the first one.
      $$
      z_1 \ge \lambda_1 \ge \cdots
      $$

- Differentiate perspective

  - $A^{-1}$ 
    $$
    \frac{dA^{-1}}{dt} = - A^{-1}\frac{dA}{dt}A^{-1}
    $$



    \frac{d\lambda}{dt}=y^T\frac{dA}{dt}x
  $$
    
  - $$
    \lambda_{max}(S+T)\le \lambda_{max}(S) + \lambda_{max}(T) 
  $$

  $$
    \lambda_{min}(S+T) \ge \lambda_{min}(S) + \lambda_{min}(T)
  $$

    However, it is hard to find the intermediate ones.

  - Weyl inequality   
    $$
    \lambda_{i+j-1}(A+B)\le \sigma_i(A)+\sigma_j(A)
    $$

- Saddle points from lagrange multipliers

  - Lagrangian:  $L(x, \lambda) = \frac{1}{2}x^TSx + \lambda^T(Ax-b)$ which will produce a Lag





- application
  - update least square, has a new row
  - Kalman filter **TODO reading P112**
  - 