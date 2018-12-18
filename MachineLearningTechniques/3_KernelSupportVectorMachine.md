# Kernel Support Vector Machine

---

### 1.Kernel Trick
推导的dual SVM是如下形式：<br>
![dualSVM](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/3_dualSVM.png?raw=true)<br>
其中$\alpha$是拉格朗日因子，共N个，这是我们要求解的，而条件共有N+1个。<br>
向量$Q_D$中的$q_{n,m}=y_ny_mz_n^Tz_m$，$z_n^Tz_m$的内积中会引入$\hat d$。即，如果$\hat d$很大，计算$z_n^Tz_m$的复杂度也会很高，会影响QP问题的计算效率。<br>
so,$q_{n,m}=y_ny_mz_n^Tz_m$这一步是计算的瓶颈所在。

其实问题的关键在于$z_n^Tz_m$内积求解上。我们知道，z是由x经过特征转换而来：

$$z_n^Tz_m=\Phi(x_n)\Phi(x_m)$$

如果从x空间来看的话，$z_n^Tz_m$分为两个步骤：<br>
1. 进行特征转换$\Phi(x_n)$和$\Phi(x_m)$；<br>
2. 计算$\Phi(x_n)$与$\Phi(x_m)$的内积。<br>
这种先转换再计算内积的方式，必然会引入$\hat d$参数，从而在$\hat d$很大的时候影响计算速度。<br>
尝试将两个步骤联合起来。<br>

***

我们先来看一个简单的例子，对于二阶多项式转换，各种排列组合为：<br>
![2nd](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/2_2nd.png?raw=true)<br>
<br>
内积推导：<br>
![neiji](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/2_neiji.png?raw=true)<br>
其中$x^Tx'$是x空间中特征向量的内积。<br>
所以，$\Phi_2(x)$与$\Phi_2(x')$的内积的复杂度由原来的$O(d^2)$变成$O(d)$，只与x空间的维度d有关，而与z空间的维度$\hat d$无关。<br>

我们把合并特征转换和计算内积这两个步骤的操作叫做Kernel Function，用大写字母K表示。例如刚刚讲的二阶多项式例子，它的kernel function为：<br>

$$K_{\Phi}(x,x')=\Phi(x)^T\Phi(x')$$

$$K_{\Phi_2}(x,x')=1+(x^Tx')+(x^Tx')^2$$

有了kernel function之后，我们来看看它在SVM里面如何使用。在dual SVM中，二次项系数$q_{n,m}$中有z的内积计算，就可以用kernel function替换：

$$q_{n,m}=y_ny_mz_n^Tz_m=y_ny_mK(x_n,x_m)$$

所以，直接计算出$K(x_n,x_m)$，再代入上式，就能得到$q_{n,m}$的值。

$q_{n,m}$值计算之后，就能通过QP得到拉格朗日因子$\alpha_n$。然后，下一步就是计算b（取$\alpha_n$>0的点，即SV），b的表达式中包含z，可以作如下推导：

$$b=y_s-w^Tz_s=y_s-(\sum_{n=1}^N\alpha_ny_nz_n)^Tz_s=y_s-\sum_{n=1}^N\alpha_ny_n(K(x_n,x_s))$$

这样得到的b就可以用kernel function表示，而与z空间无关。

最终我们要求的矩$g_{SVM}$可以作如下推导：

$$g_{SVM}(x)=sign(w^T\Phi(x)+b)=sign((\sum_{n=1}^N\alpha_ny_nz_n)^Tz+b)=sign(\sum_{n=1}^N\alpha_ny_n(K(x_n,x))+b)$$

至此，dual SVM中我们所有需要求解的参数都已经得到了，而且整个计算过程中都没有在z空间作内积，即与z无关。我们把这个过程称为kernel trick，也就是把特征转换和计算内积两个步骤结合起来，用kernel function来避免计算过程中受$\hat d$的影响，从而提高运算速度。

***
总结一下，引入kernel function后，svm算法变成：
![svm](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/3_sumsvm.png?raw=true)<br>

每个步骤的时间复杂度为:<br>
![times](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/3_times.png?raw=true)<br>

我们把这种引入kernel function的SVM称为kernel SVM，它是基于dual SVM推导而来的。kernel SVM同样只用SV（$\alpha_n$>0）就能得到最佳分类面，而且整个计算过程中摆脱了$\hat d$的影响，大大提高了计算速度。

---

### 2.Polynomial Kernel 
二次多项式的kernel形式多样。
![poly](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/3_poly.png?raw=true)<br>
系数不同，内积就会有差异，就会代表不同的距离，最终可能会得到不同的SVM margin。
所以，系数不同，可能会得到不同的SVM分界线。
![choose](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/3_choose.png?raw=true)<br>

![distances](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/3_distances.png?raw=true)<br>


---

### 3.Gaussian Kernel

---

### 4.Comparison of Kernels

