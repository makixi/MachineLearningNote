# Dual Support Vector Machine

---

### 1.Motivation of Dual SVM
![svm](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/2_svmvs.png?raw=true)<br>
Original SVM二次规划问题的变量个数是$\hat d +1$，有N个限制条件；<br>
Equivalent SVM二次规划变量个数为N个，有N+1个限制条件。<br>
这种对偶SVM的好处就是问题只跟N有关，与$\hat d$无关，这样就不会出现当$\hat d$无限大时难以求解的情况。<br>

***

![lamda](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/2_lamda.png?raw=true)<br>
Regularization中，在最小化$E_{in}$的过程中，也添加了限制条件：$w^Tw\leq C$。我们的求解方法是引入拉格朗日因子$\lambda$<br>，将有条件的最小化问题转换为无条件的最小化问题：$min\ E_{aug}(w)=E_{in}(w)+\frac{\lambda}{N}w^Tw$，最终得到的w的最优化解为：<br>
<br>
$$\nabla E_{in}(w)+\frac{2\lambda}{N}w=0$$<br>
<br>
所以，在regularization问题中，$\lambda$是已知常量，求解过程变得容易。那么，对于dual SVM问题，同样可以引入$\lambda$，将条件问题转换为非条件问题，只不过$\lambda$是未知参数，且个数是N，需要对其进行求解。<br>

***

现在要将条件问题转化成非条件问题。<br>
SVM中，目标是：$min\ \frac12w^Tw$，条件是：$y_n(w^Tz_n+b)\geq 1,\ for\ n=1,2,\cdots,N$。首先，我们令拉格朗日因子为$\alpha_n$（区别于regularization），构造一个函数：<br>
<br>
$$L(b,w,\alpha)=\frac12w^Tw+\sum_{n=1}^N\alpha_n(1-y_n(w^Tz_n+b))$$
<br>
这个函数右边第一项是SVM的目标，第二项是SVM的条件和拉格朗日因子$\alpha_n$的乘积。我们把这个函数称为拉格朗日函数，其中包含三个参数：b，w，$\alpha_n$。<br>
![lagrange](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/2_lagrange.png?raw=true)<br>

---


### 2.Lagrange Dual SVM 

---

### 3.Solving Dual SVM 


---

### 4.Messages behind Dual SVM 
