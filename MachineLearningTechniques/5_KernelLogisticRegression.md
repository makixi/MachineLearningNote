# Kernel Logistic Regression

---

### 1.Soft-Margin SVM as Regularized Model
最早有Hard-Margin Primal,然后推导出Hard-Margin Dual形式。<br>
后来，为了允许有错误点存在（或者noise），也为了避免模型太过复杂化，造成过拟合，建立了Soft-Margin Primal的数学表达式，并引入了新的参数C作为权衡因子，然后也推导了其Soft-Margin Dual形式。<br>
![soft](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/5_softad.png?raw=true)<br>

***

![zh](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/5_zh.png?raw=true)<br>
$\xi_n$描述的是点$(x_n,y_n)$ 距离$y_n(w^Tz_n+b)=1$的边界有多远。<br>
第一种情况是violating margin，即不满足$y_n(w^Tz_n+b)\geq1$。那么$\xi_n$可表示为：$\xi_n=1-y_n(w^Tz_n+b)>0$。<br>
第二种情况是not violating margin，即点$(x_n,y_n)$ 在边界之外，满足$y_n(w^Tz_n+b)\geq1$的条件，此时$\xi_n=0$。<br>
<br>
我们可以将两种情况整合到一个表达式中，对任意点：

$$\xi_n=max(1-y_n(w^Tz_n+b),0)$$

上式表明，如果有voilating margin，则$1-y_n(w^Tz_n+b)>0$，$\xi_n=1-y_n(w^Tz_n+b)$<br>
如果not violating margin，则$1-y_n(w^Tz_n+b)<0$，$\xi_n=0$。<br>
<br>
整合之后，我们可以把Soft-Margin SVM的最小化问题写成如下形式：

$$\frac12w^Tw+C\sum_{n=1}^Nmax(1-y_n(w^Tz_n+b),0)$$

经过这种转换之后，表征犯错误值大小的变量$\xi_n$就被消去了，转而由一个max操作代替。

***

![goal](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/5_goal.png?raw=true)<br>
L2 Regularization中的$\lambda$和Soft-Margin SVM中的C也是相互对应的，$\lambda$越大，w会越小，Regularization的程度就越大；C越小，$\hat{E_{in}}$会越大，相应的margin就越大。<br>
所以说增大C，或者减小$\lambda$，效果是一致的，Large-Margin等同于Regularization，都起到了防止过拟合的作用。

---

### 2.SVM versus Logistic Regression

---

### 3.SVM for Soft Binary Classification

---

### 4.Kernel Logistic Regression