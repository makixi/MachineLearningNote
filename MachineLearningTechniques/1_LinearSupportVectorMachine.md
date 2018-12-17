# Linear Support Vector Machine

---

### 1.Large-Margin Separating Hyperplane
![3lines](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/1_3lines.png?raw=true)<br>
上图三种分割方式都可以正确分割正负点，那么怎么分辨哪种方案更好？<br>
<br>
PLA可能会随机选择方案(最终结果与经过的错误点有关)<br>
都满足VC bound要求，模型复杂度一样。<br>

***

![3answers](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/1_3answers.png?raw=true)<br>
每个样本点距离分界线越远，就表明其对于测量误差的容忍度越高，就越安全。（PS:测量误差是一种类型的noise，而noise是导致过拟合的一个原因）<br>
![bestline](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/1_bestline.png?raw=true)<br>

***
分类线由权重w决定，目的就是找到使margin最大时对应的w值。即<br>
![bestmargin](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/1_maxmargin.png?raw=true)<br>

---

### 2.Standard Large-Margin Problem
如何计算点到直线的距离？<br>
首先，我们将权重$w(w_0,w_1,\cdots,w_d)$中的$w_0$拿出来，用b表示(即截距)。同时省去$x_0$项。这样，hypothesis就变成了$h(x)=sign(w^Tx+b)$。<br>
![shorten_X_and_W](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/1_shortenXW.png?raw=true)<br>
![distances](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/1_distances.png?raw=true)<br>

---

### 3.Support Vector Machine 


---


### 4.Reasons behind Large-Margin Hyperplane

