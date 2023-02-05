# -ML
有关机器学习的知识上传，会有一些系统的也有细碎的知识，还有一些小项目
> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [towardsai.net](https://towardsai.net/p/tutorials/mathematical-intuition-behind-the-gradient-descent-algorithm)

> 作者：Towards AI 编辑团队 最初发表于 Towards AI 世界领先的人工智能和......

![](https://cdn-images-1.medium.com/max/1024/1*1OsPgY9vuwuHOxUSOsmzQA.jpeg)

# 梯度下降算法及其变体

#### 梯度下降系列博客：

1.  [梯度下降算法](https://pub.towardsai.net/the-gradient-descent-algorithm-defddd1d312e)
2.  [梯度下降算法背后的数学直觉](https://pub.towardsai.net/mathematical-intuition-behind-the-gradient-descent-algorithm-143a051c3fa9)（你在这里！）
3.  [梯度下降算法及其变体](https://pub.towardsai.net/the-gradient-descent-algorithm-and-its-variants-e0915796dbf2)

1.  #3080)

#### 介绍：

欢迎！今天，我们正在努力开发一种强大的数学直觉，以了解梯度下降 算法如何为其参数找到最佳值。拥有这种感觉可以帮助您发现机器学习输出中的错误，并更加了解梯度下降 算法如何使机器学习如此强大。在接下来的几页中，我们将推导均方误差函数的梯度下降算法方程。我们将使用此博客的结果来编写梯度下降算法的代码。让我们深入研究吧！

#### 均方误差的梯度下降算法推导：

#### 1. 第 1 步：

输入数据显示在下面的矩阵中。在这里，我们可以观察到有**_m_**个训练示例和**_n_**个特征。

![](https://cdn-images-1.medium.com/max/305/1*lU8vZWRfh52PuAKwP3olsg.png)

> **维度：** X = (m, n)

#### 2. 第 2 步：

预期的输出矩阵如下所示。我们预期的输出矩阵大小为**_m*1_**，因为我们有**_m_**个训练样本。

![](https://cdn-images-1.medium.com/max/162/1*fRsvMUopHPJOb_POki5zMg.png)

> **维度：** Y = (m, 1)

#### 3. 第 3 步：

我们将在要训练的参数中添加一个偏差元素。

![](https://cdn-images-1.medium.com/max/97/1*sflcYpHY5v0BrXMyhBbv6A.png)

> **维度：** α = (1, 1)

#### 4. 第 4 步：

在我们的参数中，我们有权重矩阵。权重矩阵将有**_n 个_**元素。这里，**_n_**是我们训练 数据集的特征数量。

![](https://cdn-images-1.medium.com/max/304/1*z0-F--JyLKyqSigfmwXgIA.png)

> **维度：** β = (1, n)

#### 5. 第 5 步：

![](https://cdn-images-1.medium.com/max/922/1*v7oR4EkpqGiiz3IXF5EmNg.png)

每个训练示例的预测值由下式给出，

![](https://cdn-images-1.medium.com/max/583/1*YxqK_4DqSd2IQIbLsurMaQ.png)

请注意，我们正在对权重矩阵 (β) 进行转置，以使维度与矩阵乘法规则兼容。

> **维度：** predicted_value = (1, 1) + (m, n) * (1, n)

> — 对权重矩阵 (β) 进行转置 —

> **维度：** predicted_value = (1, 1) + (m, n) * (n, 1) = (m, 1)

#### 6. 第 6 步：

均方误差定义如下。

![](https://cdn-images-1.medium.com/max/382/1*nGvcTnyQCFJIYnwuXinlww.png)

> **维度：**成本=标量函数

#### 7. 第 7 步：

在这种情况下，我们将使用以下梯度下降规则来确定最佳参数。

![](https://cdn-images-1.medium.com/max/197/1*yzVFACxEKSdRN8_jNQqZxA.png)

> **维度：** α = (1, 1) & β = (1, n)

#### 8. 第 8 步：

现在，让我们找到成本函数关于偏置元素 ( **_α_** ) 的偏导数。

![](https://cdn-images-1.medium.com/max/303/1*D_sGwXtH-MZY4H-T8g3ICg.png)

> **维度:** (1, 1)

#### 9. 第 9 步：

现在，我们正在尝试简化上述方程以找到偏导数。

![](https://cdn-images-1.medium.com/max/254/1*qKQ6R1ys_WkNHgXdycJoeQ.png)

> **维度：** u = (m, 1)

#### 10. 第 10 步：

基于[Step — 9](#6453)，我们可以将成本函数写为，

![](https://cdn-images-1.medium.com/max/145/1*9KKmTMOxA4CFYdqRwdniLA.png)

> **维度：**标量函数

#### 11. 第 11 步：

接下来，我们将使用链式法则计算成本函数关于截距 ( **_α_** ) 的偏导数。

![](https://cdn-images-1.medium.com/max/108/1*AOz3hM8P7M9ryS72DTNLgg.png)

> **尺寸：（**米，1）

#### 12. 第 12 步：

接下来，我们正在计算Step_11的偏导数的第一部分。

![](https://cdn-images-1.medium.com/max/436/1*B_003FSCdtk7Bk7ctnbiuQ.png)

> **尺寸：（**米，1）

#### 13. 第 13 步：

接下来，我们计算Step — 11的偏导数的第二部分。

![](https://cdn-images-1.medium.com/max/578/1*IbTxwyCLQrUD65MgaybXlQ.png)

> **维度：**标量函数

#### 14. 第 14 步：

[接下来，我们将第 12](#bc41)步和[第 13](#5fad)步的结果相乘，得到最终的结果。

![](https://cdn-images-1.medium.com/max/349/1*AY9VrdGKSrK7O3IWcVsRbQ.png)

> **尺寸：（**米，1）

#### 15. 第 15 步：

接下来，我们将使用链式法则计算成本函数关于权重 ( **_β_** ) 的偏导数。

![](https://cdn-images-1.medium.com/max/125/1*UorQBxbMGW0U-UBIDH7bBQ.png)

> **尺寸：**（1，n）

#### 16. 第 16 步：

接下来，我们计算Step — 15的偏导数的第二部分。

![](https://cdn-images-1.medium.com/max/617/1*cOizwlssMLx4Rzz7JOnECg.png)

> **尺寸：（**米，n）

#### 17. 第 17 步：

接下来，我们将Step_12和Step_16的结果相乘，得到最后的偏导数结果。

![](https://cdn-images-1.medium.com/max/420/1*C9LC6dmbVsm007Lwmi3qlg.png)

现在，因为我们想要有**_n_**个权重值，所以我们将从上面的等式中删除求和部分。

![](https://cdn-images-1.medium.com/max/206/1*Wsj3CY6hIgCQhqGXQFo0Iw.png)

请注意，这里我们必须转置计算的第一部分，使其与矩阵乘法规则兼容。

> **尺寸：（**米，1）*（米，n）

> — 对错误部分进行转置 —

> **尺寸：**（1，米）*（米，n）=（1，n）

#### 18. 第 18 步：

接下来，我们把所有的计算值放在Step_7中，计算更新**α**的梯度规则 。

![](https://cdn-images-1.medium.com/max/332/1*2AMDVChWSsnBFZvXhM01CA.png)

> **维度：** α = (1, 1)

#### 19. 第 19 步：

接下来，我们把所有的计算值放在Step_7中，计算出更新**_β_**的梯度规则 。

![](https://cdn-images-1.medium.com/max/367/1*LOpzgk-Y9lOaDwkxuK7jUg.png)

请注意，我们必须转置误差值以使函数与矩阵乘法规则兼容。

> **维度：** β = (1, n) - (1, n) = (1, n)

#### 梯度下降算法的工作示例：

现在，让我们举个例子看看梯度下降算法是如何找到最佳参数值的。

#### 1. 第 1 步：

输入数据显示在下面的矩阵中。在这里，我们可以观察到有**_4 个_**训练示例和**_2 个_**特征。

![](https://cdn-images-1.medium.com/max/160/1*is1vO503YO0QRbA7OERTTQ.png)

#### 2. 第 2 步：

预期的输出矩阵如下所示。我们预期的输出矩阵大小为**_4*1_**，因为我们有**_4 个_**训练示例。

![](https://cdn-images-1.medium.com/max/145/1*wEOOKkYOuAbhRGCP3fm48A.png)

#### 3. 第 3 步：

我们将在要训练的参数中添加一个偏差元素。在这里，我们为偏置选择初始值 0。

![](https://cdn-images-1.medium.com/max/134/1*ZUSC0EQU4gm2mshwRwcAbg.png)

#### 4. 第 4 步：

在我们的参数中，我们有权重矩阵。权重矩阵将有 2 个元素。这里，2 是我们训练数据集的特征数量。最初，我们可以为权重矩阵选择任意随机数。

![](https://cdn-images-1.medium.com/max/216/1*KUNVExsHn8LZt6g84uN_1A.png)

#### 5. 第 5 步：

接下来，我们将使用输入矩阵、权重矩阵和偏差来预测值。

![](https://cdn-images-1.medium.com/max/570/1*7FYZzWCw7guSuKl1WvX_Bw.png)

#### 6. 第 6 步：

接下来，我们使用以下等式计算成本。

![](https://cdn-images-1.medium.com/max/394/1*3BvU7wSnXvqg5q0LrJBkvw.png)

#### 7. 第 7 步：

接下来，我们正在计算成本函数关于偏置元素的偏导数。我们将在梯度下降算法中使用这个结果来更新偏置参数的值。

![](https://cdn-images-1.medium.com/max/237/1*aaZ7qXfLKzLPLx07SiwL6Q.png)

#### 8. 第 8 步：

接下来，我们计算成本函数关于权重矩阵的偏导数。我们将在梯度下降算法中使用这个结果来更新权重矩阵的值。

![](https://cdn-images-1.medium.com/max/311/1*c_aIzriQhg0SS18DL_xx5A.png)

#### 9. 第 9 步：

接下来，我们定义学习率的值。学习率是控制模型学习速度的参数。

![](https://cdn-images-1.medium.com/max/190/1*BMq13d_g3J7_YiVZ6KhDjg.png)

#### 10. 第 10 步：

接下来，我们使用梯度下降规则来更新偏置元素的参数值。

![](https://cdn-images-1.medium.com/max/355/1*mGg8EiF30XDxuuD1mDHBiQ.png)

#### 11. 第 11 步：

接下来，我们使用梯度下降规则来更新权重矩阵的参数值。

![](https://cdn-images-1.medium.com/max/521/1*1O19AdxEkchpvmFIoi5EHA.png)

#### 12. 第 12 步：

现在，我们重复此过程进行多次迭代，以找到最适合我们模型的参数。在每次迭代中，我们使用参数的更新值。

#### 尾注：

因此，这就是我们如何使用均方误差的梯度下降算法找到更新规则。我们希望这能激发您的好奇心，让您渴望获得更多机器学习知识。我们将使用我们在这里推导出的规则在以后的博客中实现梯度下降算法，所以不要错过梯度下降系列的第三部分，所有这些都汇集在一起​​——大结局！

