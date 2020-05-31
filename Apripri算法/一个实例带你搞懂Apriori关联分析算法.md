# 关联分析
>Apriori算法
优点：易编码实现。
缺点：在大数据集上可能较慢。
适用数据类型：数值型或者标称型数据。

关联分析是一种在大规模数据集中寻找有趣关系的任务。这些关系可以有两种形式：**频繁项集或者关联规则。**

**频繁项集**是经常出现在一块的物品的集合
**关联规则**（association rules）暗示两种物品之间可能存在很强的关系。下面会用一个例子来说明这两种概念。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200528234031834.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTc1NTMzMg==,size_16,color_FFFFFF,t_70)
频繁项集是指那些经常出现在一起的物品集合，图中的集合{葡萄酒，尿布, 豆奶}就是频繁项集的一个例子

如尿布 ➞葡萄酒的一起出现的频率较高就是一个关联规则。这意味着如果有人买了尿布，那么他很可能也会买葡萄酒。

数据集中包含该项集的记录所占的比例就是项集的**支持度**

可以得到，{豆奶}的支持度为4/5。而在5条交易记录中有3条包含{豆奶，尿布}，因此{豆奶，尿布}的支持度为3/5。支持度是针对项集来说的，因此可以定义一个最小支持度，而只保留满足最小支持度的项集。 
**可信度或置信度**（confidence）是针对一条诸如{尿布} ➞ {葡萄酒}的关联规则来定义的。这条规则的可信度被定义为“支持度({尿布, 葡萄酒})/支持度({尿布})”。

由{尿布, 葡萄酒}的支持度为3/5，尿布的支持度为4/5，所以“尿布 ➞ 葡萄酒”的可信度为3/4=0.75。这意味着对于包含“尿布”的所有记录，我们的规则对其中75%的记录都适用。

支持度和可信度是用来量化关联分析是否成功的方法。假设想找到支持度大于0.8的所有项集，应该如何去做？一个办法是生成一个物品所有可能组合的清单，然后对每一种组合统计它出现的频繁程度，但当物品成千上万时，上述做法非常非常慢。

Apriori原理会减少关联规则学习时所需的计算量。

# Apriori 原理
**先验原理：如果一个项集是频繁的，那么他的子集也是频繁的；反过来，如果一个子集是非频繁的，该项集也是非频繁**

Apriori算法的一般过程 
1. 收集数据：使用任意方法。 
2.  准备数据：任何数据类型都可以，因为我们只保存集合。
3.  分析数据：使用任意方法。
4.  训练算法：使用Apriori算法来找到频繁项集。
5.  测试算法：不需要测试过程。 
6.  使用算法：用于发现频繁项集以及物品之间的关联规则。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200529090019972.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTc1NTMzMg==,size_16,color_FFFFFF,t_70)
上图就是扫描数据的过程，在扫描完所有数据之后，使用统计得到的总数除以总的交易记录数，就可以得到支持度。
对于包含n种物品的数据集共有$2^n -1$种项集组合。即使只出售100种商品的商店也会有1.26×1030种可能的项集组合。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200530113902482.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTc1NTMzMg==,size_16,color_FFFFFF,t_70)
若已知123是频繁项集，则他上面的那几个子集都是频繁的

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200529093349940.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTc1NTMzMg==,size_16,color_FFFFFF,t_70)
上图中，假设已知阴影项集{2,3}是非频繁的。利用这个知识，我们就知道项集{0,2,3}，{1,2,3}以及{0,1,2,3}也是非频繁的。这也就是说，一旦计算出了{2,3}的支持度，知道它是非频繁的之后，就不需要再计算{0,2,3}、{1,2,3}和{0,1,2,3}的支持度，因为我们知道这些集合不会满足我们的要求。使用该原理就可以避免项集数目的指数增长，从而在合理时间内计算出频繁项集。

#  使用 Apriori 算法来发现频繁集

**首先需要找到频繁项集，然后才能获得关联规则。**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200530120311378.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTc1NTMzMg==,size_16,color_FFFFFF,t_70)
C1，C2，... Ck分别表示1-项集，2-项集，...k-项集，是候选项集 
L1, L2, ....Lk分别表示有k个数据项的频繁项集。
create是根据数据集创建候选项集
Scan表示数据集扫描函数。该函数起到的作用是支持度过滤，满足最小支持度的项集才留下，不满足最小支持度的项集直接舍掉。


Apriori算法是发现频繁项集的一种方法。两个输入参数分别是最小支持度和数据集。该算法首先会生成所有单个物品的项集列表。接着扫描交易记录来查看哪些项集满足最小支持度要求，那些不满足最小支持度的集合会被去掉。然后，对剩下来的集合进行组合以生成包含两个元素的项集。接下来，再重新扫描交易记录，去掉不满足最小支持度的项集。该过程重复进行直到所有项集都被去掉。
## 生成候选项集
对数据集中的每条交易记录transaction
对每个候选项集c： 
		检查一下c是否是transaction的子集：
		如果是，则增加ca的计数值
对每个候选项集： 
		如果其支持度不低于最小值，则保留该项集
		返回所有频繁项集列表
		
```python
#创建一组数据
def loadDataSet():
    dataset=[[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
    return dataset

def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        #print(transaction)
        for item in transaction:
            #print(item)
            if not {item} in C1:
                #print([item])
                C1.append({item})
        #print(C1)
    C1.sort()
    #就是对c1用frozenset的方法,frozenset就是把这个数集合冰冻起来不能进行修改
    return list(map(frozenset, C1))# use frozen set so we can use it as a key in a dict

c1=createC1(dataset)
#三个参数 分别是数据集、候选项集列表Ck以及最小支持度minSupport
def scanD(D, Ck, minSupport):
    ssCnt = { }
    for tid in D:
        for can in Ck:
            #issubset方法用于判断集合的所有元素是否都包含在指定集合中，如果是则返回 True，否则返回 False。
            if can.issubset(tid):
                #判断can是否是tid的子集，has_key() 函数用于判断键key是否存在于字典中，如果键在字典dict里返回true，否则返回false。
                if can not in ssCnt:
                    ssCnt[can]=1
                else:
                    ssCnt[can] += 1

    numItems = float(len(D))
    retList = [ ]
    supportData = { }
    for key in ssCnt:
        support = ssCnt[key] /numItems
        supportData[key] = support
        if support >= minSupport:
            retList.append(key)
    return retList, supportData
dataset=loadDataSet()
c1=createC1(dataset)
l1,suppordata=scanD(dataset,c1,0.5)
print(l1)
print(suppordata)

```

```py
[frozenset({1}), frozenset({3}), frozenset({2}), frozenset({5})]

{frozenset({1}): 0.5, frozenset({3}): 0.75, frozenset({4}): 0.25, frozenset({2}): 0.75, frozenset({5}): 0.75}
```

代码实现原理如下图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200530165224572.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTc1NTMzMg==,size_16,color_FFFFFF,t_70)

