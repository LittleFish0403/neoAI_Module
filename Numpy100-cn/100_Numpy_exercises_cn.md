# 100 个 NumPy 练习



该Numpy练习翻译自项目：https://github.com/rougier/numpy-100， <br/>
对于练习题中的题目进行**重新排序并汉化**，对表述不太清晰的题目**加入例子辅助**， <br/>
直接**将hint放在题目下方**，更方便新手去查询使用方法。<br/>
——文档有问题赶紧来联系我！

Numpy官方文档：https://numpy.org/doc/stable/index.html

运行`initialize.py`，然后在代码框内答题，自行查看运行结果是否正确 


```python
%run initialise.py
```

## 基础题
简单的numpy入门级题目，看完quickstart，简单查询函数使用方法即可

#### 1. 导入名为 `np` 的 numpy 包 (★☆☆)  
`hint: import … as`


```python

```


      Cell In[8], line 1
        import numpy as
                       ^
    SyntaxError: invalid syntax
    


#### 2. 打印 numpy 的版本和配置信息 (★☆☆)    
`hint: np.__version__, np.show_config`


```python

```

#### 3. 如何从命令行获取 numpy add 函数的文档？ (★☆☆) 
`hint: np.info`


```python

```

#### 4. 如何找到任何数组的内存大小 (★☆☆) 
`hint: size, itemsize`


```python

```

#### 5. 创建一个大小为 10 的空向量 (★☆☆) 
`hint: np.zeros`


```python

```

#### 6. 创建一个大小为 10 的空向量，但第五个值是 1 (★☆☆) 
`hint: array[4]`


```python

```

#### 7. 创建一个值范围从 10 到 49 的向量 (★☆☆) 
`hint: arange`


```python

```

#### 8. 反转一个向量（第一个元素变为最后一个） (★☆☆) 
`hint: array[::-1]`


```python

```

#### 9. 创建一个 3x3 的矩阵，值范围从 0 到 8 (★☆☆) 
`hint: reshape`


```python

```

#### 10. 找到 [1,2,0,0,4,0] 中非零元素的索引 (★☆☆) 
`hint: np.nonzero`


```python

```

#### 11. 创建一个 3x3 的单位矩阵 (★☆☆) 
`hint: np.eye`


```python

```

#### 12. 创建一个包含随机值的 3x3x3 数组 (★☆☆) 
`hint: np.random.random`


```python

```

#### 13. 创建一个 10x10 的随机值数组，并找到最小值和最大值 (★☆☆) 
`hint: min, max`


```python

```

#### 14. 创建一个大小为 30 的随机向量，并找到其平均值 (★☆☆) 
`hint: mean`


```python

```

#### 15. 创建一个 5x5 的矩阵，在对角线下方填充 1,2,3,4 (★☆☆) 
`hint: np.diag`


```python

```

#### 16. 使用 tile 函数创建一个 8x8 的棋盘格矩阵 (★☆☆) 
`hint: np.tile`


```python

```

#### 17. 创建一个二维数组，边界值为 1，内部值为 0 (★☆☆) 
`hint: array[1:-1, 1:-1]`
```
eg. [[1,1,1],
     [1,0,1],
     [1,1,1]]
```


```python

```

#### 18. 如何在现有数组周围添加一个填充 0 的边框？ (★☆☆) 
`hint: np.pad`


```python

```

#### 19. 创建一个 8x8 的矩阵，并填充成棋盘格模式 (★☆☆) 
`hint: array[::2]`


```python

```

#### 20. 考虑一个形状为 (6,7,8) 的数组，第 100 个元素的索引 (x,y,z) 是什么？ (★☆☆) 
`hint: np.unravel_index`


```python

```

#### 21. 将一个 5x5 的随机矩阵标准化 (★☆☆) 
`hint: (x -mean)/std`


```python

```

#### 22. 创建一个描述颜色的自定义 dtype（四个无符号字节 - RGBA） (★☆☆) 
`hint: np.dtype`


```python

```

#### 23. 将一个 5x3 的矩阵与一个 3x2 的矩阵相乘（真正的矩阵乘积） (★☆☆) 
`hint: np.dot`


```python

```

#### 24. 给定一个一维数组，反转(即乘以-1)所有介于 3 和 8 之间的元素，就地进行。 (★☆☆) 
`hint: >, <`


```python

```

#### 25. 以下表达式的结果是什么？ (★☆☆) 
`hint: NaN = not a number, inf = infinity`
```python
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
np.nan in set([np.nan])
0.3 == 3 * 0.1
```


```python

```

#### 26. 以下脚本的输出是什么？ (★☆☆)
```python
print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```


```python

```

####  27. 考虑一个整数向量 Z，这些表达式中哪些是合法的？ (★☆☆)
```python
Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z
```


```python

```

#### 28. 以下表达式的结果是什么？ (★☆☆)
```python
np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)
```


```python

```

#### 29. 如何将一个浮点数组四舍五入到**远离零**的地方？ (★☆☆)
`hint: np.uniform, np.copysign, np.ceil, np.abs, np.where`


```python

```

#### 30. 如何找到两个数组之间的公共值？ (★☆☆) 
`hint: np.intersect1d`


```python

```

## 入门题
需要开始频繁的检索官方文档

#### 31. 如何忽略所有 numpy 警告（不推荐）？ (★☆☆) 
`hint: np.seterr, np.errstate`



```python

```

#### 32. 以下表达式为真吗？ (★☆☆)
```python
np.sqrt(-1) == np.emath.sqrt(-1)
```


```python

```

####  33. 如何就地计算 ((A+B)*(-A/2)) 而不进行复制？ (★★☆)
`hint: np.add(out=), np.negative(out=), np.multiply(out=), np.divide(out=)`


```python

```

#### 34. 使用 4 种不同的方法提取一个随机正数组的整数部分 (★★☆) 
`hint: %, np.floor, astype, np.trunc`


```python

```

#### 35. 创建一个 5x5 的矩阵，每行的值范围从 0 到 4 (★★☆) 
`hint: np.arange`


```python

```

#### 36. 创建一个大小为 10 的向量，值范围从 0 到 1（不包括 0 和 1） (★★☆) 
`hint: np.linspace`


```python

```

#### 37. 创建一个大小为 10 的随机向量，并对其进行排序 (★★☆) 
`hint: sort`


```python

```

####  38. 考虑两个随机数组 A 和 B，检查它们是否相等 (★★☆) 
`hint: np.allclose, np.array_equal`


```python

```

#### 39. 考虑一个随机 10x2 矩阵，表示笛卡尔坐标，将其转换为极坐标 (★★☆) 
`hint: np.sqrt, np.arctan2`


```python

```

#### 40. 创建一个大小为 10 的随机向量，并将最大值替换为 0 (★★☆) 
`hint: argmax`


```python

```

#### 41. 创建一个覆盖 [0,1]x[0,1] 区域的 `x` 和 `y` 坐标的结构化数组 (★★☆) 
`hint: np.meshgrid`


```python

```

#### 42. 给定两个数组，X 和 Y，构造 柯西（Cauchy） 矩阵 C (Cij =1/(xi - yj)) (★★☆) 
`hint: np.subtract.outer`


```python

```

#### 43. 打印每个 numpy 标量类型的最小和最大可表示值 (★★☆) 
`hint: np.iinfo, np.finfo, eps`


```python

```

#### 44. 如何在一个向量中找到最接近给定标量的值？ (★★☆) 
`hint: argmin`


```python

```

#### 45. 创建一个表示位置 (x,y) 和颜色 (r,g,b) 的结构化数组 (★★☆) 
`hint: dtype`


```python

```

#### 46. 考虑一个形状为 (100,2) 的随机向量，表示坐标，找到逐点距离 (★★☆) 
`hint: np.atleast_2d, T, np.sqrt`


```python

```

#### 47. 如何将一个浮点（32 位）数组就地转换为整数（32 位）？ 
`hint: view and [:] =`


```python

```

#### 48. 如何读取以下文件？ (★★☆) 
`hint: np.genfromtxt`
```python
1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11
```



```python

```

#### 49. numpy 数组的 enumerate 等效函数是什么？ (★★☆)
`hint: np.ndenumerate, np.ndindex`


```python

```

####  50. 生成一个通用的二维高斯数组 (★★☆) 
`hint: np.meshgrid, np.exp`


```python

```

#### 51. 如何在一个二维数组中随机放置 p 个元素？ (★★☆) 
`hint: np.put, np.random.choice`


```python

```

#### 52. 如何减去矩阵每一行的均值？ (★★☆)
`hint: mean(axis=,keepdims=)`


```python

```

#### 53. 如何按第 n 列对数组进行排序？ (★★☆)
`hint: argsort`
```python
eg. 排序前：
    1 2 3
    4 1 6
    7 0 5

按第二列排序后：
    7 0 5
    4 1 6
    1 2 3 
```



```python

```

#### 54. 如何判断给定的二维数组是否有空列？ (★★☆)
`hint: any, ~`


```python

```

#### 55. 从数组中找到最接近给定值的值 (★★☆)
`hint: np.abs, argmin, flat`


```python

```

#### 56. 考虑形状为 (1,3) 和 (3,1) 的两个数组，如何使用迭代器计算它们的和？ (★★☆)
`hint: np.nditer`



```python

```

#### 57. 考虑一个给定的向量，如何将第二个向量索引处的每个元素加 1（注意重复索引）？ (★★★)
`hint: np.bincount | np.add.at`
```python
处理前：
Z = [1, 2, 3]
I = [0, 0, 2, 1, 1, 0]

I索引下'0':3,'1':2,'2':1
处理后：
Z = [1+3, 2+2, 3+1]
```


```python

```

#### 58. 404 NOT FOUND


```python

```

#### 59. 如何基于索引列表 (I) 将向量 (X) 的元素累积到数组 (F) 中？ (★★★)
`hint: np.bincount`
```python
X = [1,2,3,4,5,6]
I = [1,3,9,3,4,1]

F = [0,7,0,6,5,0,0,0,0,3]
```


```python

```

#### 60. 如何获取点积的对角线？ (★★★)
`hint: np.diag`


```python

```

#### 61. 考虑一个四维数组，如何一次对最后两个轴求和？ (★★★)
`hint: sum(axis=(-2,-1))`


```python

```

#### 62. 考虑一个一维向量 D，如何使用描述子集索引的相同大小的向量 S 计算 D 的子集均值？ (★★★)
`hint: np.bincount`


```python

```

#### 63. 考虑向量 [1, 2, 3, 4, 5]，如何构建一个在每个值之间插入 3 个连续零的新向量？ (★★★)
`hint: array[::4]`


```python

```

#### 64. 考虑一个形状为 (5,5,3) 的数组，如何将其与形状为 (5,5) 的数组相乘？ (★★★)
`hint: array[:, :, None]`


```python

```

#### 65. 如何交换数组的两行？ (★★★)
`hint: array[[]] = array[[]]`



```python

```

#### 66. 如何取反一个布尔值，或就地改变一个浮点数的符号？ (★★☆)
`hint: np.logical_not, np.negative`


```python

```

#### 67. 计算矩阵的秩 (★★★)
`hint: np.linalg.svd, np.linalg.matrix_rank`


```python

```

#### 68. 如何找到数组中最频繁的值？(★★☆)
`hint: np.bincount, argmax`


```python

```

#### 69. 创建一个二维数组子类，使得 Z[i,j] == Z[j,i] (★★★)
`hint: class method`


```python

```

#### 70. 如何获得数组的 n 个最大值 (★★★)
`hint: np.argsort | np.argpartition`


```python

```

## 进阶题
更加偏向于代码编写和思路转换，需要对numpy有一定的了解

#### 71. 创建一个具有名称属性的数组类 (★★★)
`hint: class method`


```python

```

#### 72. 考虑一组描述 10 个三角形（具有共享顶点）的 10 个三元组，找到构成所有三角形的唯一线段集合 (★★★)
`hint: repeat, np.roll, np.sort, view, np.unique`


```python

```

#### 73. 给定一个对应于 bincount 的排序数组 C，如何生成一个数组 A 使得 np.bincount(A) == C？ (★★★)
`hint: np.repeat`


```python

```

#### 74. 考虑一个 (w,h,3) 形状的图像 (dtype=ubyte)，计算唯一颜色的数量 (★★☆)
`hint: np.unique`


```python

```

#### 75. 考虑两组点 P0,P1 描述线（2D）和一个点 p，如何计算从 p 到每条线 i (P0[i],P1[i]) 的距离？ (★★★)
`No hints provided...`


```python

```

#### 76. 考虑两组点 P0,P1 描述线（2D）和一组点 P，如何计算从每个点 j (P[j]) 到每条线 i (P0[i],P1[i]) 的距离？ (★★★)
`No hints provided...`


```python

```

#### 77. 考虑一个任意数组，编写一个函数以固定形状提取子部分并以给定元素为中心（必要时用 `fill` 值填充） (★★★)
`hint: minimum maximum`


```python

```

#### 78. 考虑形状为 (n,n) 的 p 个矩阵和形状为 (n,1) 的 p 个向量，如何一次计算 p 个矩阵乘积的和？（结果形状为 (n,1)）(★★★)
`hint: np.tensordot`


```python

```

#### 79. 给定任意数量的向量，构建笛卡尔积（每个项目的每种组合） (★★★)
`hint: np.indices`


```python

```

#### 80. 考虑一个大的向量 Z，使用 3 种不同的方法计算 Z 的立方 (★★★)
`hint: np.power, *, np.einsum`


```python

```

#### 81. 考虑形状为 (8,3) 和 (2,2) 的两个数组 A 和 B，如何找到 A 的行，其中包含 B 的每一行的元素，而不考虑 B 中元素的顺序？ (★★★)
`hint: np.where`


```python

```

#### 82. 考虑一个 10x3 的矩阵，提取具有不等值的行（例如 [2,2,3]）(★★★)
`No hints provided...`


```python

```

#### 83. 将一个整数向量转换为矩阵的二进制表示 (★★★)
`hint: np.unpackbits`


```python

```

#### 84. 给定一个二维数组，如何提取唯一的行？ (★★★)
`hint: np.ascontiguousarray | np.unique`


```python

```

#### 85. 考虑 2 个向量 A 和 B，编写 inner, outer, sum 和 mul 函数的 einsum 等效函数 (★★★)
`hint: np.einsum`


```python

```

#### 86. 考虑由两个向量 (X,Y) 描述的路径，如何使用等距样本对其进行采样 (★★★)?
`hint: np.cumsum, np.interp`


```python

```

#### 87. 给定一个整数 n 和一个二维数组 X，从 X 中选择可以解释为具有 n 个度的多项分布的行，即仅包含整数并且和为 n 的行。 (★★★)
`hint:np.logical_and.reduce, np.mod`


```python

```

#### 88. 计算一维数组 X 的 bootstrap 95% 置信区间（即，用替换的方式重新采样数组的元素 N 次，计算每个样本的均值，然后计算均值的百分位数）。 (★★★)
`hint: np.percentile`


```python

```

## 选做题
不太常用，可以作为了解

#### 89. 如何获取昨天、今天和明天的日期？ (★☆☆)
`hint: np.datetime64, np.timedelta64`


```python

```

#### 90. 如何获取对应于 2016 年 7 月的所有日期？ (★★☆)
`hint: np.arange(dtype=datetime64['D'])`


```python

```

#### 91. 使数组不可变（只读） (★★☆)
`hint: flags.writeable`


```python

```

#### 92. 如何比 np.sum 更快地对小数组求和？ (★★☆)
`hint: np.add.reduce`


```python

```

#### 93. 如何打印数组的所有值？ (★★☆)
`hint: np.set_printoptions`


```python

```

#### 94. 404 NOT FOUND


```python

```

#### 95. 如何使用滑动窗口计算数组的平均值？ (★★★)
`hint: np.cumsum, from numpy.lib.stride_tricks import sliding_window_view (np>=1.20.0)`


```python

```

#### 96. 考虑一个一维数组 Z，构建一个二维数组，其第一行为 (Z[0],Z[1],Z[2])，每一行向后移动 1（最后一行为 (Z[-3],Z[-2],Z[-1])） (★★★)
`hint: from numpy.lib import stride_tricks, from numpy.lib.stride_tricks import sliding_window_view (np>=1.20.0)`


```python

```

#### 97. 考虑数组 Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]，如何生成数组 R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]？ (★★★)
`hint: stride_tricks.as_strided, from numpy.lib.stride_tricks import sliding_window_view (np>=1.20.0)`


```python

```

#### 98. 从一个随机的 10x10 矩阵中提取所有连续的 3x3 块 (★★★)
`hint: stride_tricks.as_strided, from numpy.lib.stride_tricks import sliding_window_view (np>=1.20.0)`


```python

```

#### 99. 考虑一个 16x16 的数组，如何获得块和（块大小为 4x4）？ (★★★)
`hint: np.add.reduceat, from numpy.lib.stride_tricks import sliding_window_view (np>=1.20.0)`



```python

```

#### 100. 如何从常规数组创建记录数组？ (★★★)
`hint: np.core.records.fromarrays`


```python

```
