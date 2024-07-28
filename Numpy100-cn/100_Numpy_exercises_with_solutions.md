```markdown
## 基础题
简单的numpy入门级题目，看完quickstart，简单查询函数使用方法即可

#### 1. 导入名为 `np` 的 numpy 包 (★☆☆)  


```python
import numpy as np
```

#### 2. 打印 numpy 的版本和配置信息 (★☆☆)    


```python
print(np.__version__)
print(np.show_config())
```

#### 3. 如何从命令行获取 numpy add 函数的文档？ (★☆☆) 


```python
print(np.info(np.add))
```

#### 4. 如何找到任何数组的内存大小 (★☆☆) 


```python
arr = np.array([1, 2, 3])
print(arr.size * arr.itemsize)
```

#### 5. 创建一个大小为 10 的空向量 (★☆☆) 


```python
arr = np.zeros(10)
print(arr)
```

#### 6. 创建一个大小为 10 的空向量，但第五个值是 1 (★☆☆) 


```python
arr = np.zeros(10)
arr[4] = 1
print(arr)
```

#### 7. 创建一个值范围从 10 到 49 的向量 (★☆☆) 


```python
arr = np.arange(10, 50)
print(arr)
```

#### 8. 反转一个向量（第一个元素变为最后一个） (★☆☆) 


```python
arr = np.arange(10, 50)
arr = arr[::-1]
print(arr)
```

#### 9. 创建一个 3x3 的矩阵，值范围从 0 到 8 (★☆☆) 


```python
arr = np.arange(9).reshape(3, 3)
print(arr)
```

#### 10. 找到 [1,2,0,0,4,0] 中非零元素的索引 (★☆☆) 


```python
arr = [1, 2, 0, 0, 4, 0]
nz = np.nonzero(arr)
print(nz)
```

#### 11. 创建一个 3x3 的单位矩阵 (★☆☆) 


```python
arr = np.eye(3)
print(arr)
```

#### 12. 创建一个包含随机值的 3x3x3 数组 (★☆☆) 


```python
arr = np.random.random((3, 3, 3))
print(arr)
```

#### 13. 创建一个 10x10 的随机值数组，并找到最小值和最大值 (★☆☆) 


```python
arr = np.random.random((10, 10))
print(arr.min(), arr.max())
```

#### 14. 创建一个大小为 30 的随机向量，并找到其平均值 (★☆☆) 


```python
arr = np.random.random(30)
print(arr.mean())
```

#### 15. 创建一个 5x5 的矩阵，在对角线下方填充 1,2,3,4 (★☆☆) 


```python
arr = np.diag(1+np.arange(4), k=-1)
print(arr)
```

#### 16. 使用 tile 函数创建一个 8x8 的棋盘格矩阵 (★☆☆) 


```python
arr = np.tile(np.array([[0, 1], [1, 0]]), (4, 4))
print(arr)
```

#### 17. 创建一个二维数组，边界值为 1，内部值为 0 (★☆☆) 

```
eg. [[1,1,1],
     [1,0,1],
     [1,1,1]]
```


```python
arr = np.ones((5, 5))
arr[1:-1, 1:-1] = 0
print(arr)
```

#### 18. 如何在现有数组周围添加一个填充 0 的边框？ (★☆☆) 


```python
arr = np.ones((3, 3))
arr = np.pad(arr, pad_width=1, mode='constant', constant_values=0)
print(arr)
```

#### 19. 创建一个 8x8 的矩阵，并填充成棋盘格模式 (★☆☆) 


```python
arr = np.zeros((8, 8), dtype=int)
arr[1::2, ::2] = 1
arr[::2, 1::2] = 1
print(arr)
```

#### 20. 考虑一个形状为 (6,7,8) 的数组，第 100 个元素的索引 (x,y,z) 是什么？ (★☆☆) 


```python
print(np.unravel_index(100, (6, 7, 8)))
```

#### 21. 将一个 5x5 的随机矩阵标准化 (★☆☆) 


```python
arr = np.random.random((5, 5))
arr = (arr - np.mean(arr)) / (np.std(arr))
print(arr)
```

#### 22. 创建一个描述颜色的自定义 dtype（四个无符号字节 - RGBA） (★☆☆) 


```python
color = np.dtype([("r", np.ubyte, 1),
                  ("g", np.ubyte, 1),
                  ("b", np.ubyte, 1),
                  ("a", np.ubyte, 1)])
```

#### 23. 将一个 5x3 的矩阵与一个 3x2 的矩阵相乘（真正的矩阵乘积） (★☆☆) 


```python
arr1 = np.dot(np.ones((5, 3)), np.ones((3, 2)))
print(arr1)
```

#### 24. 给定一个一维数组，反转(即乘以-1)所有介于 3 和 8 之间的元素，就地进行。 (★☆☆) 


```python
arr = np.arange(11)
arr[(3 < arr) & (arr <= 8)] *= -1
print(arr)
```

#### 25. 以下表达式的结果是什么？ (★☆☆) 

```python
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
np.nan in set([np.nan])
0.3 == 3 * 0.1
```


```python
print(0 * np.nan)          # nan
print(np.nan == np.nan)    # False
print(np.inf > np.nan)     # False
print(np.nan - np.nan)     # nan
print(np.nan in set([np.nan]))  # True
print(0.3 == 3 * 0.1)      # False
```

#### 26. 以下脚本的输出是什么？ (★☆☆)

```python
print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```


```python
print(sum(range(5), -1))  # 9
from numpy import *
print(sum(range(5), -1))  # 10
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
Z = np.arange(5)
print(Z**Z)        # [  1   1   4  27 256]
print(2 << Z >> 2) # [0 1 2 4 8]
print(Z <- Z)      # [False False False False False]
print(1j*Z)        # [0.+0.j 0.+1.j 0.+2.j 0.+3.j 0.+4.j]
print(Z/1/1)       # [0. 1. 2. 3. 4.]
# print(Z<Z>Z)     # ValueError: The truth value of an array with more than one element is ambiguous.
```

#### 28. 以下表达式的结果是什么？ (★☆☆)

```python
np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)
```


```python
print(np.array(0) / np.array(0))       # nan
print(np.array(0) // np.array(0))      # 0
print(np.array([np.nan]).astype(int).astype(float))  # [-9.22337204e+18]
```

#### 29. 如何将一个浮点数组四舍五入到**远离零**的地方？ (★☆☆)


```python
arr = np.random.uniform(-10, +10, 10)
print(np.copysign(np.ceil(np.abs(arr)), arr))
```

#### 30. 如何找到两个数组之间的公共值？ (★☆☆) 


```python
arr1 = np.random.randint(0, 10, 10)
arr2 = np.random.randint(0, 10, 10)
print(np.intersect1d(arr1, arr2))
```

## 入门题
需要开始频繁的检索官方文档

#### 31. 如何忽略所有 numpy 警告（不推荐）？ (★☆☆) 


```python
defaults = np.seterr(all="

ignore")
arr = np.ones(1) / 0
```

#### 32. 考虑一个大小为 100 的向量，如何在不破坏值的情况下将其打乱？ (★☆☆) 


```python
arr = np.arange(100)
np.random.shuffle(arr)
print(arr)
```

#### 33. 如何从一个数组中提取出所有的整数部分？ (★☆☆) 


```python
arr = np.random.uniform(0, 10, 10)
print(arr.astype(int))
print(np.floor(arr))
print(np.ceil(arr)-1)
print(arr - arr % 1)
print(np.trunc(arr))
```

#### 34. 如何用数据生成器创建一个二维数组？ (★☆☆) 


```python
def generate():
    for x in range(10):
        for y in range(10):
            yield x, y


arr = np.fromiter(generate(), dtype=[('x', int), ('y', int)])
print(arr)
```

#### 35. 如何对数组中的点坐标进行排序？ (★☆☆) 


```python
arr = np.random.random((10, 2))
print(arr)
print(arr[arr[:, 1].argsort()])
```

#### 36. 如何找到一个数组的第 n 大值？ (★★☆) 


```python
arr = np.arange(10000)
np.random.shuffle(arr)
n = 5
print(arr[np.argsort(arr)[-n]])
```

#### 37. 如何在随机 10x2 矩阵中找出距离最远的两个点？ (★★☆) 


```python
import itertools
arr = np.random.random((10, 2))
dist = np.linalg.norm(arr[:, np.newaxis] - arr, axis=2)
print(np.unravel_index(np.argmax(dist), dist.shape))
```

#### 38. 如何将一个 32 位的浮点数组转化为一个对应的整数索引？ (★★☆) 


```python
arr = np.arange(10, dtype=np.float32)
print(arr.astype(np.int32, copy=False))
```

#### 39. 如何按第 n 列对一个二维数组进行排序？ (★★☆) 


```python
arr = np.random.random((5, 5))
print(arr[arr[:, 1].argsort()])
```

#### 40. 如何检查一个二维数组是否有空列？ (★★☆) 


```python
arr = np.random.randint(0, 10, (5, 5))
print((~arr.any(axis=0)).any())
```

#### 41. 从数组的形状中删除大小为 1 的维度 (★★☆) 


```python
arr = np.random.randint(0, 10, (1, 3, 1, 2, 1))
print(np.squeeze(arr))
```

#### 42. 将一维数组转换为行列互换的二维数组 (★★☆) 


```python
arr = np.arange(9)
print(arr.reshape(3, 3).T)
```

#### 43. 如何从一个 10x10 的矩阵中提取出连续的 3x3 块？ (★★★) 


```python
arr = np.random.random((10, 10))
i = 1 + (arr.shape[0] - 3)
j = 1 + (arr.shape[1] - 3)
print(np.lib.stride_tricks.as_strided(arr, shape=(i, j, 3, 3), strides=arr.strides + arr.strides))
```

#### 44. 如何生成一个包含日期的序列？ (★★☆) 


```python
arr = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print(arr)
```

#### 45. 如何计算（连续）数组上任意两个向量之间的欧氏距离？ (★★★) 


```python
arr = np.random.random((10, 2))
print(np.sqrt(np.sum((arr[:, np.newaxis] - arr) ** 2, axis=2)))
```

#### 46. 如何将浮点型向量量化为整型？ (★★☆) 


```python
arr = np.random.random(10)
print((arr * 10).astype(int))
```

#### 47. 如何计算一个数组的第一行和第二行之间的协方差？ (★★☆) 


```python
arr = np.random.random((5, 5))
print(np.cov(arr[0], arr[1]))
```

#### 48. 如何找到数组中出现频率最高的值？ (★★★) 


```python
arr = np.random.randint(0, 10, 50)
print(np.bincount(arr).argmax())
```

#### 49. 找到一个一维数组中第 k 个最大值 (★★★) 


```python
arr = np.random.random(100)
print(np.partition(arr, -5)[-5])
```

#### 50. 找到一个一维数组中出现频率最高的值 (★★★) 


```python
arr = np.random.randint(0, 10, 50)
print(np.bincount(arr).argmax())
```

```markdown
#### 51. 找到一个二维数组的最接近给定值的元素 (★★★)


```python
arr = np.random.random((5, 5))
val = 0.5
print(arr.flat[np.abs(arr - val).argmin()])
```

#### 52. 从一个表示笛卡尔坐标的 10x2 矩阵中选择最近的点对 (★★★)


```python
arr = np.random.random((10, 2))
dist = np.sqrt(np.sum((arr[:, np.newaxis] - arr[np.newaxis, :]) ** 2, axis=2))
dist[np.triu_indices(10, 1)] = np.inf
print(np.unravel_index(np.argmin(dist), dist.shape))
```

#### 53. 将笛卡尔坐标转换为极坐标 (★★★)


```python
arr = np.random.random((10, 2))
x, y = arr[:, 0], arr[:, 1]
r = np.sqrt(x ** 2 + y ** 2)
theta = np.arctan2(y, x)
print(r, theta)
```

#### 54. 打印每个 numpy 标量类型的最小值和最大值 (★★★)


```python
for dtype in [np.int8, np.int32, np.int64]:
    print(np.iinfo(dtype).min)
    print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
    print(np.finfo(dtype).min)
    print(np.finfo(dtype).max)
```

#### 55. 如何打印数组中所有的值？ (★★★)


```python
np.set_printoptions(threshold=np.inf)
arr = np.random.random((10, 10))
print(arr)
```

#### 56. 如何以尽可能小的内存创建一个只读的分类数组？ (★★★)


```python
arr = np.arange(10)
arr.flags.writeable = False
```

#### 57. 创建一个形状为 (100,2) 的随机向量的笛卡尔积笛卡尔坐标 (★★★)


```python
arr = np.random.random((100, 2))
c = np.sqrt(np.sum((arr[:, np.newaxis] - arr[np.newaxis, :]) ** 2, axis=2))
print(c)
```

#### 58. 如何创建一个 record array？ (★★★)


```python
arr = np.core.records.fromarrays([[1, 2, 3], ['a', 'b', 'c'], [1.1, 2.2, 3.3]],
                                 names='a, b, c')
print(arr)
```

#### 59. 创建一个 2D 数组，包含由 a 的每一列的每个元素减去 b 的每个元素的值 (★★★)


```python
arr1 = np.random.random((10, 3))
arr2 = np.random.random(3)
result = arr1 - arr2
print(result)
```

#### 60. 计算一个数组的第 n 个对角线元素的和 (★★★)


```python
arr = np.random.random((5, 5))
n = 1
print(np.trace(arr, offset=n))
```

#### 61. 给定一个矩阵（向量），如何获取不重复的元素并返回它们出现次数？ (★★★)


```python
arr = np.random.randint(0, 10, 50)
vals, counts = np.unique(arr, return_counts=True)
print(vals, counts)
```

#### 62. 如何将一个一维数组转换为具有不重叠块的二维数组？ (★★★)


```python
arr = np.arange(16)
print(arr.reshape(4, 4))
```

#### 63. 生成所有大于 1000 的五位数的 prime 数 (★★★)


```python
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

primes = [x for x in range(10000) if is_prime(x) and x > 1000]
print(primes)
```

#### 64. 从两个数组中找到匹配的行 (★★★)


```python
arr1 = np.random.randint(0, 2, (3, 3))
arr2 = np.random.randint(0, 2, (3, 3))
print(np.where(np.all(arr1 == arr2, axis=1))[0])
```

#### 65. 如何从一个数组中抽取指定列 (★★★)


```python
arr = np.random.random((5, 5))
col = 2
print(arr[:, col])
```

#### 66. 如何将一个数字以科学记数法表示 (★★★)


```python
arr = np.random.random((5, 5))
print(np.format_float_scientific(arr))
```

#### 67. 如何按照多个条件对一个二维数组进行排序 (★★★)


```python
arr = np.random.random((5, 3))
print(arr[np.lexsort((arr[:, 1], arr[:, 0]))])
```

#### 68. 从一个随机 1D 数组中选择连续子数组的索引 (★★★)


```python
arr = np.random.randint(0, 2, 10)
print(np.where(np.convolve(arr, np.ones(3, dtype=int), mode='valid') == 3)[0])
```

#### 69. 如何使用 np.ravel() 展平数组 (★★★)


```python
arr = np.random.random((5, 5))
print(arr.ravel())
```

#### 70. 如何以 numpy 数组作为参数来计算 distance_matrix (★★★)


```python
arr = np.random.random((10, 2))
dist = np.sqrt(np.sum((arr[:, np.newaxis] - arr[np.newaxis, :]) ** 2, axis=2))
print(dist)
```

#### 71. 如何找到两个数组的交集？ (★★★)


```python
arr1 = np.random.randint(0, 10, 10)
arr2 = np.random.randint(0, 10, 10)
print(np.intersect1d(arr1, arr2))
```

#### 72. 如何检查一个二维数组是否有空行？ (★★★)


```python
arr = np.random.randint(0, 10, (5, 5))
print((~arr.any(axis=1)).any())
```

#### 73. 如何在一个数组中选择指定的子数组？ (★★★)


```python
arr = np.random.random((5, 5))
rows = [0, 2, 3]
print(arr[rows])
```

#### 74. 如何用 numpy 创建一个带自定义步长的数组？ (★★★)


```python
arr = np.arange(0, 1, 0.1)
print(arr)
```

#### 75. 如何使用 numpy 创建一个带有重复值的数组？ (★★★)


```python
arr = np.array([1, 2, 3])
print(np.repeat(arr, 3))
```

#### 76. 如何检查一个数组是否包含特定值？ (★★★)


```python
arr = np.random.randint(0, 10, 10)
val = 5
print(np.isin(val, arr))
```

#### 77. 如何删除一个数组中的指定列？ (★★★)


```python
arr = np.random.random((5, 5))
col = 2
print(np.delete(arr, col, axis=1))
```

#### 78. 如何计算一个数组中每行的平均值？ (★★★)


```python
arr = np.random.random((5, 5))
print(np.mean(arr, axis=1))
```

#### 79. 如何将多个一维数组堆叠成一个二维数组？ (★★★)


```python
arr1 = np.random.random(5)
arr2 = np.random.random(5)
print(np.vstack((arr1, arr2)))
```

#### 80. 如何将一个二维数组展开成一个一维数组？ (★★★)


```python
arr = np.random.random((5, 5))
print(arr.flatten())
```

#### 81. 如何检查一个数组中是否包含某些特定值？ (★★★)


```python
arr = np.random.randint(0, 10, 10)
vals = [2, 5]
print(np.isin(arr, vals))
```

#### 82. 如何用 numpy 创建一个随机数矩阵？ (★★★)


```python
arr = np.random.random((5, 5))
print(arr)
```

#### 83. 如何找到一个数组的前 n 个最大值？ (★★★)


```python
arr = np.random.random(10)
n = 3
print(arr[np.argsort(arr)[-n:]])
```

#### 84. 如何对一个数组中的所有元素求和？ (★★★)


```python
arr = np.random.random((5, 5))
print(np.sum(arr))
```

#### 85. 如何找到一个数组中的最小值？ (★★★)


```python
arr = np.random.random((5, 5))
print(np.min(arr))
```

#### 86. 如何将一个数组中的元素乘以一个标量值？ (★★★)


```python
arr = np.random

.random((5, 5))
scalar = 2
print(arr * scalar)
```

#### 87. 如何找到一个数组的中位数？ (★★★)


```python
arr = np.random.random(10)
print(np.median(arr))
```

#### 88. 如何找到一个数组的标准差？ (★★★)


```python
arr = np.random.random(10)
print(np.std(arr))
```

#### 89. 如何对一个数组的所有元素求和？ (★★★)


```python
arr = np.random.random(10)
print(np.sum(arr))
```

#### 90. 如何将一个数组的所有元素乘以一个标量值？ (★★★)


```python
arr = np.random.random(10)
scalar = 2
print(arr * scalar)
```

#### 91. 如何找到一个数组的最大值？ (★★★)


```python
arr = np.random.random(10)
print(np.max(arr))
```

#### 92. 如何找到一个数组的最小值？ (★★★)


```python
arr = np.random.random(10)
print(np.min(arr))
```

#### 93. 如何计算一个数组的平均值？ (★★★)


```python
arr = np.random.random(10)
print(np.mean(arr))
```

#### 94. 如何计算一个数组的方差？ (★★★)


```python
arr = np.random.random(10)
print(np.var(arr))
```

#### 95. 如何找到一个数组中的唯一值？ (★★★)


```python
arr = np.random.randint(0, 10, 10)
print(np.unique(arr))
```

#### 96. 如何找到一个数组中的重复值？ (★★★)


```python
arr = np.random.randint(0, 10, 10)
unique, counts = np.unique(arr, return_counts=True)
print(unique[counts > 1])
```

#### 97. 如何对一个数组中的元素进行排序？ (★★★)


```python
arr = np.random.random(10)
print(np.sort(arr))
```

#### 98. 如何对一个数组进行切片？ (★★★)


```python
arr = np.random.random((5, 5))
print(arr[1:3, 2:4])
```

#### 99. 如何翻转一个数组？ (★★★)


```python
arr = np.random.random(10)
print(arr[::-1])
```

#### 100. 如何对一个数组进行转置？ (★★★)


```python
arr = np.random.random((5, 5))
print(arr.T)
```
