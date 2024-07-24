## 介绍
- 学习使用int 8，float16，bfloat16（brain flot16） 来压缩模型
- 学习压缩算法，存储一个32位数
## Class 1：处理大型模型
随着模型平均大小的增加，平均参数数量已达到了700亿的数量级，这是一个非常庞大的数据。然而像t4这样的低端计算卡，也只有16g的 RAM（运行内存）。所以，运行这些先进的模型仍然是一种挑战。这个时候，就要求在不需要访问高内存的情况下有效的运行这些模型。所以，社区现在面临的主要挑战是让这些模型可以通过压缩模型来访问。

而目前，最先进的模型压缩方法（如pruning and knowledge distillation)花费更多的时间去做量化。 
	1.pruning（修剪）：pruning只是包括删除模型中的层级，对模型的决策没有很重大的影响。他只是删除一些在计算过程中权重不大的某些指标的层级。
	2.knowledge distillation（知识蒸馏）：使用原模型（instructor）训练一个更小的模型（student）。知识蒸馏除了会难以避免地丢失一些信息，最主要的难题是需要确保你有足够的计算来拟合原始模型并且从中获取预测，以便可以再计算损失的时候将他们发送到原模型。但是如果你要提炼非常大的模型，那么他的成本将会指数级上涨😱。
### 量化 
量化是以较低精度表示模型权重，例如下图的小矩阵，里面存储着一个小模型的一些参数。由于矩阵以float32格式存储（这是大多数模型的默认存储数据类型）它为每个参数分配四个字节，8bit的四倍精读，因此如下矩阵占用 36 bytes。
![alt text](/image/QQ_1721389158056.png)
如果以八位精读量化权重矩阵，那么每个参数将只得到 1 byte。因此，我们只需要9 bytes就可以存储整个权重矩阵。
![alt text](/image/QQ_1721389582129.png)
然而，这是有代价的，那就是量化误差（quantization error）。最先进的量化方法背后的主要挑战是降低这种误差，尽可能避免任何性能下降。

## Class 2：数据类型和大小
### 1.unsigned integer:
用于表示正整数，n位无符号整数的范围是0到2ⁿ-1，例如八位无符号整数的最小值为0，最大值为255。而计算机分配一个八个比特的空间来存储这个八位整数，如下图所示
![alt text](/image/QQ_1721399716867.png)
对于它的表示方法这里就不多做赘述了。这里还要讲讲有符号整数，有符号整数用来表示负整数或正整数，他有多种表现形式，但由于2‘s补码（2’s Complement Code）的常见性，我们今天研究的只有2‘s补码。它的范围是-2^n-1^次幂到2^n-1^-1。因此，对于8位有符号整数，最小值为-128，最大值为127。与无符号整数的区别在于最后一个位置的数字。正如你在下图看到的
![alt text](/image/QQ_1721403751862.png)
这个值是负数。因此，如果我们处理与前面相同的序列，我们需要在数字前添加一个负号。结果为-（128+1）=-129。

### 2.integer-pytorch
在pytorch中创建具有整数数据类型的数据并非难事，只需要正确设置``torch.dtype``。如下表所示，要创建一个8位有符号整数，只需要设置``torch.int8``作为``torrch.dtype``
![alt text](/image/QQ_1721404514237.png)
为此，我们将使用``torch.iinfo``，代码如下所示。
```python
#torch.uint8
torch.iinfo(torch.uint8)
iinfo(min=0, max=255, dtype=uint8)
```
以下是jupyter notebook代码，用来本地运行并测试代码
```python
!pip install pytorch==2.1.1

##使用torch库
import torch

##8位无符号整数的信息
torch.iinfo(torch.uint8)
iinfo(min=0, max=255, dtype=uint8)

##8位有符号整数的信息
torch.iinfo(torch.int8)
iinfo(min=-128, max=127, dtype=int8)

##下面就该轮到你们自己完成了！

##64位有符号整数的信息


##32位有符号整数的信息


##16位有符号整数的信息


```
### 3.floating
讨论完了整数，接下来赶到战场的是浮点数。浮点数由三个部分组成：
- 符号：正/负（始终为1位）
- 指数（范围）：影响数值的可表示范围
- 小数部分（精度）：影响数值的精度

FP32、BF16、FP16、FP8 是具有特定数量的指数和小数部分位数的浮点格式。
例如FP32
```
FP32
符号：1位
指数（范围）：8位
小数部分（精度）：23位
总计：32位
```
下图展示了FP32可以存储多大和多小的数字
![alt text](/image/QQ_1721407068321.png)
这类数据类型在机器学习中非常重要，因为大多数模型将他的权重存储在FP32中。
FP16 (Half-Precision Floating-Point) 的表示方式与整数和分数部分的位数有些不同。FP16 使用 1 位符号位、5 位指数位和 10 位尾数位。具体的位数分配如下：

- 符号位 (Sign Bit, S)：1 位
- 指数位 (Exponent Bits, E)：5 位
- 尾数位 (Mantissa Bits, M)：10 位

根据 IEEE 754 标准，FP16 的表示范围和精度如下：

- 指数范围是 -14 到 15（经过偏移量为 15 的偏移后）。
- 尾数部分是 10 位，有效尾数是 1 + 尾数部分。
- 最小的非零正数：2^(-14) * 2^(-10) ≈ 6.10 x 10^-5（次正规数）。
- 最大值：2^(15) * (2 - 2^-10) ≈ 65504。

![alt text](/image/QQ_1721785559240.png)
而bfloat16 的位分配如下：

- 符号位 (Sign Bit, S)：1 位
- 指数位 (Exponent Bits, E)：8 位
- 尾数位 (Mantissa Bits, M)：7 位

由于它的指数位与 FP32 的指数位相同，因此它们的指数范围相同。这意味着 bfloat16 的指数范围与 FP32 一样，从 -126 到 +127（偏移量为 127 的偏移后）。

根据 IEEE 754 标准，bfloat16 的表示范围和精度如下：

- 指数范围是 -126 到 127（经过偏移量为 127 的偏移后）。
- 尾数部分是 7 位，有效尾数是 1 + 尾数部分。
- 最小的非零正数：2^(-126) ≈ 1.18 x 10^-38（次正规数）。
- 最大值：2^(127) * (2 - 2^-7) ≈ 3.39 x 10^38。

以下是浮点数在pytorch中的类型表：
![alt text](/image/QQ_1721785604190.png)
要在 PyTorch 中创建一个 16 位的brain浮点数，只需将 `dtype` 设置为 `torch.bfloat16`。

现在，让我们来看看将一个python值转化为具有特定类型的pytorch张量时会发生什么
```python
# 默认情况下，Python 使用 FP64 存储浮点数数据
value = 1/3

# 使用 format 函数将浮点数打印出后 60 位小数点
format(value, '.60f')
#大概率打出来的是0.33333333333333+后面的近似值

#现在，让我们创建一个张量，类型为torch.bfloat16
tensor_bf16 = 
torch.tenser(value, dtype=torch.bfloat16)
#打印它，你将会得到相同的答案
print(f"fp64 tensor: {format(tensor_fp64.item(), '.60f')}")

#以下就自己跑跑吧😋

# 32 位浮点数
tensor_fp32 = torch.tensor(value, dtype=torch.float32)
print(f"fp32 tensor: {format(tensor_fp32.item(), '.60f')}")

# 16 位浮点数
tensor_fp16 = torch.tensor(value, dtype=torch.float16)
print(f"fp16 tensor: {format(tensor_fp16.item(), '.60f')}")

# 16 位brain浮点数
tensor_bf16 = torch.tensor(value, dtype=torch.bfloat16)
print(f"bf16 tensor: {format(tensor_bf16.item(), '.60f')}")
```
到这里我们应该可以得出一个结论
**数据有的位数越少，近似值的精确度就越低，而对于bfloat16，正如我们之前看到的，精度比FP16差，但是bfloat16的表示范围大于FP16。**
可以通过`torch.finfo`函数来检验这一点
```python
# 获取 16 位brain浮点数的信息
torch.finfo(torch.bfloat16)
'''
返回：finfo(resolution=0.0078125, 
min=-3.3895313892515355e+38, 
max=3.3895313892515355e+38, 
eps=0.0078125, 
tiny=1.1754943508222875e-38)
'''



# 获取 32 位浮点数的信息
torch.finfo(torch.float32)

# 获取 16 位浮点数的信息
torch.finfo(torch.float16)

# 获取 64 位浮点数的信息
torch.finfo(torch.float64)
```

### 4.Downcasting(向下转换)
向下转换 (Downcasting) 是指将数据从一种更高精度的数值类型转换为一种更低精度的数值类型。
例如，将 64 位浮点数 (float64) 转换为 32 位浮点数 (float32)，或将 32 位浮点数 (float32) 转换为 16 位浮点数 (float16) 或 16 位brain浮点数 (bfloat16)。
向下转换在深度学习中尤为重要，因为它可以显著减少模型的内存占用和计算时间，但也会带来精度损失。