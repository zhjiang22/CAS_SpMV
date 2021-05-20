# SpMV-赛题描述

# 一、背景简介

数值线性代数是科学计算中的基础计算模块，求解线性方程组、线性最小二乘问题、特征值和奇异值等计算是科学计算中计算强度最大的部分。随着数值编程的出现，使用复杂的子程序库来解决此类问题非常有效。我们在编写涉及线性代数操作的程序代码时，通常将计算分解为点乘或者矩阵向量乘等基础子程序。由此，结构化编程应运而生，指定基本构建模块并使用独特助记符名称标识这些操作，并且为了提高这些代数程序的算法使用效率，对其中一些基本操作的名称和参数列表进行统一规划，这些统一规划的接口可以封装为数学库供开发者或者研究人员使用。

目前每个开发语言基本上都有其对应的线性代数库，常见的C++线性代数库如MKL、Eigen、BLAS、LAPACK在科学计算中使用广泛，而在python中可以使用SciPy进行此类计算。随着技术不断发展，人们又利用多线程技术或多进程技术来加速此类计算的速度。其中利用异构编程技术来进行加速是当下值得深入研究的一个方向。

---

# 二、赛题描述

SpMV是上述线性代数库中比较基本的操作(详见附录)，所以其使用频率较高，例如在有限元分析中往往会将问题转化为矩阵运算，而有限元分析在力学分析，计算电磁学等领域都有广泛应用。本次比赛，采用异构编程加速SpMV，重点关注**双精度**实数在异构计算平台上的计算性能。

目前已经为选手规范了SpMV的实现接口，为方便各位选手更好的进行优化工作，现将函数及其参数解释整理如下：

```cpp
void sparse_spmv( int         htrans,

             const int     halpha,

             const int    hbeta,

             int         hm,

             int         hn,

             const int*    hrowptr,

             const int*    hcolindex,

             const double*  hvalue,

             const double*  hx,

             double*      hy

)

enum sparse_operation {operation_none=0,operation_transpose=1};
```

### 1)输入参数

- 参数`htrans` : sparse_operation类型。 详细说明了矩阵乘法中使用的 op(A)的格式(是否转置)。

  如果 `trans_a` = operation_none ,则 op(A) = A；

  如果 `trans_a` = operation_transpose, 则 op(A) = A'；

- 参数 `halpha` : 表示矩阵A的系数，halpha>0；

- 参数 `hbeta` : 表示向量Y的系数，hbeta>0；

- 参数 `hm` : 表示矩阵A的行数,hm>0；

- 参数 `hn` : 表示矩阵A的列数,hn>0；

- 参数 `hrowptr` : csr格式中每一行首元素在value中的位置；

- 参数 `hcolindex` : csr稀疏格式中每个元素的的列序号；

- 参数 `hvalue` : csr稀疏格式中非零元素的值；

- 参数 `hx` : 稠密向量；

- 参数 `hy` : 向量y的初始值；

### 2)输出参数

- 参数`hy` : 覆盖输入的hy向量,结果写回到hy。

为充分发挥异构计算平台在此计算方式中的性能优势，需对函数实现进行进一步优化。

### 3)比赛要求

在比赛过程中遵循以下**要求**:

1. API实现完整，所有数据集下结果正确。

2. 选手在给定接口函数的基础上进行矩阵乘法性能优化，给定测试代码中，API接口函数不可更改，其中涉及的非固定参数选手可根据参与计算的矩阵大小自行调优。

3. 稀疏矩阵存储格式统一采用CSR格式，关于CSR格式的解释可以**参考附录。**

4. 选手可以参考给出的demo程序(下载方式参考交卷表)进行开发，程序包含了预热、验证以及测时部分。自行编写的程序也要有这三个部分,提交时请附上程序的编译方式和算法文档。

5. 不要修改main.cpp和common_function.hpp，其余源文件可以自定义，保证程序运行即可。

6. 选手每天可以提交**2次**程序，每周最多提交**12次**。

---

# 三、评分细则

### 1)测试方式:

测试样例部分，代码中给出以伪随机数生成稀疏矩阵作为测试数据，验证算法的性能。具体的测试算例要求如下。

本次比赛采用suitsparse稀疏矩阵测试集合，所选稀疏矩阵来自于不同应用领域，具体描述如下。

<img width="1254" alt="image" src="https://user-images.githubusercontent.com/46041980/119002948-e661e700-b9bf-11eb-8676-901bdd3a4707.png">


![SpMV-%E8%B5%9B%E9%A2%98%E6%8F%8F%E8%BF%B0%20d9c16db21c50402b827a676b8e481730/Untitled.png](SpMV-%E8%B5%9B%E9%A2%98%E6%8F%8F%E8%BF%B0%20d9c16db21c50402b827a676b8e481730/Untitled.png)
数据集位于`demo.zip`同级目录下，以`data.zip`命名，编译方式与运行方式详见demo中的`README.md`文件，提交程序时不必提交`data.zip`。

### 2)评分方式:

**初赛**时采用单size进行评比，每个程序会跑三次，取三次的平均时间作为程序的最终时间，使用时间越少排名越靠前，初赛月度排名使用的数据集来自于上述数据集中的一个或者多个。

**复赛**评选中会对10组数据进行两轮测试，每一组数据都有运行时间，在每轮测试中记录10组数据的运行总时间，对两轮测试的总时间取平均值作为选手的成绩，时间越少成绩越高，最后对所有选手成绩进行归一化，得出百分制得分。总成绩按照下方公式加权计算，其中timesocre为选手的程序运行时间得分，docscore为文档得分。

totalscore = (0.8×timescore + 0.2×docscore)×100

**复赛文档评分细则**:

1. 文档的完整性，应包含编译方式，运行方式，算法介绍，算法创新点，代码设计结构五个部分介绍，请参考交卷表填写(在官网的交卷页面处选择题目后可下载交卷表)。(30分)

2. 算法实现介绍要能与文档中代码设计结构对应。(30分)

3. 文档通顺可读性高，如果引用了某些文献的算法请给出文献的名称。(30分)

4. 目录完整。(10分)

### **3)特别说明:**

在初赛和复赛中如果出现以下情况**成绩无效**:

1. 程序无法编译。

2. 程序5分钟内无法结束。

3. 无法通过正确性测试。

4. 出现抄袭现有开源库实现的情况。

5. 选手之间出现代码抄袭或者文档抄袭的情况。

---

---

---

## 附录:

SpMV公式描述如下:

$Y=α×A×X+β×Y (1)$

其中， A表示稀疏矩阵其大小为m*k，X表示稠密向量大小为k，Y是密度向量大小为m，a和B是标量。SpMV算法优势就是在于矩阵A采用了稀疏存储格式，稀疏存储在超大规模求解过程中可以节约大量的内存，减少无效运算次数，从而提高计算速度。常用的稀疏存储格式有CSR、COO、CSC、BSR等，下面以CSR存储方式对比说明稀疏存储。

![https://cas-pra.sugon.com/ueditor/jsp/upload/image/20210409/1617947320812068697.png](https://cas-pra.sugon.com/ueditor/jsp/upload/image/20210409/1617947320812068697.png)

普通存储方式就是将一个矩阵变为一个数组进行存储，这里采用行主模式举例，上述矩阵为Value=[1 4 0 0 0 2 3 0 5 0 7 8 0 0 9 6]。

稀疏存储的目的只存储矩阵当中的非零值，这样可以节约存储空间，同时0与任何数进行运算结果都为0，这样可以减少无效的数值计算。CSR稀疏存储要开辟三个数组来描述这个矩阵，value数组代表矩阵元素的非0值，csrColPtrA描述了非零元素的列序号，csrRowIndA描述了每行首个非零元素在value中的索引序号, csrRowIndA最后一个值代表非零元素的个数。上述矩阵采用csr描述如下:

- Value=[1 4 2 3 5 7 8 9 6]
- csrColPtrA=[0 1 1 2 0 2 3 2 3]
- csrRowIndA =[0 2 4 7 9]

在csrRowIndA数组中，每两个元素的差值正好为该行元素的个数，通过csrColPtrA描述的非零元素的列索引，可以得出需要相乘的稠密向量的行索引，从而进行对位相乘。更多的关于稀疏矩阵的信息可以查询网络资料或者文献。
