# MiniMind

MiniMind 是一个基于 [MNN](https://github.com/alibaba/MNN) 扩展的 C++ 公共函数库，主要补充游戏 AI 和游戏状态建模中常用的隐藏层与损失函数。

项目使用 MNN Express 构建计算图，并接入 MNN Train 的自动微分能力。各层采用 MNN 原生 Tensor 布局，省去旧版 MiniBrain 中针对 Eigen 扁平矩阵的数据拆分和重组流程。

## 已实现功能

### 损失函数

- `mse`：对全部元素计算平方误差平均值，与 PyTorch `reduction="mean"` 的行为一致。
- `cross_entropy_multi`：接收概率和同形状多标签 target，计算标签加权交叉熵并对 batch 求平均。

### Attention

单头缩放点积自注意力层。

```text
输入： [batch, entity, feature]
输出： [batch, entity, feature]
```

### GRU

无状态 GRU Cell。调用方显式传入上一隐藏状态，并保存返回的下一隐藏状态。

```text
输入：
  x      [batch, input_size]
  hidden [batch, hidden_size]

输出：
  next_hidden [batch, hidden_size]
```

### StatePooling

沿 entity 维分别执行平均池化和最大池化，再拼接结果。

```text
输入： [batch, entity, feature]
输出： [batch, 2 * feature]
```

### Embedding

对每个实体共享同一组线性投影参数。该层处理连续浮点特征，不是整数索引查表。

```text
输入： [batch, entity, input_feature]
输出： [batch, entity, output_feature]
```

## 项目结构

```text
MiniMind/
├─ src/
│  ├─ 3rd_party/MNN/       MNN 依赖
│  ├─ attention.hpp/.cpp
│  ├─ embedding.hpp/.cpp
│  ├─ gru.hpp/.cpp
│  ├─ loss.hpp/.cpp
│  └─ state_pooling.hpp/.cpp
├─ tests/                  单元测试
├─ CMakeLists.txt
└─ build_tests.bat
```

## 构建要求

- Windows
- CMake 3.20 或更高版本
- LLVM/Clang 工具链
- MinGW Makefiles

示例：

```bat
build_tests.bat C:\Files\programs\llvm-mingw-ucrt-x86_64\bin 8
```

第二个参数是并行构建任务数。

## 测试

构建脚本会编译库和全部测试，并通过 CTest 运行：

```text
minimind.loss
minimind.attention
minimind.gru
minimind.state_pooling
minimind.embedding
```

## 使用方式

在 CMake 项目中加入 MiniMind，并链接 `MiniMind` 目标：

```cmake
add_subdirectory(PATH_TO_MINIMIND)
target_link_libraries(YOUR_TARGET PRIVATE MiniMind)
```

业务代码按需包含对应头文件：

```cpp
#include "attention.hpp"
#include "embedding.hpp"
#include "gru.hpp"
#include "loss.hpp"
#include "state_pooling.hpp"
```

MiniMind 定位为游戏项目可复用的基础神经网络组件库，不负责具体游戏状态采集、实体排序、隐藏状态生命周期或完整网络结构；这些逻辑由上层游戏模块管理。
