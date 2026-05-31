# PKU 计算机视觉导论 · 2026 春季作业

北京大学《计算机视觉导论》（2026 春季学期）编程作业仓库，包含四次循序渐进的实验，从经典图像处理到深度学习与三维视觉。

## 仓库结构

```
PKU-ICV-2026Spring-Assignment/
├── assignment-01/          # 作业 1：经典图像处理
├── assignment-02/          # 作业 2：神经网络基础
├── assignment-03/          # 作业 3：三维视觉与 NeRF
├── assignment-04/          # 作业 4：点云、序列模型与注意力
├── .gitignore
└── README.md
```

各次作业的详细说明、环境配置与提交要求见对应目录下的 README 或 PDF 说明文件。

| 作业 | 主题 | 主要技术栈 | 详细文档 |
|------|------|-----------|----------|
| [assignment-01](assignment-01/) | 卷积、边缘检测、角点、RANSAC | NumPy | — |
| [assignment-02](assignment-02/) | 反向传播、BatchNorm、CIFAR-10 分类 | NumPy, PyTorch | [README](assignment-02/README.md) |
| [assignment-03](assignment-03/) | 点云、NeRF 渲染、Mask R-CNN | NumPy, PyTorch, Jupyter | [README](assignment-03/README.md) · [PDF](assignment-03/03_assignment.pdf) |
| [assignment-04](assignment-04/) | PointNet、RNN、Self-Attention 图像描述 | PyTorch | [README](assignment-04/README.md) · [PDF](assignment-04/04_assignment.pdf) |

## 作业概览

### 作业 1 · 经典图像处理

实现图像处理的基础算法，全部基于 NumPy，无需深度学习框架。

- **卷积与滤波**（`HM1_Convolve.py`）：零填充 / 复制填充、Toeplitz 矩阵卷积、高斯滤波、Sobel 算子
- **Canny 边缘检测**（`HM1_Canny.py`）：梯度计算、非极大值抑制、双阈值与滞后连接
- **Harris 角点检测**（`HM1_HarrisCorner.py`）：结构张量与角点响应
- **RANSAC 平面拟合**（`HM1_RANSAC.py`）：三维点云中的鲁棒平面估计

### 作业 2 · 神经网络基础

从手写反向传播到 PyTorch 训练完整 CNN。

- **反向传播**（`BP.py`）：在 MNIST 子集上实现两层 MLP 的前向与反向传播
- **Batch Normalization**（`batch_normalization/bn.py`）：前向、反向及推理模式
- **CIFAR-10 分类**（`cifar-10/`）：数据增强、网络结构设计、训练与 TensorBoard 可视化

### 作业 3 · 三维视觉与 NeRF

涵盖点云、体渲染与目标检测，部分题目以 Jupyter Notebook 形式提供。

- **depth_pc**：由深度图反投影生成点云
- **mesh_pc**：从三角网格采样点云，计算 Chamfer Distance
- **mini_nerf**：实现 NeRF 的射线生成、采样与体渲染（无需训练）
- **MaskRCNN**：完成数据集封装，在小型数据集上训练实例分割模型

### 作业 4 · 点云、序列模型与注意力

面向更复杂的深度学习架构与三维数据。

- **PointNet**（`PointNet/`）：点云分类与分割，基于 ShapeNetPart 数据集
- **RNN**（`RNN/`）：实现 RNN 层与图像描述模型，在 COCO 特征上过拟合演示
- **Self-Attention**（`Attention/`）：因果自注意力与 Transformer 图像描述，含数值校验脚本
- **MaskRCNN**（`MaskRCNN/`）：目标检测 / 实例分割（自作业 3 延续）

## 环境配置

建议为每次作业创建独立的 Conda 环境，或按难度递进复用环境：

| 作业 | Python | 核心依赖 |
|------|--------|----------|
| 1 | ≥ 3.7 | NumPy, OpenCV |
| 2 | ≥ 3.7 | NumPy, PyTorch, OpenCV, TensorBoard |
| 3 | ≥ 3.7 | NumPy, Matplotlib, PyTorch, Trimesh, Jupyter |
| 4 | ≥ 3.7 | PyTorch, OpenCV, Matplotlib, h5py, imageio |

国内用户可使用 [清华大学开源软件镜像站](https://mirrors.tuna.tsinghua.edu.cn/) 加速 pip / conda 安装，具体命令见 [assignment-02/README.md](assignment-02/README.md) 与 [assignment-04/README.md](assignment-04/README.md)。

部分作业需额外下载数据集（如 CIFAR-10、ShapeNetPart、COCO 图像描述特征），下载链接与路径配置说明见各作业 README。
