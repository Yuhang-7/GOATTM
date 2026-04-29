# GOATTM 高层介绍

## GOATTM 是什么

`GOATTM` 是一个面向降阶建模的库，用来训练二次型 latent dynamics，并让模型去拟合
quantity of interest（QoI）时间序列观测。

它把下面几类能力放在了一起：

- 结构化的 latent dynamics 模型
- 从 latent state 到观测量的二次型 decoder
- 精确的离散导数与伴随计算
- decoder best-response 降维优化
- 支持 MPI 的训练与评估流程

## 它擅长什么

`GOATTM` 适合下面这类任务：

- 从时变数据里拟合 latent reduced model
- 保持 dynamics 参数化有结构、可解释
- 直接针对 QoI 轨迹训练，而不只做全状态重构
- 使用精确的离散梯度和 Hessian action
- 按 sample 维度用 MPI 扩展数据评估

## 主要组成

整个库大致分成几层：

- `core`: 代数与参数化工具
- `models`: latent dynamics 和 decoder 对象
- `solvers`: `implicit_midpoint`、`explicit_euler`、`rk4` 的 rollout 支持
- `losses`: QoI loss 和离散 adjoint
- `problems`: decoder best-response 与 reduced objective 组装
- `preprocess`: normalization 和 `OpInf` 初始化
- `train`: optimizer、checkpoint、metrics 和 run logging

## 实际 workflow

一个典型流程是：

1. 读取 `.npz` samples 并构建 manifest
2. 划分 train/test
3. 可选地对数据做 normalization
4. 可选地运行 `OpInf` 初始化
5. 用 decoder best-response 的 reduced training workflow 训练模型
6. 查看 metrics、checkpoints 和 timing summaries

## 当前状态

这个代码库现在已经不只是“研究脚本集合”，而是具有明确 library 形态的实现。

它目前最强的部分包括：

- 结构化二次型 reduced dynamics
- reduced best-response 训练组织
- 多种积分器上的精确离散导数支持
- 带 provenance 的 preprocessing 和训练输出
- MPI-aware 的数据集评估

## 一个重要提醒

高层 workflow 已经比早期说明里写得完整得多，但部分底层 helper 仍然带有更窄的假设。
比如某些内部组件仍然保留 midpoint 风格的特定实现，即使公共训练接口已经支持多种积分器。

## 一句话介绍

`GOATTM` 是一个面向 quadratic latent dynamics 的降阶建模与 reduced-QoI 训练库，提供结构化参数化、多种时间积分器、精确离散导数、decoder best-response 优化、数据预处理以及支持 MPI 的训练 workflow。
