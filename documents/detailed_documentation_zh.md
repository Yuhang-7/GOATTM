# GOATTM 详细文档

## 1. 文档范围

这份文档面向当前 `GOATTM` 代码库，重点说明：

- `src/goattm` 里已经实现了哪些能力
- 这些模块之间如何配合
- 当前实现有哪些重要假设和限制

它的目标不是写一份营销材料，而是给使用者和后续开发者一份和代码现状一致的说明。

## 2. 概念模型

这个库围绕 reduced latent state `u(t)` 工作，并同时训练两部分：

- 一个推进 `u(t)` 的 latent dynamics model
- 一个把 latent state 映射到 QoI 的 decoder

训练目标并不是简单地把 dynamics 和 decoder 参数一次性全都联合优化。
相反，库内部采用的是 reduced best-response 的组织方式：

- dynamics 参数是外层变量
- decoder 参数被视为内层最小二乘 best response

这使得它非常适合 GOAM 风格的 reduced training pipeline。

## 3. 公共包结构

### `goattm.core`

这是底层数学工具层，负责：

- 构造 quadratic features
- 在显式和压缩 quadratic 表达之间转换
- 处理 `mu_h` 参数化
- 从 `S` 和 `W` 参数构造 stabilized linear operator
- 把显式梯度 pull back 到结构化参数坐标

这一层本身不负责训练 workflow，但它是其他模块的代数基础。

### `goattm.models`

模型层定义了主要对象：

- `QuadraticDynamics`
- `StabilizedQuadraticDynamics`
- `QuadraticDecoder`

`QuadraticDynamics` 表示 latent state 的状态方程，包含线性项、二次项、输入项和常数项。
`StabilizedQuadraticDynamics` 用结构化稳定参数保存线性块，但依然向外暴露显式算子。
`QuadraticDecoder` 用线性项、二次项和仿射项把 latent state 映射成 QoI。

### `goattm.data`

这一层定义了 `.npz` 数据契约。

重要对象包括：

- `NpzQoiSample`
- `NpzSampleManifest`
- `NpzTrainTestSplit`

当前 sample 结构包含：

- `sample_id`
- `observation_times`
- latent 初值 `u0`
- `qoi_observations`
- 可选的 `input_times`
- 可选的 `input_values`
- 可选 metadata

此外还支持：

- manifest 保存和读取
- 基于 seed 的可复现实验划分
- 基于 sample id 的显式 train/test 划分
- cubic-spline 输入插值
- piecewise-linear 输入插值

这是目前代码里比较干净、可复用的一层。

### `goattm.solvers`

这一层负责 rollout、tangent propagation 和积分器分发。

当前公共接口支持：

- `implicit_midpoint`
- `explicit_euler`
- `rk4`

这点很重要，因为一些旧说明仍然容易让人误以为 `GOATTM` 主要是 midpoint 代码，
然后附带一个 explicit Euler 分支。现在这种说法已经不准确了。
`rk4` 已经进入公共积分接口，并且是 `ReducedQoiTrainerConfig` 的默认值。

积分器支持的内容不只是 forward rollout，还包括：

- discrete adjoint
- incremental discrete adjoint
- 参数梯度累积
- Hessian-action 项累积

### `goattm.losses`

这一层负责 observation-aligned 的 QoI objective 和一阶导数。

包括：

- trapezoidal quadrature weights
- decoder partial assembly
- state-space residual derivative
- 支持积分器的精确 discrete adjoint
- 精确的一阶参数梯度组装

这里的关键事实是：这些导数是对离散时间格式本身做的精确离散导数，
而不是“先写连续伴随，再事后离散”。

### `goattm.problems`

这是把局部数值核组织成优化问题的 workflow 层。

主要能力包括：

- 数据集级的 QoI loss/gradient 评估
- decoder normal equation 的组装与求解
- reduced objective preparation
- reduced gradient evaluation
- reduced Hessian-action evaluation
- 通过 repeated actions 显式拼出 reduced Hessian

这一层让整个项目更像一个完整的优化库，而不只是数值函数拼装。

但这里有一个需要诚实说明的地方：

- 高层 reduced-objective evaluator 接受公共时间积分器选项
- 但不是所有底层 helper 都同样通用

例如 `decoder_normal_equation.py` 目前仍然是基于 midpoint rollout 来组装 decoder normal equation。
这并不否定整体的多积分器 workflow，但说明“所有内部实现都已经完全 integrator-agnostic”
这种说法会过头。

### `goattm.preprocess`

这一层负责训练前的数据准备和初始化。

当前能力包括：

- 基于训练集统计量的 QoI normalization
- 可选的输入 normalization
- 归一化 train/test 数据集的 materialization
- energy-preserving quadratic fitting 的 constrained least-squares 工具
- 基于 `OpInf` 的 stabilized latent model 初始化
- 回归前的 latent embedding 构造
- 用 forward rollout 验证初始化模型

`OpInf` 初始化路径已经相当完整，不只是做一个回归：

- 可以先归一化数据
- 可以生成 latent dataset
- 可以拟合结构化 dynamics model
- 可以做 rollout 验证
- 可以保存 artifacts 和 diagnostics

### `goattm.train`

这一层提供训练接口和 run 管理。

主要组件有：

- `ReducedQoiTrainer`
- `ReducedQoiTrainerConfig`
- Adam、gradient descent、L-BFGS、Newton-action 的更新器配置
- metrics 与 summary logging
- checkpointing
- timing instrumentation
- output-directory provenance

训练器围绕 reduced best-response objective 运作，并记录完整的 run 上下文，
方便后续复现实验。

## 4. 端到端 workflow

目前推荐的工作流大致是：

1. 准备 `.npz` samples 和 manifest
2. 用 seed 或显式 ids 做 train/test split
3. 可选地对 train/test 数据做 normalization
4. 可选地运行 `OpInf` 初始化，生成 latent dataset 和初始模型
5. 构建 reduced training workflow
6. 在外层优化 dynamics 参数，同时反复求解 decoder 内层问题
7. 查看 run logs、checkpoints、metrics 和 timing summaries

这比早期把 preprocessing、initialization 和 optimization 分散在外部脚本里的方式清晰得多。

## 5. Reduced best-response 形式

核心训练对象可以写成：

`J(mu_g) = J(mu_f*(mu_g), mu_g)`

其中：

- `mu_g` 表示外层 dynamics 参数
- `mu_f*(mu_g)` 表示在该 dynamics 下 decoder 的 best response

一次 reduced objective 评估通常需要：

1. 在训练集上做分布式 forward rollout
2. 组装 decoder normal equation
3. 求解 decoder
4. 评估 QoI loss
5. 组装 reduced gradient
6. 可选地组装 Hessian action

这是 `GOATTM` 身份中最重要的设计之一。

## 6. 时间积分器支持

公共 solver 层支持：

- `implicit_midpoint`
- `explicit_euler`
- `rk4`

在 workflow 层面：

- rollout dispatch 支持三者
- 一阶 loss gradient 支持三者
- reduced Hessian action 支持三者
- trainer configuration 支持三者

在 helper 层面：

- 仍有部分例程是特定实现
- midpoint 风格假设仍然出现在个别底层路径中

所以更准确的描述应当是：
公共训练栈已经是多积分器的，而部分内部组件仍然带有专门化实现。

## 7. 导数支持

### 一阶导数

代码中已经实现了基于 exact discrete adjoint 的一阶导数。
这是一项核心优势，因为优化过程和真正使用的离散前向积分保持一致。

### 二阶导数

reduced objective workflow 还支持 Hessian-action 计算，并且可以通过重复 action evaluation
显式构造 reduced Hessian。这使得 reduced 层的 Newton-style 或近似二阶方法成为可能。

训练器里已经通过 `NewtonActionUpdater` 把这一点体现出来了。

## 8. MPI 执行模型

并行执行采用 sample-parallel 方式。

每个 rank：

- 持有 manifest 的一个子集
- 只加载本地 sample
- 计算本地 rollout/loss/gradient 贡献
- 参与全局归约

这套模式用于：

- 数据集 loss evaluation
- decoder normal-equation assembly
- reduced gradient assembly
- Hessian-action assembly

通常 root rank 负责：

- 求解某些全局系统
- 写集中式日志
- 写 checkpoint 和 summary

## 9. 输出 artifacts 与 provenance

训练和预处理代码会尽量保存可复现信息，例如：

- `config.json`
- `split.json`
- `preprocess.json`
- `metrics.jsonl`
- `summary.txt`
- `timing_summary.txt`
- `timing_summary.json`
- checkpoint 文件
- stdout/stderr logs
- failure records

这是当前实现里一个很重要但容易被低估的优点。

## 10. 当前优势

目前代码库最强的地方包括：

- quadratic latent dynamics 的结构化参数化
- reduced best-response 训练组织
- 多积分器分支上的精确离散导数
- MPI-aware 的数据集评估
- preprocessing 和 initialization artifact 管理
- 围绕 rollout 与 derivative correctness 的测试文化

## 11. 风险与限制

当前的限制并不适合粗暴总结为“数学有问题”，更准确的说法是：

- `OpInf` 初始化仍可能产生数值上过激的模型
- 对显式积分器来说，差的初始化或较大的步长会引发稳定性问题
- 某些底层 helper 仍带有 midpoint 假设
- 库的一部分仍处于研究性快速迭代阶段，而不是完全冻结的产品 API

所以文档和介绍应该做到两点：

- 充分体现这个库已经具有的 library 级能力
- 同时避免把所有内部组件都说成已经完全通用、完全产品化

## 12. 推荐的标准介绍语

如果要用一句相对简洁但又准确的话来介绍 `GOATTM`，我建议：

`GOATTM` 是一个面向 quadratic latent dynamics 的降阶建模与 reduced-QoI 训练库，提供结构化参数化、多种时间积分器、精确离散导数、decoder best-response 优化、数据预处理以及支持 MPI 的训练 workflow。
