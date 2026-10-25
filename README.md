<div align="center">

# SpaceGo 维空弈

**从零实现的 AlphaZero 式围棋引擎 · HMGS-DP 搜索 × CA-MOE 混合专家**

C++17 · CUDA · LibTorch · GTP · 约 5 万行自研代码

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](CMakeLists.txt)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900.svg)](CMakeLists.txt)
[![LibTorch](https://img.shields.io/badge/LibTorch-2.6.0%2Bcu124-EE4C2C.svg)](CMakeLists.txt)
[![GTP](https://img.shields.io/badge/Protocol-GTP-4B8BBE.svg)](src/network/protocol/gtp_protocol.cpp)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[快速开始](#快速开始) · [核心算法](#核心算法与创新点) · [推理主链路](#推理主链路与模型文件) · [训练体系](#训练体系) · [实测数据](#实测数据与当前状态) · [路线图](#路线图与贡献方向)

</div>

---

> **English summary**
>
> SpaceGo (维空弈) is an open-source, from-scratch AlphaZero-style Go engine written in C++17/CUDA with LibTorch.
> It combines **HMGS-DP** (Hierarchical Merged Graph Search with Diversity Protection: parallel UCT, Zobrist
> transposition-table DAG merging, virtual loss, work stealing and batch inference) with **CA-MOE**
> (an 8-expert, ~62M-parameter mixture-of-experts policy/value network with a gating network and opponent-style
> conditioning). The repository ships the engine, a four-mode training stack (supervised / unsupervised /
> diploid competition / ELO-gated online training with KataGo evaluation and distillation), a Win32 GUI and
> GTP support. **Status: research prototype.** The architecture and training pipeline are complete and
> reproducible; playing strength is still limited and known defects are documented below. Contributions and
> benchmark results are welcome.

---

## 为什么值得关注

- **不是调参脚本，而是一套完整引擎栈**：位棋盘、搜索、神经网络推理、训练闭环、GTP 接口、
  图形界面、单元测试与构建系统全部自研，约 5 万行 C++/CUDA。
- **两套自研算法骨架**：搜索侧 HMGS-DP，推理侧 CA-MOE。二者通过“评估器注入”解耦，
  可独立替换、单独压测。
- **训练闭环可复现**：自对弈 → 经验回放 → 门控接受/回滚 → KataGo 目差评估 → 教师蒸馏，
  日志与检查点格式完整保留。
- **面向工程而不是论文**：可裁剪构建（CUDA / LibTorch / 训练 / GUI / 测试独立开关）、
  失败自动降级、跨平台抽象、中文注释与完整文档。
- **诚实的基准**：仓库同时记录“做到了什么”和“还差什么”，所有数字都能在训练日志中核对。

> 如果这个项目对你有启发，欢迎点一个 **Star**。Star 是这个项目继续公开训练日志、
> 发布模型与基准数据的直接动力。

---

## 目录

- [引擎一览](#引擎一览)
- [推理主链路与模型文件](#推理主链路与模型文件)
- [核心算法与创新点](#核心算法与创新点)
- [训练体系](#训练体系)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [实测数据与当前状态](#实测数据与当前状态)
- [已知限制](#已知限制)
- [路线图与贡献方向](#路线图与贡献方向)
- [文档与许可证](#文档与许可证)

---

## 引擎一览

| 维度 | 实现 |
|------|------|
| 语言 / 标准 | C++17（引擎、训练、GUI）+ CUDA C++ |
| 棋盘表示 | 19 路位棋盘（`6 × uint64_t` 黑白各一份）+ Zobrist 哈希 |
| 搜索 | HMGS-DP：UCT + 换位表 DAG 合并 + 批量推理；虚拟损失 / 工作窃取队列已提供框架，GTP 主循环当前为单线程批量实现 |
| 神经网络 | CA-MOE：共享主干（10 残差块 / 256 通道）+ 8 组策略头 / 价值头 + 门控 |
| 模型规模 | 约 **6200 万**参数，检查点约 **237 MB** |
| 输入 / 输出 | `17 × 19 × 19` 特征张量 → 362 维策略（361 点 + pass）+ 1 维价值 |
| 推理后端 | LibTorch（CPU / CUDA），支持批量前向与混合精度训练 |
| 协议 | GTP（`name` / `version` / `boardsize` / `clear_board` / `komi` / `play` / `genmove` / `undo` / `showboard` / `final_score` / `time_settings`） |
| 训练模式 | 监督预训练 / 无监督自强化 / 二倍体分化竞争 / ELO 门控在线训练 |
| 工程配套 | CMake 可裁剪构建、GDI+ GUI、GoogleTest 单元测试、算法文档 17 篇 |
| 许可证 | AGPL-3.0 |

---

## 推理主链路与模型文件

这是仓库中**真正作用于 GTP 对局推理**的代码路径，按调用顺序列出：

```text
spacego.exe --gtp --model=<checkpoint>.pt --simulations=1600
        │
        ▼
src/main.cpp                     解析 --gtp / --model= / --simulations=
        │                        RunGTP()
        ▼
src/decision/engine/go_engine.cpp
        ├─ GoEngine::initialize()            判断模型后缀是否为 .pt
        ├─ GoEngine::init_neural_engine()    构造 CA_MOEExpertSystem(8, 19, 0.1, 256)
        │      ├─ CA_MOEExpertSystem::load() 读取 LibTorch OutputArchive 检查点
        │      ├─ to(CUDA) + set_eval_mode()
        │      ├─ policy_cb()                策略先验（362 维）
        │      ├─ value_cb()                 价值（当前行棋方视角）
        │      └─ batch_cb()                 批量策略 + 价值
        ├─ HMGSDP_Optimized::set_neural_evaluator()
        ├─ HMGSDP_Optimized::set_batch_evaluator()
        └─ GoEngine::genmove() → do_search() → HMGSDP_Optimized::search()
                │
                ▼
        src/go/search/hmgs_dp_optimized.cpp
                ├─ select_node_for_expansion()   UCT 下潜
                ├─ batch_eval_fn_()              批量评估叶子
                ├─ expand_node_with_prior()      top-K 展开 + 换位表合并
                └─ backpropagate()               价值回传
                │
                ▼
        src/network/protocol/gtp_protocol.cpp
                └─ genmove → "= D4" 等 GTP 响应
```

### 关键文件

| 文件 | 作用 |
|------|------|
| [src/main.cpp](src/main.cpp) | 程序入口、参数解析、GTP / 交互式 / 训练模式分发 |
| [src/decision/engine/go_engine.cpp](src/decision/engine/go_engine.cpp) | 引擎封装、神经网络初始化、评估器注入、genmove |
| [src/go/neural/ca_moe_expert.cpp](src/go/neural/ca_moe_expert.cpp) | CA-MOE 前向、门控、策略/价值融合、save/load |
| [src/go/search/hmgs_dp_optimized.cpp](src/go/search/hmgs_dp_optimized.cpp) | HMGS-DP 搜索主循环、UCT、DAG 合并、回传 |
| [src/perception/board/go_board_cuda.cpp](src/perception/board/go_board_cuda.cpp) | 位棋盘操作、`encode_to_tensor()` 特征编码 |
| [src/network/protocol/gtp_protocol.cpp](src/network/protocol/gtp_protocol.cpp) | GTP 命令实现 |

### 模型文件说明（重要）

- **实际加载的模型**：`--model=` 指向的 `*.pt`，由训练程序 `spacego_train` 产出。
- **格式**：LibTorch `torch::serialize::OutputArchive`，**不是** TorchScript，也不是裸 `state_dict`。
- **必需键**：`backbone`、`gating_network`、`policy_head_0..7`、`value_head_0..7`、
  `num_experts`、`board_size`、`hidden_dim`、`is_training`。
- **结构固定**：8 专家、10 个残差块、17 输入通道；`hidden_dim` 先按 256 加载，失败回退 128。
- **典型产物**：
  - `training_output_fresh/elo_gated/checkpoints/elo_best.pt`（ELO 门控训练最佳检查点）
  - `training_output/diploid/checkpoints/diploid_best.pt`（二倍体竞争最佳检查点）
  - `training_output_*/..._best.pt`（无监督自强化最佳检查点）
- **未随仓库分发**：单个检查点约 237 MB，超过 GitHub 普通文件限制。请自行训练，
  或将发布版模型放到 HuggingFace / Release 附件后通过 `--model=` 加载。
- **无模型时**：引擎自动回退到“规则搜索 + 启发式评估”，神经网络不参与；这不是完整棋力。
- **模型放置说明**：[models/README.txt](models/README.txt)。

> 代码中**不存在**对 `models/hybrid_go_model.pt`、`models/ca_moe_expert.pt` 的引用；
> 早期 README 中的这两个文件名是历史遗留，已在本版更正。

---

## 核心算法与创新点

### 1. HMGS-DP：分层合并图搜索 + 多样性保护

HMGS-DP = **H**ierarchical **M**erged **G**raph **S**earch with **D**iversity **P**rotection。

**（1）UCT 与并行框架**

选择准则以 UCT 为基础，节点保存访问数、累计价值、先验概率与虚拟损失：

```text
UCT(i) = Q(i) + C · P(i) · √(ln N / n_i)
```

`HMGS_Node` 已包含 `virtual_loss` 原子计数，仓库也提供 `WorkStealingQueue` 与
多 worker 配置接口；但 `HMGSDP_Optimized::search()` 目前是**单线程批量主循环**，
虚拟损失尚未接入选择/回传（见[已知限制](#已知限制)）。这一层是为后续并行化预留的骨架，
不是已生效的并行加速。

**（2）换位表状态合并（树 → DAG）**

以 Zobrist 哈希（棋子布局 + 行棋方）为键建立换位表：

```cpp
std::unordered_map<uint64_t, HMGS_Node*> tt_;
```

相同局面直接复用既有节点（`add_child_ref`），搜索结构由树变为**有向无环图**，
同一局面只评估一次，避免重复神经网络推理。换位表统一持有节点所有权，
规避 DAG 多父共享导致的重复释放。

**（3）批量推理路径**

`set_batch_evaluator()` 注入批量回调后，搜索按 `batch_size` 收集叶子统一前向，
设计目标是一次 GPU 调用同时得到策略先验与价值、降低自对弈数据生成的墙钟时间。
当前实现的批量收集存在“重复选中同一叶子仍消耗迭代预算”的缺陷，
导致有效模拟数远低于配置值（P0，见[已知限制](#已知限制)）。

**（4）拓扑加速与持久化**

`SpatialTopologyAccelerator` 批量计算空间拓扑特征，并按任务规模在 CPU 多线程与 GPU
之间分片；搜索树、合并网络、多样性图支持导出/导入（含 v1 兼容），便于离线分析。

> 设计目标与实际实现状态见 [docs/技术文档/算法说明.md](docs/技术文档/算法说明.md)
> 与 [已知限制](#已知限制)。

### 2. CA-MOE：因果感知混合专家元决策

CA-MOE = **C**ausal-**A**ware **M**ixture **o**f **E**xperts。

```text
输入 17×19×19
   │
SharedBackbone：输入卷积 + 10 × 残差块 + Dropout（256 通道）
   │  共享特征
   ├── ExpertPolicyHead × 8  → 每个专家 362 维策略
   ├── ExpertValueHead  × 8  → 每个专家 1 维价值
   └── GatingNetwork
         fc1(hidden_dim×361 → 512) → fc2(512 → 8) → softmax
         + style_projection(5 → 8)  对手风格偏置

final_policy = Σ_e w_e · policy_e
final_value  = Σ_e w_e · value_e
w = softmax(Gating(features) + style_projection(opponent_style))
```

**创新点：**

- **共享主干 + 多专家头**：主干只算一次，8 个专家头并行输出，兼顾容量与推理成本。
- **对手风格条件化**：5 维风格向量 `[攻击, 防守, 均衡, 实地, 厚势]` 经**零初始化**投影
  叠加为门控偏置，让“按对手棋风选择专家”成为可学习机制；未注入时行为与普通门控一致，
  旧检查点可兼容加载（`load_compat`）。
- **防门控坍缩**：项目在真实训练中定位到“温度软化反而削弱正则”的问题，最终固定
  softmax 温度 1.0，改用 **KL / 负载均衡正则 + Dropout + logits L2** 抑制专家坍缩。
  这段失败经验完整保留在代码注释与训练日志中。

### 3. 从零实现的位棋盘与特征编码

- 19×19 = 361 点，用 `6 × uint64_t` 位图表示黑白双方，落子/提子/气/连通块均以位运算完成。
- Zobrist 哈希叠加行棋方分量，用于换位表、经验去重与缓存。
- `encode_to_tensor()` 输出 17 通道接口；当前实现填充前 3 个平面（黑、白、行棋方），
  其余通道恒为 0（训练与推理约定一致）。这是后续可扩展点，详见[已知限制](#已知限制)。

### 4. 训练体系（四种模式）

| 模式 | 入口 | 关键机制 |
|------|------|---------|
| 监督预训练 | `spacego_train --mode=supervised` | 模拟数据冷启动，策略交叉熵 + 价值 MSE |
| 无监督自强化 | `spacego_train --mode=unsupervised` | 自对弈 + Dirichlet 噪声 + 经验回放 + 渐进式经验池 + AMP |
| 二倍体分化竞争 | `spacego_train --mode=diploid` | 双模型分支对抗，按 ELO 门槛接受/回退 |
| ELO 门控在线训练 | `spacego_train --mode=elo-gated` | 逐局学习 + KataGo 目差标尺 + 严格改善接受/回滚 + 教师蒸馏 |

**ELO 门控**是本项目最有辨识度的训练设计：每轮让当前模型与冻结基线对弈并学习，
再用 KataGo 目差作为标尺；只有标尺**严格改善**才接受新参数，否则回滚到轮初快照。
它把“训练是否真的变强”变成一个可审计的在线决策，而不是只看损失曲线。

### 5. 工程创新

- **评估器注入解耦**：搜索只依赖 `function` 回调，神经网络、规则评估、批量评估可自由替换。
- **可裁剪构建**：`USE_CUDA` / `USE_LIBTORCH` / `BUILD_TRAIN` / `BUILD_GUI` / `BUILD_TESTS`
  独立开关，无 GPU 也能编译引擎主体。
- **失败降级**：NVML 不可用时回退启发式 GPU 利用率；无模型时回退规则搜索；
  批量评估失败时回退单节点评估。
- **门面 + 专职组件**：`HMGSDP` 门面只做生命周期与转发，搜索主控、UCT、相似度、
  合并网络、节点管理、持久化各自独立，便于单测与替换。

---

## 快速开始

### 环境要求

| 组件 | 版本 |
|------|------|
| 操作系统 | Windows 10/11（GUI 仅 Windows）；Linux 可编译引擎 |
| 编译器 | Visual Studio 2022（MSVC v143）或 GCC 11+ |
| CMake | 3.21+ |
| CUDA Toolkit | 12.x（可选，推荐 12.4） |
| LibTorch | 2.6.0+cu124（可选，神经网络推理与训练必需） |

### 构建

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
      -DLIBTORCH_ROOT="D:/libtorch-win-shared-with-deps-2.6.0+cu124/libtorch"
cmake --build build --config Release
```

产物位于 `build/Release/`：`spacego.exe`、`spacego_train.exe`、`SpaceGo_GUI.exe`。

最小构建（无 CUDA / 无 LibTorch，仅规则引擎）：

```powershell
cmake -S . -B build -DUSE_CUDA=OFF -DUSE_LIBTORCH=OFF -DBUILD_TRAIN=OFF -DBUILD_GUI=OFF
cmake --build build --config Release
```

> 源码路径含中文或空格时，MSBuild 传给 `nvcc` 的参数可能被空格拆开。
> 请使用纯 ASCII 目录联结后构建，详见 [BUILD_GUIDE.md](BUILD_GUIDE.md)。

### 运行

```powershell
# GTP 模式（默认）：加载模型并设置每步搜索模拟数
.\spacego.exe --gtp --model=<checkpoint>.pt --simulations=1600

# 无模型：规则/启发式回退
.\spacego.exe --gtp --simulations=1600

# 交互式命令行
.\spacego.exe --interactive

# 训练
.\spacego_train.exe --mode=elo-gated --elo-rounds=300 --output-dir=./training_output
```

### 对接 Sabaki / Lizzie

- 引擎路径：`spacego.exe`
- 启动参数：`--gtp`（需要模型时：`--gtp --model=<路径>`）
- 工作目录：可执行文件所在目录

---

## 项目结构

```text
include/
├── config/            平台与编译期配置
├── framework/         日志、性能监控、算法基类、AMP 梯度缩放
├── go/
│   ├── board/         位棋盘、CUDA 棋盘、Zobrist 哈希
│   ├── engine/        GoEngine 引擎封装
│   ├── heterogeneous/ CPU/GPU 异构计算与负载均衡
│   ├── neural/        CA-MOE 专家网络、CUDA 推理、对手风格画像
│   ├── search/        HMGS-DP 优化实现、CUDA 搜索、拓扑加速
│   └── training/      四种训练模式与调参工具
└── network/
    ├── algorithms/   HMGS-DP 算法核心（节点/UCT/合并/持久化）
    └── protocol/     GTP 协议

src/                  与 include/ 对应的实现（含 CUDA 内核）
GUI/                  独立 Win32 + GDI+ 图形界面
tests/                GoogleTest 单元测试（可选）
docs/                 技术文档与算法设计资料
models/               模型放置说明（权重不随仓库分发）
```

> `network/algorithms` 是历史命名，实际承载 HMGS-DP 搜索算法核心，与网络通信无关。

---

## 实测数据与当前状态

以下数字来自仓库内的训练日志与检查点，可复现：

| 指标 | 数值 | 说明 |
|------|------|------|
| CA-MOE 参数量 | 约 6200 万 | 8 专家，`hidden_dim=256` |
| 检查点大小 | 约 237 MB | 单个 `*.pt` |
| 搜索模拟数 | 1600 / 步（可配） | ELO 门控训练配置 |
| 单步耗时 | 约 35–55 s | GTX 1660 Ti，1600 模拟 |
| 对 KataGo 目差 | -350 ~ -368 目 | 最优观测 **-349.5 目**（KataGo 10 visits） |
| 代码规模 | 约 5 万行 / 139 个 C++/CUDA 文件 | 含 GUI 与测试 |

**当前定位：研究原型。** 架构、训练闭环与工程链路已经跑通，但模型棋力仍远低于主流引擎，
搜索与特征编码也存在下文列出的已知缺陷。项目选择公开这些数据，而不是用未经测量的
“工业级棋力”做宣传。

---

## 已知限制

> 这些问题来自对当前代码的逐行审查，欢迎通过 Issue / PR 一起解决。
> 详细分析见 [docs/技术文档/算法说明.md](docs/技术文档/算法说明.md) 与
> [docs/技术文档/成果说明.md](docs/技术文档/成果说明.md)。

| 级别 | 问题 | 影响 |
|------|------|------|
| P0 | 批量收集循环按 `batch_size` 消耗迭代预算，同一批内重复选中同一叶子 | `--simulations=800` 实际只完成约 6–7 次有效展开 |
| P0 | UCT 直接最大化“子节点视角”价值，父节点未取反 | 选择方向错误，搜索统计失真 |
| P0 | 价值翻转判断使用 `current_player == -1`，而 `StoneColor::WHITE == 2` | 白方行棋局面的价值未翻转 |
| P1 | `genmove` 返回 pass 时未切换行棋方/累加连续 pass | 连续两 pass 无法正常终局 |
| P1 | `boardsize` 接受 9/13，但坐标、编码、策略索引均按 19 路硬编码 | 非 19 路对局会错位 |
| P1 | CUDA 棋盘内核目前是接口占位（气数恒 0、合法性仅判空） | GPU 批量棋盘计算尚不可用于实战 |
| P2 | 17 通道输入仅填充 3 个平面 | 输入表达力受限，但与训练约定一致 |
| P2 | GUI 尚未接入引擎；`time_settings` / `stop` 未实现 | 界面可独立运行，但不参与对弈 |

---

## 路线图与贡献方向

**近期（正确性优先）**

1. 修复搜索预算、UCT 价值方向、白方价值翻转三处 P0 缺陷，并补充回归测试。
2. 统一 `StoneColor` / `±1` 价值约定，增加黑/白对称性测试。
3. 修复 GTP pass 与行棋方同步，补齐 `stop`、`time_left`。
4. 明确只支持 19 路，或把棋盘尺寸参数化到编码与策略索引。

**中期（棋力与性能）**

5. 扩展 17 通道特征（历史、气、打劫、征子等），重训基线。
6. 将 CUDA 棋盘内核从占位升级为真实批量实现，接入自对弈数据生成。
7. 让批量推理在 GTP 路径默认生效，压测节点/秒与 GPU 利用率。
8. 扩大评估局数，降低 ELO 门控的标尺噪声，抑制门控坍缩。

**长期（生态）**

9. 发布可下载的预训练检查点与基准对局。
10. GUI 通过 GTP 子进程接入引擎。
11. 建立 CI（Windows / Linux 构建 + 单元测试 + GTP 冒烟测试）。

**Good first issues**：文档纠错、GTP 命令补全、单元测试、基准脚本、英文文档翻译。

---

## 文档与许可证

- [算法说明](docs/技术文档/算法说明.md)
- [训练日志总结](docs/技术文档/训练日志总结.md)
- [成果说明](docs/技术文档/成果说明.md)
- [构建指南](BUILD_GUIDE.md)
- [贡献指南](CONTRIBUTING.md)
- [算法设计资料](docs/算法数据/)

本项目基于 **GNU Affero General Public License v3.0 (AGPL-3.0)** 开源。
若修改后通过网络提供服务，须公开修改后的源代码；衍生作品须同样采用 AGPL-3.0。
完整条款见 [LICENSE](LICENSE)。

---

<div align="center">

**SpaceGo 维空弈** —— 把一套完整的围棋 AI 引擎拆开给你看。

如果这个项目对你有帮助，欢迎 **Star**、**Fork**、提交 Issue 或 PR。

</div>
