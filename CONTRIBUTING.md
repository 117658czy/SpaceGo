# 贡献指南

感谢参与 SpaceGo 维空弈的开发。

## 开发环境

| 组件 | 版本要求 |
|------|---------|
| 操作系统 | Windows 10/11（Linux 可编译引擎） |
| 编译器 | MSVC 2022 (v143) / GCC 11+ |
| CMake | 3.21+ |
| CUDA Toolkit | 12.x（推荐 12.4） |
| LibTorch | 2.6.0+cu124 |

环境变量与路径配置参见 [BUILD_GUIDE.md](BUILD_GUIDE.md)。

## 代码规范

1. **语言标准**：C++17。
2. **命名空间**：
   - 围棋核心：`spacego::go`
   - 训练：`spacego::go::training`
   - 框架：`spacego::framework`
   - 协议：`spacego::network::protocol`
3. **注释**：中文注释（UTF-8），说明“为什么”而非“是什么”。
4. **风格**：遵循 Google C++ 风格，优先可读性。
5. **禁止简化**：不得省略必要实现细节。
6. **禁止重复**：抽取公共逻辑，避免复制粘贴。
7. **批量处理**：涉及 GPU 的组件应支持批量处理以提高利用率。

## 模块结构

新增代码请放入对应模块：

| 模块 | 头文件目录 | 实现目录 | 说明 |
|------|-----------|---------|------|
| 感知层 | `include/go/board/` | `src/perception/` | 位棋盘、Zobrist、CUDA 内核 |
| 异构计算 | `include/go/heterogeneous/` | `src/go/heterogeneous/` | CPU/GPU 负载均衡与任务分解 |
| 推理层 | `include/go/neural/` | `src/go/neural/` | CA-MOE 专家网络、CUDA 推理 |
| 搜索加速 | `include/go/search/` | `src/go/search/` | HMGS-DP 优化、拓扑加速 |
| 决策层 | `include/network/algorithms/` | `src/decision/algorithms/` | HMGS-DP 算法核心 |
| 训练 | `include/go/training/` | `src/go/training/` | 监督/无监督/二倍体/ELO 门控 |
| 引擎封装 | `include/go/engine/` | `src/decision/engine/` | GoEngine |
| 框架层 | `include/framework/` | `src/framework/` | 日志、性能监控 |
| 协议 | `include/network/protocol/` | `src/network/protocol/` | GTP |

> **重要**：新增源文件后必须同步更新根级 `CMakeLists.txt` 的 `SHARED_SOURCES`，否则不会被编译。

## 工作流程

1. 确认需求与设计（优先在 `docs/` 中补充设计说明）。
2. 在对应模块中实现（头文件 + 实现文件）。
3. 更新 `CMakeLists.txt` 源文件列表。
4. 编译验证：`cmake --build build --config Release`。
5. 更新文档（README、`docs/`）。

## 提交规范

提交信息使用中文，格式：

```
<类型>: <简要描述>

<详细说明（可选）>
```

类型：`feat` / `fix` / `docs` / `refactor` / `perf` / `test` / `chore`

示例：

```
perf: HMGS-DP 换位表改用开放寻址哈希

将换位表由 std::unordered_map 替换为开放寻址实现，
降低高并发模拟下的哈希冲突与内存开销。
```

## 测试

提交前请确保：

1. 代码可正常编译（Release）。
2. 核心功能可运行：GTP 协议响应、`genmove` 搜索、训练入口。
3. 关键路径性能无明显回退。

## 联系方式

- 邮箱：3040811677@qq.com
