# SpaceGo 维空弈 - 核心技术架构文档

## 一、系统概述

SpaceGo 维空弈是一个**工业级、高性能、现代化**的围棋 AI 系统，融合了最新的高性能计算、深度学习和因果推理技术。系统采用五层架构设计，实现了从底层棋盘表示到顶层用户界面的完整功能链。

### 1.1 核心特性

- **高性能计算**：CUDA GPU 加速，支持 Tensor Core 和混合精度计算 (FP16/FP32)
- **先进算法**：HMGS-DP（分层合并图搜索与多样性保护）+ CA-MOE（因果混合专家元决策系统）
- **深度学习**：LibTorch 集成，支持三种神经网络架构（EnhancedNeuralNetwork、StyleAdaptedNN、HybridNeuralNetwork）
- **因果推理**：创新的因果链建模、关键点检测和高斯泼溅技术
- **模块化架构**：清晰的五层架构（感知层、推理层、决策层、框架层、表现层）
- **跨平台支持**：Windows 优先，Linux 兼容，CMake 构建系统

### 1.2 技术栈

| 技术类别 | 技术选型 | 说明 |
|---------|---------|------|
| **编程语言** | C++20, CUDA | 核心逻辑使用 C++20，GPU 加速使用 CUDA |
| **深度学习** | LibTorch | PyTorch C++ 接口，用于神经网络推理 |
| **构建系统** | CMake 3.18+ | 跨平台构建，支持 Visual Studio 2022 |
| **GPU 计算** | CUDA Toolkit 11.0+ | 支持 Turing、Ampere、Ada 架构 |
| **GUI 框架** | GDI+ (Win32) | 自研 GUI 框架，支持现代 UI 效果 |

---

## 二、五层架构详解

### 2.1 感知层 (Perception Layer)

感知层负责棋盘状态的表示、特征提取和 CUDA 加速计算。

#### 2.1.1 位棋盘表示

SpaceGo 使用**位棋盘 (bitboard)** 优化内存访问和计算效率：

```cpp
// 19x19 棋盘 = 361 个交叉点，使用 6 个 uint64_t 存储
static const int BITBOARD_WORDS = 6;

class GoBoardCPU {
    uint64_t black_bits_[BITBOARD_WORDS];  // 黑棋位图
    uint64_t white_bits_[BITBOARD_WORDS];  // 白棋位图
    
    // 支持高效的气计算、合法着法生成、提子等操作
    int calculate_liberties(int x, int y) const;
    std::vector<Move> get_legal_moves(StoneColor color) const;
    bool play_move(Move move);
};
```

**核心优化**：
- 使用位运算快速计算气、连接、提子
- 增量式 Zobrist 哈希更新
- 支持快速的棋盘状态克隆和比较

#### 2.1.2 CUDA 加速内核

SpaceGo 实现了完整的 CUDA 加速内核，支持大规模并行计算：

```cpp
class GoBoardCUDA {
    // GPU 端批量处理，支持大规模并行计算
    void upload_batch(const std::vector<GoBoardCPU>& boards);
    std::vector<int> batch_calculate_liberties();
    std::vector<std::vector<Move>> batch_get_legal_moves();
    
private:
    thrust::device_vector<uint64_t> black_bits_device_;
    thrust::device_vector<uint64_t> white_bits_device_;
    cudaStream_t* streams_;  // 多 CUDA 流支持
};
```

**CUDA 优化策略**：
- **批量处理**：支持 64/128/256/512 批次大小，减少内核启动开销
- **内存访问优化**：合并全局内存访问，使用共享内存缓存
- **异步计算**：多 CUDA 流并发执行，隐藏数据传输延迟
- **Tensor Core 支持**：FP16 混合精度计算，加速神经网络推理

#### 2.1.3 特征提取

将棋盘状态转换为神经网络输入（17 通道特征平面）：

```cpp
torch::Tensor board_to_tensor(const GoBoardCPU& board, StoneColor player) {
    // 创建 17 通道特征张量
    torch::Tensor features = torch::zeros({1, 17, 19, 19});
    
    // 通道 1-8: 当前玩家棋子（最近 8 步）
    // 通道 9-16: 对手棋子（最近 8 步）
    // 通道 17: 玩家指示符
    
    return features;
}
```

---

### 2.2 推理层 (Reasoning Layer)

推理层负责神经网络推理、因果推理和风格迁移。

#### 2.2.1 神经网络集成

SpaceGo 支持三种先进的神经网络架构：

**1. EnhancedNeuralNetwork（基础增强版）**
- 29 层残差块
- 多头注意力机制
- 多尺度特征提取
- 双头输出（策略 + 价值）

```cpp
class EnhancedNeuralNetwork {
    // 初始卷积层
    Conv2d conv_input_;  // 17 -> 96 通道
    BatchNorm2d bn_input_;
    
    // 29 个残差块
    std::vector<ResBlock> res_blocks_;
    
    // 注意力机制
    MultiheadAttention attention_;
    
    // 策略头和价值头
    SpatialAttentionHead policy_head_;
    EnhancedValueHead value_head_;
};
```

**2. StyleAdaptedNN（风格适配网络）**
- 风格编码器
- 门控机制
- 风格路由器
- 元风格控制器

```cpp
class StyleAdaptedNN {
    // 风格编码器
    StyleEncoder style_encoder_;
    
    // 门控机制
    GatingNetwork gating_;
    
    // 风格适配器
    StyleAdapter style_adapter_;
};
```

**3. HybridNeuralNetwork（混合架构）**
- Transformer 编码器
- 图卷积网络 (GCN)
- 多尺度注意力机制
- 专门化头设计（空间策略、全局策略、局部策略）

```cpp
class HybridNeuralNetwork {
    // Transformer 增强
    TransformerEncoder transformer_encoder_;
    
    // 图卷积网络
    std::vector<GCNLayer> gcn_layers_;
    
    // 专门化注意力头
    SpatialAttention spatial_attention_;
    ConnectionAttention connection_attention_;
    TerritoryAttention territory_attention_;
    
    // 多策略头
    SpatialPolicyHead spatial_policy_head_;
    GlobalPolicyHead global_policy_head_;
    LocalPolicyHead local_policy_head_;
};
```

#### 2.2.2 混合精度推理

支持 FP16/FP32 混合精度计算，利用 Tensor Core 加速：

```cpp
class NeuralNetworkCUDA {
    bool use_fp16_;              // 启用 FP16
    bool use_tensor_core_;       // 启用 Tensor Core
    size_t optimal_batch_size_;  // 最优批次大小
    
    void enable_mixed_precision(bool enable);
    void enable_tensor_core(bool enable);
    
    PolicyValueOutput inference(const std::string& board_state);
    std::vector<PolicyValueOutput> batch_inference(
        const std::vector<std::string>& board_states);
};
```

#### 2.2.3 因果推理系统

SpaceGo 实现了完整的因果推理框架：

**因果链生成器**
```cpp
class CausalChainGenerator {
    // 构建着法 - 结果因果关系链
    CausalChain generate_chain(const BoardState& state, const Move& move);
    
    // 支持多步推理
    std::vector<CausalLink> multi_step_reasoning(const CausalChain& chain);
};
```

**关键点检测器**
```cpp
class KeypointDetector {
    // 识别局面关键点
    std::vector<Keypoint> detect_keypoints(const BoardState& state);
    
    // 提取因果特征
    CausalFeatures extract_features(const Keypoint& keypoint);
};
```

**高斯泼溅**
```cpp
class GaussianSplatting {
    // 3D 高斯分布建模
    GaussianDistribution model_uncertainty(const BoardState& state);
    
    // 不确定性量化
    float quantify_uncertainty(const GaussianDistribution& dist);
};
```

---

### 2.3 决策层 (Decision Layer)

决策层是 SpaceGo 的核心，实现了先进的搜索算法和元决策系统。

#### 2.3.1 HMGS-DP 算法

**分层合并图搜索与多样性保护（HMGS-DP）** 是 SpaceGo 的核心搜索算法。

**核心特性**：
- **8 分区立方体架构**：将棋盘分为 8 个区域独立探索
- **三副本并行搜索**：3 个并行搜索副本
- **智能状态合并**：基于语义相似性的状态合并
- **多样性保护**：对未合并分支给予特殊价值奖励
- **拓扑感知 UCT**：考虑棋盘拓扑结构的 UCT 变种

```cpp
class HMGSDP_CUDA : public HMGS_DP {
    // CUDA 加速的搜索实现
    std::pair<int, int> search(const std::string& board_state, 
                               int max_iterations = 10000) override;
    
    // 批量搜索（CUDA 特有功能）
    std::vector<std::pair<int, int>> batch_search(
        const std::vector<std::string>& board_states,
        int max_iterations_per_board = 10000);
    
    // 异步搜索支持
    uint64_t start_async_search(const std::string& board_state);
    bool get_async_search_result(uint64_t search_id, AsyncSearchResult& result);
    
private:
    // 多样性保护机制
    DiversityProtection diversity_protection_;
    
    // 合并策略
    MergeStrategy merge_strategy_;
    
    // 拓扑感知 UCT
    TopologyAwareUCT uct_calculator_;
};
```

#### 2.3.2 改进的 UCT 公式

SpaceGo 提出了**拓扑感知的 UCT 公式**：

```
UCT_topo(i) = Q_topo(i) + C × P(i) × √(lnN / n_i) × M(i) × Novelty(i)
```

其中：
- **Q_topo(i)** = 拓扑修正价值 = Q(i) × [1 + α₁P_m(i,2) + α₂P_m(i,3) + α₃P_b(i,2)]
- **P(i)** = 链接神经元的先验概率
- **M(i)** = 合并增强项 = 1 + β₁P_m(i,2) + β₂P_m(i,3) + β₃P_b(i,2)
- **Novelty(i)** = 新奇性奖励 = 1 + δ × U(i) × R(t)

**概率指标**：
- **P_m(i,k)**：节点 i 在 k 层搜索中的**合并概率**
- **P_b(i,k)**：节点 i 在 k 层搜索中的**被合并概率**

#### 2.3.3 CA-MOE 系统

**因果混合专家元决策系统（CA-MOE）** 是 SpaceGo 的创新决策机制。

**6 种专家类型**：
```cpp
enum class ExpertType {
    OPENING_EXPERT,      // 开局专家
    MIDDLE_GAME_EXPERT,  // 中盘专家
    ENDGAME_EXPERT,      // 收官专家
    TACTICAL_EXPERT,     // 战术专家
    POSITIONAL_EXPERT,   // 局面专家
    UNIVERSAL_EXPERT     // 通用专家
};
```

**门控网络架构**：
```cpp
class CAMOE {
    // 门控网络
    GatingNetwork gating_network_;  // 输入 10 维，隐藏层 32，Top-K=3
    
    // 专家模型池
    std::vector<ExpertModel> experts_;
    
    // 动态路由
    std::vector<ExpertType> select_experts(const BoardState& state);
    
    // 专家输出融合
    float fuse_expert_outputs(const std::vector<float>& expert_outputs,
                             const std::vector<float>& gating_weights);
};
```

**动态路由机制**：
- 输入维度：10（棋盘特征）
- 隐藏层：32
- Top-K 选择：3 个专家
- 温度参数：1.0
- 探索噪声：0.1

#### 2.3.4 合并策略

SpaceGo 实现了**分层精度的 CNN 合并函数**和**合并决策的 UCT 框架**：

```cpp
class MergeStrategy {
    // 分层精度的 CNN 合并函数
    float hierarchical_cnn_merge(HMGSNode& node1, HMGSNode& node2) {
        // 根据节点深度选择 CNN 精度等级
        int max_depth = std::max(node1.depth, node2.depth);
        CNNAccuracyLevel level = select_cnn_accuracy_level(max_depth);
        
        // 浅层节点 → 轻量 CNN
        // 中层节点 → 标准 CNN
        // 深层节点 → 高精度 CNN
        
        return cnn_merge_with_accuracy_level(node1, node2, level);
    }
    
    // 合并质量评分
    float calculate_merge_score(HMGSNode& node1, HMGSNode& node2) {
        float state_similarity = calculate_state_similarity(node1, node2);
        float value_compatibility = calculate_value_compatibility(node1, node2);
        float topology_consistency = calculate_topology_consistency(node1, node2);
        
        return (0.4 * state_similarity + 
                0.3 * value_compatibility + 
                0.3 * topology_consistency);
    }
    
    // 合并 UCT 框架
    float calculate_merge_uct(HMGSNode& node, HMGSNode& candidate) {
        float q_merge = calculate_merge_score(node, candidate);
        float exploration_term = c_merge_ * std::sqrt(std::log(parent.visit_count) / 
                                                      (1 + candidate.visit_count));
        float merge_cost = calculate_merge_cost(node, candidate);
        
        return q_merge + exploration_term - merge_cost;
    }
};
```

#### 2.3.5 多样性保护机制

SpaceGo 设计了完善的**多样性保护机制**，防止搜索过早收敛：

```cpp
class DiversityProtection {
    // 新奇性价值函数
    float calculate_novelty_value(HMGSNode& node) {
        float v_original = node.total_value / node.visit_count;
        float time_decay = std::exp(-lambda_ * (current_time - node.last_visit_time));
        float backtrack_boost = 1 + eta_ * node.backtrack_count;
        
        // V_novel = V_original + γ × (1 - P_merge) × D(t) × B(k)
        return v_original + gamma_ * (1 - node.merge_prob) * time_decay * backtrack_boost;
    }
    
    // 保护性探索概率
    float calculate_protected_exploration_prob(HMGSNode& node, float base_prob) {
        float protected_prob = std::max(base_prob, epsilon_);
        
        if (node.continuous_unmerged > 0) {
            // 连续未合并次数越多，探索概率越高
            float novelty_bonus = beta_ * (node.continuous_unmerged / 100.0);
            protected_prob += novelty_bonus;
        }
        
        return protected_prob;
    }
    
    // 被忽视分支识别
    std::vector<HMGSNode*> identify_neglected_branches(
        std::vector<HMGSNode*>& nodes,
        int u_threshold = 5,
        float t_threshold = 10.0) {
        
        std::vector<HMGSNode*> neglected;
        for (auto* node : nodes) {
            if (node.continuous_unmerged > u_threshold &&
                (current_time - node.last_visit_time) > t_threshold) {
                neglected.push_back(node);
            }
        }
        return neglected;
    }
};
```

---

### 2.4 框架层 (Framework Layer)

框架层提供日志、配置、性能监控等基础设施。

#### 2.4.1 日志系统

```cpp
class Logger {
    // 多级别日志
    enum class LogLevel {
        TRACE, DEBUG, INFO, WARNING, ERROR, FATAL
    };
    
    // 异步写入
    void log(LogLevel level, const std::string& message);
    
    // 支持控制台和文件输出
    void set_output_target(OutputTarget target);
    
    // 中文支持
    void set_language(Language lang);
};
```

#### 2.4.2 配置管理

```cpp
class ConfigManager {
    // INI 格式配置文件
    void load_config(const std::string& config_path);
    
    // 热重载支持
    void reload_config();
    
    // 运行时参数调整
    template<typename T>
    T get(const std::string& key, const T& default_value);
};
```

#### 2.4.3 性能监控

```cpp
class PerformanceMonitor {
    // CPU/GPU 使用率监控
    double get_cpu_utilization();
    double get_gpu_utilization();
    
    // 内存占用跟踪
    size_t get_memory_usage();
    
    // 组件计时器
    void start_timer(const std::string& component);
    double stop_timer(const std::string& component);
};
```

---

### 2.5 表现层 (Presentation Layer)

表现层实现用户界面和交互。

#### 2.5.1 GUI 架构

SpaceGo 的 GUI 采用自研的 GDI+ 渲染引擎：

**核心组件**：
- **主窗口**：[main_window.cpp](file://e:\SpaceGo%20维空弈\GUI\src\components\main_window.cpp)
- **棋盘组件**：[board.cpp](file://e:\SpaceGo%20维空弈\GUI\src\components\board.cpp)
- **动画系统**：[animation.cpp](file://e:\SpaceGo%20维空弈\GUI\src\core\animation.cpp)
- **渲染器**：[renderer.cpp](file://e:\SpaceGo%20维空弈\GUI\src\core\renderer.cpp)

#### 2.5.2 设计特色

- **视觉风格**：融合 Sabaki 的专注与 Trae 的现代柔和
- **特效**：
  - 圆角弧矩形标签页
  - 半透明毛玻璃效果
  - 动感背景（像素小球涡流运动）
- **色彩系统**：HSL 色彩模型智能配色

---

## 三、核心算法详解

### 3.1 神经网络输入输出

**输入**：17 通道特征平面
- 通道 1-8：当前玩家棋子（最近 8 步）
- 通道 9-16：对手棋子（最近 8 步）
- 通道 17：玩家指示符

**输出**：
- **策略头**：361+1 维概率分布（着法概率，+1 为 pass）
- **价值头**：[-1, 1] 标量（胜率估计）
- **状态编码**：64 维特征向量（可选）

```cpp
struct PolicyValueOutput {
    std::vector<float> policy_probs;  // 着法概率分布
    float value_estimate;             // 价值估计
    std::vector<float> state_encoding; // 状态编码（可选）
    float uncertainty;                // 不确定性估计
};
```

### 3.2 搜索流程

SpaceGo 的搜索流程包含以下关键步骤：

```
1. 初始化阶段
   ├─ 状态空间划分为 8 个初始立方体区域
   ├─ 每个立方体初始化链接神经元
   └─ 建立空合并图谱

2. 并行搜索阶段
   ├─ 对于每个立方体，启动三副本独立搜索
   ├─ 使用改进的 UCT 公式选择节点
   └─ 进行神经网络推理

3. 结果合并与权重调整
   ├─ 计算合并质量评分
   ├─ 执行合并决策
   └─ 调整节点权重

4. 多样性保护
   ├─ 识别未合并分支
   ├─ 计算新奇性价值
   └─ 分配保护性探索资源

5. 回溯与更新
   ├─ 沿搜索路径反向传播
   ├─ 更新节点统计信息
   └─ 更新合并概率和被合并概率
```

### 3.3 性能优化策略

#### 3.3.1 GPU 优化

1. **批量处理优化**
   - 最优批次大小自动计算
   - 减少内核启动开销
   - 提高 GPU 利用率

2. **内存访问优化**
   - 合并访问模式
   - 共享内存缓存
   - 常量内存使用

3. **计算优化**
   - Tensor Core 加速（FP16）
   - 内核融合
   - 异步 CUDA 流

#### 3.3.2 CPU 优化

1. **多线程并行**
   - 并行搜索
   - 异步推理

2. **缓存优化**
   - 棋盘状态缓存
   - 路由决策缓存

3. **SIMD 指令**
   - 位运算优化
   - 向量化计算

---

## 四、构建系统

### 4.1 CMake 配置

```cmake
# C++ 标准
set(CMAKE_CXX_STANDARD 20)

# CUDA 支持（可选）
set(CUDA_ENABLED OFF CACHE BOOL "Enable CUDA compilation")
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")

# 模块开关
set(BUILD_NETWORK OFF)
set(BUILD_GUI OFF)

# 编译器选项（MSVC）
add_compile_options(/MP /FS /W4 /wd4819 /utf-8)
```

### 4.2 库目标划分

项目创建了多个静态库：
- `spacego_common` - 通用工具
- `spacego_framework` - 框架核心
- `spacego_go_board` - 棋盘模块
- `spacego_go_search` - 搜索算法
- `spacego_go_analysis` - 局面分析
- `spacego_go_evaluation` - 评估函数
- `spacego_causal` - 因果推理
- `spacego_network` - 网络模块（可选）
- `spacego_cuda` - CUDA 内核（可选）

---

## 五、性能指标

### 5.1 关键性能指标 (KPI)

```cpp
struct GoAIPerformanceMetrics {
    // 推理性能
    double inference_latency_ms;          // 单次推理延迟
    double inference_throughput_pos_s;    // 位置/秒
    
    // 搜索性能
    double search_nodes_per_second;       // 搜索节点/秒
    double search_depth_average;          // 平均搜索深度
    
    // GPU 性能
    double gpu_utilization_percent;       // GPU 利用率
    double memory_bandwidth_gb_s;         // 内存带宽
    double sm_efficiency_percent;         // SM 效率
};
```

### 5.2 预期性能基准

| 硬件配置 | 推理延迟 | 搜索节点/秒 | GPU 利用率 |
|---------|---------|-----------|----------|
| RTX 3060 | <5ms | >100k | >80% |
| RTX 3080 | <3ms | >200k | >85% |
| RTX 4090 | <2ms | >500k | >90% |

---

## 六、技术创新点

### 6.1 算法创新

1. **HMGS-DP 搜索算法**
   - 分层合并图搜索
   - 多样性保护机制
   - 拓扑感知 UCT

2. **CA-MOE 元决策系统**
   - 6 种专家动态路由
   - 自适应融合权重
   - 局面感知选择

3. **因果推理集成**
   - 因果链建模
   - 关键点检测
   - 高斯泼溅不确定性量化

### 6.2 工程创新

1. **模块化架构**
   - 五层架构，清晰的模块划分
   - 低耦合高内聚
   - 易于扩展维护

2. **跨平台支持**
   - Windows 优先
   - Linux 兼容
   - CMake 构建

3. **性能监控**
   - Nsight 工具集成
   - 实时性能分析
   - 自动化告警

---

## 七、总结

SpaceGo 维空弈是一个**技术先进、架构清晰、性能卓越**的现代化围棋 AI 系统。它融合了深度学习、强化学习、高性能计算三大技术领域，具有以下核心优势：

### 技术实力
✅ **高性能计算**：CUDA GPU 加速，支持 Tensor Core 和混合精度  
✅ **先进算法**：HMGS-DP + CA-MOE 混合决策系统  
✅ **深度学习**：LibTorch 集成，多种神经网络架构  
✅ **因果推理**：创新的因果链与风格迁移技术  

### 工程实践
✅ **模块化设计**：五层架构，清晰的模块划分  
✅ **跨平台构建**：CMake 支持，多环境兼容  
✅ **完整文档**：详细的技术文档和使用指南  
✅ **测试覆盖**：单元测试和集成测试  

### 创新亮点
✅ **8 分区立方体搜索**：创新的棋盘分区探索策略  
✅ **混合专家系统**：6 种专家动态路由选择  
✅ **风格化 AI**：支持多种棋风和智能体创建  
✅ **注意力可视化**：AI 决策过程可解释性  

SpaceGo 维空弈不仅是一个强大的围棋 AI，更是一个**研究平台**，为高性能计算、深度学习和博弈论的交叉研究提供了宝贵的实验环境。
