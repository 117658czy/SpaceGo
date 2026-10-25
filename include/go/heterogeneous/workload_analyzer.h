#pragma once

// ============================================================================
// 工作负载分析器 - 任务特征提取与设备亲和度计算
// 命名空间: spacego::go::heterogeneous
// 功能: 分析计算任务的特征向量（四维），计算任务对各计算设备的亲和度评分
// ============================================================================

#include "heterogeneous_types.h"

#include <memory>

namespace spacego {
namespace go {
namespace heterogeneous {

// ============================================================================
// TaskCategory - 任务类型枚举
// ============================================================================

/**
 * @brief 标识计算任务的语义类别，用于选择专用分析策略
 *
 * 不同类别的任务具有本质不同的计算特征（矩阵密度、访存模式、并行度等），
 * 通过分类可以为每种任务类型设计专用的特征分析策略，提高亲和度计算精度。
 *
 * 选择原则：
 *   - 有明确围棋语义的任务使用专用类别（评估、推理、拓扑、搜索）
 *   - 无法归类到上述四类的通用计算任务使用 GENERIC_COMPUTE
 */
enum class TaskCategory : uint8_t {
    BOARD_EVALUATION = 0,       // 棋盘评估 - 判断局面优劣，涉及大量随机棋盘状态访问
    NEURAL_INFERENCE = 1,       // 神经网络推理 - 前向传播，密集矩阵运算
    TOPOLOGY_CALCULATION = 2,   // 拓扑计算 - 棋盘连通性/气/眼位分析
    SEARCH_SIMULATION = 3,      // 搜索仿真 - 蒙特卡洛树搜索的模拟对局
    GENERIC_COMPUTE = 4         // 通用计算 - 无法归类的其他计算任务
};

// ============================================================================
// WorkloadAnalyzer - 工作负载分析器
// ============================================================================

/**
 * @brief 工作负载分析器：分析任务特征并计算设备亲和度
 *
 * 核心功能分为两个阶段：
 *   阶段一：特征提取 - 将任务描述（类别、规模、元数据）转换为四维特征向量
 *   阶段二：亲和度计算 - 将任务特征与设备能力对比，输出设备亲和度评分
 *
 * 四维特征向量定义（详见 TaskFeature 结构体）：
 *   1. matrix_density   - 矩阵/张量密度（0.0稀疏 ~ 1.0密集）
 *   2. data_scale       - 数据规模（元素数量）
 *   3. mem_pattern      - 访存模式（顺序/跨步/随机/混合）
 *   4. parallelism_degree - 内在并行度（0.0串行 ~ 1.0完全并行）
 *
 * 使用示例：
 * @code
 *   WorkloadAnalyzer analyzer;
 *   auto cap_cpu = DeviceCapability::detect_cpu();
 *   #if SPACEGO_CUDA_ENABLED
 *   auto cap_gpu = DeviceCapability::detect_gpu();
 *   #endif
 *   analyzer.update_device_capabilities(cap_cpu, cap_gpu);
 *
 *   // 分析一个批量大小为64的神经网络推理任务
 *   auto feature = analyzer.analyze_task(
 *       TaskCategory::NEURAL_INFERENCE, 64, 1, true, 4);
 *   auto affinity = analyzer.calculate_device_affinity(feature, cap_cpu, cap_gpu);
 *   // affinity.gpu_affinity_score 接近 1.0，推荐使用 GPU
 * @endcode
 */
class WorkloadAnalyzer {
public:
    // ========================================================================
    // 构造函数与析构函数
    // ========================================================================

    /**
     * @brief 构造工作负载分析器
     *
     * 初始化时使用默认的设备能力。建议在构造后调用
     * update_device_capabilities() 更新为实际检测到的设备能力。
     */
    WorkloadAnalyzer();

    /**
     * @brief 析构函数
     */
    ~WorkloadAnalyzer();

    // ========================================================================
    // 综合任务分析 - 一站式特征提取
    // ========================================================================

    /**
     * @brief 综合分析任务的四维特征
     *
     * 根据任务类别选择专用分析策略，计算完整的 TaskFeature 四维特征向量。
     * 这是最常用的入口方法，封装了所有特征计算逻辑。
     *
     * @param category       任务类别（棋盘评估/神经网络推理/拓扑计算/搜索仿真/通用计算）
     * @param batch_size     批次大小（同时处理的任务实例数量）
     *                       批次越大，GPU的并行优势越明显
     * @param element_count  每个实例处理的元素数量
     *                       如棋盘评估的361个点，张量的大小等
     * @param is_tensor_op   是否为张量/矩阵运算
     *                       张量运算具有高矩阵密度，适合Tensor Core加速
     * @param tensor_dims    张量维度数（仅在 is_tensor_op=true 时有意义）
     *                       影响数据规模的计算，维度越高数据量越大
     * @return TaskFeature   包含四维特征的任务特征结构体
     */
    TaskFeature analyze_task(TaskCategory category,
                             size_t batch_size,
                             size_t element_count,
                             bool is_tensor_op = false,
                             int tensor_dims = 0);

    // ========================================================================
    // 单独计算四维特征的方法
    // ========================================================================

    /**
     * @brief 计算任务的矩阵/张量密度
     *
     * 矩阵密度反映任务的计算模式密集程度：
     *   - 神经网络推理：权重矩阵密集，密度高(0.9)
     *   - 拓扑计算：区域计算有规律，密度中等(0.6)
     *   - 棋盘评估：棋盘状态稀疏，密度低(0.3)
     *   - 搜索仿真：树搜索分支稀疏，密度低(0.2)
     *   - 通用计算：根据 is_tensor_op 标志判断
     *
     * @param category     任务类别
     * @param batch_size   批次大小（用于微调密度值）
     * @param is_tensor_op 是否为张量运算
     * @return double      矩阵密度值，范围 [0.0, 1.0]
     */
    double compute_matrix_density(TaskCategory category,
                                  size_t batch_size,
                                  bool is_tensor_op);

    /**
     * @brief 计算任务的数据规模
     *
     * 数据规模 = 元素数量 * 批次大小 * 维度放大因子
     * 维度放大因子：对于张量操作，考虑各维度的累积效应
     *
     * @param category      任务类别（用于特定类别的规模估算）
     * @param batch_size    批次大小
     * @param element_count 每个实例的元素数量
     * @return size_t       等效数据规模（元素总数）
     */
    size_t compute_data_scale(TaskCategory category,
                              size_t batch_size,
                              size_t element_count);

    /**
     * @brief 推断任务的访存模式
     *
     * 访存模式对设备选择有重要影响：
     *   - SEQUENTIAL（顺序）：GPU合并访问高效，CPU预取器友好
     *   - STRIDED（跨步）：大跨步时GPU和CPU缓存效率均下降
     *   - RANDOM（随机）：CPU多级缓存可能优于GPU
     *   - MIXED（混合）：需要综合分析
     *
     * @param category     任务类别
     * @param is_tensor_op 是否为张量运算（张量运算倾向顺序访问）
     * @return MemoryAccessPattern 访存模式枚举值
     */
    MemoryAccessPattern compute_memory_pattern(TaskCategory category,
                                               bool is_tensor_op);

    /**
     * @brief 计算任务的内在并行度
     *
     * 内在并行度表示任务可以被并行化的程度：
     *   - 神经网络推理(大批次)：0.95（每个样本独立前向传播）
     *   - 棋盘评估：0.85（各棋盘独立评估）
     *   - 搜索仿真：0.75（树搜索分支有依赖）
     *   - 拓扑计算：0.70（区域间可能有依赖）
     *   - 小批次任务：0.50（并行度受限于批次大小）
     *
     * @param category   任务类别
     * @param batch_size 批次大小（影响并行度，批次越大并行度越高）
     * @return double    并行度值，范围 [0.0, 1.0]
     */
    double compute_parallelism_degree(TaskCategory category,
                                     size_t batch_size);

    // ========================================================================
    // 设备亲和度计算 - 核心决策算法
    // ========================================================================

    /**
     * @brief 计算任务对CPU和GPU的设备亲和度评分
     *
     * 亲和度计算的核心算法：
     *
     * GPU亲和度 = (matrix_density       * 0.40   // 矩阵密度最重要
     *            + parallelism_degree   * 0.30   // 并行度次重要
     *            + sequential_mem_bonus * 0.15   // 顺序访问奖励
     *            + tensor_core_bonus    * 0.15)  // Tensor Core奖励
     *            * data_scale_factor            // 数据规模因子
     *
     * CPU亲和度 = (1.0 - gpu_affinity)
     *            * random_mem_bonus              // 随机访问给CPU加分
     *            * small_batch_bonus             // 小批次给CPU加分
     *
     * 各奖励/惩罚因子的含义：
     *   - sequential_mem_bonus: 顺序访问模式让GPU合并访存高效
     *   - tensor_core_bonus:    GPU Tensor Core对密集矩阵的加成
     *   - random_mem_bonus:     随机访问CPU多级缓存更优
     *   - small_batch_bonus:    小批次任务不值得承受GPU启动开销
     *   - data_scale_factor:    数据量越大，GPU的并行优势越明显
     *
     * @param feature  任务的四维特征向量
     * @param cpu_cap  CPU设备能力描述
     * @param gpu_cap  GPU设备能力描述
     * @return DeviceAffinity 包含CPU/GPU亲和度评分、推荐设备和置信度
     */
    DeviceAffinity calculate_device_affinity(const TaskFeature& feature,
                                             const DeviceCapability& cpu_cap,
                                             const DeviceCapability& gpu_cap);

    // ========================================================================
    // 设备能力管理
    // ========================================================================

    /**
     * @brief 更新内部缓存的设备能力信息
     *
     * 将检测到的CPU和GPU设备能力保存到分析器内部，
     * 后续的 analyze_task 调用将自动使用这些信息进行亲和度计算。
     *
     * @param cpu_cap CPU设备能力描述
     * @param gpu_cap GPU设备能力描述
     */
    void update_device_capabilities(const DeviceCapability& cpu_cap,
                                    const DeviceCapability& gpu_cap);

private:
    // ========================================================================
    // 内部状态 - 设备能力缓存
    // ========================================================================

    DeviceCapability cpu_capability_;    // 缓存的CPU设备能力
    DeviceCapability gpu_capability_;    // 缓存的GPU设备能力

    // ========================================================================
    // 专用分析策略 - 每种任务类别的特征提取逻辑
    // ========================================================================

    /**
     * @brief 棋盘评估特征分析
     *
     * 特征概述：
     *   - 访问模式：随机（棋盘上任意位置的点都可能被访问）
     *   - 并行度：高（每块棋盘独立评估，互不依赖）
     *   - 数据规模：batch_size * 361（棋盘共361个交叉点）
     *   - 矩阵密度：低（棋盘状态是稀疏的，大部分位置为空）
     *
     * 适合GPU的场景：同时评估大量棋盘（batch_size >= 32）
     * 适合CPU的场景：评估少量棋盘，或需要复杂逻辑判断
     */
    TaskFeature analyze_board_evaluation(size_t batch_size,
                                         size_t board_count);

    /**
     * @brief 神经网络推理特征分析
     *
     * 特征概述：
     *   - 访问模式：顺序（权重矩阵按行/列顺序读取）
     *   - 并行度：极高（矩阵乘法天然适合并行）
     *   - 数据规模：batch_size * tensor_dims^2（矩阵乘法复杂度）
     *   - 矩阵密度：高（权重矩阵几乎完全密集）
     *
     * 适合GPU的场景：几乎所有情况（神经网络推理是GPU的传统优势领域）
     * 适合CPU的场景：极小模型或单次推理（batch_size < 4）
     */
    TaskFeature analyze_neural_inference(size_t batch_size,
                                         int tensor_dims);

    /**
     * @brief 拓扑计算特征分析
     *
     * 特征概述：
     *   - 访问模式：跨步（以连通区域为单位访问，跨越棋盘区域）
     *   - 并行度：中等（区域间可能有依赖，不完全独立）
     *   - 数据规模：batch_size * region_count
     *   - 矩阵密度：中等（区域计算比纯评估密集，不如矩阵乘法）
     *
     * 适合GPU的场景：大批次区域并行分析
     * 适合CPU的场景：复杂拓扑依赖关系处理
     */
    TaskFeature analyze_topology_calculation(size_t batch_size,
                                             size_t region_count);

    /**
     * @brief 搜索仿真特征分析
     *
     * 特征概述：
     *   - 访问模式：混合（有规则的走子生成，也有随机的胜率回溯）
     *   - 并行度：中高（树搜索分支可并行，但叶节点评估需要汇总）
     *   - 数据规模：batch_size * 仿真步数
     *   - 矩阵密度：低（树结构天然稀疏）
     *
     * 适合GPU的场景：大量独立仿真并行执行
     * 适合CPU的场景：需要维护全局树结构的搜索
     */
    TaskFeature analyze_search_simulation(size_t batch_size);

    /**
     * @brief 通用计算特征分析
     *
     * 特征概述：
     *   - 访问模式：根据 is_tensor_op 判断（张量->顺序，否则->混合）
     *   - 并行度：中等（通用任务保守估算）
     *   - 数据规模：batch_size * element_count
     *   - 矩阵密度：根据 is_tensor_op 判断（张量->高，否则->中等）
     *
     * 适合GPU的场景：大规模数据并行处理
     * 适合CPU的场景：逻辑复杂的通用计算
     */
    TaskFeature analyze_generic_compute(size_t batch_size,
                                        size_t element_count,
                                        bool is_tensor_op = false);

    // ========================================================================
    // 辅助估算函数
    // ========================================================================

    /**
     * @brief 估算GPU内核启动开销（微秒级）
     *
     * GPU启动一个CUDA内核的开销通常在5-50微秒之间，取决于：
     *   - 内核参数的复杂度和数量
     *   - 驱动程序的调度延迟
     *   - PCIe总线上的命令传输延迟
     *
     * 批次大小越大，平摊到每个元素的启动开销越小。
     *
     * 估算公式：
     *   基础开销 20us + 批次相关开销 batch_size * 0.1us
     *   对于大批次，上限设为 100us（驱动程序优化）
     *
     * @param batch_size 批次大小
     * @return double    GPU启动开销估算值（微秒）
     */
    double estimate_gpu_launch_overhead(size_t batch_size) const;

    // ========================================================================
    // 阈值常量 - 用于分类决策的关键分界点
    // ========================================================================

    static constexpr size_t GPU_MIN_BATCH_THRESHOLD = 16;     // GPU最小批次阈值
                                                               // 批次小于此值的任务不适合GPU
                                                               // 因为GPU启动开销 > 并行收益

    static constexpr size_t LARGE_TENSOR_THRESHOLD = 1024;    // 大张量阈值
                                                               // 张量维度超过此值认为是大张量
                                                               // 大张量运算GPU优势极为明显

    static constexpr size_t SMALL_BATCH_THRESHOLD = 8;        // 小批次阈值
                                                               // 批次小于此值CPU更优
                                                               // 避免GPU启动开销占比过高

    static constexpr double DEFAULT_GPU_LAUNCH_US = 20.0;     // GPU内核默认启动开销（微秒）
    static constexpr double MAX_GPU_LAUNCH_US = 100.0;        // GPU内核最大启动开销（微秒）
};

}  // namespace heterogeneous
}  // namespace go
}  // namespace spacego