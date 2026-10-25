#pragma once

// ============================================================================
// 异构计算引擎 - 任务分解器 (TaskDecomposer)
// 命名空间: spacego::go::heterogeneous
// 功能: 将大规模计算任务按设备能力和任务特征拆分为多个子任务，
//       实现CPU和GPU间的细粒度负载均衡
// ============================================================================

#include "heterogeneous_types.h"

#include <cstddef>
#include <utility>
#include <vector>

namespace spacego {
namespace go {
namespace heterogeneous {

// ============================================================================
// DecompositionResult - 任务分解结果结构体
// ============================================================================

/**
 * @brief 描述任务分解后的结果，包含CPU和GPU两路的子任务分配
 *
 * 该结构体是 TaskDecomposer 所有分解方法的统一返回类型，
 * 调度器根据此结果将子任务分发到不同设备队列中。
 *
 * 关键设计：
 *   - cpu_ratio + gpu_ratio 恒等于 1.0
 *   - cpu_subtasks 和 gpu_subtasks 分别对应两种设备的子任务列表
 *   - total_subtask_count = cpu_subtasks.size() + gpu_subtasks.size()
 */
struct DecompositionResult {
    std::vector<SubTask> cpu_subtasks;          // 分配给CPU的子任务列表
                                                // 每个子任务包含数据偏移、长度和优先级
    std::vector<SubTask> gpu_subtasks;          // 分配给GPU的子任务列表
                                                // GPU子任务通常使用更大的数据块以提高吞吐
    size_t total_subtask_count;                 // 子任务总数 (cpu + gpu)
    double cpu_ratio;                           // CPU分配比例 (0.0 ~ 1.0)
                                                // 基于元素数量计算的比例值
    double gpu_ratio;                           // GPU分配比例 (0.0 ~ 1.0)
                                                // cpu_ratio + gpu_ratio = 1.0

    // ========================================================================
    // 默认构造函数 - 初始化为空分解结果
    // ========================================================================
    DecompositionResult()
        : cpu_subtasks()
        , gpu_subtasks()
        , total_subtask_count(0)
        , cpu_ratio(0.0)
        , gpu_ratio(0.0)
    {
    }
};

// ============================================================================
// TaskDecomposer - 任务分解器类
// ============================================================================

/**
 * @brief 围棋AI异构计算的任务分解器
 *
 * 核心职责：
 *   1. 将大规模计算任务（棋盘评估、神经网络推理、搜索仿真、拓扑计算）
 *      按设备能力和任务特征拆分为CPU和GPU两路的子任务
 *   2. 根据设备能力估算最优的CPU/GPU分配比例
 *   3. 确保子任务粒度合理，既不过度拆分（调度开销大）也不过于粗放（负载不均）
 *
 * 设计原则：
 *   - 所有分解方法最终委托给 decompose_by_batch 统一处理
 *   - 各层方法负责特定于任务类型的策略调整（如神经网络推理的小批次偏向CPU）
 *   - 分解粒度受 max_subtask_count_ 约束，避免子任务数量爆炸
 *
 * 使用示例：
 * @code
 *   TaskDecomposer decomposer(64);  // 最多64个子任务
 *
 *   // 棋盘评估分解：10000个棋盘，CPU评估3000个，GPU评估7000个
 *   auto result = decomposer.decompose_board_evaluation(10000, 3000, 7000);
 *
 *   // 最优分配估算
 *   auto cpu_cap = DeviceCapability::detect_cpu();
 *   auto gpu_cap = DeviceCapability::detect_gpu(0);
 *   auto optimal = decomposer.estimate_optimal_split(10000, cpu_cap, gpu_cap, affinity);
 * @endcode
 */
class TaskDecomposer {
public:
    // ========================================================================
    // 构造函数与析构函数
    // ========================================================================

    /**
     * @brief 默认构造函数 - 使用默认最大子任务数 (64)
     */
    TaskDecomposer();

    /**
     * @brief 带参构造函数 - 指定最大子任务数上限
     *
     * @param max_subtask_count 允许的最大子任务总数（CPU + GPU）
     *                         合理范围：8 ~ 256
     *                         过小导致负载不均，过大导致调度开销增加
     */
    explicit TaskDecomposer(size_t max_subtask_count);

    /**
     * @brief 析构函数
     */
    ~TaskDecomposer();

    // 禁止拷贝（内部状态复杂，不支持浅拷贝语义）
    TaskDecomposer(const TaskDecomposer&) = delete;
    TaskDecomposer& operator=(const TaskDecomposer&) = delete;

    // ========================================================================
    // 公共分解接口 - 按任务类型分类
    // ========================================================================

    /**
     * @brief 通用批次维度分解（核心方法）
     *
     * 按CPU和GPU各自分配的元素数量创建对应的子任务。
     * 每个子任务分配连续的数据范围，数据偏移从0开始递增。
     * CPU子任务和GPU子任务分别给出递增的子任务ID。
     *
     * 数据布局约定：
     *   - 数据缓冲区中 [0, cpu_elements) 区域分配给CPU处理
     *   - 数据缓冲区中 [cpu_elements, cpu_elements + gpu_elements) 分配给GPU处理
     *   - 调用者负责确保 total_elements == cpu_elements + gpu_elements
     *
     * @param total_elements  待处理的总元素数量
     * @param cpu_elements    分配给CPU处理的元素数量
     * @param gpu_elements    分配给GPU处理的元素数量
     * @return DecompositionResult 包含CPU和GPU子任务列表的分解结果
     */
    DecompositionResult decompose_by_batch(size_t total_elements,
                                           size_t cpu_elements,
                                           size_t gpu_elements);

    /**
     * @brief 棋盘评估任务分解
     *
     * 将一批棋盘评估任务按指定比例分配给CPU和GPU。
     * CPU子任务使用传统GoBoardCPU评估逻辑（适合串行+复杂局面判断），
     * GPU子任务使用GoBoardCUDA批量评估（适合大量并行局面评估）。
     *
     * 典型使用场景：MCTS搜索中需要对多个候选局面进行同时评估。
     *
     * @param total_boards  待评估的围棋棋盘总数
     * @param cpu_boards    分配给CPU评估的棋盘数
     * @param gpu_boards    分配给GPU评估的棋盘数
     * @return DecompositionResult 包含CPU和GPU子任务列表的分解结果
     */
    DecompositionResult decompose_board_evaluation(size_t total_boards,
                                                    size_t cpu_boards,
                                                    size_t gpu_boards);

    /**
     * @brief 神经网络推理任务分解
     *
     * 将批量的神经网络推理请求分配给CPU和GPU。
     *
     * 特殊策略：
     *   - 小批次（batch_size < 16）：全部分配给CPU，避免GPU启动开销大于计算收益
     *   - 大批次：按指定比例分配，GPU子任务标记为HIGH优先级（GPU推理延迟更低）
     *
     * 支持的推理场景：
     *   - 策略网络（Policy Network）前向传播
     *   - 价值网络（Value Network）局面评估
     *   - CA-MOE混合专家网络路由推理
     *
     * @param total_batch_size  神经网络推理的总批次大小
     * @param cpu_batch_size    分配给CPU推理的批次大小
     * @param gpu_batch_size    分配给GPU推理的批次大小
     * @return DecompositionResult 包含CPU和GPU子任务列表的分解结果
     */
    DecompositionResult decompose_neural_inference(size_t total_batch_size,
                                                    size_t cpu_batch_size,
                                                    size_t gpu_batch_size);

    /**
     * @brief 搜索仿真任务分解
     *
     * 将MCTS/HMGS-DP搜索中的仿真任务分配给CPU和GPU。
     *
     * CPU子任务使用完整的CPU棋盘+搜索逻辑（适合复杂局面、分支多的情况），
     * GPU子任务使用CUDA加速的批量搜索仿真（适合大量快速推演）。
     *
     * @param total_simulations  搜索仿真总数
     * @param cpu_simulations    分配给CPU执行的仿真数
     * @param gpu_simulations    分配给GPU执行的仿真数
     * @return DecompositionResult 包含CPU和GPU子任务列表的分解结果
     */
    DecompositionResult decompose_simulation(size_t total_simulations,
                                              size_t cpu_simulations,
                                              size_t gpu_simulations);

    /**
     * @brief 拓扑计算任务分解
     *
     * 将围棋拓扑（区域、连通性、气数等）计算任务分配给CPU和GPU。
     *
     * 拓扑计算具有以下特征，因此默认偏向CPU：
     *   - 数据依赖性强，不同区块间有复杂的依赖关系
     *   - 计算密度相对较低，分支预测多
     *   - CPU的缓存层次结构和分支预测能力更适合此类任务
     *
     * 但如果用户显式指定了较大的GPU比例（如gpu_regions > cpu_regions），
     * 则按用户指定的比例分解。
     *
     * @param total_regions  待计算的围棋拓扑区域总数
     * @param cpu_regions    分配给CPU计算的区域数
     * @param gpu_regions    分配给GPU计算的区域数
     * @return DecompositionResult 包含CPU和GPU子任务列表的分解结果
     */
    DecompositionResult decompose_topology_calculation(size_t total_regions,
                                                        size_t cpu_regions,
                                                        size_t gpu_regions);

    /**
     * @brief 基于设备能力和任务特征估算最优分解粒度
     *
     * 综合考量以下因素计算最优的CPU/GPU分配比例：
     *   1. 设备理论计算能力比值（compute_power_tflops）
     *   2. 设备内存带宽比值（memory_bandwidth_gb_per_sec）
     *   3. 任务亲和度（DeviceAffinity）
     *   4. 当前可用内存容量
     *
     * 估算公式：
     *   原始CPU比例 = CPU_TFLOPS / (CPU_TFLOPS + GPU_TFLOPS)
     *   亲和度调整 = 原始比例 * (CPU亲和度 / GPU亲和度) 的归一化权重
     *   带宽约束   = 如果GPU带宽远高于CPU，适当增加GPU分配
     *   内存约束   = 如果某设备可用内存不足，减少其分配
     *
     * @param total_elements  待处理的总元素数量
     * @param cpu_cap         CPU设备能力信息
     * @param gpu_cap         GPU设备能力信息
     * @param affinity        任务对CPU和GPU的亲和度评估
     * @return std::pair<size_t, size_t> {建议CPU元素数, 建议GPU元素数}
     */
    std::pair<size_t, size_t> estimate_optimal_split(
        size_t total_elements,
        const DeviceCapability& cpu_cap,
        const DeviceCapability& gpu_cap,
        const DeviceAffinity& affinity);

    // ========================================================================
    // 属性访问器
    // ========================================================================

    /**
     * @brief 设置最大子任务数量上限
     *
     * 子任务总数（CPU + GPU）不会超过此值。
     * 如果设置的值不合理（< 2），会自动调整为2以保证至少CPU和GPU各一个子任务。
     *
     * @param count 新的最大子任务数上限
     */
    void set_max_subtask_count(size_t count);

    /**
     * @brief 获取当前最大子任务数量上限
     *
     * @return size_t 最大子任务数上限
     */
    size_t get_max_subtask_count() const;

private:
    // ========================================================================
    // 内部辅助方法
    // ========================================================================

    /**
     * @brief 创建一个子任务实例
     *
     * 统一创建子任务，确保所有子任务具有一致的初始化逻辑。
     *
     * @param subtask_id   子任务唯一ID
     * @param device       目标计算设备类型
     * @param data_offset  在数据缓冲区中的起始偏移
     * @param data_length  子任务处理的数据长度（元素数）
     * @param priority     调度优先级（默认NORMAL）
     * @return SubTask 新创建的子任务实例
     */
    SubTask create_subtask(size_t subtask_id,
                           ComputeDeviceType device,
                           size_t data_offset,
                           size_t data_length,
                           SubTaskPriority priority = SubTaskPriority::NORMAL);

    /**
     * @brief 基于设备计算能力比例估算CPU可处理的元素数量
     *
     * 使用TFLOPS比值作为主要权重，结合内存带宽进行微调。
     *
     * @param cpu_cap         CPU设备能力
     * @param gpu_cap         GPU设备能力
     * @param total_elements  总元素数量
     * @return size_t CPU应处理的元素数估算值
     */
    size_t estimate_cpu_capacity(const DeviceCapability& cpu_cap,
                                  const DeviceCapability& gpu_cap,
                                  size_t total_elements);

    /**
     * @brief 基于设备计算能力比例估算GPU可处理的元素数量
     *
     * 使用TFLOPS比值作为主要权重，结合内存带宽进行微调。
     *
     * @param cpu_cap         CPU设备能力
     * @param gpu_cap         GPU设备能力
     * @param total_elements  总元素数量
     * @return size_t GPU应处理的元素数估算值
     */
    size_t estimate_gpu_capacity(const DeviceCapability& cpu_cap,
                                  const DeviceCapability& gpu_cap,
                                  size_t total_elements);

    // ========================================================================
    // 成员变量
    // ========================================================================

    /**
     * @brief 最大子任务数量上限
     *
     * 限制一次分解产生的子任务总数（CPU + GPU），
     * 防止过度拆分导致调度开销淹没计算收益。
     *
     * 默认值：64（经验值，在大多数硬件配置下提供良好的粒度）
     */
    size_t max_subtask_count_;
};

}  // namespace heterogeneous
}  // namespace go
}  // namespace spacego