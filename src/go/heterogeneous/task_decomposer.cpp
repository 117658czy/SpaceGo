// ============================================================================
// 异构计算引擎 - 任务分解器实现 (TaskDecomposer)
// 命名空间: spacego::go::heterogeneous
// 功能: 实现各类围棋AI计算任务的CPU/GPU分解逻辑
// ============================================================================
//
// 核心算法说明：
//   decompose_by_batch() 是所有分解方法的底层实现，采用以下策略：
//     1. 按CPU和GPU各自的元素数量计算应分配的子任务数（按比例分配）
//     2. 将CPU/GPU数据区间均匀切分为指定数量的子任务
//     3. 前 (remainder) 个子任务多分配一个元素，确保总数精确
//     4. 每个子任务分配连续的数据偏移范围
//
//   estimate_optimal_split() 基于四因子模型：
//     F1: 设备理论计算能力比值 (TFLOPS)
//     F2: 任务亲和度调整 (DeviceAffinity)
//     F3: 内存带宽约束 (Memory Bandwidth)
//     F4: 可用内存容量限制 (Available Memory)
// ============================================================================

#include "go/heterogeneous/task_decomposer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#undef max
#undef min

namespace spacego {
namespace go {
namespace heterogeneous {

// ============================================================================
// 常量定义
// ============================================================================

// 默认最大子任务数（CPU + GPU 总和）
constexpr size_t kDefaultMaxSubtaskCount = 64;

// 最小合理的最大子任务数（至少保证CPU和GPU各一个子任务）
constexpr size_t kMinReasonableSubtaskCount = 2;

// 神经网络推理的小批次阈值：低于此值的批次全部在CPU执行
constexpr size_t kNeuralSmallBatchThreshold = 16;

// 内存约束估算：假设每个数据元素平均占用 1 KB 内存
constexpr size_t kEstimatedBytesPerElement = 1024;

// 亲和度融合权重：50% 来自计算能力比值，50% 来自亲和度分数
constexpr double kAffinityBlendWeight = 0.5;

// 带宽调整因子：GPU带宽优势的权重
constexpr double kBandwidthAdjustBase = 0.7;
constexpr double kBandwidthAdjustScale = 0.3;

// 拓扑计算默认CPU比例：拓扑计算更偏向CPU
constexpr double kTopologyDefaultCPURatio = 0.70;

// ============================================================================
// TaskDecomposer 构造函数与析构函数
// ============================================================================

TaskDecomposer::TaskDecomposer()
    : max_subtask_count_(kDefaultMaxSubtaskCount)
{
    // 默认使用64作为最大子任务数，这是在大多数硬件配置下
    // 经过调优的经验值，能平衡调度开销和负载均衡粒度
}

TaskDecomposer::TaskDecomposer(size_t max_subtask_count)
    : max_subtask_count_(max_subtask_count)
{
    // 参数合法性检查：如果传入的值不合理（< 2），自动修正
    if (max_subtask_count_ < kMinReasonableSubtaskCount) {
        max_subtask_count_ = kMinReasonableSubtaskCount;
    }
}

TaskDecomposer::~TaskDecomposer()
{
    // 无动态资源需要释放，使用默认析构即可
}

// ============================================================================
// 属性访问器
// ============================================================================

void TaskDecomposer::set_max_subtask_count(size_t count)
{
    // 合法性检查：不允许设置为小于2的值
    if (count < kMinReasonableSubtaskCount) {
        max_subtask_count_ = kMinReasonableSubtaskCount;
    } else {
        max_subtask_count_ = count;
    }
}

size_t TaskDecomposer::get_max_subtask_count() const
{
    return max_subtask_count_;
}

// ============================================================================
// create_subtask - 内部辅助方法：创建单个子任务
// ============================================================================

SubTask TaskDecomposer::create_subtask(
    size_t subtask_id,
    ComputeDeviceType device,
    size_t data_offset,
    size_t data_length,
    SubTaskPriority priority)
{
    SubTask task;
    task.subtask_id    = subtask_id;      // 子任务唯一ID
    task.target_device = device;          // 目标执行设备（CPU或GPU）
    task.priority      = priority;        // 调度优先级
    task.status        = SubTaskStatus::PENDING;  // 初始状态：待处理
    task.data_offset   = data_offset;     // 在父任务缓冲区中的起始偏移
    task.data_length   = data_length;     // 该子任务处理的元素数量
    // dependencies 留空：默认无前置依赖，可被立即调度
    // submit_time 在构造时由 SubTask 构造函数自动设置
    return task;
}

// ============================================================================
// decompose_by_batch - 核心分解方法
// ============================================================================

DecompositionResult TaskDecomposer::decompose_by_batch(
    size_t total_elements,
    size_t cpu_elements,
    size_t gpu_elements)
{
    DecompositionResult result;

    // ---- 边界条件处理 ----
    // 如果总元素数为0，返回空结果
    if (total_elements == 0) {
        result.total_subtask_count = 0;
        result.cpu_ratio = 0.0;
        result.gpu_ratio = 0.0;
        return result;
    }

    // 参数合法性检查：CPU元素 + GPU元素 应等于总元素数
    // 在Debug模式下列出警告，Release模式下静默修正
    if (cpu_elements + gpu_elements != total_elements) {
        // 如果参数不一致，以总元素数为准，按原比例重新分配
        double total = static_cast<double>(cpu_elements + gpu_elements);
        if (total > 0.0) {
            double cpu_proportion = static_cast<double>(cpu_elements) / total;
            cpu_elements = static_cast<size_t>(
                static_cast<double>(total_elements) * cpu_proportion
            );
            gpu_elements = total_elements - cpu_elements;
        } else {
            // 如果两个都是0但total_elements > 0，均分
            cpu_elements = total_elements / 2;
            gpu_elements = total_elements - cpu_elements;
        }
    }

    // ---- 计算CPU和GPU的分配比例 ----
    double total_d = static_cast<double>(total_elements);
    result.cpu_ratio = static_cast<double>(cpu_elements) / total_d;
    result.gpu_ratio = static_cast<double>(gpu_elements) / total_d;

    // ---- 计算各设备的子任务数量 ----
    // 策略：按CPU和GPU的元素比例分配子任务槽位
    // 确保每个有元素的设备至少分配1个子任务

    size_t cpu_subtask_count = 0;
    size_t gpu_subtask_count = 0;

    if (cpu_elements > 0 && gpu_elements > 0) {
        // 两个设备都有元素：按比例分配子任务数
        cpu_subtask_count = static_cast<size_t>(
            std::round(result.cpu_ratio * static_cast<double>(max_subtask_count_))
        );
        // 确保至少各1个子任务
        if (cpu_subtask_count < 1) {
            cpu_subtask_count = 1;
        }
        gpu_subtask_count = max_subtask_count_ - cpu_subtask_count;
        if (gpu_subtask_count < 1) {
            gpu_subtask_count = 1;
            // 如果总子任务数超限，削减CPU子任务数
            if (cpu_subtask_count + gpu_subtask_count > max_subtask_count_) {
                cpu_subtask_count = max_subtask_count_ - gpu_subtask_count;
            }
        }
    } else if (cpu_elements > 0) {
        // 仅CPU有元素
        cpu_subtask_count = max_subtask_count_;
        gpu_subtask_count = 0;
    } else if (gpu_elements > 0) {
        // 仅GPU有元素
        cpu_subtask_count = 0;
        gpu_subtask_count = max_subtask_count_;
    } else {
        // 两个都没有元素（理论上不会到达此处，但安全处理）
        cpu_subtask_count = 0;
        gpu_subtask_count = 0;
    }

    // ---- 创建CPU子任务 ----
    // CPU子任务覆盖数据范围：[0, cpu_elements)
    // 将cpu_elements个元素均匀分配给cpu_subtask_count个子任务
    if (cpu_subtask_count > 0 && cpu_elements > 0) {
        size_t cpu_chunk_size = cpu_elements / cpu_subtask_count;   // 每个子任务的基础大小
        size_t cpu_remainder  = cpu_elements % cpu_subtask_count;   // 余数分配给前几个子任务

        size_t current_offset = 0;  // 当前数据偏移位置（CPU数据从0开始）
        for (size_t i = 0; i < cpu_subtask_count; ++i) {
            // 前 cpu_remainder 个子任务多分配1个元素，确保总数精确
            size_t this_chunk = cpu_chunk_size + (i < cpu_remainder ? 1 : 0);

            SubTask task = create_subtask(
                i,                              // subtask_id: CPU子任务从0开始编号
                ComputeDeviceType::CPU,          // 目标设备：CPU
                current_offset,                  // 数据偏移
                this_chunk,                      // 数据长度（元素数）
                SubTaskPriority::NORMAL          // CPU子任务默认普通优先级
            );

            result.cpu_subtasks.push_back(std::move(task));
            current_offset += this_chunk;
        }
    }

    // ---- 创建GPU子任务 ----
    // GPU子任务覆盖数据范围：[cpu_elements, cpu_elements + gpu_elements)
    // 将gpu_elements个元素均匀分配给gpu_subtask_count个子任务
    if (gpu_subtask_count > 0 && gpu_elements > 0) {
        size_t gpu_chunk_size = gpu_elements / gpu_subtask_count;   // 每个子任务的基础大小
        size_t gpu_remainder  = gpu_elements % gpu_subtask_count;   // 余数分配给前几个子任务

        // GPU子任务的数据偏移从cpu_elements开始（CPU数据之后）
        size_t current_offset = cpu_elements;
        for (size_t i = 0; i < gpu_subtask_count; ++i) {
            // 前 gpu_remainder 个子任务多分配1个元素
            size_t this_chunk = gpu_chunk_size + (i < gpu_remainder ? 1 : 0);

            SubTask task = create_subtask(
                // GPU子任务ID从 cpu_subtask_count 开始编号，确保全局唯一
                cpu_subtask_count + i,
                ComputeDeviceType::GPU,          // 目标设备：GPU
                current_offset,                  // 数据偏移（紧接CPU数据之后）
                this_chunk,                      // 数据长度（元素数）
                SubTaskPriority::NORMAL          // GPU子任务默认普通优先级
            );

            result.gpu_subtasks.push_back(std::move(task));
            current_offset += this_chunk;
        }
    }

    // ---- 统计结果汇总 ----
    result.total_subtask_count = result.cpu_subtasks.size() + result.gpu_subtasks.size();

    return result;
}

// ============================================================================
// decompose_board_evaluation - 棋盘评估任务分解
// ============================================================================

DecompositionResult TaskDecomposer::decompose_board_evaluation(
    size_t total_boards,
    size_t cpu_boards,
    size_t gpu_boards)
{
    // 棋盘评估任务的特点：
    //   - CPU评估适合复杂局面、需要深层分析的个案
    //   - GPU评估适合大批量并行评估，每块评估计算密集但高度并行
    //
    // 委托给通用的 decompose_by_batch 方法处理，
    // 子任务创建时使用NORMAL优先级（评估任务不是最关键路径）
    //
    // 实际使用中，调度器可以根据局面复杂度动态调整：
    //   复杂局面 -> CPU（更好的分支预测和复杂逻辑处理）
    //   简单局面 -> GPU（批量并行评估效率高）

    DecompositionResult result = decompose_by_batch(total_boards, cpu_boards, gpu_boards);

    // 棋盘评估特有的后处理：为CPU子任务设置合适的数据偏移
    // （decompose_by_batch 已经正确设置了偏移，此处的返回结果直接可用）
    // 可根据需要在此添加评估任务特有的元数据标记

    return result;
}

// ============================================================================
// decompose_neural_inference - 神经网络推理任务分解
// ============================================================================

DecompositionResult TaskDecomposer::decompose_neural_inference(
    size_t total_batch_size,
    size_t cpu_batch_size,
    size_t gpu_batch_size)
{
    // ---- 神经网络推理特有的策略调整 ----

    // 策略1：小批次优化
    // 如果总批次小于阈值（16），GPU的kernel启动开销将大于计算收益，
    // 此时将所有数据分配给CPU执行
    if (total_batch_size < kNeuralSmallBatchThreshold) {
        // 小批次全部交给CPU处理
        DecompositionResult result = decompose_by_batch(
            total_batch_size,       // 总元素数
            total_batch_size,       // CPU处理全部
            0                       // GPU不参与
        );
        // 设置CPU子任务为HIGH优先级（虽然是CPU执行，但推理任务通常比较关键）
        for (auto& task : result.cpu_subtasks) {
            task.priority = SubTaskPriority::HIGH;
        }
        return result;
    }

    // 策略2：大批次按比例分配
    // GPU的Tensor Core在大批次推理时吞吐远超CPU，因此GPU子任务标记为HIGH优先级
    DecompositionResult result = decompose_by_batch(
        total_batch_size, cpu_batch_size, gpu_batch_size
    );

    // GPU子任务设置HIGH优先级，因为GPU推理延迟更低，应优先调度
    for (auto& task : result.gpu_subtasks) {
        task.priority = SubTaskPriority::HIGH;
    }

    // CPU子任务保持NORMAL优先级（decompose_by_batch默认已设为NORMAL）
    // 不需要额外修改

    return result;
}

// ============================================================================
// decompose_simulation - 搜索仿真任务分解
// ============================================================================

DecompositionResult TaskDecomposer::decompose_simulation(
    size_t total_simulations,
    size_t cpu_simulations,
    size_t gpu_simulations)
{
    // 搜索仿真任务的特点：
    //   - CPU仿真：适合需要复杂局面判断的深度仿真
    //   - GPU仿真：适合大量快速推演的批量仿真
    //
    // 搜索仿真直接影响走子决策质量，属于关键路径任务。
    // 但优先级已在调度器层面控制，此处保持默认NORMAL即可。

    DecompositionResult result = decompose_by_batch(
        total_simulations, cpu_simulations, gpu_simulations
    );

    // 搜索仿真的子任务可以标记依赖关系：
    //   如果有多个搜索树并行，可在此设置dependencies字段，
    //   确保某些仿真结果被合并后才进行后续评估。
    //   当前实现不设置显式依赖，由调度器统一管理。

    return result;
}

// ============================================================================
// decompose_topology_calculation - 拓扑计算任务分解
// ============================================================================

DecompositionResult TaskDecomposer::decompose_topology_calculation(
    size_t total_regions,
    size_t cpu_regions,
    size_t gpu_regions)
{
    // 拓扑计算的特点：
    //   - 围棋拓扑（区域连通性、气数计算）数据依赖性很强
    //   - CPU的缓存层次结构和分支预测能力更适合此类计算
    //   - 如果调用者显式指定了分配比例，则尊重调用者的选择
    //
    // 默认策略：CPU占70%，GPU占30%（kTopologyDefaultCPURatio）

    // 如果总区域数较小（< 32），全部交给CPU处理更高效
    if (total_regions < 32) {
        DecompositionResult result = decompose_by_batch(
            total_regions, total_regions, 0
        );
        // 小规模拓扑计算CPU子任务HIGH优先级
        for (auto& task : result.cpu_subtasks) {
            task.priority = SubTaskPriority::HIGH;
        }
        return result;
    }

    // 如果调用者两个参数都为0，使用默认CPU偏向比例自动分配
    if (cpu_regions == 0 && gpu_regions == 0) {
        cpu_regions = static_cast<size_t>(
            static_cast<double>(total_regions) * kTopologyDefaultCPURatio
        );
        // 确保CPU至少处理1个区域
        if (cpu_regions < 1 && total_regions > 0) {
            cpu_regions = 1;
        }
        gpu_regions = total_regions - cpu_regions;
    }

    // 注意：如果调用者显式指定了GPU比例异常高于CPU（> 60%），
    // 仍然尊重调用者的选择——拓扑计算对GPU不友好，但调用者可能有
    // 特殊的批量处理场景（如几千个简单区域的连通性GPU批量计算）。
    // 不做自动修正，由调用者自行承担性能后果。

    // 使用调用者指定（或默认计算）的比例进行分解
    DecompositionResult result = decompose_by_batch(
        total_regions, cpu_regions, gpu_regions
    );

    // 拓扑计算的CPU子任务可以适当提高优先级
    //（拓扑信息是后续推理和评估的基础，应尽早完成）
    for (auto& task : result.cpu_subtasks) {
        task.priority = SubTaskPriority::HIGH;
    }

    return result;
}

// ============================================================================
// estimate_optimal_split - 最优分解粒度估算
// ============================================================================

std::pair<size_t, size_t> TaskDecomposer::estimate_optimal_split(
    size_t total_elements,
    const DeviceCapability& cpu_cap,
    const DeviceCapability& gpu_cap,
    const DeviceAffinity& affinity)
{
    // ---- 边界条件：总元素为0 ----
    if (total_elements == 0) {
        return {0, 0};
    }

    // ---- 边界条件：GPU不可用 ----
    // 如果GPU计算能力为0（未检测到或不可用），全部分配给CPU
    if (gpu_cap.compute_power_tflops <= 0.0 || gpu_cap.compute_cores == 0) {
        return {total_elements, 0};
    }

    // ---- 边界条件：CPU不可用 ----
    // 如果CPU计算能力为0（极不可能但需防御），全部分配给GPU
    if (cpu_cap.compute_power_tflops <= 0.0 || cpu_cap.compute_cores == 0) {
        return {0, total_elements};
    }

    // ========================================================================
    // 因子1：基于理论计算能力的原始分配比例
    // ========================================================================
    // 公式：原始CPU比例 = CPU_TFLOPS / (CPU_TFLOPS + GPU_TFLOPS)
    //
    // 例如：CPU 0.5 TFLOPS, GPU 10 TFLOPS
    //       raw_cpu_ratio = 0.5 / 10.5 ~= 0.048 (约5%给CPU)
    double total_tflops = cpu_cap.compute_power_tflops + gpu_cap.compute_power_tflops;
    double raw_cpu_ratio = cpu_cap.compute_power_tflops / total_tflops;

    // ========================================================================
    // 因子2：基于任务亲和度的比例调整
    // ========================================================================
    // 亲和度反映了任务本身的特性（稠密度、并行度、访存模式等）更适合哪个设备。
    // 如果任务对CPU的亲和度远高于GPU，应增加CPU分配比例。
    //
    // 调整策略：计算能力比值和亲和度比值各占50%权重进行融合
    double adjusted_cpu_ratio = raw_cpu_ratio;

    double cpu_aff = affinity.cpu_affinity_score;
    double gpu_aff = affinity.gpu_affinity_score;
    double total_aff = cpu_aff + gpu_aff;

    if (total_aff > 0.0) {
        // 基于亲和度的CPU比例
        double affinity_cpu_ratio = cpu_aff / total_aff;

        // 融合计算能力比值和亲和度比值（加权平均）
        adjusted_cpu_ratio = kAffinityBlendWeight * raw_cpu_ratio
                           + (1.0 - kAffinityBlendWeight) * affinity_cpu_ratio;
    }
    // 如果total_aff为0（两者亲和度都未设置），仅使用原始计算能力比值

    // ========================================================================
    // 因子3：内存带宽约束调整
    // ========================================================================
    // 内存带宽影响大规模数据传输效率。
    // 如果GPU的HBM/HBM2带宽远高于CPU的DDR4带宽，
    // GPU可以在单位时间内处理更多数据，应增加GPU分配。
    //
    // 例如：CPU 40 GB/s, GPU 360 GB/s
    //       bw_ratio = 40/360 ~= 0.111, adjustment = 0.7 + 0.3*0.111 = 0.733
    //       这将减少CPU比例约27%
    if (cpu_cap.memory_bandwidth_gb_per_sec > 0
        && gpu_cap.memory_bandwidth_gb_per_sec > 0) {

        double bw_ratio = static_cast<double>(cpu_cap.memory_bandwidth_gb_per_sec)
                        / static_cast<double>(gpu_cap.memory_bandwidth_gb_per_sec);

        // 带宽调整因子：GPU带宽越高，CPU比例越低
        // 范围：(kBandwidthAdjustBase, kBandwidthAdjustBase + kBandwidthAdjustScale] = (0.7, 1.0]
        double bw_adjustment = kBandwidthAdjustBase
                             + kBandwidthAdjustScale * bw_ratio;

        // 应用带宽调整
        adjusted_cpu_ratio *= bw_adjustment;
    }
    // 如果任一设备的带宽信息不可用，跳过带宽调整

    // ========================================================================
    // 因子4：可用内存容量约束
    // ========================================================================
    // 确保分配到每个设备的元素数量不超过其可用内存能承载的上限。
    // 估算每个元素占用 kEstimatedBytesPerElement 字节的内存。
    //
    // 注意：这不是精确的内存需求计算，而是启发式估算。
    // 实际任务可能有不同的内存需求，调用者可通过调整total_elements来间接控制。

    size_t cpu_max_elements = 0;
    if (kEstimatedBytesPerElement > 0 && cpu_cap.available_memory_mb > 0) {
        cpu_max_elements = (cpu_cap.available_memory_mb * 1024 * 1024)
                         / kEstimatedBytesPerElement;
    } else {
        // 如果无法获取可用内存，不施加限制
        cpu_max_elements = total_elements;
    }

    size_t gpu_max_elements = 0;
    if (kEstimatedBytesPerElement > 0 && gpu_cap.available_memory_mb > 0) {
        gpu_max_elements = (gpu_cap.available_memory_mb * 1024 * 1024)
                         / kEstimatedBytesPerElement;
    } else {
        gpu_max_elements = total_elements;
    }

    // ---- 最终计算 ----
    // 将调整后的CPU比例转换为具体元素数量
    size_t cpu_result = static_cast<size_t>(
        static_cast<double>(total_elements) * adjusted_cpu_ratio
    );

    // 钳位：确保CPU分配数不超过可用内存约束
    if (cpu_result > cpu_max_elements) {
        cpu_result = cpu_max_elements;
    }

    // 钳位：确保CPU分配数在有效范围内 [0, total_elements]
    if (cpu_result > total_elements) {
        cpu_result = total_elements;
    }

    // GPU分配数 = 总数 - CPU分配数
    size_t gpu_result = total_elements - cpu_result;

    // 钳位：确保GPU分配数不超过可用内存约束
    if (gpu_result > gpu_max_elements) {
        gpu_result = gpu_max_elements;
        // 回补到CPU（如果CPU还有余量）
        size_t overflow = total_elements - gpu_result;
        if (overflow > cpu_result && cpu_result < cpu_max_elements) {
            cpu_result = std::min(overflow, cpu_max_elements);
            gpu_result = total_elements - cpu_result;
        } else {
            cpu_result = total_elements - gpu_result;
        }
    }

    return {cpu_result, gpu_result};
}

// ============================================================================
// estimate_cpu_capacity - 估算CPU可处理元素数
// ============================================================================

size_t TaskDecomposer::estimate_cpu_capacity(
    const DeviceCapability& cpu_cap,
    const DeviceCapability& gpu_cap,
    size_t total_elements)
{
    // 基于计算能力比值估算CPU能处理的元素数
    // 本质上是 estimate_optimal_split 的简化版本，仅考虑TFLOPS比值

    if (total_elements == 0) {
        return 0;
    }

    // 如果GPU不可用，CPU处理全部
    if (gpu_cap.compute_power_tflops <= 0.0) {
        return total_elements;
    }

    // 如果CPU不可用，CPU处理0个
    if (cpu_cap.compute_power_tflops <= 0.0) {
        return 0;
    }

    // 计算TFLOPS比值
    double total_tflops = cpu_cap.compute_power_tflops + gpu_cap.compute_power_tflops;
    double cpu_ratio = cpu_cap.compute_power_tflops / total_tflops;

    // 考虑了内存带宽的微调（使结果略有偏向）
    if (cpu_cap.memory_bandwidth_gb_per_sec > 0
        && gpu_cap.memory_bandwidth_gb_per_sec > 0) {

        double bw_ratio = static_cast<double>(cpu_cap.memory_bandwidth_gb_per_sec)
                        / static_cast<double>(gpu_cap.memory_bandwidth_gb_per_sec);

        // 带宽比例影响：CPU带宽相对越低，CPU应处理的越少
        double bw_adjust = kBandwidthAdjustBase + kBandwidthAdjustScale * bw_ratio;
        cpu_ratio *= bw_adjust;
    }

    // 转换为元素数量并钳位
    size_t result = static_cast<size_t>(
        static_cast<double>(total_elements) * cpu_ratio
    );

    if (result > total_elements) {
        result = total_elements;
    }

    return result;
}

// ============================================================================
// estimate_gpu_capacity - 估算GPU可处理元素数
// ============================================================================

size_t TaskDecomposer::estimate_gpu_capacity(
    const DeviceCapability& cpu_cap,
    const DeviceCapability& gpu_cap,
    size_t total_elements)
{
    // GPU容量 = 总数 - CPU容量（两者互补）
    if (total_elements == 0) {
        return 0;
    }

    size_t cpu_capacity = estimate_cpu_capacity(cpu_cap, gpu_cap, total_elements);
    return total_elements - cpu_capacity;
}

}  // namespace heterogeneous
}  // namespace go
}  // namespace spacego