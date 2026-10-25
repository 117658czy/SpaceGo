#pragma once

// ============================================================================
// 结果合并器 - 异构计算结果汇总与精度对齐
// 命名空间: spacego::go::heterogeneous
// 功能: 合并CPU和GPU分别产出的子任务计算结果，处理精度差异，保证结果一致性
// ============================================================================

#include "heterogeneous_types.h"
#include "config/platform_config.h"

#include <memory>
#include <vector>
#include <mutex>
#include <functional>
#include <cstdint>
#include <string>
#include <sstream>

namespace spacego {
namespace go {
namespace heterogeneous {

// ============================================================================
// MergedBoardResult - 棋盘评估合并结果
// ============================================================================

/**
 * @brief 存储多个棋盘的合法着法数、气数和合法着法列表的合并结果
 *
 * 该结构体是棋盘评估子任务合并后的最终输出。CPU和GPU分别处理不同
 * 的棋盘批次，合并器按原始索引顺序重新排列所有结果。
 *
 * 典型数据规模：
 *   - legal_moves_counts: 总棋盘数 * sizeof(int)
 *   - liberties:         总棋盘数 * sizeof(int)
 *   - legal_moves_per_board: 总棋盘数 * 平均合法着法数 * sizeof(int)
 */
struct MergedBoardResult {
    std::vector<int>                legal_moves_counts;     // 每个棋盘的合法着法总数
    std::vector<int>                liberties;              // 每个棋盘的总气数（所有组气数之和）
    std::vector<std::vector<int>>   legal_moves_per_board;  // 每个棋盘的合法着法位置列表
                                                            // 每个内部vector存放该棋盘所有合法着法的
                                                            // 位置索引（0~360）
    size_t                          total_boards;           // 合并后的总棋盘数量
    bool                            merge_success;          // 合并是否成功完成
                                                            // false时表示至少一个子任务失败或数据缺失

    // ========================================================================
    // 默认构造函数 - 初始化为空状态
    // ========================================================================
    MergedBoardResult()
        : total_boards(0)
        , merge_success(false)
    {
    }
};

// ============================================================================
// MergedInferenceResult - 神经网络推理合并结果
// ============================================================================

/**
 * @brief 存储批量神经网络推理的策略和价值合并结果
 *
 * CPU和GPU可以分别处理不同的推理批次（如CPU处理小批次，
 * GPU处理大批次），合并器将所有结果按原始顺序拼接。
 *
 * 数据布局：
 *   - policy_output: [sample0_policy(362 floats), sample1_policy(362 floats), ...]
 *   - value_output:  [sample0_value, sample1_value, ...]
 *     每个value是单精度浮点数，范围[-1.0, 1.0]，表示黑棋胜率估计
 */
struct MergedInferenceResult {
    std::vector<float>  policy_output;     // 合并后的策略输出（扁平化存储）
                                           // 大小 = total_batch_size * policy_dim
    std::vector<float>  value_output;      // 合并后的价值输出
                                           // 大小 = total_batch_size
    size_t              total_batch_size;  // 总批次大小（所有子任务样本数之和）
    int                 policy_dim;        // 策略输出维度（通常为362：361个着法点 + pass）
    bool                merge_success;     // 合并是否成功完成

    // ========================================================================
    // 默认构造函数 - 初始化为空状态
    // ========================================================================
    MergedInferenceResult()
        : total_batch_size(0)
        , policy_dim(0)
        , merge_success(false)
    {
    }
};

// ============================================================================
// MergedTopologyResult - 拓扑计算合并结果
// ============================================================================

/**
 * @brief 存储空间拓扑计算的连通性、中心性和战略价值合并结果
 *
 * 拓扑计算可以按区域分片，CPU和GPU分别计算不同的区域组，
 * 合并器将所有区域计算结果汇总。
 *
 * 数据含义：
 *   - connectivity_scores: 每个区域的棋子连通性评分 [0.0, 1.0]
 *   - centrality_scores:   每个区域在棋盘上的中心性评分 [0.0, 1.0]
 *   - strategic_values:    每个区域的综合战略价值评分 [0.0, 1.0]
 */
struct MergedTopologyResult {
    std::vector<double> connectivity_scores;    // 每个区域的连通性分数
    std::vector<double> centrality_scores;      // 每个区域的中心性分数
    std::vector<double> strategic_values;       // 每个区域的战略价值分数
    size_t              total_regions;          // 合并后的总区域数量
    bool                merge_success;          // 合并是否成功完成

    // ========================================================================
    // 默认构造函数 - 初始化为空状态
    // ========================================================================
    MergedTopologyResult()
        : total_regions(0)
        , merge_success(false)
    {
    }
};

// ============================================================================
// SubTaskResult - 子任务执行结果（通用容器）
// ============================================================================

/**
 * @brief 存储单个子任务的执行结果，包含序列化的原始数据
 *
 * 与 SubTask（任务描述符）不同，SubTaskResult 是子任务执行完毕后
 * 产生的结果容器。raw_data 使用二进制序列化格式存储结果数据，
 * 不同类型的结果有不同的序列化格式（由各自的 deserialize 方法解析）。
 *
 * 关键字段：
 *   - subtask_id:    关联回原始 SubTask 的 subtask_id
 *   - source_device: 记录在哪个设备上执行（CPU 或 GPU）
 *   - data_offset:   该结果对应的数据起始偏移位置
 *   - data_length:   该结果覆盖的数据长度
 *   - raw_data:      序列化的二进制结果数据
 *   - success:       子任务执行是否成功
 *   - error_message: 失败时的错误描述信息
 *
 * 序列化格式说明（不同任务类型使用不同的反序列化方法）：
 *
 *   棋盘评估结果格式（raw_data布局）：
 *     [boards_count:4B][per_board: board_count个]
 *       每个board: [legal_moves_count:4B][liberty_sum:4B][move_count:4B]
 *                 [moves: move_count*4B]
 *
 *   神经网络推理结果格式（raw_data布局）：
 *     [samples_count:4B]
 *     [per_sample: samples_count个]
 *       每个sample: [policy: policy_dim*4B][value: 4B]
 *
 *   拓扑计算结果格式（raw_data布局）：
 *     [regions_count:4B]
 *     [per_region: regions_count个]
 *       每个region: [connectivity:8B][centrality:8B][strategic_value:8B]
 */
struct SubTaskResult {
    size_t                  subtask_id;         // 子任务唯一ID（关联原始 SubTask）
    ComputeDeviceType       source_device;      // 执行该子任务的设备类型
    size_t                  data_offset;        // 该结果对应的数据起始偏移
    size_t                  data_length;        // 该结果覆盖的数据长度（元素数量）
    std::vector<uint8_t>    raw_data;           // 序列化的二进制结果数据
    bool                    success;            // 子任务执行是否成功
    std::string             error_message;      // 失败时的错误描述

    // ========================================================================
    // 默认构造函数 - 初始化为空结果状态
    // ========================================================================
    SubTaskResult()
        : subtask_id(0)
        , source_device(ComputeDeviceType::CPU)
        , data_offset(0)
        , data_length(0)
        , success(false)
        , error_message()
    {
    }
};

// ============================================================================
// ResultMerger - 结果合并器
// ============================================================================

/**
 * @brief 异构计算结果的合并器，负责汇总CPU和GPU的子任务结果
 *
 * 核心职责：
 *   1. 棋盘评估结果合并 - 合并各设备产出的legal_moves和liberties数据
 *   2. 神经网络推理结果合并 - 合并policy和value输出向量
 *   3. 拓扑计算结果合并 - 合并连通性、中心性和战略价值数据
 *   4. 精度对齐 - 补偿GPU FP16计算相对于CPU FP32的精度损失
 *
 * 使用模式：
 *   1. 构造 ResultMerger 实例
 *   2. 收集所有子任务结果到 SubTaskResult 容器中
 *   3. 调用对应的 merge_*_results 方法
 *   4. 检查返回结构体的 merge_success 字段确认合并状态
 *   5. 如有需要，调用 get_last_merge_diagnostic() 获取诊断信息
 *
 * 线程安全性：
 *   - 所有公有方法通过内部 mutex 保护，可在多线程环境中安全使用
 *   - 但建议在外部分配独立的 ResultMerger 实例给不同的合并任务
 *
 * 使用示例：
 * @code
 *   ResultMerger merger;
 *
 *   // 收集CPU和GPU的子任务结果
 *   std::vector<SubTaskResult> cpu_results = ...;
 *   std::vector<SubTaskResult> gpu_results = ...;
 *
 *   // 合并棋盘评估结果
 *   auto board_merged = merger.merge_board_results(
 *       cpu_results, gpu_results, total_boards);
 *
 *   if (!board_merged.merge_success) {
 *       std::cerr << "合并诊断: "
 *                 << merger.get_last_merge_diagnostic() << std::endl;
 *   }
 * @endcode
 */
class ResultMerger {
public:
    // ========================================================================
    // 构造函数与析构函数
    // ========================================================================

    /**
     * @brief 构造结果合并器
     *
     * 初始化内部状态，确保线程安全锁就绪。
     */
    ResultMerger();

    /**
     * @brief 析构函数
     */
    ~ResultMerger();

    // ---- 禁止拷贝，允许移动 ----
    ResultMerger(const ResultMerger&) = delete;
    ResultMerger& operator=(const ResultMerger&) = delete;

    // ========================================================================
    // 棋盘评估结果合并
    // ========================================================================

    /**
     * @brief 合并CPU和GPU分别产出的棋盘评估子任务结果
     *
     * 处理流程：
     *   1. 验证所有子任务结果的有效性（success标志、data_range合理性）
     *   2. 将所有结果（CPU + GPU）合并到一个列表中
     *   3. 按 data_offset 排序，确保结果按原始索引顺序排列
     *   4. 检查结果连续性：所有子任务的 data_offset+data_length
     *      应完整覆盖 0 到 total_boards
     *   5. 按顺序反序列化每个子任务的棋盘评估数据
     *   6. 汇总到 MergedBoardResult 中返回
     *
     * 错误处理：
     *   - 如果发现任何子任务 success=false，标记 merge_success=false
     *   - 如果结果不连续（有空隙），标记 merge_success=false
     *   - 如果总覆盖数量不等于 total_boards，标记 merge_success=false
     *   - 所有错误信息记录到 last_diagnostic_
     *
     * @param cpu_results  CPU设备产出的棋盘评估子任务结果列表
     * @param gpu_results  GPU设备产出的棋盘评估子任务结果列表
     * @param total_boards 期望的总棋盘数量（用于验证完整性）
     * @return MergedBoardResult 合并后的棋盘评估结果
     */
    MergedBoardResult merge_board_results(
        const std::vector<SubTaskResult>& cpu_results,
        const std::vector<SubTaskResult>& gpu_results,
        size_t total_boards);

    // ========================================================================
    // 神经网络推理结果合并
    // ========================================================================

    /**
     * @brief 合并CPU和GPU分别产出的神经网络推理子任务结果
     *
     * 处理流程：
     *   1. 验证所有子任务结果的有效性
     *   2. 合并并排序所有结果
     *   3. 检查连续性
     *   4. 反序列化每个子任务的策略和价值数据
     *   5. 按原始顺序拼接 policy_output 和 value_output 向量
     *   6. 如果GPU使用了FP16计算，调用 align_precision 进行精度对齐
     *
     * 精度对齐说明：
     *   - 当检测到 source_device=GPU 的子任务时，自动将GPU产出的
     *     浮点值与CPU参考值比较
     *   - 差异超过 epsilon 阈值的值使用CPU参考值替代
     *   - 目的：消除GPU FP16半精度计算引入的累积误差
     *
     * @param cpu_results      CPU设备产出的推理子任务结果列表
     * @param gpu_results      GPU设备产出的推理子任务结果列表
     * @param total_batch_size 期望的总批次大小（用于验证完整性）
     * @param policy_dim       策略输出维度（通常为362 = 361个着法点 + pass）
     * @return MergedInferenceResult 合并后的推理结果
     */
    MergedInferenceResult merge_inference_results(
        const std::vector<SubTaskResult>& cpu_results,
        const std::vector<SubTaskResult>& gpu_results,
        size_t total_batch_size,
        int policy_dim);

    // ========================================================================
    // 拓扑计算结果合并
    // ========================================================================

    /**
     * @brief 合并CPU和GPU分别产出的空间拓扑计算子任务结果
     *
     * 处理流程：
     *   1. 验证所有子任务结果的有效性
     *   2. 合并并排序所有结果
     *   3. 检查连续性
     *   4. 反序列化每个子任务的拓扑特征数据
     *   5. 汇总 connectivity_scores、centrality_scores、strategic_values
     *
     * 拓扑计算通常按棋盘区域分片，每个子任务处理一组区域。
     * 合并后得到完整的区域拓扑特征数组。
     *
     * @param cpu_results   CPU设备产出的拓扑计算子任务结果列表
     * @param gpu_results   GPU设备产出的拓扑计算子任务结果列表
     * @param total_regions 期望的总区域数量（用于验证完整性）
     * @return MergedTopologyResult 合并后的拓扑计算结果
     */
    MergedTopologyResult merge_topology_results(
        const std::vector<SubTaskResult>& cpu_results,
        const std::vector<SubTaskResult>& gpu_results,
        size_t total_regions);

    // ========================================================================
    // 精度对齐处理
    // ========================================================================

    /**
     * @brief 将GPU FP16精度结果对齐到CPU FP32精度
     *
     * 当CPU使用FP32计算、GPU使用FP16计算时，GPU结果可能存在精度损失。
     * 此方法比较GPU值和CPU参考值，如果差异超过epsilon阈值，
     * 则使用CPU参考值替代GPU值。
     *
     * 对齐策略：
     *   - 遍历 gpu_values 中的每个元素
     *   - 与 cpu_reference_values 中对应位置的参考值比较
     *   - 如果 |gpu - cpu| > epsilon，将 gpu 值替换为 cpu 值
     *   - 目的：确保最终结果满足精度要求，消除FP16累积误差
     *
     * 使用前提：
     *   - cpu_reference_values 的大小必须 >= gpu_values 的大小
     *   - 两个向量中相同索引位置的元素对应相同的计算含义
     *
     * @param gpu_values              GPU产出的值向量（将被就地修改）
     * @param cpu_reference_values    CPU产出的FP32参考值向量
     * @param epsilon                 允许的最大差异阈值（默认1e-5）
     *                                超过此阈值的GPU值被认为精度不足
     */
    void align_precision(std::vector<float>& gpu_values,
                         const std::vector<float>& cpu_reference_values,
                         double epsilon = 1e-5);

    // ========================================================================
    // 诊断信息
    // ========================================================================

    /**
     * @brief 获取最后一次合并操作的诊断信息
     *
     * 返回包含错误详情、警告和统计信息的字符串。
     * 当 merge_success=false 时调用此方法获取失败原因。
     *
     * @return const std::string& 最后一次合并的诊断信息
     */
    const std::string& get_last_merge_diagnostic() const { return last_diagnostic_; }

private:
    // ========================================================================
    // 线程同步
    // ========================================================================

    mutable std::mutex  merge_mutex_;        // 合并操作互斥锁，保护内部状态
                                             // 使用 mutable 允许在 const 方法中加锁

    // ========================================================================
    // 诊断信息
    // ========================================================================

    std::string         last_diagnostic_;    // 最后一次合并的诊断信息字符串
                                             // 包含错误详情、警告和统计信息

    // ========================================================================
    // 内部辅助方法
    // ========================================================================

    /**
     * @brief 验证子任务结果列表的有效性
     *
     * 对每个子任务执行以下检查：
     *   - success 标志是否为 true
     *   - data_offset 是否在合理范围内（不超过 SIZE_MAX / 2）
     *   - data_length 是否大于 0
     *   - raw_data 是否非空
     *
     * @param results      待验证的子任务结果列表
     * @param source_name  来源名称（用于诊断信息，如"CPU"或"GPU"）
     * @return bool        所有子任务均有效返回true，否则返回false
     */
    bool validate_subtask_results(const std::vector<SubTaskResult>& results,
                                  const std::string& source_name);

    /**
     * @brief 按 data_offset 对子任务结果进行升序排序
     *
     * 排序后的结果可以用于连续性检查。使用 std::sort 进行原地排序。
     *
     * @param results 待排序的子任务结果列表（将被就地修改）
     */
    void sort_by_offset(std::vector<SubTaskResult>& results);

    /**
     * @brief 检查排序后的子任务结果的连续性和完整性
     *
     * 检查内容：
     *   1. 排序后的第一个结果的 data_offset 必须为 0
     *   2. 相邻结果的 data_offset + data_length 必须等于下一个结果的 data_offset
     *   3. 最后一个结果的 data_offset + data_length 必须等于 total_size
     *
     * 只有所有检查通过才返回 true，表示数据完整覆盖 [0, total_size)。
     *
     * @param results    已按 data_offset 排序的子任务结果列表
     * @param total_size 期望覆盖的总大小
     * @return bool      数据完整连续返回true，否则返回false
     */
    bool check_result_continuity(const std::vector<SubTaskResult>& results,
                                  size_t total_size);

    /**
     * @brief 从 SubTaskResult 的 raw_data 中反序列化棋盘评估结果
     *
     * 解析 raw_data 的二进制格式，提取单个棋盘的评估数据。
     * 格式说明详见 SubTaskResult 的文档注释。
     *
     * @param result             包含序列化数据的子任务结果
     * @param legal_moves_counts 输出参数：每个棋盘的合法着法数（追加模式）
     * @param liberties          输出参数：每个棋盘的总气数（追加模式）
     * @param legal_moves        输出参数：所有棋盘的合法着法列表（追加模式）
     * @return bool              反序列化成功返回true，否则返回false
     */
    bool deserialize_board_result(const SubTaskResult& result,
                                   std::vector<int>& legal_moves_counts,
                                   std::vector<int>& liberties,
                                   std::vector<std::vector<int>>& legal_moves);

    /**
     * @brief 从 SubTaskResult 的 raw_data 中反序列化神经网络推理结果
     *
     * 解析 raw_data 的二进制格式，提取策略和价值数据。
     * 格式说明详见 SubTaskResult 的文档注释。
     *
     * @param result      包含序列化数据的子任务结果
     * @param policy      输出参数：策略输出（追加模式，按样本顺序排列）
     * @param value       输出参数：价值输出（追加模式）
     * @param policy_dim  策略输出维度（用于验证数据长度）
     * @return bool       反序列化成功返回true，否则返回false
     */
    bool deserialize_inference_result(const SubTaskResult& result,
                                       std::vector<float>& policy,
                                       std::vector<float>& value,
                                       int policy_dim);

    /**
     * @brief 从 SubTaskResult 的 raw_data 中反序列化拓扑计算结果
     *
     * 解析 raw_data 的二进制格式，提取单个区域的拓扑特征。
     * 格式说明详见 SubTaskResult 的文档注释。
     *
     * @param result           包含序列化数据的子任务结果
     * @param connectivity     输出参数：连通性分数（追加模式）
     * @param centrality       输出参数：中心性分数（追加模式）
     * @param strategic_value  输出参数：战略价值（追加模式）
     * @return bool            反序列化成功返回true，否则返回false
     */
    bool deserialize_topology_result(const SubTaskResult& result,
                                      std::vector<double>& connectivity,
                                      std::vector<double>& centrality,
                                      std::vector<double>& strategic_value);

    // ========================================================================
    // 二进制读取辅助方法（从 uint8_t 缓冲区安全读取各类型数据）
    // ========================================================================

    /**
     * @brief 从字节缓冲区读取 uint32_t（小端序）
     *
     * @param data   字节缓冲区指针
     * @param offset 读取起始偏移
     * @param value  输出参数：读取的值
     * @param max_len 缓冲区总长度（用于边界检查）
     * @return bool  读取成功返回true（有足够字节），否则返回false
     */
    static bool read_uint32_le(const std::vector<uint8_t>& data, size_t offset,
                               uint32_t& value, size_t max_len);

    /**
     * @brief 从字节缓冲区读取 int32_t（小端序）
     *
     * @param data   字节缓冲区指针
     * @param offset 读取起始偏移
     * @param value  输出参数：读取的值
     * @param max_len 缓冲区总长度（用于边界检查）
     * @return bool  读取成功返回true（有足够字节），否则返回false
     */
    static bool read_int32_le(const std::vector<uint8_t>& data, size_t offset,
                              int32_t& value, size_t max_len);

    /**
     * @brief 从字节缓冲区读取 float（小端序，IEEE 754 binary32）
     *
     * @param data   字节缓冲区指针
     * @param offset 读取起始偏移
     * @param value  输出参数：读取的值
     * @param max_len 缓冲区总长度（用于边界检查）
     * @return bool  读取成功返回true（有足够字节），否则返回false
     */
    static bool read_float_le(const std::vector<uint8_t>& data, size_t offset,
                              float& value, size_t max_len);

    /**
     * @brief 从字节缓冲区读取 double（小端序，IEEE 754 binary64）
     *
     * @param data   字节缓冲区指针
     * @param offset 读取起始偏移
     * @param value  输出参数：读取的值
     * @param max_len 缓冲区总长度（用于边界检查）
     * @return bool  读取成功返回true（有足够字节），否则返回false
     */
    static bool read_double_le(const std::vector<uint8_t>& data, size_t offset,
                               double& value, size_t max_len);
};

}  // namespace heterogeneous
}  // namespace go
}  // namespace spacego