// ============================================================================
// 结果合并器 - 异构计算结果汇总与精度对齐（实现文件）
// 命名空间: spacego::go::heterogeneous
// 功能: 实现 ResultMerger 类的所有方法
// ============================================================================

#include "go/heterogeneous/result_merger.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace spacego {
namespace go {
namespace heterogeneous {

// ============================================================================
// 构造函数与析构函数
// ============================================================================

ResultMerger::ResultMerger()
    : last_diagnostic_("结果合并器已初始化，等待合并操作。")
{
}

ResultMerger::~ResultMerger()
{
}

// ============================================================================
// merge_board_results - 棋盘评估结果合并
// ============================================================================

MergedBoardResult ResultMerger::merge_board_results(
    const std::vector<SubTaskResult>& cpu_results,
    const std::vector<SubTaskResult>& gpu_results,
    size_t total_boards)
{
    std::lock_guard<std::mutex> lock(merge_mutex_);

    std::ostringstream diag;
    diag << "棋盘评估结果合并开始，总棋盘数=" << total_boards
         << "，CPU子任务数=" << cpu_results.size()
         << "，GPU子任务数=" << gpu_results.size();

    MergedBoardResult merged;
    merged.total_boards = total_boards;
    merged.merge_success = false;

    // ---- 第1步：验证CPU和GPU子任务结果 ----
    if (!cpu_results.empty() && !validate_subtask_results(cpu_results, "CPU")) {
        diag << " | 错误：CPU子任务结果验证失败";
        last_diagnostic_ = diag.str();
        return merged;
    }
    if (!gpu_results.empty() && !validate_subtask_results(gpu_results, "GPU")) {
        diag << " | 错误：GPU子任务结果验证失败";
        last_diagnostic_ = diag.str();
        return merged;
    }

    // ---- 第2步：合并所有结果到一个列表并排序 ----
    std::vector<SubTaskResult> all_results;
    all_results.reserve(cpu_results.size() + gpu_results.size());

    // 合并CPU结果
    for (const auto& r : cpu_results) {
        all_results.push_back(r);
    }
    // 合并GPU结果
    for (const auto& r : gpu_results) {
        all_results.push_back(r);
    }

    // 按 data_offset 升序排序
    sort_by_offset(all_results);

    // ---- 第3步：检查结果连续性和完整性 ----
    if (!check_result_continuity(all_results, total_boards)) {
        diag << " | 错误：子任务结果数据不连续，存在空缺或重叠";
        last_diagnostic_ = diag.str();
        return merged;
    }

    // ---- 第4步：按顺序反序列化每个子任务的棋盘评估数据 ----
    merged.legal_moves_counts.reserve(total_boards);
    merged.liberties.reserve(total_boards);
    merged.legal_moves_per_board.reserve(total_boards);

    for (const auto& result : all_results) {
        if (!deserialize_board_result(result,
                                       merged.legal_moves_counts,
                                       merged.liberties,
                                       merged.legal_moves_per_board)) {
            diag << " | 错误：子任务ID=" << result.subtask_id
                 << " 的棋盘评估数据反序列化失败";
            last_diagnostic_ = diag.str();
            return merged;
        }
    }

    // ---- 第5步：验证合并结果的数据一致性 ----
    if (merged.legal_moves_counts.size() != total_boards ||
        merged.liberties.size() != total_boards ||
        merged.legal_moves_per_board.size() != total_boards) {
        diag << " | 错误：合并后数据量不一致，期望=" << total_boards
             << "，实际legal_moves_counts=" << merged.legal_moves_counts.size()
             << "，liberties=" << merged.liberties.size()
             << "，legal_moves_per_board=" << merged.legal_moves_per_board.size();
        last_diagnostic_ = diag.str();
        return merged;
    }

    // ---- 合并成功 ----
    merged.merge_success = true;
    diag << " | 合并成功，总合法着法数="
         << [&]() { size_t sum = 0; for (auto c : merged.legal_moves_counts) sum += static_cast<size_t>(c); return sum; }()
         << "，总气数="
         << [&]() { int64_t sum = 0; for (auto l : merged.liberties) sum += l; return sum; }();

    last_diagnostic_ = diag.str();
    return merged;
}

// ============================================================================
// merge_inference_results - 神经网络推理结果合并
// ============================================================================

MergedInferenceResult ResultMerger::merge_inference_results(
    const std::vector<SubTaskResult>& cpu_results,
    const std::vector<SubTaskResult>& gpu_results,
    size_t total_batch_size,
    int policy_dim)
{
    std::lock_guard<std::mutex> lock(merge_mutex_);

    std::ostringstream diag;
    diag << "神经网络推理结果合并开始，总批次大小=" << total_batch_size
         << "，策略维度=" << policy_dim
         << "，CPU子任务数=" << cpu_results.size()
         << "，GPU子任务数=" << gpu_results.size();

    MergedInferenceResult merged;
    merged.total_batch_size = total_batch_size;
    merged.policy_dim = policy_dim;
    merged.merge_success = false;

    // ---- 第1步：验证所有子任务结果 ----
    if (!cpu_results.empty() && !validate_subtask_results(cpu_results, "CPU")) {
        diag << " | 错误：CPU推理结果验证失败";
        last_diagnostic_ = diag.str();
        return merged;
    }
    if (!gpu_results.empty() && !validate_subtask_results(gpu_results, "GPU")) {
        diag << " | 错误：GPU推理结果验证失败";
        last_diagnostic_ = diag.str();
        return merged;
    }

    // ---- 第2步：合并所有结果并排序 ----
    std::vector<SubTaskResult> all_results;
    all_results.reserve(cpu_results.size() + gpu_results.size());

    for (const auto& r : cpu_results) {
        all_results.push_back(r);
    }
    for (const auto& r : gpu_results) {
        all_results.push_back(r);
    }

    sort_by_offset(all_results);

    // ---- 第3步：检查连续性 ----
    if (!check_result_continuity(all_results, total_batch_size)) {
        diag << " | 错误：推理子任务结果数据不连续";
        last_diagnostic_ = diag.str();
        return merged;
    }

    // ---- 第4步：预分配合并后的输出缓冲区 ----
    merged.policy_output.reserve(total_batch_size * static_cast<size_t>(policy_dim));
    merged.value_output.reserve(total_batch_size);

    // ---- 第5步：按顺序反序列化每个子任务的推理数据 ----
    for (const auto& result : all_results) {
        if (!deserialize_inference_result(result,
                                           merged.policy_output,
                                           merged.value_output,
                                           policy_dim)) {
            diag << " | 错误：子任务ID=" << result.subtask_id
                 << " 的推理数据反序列化失败";
            last_diagnostic_ = diag.str();
            return merged;
        }
    }

    // ---- 第6步：验证合并后数据量 ----
    const size_t expected_policy_size = total_batch_size * static_cast<size_t>(policy_dim);
    if (merged.policy_output.size() != expected_policy_size ||
        merged.value_output.size() != total_batch_size) {
        diag << " | 错误：合并后推理数据量不一致，期望policy_size="
             << expected_policy_size
             << "，实际=" << merged.policy_output.size()
             << "，期望value_size=" << total_batch_size
             << "，实际=" << merged.value_output.size();
        last_diagnostic_ = diag.str();
        return merged;
    }

    // ---- 第7步：精度对齐处理 ----
    // 当存在GPU子任务时，对GPU产出的浮点值进行精度对齐。
    // 对齐策略：对于GPU产出结果覆盖的每个数据偏移范围，
    //   如果存在CPU子任务覆盖完全相同的偏移范围（即冗余计算验证场景），
    //   则从merged结果中提取对应的CPU参考值进行逐元素比较和纠正。
    //   如果CPU和GPU处理的是互补范围（正常异构分片场景），
    //   则跳过对齐（因为没有可比较的参考值）。
    //
    // 典型场景说明：
    //   场景A（冗余验证）：CPU FP32全量计算 + GPU FP16全量计算
    //     -> GPU的所有值都可以用CPU参考值对齐
    //   场景B（异构分片）：CPU处理[0,N) + GPU处理[N,N+M)
    //     -> 没有重叠，跳过对齐
    //   场景C（混合）：CPU处理[0,N) + GPU处理[0,N+M)（GPU覆盖CPU范围）
    //     -> 仅对齐GPU中与CPU重叠的[0,N)部分
    bool has_gpu_results = std::any_of(all_results.begin(), all_results.end(),
        [](const SubTaskResult& r) {
            return r.source_device == ComputeDeviceType::GPU;
        });

    if (has_gpu_results) {
        size_t total_aligned_policy = 0;
        size_t total_aligned_value = 0;

        // 遍历每个GPU子任务结果，尝试找到覆盖相同偏移范围的CPU参考值
        for (const auto& gpu_result : all_results) {
            if (gpu_result.source_device != ComputeDeviceType::GPU) {
                continue;
            }

            const size_t gpu_offset = gpu_result.data_offset;
            const size_t gpu_length = gpu_result.data_length;

            // 查找覆盖相同偏移范围的CPU子任务结果
            // 对齐条件：CPU结果的偏移范围完全包含GPU结果的偏移范围
            const SubTaskResult* cpu_ref = nullptr;
            for (const auto& cpu_result : all_results) {
                if (cpu_result.source_device != ComputeDeviceType::CPU) {
                    continue;
                }
                // 检查CPU结果是否覆盖GPU结果的完整偏移范围
                if (cpu_result.data_offset <= gpu_offset &&
                    cpu_result.data_offset + cpu_result.data_length >= gpu_offset + gpu_length) {
                    cpu_ref = &cpu_result;
                    break;
                }
            }

            if (cpu_ref == nullptr) {
                // 没有CPU参考值覆盖该GPU范围，跳过此GPU子任务的对齐
                continue;
            }

            // ---- 存在匹配的CPU参考值，执行精度对齐 ----
            const size_t policy_start = gpu_offset * static_cast<size_t>(policy_dim);
            const size_t cpu_policy_start = cpu_ref->data_offset * static_cast<size_t>(policy_dim);

            // 提取GPU子任务的策略输出子向量
            std::vector<float> gpu_policy_chunk;
            gpu_policy_chunk.reserve(gpu_length * static_cast<size_t>(policy_dim));
            for (size_t i = 0; i < gpu_length; ++i) {
                for (int p = 0; p < policy_dim; ++p) {
                    gpu_policy_chunk.push_back(
                        merged.policy_output[policy_start + i * static_cast<size_t>(policy_dim) + static_cast<size_t>(p)]);
                }
            }

            // 提取对应的CPU参考策略数据（从merged中已有的CPU数据）
            std::vector<float> cpu_ref_policy;
            cpu_ref_policy.reserve(gpu_length * static_cast<size_t>(policy_dim));
            // CPU结果中对应GPU偏移范围的起始位置
            const size_t cpu_offset_in_result = gpu_offset - cpu_ref->data_offset;
            const size_t cpu_policy_start_in_result = cpu_offset_in_result * static_cast<size_t>(policy_dim);
            for (size_t i = 0; i < gpu_length; ++i) {
                for (int p = 0; p < policy_dim; ++p) {
                    cpu_ref_policy.push_back(
                        merged.policy_output[cpu_policy_start + cpu_policy_start_in_result
                                             + i * static_cast<size_t>(policy_dim) + static_cast<size_t>(p)]);
                }
            }

            // 执行策略精度对齐
            if (!gpu_policy_chunk.empty() && !cpu_ref_policy.empty()) {
                align_precision(gpu_policy_chunk, cpu_ref_policy);
            }

            // 提取GPU子任务的价值输出子向量
            std::vector<float> gpu_value_chunk;
            gpu_value_chunk.reserve(gpu_length);
            for (size_t i = 0; i < gpu_length; ++i) {
                gpu_value_chunk.push_back(merged.value_output[gpu_offset + i]);
            }

            // 提取对应的CPU参考价值数据
            std::vector<float> cpu_ref_value;
            cpu_ref_value.reserve(gpu_length);
            for (size_t i = 0; i < gpu_length; ++i) {
                cpu_ref_value.push_back(
                    merged.value_output[cpu_ref->data_offset + cpu_offset_in_result + i]);
            }

            // 执行价值精度对齐
            if (!gpu_value_chunk.empty() && !cpu_ref_value.empty()) {
                align_precision(gpu_value_chunk, cpu_ref_value);
            }

            // 将对齐后的数据写回 merged 向量
            for (size_t i = 0; i < gpu_length; ++i) {
                for (int p = 0; p < policy_dim; ++p) {
                    merged.policy_output[policy_start + i * static_cast<size_t>(policy_dim) + static_cast<size_t>(p)]
                        = gpu_policy_chunk[i * static_cast<size_t>(policy_dim) + static_cast<size_t>(p)];
                }
                merged.value_output[gpu_offset + i] = gpu_value_chunk[i];
            }

            total_aligned_policy += gpu_length * static_cast<size_t>(policy_dim);
            total_aligned_value += gpu_length;
        }

        if (total_aligned_policy > 0) {
            diag << " | 已执行GPU精度对齐（FP16->FP32纠正），"
                 << "对齐策略值=" << total_aligned_policy
                 << "，对齐价值值=" << total_aligned_value;
        } else {
            diag << " | GPU精度对齐跳过：未找到匹配的CPU参考偏移范围";
        }
    }

    // ---- 合并成功 ----
    merged.merge_success = true;
    diag << " | 合并成功，策略向量大小=" << merged.policy_output.size()
         << "，价值向量大小=" << merged.value_output.size();

    last_diagnostic_ = diag.str();
    return merged;
}

// ============================================================================
// merge_topology_results - 拓扑计算结果合并
// ============================================================================

MergedTopologyResult ResultMerger::merge_topology_results(
    const std::vector<SubTaskResult>& cpu_results,
    const std::vector<SubTaskResult>& gpu_results,
    size_t total_regions)
{
    std::lock_guard<std::mutex> lock(merge_mutex_);

    std::ostringstream diag;
    diag << "拓扑计算结果合并开始，总区域数=" << total_regions
         << "，CPU子任务数=" << cpu_results.size()
         << "，GPU子任务数=" << gpu_results.size();

    MergedTopologyResult merged;
    merged.total_regions = total_regions;
    merged.merge_success = false;

    // ---- 第1步：验证所有子任务结果 ----
    if (!cpu_results.empty() && !validate_subtask_results(cpu_results, "CPU")) {
        diag << " | 错误：CPU拓扑结果验证失败";
        last_diagnostic_ = diag.str();
        return merged;
    }
    if (!gpu_results.empty() && !validate_subtask_results(gpu_results, "GPU")) {
        diag << " | 错误：GPU拓扑结果验证失败";
        last_diagnostic_ = diag.str();
        return merged;
    }

    // ---- 第2步：合并所有结果并排序 ----
    std::vector<SubTaskResult> all_results;
    all_results.reserve(cpu_results.size() + gpu_results.size());

    for (const auto& r : cpu_results) {
        all_results.push_back(r);
    }
    for (const auto& r : gpu_results) {
        all_results.push_back(r);
    }

    sort_by_offset(all_results);

    // ---- 第3步：检查连续性 ----
    if (!check_result_continuity(all_results, total_regions)) {
        diag << " | 错误：拓扑子任务结果数据不连续";
        last_diagnostic_ = diag.str();
        return merged;
    }

    // ---- 第4步：预分配输出缓冲区 ----
    merged.connectivity_scores.reserve(total_regions);
    merged.centrality_scores.reserve(total_regions);
    merged.strategic_values.reserve(total_regions);

    // ---- 第5步：按顺序反序列化每个子任务的拓扑数据 ----
    for (const auto& result : all_results) {
        if (!deserialize_topology_result(result,
                                          merged.connectivity_scores,
                                          merged.centrality_scores,
                                          merged.strategic_values)) {
            diag << " | 错误：子任务ID=" << result.subtask_id
                 << " 的拓扑数据反序列化失败";
            last_diagnostic_ = diag.str();
            return merged;
        }
    }

    // ---- 第6步：验证合并后数据量 ----
    if (merged.connectivity_scores.size() != total_regions ||
        merged.centrality_scores.size() != total_regions ||
        merged.strategic_values.size() != total_regions) {
        diag << " | 错误：合并后拓扑数据量不一致，期望=" << total_regions
             << "，实际connectivity=" << merged.connectivity_scores.size()
             << "，centrality=" << merged.centrality_scores.size()
             << "，strategic=" << merged.strategic_values.size();
        last_diagnostic_ = diag.str();
        return merged;
    }

    // ---- 合并成功 ----
    merged.merge_success = true;
    diag << " | 合并成功，合并区域数=" << total_regions;

    last_diagnostic_ = diag.str();
    return merged;
}

// ============================================================================
// align_precision - 精度对齐处理
// ============================================================================

void ResultMerger::align_precision(std::vector<float>& gpu_values,
                                    const std::vector<float>& cpu_reference_values,
                                    double epsilon)
{
    // 参数验证：确保参考值向量至少与GPU值向量一样大
    if (cpu_reference_values.size() < gpu_values.size()) {
        // CPU参考值不足时，无法进行完整的精度对齐
        // 仅对前 cpu_reference_values.size() 个元素进行对齐
        // 这是一个降级策略，不应在正常运行时发生
        size_t alignable_count = cpu_reference_values.size();
        for (size_t i = 0; i < alignable_count; ++i) {
            double diff = std::abs(static_cast<double>(gpu_values[i]) -
                                   static_cast<double>(cpu_reference_values[i]));
            if (diff > epsilon) {
                gpu_values[i] = cpu_reference_values[i];
            }
        }
        return;
    }

    // 标准精度对齐流程：
    // 遍历所有GPU值，与CPU参考值逐一比较
    // 差异超过epsilon阈值的GPU值视为精度不足，使用CPU参考值替代
    size_t count = gpu_values.size();
    size_t aligned_count = 0;  // 统计被对齐的元素数量

    for (size_t i = 0; i < count; ++i) {
        // 计算GPU值与CPU参考值之间的绝对差异
        double diff = std::abs(static_cast<double>(gpu_values[i]) -
                               static_cast<double>(cpu_reference_values[i]));

        // 检查是否超过阈值：
        //   - 如果不是NaN且差异大于epsilon，则GPU值精度不足
        //   - 如果GPU值是NaN而CPU值不是NaN，也需要对齐
        //   - 如果两者都是NaN，则不需要对齐（都不可靠）
        bool gpu_is_nan = std::isnan(gpu_values[i]);
        bool cpu_is_nan = std::isnan(cpu_reference_values[i]);

        if ((gpu_is_nan && !cpu_is_nan) || (!gpu_is_nan && !cpu_is_nan && diff > epsilon)) {
            // GPU值精度不足：用CPU参考值替代
            gpu_values[i] = cpu_reference_values[i];
            ++aligned_count;
        }
    }

    // 将对齐统计信息附加到诊断信息中（如果对齐数量较大，说明精度问题严重）
    if (aligned_count > 0 && count > 0) {
        double align_ratio = static_cast<double>(aligned_count) /
                             static_cast<double>(count) * 100.0;
        if (align_ratio > 10.0) {
            // 超过10%的值需要对齐，记录警告
            last_diagnostic_ += " | 精度对齐警告："
                              + std::to_string(aligned_count) + "/"
                              + std::to_string(count) + " ("
                              + std::to_string(static_cast<int>(align_ratio))
                              + "%) 个值被纠正，GPU精度偏差较大";
        }
    }
}

// ============================================================================
// validate_subtask_results - 验证子任务结果
// ============================================================================

bool ResultMerger::validate_subtask_results(const std::vector<SubTaskResult>& results,
                                             const std::string& source_name)
{
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];

        // 检查1：success标志必须为true
        if (!result.success) {
            last_diagnostic_ = source_name + "子任务[" + std::to_string(i)
                             + "] (ID=" + std::to_string(result.subtask_id)
                             + ") 执行失败: " + result.error_message;
            return false;
        }

        // 检查2：data_offset 必须在合理范围内
        // 使用 SIZE_MAX / 2 作为上限，防止溢出
        if (result.data_offset > SIZE_MAX / 2) {
            last_diagnostic_ = source_name + "子任务[" + std::to_string(i)
                             + "] (ID=" + std::to_string(result.subtask_id)
                             + ") data_offset 值异常: "
                             + std::to_string(result.data_offset);
            return false;
        }

        // 检查3：data_length 不能为0
        if (result.data_length == 0) {
            last_diagnostic_ = source_name + "子任务[" + std::to_string(i)
                             + "] (ID=" + std::to_string(result.subtask_id)
                             + ") data_length 为0，没有有效数据";
            return false;
        }

        // 检查4：raw_data 不能为空
        if (result.raw_data.empty()) {
            last_diagnostic_ = source_name + "子任务[" + std::to_string(i)
                             + "] (ID=" + std::to_string(result.subtask_id)
                             + ") raw_data 为空，无法反序列化";
            return false;
        }
    }

    return true;
}

// ============================================================================
// sort_by_offset - 按 data_offset 排序
// ============================================================================

void ResultMerger::sort_by_offset(std::vector<SubTaskResult>& results)
{
    // 使用 lambda 按 data_offset 升序排序
    std::sort(results.begin(), results.end(),
              [](const SubTaskResult& a, const SubTaskResult& b) {
                  return a.data_offset < b.data_offset;
              });
}

// ============================================================================
// check_result_continuity - 检查结果连续性
// ============================================================================

bool ResultMerger::check_result_continuity(const std::vector<SubTaskResult>& results,
                                            size_t total_size)
{
    // 空结果列表无法覆盖任何数据
    if (results.empty()) {
        last_diagnostic_ = "子任务结果列表为空，无法验证连续性";
        return false;
    }

    // 检查1：第一个结果的 data_offset 必须为 0
    if (results[0].data_offset != 0) {
        last_diagnostic_ = "结果连续性检查失败：首个结果的data_offset不为0，实际值="
                         + std::to_string(results[0].data_offset)
                         + "（子任务ID=" + std::to_string(results[0].subtask_id) + "）";
        return false;
    }

    // 检查2：遍历所有结果，验证相邻结果之间没有空隙或重叠
    for (size_t i = 0; i < results.size(); ++i) {
        // 计算当前结果覆盖范围的结束位置（不含）
        size_t expected_end = results[i].data_offset + results[i].data_length;

        if (i + 1 < results.size()) {
            // 如果有下一个结果，检查当前结果的结束位置是否等于下一个结果的起始位置
            size_t next_offset = results[i + 1].data_offset;

            if (expected_end < next_offset) {
                // 有空隙：当前结果结束得太早
                last_diagnostic_ = "结果连续性检查失败：在偏移 "
                                 + std::to_string(expected_end) + " 到 "
                                 + std::to_string(next_offset)
                                 + " 之间存在数据空缺（gap="
                                 + std::to_string(next_offset - expected_end) + "）"
                                 + "（当前子任务ID=" + std::to_string(results[i].subtask_id)
                                 + "，下一子任务ID=" + std::to_string(results[i + 1].subtask_id)
                                 + "）";
                return false;
            }

            if (expected_end > next_offset) {
                // 有重叠：两个结果覆盖了相同的数据区域
                last_diagnostic_ = "结果连续性检查失败：在偏移 "
                                 + std::to_string(next_offset) + " 处存在数据重叠（overlap="
                                 + std::to_string(expected_end - next_offset) + "）"
                                 + "（当前子任务ID=" + std::to_string(results[i].subtask_id)
                                 + "，下一子任务ID=" + std::to_string(results[i + 1].subtask_id)
                                 + "）";
                return false;
            }
            // expected_end == next_offset，数据紧密衔接，正确
        } else {
            // 这是最后一个结果，检查其结束位置是否等于 total_size
            if (expected_end != total_size) {
                last_diagnostic_ = "结果连续性检查失败：最后一个结果结束于 "
                                 + std::to_string(expected_end)
                                 + "，但期望结束于 "
                                 + std::to_string(total_size)
                                 + "（子任务ID=" + std::to_string(results[i].subtask_id) + "）";
                return false;
            }
        }
    }

    return true;
}

// ============================================================================
// deserialize_board_result - 反序列化棋盘评估结果
// ============================================================================

bool ResultMerger::deserialize_board_result(const SubTaskResult& result,
                                             std::vector<int>& legal_moves_counts,
                                             std::vector<int>& liberties,
                                             std::vector<std::vector<int>>& legal_moves)
{
    const auto& data = result.raw_data;
    const size_t data_len = data.size();
    size_t offset = 0;

    // ---- 读取该子任务包含的棋盘数量 ----
    uint32_t boards_count = 0;
    if (!read_uint32_le(data, offset, boards_count, data_len)) {
        last_diagnostic_ = "反序列化棋盘评估结果失败：无法读取棋盘数量（子任务ID="
                         + std::to_string(result.subtask_id) + "）";
        return false;
    }
    offset += sizeof(uint32_t);

    // ---- 验证：棋盘数量应与子任务的 data_length 一致 ----
    if (static_cast<size_t>(boards_count) != result.data_length) {
        last_diagnostic_ = "反序列化棋盘评估结果失败：棋盘数量不匹配"
                         + std::string("，data_length=") + std::to_string(result.data_length)
                         + "，但serialized_count=" + std::to_string(boards_count)
                         + "（子任务ID=" + std::to_string(result.subtask_id) + "）";
        return false;
    }

    // ---- 逐个反序列化每个棋盘的评估数据 ----
    for (uint32_t b = 0; b < boards_count; ++b) {
        // 读取合法着法总数
        int32_t moves_count = 0;
        if (!read_int32_le(data, offset, moves_count, data_len)) {
            last_diagnostic_ = "反序列化棋盘评估结果失败：无法读取棋盘["
                             + std::to_string(b) + "]的合法着法数"
                             + "（子任务ID=" + std::to_string(result.subtask_id) + "）";
            return false;
        }
        offset += sizeof(int32_t);

        // 读取总气数
        int32_t liberty_sum = 0;
        if (!read_int32_le(data, offset, liberty_sum, data_len)) {
            last_diagnostic_ = "反序列化棋盘评估结果失败：无法读取棋盘["
                             + std::to_string(b) + "]的总气数"
                             + "（子任务ID=" + std::to_string(result.subtask_id) + "）";
            return false;
        }
        offset += sizeof(int32_t);

        // 读取实际合法着法数量（用于确定后续要读取的着法列表长度）
        int32_t move_count = 0;
        if (!read_int32_le(data, offset, move_count, data_len)) {
            last_diagnostic_ = "反序列化棋盘评估结果失败：无法读取棋盘["
                             + std::to_string(b) + "]的着法数量"
                             + "（子任务ID=" + std::to_string(result.subtask_id) + "）";
            return false;
        }
        offset += sizeof(int32_t);

        // 读取合法着法位置列表
        std::vector<int> moves;
        if (move_count > 0) {
            moves.reserve(static_cast<size_t>(move_count));
            for (int32_t m = 0; m < move_count; ++m) {
                int32_t move_pos = 0;
                if (!read_int32_le(data, offset, move_pos, data_len)) {
                    last_diagnostic_ = "反序列化棋盘评估结果失败：无法读取棋盘["
                                     + std::to_string(b) + "]的着法[" + std::to_string(m) + "]"
                                     + "（子任务ID=" + std::to_string(result.subtask_id) + "）";
                    return false;
                }
                offset += sizeof(int32_t);
                moves.push_back(static_cast<int>(move_pos));
            }
        }

        // 追加到合并结果向量中
        legal_moves_counts.push_back(static_cast<int>(moves_count));
        liberties.push_back(static_cast<int>(liberty_sum));
        legal_moves.push_back(std::move(moves));
    }

    return true;
}

// ============================================================================
// deserialize_inference_result - 反序列化神经网络推理结果
// ============================================================================

bool ResultMerger::deserialize_inference_result(const SubTaskResult& result,
                                                 std::vector<float>& policy,
                                                 std::vector<float>& value,
                                                 int policy_dim)
{
    const auto& data = result.raw_data;
    const size_t data_len = data.size();
    size_t offset = 0;

    // ---- 读取该子任务包含的样本数量 ----
    uint32_t samples_count = 0;
    if (!read_uint32_le(data, offset, samples_count, data_len)) {
        last_diagnostic_ = "反序列化推理结果失败：无法读取样本数量（子任务ID="
                         + std::to_string(result.subtask_id) + "）";
        return false;
    }
    offset += sizeof(uint32_t);

    // ---- 验证：样本数量应与子任务的 data_length 一致 ----
    if (static_cast<size_t>(samples_count) != result.data_length) {
        last_diagnostic_ = "反序列化推理结果失败：样本数量不匹配"
                         + std::string("，data_length=") + std::to_string(result.data_length)
                         + "，但serialized_count=" + std::to_string(samples_count)
                         + "（子任务ID=" + std::to_string(result.subtask_id) + "）";
        return false;
    }

    // ---- 验证：raw_data 总长度是否足够容纳所有样本数据 ----
    // 每个样本占用: policy_dim * sizeof(float) + sizeof(float)
    const size_t bytes_per_sample = static_cast<size_t>(policy_dim) * sizeof(float)
                                    + sizeof(float);
    const size_t expected_total_bytes = sizeof(uint32_t)
                                        + static_cast<size_t>(samples_count) * bytes_per_sample;
    if (data_len < expected_total_bytes) {
        last_diagnostic_ = "反序列化推理结果失败：raw_data长度不足，期望>="
                         + std::to_string(expected_total_bytes)
                         + "，实际=" + std::to_string(data_len)
                         + "（子任务ID=" + std::to_string(result.subtask_id) + "）";
        return false;
    }

    // ---- 逐个反序列化每个样本的推理数据 ----
    for (uint32_t s = 0; s < samples_count; ++s) {
        // 读取策略输出（policy_dim 个 float 值）
        for (int p = 0; p < policy_dim; ++p) {
            float policy_val = 0.0f;
            if (!read_float_le(data, offset, policy_val, data_len)) {
                last_diagnostic_ = "反序列化推理结果失败：无法读取样本["
                                 + std::to_string(s) + "]的策略值[" + std::to_string(p) + "]"
                                 + "（子任务ID=" + std::to_string(result.subtask_id) + "）";
                return false;
            }
            offset += sizeof(float);
            policy.push_back(policy_val);
        }

        // 读取价值输出（1个 float 值）
        float value_val = 0.0f;
        if (!read_float_le(data, offset, value_val, data_len)) {
            last_diagnostic_ = "反序列化推理结果失败：无法读取样本["
                             + std::to_string(s) + "]的价值输出"
                             + "（子任务ID=" + std::to_string(result.subtask_id) + "）";
            return false;
        }
        offset += sizeof(float);
        value.push_back(value_val);
    }

    return true;
}

// ============================================================================
// deserialize_topology_result - 反序列化拓扑计算结果
// ============================================================================

bool ResultMerger::deserialize_topology_result(const SubTaskResult& result,
                                                std::vector<double>& connectivity,
                                                std::vector<double>& centrality,
                                                std::vector<double>& strategic_value)
{
    const auto& data = result.raw_data;
    const size_t data_len = data.size();
    size_t offset = 0;

    // ---- 读取该子任务包含的区域数量 ----
    uint32_t regions_count = 0;
    if (!read_uint32_le(data, offset, regions_count, data_len)) {
        last_diagnostic_ = "反序列化拓扑结果失败：无法读取区域数量（子任务ID="
                         + std::to_string(result.subtask_id) + "）";
        return false;
    }
    offset += sizeof(uint32_t);

    // ---- 验证：区域数量应与子任务的 data_length 一致 ----
    if (static_cast<size_t>(regions_count) != result.data_length) {
        last_diagnostic_ = "反序列化拓扑结果失败：区域数量不匹配"
                         + std::string("，data_length=") + std::to_string(result.data_length)
                         + "，但serialized_count=" + std::to_string(regions_count)
                         + "（子任务ID=" + std::to_string(result.subtask_id) + "）";
        return false;
    }

    // ---- 验证：raw_data 总长度是否足够容纳所有区域数据 ----
    // 每个区域占用: 3 * sizeof(double) = 24 字节
    const size_t bytes_per_region = 3 * sizeof(double);
    const size_t expected_total_bytes = sizeof(uint32_t)
                                        + static_cast<size_t>(regions_count) * bytes_per_region;
    if (data_len < expected_total_bytes) {
        last_diagnostic_ = "反序列化拓扑结果失败：raw_data长度不足，期望>="
                         + std::to_string(expected_total_bytes)
                         + "，实际=" + std::to_string(data_len)
                         + "（子任务ID=" + std::to_string(result.subtask_id) + "）";
        return false;
    }

    // ---- 逐个反序列化每个区域的拓扑特征 ----
    for (uint32_t r = 0; r < regions_count; ++r) {
        // 读取连通性分数（double, 8字节）
        double conn_val = 0.0;
        if (!read_double_le(data, offset, conn_val, data_len)) {
            last_diagnostic_ = "反序列化拓扑结果失败：无法读取区域["
                             + std::to_string(r) + "]的连通性分数"
                             + "（子任务ID=" + std::to_string(result.subtask_id) + "）";
            return false;
        }
        offset += sizeof(double);
        connectivity.push_back(conn_val);

        // 读取中心性分数（double, 8字节）
        double cent_val = 0.0;
        if (!read_double_le(data, offset, cent_val, data_len)) {
            last_diagnostic_ = "反序列化拓扑结果失败：无法读取区域["
                             + std::to_string(r) + "]的中心性分数"
                             + "（子任务ID=" + std::to_string(result.subtask_id) + "）";
            return false;
        }
        offset += sizeof(double);
        centrality.push_back(cent_val);

        // 读取战略价值（double, 8字节）
        double strat_val = 0.0;
        if (!read_double_le(data, offset, strat_val, data_len)) {
            last_diagnostic_ = "反序列化拓扑结果失败：无法读取区域["
                             + std::to_string(r) + "]的战略价值"
                             + "（子任务ID=" + std::to_string(result.subtask_id) + "）";
            return false;
        }
        offset += sizeof(double);
        strategic_value.push_back(strat_val);
    }

    return true;
}

// ============================================================================
// 二进制读取辅助方法 - 小端序数据读取
// ============================================================================

// ---- 读取 uint32_t（4字节，小端序） ----
bool ResultMerger::read_uint32_le(const std::vector<uint8_t>& data,
                                   size_t offset,
                                   uint32_t& value,
                                   size_t max_len)
{
    // 边界检查：确保有4个字节可读
    if (offset + sizeof(uint32_t) > max_len) {
        return false;
    }

    // 小端序：最低有效字节在最低地址
    value = (static_cast<uint32_t>(data[offset]))
          | (static_cast<uint32_t>(data[offset + 1]) << 8)
          | (static_cast<uint32_t>(data[offset + 2]) << 16)
          | (static_cast<uint32_t>(data[offset + 3]) << 24);

    return true;
}

// ---- 读取 int32_t（4字节，小端序） ----
bool ResultMerger::read_int32_le(const std::vector<uint8_t>& data,
                                  size_t offset,
                                  int32_t& value,
                                  size_t max_len)
{
    // 边界检查
    if (offset + sizeof(int32_t) > max_len) {
        return false;
    }

    // 先读取为 uint32_t，然后强制转换为 int32_t（二进制位相同）
    uint32_t raw = 0;
    raw = (static_cast<uint32_t>(data[offset]))
        | (static_cast<uint32_t>(data[offset + 1]) << 8)
        | (static_cast<uint32_t>(data[offset + 2]) << 16)
        | (static_cast<uint32_t>(data[offset + 3]) << 24);

    value = static_cast<int32_t>(raw);
    return true;
}

// ---- 读取 float（4字节，小端序，IEEE 754 binary32） ----
bool ResultMerger::read_float_le(const std::vector<uint8_t>& data,
                                  size_t offset,
                                  float& value,
                                  size_t max_len)
{
    // 边界检查
    if (offset + sizeof(float) > max_len) {
        return false;
    }

    // float 在内存中为32位IEEE 754格式
    // 先读取为 uint32_t，然后通过 memcpy 安全地转换为 float
    // 这是符合C++标准的方式（不使用未定义行为的类型双关）
    uint32_t raw = 0;
    raw = (static_cast<uint32_t>(data[offset]))
        | (static_cast<uint32_t>(data[offset + 1]) << 8)
        | (static_cast<uint32_t>(data[offset + 2]) << 16)
        | (static_cast<uint32_t>(data[offset + 3]) << 24);

    std::memcpy(&value, &raw, sizeof(float));
    return true;
}

// ---- 读取 double（8字节，小端序，IEEE 754 binary64） ----
bool ResultMerger::read_double_le(const std::vector<uint8_t>& data,
                                   size_t offset,
                                   double& value,
                                   size_t max_len)
{
    // 边界检查
    if (offset + sizeof(double) > max_len) {
        return false;
    }

    // double 在内存中为64位IEEE 754格式
    // 先读取为 uint64_t，然后通过 memcpy 安全地转换为 double
    uint64_t raw = 0;
    raw = (static_cast<uint64_t>(data[offset]))
        | (static_cast<uint64_t>(data[offset + 1]) << 8)
        | (static_cast<uint64_t>(data[offset + 2]) << 16)
        | (static_cast<uint64_t>(data[offset + 3]) << 24)
        | (static_cast<uint64_t>(data[offset + 4]) << 32)
        | (static_cast<uint64_t>(data[offset + 5]) << 40)
        | (static_cast<uint64_t>(data[offset + 6]) << 48)
        | (static_cast<uint64_t>(data[offset + 7]) << 56);

    std::memcpy(&value, &raw, sizeof(double));
    return true;
}

}  // namespace heterogeneous
}  // namespace go
}  // namespace spacego