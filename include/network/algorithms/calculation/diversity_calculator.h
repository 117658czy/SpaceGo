#pragma once

// 多样性计算器头文件
// 职责：负责搜索树中多样性相关的所有计算和调整操作
// 包括新颖性评估、多样性保护、权重调整等核心功能

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <chrono>

#include "network/algorithms/config/hmgs_config.h"
#include "network/algorithms/hmgs_node.h"

namespace spacego {
namespace algorithms {

// 前向声明
class HMGSStatistics;

// 多样性计算器类
class DiversityCalculator {
public:
    // 构造函数和析构函数
    DiversityCalculator();
    ~DiversityCalculator();

    // 初始化
    void initialize(const HMGSConfig& config);
    
    // 核心多样性计算方法
    
    // 计算节点的新颖性价值
    // 综合考虑状态唯一性、价值差异、结构位置和时间因素
    // 返回值范围：[0.0, 1.0]，值越大表示越新颖
    double calculate_novelty_value(
        std::shared_ptr<HMGSNode> node,
        const std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
        const HMGSStatistics& stats) const;
    
    // 保护多样性分支
    // 将节点标记为受保护状态，增加探索激励
    // 防止有价值但被忽视的分支被过早合并或丢弃
    void protect_diversity_branch(
        std::shared_ptr<HMGSNode> node,
        const HMGSConfig& config,
        HMGSStatistics& stats,
        std::unordered_map<std::string, std::chrono::steady_clock::time_point>& last_visit_times);
    
    // 根据一致性水平调整权重
    // 低一致性 -> 高多样性需求 -> 增加探索权重
    // 高一致性 -> 低多样性需求 -> 减少探索权重
    void adjust_weights_for_diversity(
        std::shared_ptr<HMGSNode> branch,
        double consistency_level,
        const HMGSConfig& config,
        HMGSStatistics& stats);
    
    // 计算当前整体的多样性分数
    // 基于节点的平均新颖性、价值分布、访问分布等指标
    // 返回值范围：[0.0, 1.0]
    double calculate_current_diversity_score(
        const std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
        const std::vector<std::shared_ptr<HMGSNode>>& main_branches,
        std::shared_ptr<HMGSNode> root_node) const;
    
    // 识别被忽视的分支
    // 基于访问次数、最后访问时间、合并概率、价值改进等条件判断
    // 返回被忽视分支的ID列表
    std::vector<std::string> identify_neglected_branches(
        const std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
        std::shared_ptr<HMGSNode> root_node,
        const HMGSConfig& config,
        const HMGSStatistics& stats);
    
    // 为被忽视的分支分配资源
    // 均匀分配模拟次数到各被忽视分支，增加其优先权
    void allocate_resources_to_neglected_branches(
        const std::vector<std::string>& neglected_branches,
        double total_resources,
        std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
        HMGSStatistics& stats,
        std::unordered_map<std::string, std::chrono::steady_clock::time_point>& last_visit_times);
    
    // 调整多样性参数
    // 基于当前多样性水平动态调整探索参数
    void adjust_diversity_parameters(
        HMGSConfig& config,
        const std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
        std::shared_ptr<HMGSNode> root_node,
        HMGSStatistics& stats);
    
    // 选择回溯节点
    // 从候选节点中选择最适合回溯的节点（高不确定性、高新颖性、低访问次数）
    std::shared_ptr<HMGSNode> select_backtrack_node(
        const std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
        std::shared_ptr<HMGSNode> root_node,
        const std::vector<std::string>& neglected_branches) const;
    
    // 执行回溯操作
    // 重置节点统计信息以鼓励重新探索
    void perform_backtrack(
        std::shared_ptr<HMGSNode> backtrack_node,
        HMGSStatistics& stats,
        std::vector<std::string>& neglected_branches,
        std::unordered_map<std::string, std::chrono::steady_clock::time_point>& last_visit_times);
    
    // 重新发现被忽视的分支
    // 重置分支统计信息并标记为受保护
    void rediscover_neglected_branches(
        const std::vector<std::string>& neglected_branches,
        std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
        HMGSStatistics& stats,
        std::unordered_map<std::string, std::chrono::steady_clock::time_point>& last_visit_times);

private:
    // 配置
    HMGSConfig config_;
    
    // 线程安全锁
    mutable std::mutex mutex_;
    
    // 是否已初始化
    std::atomic<bool> initialized_{false};
};

} // namespace algorithms
} // namespace spacego
