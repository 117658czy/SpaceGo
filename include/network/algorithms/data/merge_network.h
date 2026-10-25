#pragma once

// 合并网络头文件
// 职责：负责搜索树中节点合并的所有操作
// 包括合并检测、合并决策、概率计算和网络重建等核心功能

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>

#include "network/algorithms/config/hmgs_config.h"
#include "network/algorithms/hmgs_node.h"

namespace spacego {
namespace algorithms {

// 前向声明
class HMGSStatistics;

// 合并网络类
class MergeNetwork {
public:
    // 构造函数和析构函数
    MergeNetwork();
    ~MergeNetwork();

    // 初始化
    void initialize(const HMGSConfig& config);
    
    // 核心合并方法
    
    // 检查是否需要执行合并操作
    // 遍历所有候选节点对，判断是否应该进行合并
    bool check_for_merges(
        std::shared_ptr<HMGSNode> root_node,
        std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
        const HMGSConfig& config,
        HMGSStatistics& stats,
        std::unordered_map<std::string, double>& merge_probabilities,
        std::unordered_map<std::string, size_t>& unmerged_counts,
        std::unordered_map<std::string, std::chrono::steady_clock::time_point>& last_visit_times);
    
    // 判断两个节点是否应该合并
    // 综合考虑状态相似性、价值兼容性、拓扑一致性等因素
    bool should_merge_nodes(
        std::shared_ptr<HMGSNode> node1,
        std::shared_ptr<HMGSNode> node2,
        const HMGSConfig& config,
        const std::unordered_map<std::string, std::vector<std::string>>& merge_relationships) const;
    
    // 执行两个节点的合并操作
    // 将次节点的统计信息、子节点等合并到主节点，并标记次节点为已合并
    std::shared_ptr<HMGSNode> merge_nodes(
        std::shared_ptr<HMGSNode> node1,
        std::shared_ptr<HMGSNode> node2,
        const HMGSConfig& config,
        std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
        HMGSStatistics& stats,
        std::unordered_map<std::string, double>& merge_probabilities,
        std::unordered_map<std::string, double>& merged_into_probabilities,
        std::unordered_map<std::string, std::vector<std::string>>& merge_relationships,
        std::unordered_map<std::string, size_t>& unmerged_counts,
        std::unordered_map<std::string, std::chrono::steady_clock::time_point>& last_visit_times);
    
    // 更新单个节点的合并概率
    // 基于访问频率、价值、不确定性、新颖性和拓扑一致性综合计算
    void update_merge_probabilities(
        std::shared_ptr<HMGSNode> node,
        const HMGSConfig& config,
        const std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
        std::unordered_map<std::string, double>& merge_probabilities,
        std::unordered_map<std::string, double>& merged_into_probabilities,
        const std::unordered_map<std::string, std::vector<std::string>>& merge_relationships);
    
    // 强制执行一次合并检查
    // 公共接口方法，内部调用check_for_merges
    void force_merge_check(
        std::shared_ptr<HMGSNode> root_node,
        std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
        const HMGSConfig& config,
        HMGSStatistics& stats,
        std::unordered_map<std::string, double>& merge_probabilities,
        std::unordered_map<std::string, size_t>& unmerged_counts,
        std::unordered_map<std::string, std::chrono::steady_clock::time_point>& last_visit_times);
    
    // 重新计算所有节点的合并概率
    // 遍历节点注册表，重新评估每个节点的合并概率
    void recalculate_merge_probabilities(
        const std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
        const HMGSConfig& config,
        HMGSStatistics& stats,
        std::unordered_map<std::string, double>& merge_probabilities,
        std::unordered_map<std::string, double>& merged_into_probabilities,
        const std::unordered_map<std::string, std::vector<std::string>>& merge_relationships);
    
    // 重建整个合并网络
    // 清空现有数据并基于当前节点状态重新构建完整的合并网络
    void rebuild_merge_network(
        const std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
        const HMGSConfig& config,
        HMGSStatistics& stats,
        std::unordered_map<std::string, double>& merge_probabilities,
        std::unordered_map<std::string, double>& merged_into_probabilities,
        std::unordered_map<std::string, std::vector<std::string>>& merge_relationships);

private:
    // 配置
    HMGSConfig config_;
    
    // 线程安全锁
    mutable std::mutex mutex_;
    
    // 是否已初始化
    std::atomic<bool> initialized_{false};
    
    // 计算状态相似性（辅助方法）
    double calculate_state_similarity(
        const std::string& state1,
        const std::string& state2) const;
    
    // 计算价值兼容性（辅助方法）
    double calculate_value_compatibility(double value1, double value2) const;
    
    // 计算拓扑一致性（辅助方法）
    double calculate_topology_consistency(
        std::shared_ptr<HMGSNode> node1,
        std::shared_ptr<HMGSNode> node2,
        const std::unordered_map<std::string, std::vector<std::string>>& merge_relationships) const;
};

} // namespace algorithms
} // namespace spacego
