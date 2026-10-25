#pragma once

// 数据导出器头文件
// 职责：负责将HMGS-DP算法的所有运行时数据导出为结构化文本格式
// 支持搜索树、合并网络、多样性地图等数据的完整导出

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>

#include "network/algorithms/hmgs_node.h"

// 前向声明
namespace spacego {
namespace algorithms {
struct HMGSStatistics;
}
}

namespace spacego {
namespace algorithms {

// 数据导出器类
class DataExporter {
public:
    // 构造函数和析构函数
    DataExporter();
    ~DataExporter();

    // 初始化
    void initialize();
    
    // 核心导出方法
    
    // 导出完整的搜索树结构
    // 包含所有节点信息、父子关系、统计属性等完整数据
    // 返回格式化的文本字符串，可用于持久化存储或调试分析
    std::string export_search_tree(
        std::shared_ptr<HMGSNode> root_node,
        const std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
        const std::vector<std::shared_ptr<HMGSNode>>& main_branches,
        const HMGSStatistics& stats) const;
    
    // 导出合并网络信息
    // 包含所有节点的合并概率、被合并概率、合并关系记录等
    std::string export_merge_network(
        const std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
        const std::unordered_map<std::string, double>& merge_probabilities,
        const std::unordered_map<std::string, double>& merged_into_probabilities,
        const std::unordered_map<std::string, std::vector<std::string>>& merge_relationships,
        const HMGSStatistics& stats) const;
    
    // 导出多样性保护地图
    // 包含被忽视分支列表、未合并计数、访问时间分布、多样性指标等
    std::string export_diversity_map(
        const std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
        const std::vector<std::shared_ptr<HMGSNode>>& main_branches,
        std::shared_ptr<HMGSNode> root_node,
        const std::vector<std::string>& neglected_branches,
        const std::unordered_map<std::string, size_t>& unmerged_counts,
        const std::unordered_map<std::string, std::chrono::steady_clock::time_point>& last_visit_times,
        const HMGSStatistics& stats) const;

private:
    // 线程安全锁
    mutable std::mutex mutex_;
    
    // 是否已初始化
    std::atomic<bool> initialized_{false};
};

} // namespace algorithms
} // namespace spacego
