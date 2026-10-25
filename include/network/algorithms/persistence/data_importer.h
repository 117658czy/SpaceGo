#pragma once

// 数据导入器头文件
// 职责：负责将导出的结构化文本数据重新导入到HMGS-DP算法中
// 支持搜索树、合并网络、多样性地图等数据的完整恢复
// 提供向后兼容性支持，可处理不同版本的导出格式

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>

#include "network/algorithms/hmgs_node.h"
#include "network/algorithms/config/hmgs_config.h"

namespace spacego {
namespace algorithms {

// 数据导入器类
class DataImporter {
public:
    // 构造函数和析构函数
    DataImporter();
    ~DataImporter();

    // 初始化
    void initialize();
    
    // 核心导入方法
    
    // 导入完整的搜索树结构
    // 从结构化文本重建节点注册表、根节点、主分支列表等完整数据
    // 返回是否成功导入，通过输出参数返回重建的数据结构
    bool import_search_tree(
        const std::string& data,
        std::shared_ptr<HMGSNode>& root_node,
        std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
        std::vector<std::shared_ptr<HMGSNode>>& main_branches,
        HMGSStatistics& stats);
    
    // 导入合并网络信息
    // 恢复合并概率、被合并概率、合并关系记录等数据
    bool import_merge_network(
        const std::string& data,
        std::unordered_map<std::string, double>& merge_probabilities,
        std::unordered_map<std::string, double>& merged_into_probabilities,
        std::unordered_map<std::string, std::vector<std::string>>& merge_relationships,
        HMGSStatistics& stats);
    
    // 导入多样性保护地图
    // 恢复被忽视分支列表、未合并计数、访问时间分布等数据
    bool import_diversity_map(
        const std::string& data,
        std::vector<std::string>& neglected_branches,
        std::unordered_map<std::string, size_t>& unmerged_counts,
        std::unordered_map<std::string, std::chrono::steady_clock::time_point>& last_visit_times,
        HMGSStatistics& stats);
    
    // 向后兼容性方法：支持V1.0版本的旧格式导入
    
    // 导入V1.0格式的搜索树（简化版）
    bool import_search_tree_v1(
        const std::string& data,
        std::shared_ptr<HMGSNode>& root_node,
        std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
        HMGSStatistics& stats);
    
    // 导入V1.0格式的合并网络（简化版）
    bool import_merge_network_v1(
        const std::string& data,
        std::unordered_map<std::string, double>& merge_probabilities,
        HMGSStatistics& stats);
    
    // 导入V1.0格式的多样性地图（简化版）
    bool import_diversity_map_v1(
        const std::string& data,
        std::vector<std::string>& neglected_branches,
        HMGSStatistics& stats);

private:
    // 线程安全锁
    mutable std::mutex mutex_;
    
    // 是否已初始化
    std::atomic<bool> initialized_{false};
    
    // 辅助解析方法
    
    // 解析键值对行
    // 格式: "KEY:VALUE" -> 返回pair(KEY, VALUE)
    std::pair<std::string, std::string> parse_key_value(const std::string& line) const;
    
    // 分割字符串为子串列表
    // 根据分隔符将字符串分割为多个部分
    std::vector<std::string> split_string(const std::string& str, char delimiter) const;
    
    // 重建节点间的父子关系
    // 在所有节点创建后，根据父节点ID建立关系连接
    void rebuild_node_relationships(
        std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
        const std::unordered_map<std::string, std::string>& parent_map);
};

} // namespace algorithms
} // namespace spacego
