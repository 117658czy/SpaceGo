#pragma once

// 节点管理器头文件
// 职责：负责搜索树中所有节点的生命周期管理
// 包括节点创建、扩展、价值回溯等核心操作

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <functional>

#include "network/algorithms/config/hmgs_config.h"
#include "network/algorithms/hmgs_node.h"

namespace spacego {
namespace algorithms {

// 节点管理器类
class NodeManager {
public:
    // 构造函数和析构函数
    NodeManager();
    ~NodeManager();

    // 初始化
    void initialize(const HMGSConfig& config);
    
    // 核心节点管理方法
    
    // 创建根节点
    // 根节点是搜索树的入口，包含分支管理功能
    std::shared_ptr<HMGSRootNode> create_root_node(
        const std::string& state,
        std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry);
    
    // 创建普通节点
    // 为指定状态创建新节点，并建立父子关系
    std::shared_ptr<HMGSNode> create_node(
        const std::string& state,
        std::shared_ptr<HMGSNode> parent,
        std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
        std::unordered_map<std::string, std::chrono::steady_clock::time_point>& last_visit_times);
    
    // 扩展节点
    // 为未扩展的节点生成所有合法子节点，使用均匀先验概率
    void expand_node(
        std::shared_ptr<HMGSNode> node,
        const HMGSConfig& config,
        std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
        HMGSStatistics& stats,
        std::function<std::vector<std::string>(const std::string&)> generate_legal_moves_fn);
    
    // 回溯价值
    // 从叶节点沿路径回溯更新价值到根节点
    void backup_value(
        std::shared_ptr<HMGSNode> node,
        double value,
        const HMGSConfig& config,
        HMGSStatistics& stats,
        std::unordered_map<std::string, std::chrono::steady_clock::time_point>& last_visit_times);
    
    // 选择回溯路径
    // 支持多种回溯策略：最佳价值路径、拓扑感知路径等
    std::vector<std::shared_ptr<HMGSNode>> select_backup_path(
        std::shared_ptr<HMGSNode> leaf_node,
        const HMGSConfig& config) const;
    
    // 选择最佳子节点
    // 使用拓扑感知UCT算法选择最有潜力的子节点进行扩展
    std::shared_ptr<HMGSNode> select_best_child(
        std::shared_ptr<HMGSNode> node,
        const HMGSConfig& config) const;
    
    // 选择叶子节点
    // 从起始节点开始，沿着最佳子节点路径选择直到叶子节点
    std::shared_ptr<HMGSNode> select_leaf_node(
        std::shared_ptr<HMGSNode> start_node,
        const HMGSConfig& config) const;

private:
    // 配置
    HMGSConfig config_;
    
    // 线程安全锁
    mutable std::mutex mutex_;
    
    // 是否已初始化
    std::atomic<bool> initialized_{false};
    
    // 计算拓扑感知UCT值
    double calculate_topology_uct(
        std::shared_ptr<HMGSNode> child,
        std::shared_ptr<HMGSNode> parent,
        size_t child_index,
        const HMGSConfig& config) const;
};

} // namespace algorithms
} // namespace spacego
