// 数据导出器实现文件
// 包含所有数据导出操作的具体实现

#include "network/algorithms/persistence/data_exporter.h"
#include "network/algorithms/config/hmgs_config.h"
#include "framework/logger.h"
#include <sstream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace spacego {
namespace algorithms {

// 构造函数
DataExporter::DataExporter()
    : initialized_(false) {
}

// 析构函数
DataExporter::~DataExporter() {
}

// 初始化
void DataExporter::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    initialized_.store(true, std::memory_order_release);
    
    LOG_INFO("数据导出器初始化完成");
}

// 导出完整的搜索树结构
std::string DataExporter::export_search_tree(
    std::shared_ptr<HMGSNode> root_node,
    const std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
    const std::vector<std::shared_ptr<HMGSNode>>& main_branches,
    const HMGSStatistics& stats) const {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 创建字符串流用于构建导出文本
    std::stringstream ss;
    
    // 写入文件头和版本信息
    ss << "SEARCH_TREE_EXPORT_V2.0\n";
    ss << "FORMAT:STRUCTURED_GO_TREE\n";
    ss << "TIMESTAMP:" << std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() << "\n";
    
    // 导出树元数据（全局统计信息）
    ss << "METADATA_BEGIN\n";
    ss << "NODE_COUNT:" << node_registry.size() << "\n";
    ss << "ROOT_NODE:" << (root_node ? root_node->get_id() : "NULL") << "\n";
    ss << "MAIN_BRANCHES:" << main_branches.size() << "\n";
    ss << "TOTAL_SIMULATIONS:" << stats.total_simulations << "\n";
    ss << "COMPLETED_SIMULATIONS:" << stats.completed_simulations << "\n";
    ss << "SUCCESSFUL_MERGES:" << stats.successful_merges << "\n";
    ss << "FAILED_MERGES:" << stats.failed_merges << "\n";
    ss << "NEGLECTED_BRANCHES:" << stats.neglected_branches << "\n";
    ss << "BACKTRACK_EVENTS:" << stats.backtrack_events << "\n";
    ss << "METADATA_END\n";
    
    // 导出每个节点的详细信息
    ss << "NODES_BEGIN\n";
    for (const auto& entry : node_registry) {
        const auto& node = entry.second;
        if (!node) continue;  // 跳过空节点
        
        std::string node_id = node->get_id();
        
        // 获取父节点ID（如果存在）
        std::string parent_id = node->get_parent() ? node->get_parent()->get_id() : "NULL";
        
        // 节点基本信息
        ss << "NODE_START:" << node_id << "\n";
        ss << "PARENT:" << parent_id << "\n";
        ss << "DEPTH:" << node->get_depth() << "\n";
        ss << "STATE:" << node->get_state() << "\n";
        
        // 访问统计信息
        ss << "VISITS:" << node->get_visit_count() << "\n";
        ss << "TOTAL_VALUE:" << node->get_total_value() << "\n";
        ss << "AVERAGE_VALUE:" << std::fixed << std::setprecision(6) 
           << node->get_average_value() << "\n";
        
        // 合并相关属性
        ss << "MERGE_PROB:" << std::fixed << std::setprecision(6) 
           << node->get_merge_prob() << "\n";
        ss << "MERGED_INTO_PROB:" << std::fixed << std::setprecision(6) 
           << node->get_merged_into_prob() << "\n";
        ss << "CONTINUOUS_UNMERGED:" << node->get_continuous_unmerged() << "\n";
        
        // 多样性相关属性
        ss << "NOVELTY_SCORE:" << std::fixed << std::setprecision(6) 
           << node->get_novelty_score() << "\n";
        ss << "BACKTRACK_COUNT:" << node->get_backtrack_count() << "\n";
        ss << "UNCERTAINTY:" << std::fixed << std::setprecision(6) 
           << node->get_uncertainty() << "\n";
        
        // 节点状态标志
        ss << "IS_EXPANDED:" << (node->is_expanded() ? "true" : "false") << "\n";
        ss << "IS_LEAF:" << (node->is_leaf() ? "true" : "false") << "\n";
        ss << "IS_REPLICA:" << (node->is_replica() ? "true" : "false") << "\n";
        
        // 分支和副本管理属性
        ss << "BRANCH_INDEX:" << node->get_branch_index() << "\n";
        ss << "REPLICA_INDEX:" << node->get_replica_index() << "\n";
        ss << "REPLICA_COUNT:" << node->get_replica_count() << "\n";
        ss << "BRANCH_QUALITY:" << std::fixed << std::setprecision(6) 
           << node->get_branch_quality() << "\n";
        
        // 动态调整参数
        ss << "EXPLORATION_BIAS:" << std::fixed << std::setprecision(6) 
           << node->get_exploration_bias() << "\n";
        ss << "TOPOLOGY_WEIGHT:" << std::fixed << std::setprecision(6) 
           << node->get_topology_weight() << "\n";
        ss << "NOVELTY_REWARD:" << std::fixed << std::setprecision(6) 
           << node->get_novelty_reward() << "\n";
        
        // 子节点列表
        const auto& children = node->get_children();
        if (!children.empty()) {
            ss << "CHILDREN:";
            for (size_t i = 0; i < children.size(); ++i) {
                if (children[i]) {
                    ss << children[i]->get_id();
                    if (i < children.size() - 1) ss << ",";
                }
            }
            ss << "\n";
        }
        
        // 相似节点列表
        const auto& similar_nodes = node->get_similar_nodes();
        if (!similar_nodes.empty()) {
            ss << "SIMILAR_NODES:";
            for (size_t i = 0; i < similar_nodes.size(); ++i) {
                if (similar_nodes[i]) {
                    ss << similar_nodes[i]->get_id();
                    if (i < similar_nodes.size() - 1) ss << ",";
                }
            }
            ss << "\n";
        }
        
        // 走法和先验概率
        const auto& moves = node->get_moves();
        const auto& prior_probs = node->get_prior_probs();
        if (!moves.empty() && !prior_probs.empty() && moves.size() == prior_probs.size()) {
            ss << "MOVES_BEGIN\n";
            for (size_t i = 0; i < moves.size(); ++i) {
                ss << "MOVE:" << moves[i] << ":PROB:" << std::fixed << std::setprecision(6) 
                   << prior_probs[i] << "\n";
            }
            ss << "MOVES_END\n";
        }
        
        // 状态编码向量（如果可用）
        auto state_encoding = node->get_state_encoding();
        if (!state_encoding.empty()) {
            ss << "STATE_ENCODING_BEGIN\n";
            for (size_t i = 0; i < state_encoding.size(); ++i) {
                ss << std::fixed << std::setprecision(6) << state_encoding[i];
                if (i < state_encoding.size() - 1) ss << ",";
            }
            ss << "\nSTATE_ENCODING_END\n";
        }
        
        ss << "NODE_END:" << node_id << "\n";
    }
    ss << "NODES_END\n";
    
    // 导出主分支信息
    ss << "MAIN_BRANCHES_BEGIN\n";
    for (size_t i = 0; i < main_branches.size(); ++i) {
        if (main_branches[i]) {
            ss << "BRANCH:" << i << ":ROOT:" << main_branches[i]->get_id() << "\n";
        }
    }
    ss << "MAIN_BRANCHES_END\n";
    
    ss << "SEARCH_TREE_EXPORT_END\n";
    
    LOG_DEBUG("搜索树导出完成: %zu个节点", node_registry.size());
    
    return ss.str();
}

// 导出合并网络信息
std::string DataExporter::export_merge_network(
    const std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
    const std::unordered_map<std::string, double>& merge_probabilities,
    const std::unordered_map<std::string, double>& merged_into_probabilities,
    const std::unordered_map<std::string, std::vector<std::string>>& merge_relationships,
    const HMGSStatistics& stats) const {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::stringstream ss;
    
    // 文件头信息
    ss << "MERGE_NETWORK_EXPORT_V2.0\n";
    ss << "FORMAT:STRUCTURED_MERGE_NETWORK\n";
    ss << "TIMESTAMP:" << std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() << "\n";
    
    // 合并网络元数据
    ss << "METADATA_BEGIN\n";
    ss << "TOTAL_NODES:" << node_registry.size() << "\n";
    ss << "MERGE_RELATIONSHIPS:" << merge_relationships.size() << "\n";
    ss << "MERGE_PROBABILITIES:" << merge_probabilities.size() << "\n";
    ss << "MERGED_INTO_PROBABILITIES:" << merged_into_probabilities.size() << "\n";
    ss << "METADATA_END\n";
    
    // 导出合并关系记录
    ss << "MERGE_RELATIONSHIPS_BEGIN\n";
    for (const auto& entry : merge_relationships) {
        const std::string& node_id = entry.first;
        const auto& relationships = entry.second;
        
        if (!relationships.empty()) {
            ss << "NODE:" << node_id << ":MERGE_TARGETS:";
            for (size_t i = 0; i < relationships.size(); ++i) {
                ss << relationships[i];
                if (i < relationships.size() - 1) ss << ",";
            }
            ss << "\n";
        }
    }
    ss << "MERGE_RELATIONSHIPS_END\n";
    
    // 导出合并概率映射
    ss << "MERGE_PROBABILITIES_BEGIN\n";
    for (const auto& entry : merge_probabilities) {
        ss << "NODE:" << entry.first << ":PROB:" << std::fixed << std::setprecision(6) 
           << entry.second << "\n";
    }
    ss << "MERGE_PROBABILITIES_END\n";
    
    // 导出被合并概率映射
    ss << "MERGED_INTO_PROBABILITIES_BEGIN\n";
    for (const auto& entry : merged_into_probabilities) {
        ss << "NODE:" << entry.first << ":MERGED_INTO_PROB:" << std::fixed << std::setprecision(6) 
           << entry.second << "\n";
    }
    ss << "MERGED_INTO_PROBABILITIES_END\n";
    
    // 导出每个节点的合并状态快照
    ss << "NODE_MERGE_STATES_BEGIN\n";
    for (const auto& entry : node_registry) {
        const auto& node = entry.second;
        if (!node) continue;
        
        std::string node_id = node->get_id();
        
        ss << "NODE:" << node_id 
           << ":NODE_MERGE_PROB:" << std::fixed << std::setprecision(6) << node->get_merge_prob()
           << ":NODE_MERGED_INTO_PROB:" << std::fixed << std::setprecision(6) << node->get_merged_into_prob()
           << ":CONTINUOUS_UNMERGED:" << node->get_continuous_unmerged() << "\n";
    }
    ss << "NODE_MERGE_STATES_END\n";
    
    // 导出合并历史统计
    ss << "MERGE_HISTORY_BEGIN\n";
    ss << "SUCCESSFUL_MERGES:" << stats.successful_merges << "\n";
    ss << "FAILED_MERGES:" << stats.failed_merges << "\n";
    ss << "MERGE_HISTORY_END\n";
    
    ss << "MERGE_NETWORK_EXPORT_END\n";
    
    LOG_DEBUG("合并网络导出完成: %zu个关系", merge_relationships.size());
    
    return ss.str();
}

// 导出多样性保护地图
std::string DataExporter::export_diversity_map(
    const std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
    const std::vector<std::shared_ptr<HMGSNode>>& main_branches,
    std::shared_ptr<HMGSNode> root_node,
    const std::vector<std::string>& neglected_branches,
    const std::unordered_map<std::string, size_t>& unmerged_counts,
    const std::unordered_map<std::string, std::chrono::steady_clock::time_point>& last_visit_times,
    const HMGSStatistics& stats) const {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::stringstream ss;
    
    // 文件头信息
    ss << "DIVERSITY_MAP_EXPORT_V2.0\n";
    ss << "FORMAT:STRUCTURED_DIVERSITY_MAP\n";
    ss << "TIMESTAMP:" << std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() << "\n";
    
    // 多样性地图元数据
    ss << "METADATA_BEGIN\n";
    ss << "TOTAL_NODES:" << node_registry.size() << "\n";
    ss << "NEGLECTED_BRANCHES:" << neglected_branches.size() << "\n";
    ss << "UNMERGED_COUNTS:" << unmerged_counts.size() << "\n";
    ss << "LAST_VISIT_TIMES:" << last_visit_times.size() << "\n";
    ss << "METADATA_END\n";
    
    // 导出被忽视分支列表及详细信息
    ss << "NEGLECTED_BRANCHES_BEGIN\n";
    for (const auto& branch_id : neglected_branches) {
        // 在主分支列表中查找对应的根节点
        auto it = std::find_if(main_branches.begin(), main_branches.end(),
            [&branch_id](const std::shared_ptr<HMGSNode>& node) {
                return node && node->get_id() == branch_id;
            });
        
        if (it != main_branches.end()) {
            const auto& branch_root = *it;
            size_t branch_index = std::distance(main_branches.begin(), it);
            
            ss << "BRANCH_ID:" << branch_id 
               << ":BRANCH_INDEX:" << branch_index
               << ":BRANCH_QUALITY:" << std::fixed << std::setprecision(6) 
               << branch_root->get_branch_quality()
               << ":AVERAGE_VALUE:" << std::fixed << std::setprecision(6) 
               << branch_root->get_average_value()
               << ":VISIT_COUNT:" << branch_root->get_visit_count() << "\n";
        } else {
            // 分支可能已被移除
            ss << "BRANCH_ID:" << branch_id << ":STATUS:REMOVED\n";
        }
    }
    ss << "NEGLECTED_BRANCHES_END\n";
    
    // 导出未合并计数信息
    ss << "UNMERGED_COUNTS_BEGIN\n";
    for (const auto& entry : unmerged_counts) {
        const std::string& node_id = entry.first;
        size_t count = entry.second;
        
        // 查找对应节点以获取额外信息
        auto node_it = node_registry.find(node_id);
        if (node_it != node_registry.end() && node_it->second) {
            const auto& node = node_it->second;
            
            ss << "NODE:" << node_id 
               << ":UNMERGED_COUNT:" << count
               << ":CONTINUOUS_UNMERGED:" << node->get_continuous_unmerged()
               << ":NOVELTY_SCORE:" << std::fixed << std::setprecision(6) 
               << node->get_novelty_score()
               << ":BACKTRACK_COUNT:" << node->get_backtrack_count() << "\n";
        } else {
            ss << "NODE:" << node_id << ":UNMERGED_COUNT:" << count << ":STATUS:MISSING\n";
        }
    }
    ss << "UNMERGED_COUNTS_END\n";
    
    // 导出最后访问时间信息
    ss << "LAST_VISIT_TIMES_BEGIN\n";
    for (const auto& entry : last_visit_times) {
        const std::string& node_id = entry.first;
        const auto& last_visit_time = entry.second;
        
        // 计算距离上次访问的时间
        auto now = std::chrono::steady_clock::now();
        auto time_since_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_visit_time).count();
        auto time_since_sec = time_since_ms / 1000.0;
        
        // 查找对应节点
        auto node_it = node_registry.find(node_id);
        if (node_it != node_registry.end() && node_it->second) {
            const auto& node = node_it->second;
            auto node_last_visit = node->get_last_visit_time();
            auto node_time_since_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - node_last_visit).count();
            
            ss << "NODE:" << node_id 
               << ":SYSTEM_TIME_SINCE_MS:" << time_since_ms
               << ":SYSTEM_TIME_SINCE_SEC:" << std::fixed << std::setprecision(3) << time_since_sec
               << ":NODE_TIME_SINCE_MS:" << node_time_since_ms
               << ":VISIT_COUNT:" << node->get_visit_count()
               << ":RECENT_VISITS:" << (time_since_ms < 1000 ? "true" : "false") << "\n";
        } else {
            ss << "NODE:" << node_id << ":TIME_SINCE_MS:" << time_since_ms << ":STATUS:MISSING\n";
        }
    }
    ss << "LAST_VISIT_TIMES_END\n";
    
    // 导出多样性统计指标
    ss << "DIVERSITY_STATISTICS_BEGIN\n";
    ss << "NEGLECTED_BRANCHES_TOTAL:" << stats.neglected_branches << "\n";
    ss << "BACKTRACK_EVENTS_TOTAL:" << stats.backtrack_events << "\n";
    ss << "AVERAGE_BRANCH_VALUE:" << std::fixed << std::setprecision(6) 
       << stats.average_branch_value << "\n";
    ss << "MAX_BRANCH_VALUE:" << std::fixed << std::setprecision(6) 
       << stats.max_branch_value << "\n";
    ss << "MIN_BRANCH_VALUE:" << std::fixed << std::setprecision(6) 
       << stats.min_branch_value << "\n";
    
    // 计算并导出分支价值分布统计
    if (!main_branches.empty()) {
        std::vector<double> branch_values;
        for (const auto& branch : main_branches) {
            if (branch) {
                branch_values.push_back(branch->get_average_value());
            }
        }
        
        if (!branch_values.empty()) {
            // 计算均值
            double mean_value = 0.0;
            for (double val : branch_values) {
                mean_value += val;
            }
            mean_value /= branch_values.size();
            
            // 计算方差
            double variance = 0.0;
            for (double val : branch_values) {
                variance += (val - mean_value) * (val - mean_value);
            }
            variance /= branch_values.size();
            
            ss << "BRANCH_VALUE_MEAN:" << std::fixed << std::setprecision(6) << mean_value << "\n";
            ss << "BRANCH_VALUE_VARIANCE:" << std::fixed << std::setprecision(6) << variance << "\n";
            ss << "BRANCH_VALUE_STDDEV:" << std::fixed << std::setprecision(6) 
               << std::sqrt(variance) << "\n";
        }
    }
    
    ss << "DIVERSITY_STATISTICS_END\n";
    
    // 导出各节点的多样性详细信息（限制输出数量以提高性能）
    ss << "NODE_DIVERSITY_INFO_BEGIN\n";
    size_t nodes_exported = 0;
    for (const auto& entry : node_registry) {
        const auto& node = entry.second;
        if (!node || nodes_exported >= 100) continue;  // 最多输出100个节点
        
        ss << "NODE:" << node->get_id()
           << ":NOVELTY_SCORE:" << std::fixed << std::setprecision(6) << node->get_novelty_score()
           << ":UNCERTAINTY:" << std::fixed << std::setprecision(6) << node->get_uncertainty()
           << ":EXPLORATION_BIAS:" << std::fixed << std::setprecision(6) << node->get_exploration_bias()
           << ":TOPOLOGY_WEIGHT:" << std::fixed << std::setprecision(6) << node->get_topology_weight()
           << ":NOVELTY_REWARD:" << std::fixed << std::setprecision(6) << node->get_novelty_reward()
           << ":SIMILAR_NODES_COUNT:" << node->get_similar_nodes().size() << "\n";
        
        nodes_exported++;
    }
    ss << "NODE_DIVERSITY_INFO_END\n";
    
    ss << "DIVERSITY_MAP_EXPORT_END\n";
    
    LOG_DEBUG("多样性地图导出完成: %zu个被忽视分支", neglected_branches.size());
    
    return ss.str();
}

} // namespace algorithms
} // namespace spacego
