// 合并网络实现文件
// 包含所有合并操作的具体实现

#include "network/algorithms/data/merge_network.h"
#include "framework/logger.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

namespace spacego {
namespace algorithms {

// 构造函数
MergeNetwork::MergeNetwork()
    : initialized_(false) {
}

// 析构函数
MergeNetwork::~MergeNetwork() {
}

// 初始化
void MergeNetwork::initialize(const HMGSConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    config_ = config;
    initialized_.store(true, std::memory_order_release);
    
    LOG_INFO("合并网络初始化完成");
}

// 检查是否需要执行合并操作
bool MergeNetwork::check_for_merges(
    std::shared_ptr<HMGSNode> root_node,
    std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
    const HMGSConfig& config,
    HMGSStatistics& stats,
    std::unordered_map<std::string, double>& merge_probabilities,
    std::unordered_map<std::string, size_t>& unmerged_counts,
    std::unordered_map<std::string, std::chrono::steady_clock::time_point>& last_visit_times) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 前置条件检查：必须有根节点且节点数足够
    if (!root_node || node_registry.size() < 2) {
        return false;  // 节点不足，无法进行合并
    }
    
    bool merges_performed = false;
    size_t merge_checks = 0;
    const size_t max_merge_checks = 100;  // 限制每次检查的节点对数量，避免性能问题
    
    // 收集潜在的合并候选节点（排除根节点和未访问的节点）
    std::vector<std::shared_ptr<HMGSNode>> candidates;
    for (const auto& entry : node_registry) {
        auto node = entry.second;
        if (node && node != root_node && node->get_visit_count() > 0) {
            candidates.push_back(node);
        }
    }
    
    if (candidates.size() < 2) {
        return false;  // 候选节点不足
    }
    
    // 按访问次数降序排序，优先检查访问次数多的节点（更可能需要合并）
    std::sort(candidates.begin(), candidates.end(), 
              [](const std::shared_ptr<HMGSNode>& a, const std::shared_ptr<HMGSNode>& b) {
                  return a->get_visit_count() > b->get_visit_count();
              });
    
    // 遍历所有候选节点对，检查是否应该合并
    for (size_t i = 0; i < candidates.size() && merge_checks < max_merge_checks; ++i) {
        for (size_t j = i + 1; j < candidates.size() && merge_checks < max_merge_checks; ++j) {
            auto node1 = candidates[i];
            auto node2 = candidates[j];
            
            ++merge_checks;  // 增加检查计数
            
            // 检查节点是否已经被合并过
            if (node1->is_merged() || node2->is_merged()) {
                continue;  // 已合并的节点跳过
            }
            
            // 判断这两个节点是否应该合并
            if (should_merge_nodes(node1, node2, config, {})) {
                std::unordered_map<std::string, double> local_merged_into_probs;
                std::unordered_map<std::string, std::vector<std::string>> local_merge_rels;
                auto merged_node = merge_nodes(
                    node1, node2, config, node_registry, stats,
                    merge_probabilities, local_merged_into_probs, local_merge_rels, unmerged_counts, last_visit_times);
                
                if (merged_node) {
                    merges_performed = true;
                    stats.successful_merges++;
                    
                    LOG_DEBUG("成功合并节点: %s + %s -> %s", 
                             node1->get_id().c_str(), node2->get_id().c_str(), merged_node->get_id().c_str());
                } else {
                    stats.failed_merges++;
                    
                    LOG_WARNING("合并失败: %s + %s", node1->get_id().c_str(), node2->get_id().c_str());
                }
                
                // 限制每次检查的合并数量，避免过度合并
                if (stats.successful_merges % 10 == 0) {
                    break;  // 达到本次合并上限
                }
            }
        }
    }
    
    // 更新统计信息时间戳
    if (merges_performed) {
        stats.last_update_time = std::chrono::steady_clock::now();
    }
    
    return merges_performed;
}

// 判断两个节点是否应该合并
bool MergeNetwork::should_merge_nodes(
    std::shared_ptr<HMGSNode> node1,
    std::shared_ptr<HMGSNode> node2,
    const HMGSConfig& config,
    const std::unordered_map<std::string, std::vector<std::string>>& merge_relationships) const {
    
    // 基本有效性检查
    if (!node1 || !node2 || node1 == node2) {
        return false;  // 无效输入或相同节点
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 检查节点是否已经被合并
    if (node1->is_merged() || node2->is_merged()) {
        return false;  // 已合并的节点不能再参与合并
    }
    
    // 维度1：计算状态相似性（权重40%）
    double state_sim = calculate_state_similarity(node1->get_state(), node2->get_state());
    
    // 维度2：计算价值兼容性（权重30%）
    double value1 = node1->get_average_value();
    double value2 = node2->get_average_value();
    double value_compat = calculate_value_compatibility(value1, value2);
    
    // 维度3：计算拓扑一致性（权重30%）
    double topology_consist = calculate_topology_consistency(node1, node2, merge_relationships);
    
    // 计算综合合并分数（加权求和）
    double state_weight = 0.4;
    double value_weight = 0.3;
    double topology_weight = 0.3;
    
    double merge_score = state_weight * state_sim +
                        value_weight * value_compat +
                        topology_weight * topology_consist;
    
    // 获取节点的合并概率用于修正
    double merge_prob1 = node1->get_merge_prob();
    double merge_prob2 = node2->get_merge_prob();
    double merged_into_prob1 = node1->get_merged_into_prob();
    double merged_into_prob2 = node2->get_merged_into_prob();
    
    // 合并概率修正：高合并概率的节点更容易被选中合并
    double merge_prob_correction = (merge_prob1 + merge_prob2) / 2.0;
    double merged_into_correction = (merged_into_prob1 + merged_into_prob2) / 2.0;
    
    merge_score *= (1.0 + config.merge_probability_weight * merge_prob_correction);
    merge_score *= (1.0 + config.merged_into_probability_weight * merged_into_correction);
    
    // 应用合并阈值判断是否应该合并
    bool should_merge = merge_score >= config.merge_threshold;
    
    // 额外检查：如果任一节点的合并概率过低，不执行合并
    if (merge_prob1 < config.merge_probability_decline_threshold ||
        merge_prob2 < config.merge_probability_decline_threshold) {
        should_merge = false;
    }
    
    return should_merge;
}

// 执行两个节点的合并操作
std::shared_ptr<HMGSNode> MergeNetwork::merge_nodes(
    std::shared_ptr<HMGSNode> node1,
    std::shared_ptr<HMGSNode> node2,
    const HMGSConfig& config,
    std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
    HMGSStatistics& stats,
    std::unordered_map<std::string, double>& merge_probabilities,
    std::unordered_map<std::string, double>& merged_into_probabilities,
    std::unordered_map<std::string, std::vector<std::string>>& merge_relationships,
    std::unordered_map<std::string, size_t>& unmerged_counts,
    std::unordered_map<std::string, std::chrono::steady_clock::time_point>& last_visit_times) {
    
    // 基本有效性检查
    if (!node1 || !node2 || node1 == node2) {
        return nullptr;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 检查节点是否已经被合并
    if (node1->is_merged() || node2->is_merged()) {
        return nullptr;
    }
    
    // 选择主节点（访问次数更多的作为主节点，保留其身份）
    std::shared_ptr<HMGSNode> main_node, secondary_node;
    if (node1->get_visit_count() >= node2->get_visit_count()) {
        main_node = node1;
        secondary_node = node2;
    } else {
        main_node = node2;
        secondary_node = node1;
    }
    
    // 记录合并前的统计信息用于日志
    size_t main_visits_before = main_node->get_visit_count();
    double main_value_before = main_node->get_average_value();
    size_t secondary_visits_before = secondary_node->get_visit_count();
    double secondary_value_before = secondary_node->get_average_value();
    
    // 执行合并操作
    
    // 1. 合并访问计数和价值（使用加权平均）
    size_t total_visits = main_visits_before + secondary_visits_before;
    double weighted_value = (main_value_before * main_visits_before + 
                           secondary_value_before * secondary_visits_before) / total_visits;
    
    // 重置主节点统计信息并设置合并后的值
    main_node->reset_visit_count();
    main_node->reset_total_value();
    main_node->add_value(weighted_value);
    
    // 用加权价值填充总访问次数（保留历史记录）
    for (size_t i = 0; i < total_visits - 1; ++i) {
        main_node->add_value(weighted_value);
    }
    
    // 2. 合并子节点（将次节点的子节点转移到主节点，去重处理）
    const auto& secondary_children = secondary_node->get_children();
    const auto& main_existing_children = main_node->get_children();
    
    for (const auto& child : secondary_children) {
        if (child) {
            // 检查主节点是否已有相同状态的子节点，避免重复添加
            bool is_duplicate = false;
            for (const auto& existing_child : main_existing_children) {
                if (existing_child && existing_child->get_state() == child->get_state()) {
                    is_duplicate = true;
                    break;
                }
            }
            
            if (!is_duplicate) {
                // 重新设置父节点为主节点
                child->set_parent(main_node);
                main_node->add_child(child);
                
                // 更新子节点的深度信息
                child->set_depth(main_node->get_depth() + 1);
            }
        }
    }
    
    // 清空次节点的子节点列表（已转移或去重，断开引用防止悬垂指针）
    secondary_node->clear_children();
    
    // 3. 合并先验概率（使用加权平均）
    const auto& secondary_prior_probs = secondary_node->get_prior_probs();
    const auto& main_prior_probs = main_node->get_prior_probs();
    
    if (!secondary_prior_probs.empty() && !main_prior_probs.empty() &&
        secondary_prior_probs.size() == main_prior_probs.size()) {
        
        std::vector<double> merged_prior_probs(main_prior_probs.size());
        for (size_t i = 0; i < main_prior_probs.size(); ++i) {
            merged_prior_probs[i] = (main_prior_probs[i] * main_visits_before + 
                                    secondary_prior_probs[i] * secondary_visits_before) / total_visits;
        }
        main_node->set_prior_probs(merged_prior_probs);
    }
    
    // 4. 合并不确定性估计（使用加权平均）
    double main_uncertainty = main_node->get_uncertainty();
    double secondary_uncertainty = secondary_node->get_uncertainty();
    double merged_uncertainty = (main_uncertainty * main_visits_before + 
                                secondary_uncertainty * secondary_visits_before) / total_visits;
    main_node->set_uncertainty(merged_uncertainty);
    
    // 5. 合并新颖性分数（使用加权平均）
    double main_novelty = main_node->get_novelty_score();
    double secondary_novelty = secondary_node->get_novelty_score();
    double merged_novelty = (main_novelty * main_visits_before + 
                            secondary_novelty * secondary_visits_before) / total_visits;
    main_node->set_novelty_score(merged_novelty);
    
    // 6. 标记次节点为已合并状态
    secondary_node->mark_as_merged();
    secondary_node->set_merged_into_node(main_node);
    
    // 6.1 从父节点的children列表中移除次节点（防止僵尸引用和内存泄漏）
    // 合并后次节点已从注册表删除，但父节点仍持有其shared_ptr引用
    auto secondary_parent = secondary_node->get_parent();
    if (secondary_parent) {
        secondary_parent->remove_child(secondary_node);
    }
    
    // 7. 更新合并关系记录（双向记录）
    std::string main_id = main_node->get_id();
    std::string secondary_id = secondary_node->get_id();
    
    merge_relationships[main_id].push_back(secondary_id);
    merge_relationships[secondary_id].push_back(main_id);
    
    // 8. 从全局节点注册表中移除次节点
    node_registry.erase(secondary_id);
    
    // 9. 更新未合并计数
    unmerged_counts[main_id] = unmerged_counts[main_id] + unmerged_counts[secondary_id] + 1;
    unmerged_counts.erase(secondary_id);
    
    // 10. 更新最后访问时间
    auto now = std::chrono::steady_clock::now();
    main_node->update_last_visit_time();
    last_visit_times[main_id] = now;
    last_visit_times.erase(secondary_id);
    
    // 11. 更新主节点的合并概率
    update_merge_probabilities(main_node, config, node_registry, 
                              merge_probabilities, merged_into_probabilities, merge_relationships);
    
    // 更新统计信息
    stats.total_simulations += secondary_visits_before;
    stats.completed_simulations += secondary_visits_before;
    
    LOG_INFO("完成节点合并: 主节点=%s, 次节点=%s, 总访问=%zu", 
             main_id.c_str(), secondary_id.c_str(), total_visits);
    
    return main_node;
}

// 更新单个节点的合并概率
void MergeNetwork::update_merge_probabilities(
    std::shared_ptr<HMGSNode> node,
    const HMGSConfig& config,
    const std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
    std::unordered_map<std::string, double>& merge_probabilities,
    std::unordered_map<std::string, double>& merged_into_probabilities,
    const std::unordered_map<std::string, std::vector<std::string>>& merge_relationships) {
    
    if (!node) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::string node_id = node->get_id();
    
    // 计算新的合并概率（基于多个维度）
    double new_merge_prob = 0.0;
    
    // 维度1：基于节点访问频率（权重30%）
    size_t total_visits_in_tree = 0;
    for (const auto& entry : node_registry) {
        total_visits_in_tree += entry.second->get_visit_count();
    }
    
    if (total_visits_in_tree > 0) {
        double visit_frequency = static_cast<double>(node->get_visit_count()) / total_visits_in_tree;
        new_merge_prob += 0.3 * visit_frequency;
    }
    
    // 维度2：基于节点价值（权重30%）
    double node_value = node->get_average_value();
    double normalized_value = (node_value + 1.0) / 2.0;  // 归一化到 [0, 1]
    new_merge_prob += 0.3 * normalized_value;
    
    // 维度3：基于节点不确定性（权重20%）
    double uncertainty = node->get_uncertainty();
    new_merge_prob += 0.2 * uncertainty;
    
    // 维度4：基于新颖性分数（权重10%）
    double novelty = node->get_novelty_score();
    new_merge_prob += 0.1 * novelty;
    
    // 维度5：基于拓扑一致性（权重10%）
    double topology_score = 0.0;
    size_t neighbor_count = 0;
    
    const auto& similar_nodes = node->get_similar_nodes();
    for (const auto& similar_node : similar_nodes) {
        if (similar_node) {
            topology_score += calculate_topology_consistency(node, similar_node, merge_relationships);
            ++neighbor_count;
        }
    }
    
    if (neighbor_count > 0) {
        topology_score /= neighbor_count;  // 平均拓扑一致性
        new_merge_prob += 0.1 * topology_score;
    }
    
    // 应用指数移动平均平滑处理（避免剧烈波动）
    auto it = merge_probabilities.find(node_id);
    if (it != merge_probabilities.end()) {
        double old_prob = it->second;
        double alpha = 0.7;  // 衰减因子，给予新值更高权重
        new_merge_prob = alpha * new_merge_prob + (1.0 - alpha) * old_prob;
    }
    
    // 限制概率范围到[0, 1]
    new_merge_prob = std::max(0.0, std::min(1.0, new_merge_prob));
    
    // 更新合并概率存储
    merge_probabilities[node_id] = new_merge_prob;
    
    // 同步更新节点对象的合并概率属性
    node->set_merge_prob(new_merge_prob);
    
    // 计算被其他节点合并的概率
    double merged_into_prob = 0.0;
    auto rel_it = merge_relationships.find(node_id);
    if (rel_it != merge_relationships.end() && !rel_it->second.empty()) {
        merged_into_prob = 0.5;  // 基础概率
        
        // 基于已存在的合并关系数量调整概率
        size_t merge_count = rel_it->second.size();
        merged_into_prob += 0.1 * std::min(static_cast<double>(merge_count), 5.0);
    }
    
    // 存储被合并概率
    merged_into_probabilities[node_id] = merged_into_prob;
    node->set_merged_into_prob(merged_into_prob);
}

// 强制执行一次合并检查
void MergeNetwork::force_merge_check(
    std::shared_ptr<HMGSNode> root_node,
    std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
    const HMGSConfig& config,
    HMGSStatistics& stats,
    std::unordered_map<std::string, double>& merge_probabilities,
    std::unordered_map<std::string, size_t>& unmerged_counts,
    std::unordered_map<std::string, std::chrono::steady_clock::time_point>& last_visit_times) {
    
    // 直接调用内部合并检查方法
    check_for_merges(root_node, node_registry, config, stats, 
                    merge_probabilities, unmerged_counts, last_visit_times);
    
    LOG_INFO("强制合并检查完成");
}

// 重新计算所有节点的合并概率
void MergeNetwork::recalculate_merge_probabilities(
    const std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
    const HMGSConfig& config,
    HMGSStatistics& stats,
    std::unordered_map<std::string, double>& merge_probabilities,
    std::unordered_map<std::string, double>& merged_into_probabilities,
    const std::unordered_map<std::string, std::vector<std::string>>& merge_relationships) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 遍历所有节点，重新计算每个节点的合并概率
    for (const auto& entry : node_registry) {
        auto node = entry.second;
        if (node) {
            // 调用单节点概率更新方法
            update_merge_probabilities(node, config, node_registry,
                                      merge_probabilities, merged_into_probabilities,
                                      merge_relationships);
        }
    }
    
    // 更新统计信息时间戳
    stats.last_update_time = std::chrono::steady_clock::now();
    
    LOG_DEBUG("重新计算%zu个节点的合并概率", node_registry.size());
}

// 重建整个合并网络
void MergeNetwork::rebuild_merge_network(
    const std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
    const HMGSConfig& config,
    HMGSStatistics& stats,
    std::unordered_map<std::string, double>& merge_probabilities,
    std::unordered_map<std::string, double>& merged_into_probabilities,
    std::unordered_map<std::string, std::vector<std::string>>& merge_relationships) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 清空现有的合并网络数据结构
    merge_probabilities.clear();
    merged_into_probabilities.clear();
    merge_relationships.clear();
    
    // 基于当前节点状态重新构建完整的合并网络
    for (const auto& entry : node_registry) {
        auto node = entry.second;
        if (node) {
            // 重新计算该节点的合并概率
            update_merge_probabilities(node, config, node_registry,
                                      merge_probabilities, merged_into_probabilities,
                                      merge_relationships);
            
            // 重建合并关系（基于相似节点列表）
            std::string node_id = node->get_id();
            const auto& similar_nodes = node->get_similar_nodes();
            
            for (const auto& similar_node : similar_nodes) {
                if (similar_node) {
                    std::string similar_id = similar_node->get_id();
                    merge_relationships[node_id].push_back(similar_id);
                }
            }
        }
    }
    
    // 更新统计信息时间戳
    stats.last_update_time = std::chrono::steady_clock::now();
    
    LOG_INFO("重建合并网络完成: %zu个节点", node_registry.size());
}

// 计算状态相似性（私有辅助方法）
double MergeNetwork::calculate_state_similarity(
    const std::string& state1,
    const std::string& state2) const {
    
    // 完全相同的状态返回最高相似性
    if (state1 == state2) {
        return 1.0;
    }
    
    // 检查是否为标准19x19棋盘编码（361字符）
    const size_t BOARD_SIZE = 361;
    
    if (state1.size() != BOARD_SIZE || state2.size() != BOARD_SIZE) {
        // 非标准编码，使用简化的字符匹配作为后备方案
        size_t min_len = std::min(state1.size(), state2.size());
        size_t max_len = std::max(state1.size(), state2.size());
        
        if (max_len == 0) {
            return 0.0;
        }
        
        // 简单的字符匹配相似性
        size_t matches = 0;
        for (size_t i = 0; i < min_len; ++i) {
            if (state1[i] == state2[i]) {
                ++matches;
            }
        }
        
        double base_similarity = static_cast<double>(matches) / max_len;
        double length_ratio = static_cast<double>(min_len) / max_len;
        
        return base_similarity * length_ratio;
    }
    
    // 标准棋盘编码：计算位置匹配率
    size_t same_position_count = 0;
    size_t valid_positions = 0;
    
    for (size_t i = 0; i < BOARD_SIZE; ++i) {
        char c1 = state1[i];
        char c2 = state2[i];
        
        // 跳过无效位置
        if (c1 == 'X' || c2 == 'X') {
            continue;
        }
        
        valid_positions++;
        if (c1 == c2) {
            same_position_count++;
        }
    }
    
    if (valid_positions == 0) {
        return 0.0;
    }
    
    return static_cast<double>(same_position_count) / valid_positions;
}

// 计算价值兼容性（私有辅助方法）
double MergeNetwork::calculate_value_compatibility(double value1, double value2) const {
    // 使用高斯核函数计算价值兼容性
    // 价值差异越小，兼容性越高
    double diff = std::abs(value1 - value2);
    return std::exp(-diff * diff);  // 高斯核
}

// 计算拓扑一致性（私有辅助方法）
double MergeNetwork::calculate_topology_consistency(
    std::shared_ptr<HMGSNode> node1,
    std::shared_ptr<HMGSNode> node2,
    const std::unordered_map<std::string, std::vector<std::string>>& merge_relationships) const {
    
    if (!node1 || !node2) {
        return 0.0;
    }
    
    // 检查双向相似节点关系
    const auto& similar_nodes1 = node1->get_similar_nodes();
    const auto& similar_nodes2 = node2->get_similar_nodes();
    
    bool found1 = std::find(similar_nodes1.begin(), similar_nodes1.end(), node2) != similar_nodes1.end();
    bool found2 = std::find(similar_nodes2.begin(), similar_nodes2.end(), node1) != similar_nodes2.end();
    
    if (found1 && found2) {
        return 1.0;  // 双向确认，完全一致
    } else if (found1 || found2) {
        return 0.5;  // 单向确认，部分一致
    }
    
    // 检查合并关系记录
    auto node1_id = node1->get_id();
    auto node2_id = node2->get_id();
    
    auto it1 = merge_relationships.find(node1_id);
    if (it1 != merge_relationships.end()) {
        const auto& relationships = it1->second;
        if (std::find(relationships.begin(), relationships.end(), node2_id) != relationships.end()) {
            return 0.8;  // 存在合并关系记录
        }
    }
    
    auto it2 = merge_relationships.find(node2_id);
    if (it2 != merge_relationships.end()) {
        const auto& relationships = it2->second;
        if (std::find(relationships.begin(), relationships.end(), node1_id) != relationships.end()) {
            return 0.8;  // 存在反向合并关系记录
        }
    }
    
    return 0.0;  // 无任何一致性证据
}

} // namespace algorithms
} // namespace spacego
