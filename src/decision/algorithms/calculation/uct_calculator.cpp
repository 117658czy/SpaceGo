#include "network/algorithms/calculation/uct_calculator.h"

namespace spacego {
namespace algorithms {

UCTCalculator::UCTCalculator(const HMGSConfig& config) : config_(config) {}

double UCTCalculator::calculate_topology_uct(std::shared_ptr<HMGSNode> child, 
                                             std::shared_ptr<HMGSNode> parent, 
                                             size_t child_index) const {
    if (!child || !parent) {
        return 0.0;
    }
    
    size_t parent_visits = parent->get_visit_count();
    size_t child_visits = child->get_visit_count();
    
    if (child_visits == 0) {
        return std::numeric_limits<double>::infinity();
    }
    
    double child_value = child->get_average_value();
    
    // 基础UCT公式
    double exploration_term = config_.uct_c_puct * 
                              child->get_prior_prob(child_index) * 
                              std::sqrt(parent_visits) / (1.0 + child_visits);
    
    // 拓扑感知修正
    double topology_correction = 1.0;
    
    // 合并概率修正
    double merge_prob = child->get_merge_prob();
    double merged_into_prob = child->get_merged_into_prob();
    
    if (config_.merge_probability_weight > 0.0) {
        topology_correction *= (1.0 + config_.merge_probability_weight * merge_prob);
    }
    
    if (config_.merged_into_probability_weight > 0.0) {
        topology_correction *= (1.0 + config_.merged_into_probability_weight * merged_into_prob);
    }
    
    // 新颖性奖励
    double novelty_score = child->get_novelty_score();
    double novelty_reward = config_.novelty_reward_base * novelty_score;
    
    // 回溯增强
    size_t backtrack_count = child->get_backtrack_count();
    double backtrack_boost = (backtrack_count > 0) ? 
                             1.0 + config_.backtrack_boost_factor * std::log1p(backtrack_count) : 
                             1.0;
    
    // 不确定性引导
    double uncertainty = child->get_uncertainty();
    double uncertainty_exploration = 1.0 + config_.diversity_gamma * uncertainty;
    
    // 组合所有修正项
    double topology_uct = child_value + 
                          config_.topology_uct_weight * exploration_term * topology_correction +
                          novelty_reward * backtrack_boost * uncertainty_exploration;
    
    return topology_uct;
}

std::shared_ptr<HMGSNode> UCTCalculator::select_best_child(std::shared_ptr<HMGSNode> node) const {
    if (!node || node->get_children().empty()) {
        return nullptr;
    }
    
    const auto& children = node->get_children();
    std::shared_ptr<HMGSNode> best_child = nullptr;
    double best_score = -std::numeric_limits<double>::infinity();
    
    for (size_t i = 0; i < children.size(); ++i) {
        auto child = children[i];
        if (!child) continue;
        
        double score = calculate_topology_uct(child, node, i);
        if (score > best_score) {
            best_score = score;
            best_child = child;
        }
    }
    
    return best_child;
}

std::shared_ptr<HMGSNode> UCTCalculator::select_leaf_node(std::shared_ptr<HMGSNode> start_node) const {
    if (!start_node) {
        return nullptr;
    }
    
    auto current = start_node;
    size_t depth = 0;
    
    while (depth < config_.max_search_depth) {
        // 如果节点未扩展或者是叶子节点，返回
        if (!current->is_expanded() || current->is_leaf()) {
            return current;
        }
        
        // 选择最佳子节点继续向下搜索
        auto best_child = select_best_child(current);
        if (!best_child) {
            return current;
        }
        
        current = best_child;
        ++depth;
    }
    
    return current;
}

} // namespace algorithms
} // namespace spacego
