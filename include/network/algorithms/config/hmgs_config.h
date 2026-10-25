#pragma once

#include <string>
#include <cstddef>
#include <chrono>
#include <limits>

namespace spacego {
namespace algorithms {

// 搜索统计信息结构
// 记录HMGS-DP搜索过程中的各项统计数据
// 从hmgs_dp.h中提取以避免循环依赖
struct HMGSStatistics {
    size_t total_simulations = 0;
    size_t completed_simulations = 0;
    size_t successful_merges = 0;
    size_t failed_merges = 0;
    size_t neglected_branches = 0;
    size_t backtrack_events = 0;
    size_t node_expansions = 0;
    double average_branch_value = 0.0;
    double max_branch_value = -std::numeric_limits<double>::infinity();
    double min_branch_value = std::numeric_limits<double>::infinity();
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point last_update_time;
};

struct HMGSConfig {
    // 合并相关参数
    double merge_threshold = 0.8;
    double merge_probability_weight = 1.0;
    double merged_into_probability_weight = 1.0;
    
    // UCT参数
    double uct_c_puct = 1.0;
    double uct_exploration_weight = 1.414;
    double topology_uct_weight = 1.0;
    
    // 多样性保护参数
    double diversity_gamma = 0.1;
    double novelty_reward_base = 0.2;
    double novelty_decay_factor = 0.95;
    double backtrack_boost_factor = 1.5;
    
    // 搜索参数
    size_t max_search_depth = 100;
    size_t max_simulations = 10000;
    size_t parallel_branches = 8;
    size_t replicas_per_branch = 3;
    size_t branch_update_frequency = 10;
    size_t merge_check_frequency = 5;
    
    // 回溯参数
    std::string backup_path_selection = "best_value";
    double value_improvement_threshold = 0.01;
    double merge_probability_decline_threshold = 0.1;
    double diversity_loss_threshold = 0.2;
};

} // namespace algorithms
} // namespace spacego
