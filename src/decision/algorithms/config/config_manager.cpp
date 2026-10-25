#include "network/algorithms/config/config_manager.h"

namespace spacego {
namespace algorithms {

ConfigManager::ConfigManager() : config_() {}

ConfigManager::ConfigManager(const HMGSConfig& config) : config_(config) {}

void ConfigManager::set_config(const HMGSConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
}

HMGSConfig ConfigManager::get_config() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_;
}

void ConfigManager::update_parameter(const std::string& param_name, double value) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (param_name == "merge_threshold") {
        config_.merge_threshold = value;
    } else if (param_name == "merge_probability_weight") {
        config_.merge_probability_weight = value;
    } else if (param_name == "merged_into_probability_weight") {
        config_.merged_into_probability_weight = value;
    } else if (param_name == "uct_c_puct") {
        config_.uct_c_puct = value;
    } else if (param_name == "uct_exploration_weight") {
        config_.uct_exploration_weight = value;
    } else if (param_name == "topology_uct_weight") {
        config_.topology_uct_weight = value;
    } else if (param_name == "diversity_gamma") {
        config_.diversity_gamma = value;
    } else if (param_name == "novelty_reward_base") {
        config_.novelty_reward_base = value;
    } else if (param_name == "novelty_decay_factor") {
        config_.novelty_decay_factor = value;
    } else if (param_name == "backtrack_boost_factor") {
        config_.backtrack_boost_factor = value;
    } else if (param_name == "value_improvement_threshold") {
        config_.value_improvement_threshold = value;
    } else if (param_name == "merge_probability_decline_threshold") {
        config_.merge_probability_decline_threshold = value;
    } else if (param_name == "diversity_loss_threshold") {
        config_.diversity_loss_threshold = value;
    }
}

void ConfigManager::update_parameter(const std::string& param_name, size_t value) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (param_name == "max_search_depth") {
        config_.max_search_depth = value;
    } else if (param_name == "max_simulations") {
        config_.max_simulations = value;
    } else if (param_name == "parallel_branches") {
        config_.parallel_branches = value;
    } else if (param_name == "replicas_per_branch") {
        config_.replicas_per_branch = value;
    } else if (param_name == "branch_update_frequency") {
        config_.branch_update_frequency = value;
    } else if (param_name == "merge_check_frequency") {
        config_.merge_check_frequency = value;
    }
}

void ConfigManager::update_parameter(const std::string& param_name, const std::string& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (param_name == "backup_path_selection") {
        config_.backup_path_selection = value;
    }
}

void ConfigManager::update_parameter(const std::string& param_name, bool value) {
    std::lock_guard<std::mutex> lock(mutex_);
}

} // namespace algorithms
} // namespace spacego
