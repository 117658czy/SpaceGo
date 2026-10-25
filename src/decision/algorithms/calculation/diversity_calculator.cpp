// 多样性计算器实现文件
// 包含所有多样性相关计算的具体实现

#include "network/algorithms/calculation/diversity_calculator.h"
#include "framework/logger.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_set>

namespace spacego {
namespace algorithms {

// 构造函数
DiversityCalculator::DiversityCalculator()
    : initialized_(false) {
}

// 析构函数
DiversityCalculator::~DiversityCalculator() {
}

// 初始化
void DiversityCalculator::initialize(const HMGSConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    config_ = config;
    initialized_.store(true, std::memory_order_release);
    
    LOG_INFO("多样性计算器初始化完成");
}

// 计算节点的新颖性价值
double DiversityCalculator::calculate_novelty_value(
    std::shared_ptr<HMGSNode> node,
    const std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
    const HMGSStatistics& stats) const {
    
    if (!node) {
        return 0.0;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    double novelty_score = 0.0;
    
    // 1. 状态新颖性：检查状态在搜索树中的唯一性
    std::string node_state = node->get_state();
    size_t same_state_count = 0;
    
    for (const auto& entry : node_registry) {
        if (entry.second && entry.second->get_state() == node_state) {
            ++same_state_count;
        }
    }
    
    // 状态越稀有，新颖性越高
    double state_novelty = (same_state_count == 0) ? 1.0 : 1.0 / same_state_count;
    novelty_score += 0.4 * state_novelty;
    
    // 2. 价值新颖性：价值与平均价值的差异
    double node_value = node->get_average_value();
    double avg_tree_value = stats.average_branch_value;
    
    if (stats.completed_simulations > 0) {
        double value_diff = std::abs(node_value - avg_tree_value);
        double max_value_range = std::max(1.0, stats.max_branch_value - stats.min_branch_value);
        double value_novelty = value_diff / max_value_range;
        novelty_score += 0.3 * value_novelty;
    }
    
    // 3. 结构新颖性：节点在树中的结构位置
    size_t node_depth = node->get_depth();
    // 深度越大，结构越深入，新颖性贡献增加（使用指数衰减）
    double depth_factor = 1.0 - std::exp(-static_cast<double>(node_depth) / 10.0);
    novelty_score += 0.2 * depth_factor;
    
    // 4. 时间新颖性：最近访问的新颖性
    auto now = std::chrono::steady_clock::now();
    auto last_visit = node->get_last_visit_time();
    auto time_since_visit = std::chrono::duration_cast<std::chrono::seconds>(now - last_visit).count();
    // 越久未被访问，时间新颖性越高
    double time_novelty = 1.0 - std::exp(-time_since_visit / 60.0);  // 60秒衰减常数
    novelty_score += 0.1 * time_novelty;
    
    return novelty_score;
}

// 保护多样性分支
void DiversityCalculator::protect_diversity_branch(
    std::shared_ptr<HMGSNode> node,
    const HMGSConfig& config,
    HMGSStatistics& stats,
    std::unordered_map<std::string, std::chrono::steady_clock::time_point>& last_visit_times) {
    
    if (!node) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::string node_id = node->get_id();
    
    // 标记节点为受保护状态，防止被过早合并或丢弃
    node->set_protected(true);
    
    // 增加新颖性分数以鼓励探索该分支
    double current_novelty = node->get_novelty_score();
    node->set_novelty_score(current_novelty + config.novelty_reward_base);
    
    // 调整不确定性以增加探索倾向
    double current_uncertainty = node->get_uncertainty();
    node->set_uncertainty(std::min(1.0, current_uncertainty + 0.2));
    
    // 记录保护操作到统计信息
    stats.neglected_branches++;
    
    // 更新最后访问时间
    node->update_last_visit_time();
    last_visit_times[node_id] = node->get_last_visit_time();
    
    LOG_DEBUG("保护多样性分支: %s", node_id.c_str());
}

// 根据一致性水平调整权重
void DiversityCalculator::adjust_weights_for_diversity(
    std::shared_ptr<HMGSNode> branch,
    double consistency_level,
    const HMGSConfig& config,
    HMGSStatistics& stats) {
    
    if (!branch) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 根据一致性水平计算多样性因子
    // 低一致性 -> 高多样性需求 -> 增加探索权重
    // 高一致性 -> 低多样性需求 -> 减少探索权重
    double diversity_factor = 1.0 - consistency_level;
    
    // 调整UCT探索参数（Cpuct值）
    double adjusted_c_puct = config.uct_c_puct * (1.0 + config.diversity_gamma * diversity_factor);
    
    // 调整拓扑UCT权重
    double adjusted_topology_weight = config.topology_uct_weight * (1.0 + 0.5 * diversity_factor);
    
    // 调整新颖性奖励
    double adjusted_novelty_reward = config.novelty_reward_base * (1.0 + diversity_factor);
    
    // 应用调整后的参数到分支节点
    branch->set_exploration_bias(adjusted_c_puct);
    branch->set_topology_weight(adjusted_topology_weight);
    branch->set_novelty_reward(adjusted_novelty_reward);
    
    // 更新统计信息时间戳
    stats.last_update_time = std::chrono::steady_clock::now();
    
    LOG_DEBUG("调整分支权重: diversity_factor=%.3f", diversity_factor);
}

// 计算当前整体的多样性分数
double DiversityCalculator::calculate_current_diversity_score(
    const std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
    const std::vector<std::shared_ptr<HMGSNode>>& main_branches,
    std::shared_ptr<HMGSNode> root_node) const {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (node_registry.empty() || !root_node) {
        return 0.0;
    }
    
    // 多样性分数由多个维度组成
    
    // 维度1：状态多样性 - 不同状态的节点比例
    std::unordered_set<std::string> unique_states;
    for (const auto& entry : node_registry) {
        if (entry.second && entry.second != root_node) {
            unique_states.insert(entry.second->get_state());
        }
    }
    double state_diversity = static_cast<double>(unique_states.size()) / 
                            std::max<size_t>(1, node_registry.size() - 1);
    
    // 维度2：价值分布多样性 - 分支价值的方差
    double value_diversity = 0.0;
    if (!main_branches.empty()) {
        std::vector<double> branch_values;
        for (const auto& branch : main_branches) {
            if (branch && branch->get_visit_count() > 0) {
                branch_values.push_back(branch->get_average_value());
            }
        }
        
        if (branch_values.size() > 1) {
            double mean = std::accumulate(branch_values.begin(), branch_values.end(), 0.0) / 
                         branch_values.size();
            double variance = 0.0;
            for (double val : branch_values) {
                variance += (val - mean) * (val - mean);
            }
            variance /= branch_values.size();
            
            // 归一化方差到[0, 1]范围（假设最大方差为4.0）
            value_diversity = std::min(1.0, variance / 4.0);
        } else if (branch_values.size() == 1) {
            value_diversity = 0.0;  // 只有一个分支，无多样性
        }
    }
    
    // 维度3：访问分布多样性 - 访问次数的均匀程度
    double visit_diversity = 0.0;
    if (!main_branches.empty()) {
        std::vector<size_t> visit_counts;
        for (const auto& branch : main_branches) {
            if (branch) {
                visit_counts.push_back(branch->get_visit_count());
            }
        }
        
        if (!visit_counts.empty()) {
            size_t total_visits = 0;
            for (size_t count : visit_counts) {
                total_visits += count;
            }
            
            if (total_visits > 0) {
                // 计算基尼系数的补数来衡量均匀度
                // 完全均匀时为1.0，完全不均匀时为0.0
                double gini_sum = 0.0;
                for (size_t i = 0; i < visit_counts.size(); ++i) {
                    for (size_t j = 0; j < visit_counts.size(); ++j) {
                        gini_sum += std::abs(
                            static_cast<double>(visit_counts[i]) - static_cast<double>(visit_counts[j]));
                    }
                }
                double max_gini = 2.0 * static_cast<double>(visit_counts.size()) * 
                                 static_cast<double>(total_visits) / visit_counts.size();
                visit_diversity = 1.0 - (max_gini > 0 ? gini_sum / max_gini : 0.0);
            }
        }
    }
    
    // 维度4：深度分布多样性 - 节点深度的变化程度
    double depth_diversity = 0.0;
    {
        std::vector<size_t> depths;
        for (const auto& entry : node_registry) {
            if (entry.second && entry.second != root_node) {
                depths.push_back(entry.second->get_depth());
            }
        }
        
        if (depths.size() > 1) {
            double mean_depth = std::accumulate(depths.begin(), depths.end(), 0.0) / depths.size();
            double depth_variance = 0.0;
            for (size_t depth : depths) {
                depth_variance += (static_cast<double>(depth) - mean_depth) * 
                                 (static_cast<double>(depth) - mean_depth);
            }
            depth_variance /= depths.size();
            
            // 归一化到[0, 1]
            depth_diversity = std::min(1.0, depth_variance / 100.0);  // 假设最大深度方差为100
        }
    }
    
    // 综合加权计算最终多样性分数
    double total_diversity = 
        0.35 * state_diversity +      // 状态多样性权重最高
        0.30 * value_diversity +      // 价值分布多样性
        0.20 * visit_diversity +      // 访问分布多样性
        0.15 * depth_diversity;       // 深度分布多样性
    
    // 应用非线性变换以增强区分度
    total_diversity = std::pow(total_diversity, 0.8);
    
    return std::max(0.0, std::min(1.0, total_diversity));
}

// 识别被忽视的分支
std::vector<std::string> DiversityCalculator::identify_neglected_branches(
    const std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
    std::shared_ptr<HMGSNode> root_node,
    const HMGSConfig& config,
    const HMGSStatistics& stats) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> neglected_branches;
    
    if (!root_node || node_registry.empty()) {
        return neglected_branches;
    }
    
    // 计算平均访问次数（排除根节点）
    size_t total_visits = 0;
    size_t node_count = 0;
    
    for (const auto& entry : node_registry) {
        auto node = entry.second;
        if (node && node != root_node) {
            total_visits += node->get_visit_count();
            ++node_count;
        }
    }
    
    if (node_count == 0) {
        return neglected_branches;
    }
    
    double avg_visits = static_cast<double>(total_visits) / node_count;
    
    // 遍历所有节点判断是否被忽视
    for (const auto& entry : node_registry) {
        auto node = entry.second;
        if (!node || node == root_node) {
            continue;
        }
        
        std::string node_id = node->get_id();
        
        // 条件1：访问次数远低于平均（低于30%且绝对次数小于10）
        size_t visits = node->get_visit_count();
        double visit_ratio = (avg_visits > 0) ? static_cast<double>(visits) / avg_visits : 0.0;
        bool condition1 = (visit_ratio < 0.3 && visits < 10);
        
        // 条件2：长时间未被访问（超过30秒）
        auto now = std::chrono::steady_clock::now();
        auto last_visit = node->get_last_visit_time();
        auto time_since_visit = std::chrono::duration_cast<std::chrono::seconds>(now - last_visit).count();
        bool condition2 = (time_since_visit > 30);
        
        // 条件3：合并概率很低且价值改进不明显
        double merge_prob = node->get_merge_prob();
        double node_value = node->get_average_value();
        double value_improvement = node_value - stats.average_branch_value;
        bool condition3 = (merge_prob < config.merge_probability_decline_threshold && 
                          value_improvement < config.value_improvement_threshold);
        
        // 条件4：多样性损失超过阈值（新颖性过低）
        double novelty_value = calculate_novelty_value(node, node_registry, stats);
        bool condition4 = (novelty_value < config.diversity_loss_threshold);
        
        // 综合判断：满足任一条件即为被忽视分支
        bool is_neglected = condition1 || condition2 || condition3 || condition4;
        
        if (is_neglected) {
            neglected_branches.push_back(node_id);
            LOG_DEBUG("识别到被忽视分支: %s (条件: %d%d%d%d)", 
                     node_id.c_str(), (int)condition1, (int)condition2, (int)condition3, (int)condition4);
        }
    }
    
    return neglected_branches;
}

// 为被忽视的分支分配资源
void DiversityCalculator::allocate_resources_to_neglected_branches(
    const std::vector<std::string>& neglected_branches,
    double total_resources,
    std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
    HMGSStatistics& stats,
    std::unordered_map<std::string, std::chrono::steady_clock::time_point>& last_visit_times) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (neglected_branches.empty() || total_resources <= 0.0) {
        return;
    }
    
    // 均匀分配资源到每个被忽视分支
    size_t branch_count = neglected_branches.size();
    double resources_per_branch = total_resources / branch_count;
    
    LOG_INFO("为%zu个被忽视分支分配资源，每分支: %.1f次模拟", 
             branch_count, resources_per_branch);
    
    // 为每个被忽视分支分配资源并更新属性
    for (const auto& branch_id : neglected_branches) {
        auto it = node_registry.find(branch_id);
        if (it == node_registry.end()) {
            continue;
        }
        
        auto branch = it->second;
        if (!branch) {
            continue;
        }
        
        // 计算应分配的模拟次数
        size_t simulations_to_allocate = static_cast<size_t>(resources_per_branch);
        
        // 增加分支的优先权以提升选择概率
        double current_priority = branch->get_priority();
        branch->set_priority(current_priority + simulations_to_allocate);
        
        // 记录资源分配到统计信息
        stats.total_simulations += simulations_to_allocate;
        
        // 更新最后访问时间
        branch->update_last_visit_time();
        last_visit_times[branch_id] = branch->get_last_visit_time();
    }
    
    LOG_DEBUG("资源分配完成");
}

// 调整多样性参数
void DiversityCalculator::adjust_diversity_parameters(
    HMGSConfig& config,
    const std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
    std::shared_ptr<HMGSNode> root_node,
    HMGSStatistics& stats) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 采样计算当前多样性水平
    double current_diversity = 0.0;
    size_t sample_count = 0;
    
    // 采样最多50个节点计算平均新颖性
    for (const auto& entry : node_registry) {
        auto node = entry.second;
        if (node && node != root_node) {
            double novelty = calculate_novelty_value(node, node_registry, stats);
            current_diversity += novelty;
            ++sample_count;
            
            if (sample_count >= 50) {  // 限制采样数量以提高性能
                break;
            }
        }
    }
    
    if (sample_count > 0) {
        current_diversity /= sample_count;
    }
    
    // 根据当前多样性水平动态调整参数
    if (current_diversity < 0.3) {
        // 多样性过低：需要增加探索激励
        config.diversity_gamma = std::min(0.5, config.diversity_gamma * 1.2);           // 增加探索权重
        config.novelty_reward_base = std::min(0.5, config.novelty_reward_base * 1.3);   // 增加新颖性奖励
        config.backtrack_boost_factor = std::min(3.0, config.backtrack_boost_factor * 1.1);  // 增加回溯增强
        
        LOG_DEBUG("多样性偏低(%.3f)，增加探索参数", current_diversity);
    } else if (current_diversity > 0.7) {
        // 多样性过高：需要减少探索，加强利用
        config.diversity_gamma = std::max(0.05, config.diversity_gamma * 0.8);          // 减少探索权重
        config.novelty_reward_base = std::max(0.05, config.novelty_reward_base * 0.7);   // 减少新颖性奖励
        config.backtrack_boost_factor = std::max(0.5, config.backtrack_boost_factor * 0.9);  // 减少回溯增强
        
        LOG_DEBUG("多样性偏高(%.3f)，减少探索参数", current_diversity);
    }
    
    // 更新统计信息时间戳
    stats.last_update_time = std::chrono::steady_clock::now();
}

// 选择回溯节点
std::shared_ptr<HMGSNode> DiversityCalculator::select_backtrack_node(
    const std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
    std::shared_ptr<HMGSNode> root_node,
    const std::vector<std::string>& neglected_branches) const {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!root_node || node_registry.size() <= 1) {
        return nullptr;
    }
    
    // 收集潜在的回溯候选节点
    std::vector<std::shared_ptr<HMGSNode>> candidates;
    
    for (const auto& entry : node_registry) {
        auto node = entry.second;
        if (!node || node == root_node) {
            continue;
        }
        
        // 判断节点是否适合作为回溯候选
        bool is_candidate = false;
        
        // 条件1：在被忽视分支列表中
        auto it = std::find(neglected_branches.begin(), neglected_branches.end(), node->get_id());
        if (it != neglected_branches.end()) {
            is_candidate = true;
        }
        
        // 条件2：低访问次数（少于5次）
        if (node->get_visit_count() < 5) {
            is_candidate = true;
        }
        
        // 条件3：低价值但具有高潜力（不确定性高）
        double node_value = node->get_average_value();
        double value_threshold = 0.0;  // 可以从外部传入统计平均值
        if (node_value < value_threshold && node->get_uncertainty() > 0.5) {
            is_candidate = true;
        }
        
        // 条件4：高新颖性但探索不足
        double novelty = node->get_novelty_score();
        if (novelty > 0.7 && node->get_visit_count() < 10) {
            is_candidate = true;
        }
        
        if (is_candidate) {
            candidates.push_back(node);
        }
    }
    
    if (candidates.empty()) {
        return nullptr;
    }
    
    // 按回溯优先级排序（综合考虑不确定性、新颖性和访问次数）
    std::sort(candidates.begin(), candidates.end(), 
              [](const std::shared_ptr<HMGSNode>& a, const std::shared_ptr<HMGSNode>& b) {
                  // 优先级公式：不确定性权重40% + 新颖性权重30% + 访问倒数权重30%
                  double score_a = a->get_uncertainty() * 0.4 +
                                 a->get_novelty_score() * 0.3 +
                                 (1.0 / (1.0 + a->get_visit_count())) * 0.3;
                  double score_b = b->get_uncertainty() * 0.4 +
                                 b->get_novelty_score() * 0.3 +
                                 (1.0 / (1.0 + b->get_visit_count())) * 0.3;
                  return score_a > score_b;
              });
    
    // 返回优先级最高的候选节点
    return candidates.front();
}

// 执行回溯操作
void DiversityCalculator::perform_backtrack(
    std::shared_ptr<HMGSNode> backtrack_node,
    HMGSStatistics& stats,
    std::vector<std::string>& neglected_branches,
    std::unordered_map<std::string, std::chrono::steady_clock::time_point>& last_visit_times) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!backtrack_node) {
        return;
    }
    
    std::string node_id = backtrack_node->get_id();
    
    // 记录原始访问次数用于日志
    size_t original_visits = backtrack_node->get_visit_count();
    
    // 1. 重置节点的部分统计信息以鼓励重新探索
    if (original_visits > 10) {
        // 如果访问次数较多，保留一半记录但降低权重
        size_t new_visits = original_visits / 2;
        double original_value = backtrack_node->get_average_value();
        
        backtrack_node->reset_visit_count();
        backtrack_node->reset_total_value();
        
        // 用原始价值填充减少后的访问记录
        for (size_t i = 0; i < new_visits; ++i) {
            backtrack_node->add_value(original_value);
        }
    }
    
    // 2. 增加不确定性以鼓励更多探索
    double current_uncertainty = backtrack_node->get_uncertainty();
    backtrack_node->set_uncertainty(std::min(1.0, current_uncertainty + 0.3));
    
    // 3. 增加新颖性分数
    double current_novelty = backtrack_node->get_novelty_score();
    backtrack_node->set_novelty_score(current_novelty + 0.2);
    
    // 4. 更新回溯计数
    backtrack_node->increment_backtrack_count();
    
    // 5. 更新最后访问时间
    backtrack_node->update_last_visit_time();
    last_visit_times[node_id] = backtrack_node->get_last_visit_time();
    
    // 6. 从被忽视分支列表中移除（如果存在）
    auto it = std::find(neglected_branches.begin(), neglected_branches.end(), node_id);
    if (it != neglected_branches.end()) {
        neglected_branches.erase(it);
    }
    
    // 7. 记录回溯事件到统计信息
    stats.backtrack_events++;
    stats.last_update_time = std::chrono::steady_clock::now();
    
    LOG_INFO("执行回溯操作: %s (原始访问: %zu, 当前访问: %zu)", 
             node_id.c_str(), original_visits, backtrack_node->get_visit_count());
}

// 重新发现被忽视的分支
void DiversityCalculator::rediscover_neglected_branches(
    const std::vector<std::string>& neglected_branches,
    std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
    HMGSStatistics& stats,
    std::unordered_map<std::string, std::chrono::steady_clock::time_point>& last_visit_times) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (neglected_branches.empty()) {
        return;
    }
    
    LOG_INFO("重新发现%zu个被忽视的分支", neglected_branches.size());
    
    // 对每个被忽视分支执行重新发现操作
    for (const auto& branch_id : neglected_branches) {
        auto it = node_registry.find(branch_id);
        if (it == node_registry.end()) {
            continue;
        }
        
        auto branch = it->second;
        if (!branch) {
            continue;
        }
        
        // 1. 重置访问计数（保留部分历史记录）
        size_t original_visits = branch->get_visit_count();
        if (original_visits > 0) {
            // 保留25%的历史记录
            size_t new_visits = std::max<size_t>(1, original_visits / 4);
            double original_value = branch->get_average_value();
            
            branch->reset_visit_count();
            branch->reset_total_value();
            
            for (size_t i = 0; i < new_visits; ++i) {
                branch->add_value(original_value);
            }
        } else {
            // 从未被访问过，添加初始访问记录
            branch->add_value(0.0);
            branch->increment_visit_count();
        }
        
        // 2. 设置高不确定性以鼓励探索
        branch->set_uncertainty(0.8);
        
        // 3. 重置新颖性分数到中等水平
        branch->set_novelty_score(0.5);
        
        // 4. 清零回溯计数
        branch->reset_backtrack_count();
        
        // 5. 更新最后访问时间
        branch->update_last_visit_time();
        last_visit_times[branch_id] = branch->get_last_visit_time();
        
        // 6. 暂时标记为受保护状态，防止立即再次被忽视
        branch->set_protected(true);
        
        // 7. 更新统计信息
        stats.total_simulations += 1;
        stats.completed_simulations += 1;
    }
    
    // 更新统计信息时间戳
    stats.last_update_time = std::chrono::steady_clock::now();
    
    LOG_DEBUG("重新发现操作完成");
}

} // namespace algorithms
} // namespace spacego
