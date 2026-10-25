// 节点管理器实现文件
// 包含所有节点管理操作的具体实现

#include "network/algorithms/data/node_manager.h"
#include "framework/logger.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace spacego {
namespace algorithms {

// 构造函数
NodeManager::NodeManager()
    : initialized_(false) {
}

// 析构函数
NodeManager::~NodeManager() {
}

// 初始化
void NodeManager::initialize(const HMGSConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    config_ = config;
    initialized_.store(true, std::memory_order_release);
    
    LOG_INFO("节点管理器初始化完成");
}

// 创建根节点
std::shared_ptr<HMGSRootNode> NodeManager::create_root_node(
    const std::string& state,
    std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 创建根节点实例
    auto root = std::make_shared<HMGSRootNode>(state);
    
    // 设置合并阈值（从配置中获取）
    root->set_merge_threshold(config_.merge_threshold);
    
    // 将根节点注册到全局节点注册表
    node_registry[root->get_id()] = root;
    
    LOG_INFO("创建根节点: %s (状态长度: %zu)", root->get_id().c_str(), state.size());
    
    return root;
}

// 创建普通节点
std::shared_ptr<HMGSNode> NodeManager::create_node(
    const std::string& state,
    std::shared_ptr<HMGSNode> parent,
    std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
    std::unordered_map<std::string, std::chrono::steady_clock::time_point>& last_visit_times) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 创建新节点
    auto node = std::make_shared<HMGSNode>(state);
    
    // 如果提供了父节点，建立父子关系
    if (parent) {
        node->set_parent(parent);
        node->set_depth(parent->get_depth() + 1);
        parent->add_child(node);
    }
    
    // 将新节点注册到全局节点注册表
    node_registry[node->get_id()] = node;
    
    // 初始化时间戳
    node->update_last_visit_time();
    last_visit_times[node->get_id()] = node->get_last_visit_time();
    
    LOG_DEBUG("创建节点: %s (深度: %zu, 父节点: %s)", 
              node->get_id().c_str(), node->get_depth(),
              (parent ? parent->get_id().c_str() : "无"));
    
    return node;
}

// 扩展节点
void NodeManager::expand_node(
    std::shared_ptr<HMGSNode> node,
    const HMGSConfig& config,
    std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
    HMGSStatistics& stats,
    std::function<std::vector<std::string>(const std::string&)> generate_legal_moves_fn) {
    
    if (!node || node->is_expanded()) {
        return;  // 节点为空或已经扩展，直接返回
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 生成合法落子位置列表
    std::vector<std::string> legal_moves;
    double value_estimate = 0.0;  // 初始价值估计（无神经网络时使用均匀值）
    
    // 调用外部传入的合法走法生成函数
    if (generate_legal_moves_fn) {
        legal_moves = generate_legal_moves_fn(node->get_state());
    }
    
    // 如果没有合法走法，添加pass选项
    if (legal_moves.empty()) {
        legal_moves.push_back(node->get_state() + "_pass");
    }
    
    // 为每个合法走法生成均匀先验概率
    std::vector<double> prior_probs(legal_moves.size(), 1.0 / legal_moves.size());
    
    // 为每个合法落子位置创建子节点
    if (legal_moves.empty()) {
        // 特殊情况：没有合法走法，只添加pass子节点
        auto pass_state = node->get_state() + "_pass";
        std::unordered_map<std::string, std::chrono::steady_clock::time_point> pass_visit_times;
        auto pass_child = create_node(pass_state, node, node_registry, pass_visit_times);
        
        if (pass_child) {
            pass_child->set_prior_probs({1.0});
            pass_child->add_value(value_estimate);
            pass_child->increment_visit_count();
            node->add_child(pass_child);
        }
    } else {
        // 正常情况：为每个合法走法创建子节点
        for (size_t i = 0; i < legal_moves.size(); ++i) {
            const auto& move_state = legal_moves[i];
            
            // 获取该走法的先验概率（如果存在）
            double move_prior = (i < prior_probs.size()) ? prior_probs[i] : (1.0 / legal_moves.size());
            
            // 创建子节点
            std::unordered_map<std::string, std::chrono::steady_clock::time_point> child_visit_times;
            auto child = create_node(move_state, node, node_registry, child_visit_times);
            
            if (child) {
                // 设置先验概率和初始价值
                child->set_prior_probs({move_prior});
                child->add_value(value_estimate);
                child->increment_visit_count();
                
                // 添加为当前节点的子节点
                node->add_child(child);
                
                // 记录子节点索引用于后续UCT计算
                child->set_child_index(i);
            }
        }
    }
    
    // 标记节点为已扩展状态
    node->set_expanded(true);
    
    // 更新不确定性估计（访问次数越多，不确定性越低）
    double uncertainty = 1.0 / (1.0 + node->get_visit_count());
    node->set_uncertainty(uncertainty);
    
    // 更新统计信息
    stats.node_expansions++;
    stats.last_update_time = std::chrono::steady_clock::now();
    
    LOG_DEBUG("扩展节点: %s (生成%zu个子节点)", node->get_id().c_str(), legal_moves.size());
}

// 回溯价值
void NodeManager::backup_value(
    std::shared_ptr<HMGSNode> node,
    double value,
    const HMGSConfig& config,
    HMGSStatistics& stats,
    std::unordered_map<std::string, std::chrono::steady_clock::time_point>& last_visit_times) {
    
    if (!node) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 选择回溯路径（根据配置的策略）
    auto backup_path = select_backup_path(node, config);
    
    // 沿着回溯路径更新每个节点的价值和统计信息
    for (auto& current : backup_path) {
        if (!current) continue;  // 跳过空节点
        
        // 更新节点的累计价值
        current->add_value(value);
        
        // 增加访问计数
        current->increment_visit_count();
        
        // 更新最后访问时间戳
        current->update_last_visit_time();
        
        // 同步更新全局最后访问时间记录
        last_visit_times[current->get_id()] = current->get_last_visit_time();
        
        // 收集搜索反馈到统计信息
        stats.total_simulations++;
        stats.completed_simulations++;
        
        // 更新分支价值统计信息
        double avg_value = current->get_average_value();
        
        // 使用增量方式计算新的平均分支价值，避免数值溢出
        size_t completed = stats.completed_simulations;
        stats.average_branch_value = 
            (stats.average_branch_value * (completed - 1) + avg_value) / completed;
        
        // 更新最大/最小分支价值
        stats.max_branch_value = std::max(stats.max_branch_value, avg_value);
        stats.min_branch_value = std::min(stats.min_branch_value, avg_value);
    }
    
    // 更新统计信息时间戳
    stats.last_update_time = std::chrono::steady_clock::now();
    
    LOG_DEBUG("完成价值回溯: 路径长度=%zu, 价值=%.4f", backup_path.size(), value);
}

// 选择回溯路径
std::vector<std::shared_ptr<HMGSNode>> NodeManager::select_backup_path(
    std::shared_ptr<HMGSNode> leaf_node,
    const HMGSConfig& config) const {
    
    std::vector<std::shared_ptr<HMGSNode>> path;
    
    if (!leaf_node) {
        return path;  // 空叶节点，返回空路径
    }
    
    // 根据配置选择不同的回溯策略
    if (config.backup_path_selection == "best_value") {
        // 策略1：最佳价值路径 - 直接从叶节点回溯到根节点
        auto current = leaf_node;
        while (current) {
            path.push_back(current);
            current = current->get_parent();  // 向上遍历父节点
        }
    } 
    else if (config.backup_path_selection == "topology_aware") {
        // 策略2：拓扑感知路径 - 考虑合并关系进行智能回溯
        auto current = leaf_node;
        while (current) {
            path.push_back(current);
            
            // 检查是否有合并伙伴可以跳转以优化回溯路径
            const auto& similar_nodes = current->get_similar_nodes();
            if (!similar_nodes.empty()) {
                // 从相似节点中选择价值最高的作为跳转目标
                std::shared_ptr<HMGSNode> best_similar = nullptr;
                double best_value = -std::numeric_limits<double>::infinity();
                
                for (const auto& similar : similar_nodes) {
                    if (similar && similar->get_average_value() > best_value) {
                        best_value = similar->get_average_value();
                        best_similar = similar;
                    }
                }
                
                // 如果找到更好的相似节点，跳转到其父节点继续回溯
                if (best_similar && best_similar != current) {
                    current = best_similar->get_parent();
                    continue;  // 跳过常规的父节点遍历
                }
            }
            
            // 没有可跳转的相似节点，继续向上遍历父节点
            current = current->get_parent();
        }
    } 
    else {
        // 默认策略：简单路径 - 直接从叶节点回溯到根节点
        auto current = leaf_node;
        while (current) {
            path.push_back(current);
            current = current->get_parent();
        }
    }
    
    return path;
}

// 选择最佳子节点（使用拓扑感知UCT算法）
std::shared_ptr<HMGSNode> NodeManager::select_best_child(
    std::shared_ptr<HMGSNode> node,
    const HMGSConfig& config) const {
    
    if (!node || node->get_children().empty()) {
        return nullptr;  // 节点为空或没有子节点
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    const auto& children = node->get_children();
    std::shared_ptr<HMGSNode> best_child = nullptr;
    double best_score = -std::numeric_limits<double>::infinity();
    
    // 遍历所有子节点，计算UCT分数并选择最优的
    for (size_t i = 0; i < children.size(); ++i) {
        auto child = children[i];
        if (!child) continue;  // 跳过空子节点
        
        // 使用拓扑感知UCT算法计算该子节点的分数
        double score = calculate_topology_uct(child, node, i, config);
        
        if (score > best_score) {
            best_score = score;
            best_child = child;
        }
    }
    
    return best_child;
}

// 选择叶子节点
std::shared_ptr<HMGSNode> NodeManager::select_leaf_node(
    std::shared_ptr<HMGSNode> start_node,
    const HMGSConfig& config) const {
    
    if (!start_node) {
        return nullptr;  // 起始节点为空
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto current = start_node;
    size_t depth = 0;
    
    // 沿着树向下选择直到达到最大搜索深度或叶子节点
    while (depth < config.max_search_depth) {
        // 检查是否已到达叶子节点或未扩展节点
        if (!current->is_expanded() || current->is_leaf()) {
            return current;  // 返回当前节点作为叶子节点
        }
        
        // 选择当前节点的最佳子节点继续向下
        auto best_child = select_best_child(current, config);
        if (!best_child) {
            break;  // 无法选择子节点，停止遍历
        }
        
        current = best_child;
        ++depth;
    }
    
    return current;
}

// 计算拓扑感知UCT值（私有方法）
double NodeManager::calculate_topology_uct(
    std::shared_ptr<HMGSNode> child,
    std::shared_ptr<HMGSNode> parent,
    size_t child_index,
    const HMGSConfig& config) const {
    
    if (!child || !parent) {
        return 0.0;  // 无效节点返回0
    }
    
    // 获取基本统计信息
    size_t parent_visits = parent->get_visit_count();
    size_t child_visits = child->get_visit_count();
    
    // 未访问过的子节点给予无限优先级以确保被选中
    if (child_visits == 0) {
        return std::numeric_limits<double>::infinity();
    }
    
    double child_value = child->get_average_value();
    
    // 基础UCT公式：利用项 + 探索项
    // 利用项：子节点的平均价值
    // 探索项：基于先验概率和访问次数的探索激励
    double exploration_term = config.uct_c_puct * 
                              child->get_prior_prob(child_index) * 
                              std::sqrt(parent_visits) / (1.0 + child_visits);
    
    // 拓扑感知修正因子
    double topology_correction = 1.0;
    
    // 合并概率修正：高合并概率的节点更容易被探索
    double merge_prob = child->get_merge_prob();
    if (config.merge_probability_weight > 0.0) {
        topology_correction *= (1.0 + config.merge_probability_weight * merge_prob);
    }
    
    // 被合并概率修正：容易被其他节点合并的节点给予额外探索机会
    double merged_into_prob = child->get_merged_into_prob();
    if (config.merged_into_probability_weight > 0.0) {
        topology_correction *= (1.0 + config.merged_into_probability_weight * merged_into_prob);
    }
    
    // 新颖性奖励：新颖性高的节点获得额外奖励
    double novelty_score = child->get_novelty_score();
    double novelty_reward = config.novelty_reward_base * novelty_score;
    
    // 回溯增强：经常被回溯的节点可能包含有价值的信息
    size_t backtrack_count = child->get_backtrack_count();
    double backtrack_boost = (backtrack_count > 0) ? 
                             1.0 + config.backtrack_boost_factor * std::log1p(backtrack_count) : 
                             1.0;
    
    // 不确定性引导：高不确定性的节点需要更多探索以降低不确定性
    double uncertainty = child->get_uncertainty();
    double uncertainty_exploration = 1.0 + config.diversity_gamma * uncertainty;
    
    // 组合所有修正项计算最终的拓扑感知UCT值
    double topology_uct = child_value + 
                          config.topology_uct_weight * exploration_term * topology_correction +
                          novelty_reward * backtrack_boost * uncertainty_exploration;
    
    return topology_uct;
}

} // namespace algorithms
} // namespace spacego
