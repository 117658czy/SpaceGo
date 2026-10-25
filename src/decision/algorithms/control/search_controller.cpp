// 搜索控制器实现文件
// 包含所有搜索流程控制和协调的具体实现

#include "network/algorithms/control/search_controller.h"
#include "network/algorithms/calculation/diversity_calculator.h"
#include "network/algorithms/data/node_manager.h"
#include "network/algorithms/data/merge_network.h"
#include "network/algorithms/execution/simulation_executor.h"
#include "framework/logger.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace spacego {
namespace algorithms {

// 构造函数
SearchController::SearchController()
    : diversity_calculator_(nullptr)
    , node_manager_(nullptr)
    , merge_network_(nullptr)
    , simulation_executor_(nullptr) {
}

// 析构函数
SearchController::~SearchController() {
    // 确保在析构时停止所有运行中的线程
    if (searching_.load(std::memory_order_acquire)) {
        stop_search();
    }
}

// 初始化（依赖注入）
void SearchController::initialize(
    DiversityCalculator* diversity_calc,
    NodeManager* node_mgr,
    MergeNetwork* merge_net,
    SimulationExecutor* sim_executor,
    const HMGSConfig& config) {
    
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    // 注入依赖的专职类实例
    diversity_calculator_ = diversity_calc;
    node_manager_ = node_mgr;
    merge_network_ = merge_net;
    simulation_executor_ = sim_executor;
    
    // 保存配置
    config_ = config;
    
    // 初始化统计信息
    stats_ = HMGSStatistics();
    stats_.start_time = std::chrono::steady_clock::now();
    stats_.last_update_time = stats_.start_time;
    
    initialized_.store(true, std::memory_order_release);
    
    LOG_INFO("搜索控制器初始化完成");
}

// 启动搜索
void SearchController::start_search(const std::string& initial_state) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    // 防止重复启动
    if (searching_.load(std::memory_order_acquire)) {
        LOG_WARNING("搜索已在运行中，忽略启动请求");
        return;
    }
    
    // 创建根节点
    root_node_ = node_manager_->create_root_node(initial_state, node_registry_);
    
    // 初始化并行分支结构
    initialize_parallel_branches();
    
    // 设置状态标志
    searching_.store(true, std::memory_order_release);
    paused_.store(false, std::memory_order_release);
    
    // 启动后台搜索线程
    search_thread_ = std::make_unique<std::thread>([this]() {
        this->background_search_thread();
    });
    
    // 启动监控线程
    monitoring_thread_ = std::make_unique<std::thread>([this]() {
        this->monitoring_thread();
    });
    
    // 更新统计时间戳
    stats_.start_time = std::chrono::steady_clock::now();
    stats_.last_update_time = stats_.start_time;
    
    LOG_INFO("搜索已启动: 初始状态长度=%zu", initial_state.size());
}

// 停止搜索
void SearchController::stop_search() {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    // 检查是否正在运行
    if (!searching_.load(std::memory_order_acquire)) {
        return;  // 未在运行，直接返回
    }
    
    // 设置停止标志
    searching_.store(false, std::memory_order_release);
    paused_.store(false, std::memory_order_release);
    
    LOG_INFO("正在停止搜索...");
    
    // 等待后台搜索线程安全退出
    if (search_thread_ && search_thread_->joinable()) {
        search_thread_->join();
        search_thread_.reset();
    }
    
    // 等待监控线程安全退出
    if (monitoring_thread_ && monitoring_thread_->joinable()) {
        monitoring_thread_->join();
        monitoring_thread_.reset();
    }
    
    LOG_INFO("搜索已完全停止");
}

// 暂停搜索
void SearchController::pause_search() {
    paused_.store(true, std::memory_order_release);
    LOG_DEBUG("搜索已暂停");
}

// 恢复搜索
void SearchController::resume_search() {
    paused_.store(false, std::memory_order_release);
    LOG_DEBUG("搜索已恢复");
}

// 执行单次模拟
void SearchController::run_simulation() {
    // 前置条件检查：必须有根节点和主分支
    if (!root_node_ || main_branches_.empty()) {
        return;
    }
    
    // 使用轮询策略选择要模拟的分支（确保负载均衡）
    static size_t current_branch_index = 0;
    size_t branch_index = current_branch_index;
    current_branch_index = (current_branch_index + 1) % main_branches_.size();
    
    // 运行选中分支的完整模拟流程
    run_branch_simulation(branch_index);
    
    // 更新统计信息时间戳
    stats_.last_update_time = std::chrono::steady_clock::now();
}

// 执行多次模拟
void SearchController::run_simulations(size_t count) {
    for (size_t i = 0; i < count; ++i) {
        // 检查是否应该停止
        if (!searching_.load(std::memory_order_acquire)) {
            break;  // 收到停止信号，提前退出
        }
        
        run_simulation();  // 执行单次模拟
    }
}

// 获取最佳走法
std::string SearchController::get_best_move() const {
    std::lock_guard<std::mutex> tree_lock(tree_mutex_);
    
    // 前置条件检查
    if (!root_node_ || main_branches_.empty()) {
        return "";  // 无有效数据，返回空字符串
    }
    
    // 选择价值最高的分支作为推荐走法
    std::shared_ptr<HMGSNode> best_branch = nullptr;
    double best_value = -std::numeric_limits<double>::infinity();
    
    for (const auto& branch : main_branches_) {
        if (branch && branch->get_visit_count() > 0) {
            double branch_value = branch->get_average_value();
            
            if (branch_value > best_value) {
                best_value = branch_value;
                best_branch = branch;
            }
        }
    }
    
    if (!best_branch) {
        return "";  // 没有有效分支
    }
    
    // 从分支节点状态中提取实际走法坐标
    std::string root_state = root_node_->get_state();
    std::string branch_state = best_branch->get_state();
    
    // 检查分支状态是否为根状态的扩展
    if (branch_state.size() >= root_state.size() && 
        branch_state.substr(0, root_state.size()) == root_state) {
        
        // 提取扩展部分作为走法描述
        if (branch_state.size() > root_state.size()) {
            std::string move_desc = branch_state.substr(root_state.size());
            
            // 检查是否是pass走法
            if (move_desc.find("_pass") != std::string::npos) {
                return "pass";
            }
            
            // 尝试解析坐标信息并返回格式化的走法字符串
            return "move_" + move_desc;
        }
    }
    
    // 如果不是扩展状态，比较棋盘状态找出差异位置
    if (root_state.size() == 361 && branch_state.size() == 361) {
        // 标准19x19棋盘编码，逐位比较找出落子位置
        for (size_t i = 0; i < 361; ++i) {
            if (root_state[i] != branch_state[i]) {
                // 发现差异位置，转换为围棋坐标
                int row = static_cast<int>(i / 19);
                int col = static_cast<int>(i % 19);
                
                // 调用坐标转换工具方法
                return simulation_executor_->coordinate_to_string(row, col);
            }
        }
    }
    
    // 无法提取坐标时返回通用标识符
    return "state_based_move";
}

// 获取走法概率分布
std::vector<std::pair<std::string, double>> SearchController::get_move_probabilities() const {
    std::lock_guard<std::mutex> tree_lock(tree_mutex_);
    
    std::vector<std::pair<std::string, double>> result;
    
    // 前置条件检查
    if (!root_node_ || main_branches_.empty()) {
        return result;  // 返回空列表
    }
    
    // 计算总访问次数用于归一化
    size_t total_visits = 0;
    for (const auto& branch : main_branches_) {
        if (branch) {
            total_visits += branch->get_visit_count();
        }
    }
    
    if (total_visits == 0) {
        return result;  // 无访问记录，返回空列表
    }
    
    // 计算每个分支的选择概率（基于访问次数占比）
    for (const auto& branch : main_branches_) {
        if (branch && branch->get_visit_count() > 0) {
            double probability = static_cast<double>(branch->get_visit_count()) / total_visits;
            result.emplace_back(branch->get_state(), probability);
        }
    }
    
    return result;
}

// 更新根节点
void SearchController::update_root(const std::string& move) {
    std::lock_guard<std::mutex> tree_lock(tree_mutex_);
    
    if (!root_node_) {
        return;
    }
    
    // 在现有分支中查找与走法匹配的新根节点
    std::shared_ptr<HMGSNode> new_root = nullptr;
    
    for (const auto& branch : main_branches_) {
        if (branch && branch->get_state() == move) {
            new_root = branch;
            break;
        }
    }
    
    if (!new_root) {
        // 没有匹配的分支，创建全新的根节点
        root_node_ = node_manager_->create_root_node(move, node_registry_);
        initialize_parallel_branches();  // 重新初始化分支结构
        
        stats_.last_update_time = std::chrono::steady_clock::now();
        return;
    }
    
    // 创建新的HMGSRootNode，继承匹配分支的子树和统计信息
    // 禁用dynamic_pointer_cast<HMGSRootNode>：分支节点是HMGSNode类型，转换必然失败
    // 改为直接创建新根节点并转移子树指针，避免丢弃整棵搜索树
    auto new_root_node = std::make_shared<HMGSRootNode>(move);
    
    // 继承旧根节点的合并阈值配置
    new_root_node->set_merge_threshold(root_node_->get_merge_threshold());
    
    // 继承匹配分支的访问统计（访问次数、总价值）
    new_root_node->set_visit_count(new_root->get_visit_count());
    new_root_node->set_total_value(new_root->get_total_value());
    
    // 继承匹配分支的不确定性和新颖性
    new_root_node->set_uncertainty(new_root->get_uncertainty());
    new_root_node->set_novelty_score(new_root->get_novelty_score());
    
    // 转移匹配分支的所有子节点到新根节点（保留子树，不丢弃）
    const auto& branch_children = new_root->get_children();
    for (const auto& child : branch_children) {
        if (child) {
            // 重新设置父节点为新根节点
            child->set_parent(new_root_node);
            // 子节点深度重置为1（根的直接子节点）
            child->set_depth(1);
            new_root_node->add_child(child);
        }
    }
    
    // 继承先验概率和走法列表
    const auto& branch_prior_probs = new_root->get_prior_probs();
    if (!branch_prior_probs.empty()) {
        new_root_node->set_prior_probs(branch_prior_probs);
    }
    const auto& branch_moves = new_root->get_moves();
    if (!branch_moves.empty()) {
        new_root_node->set_moves(branch_moves);
    }
    
    // 继承扩展状态
    if (new_root->is_expanded()) {
        new_root_node->set_expanded(true);
    }
    
    // 注册新根节点到全局节点注册表
    node_registry_[new_root_node->get_id()] = new_root_node;
    
    // 标记旧分支节点为已合并状态（防止后续搜索误用已转移的节点）
    new_root->mark_as_merged();
    new_root->set_merged_into_node(new_root_node);
    
    // 更新根节点指针
    root_node_ = new_root_node;
    
    // 重新初始化并行分支
    initialize_parallel_branches();
    
    stats_.last_update_time = std::chrono::steady_clock::now();
    
    LOG_DEBUG("根节点已更新: %s (继承子树: %zu个子节点)", 
              move.substr(0, 50).c_str(), branch_children.size());
}

// 重置搜索树
void SearchController::reset_tree() {
    std::lock_guard<std::mutex> tree_lock(tree_mutex_);
    
    // 清空树结构但保留根节点引用
    clear_tree();
    
    if (root_node_) {
        // 重置根节点的统计信息
        root_node_->reset_visit_count();
        root_node_->reset_total_value();
        
        // 重新初始化分支结构
        initialize_parallel_branches();
    }
    
    // 重置统计信息到初始状态
    stats_ = HMGSStatistics();
    stats_.start_time = std::chrono::steady_clock::now();
    stats_.last_update_time = stats_.start_time;
    
    LOG_INFO("搜索树已重置");
}

// 清空搜索树
void SearchController::clear_tree() {
    std::lock_guard<std::mutex> tree_lock(tree_mutex_);
    
    // 清除所有节点数据
    node_registry_.clear();
    main_branches_.clear();
    
    // 清除合并网络数据
    merge_probabilities_.clear();
    merged_into_probabilities_.clear();
    merge_relationships_.clear();
    unmerged_counts_.clear();
    
    // 清除多样性保护数据
    last_visit_times_.clear();
    neglected_branches_.clear();
    
    // 重置根节点指针
    root_node_.reset();
    
    LOG_DEBUG("搜索树数据已清空");
}

// 后台搜索主线程
void SearchController::background_search_thread() {
    LOG_INFO("后台搜索线程已启动");
    
    // 主循环：持续执行直到收到停止信号
    while (searching_.load(std::memory_order_acquire) && 
           !shutdown_requested_.load(std::memory_order_acquire)) {
        
        // 检查暂停状态
        if (paused_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;  // 暂停状态下等待
        }
        
        // 检查模拟次数限制
        if (config_.max_simulations > 0 && 
            stats_.completed_simulations >= config_.max_simulations) {
            searching_.store(false, std::memory_order_release);  // 达到上限，自动停止
            break;
        }
        
        // 执行一轮完整的模拟-扩展-回溯流程
        run_simulation();
        
        // 定期管理分支（按配置频率触发）
        static size_t branch_management_counter = 0;
        ++branch_management_counter;
        
        if (branch_management_counter >= config_.branch_update_frequency) {
            branch_management_counter = 0;
            manage_branches();  // 执行分支管理操作
        }
        
        // 定期检查合并机会（按配置频率触发）
        static size_t merge_check_counter = 0;
        ++merge_check_counter;
        
        if (merge_check_counter >= config_.merge_check_frequency) {
            merge_check_counter = 0;
            merge_network_->check_for_merges(
                root_node_, node_registry_, config_, stats_,
                merge_probabilities_, unmerged_counts_, last_visit_times_);
        }
        
        // 定期识别被忽视的分支（每20次模拟检查一次）
        static size_t neglect_check_counter = 0;
        ++neglect_check_counter;
        
        if (neglect_check_counter >= 20) {
            neglect_check_counter = 0;
            identify_neglected_branches();
        }
        
        // 小延迟以防止CPU占用过高
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    LOG_INFO("后台搜索线程已退出");
}

// 监控线程
void SearchController::monitoring_thread() {
    LOG_INFO("监控线程已启动");
    
    // 监控循环：持续运行直到搜索停止
    while (searching_.load(std::memory_order_acquire) && 
           !shutdown_requested_.load(std::memory_order_acquire)) {
        
        // 更新统计信息时间戳
        stats_.last_update_time = std::chrono::steady_clock::now();
        
        // 定期调整多样性参数（每50次循环）
        static size_t diversity_adjust_counter = 0;
        ++diversity_adjust_counter;
        
        if (diversity_adjust_counter >= 50) {
            diversity_adjust_counter = 0;
            adjust_diversity_parameters();  // 动态调整探索参数
        }
        
        // 定期重新计算合并概率（每30次循环）
        static size_t merge_prob_recalc_counter = 0;
        ++merge_prob_recalc_counter;
        
        if (merge_prob_recalc_counter >= 30) {
            merge_prob_recalc_counter = 0;
            merge_network_->recalculate_merge_probabilities(
                node_registry_, config_, stats_,
                merge_probabilities_, merged_into_probabilities_, merge_relationships_);
        }
        
        // 休眠以降低监控开销
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    LOG_INFO("监控线程已退出");
}

// 初始化并行分支
void SearchController::initialize_parallel_branches() {
    std::lock_guard<std::mutex> tree_lock(tree_mutex_);
    
    if (!root_node_) {
        return;  // 必须先有根节点
    }
    
    // 清空现有的分支列表
    main_branches_.clear();
    
    LOG_INFO("初始化%zu个并行分支...", config_.parallel_branches);
    
    // 创建配置数量的主分支
    for (size_t branch_idx = 0; branch_idx < config_.parallel_branches; ++branch_idx) {
        // 为每个分支创建唯一的初始状态
        std::string branch_state = root_node_->get_state() + "_branch_" + std::to_string(branch_idx);
        
        // 通过节点管理器创建分支根节点
        auto branch_node = node_manager_->create_node(branch_state, root_node_, node_registry_, last_visit_times_);
        
        if (branch_node) {
            // 设置分支特定属性
            branch_node->set_branch_index(branch_idx);
            branch_node->set_replica_count(config_.replicas_per_branch);
            
            // 初始化分支质量评分
            branch_node->set_branch_quality(0.5);  // 初始中等质量
            
            // 添加到主分支列表
            main_branches_.push_back(branch_node);
            
            // 初始化该分支的合并相关数据
            unmerged_counts_[branch_node->get_id()] = 0;
            merge_probabilities_[branch_node->get_id()] = 0.5;
            merged_into_probabilities_[branch_node->get_id()] = 0.0;
            
            // 为每个分支创建副本节点（用于并行模拟）
            for (size_t replica_idx = 0; replica_idx < config_.replicas_per_branch; ++replica_idx) {
                std::string replica_state = branch_state + "_replica_" + std::to_string(replica_idx);
                
                auto replica_node = node_manager_->create_node(
                    replica_state, branch_node, node_registry_, last_visit_times_);
                
                if (replica_node) {
                    // 设置副本特定属性
                    replica_node->set_replica_index(replica_idx);
                    replica_node->set_branch_index(branch_idx);
                    replica_node->set_is_replica(true);
                    
                    // 初始副本具有最高不确定性和较高新颖性
                    replica_node->set_uncertainty(1.0);
                    replica_node->set_novelty_score(0.7);
                }
            }
        }
    }
    
    // 更新统计信息
    stats_.last_update_time = std::chrono::steady_clock::now();
    
    LOG_INFO("并行分支初始化完成: %zu个主分支", main_branches_.size());
}

// 运行单个分支的模拟
void SearchController::run_branch_simulation(size_t branch_index) {
    // 边界检查
    if (branch_index >= main_branches_.size()) {
        return;
    }
    
    std::lock_guard<std::mutex> tree_lock(tree_mutex_);
    
    auto branch_node = main_branches_[branch_index];
    if (!branch_node) {
        return;
    }
    
    // 记录开始时间用于性能监控
    auto start_time = std::chrono::steady_clock::now();
    
    // 异步运行所有副本的模拟任务
    std::vector<std::future<std::pair<double, bool>>> replica_futures;
    
    for (size_t replica_idx = 0; replica_idx < config_.replicas_per_branch; ++replica_idx) {
        // 使用std::async异步启动每个副本的模拟
        replica_futures.push_back(std::async(std::launch::async, 
            [this, branch_index, replica_idx]() {
                return this->run_replica_simulation(branch_index, replica_idx);
            }));
    }
    
    // 收集所有副本的模拟结果
    std::vector<double> replica_values;
    std::vector<bool> replica_successes;
    
    for (auto& future : replica_futures) {
        auto result = future.get();  // 等待异步任务完成并获取结果
        replica_values.push_back(result.first);
        replica_successes.push_back(result.second);
    }
    
    // 合并所有副本的结果到分支节点
    merge_replica_results(branch_index);
    
    // 更新分支质量评估
    if (!replica_values.empty()) {
        // 计算平均价值和成功率
        double avg_value = 0.0;
        size_t success_count = 0;
        
        for (size_t i = 0; i < replica_values.size(); ++i) {
            avg_value += replica_values[i];
            if (replica_successes[i]) {
                ++success_count;
            }
        }
        
        avg_value /= replica_values.size();
        double success_rate = static_cast<double>(success_count) / replica_values.size();
        
        // 分支质量 = 平均价值 x 成功率（综合评价指标）
        double branch_quality = avg_value * success_rate;
        branch_node->set_branch_quality(branch_quality);
        
        // 更新分支的价值统计
        branch_node->add_value(avg_value);
        branch_node->increment_visit_count();
        
        // 更新全局统计信息
        stats_.total_simulations += replica_values.size();
        stats_.completed_simulations += replica_values.size();
        
        // 增量更新平均分支价值
        size_t completed = stats_.completed_simulations;
        stats_.average_branch_value = 
            (stats_.average_branch_value * (completed - replica_values.size()) + 
             avg_value * replica_values.size()) / completed;
        
        stats_.max_branch_value = std::max(stats_.max_branch_value, avg_value);
        stats_.min_branch_value = std::min(stats_.min_branch_value, avg_value);
    }
    
    // 记录模拟耗时
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // 更新分支的最后访问时间
    branch_node->update_last_visit_time();
    last_visit_times_[branch_node->get_id()] = branch_node->get_last_visit_time();
    
    stats_.last_update_time = std::chrono::steady_clock::now();
}

// 运行单个副本的模拟
std::pair<double, bool> SearchController::run_replica_simulation(size_t branch_index, size_t replica_index) {
    // 边界检查
    if (branch_index >= main_branches_.size()) {
        return {0.0, false};  // 无效分支索引
    }
    
    auto branch_node = main_branches_[branch_index];
    if (!branch_node) {
        return {0.0, false};  // 分支不存在
    }
    
    // 查找或获取对应的副本节点
    std::shared_ptr<HMGSNode> replica_node = nullptr;
    std::string replica_id = branch_node->get_id() + "_replica_" + std::to_string(replica_index);
    
    auto it = node_registry_.find(replica_id);
    if (it != node_registry_.end()) {
        replica_node = it->second;  // 已存在的副本节点
    } else {
        // 副本节点不存在，动态创建新的副本节点
        std::string replica_state = branch_node->get_state() + "_replica_" + std::to_string(replica_index);
        replica_node = node_manager_->create_node(replica_state, branch_node, node_registry_, last_visit_times_);
        
        if (replica_node) {
            // 初始化新副本的属性
            replica_node->set_replica_index(replica_index);
            replica_node->set_branch_index(branch_index);
            replica_node->set_is_replica(true);
            replica_node->set_uncertainty(1.0);
            replica_node->set_novelty_score(0.7);
        }
    }
    
    if (!replica_node) {
        return {0.0, false};  // 副本节点创建失败
    }
    
    // 执行指定次数的完整模拟
    double total_value = 0.0;
    size_t successful_simulations = 0;
    const size_t simulations_per_replica = 10;  // 每个副本默认执行10次模拟
    
    for (size_t sim_idx = 0; sim_idx < simulations_per_replica; ++sim_idx) {
        // 选择叶子节点进行扩展
        auto leaf_node = node_manager_->select_leaf_node(replica_node, config_);
        
        if (leaf_node && leaf_node != replica_node) {
            // 扩展叶子节点（生成子节点）
            node_manager_->expand_node(
                leaf_node, config_, node_registry_, stats_,
                [this](const std::string& state) {
                    return simulation_executor_->generate_legal_moves(state);
                });
            
            // 从叶子节点状态开始执行完整围棋模拟
            double simulation_value = simulation_executor_->perform_full_simulation(leaf_node->get_state());
            
            // 回溯模拟结果价值
            node_manager_->backup_value(
                leaf_node, simulation_value, config_, stats_, last_visit_times_);
            
            // 累加结果
            total_value += simulation_value;
            ++successful_simulations;
            
            // 更新副本节点的统计信息
            replica_node->add_value(simulation_value);
            replica_node->increment_visit_count();
        }
    }
    
    // 计算本次运行的统计指标
    double avg_value = (successful_simulations > 0) ? total_value / successful_simulations : 0.0;
    bool success = (successful_simulations > 0);
    
    // 根据访问次数动态调整副本的不确定性（访问越多，不确定性越低）
    if (replica_node->get_visit_count() > 0) {
        double new_uncertainty = 1.0 / (1.0 + std::log1p(replica_node->get_visit_count()));
        replica_node->set_uncertainty(new_uncertainty);
    }
    
    // 衰减新颖性分数（随着探索增加，新颖性逐渐降低）
    if (replica_node->get_visit_count() > 0) {
        double new_novelty = replica_node->get_novelty_score() * 
                            std::pow(config_.novelty_decay_factor, successful_simulations);
        replica_node->set_novelty_score(new_novelty);
    }
    
    // 更新最后访问时间
    replica_node->update_last_visit_time();
    last_visit_times_[replica_node->get_id()] = replica_node->get_last_visit_time();
    
    return {avg_value, success};
}

// 合并副本结果
void SearchController::merge_replica_results(size_t branch_index) {
    // 边界检查
    if (branch_index >= main_branches_.size()) {
        return;
    }
    
    std::lock_guard<std::mutex> tree_lock(tree_mutex_);
    
    auto branch_node = main_branches_[branch_index];
    if (!branch_node) {
        return;
    }
    
    // 收集该分支下的所有副本节点
    std::vector<std::shared_ptr<HMGSNode>> replica_nodes;
    for (const auto& entry : node_registry_) {
        auto node = entry.second;
        if (node && node->is_replica() && node->get_branch_index() == branch_index) {
            replica_nodes.push_back(node);
        }
    }
    
    if (replica_nodes.empty()) {
        return;  // 没有副本节点
    }
    
    // 汇总所有副本的价值和访问数据
    double total_value = 0.0;
    size_t total_visits = 0;
    
    for (const auto& replica : replica_nodes) {
        if (replica->get_visit_count() > 0) {
            total_value += replica->get_average_value() * replica->get_visit_count();
            total_visits += replica->get_visit_count();
        }
    }
    
    if (total_visits > 0) {
        // 计算加权平均值
        double merged_value = total_value / total_visits;
        
        // 与分支节点原有数据进行加权融合
        size_t branch_visits = branch_node->get_visit_count();
        double branch_value = branch_node->get_average_value();
        
        double new_value = (branch_value * branch_visits + merged_value * total_visits) / 
                          (branch_visits + total_visits);
        
        // 重置并设置融合后的新值
        branch_node->reset_visit_count();
        branch_node->reset_total_value();
        
        for (size_t i = 0; i < branch_visits + total_visits; ++i) {
            branch_node->add_value(new_value);
        }
        
        // 更新分支不确定性（取副本不确定性的平均值）
        double avg_uncertainty = 0.0;
        size_t replica_count = 0;
        
        for (const auto& replica : replica_nodes) {
            avg_uncertainty += replica->get_uncertainty();
            ++replica_count;
        }
        
        if (replica_count > 0) {
            avg_uncertainty /= replica_count;
            branch_node->set_uncertainty(avg_uncertainty);
        }
        
        // 更新分支新颖性（取副本新颖性的最大值）
        double max_novelty = 0.0;
        for (const auto& replica : replica_nodes) {
            max_novelty = std::max(max_novelty, replica->get_novelty_score());
        }
        branch_node->set_novelty_score(max_novelty);
        
        // 基于副本一致性计算合并概率
        double replica_consistency = 0.0;
        if (replica_nodes.size() > 1) {
            // 计算副本间价值的方差作为一致性度量
            std::vector<double> replica_values;
            
            for (const auto& replica : replica_nodes) {
                if (replica->get_visit_count() > 0) {
                    replica_values.push_back(replica->get_average_value());
                }
            }
            
            if (replica_values.size() > 1) {
                double sum = 0.0;
                for (double val : replica_values) sum += val;
                double mean = sum / replica_values.size();
                
                double variance = 0.0;
                for (double val : replica_values) variance += (val - mean) * (val - mean);
                variance /= replica_values.size();
                
                // 使用指数函数将方差映射到[0,1]的一致性分数
                replica_consistency = std::exp(-variance);
            }
        }
        
        // 更新分支的合并概率
        double new_merge_prob = replica_consistency;
        merge_probabilities_[branch_node->get_id()] = new_merge_prob;
        branch_node->set_merge_prob(new_merge_prob);
    }
    
    stats_.last_update_time = std::chrono::steady_clock::now();
}

// 管理所有分支
void SearchController::manage_branches() {
    std::lock_guard<std::mutex> tree_lock(tree_mutex_);
    
    if (main_branches_.empty()) {
        return;
    }
    
    // 使用静态计数器控制执行频率
    static size_t branch_update_counter = 0;
    ++branch_update_counter;
    
    if (branch_update_counter < config_.branch_update_frequency) {
        return;  // 未达到执行频率
    }
    
    branch_update_counter = 0;
    
    // 1. 计算各分支的质量分布
    std::vector<double> branch_qualities;
    for (const auto& branch : main_branches_) {
        if (branch) {
            branch_qualities.push_back(branch->get_branch_quality());
        }
    }
    
    if (branch_qualities.empty()) {
        return;
    }
    
    // 2. 计算平均质量
    double avg_quality = 0.0;
    for (double quality : branch_qualities) avg_quality += quality;
    avg_quality /= branch_qualities.size();
    
    // 3. 根据质量比率调整各分支的资源分配
    for (size_t i = 0; i < main_branches_.size(); ++i) {
        auto branch = main_branches_[i];
        if (!branch) continue;
        
        double quality = branch_qualities[i];
        double quality_ratio = quality / avg_quality;
        
        // 动态调整探索参数
        if (quality_ratio < 0.5) {
            // 低质量分支：增加探索激励（提高Cpuct值）
            branch->set_exploration_bias(config_.uct_c_puct * 1.5);
        } else if (quality_ratio > 1.5) {
            // 高质量分支：减少探索，加强利用（降低Cpuct值）
            branch->set_exploration_bias(config_.uct_c_puct * 0.7);
        }
        
        // 更新分支优先级（基于质量）
        branch->set_priority(quality);
    }
    
    // 4. 触发合并检查（如果配置了合并功能）
    if (config_.merge_check_frequency > 0) {
        static size_t merge_check_counter = 0;
        ++merge_check_counter;
        
        if (merge_check_counter >= config_.merge_check_frequency) {
            merge_check_counter = 0;
            merge_network_->check_for_merges(
                root_node_, node_registry_, config_, stats_,
                merge_probabilities_, unmerged_counts_, last_visit_times_);
        }
    }
    
    // 5. 触发多样性保护机制
    identify_neglected_branches();
    
    stats_.last_update_time = std::chrono::steady_clock::now();
}

// 多样性和回溯方法的委托实现

void SearchController::perform_backtrack() {
    auto backtrack_node = select_backtrack_node();
    if (backtrack_node) {
        diversity_calculator_->perform_backtrack(
            backtrack_node, stats_, neglected_branches_, last_visit_times_);
    }
}

void SearchController::rediscover_neglected_branches() {
    diversity_calculator_->rediscover_neglected_branches(
        neglected_branches_, node_registry_, stats_, last_visit_times_);
}

void SearchController::identify_neglected_branches() {
    neglected_branches_ = diversity_calculator_->identify_neglected_branches(
        node_registry_, root_node_, config_, stats_);
    
    // 自动保护被识别出的被忽视分支
    for (const auto& branch_id : neglected_branches_) {
        auto it = node_registry_.find(branch_id);
        if (it != node_registry_.end() && it->second) {
            diversity_calculator_->protect_diversity_branch(
                it->second, config_, stats_, last_visit_times_);
        }
    }
}

void SearchController::allocate_resources_to_neglected_branches(double total_resources) {
    diversity_calculator_->allocate_resources_to_neglected_branches(
        neglected_branches_, total_resources, node_registry_, stats_, last_visit_times_);
}

void SearchController::adjust_diversity_parameters() {
    diversity_calculator_->adjust_diversity_parameters(
        config_, node_registry_, root_node_, stats_);
}

std::shared_ptr<HMGSNode> SearchController::select_backtrack_node() const {
    return diversity_calculator_->select_backtrack_node(
        node_registry_, root_node_, neglected_branches_);
}

// 状态查询方法

bool SearchController::is_searching() const {
    return searching_.load(std::memory_order_acquire);
}

bool SearchController::is_initialized() const {
    return initialized_.load(std::memory_order_acquire);
}

HMGSStatistics SearchController::get_statistics() const {
    std::lock_guard<std::mutex> stats_lock(stats_mutex_);
    return stats_;
}

std::vector<std::string> SearchController::get_search_debug_info() const {
    std::vector<std::string> info;
    info.push_back("Searching: " + std::string(searching_.load() ? "true" : "false"));
    info.push_back("Initialized: " + std::string(initialized_.load() ? "true" : "false"));
    info.push_back("Paused: " + std::string(paused_.load() ? "true" : "false"));
    info.push_back("Node count: " + std::to_string(node_registry_.size()));
    info.push_back("Main branches: " + std::to_string(main_branches_.size()));
    info.push_back("Merge probabilities: " + std::to_string(merge_probabilities_.size()));
    info.push_back("Neglected branches: " + std::to_string(neglected_branches_.size()));
    return info;
}

} // namespace algorithms
} // namespace spacego
