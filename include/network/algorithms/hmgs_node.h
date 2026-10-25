#pragma once

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <chrono>

namespace spacego {
namespace algorithms {

class HMGSNode : public std::enable_shared_from_this<HMGSNode> {
public:
    // 构造函数和析构函数
    HMGSNode(const std::string& state);
    virtual ~HMGSNode();

    // 状态访问
    std::string get_state() const;
    void set_state(const std::string& state);
    
    // 访问统计
    size_t get_visit_count() const;
    void increment_visit_count();
    void set_visit_count(size_t count);
    
    double get_total_value() const;
    void add_value(double value);
    void set_total_value(double value);
    
    double get_average_value() const;
    
    // 节点关系
    std::shared_ptr<HMGSNode> get_parent() const;
    void set_parent(std::shared_ptr<HMGSNode> parent);
    
    const std::vector<std::shared_ptr<HMGSNode>>& get_children() const;
    void add_child(std::shared_ptr<HMGSNode> child);
    void remove_child(std::shared_ptr<HMGSNode> child);
    void clear_children();
    
    // 走法和先验概率
    const std::vector<std::string>& get_moves() const;
    void set_moves(const std::vector<std::string>& moves);
    
    const std::vector<double>& get_prior_probs() const;
    void set_prior_probs(const std::vector<double>& probs);
    
    double get_prior_prob(size_t move_index) const;
    void set_prior_prob(size_t move_index, double prob);
    
    // 子节点索引
    size_t get_child_index() const;
    void set_child_index(size_t index);
    
    // 合并相关属性
    double get_merge_prob() const;
    void set_merge_prob(double prob);
    void update_merge_prob(double delta);
    
    double get_merged_into_prob() const;
    void set_merged_into_prob(double prob);
    void update_merged_into_prob(double delta);
    
    size_t get_continuous_unmerged() const;
    void set_continuous_unmerged(size_t count);
    void increment_continuous_unmerged();
    void reset_continuous_unmerged();
    
    // 新颖性评分
    double get_novelty_score() const;
    void set_novelty_score(double score);
    void update_novelty_score(double delta);
    
    // 回溯计数
    size_t get_backtrack_count() const;
    void set_backtrack_count(size_t count);
    void increment_backtrack_count();
    void reset_backtrack_count();
    
    // 状态编码
    std::vector<double> get_state_encoding() const;
    void set_state_encoding(const std::vector<double>& encoding);
    
    // 不确定性估计
    double get_uncertainty() const;
    void set_uncertainty(double uncertainty);
    
    // 节点深度
    size_t get_depth() const;
    void set_depth(size_t depth);
    
    // 时间戳
    std::chrono::steady_clock::time_point get_last_visit_time() const;
    void update_last_visit_time();
    void set_last_visit_time(const std::chrono::steady_clock::time_point& time);
    
    // 相似节点管理
    const std::vector<std::shared_ptr<HMGSNode>>& get_similar_nodes() const;
    void add_similar_node(std::shared_ptr<HMGSNode> node);
    void remove_similar_node(std::shared_ptr<HMGSNode> node);
    void clear_similar_nodes();
    
    // 节点扩展状态
    bool is_expanded() const;
    void set_expanded(bool expanded);
    
    bool is_leaf() const;
    
    // 副本一致性计算
    double calculate_replica_consistency() const;
    void update_replica_consistency(double value);
    
    // 节点合并检查
    bool can_merge_with(std::shared_ptr<HMGSNode> other, double threshold = 0.8) const;
    
    // 线程安全操作
    void lock() const;
    void unlock() const;
    bool try_lock() const;
    
    // 节点克隆
    std::shared_ptr<HMGSNode> clone() const;
    
    // 节点ID
    std::string get_id() const;
    
    // 调试信息
    std::string to_string() const;
    
    // 分支和副本管理
    size_t get_branch_index() const;
    void set_branch_index(size_t index);
    size_t get_replica_index() const;
    void set_replica_index(size_t index);
    bool is_replica() const;
    void set_is_replica(bool is_replica);
    size_t get_replica_count() const;
    void set_replica_count(size_t count);
    
    // 分支质量
    double get_branch_quality() const;
    void set_branch_quality(double quality);
    
    // 参数调整
    double get_exploration_bias() const;
    void set_exploration_bias(double bias);
    double get_topology_weight() const;
    void set_topology_weight(double weight);
    double get_novelty_reward() const;
    void set_novelty_reward(double reward);
    double get_priority() const;
    void set_priority(double priority);
    
    // 保护状态
    bool is_protected() const;
    void set_protected(bool protected_status);
    
    // 合并状态
    bool is_merged() const;
    void mark_as_merged();
    std::shared_ptr<HMGSNode> get_merged_into_node() const;
    void set_merged_into_node(std::shared_ptr<HMGSNode> node);
    
    // 重置操作
    void reset_visit_count();
    void reset_total_value();
    
protected:
    // 节点状态
    std::string state_;
    std::vector<double> state_encoding_;
    
    // 访问统计
    std::atomic<size_t> visit_count_;
    std::atomic<double> total_value_;
    
    // 节点关系
    std::weak_ptr<HMGSNode> parent_;
    std::vector<std::shared_ptr<HMGSNode>> children_;
    
    // 走法和先验概率
    std::vector<std::string> moves_;
    std::vector<double> prior_probs_;
    
    // 子节点索引
    std::atomic<size_t> child_index_;
    
    // 合并相关属性
    std::atomic<double> merge_prob_;
    std::atomic<double> merged_into_prob_;
    std::atomic<size_t> continuous_unmerged_;
    
    // 新颖性评分
    std::atomic<double> novelty_score_;
    
    // 回溯计数
    std::atomic<size_t> backtrack_count_;
    
    // 不确定性估计
    std::atomic<double> uncertainty_;
    
    // 节点深度
    std::atomic<size_t> depth_;
    
    // 时间戳
    std::chrono::steady_clock::time_point last_visit_time_;
    
    // 相似节点
    std::vector<std::shared_ptr<HMGSNode>> similar_nodes_;
    
    // 节点扩展状态
    std::atomic<bool> expanded_;
    
    // 副本一致性
    std::atomic<double> replica_consistency_;
    
    // 分支和副本管理
    std::atomic<size_t> branch_index_;
    std::atomic<size_t> replica_index_;
    std::atomic<bool> is_replica_;
    std::atomic<size_t> replica_count_;
    
    // 分支质量
    std::atomic<double> branch_quality_;
    
    // 参数调整
    std::atomic<double> exploration_bias_;
    std::atomic<double> topology_weight_;
    std::atomic<double> novelty_reward_;
    std::atomic<double> priority_;
    
    // 保护状态
    std::atomic<bool> is_protected_;
    
    // 合并状态
    std::atomic<bool> is_merged_;
    std::weak_ptr<HMGSNode> merged_into_node_;
    
    // 线程锁
    mutable std::mutex mutex_;
    
    // 节点ID
    std::string id_;
    
private:
    // 生成唯一ID
    static std::string generate_id();
};

class HMGSRootNode : public HMGSNode {
public:
    HMGSRootNode(const std::string& state);
    ~HMGSRootNode();
    
    // 分支管理
    const std::vector<std::shared_ptr<HMGSNode>>& get_branches() const;
    std::shared_ptr<HMGSNode> get_branch(size_t index) const;
    void set_branch(size_t index, std::shared_ptr<HMGSNode> branch);
    void initialize_branches();
    void clear_branches();
    
    // 合并阈值
    double get_merge_threshold() const;
    void set_merge_threshold(double threshold);
    
    // 分支更新计数
    size_t get_branch_update_count() const;
    void set_branch_update_count(size_t count);
    void increment_branch_update_count();
    
    // 合并检查计数
    size_t get_merge_check_count() const;
    void set_merge_check_count(size_t count);
    void increment_merge_check_count();
    
    // 根节点特定属性
    bool is_root() const;
    
    // 分支质量评分
    double get_branch_quality(size_t index) const;
    void set_branch_quality(size_t index, double quality);
    void update_branch_quality(size_t index, double delta);
    
    // 分支访问时间
    std::chrono::steady_clock::time_point get_branch_last_access(size_t index) const;
    void update_branch_last_access(size_t index);
    
    // 分支统计
    size_t get_branch_simulation_count(size_t index) const;
    void increment_branch_simulation_count(size_t index);
    
    // 根节点克隆
    std::shared_ptr<HMGSRootNode> clone_root() const;
    
private:
    // 分支管理
    std::vector<std::shared_ptr<HMGSNode>> branches_;
    
    // 合并阈值
    std::atomic<double> merge_threshold_;
    
    // 分支更新计数
    std::atomic<size_t> branch_update_count_;
    
    // 合并检查计数
    std::atomic<size_t> merge_check_count_;
    
    // 分支质量评分
    std::vector<double> branch_qualities_;
    
    // 分支访问时间
    std::vector<std::chrono::steady_clock::time_point> branch_last_access_;
    
    // 分支模拟计数
    std::vector<size_t> branch_simulation_counts_;
};

} // namespace algorithms
} // namespace spacego