#include "network/algorithms/hmgs_node.h"
#include <algorithm>
#include <random>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace spacego {
namespace algorithms {

// 静态ID生成器
static std::atomic<size_t> global_node_id(0);

std::string HMGSNode::generate_id() {
    size_t id = global_node_id.fetch_add(1);
    std::stringstream ss;
    ss << "node_" << std::hex << std::setfill('0') << std::setw(8) << id;
    return ss.str();
}

// HMGSNode实现
HMGSNode::HMGSNode(const std::string& state) 
    : state_(state)
    , visit_count_(0)
    , total_value_(0.0)
    , merge_prob_(0.0)
    , merged_into_prob_(0.0)
    , continuous_unmerged_(0)
    , novelty_score_(0.0)
    , backtrack_count_(0)
    , uncertainty_(1.0)
    , depth_(0)
    , expanded_(false)
    , replica_consistency_(0.0)
    , branch_index_(0)
    , replica_index_(0)
    , is_replica_(false)
    , replica_count_(0)
    , branch_quality_(0.5)
    , exploration_bias_(1.0)
    , topology_weight_(1.0)
    , novelty_reward_(0.2)
    , priority_(0.0)
    , is_protected_(false)
    , is_merged_(false)
    , id_(generate_id()) {
    last_visit_time_ = std::chrono::steady_clock::now();
}

HMGSNode::~HMGSNode() {
    // 自动清理
}

std::string HMGSNode::get_state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

void HMGSNode::set_state(const std::string& state) {
    std::lock_guard<std::mutex> lock(mutex_);
    state_ = state;
}

size_t HMGSNode::get_visit_count() const {
    return visit_count_.load(std::memory_order_acquire);
}

void HMGSNode::increment_visit_count() {
    visit_count_.fetch_add(1, std::memory_order_release);
}

void HMGSNode::set_visit_count(size_t count) {
    visit_count_.store(count, std::memory_order_release);
}

double HMGSNode::get_total_value() const {
    return total_value_.load(std::memory_order_acquire);
}

void HMGSNode::add_value(double value) {
    // C++17的std::atomic<double>不支持fetch_add，使用CAS循环实现
    double current = total_value_.load(std::memory_order_acquire);
    while (!total_value_.compare_exchange_weak(current, current + value,
                                                std::memory_order_release,
                                                std::memory_order_acquire)) {}
}

void HMGSNode::set_total_value(double value) {
    total_value_.store(value, std::memory_order_release);
}

double HMGSNode::get_average_value() const {
    size_t visits = get_visit_count();
    if (visits == 0) return 0.0;
    return get_total_value() / visits;
}

std::shared_ptr<HMGSNode> HMGSNode::get_parent() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return parent_.lock();
}

void HMGSNode::set_parent(std::shared_ptr<HMGSNode> parent) {
    std::lock_guard<std::mutex> lock(mutex_);
    parent_ = parent;
}

const std::vector<std::shared_ptr<HMGSNode>>& HMGSNode::get_children() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return children_;
}

void HMGSNode::add_child(std::shared_ptr<HMGSNode> child) {
    std::lock_guard<std::mutex> lock(mutex_);
    children_.push_back(child);
}

void HMGSNode::remove_child(std::shared_ptr<HMGSNode> child) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = std::find(children_.begin(), children_.end(), child);
    if (it != children_.end()) {
        children_.erase(it);
    }
}

void HMGSNode::clear_children() {
    std::lock_guard<std::mutex> lock(mutex_);
    children_.clear();
}

const std::vector<std::string>& HMGSNode::get_moves() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return moves_;
}

void HMGSNode::set_moves(const std::vector<std::string>& moves) {
    std::lock_guard<std::mutex> lock(mutex_);
    moves_ = moves;
}

const std::vector<double>& HMGSNode::get_prior_probs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return prior_probs_;
}

void HMGSNode::set_prior_probs(const std::vector<double>& probs) {
    std::lock_guard<std::mutex> lock(mutex_);
    prior_probs_ = probs;
}

double HMGSNode::get_prior_prob(size_t move_index) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (move_index >= prior_probs_.size()) return 0.0;
    return prior_probs_[move_index];
}

void HMGSNode::set_prior_prob(size_t move_index, double prob) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (move_index >= prior_probs_.size()) {
        prior_probs_.resize(move_index + 1, 0.0);
    }
    prior_probs_[move_index] = prob;
}

double HMGSNode::get_merge_prob() const {
    return merge_prob_.load(std::memory_order_acquire);
}

void HMGSNode::set_merge_prob(double prob) {
    merge_prob_.store(std::max(0.0, std::min(1.0, prob)), std::memory_order_release);
}

void HMGSNode::update_merge_prob(double delta) {
    double current = merge_prob_.load(std::memory_order_acquire);
    double new_value = std::max(0.0, std::min(1.0, current + delta));
    merge_prob_.store(new_value, std::memory_order_release);
}

double HMGSNode::get_merged_into_prob() const {
    return merged_into_prob_.load(std::memory_order_acquire);
}

void HMGSNode::set_merged_into_prob(double prob) {
    merged_into_prob_.store(std::max(0.0, std::min(1.0, prob)), std::memory_order_release);
}

void HMGSNode::update_merged_into_prob(double delta) {
    double current = merged_into_prob_.load(std::memory_order_acquire);
    double new_value = std::max(0.0, std::min(1.0, current + delta));
    merged_into_prob_.store(new_value, std::memory_order_release);
}

size_t HMGSNode::get_continuous_unmerged() const {
    return continuous_unmerged_.load(std::memory_order_acquire);
}

void HMGSNode::set_continuous_unmerged(size_t count) {
    continuous_unmerged_.store(count, std::memory_order_release);
}

void HMGSNode::increment_continuous_unmerged() {
    continuous_unmerged_.fetch_add(1, std::memory_order_release);
}

void HMGSNode::reset_continuous_unmerged() {
    continuous_unmerged_.store(0, std::memory_order_release);
}

double HMGSNode::get_novelty_score() const {
    return novelty_score_.load(std::memory_order_acquire);
}

void HMGSNode::set_novelty_score(double score) {
    novelty_score_.store(std::max(0.0, score), std::memory_order_release);
}

void HMGSNode::update_novelty_score(double delta) {
    double current = novelty_score_.load(std::memory_order_acquire);
    novelty_score_.store(std::max(0.0, current + delta), std::memory_order_release);
}

size_t HMGSNode::get_backtrack_count() const {
    return backtrack_count_.load(std::memory_order_acquire);
}

void HMGSNode::set_backtrack_count(size_t count) {
    backtrack_count_.store(count, std::memory_order_release);
}

void HMGSNode::increment_backtrack_count() {
    backtrack_count_.fetch_add(1, std::memory_order_release);
}

void HMGSNode::reset_backtrack_count() {
    backtrack_count_.store(0, std::memory_order_release);
}

size_t HMGSNode::get_child_index() const {
    return child_index_.load(std::memory_order_acquire);
}

void HMGSNode::set_child_index(size_t index) {
    child_index_.store(index, std::memory_order_release);
}

std::vector<double> HMGSNode::get_state_encoding() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_encoding_;
}

void HMGSNode::set_state_encoding(const std::vector<double>& encoding) {
    std::lock_guard<std::mutex> lock(mutex_);
    state_encoding_ = encoding;
}

double HMGSNode::get_uncertainty() const {
    return uncertainty_.load(std::memory_order_acquire);
}

void HMGSNode::set_uncertainty(double uncertainty) {
    uncertainty_.store(std::max(0.0, std::min(1.0, uncertainty)), std::memory_order_release);
}

size_t HMGSNode::get_depth() const {
    return depth_.load(std::memory_order_acquire);
}

void HMGSNode::set_depth(size_t depth) {
    depth_.store(depth, std::memory_order_release);
}

std::chrono::steady_clock::time_point HMGSNode::get_last_visit_time() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_visit_time_;
}

void HMGSNode::update_last_visit_time() {
    std::lock_guard<std::mutex> lock(mutex_);
    last_visit_time_ = std::chrono::steady_clock::now();
}

void HMGSNode::set_last_visit_time(const std::chrono::steady_clock::time_point& time) {
    std::lock_guard<std::mutex> lock(mutex_);
    last_visit_time_ = time;
}

const std::vector<std::shared_ptr<HMGSNode>>& HMGSNode::get_similar_nodes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return similar_nodes_;
}

void HMGSNode::add_similar_node(std::shared_ptr<HMGSNode> node) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (std::find(similar_nodes_.begin(), similar_nodes_.end(), node) == similar_nodes_.end()) {
        similar_nodes_.push_back(node);
    }
}

void HMGSNode::remove_similar_node(std::shared_ptr<HMGSNode> node) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = std::find(similar_nodes_.begin(), similar_nodes_.end(), node);
    if (it != similar_nodes_.end()) {
        similar_nodes_.erase(it);
    }
}

void HMGSNode::clear_similar_nodes() {
    std::lock_guard<std::mutex> lock(mutex_);
    similar_nodes_.clear();
}

bool HMGSNode::is_expanded() const {
    return expanded_.load(std::memory_order_acquire);
}

void HMGSNode::set_expanded(bool expanded) {
    expanded_.store(expanded, std::memory_order_release);
}

bool HMGSNode::is_leaf() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return children_.empty();
}

double HMGSNode::calculate_replica_consistency() const {
    return replica_consistency_.load(std::memory_order_acquire);
}

void HMGSNode::update_replica_consistency(double value) {
    replica_consistency_.store(std::max(0.0, std::min(1.0, value)), std::memory_order_release);
}

bool HMGSNode::can_merge_with(std::shared_ptr<HMGSNode> other, double threshold) const {
    if (!other) return false;
    
    // 检查状态相似性（简单实现，实际应使用状态编码）
    if (state_ == other->state_) return true;
    
    // 检查价值兼容性
    double value_diff = std::abs(get_average_value() - other->get_average_value());
    if (value_diff > 0.5) return false; // 价值差异太大
    
    // 检查合并概率
    double merge_score = (get_merge_prob() + other->get_merge_prob()) / 2.0;
    return merge_score >= threshold;
}

void HMGSNode::lock() const {
    mutex_.lock();
}

void HMGSNode::unlock() const {
    mutex_.unlock();
}

bool HMGSNode::try_lock() const {
    return mutex_.try_lock();
}

// 分支和副本管理
size_t HMGSNode::get_branch_index() const {
    return branch_index_.load(std::memory_order_acquire);
}

void HMGSNode::set_branch_index(size_t index) {
    branch_index_.store(index, std::memory_order_release);
}

size_t HMGSNode::get_replica_index() const {
    return replica_index_.load(std::memory_order_acquire);
}

void HMGSNode::set_replica_index(size_t index) {
    replica_index_.store(index, std::memory_order_release);
}

bool HMGSNode::is_replica() const {
    return is_replica_.load(std::memory_order_acquire);
}

void HMGSNode::set_is_replica(bool is_replica) {
    is_replica_.store(is_replica, std::memory_order_release);
}

size_t HMGSNode::get_replica_count() const {
    return replica_count_.load(std::memory_order_acquire);
}

void HMGSNode::set_replica_count(size_t count) {
    replica_count_.store(count, std::memory_order_release);
}

// 分支质量
double HMGSNode::get_branch_quality() const {
    return branch_quality_.load(std::memory_order_acquire);
}

void HMGSNode::set_branch_quality(double quality) {
    branch_quality_.store(quality, std::memory_order_release);
}

// 参数调整
double HMGSNode::get_exploration_bias() const {
    return exploration_bias_.load(std::memory_order_acquire);
}

void HMGSNode::set_exploration_bias(double bias) {
    exploration_bias_.store(bias, std::memory_order_release);
}

double HMGSNode::get_topology_weight() const {
    return topology_weight_.load(std::memory_order_acquire);
}

void HMGSNode::set_topology_weight(double weight) {
    topology_weight_.store(weight, std::memory_order_release);
}

double HMGSNode::get_novelty_reward() const {
    return novelty_reward_.load(std::memory_order_acquire);
}

void HMGSNode::set_novelty_reward(double reward) {
    novelty_reward_.store(reward, std::memory_order_release);
}

double HMGSNode::get_priority() const {
    return priority_.load(std::memory_order_acquire);
}

void HMGSNode::set_priority(double priority) {
    priority_.store(priority, std::memory_order_release);
}

// 保护状态
bool HMGSNode::is_protected() const {
    return is_protected_.load(std::memory_order_acquire);
}

void HMGSNode::set_protected(bool protected_status) {
    is_protected_.store(protected_status, std::memory_order_release);
}

// 合并状态
bool HMGSNode::is_merged() const {
    return is_merged_.load(std::memory_order_acquire);
}

void HMGSNode::mark_as_merged() {
    is_merged_.store(true, std::memory_order_release);
}

std::shared_ptr<HMGSNode> HMGSNode::get_merged_into_node() const {
    return merged_into_node_.lock();
}

void HMGSNode::set_merged_into_node(std::shared_ptr<HMGSNode> node) {
    merged_into_node_ = node;
}

// 重置操作
void HMGSNode::reset_visit_count() {
    visit_count_.store(0, std::memory_order_release);
}

void HMGSNode::reset_total_value() {
    total_value_.store(0.0, std::memory_order_release);
}

std::shared_ptr<HMGSNode> HMGSNode::clone() const {
    auto new_node = std::make_shared<HMGSNode>(state_);
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 复制基本属性
    new_node->visit_count_.store(visit_count_.load(std::memory_order_acquire), std::memory_order_release);
    new_node->total_value_.store(total_value_.load(std::memory_order_acquire), std::memory_order_release);
    new_node->merge_prob_.store(merge_prob_.load(std::memory_order_acquire), std::memory_order_release);
    new_node->merged_into_prob_.store(merged_into_prob_.load(std::memory_order_acquire), std::memory_order_release);
    new_node->continuous_unmerged_.store(continuous_unmerged_.load(std::memory_order_acquire), std::memory_order_release);
    new_node->novelty_score_.store(novelty_score_.load(std::memory_order_acquire), std::memory_order_release);
    new_node->backtrack_count_.store(backtrack_count_.load(std::memory_order_acquire), std::memory_order_release);
    new_node->uncertainty_.store(uncertainty_.load(std::memory_order_acquire), std::memory_order_release);
    new_node->depth_.store(depth_.load(std::memory_order_acquire), std::memory_order_release);
    new_node->expanded_.store(expanded_.load(std::memory_order_acquire), std::memory_order_release);
    new_node->replica_consistency_.store(replica_consistency_.load(std::memory_order_acquire), std::memory_order_release);
    
    // 复制新增的原子变量
    new_node->branch_index_.store(branch_index_.load(std::memory_order_acquire), std::memory_order_release);
    new_node->replica_index_.store(replica_index_.load(std::memory_order_acquire), std::memory_order_release);
    new_node->is_replica_.store(is_replica_.load(std::memory_order_acquire), std::memory_order_release);
    new_node->replica_count_.store(replica_count_.load(std::memory_order_acquire), std::memory_order_release);
    new_node->branch_quality_.store(branch_quality_.load(std::memory_order_acquire), std::memory_order_release);
    new_node->exploration_bias_.store(exploration_bias_.load(std::memory_order_acquire), std::memory_order_release);
    new_node->topology_weight_.store(topology_weight_.load(std::memory_order_acquire), std::memory_order_release);
    new_node->novelty_reward_.store(novelty_reward_.load(std::memory_order_acquire), std::memory_order_release);
    new_node->priority_.store(priority_.load(std::memory_order_acquire), std::memory_order_release);
    new_node->is_protected_.store(is_protected_.load(std::memory_order_acquire), std::memory_order_release);
    new_node->is_merged_.store(is_merged_.load(std::memory_order_acquire), std::memory_order_release);
    
    // 复制弱指针
    new_node->merged_into_node_ = merged_into_node_;
    
    // 复制向量数据
    new_node->state_encoding_ = state_encoding_;
    new_node->moves_ = moves_;
    new_node->prior_probs_ = prior_probs_;
    new_node->last_visit_time_ = last_visit_time_;
    
    // 注意：不复制父节点、子节点和相似节点，这些需要外部设置
    
    return new_node;
}

std::string HMGSNode::get_id() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return id_;
}

std::string HMGSNode::to_string() const {
    std::stringstream ss;
    ss << "HMGSNode[id=" << get_id() 
       << ", visits=" << get_visit_count()
       << ", value=" << get_average_value()
       << ", depth=" << get_depth()
       << ", expanded=" << (is_expanded() ? "true" : "false")
       << ", children=" << get_children().size()
       << ", merge_prob=" << get_merge_prob()
       << ", novelty=" << get_novelty_score()
       << "]";
    return ss.str();
}

// HMGSRootNode实现
HMGSRootNode::HMGSRootNode(const std::string& state) 
    : HMGSNode(state)
    , merge_threshold_(0.8)
    , branch_update_count_(0)
    , merge_check_count_(0) {
    
    size_t default_branches = 8;
    branches_.resize(default_branches);
    branch_qualities_.resize(default_branches, 0.5);
    branch_last_access_.resize(default_branches, std::chrono::steady_clock::now());
    branch_simulation_counts_.resize(default_branches, 0);
}

HMGSRootNode::~HMGSRootNode() {
    // 自动清理
}

const std::vector<std::shared_ptr<HMGSNode>>& HMGSRootNode::get_branches() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return branches_;
}

std::shared_ptr<HMGSNode> HMGSRootNode::get_branch(size_t index) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (index >= branches_.size()) return nullptr;
    return branches_[index];
}

void HMGSRootNode::set_branch(size_t index, std::shared_ptr<HMGSNode> branch) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (index >= branches_.size()) {
        branches_.resize(index + 1);
        branch_qualities_.resize(index + 1, 0.5);
        branch_last_access_.resize(index + 1, std::chrono::steady_clock::now());
        branch_simulation_counts_.resize(index + 1, 0);
    }
    branches_[index] = branch;
    if (branch_qualities_[index] == 0.0) {
        branch_qualities_[index] = 0.5;
    }
}

void HMGSRootNode::initialize_branches() {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t num_branches = branches_.size();
    for (size_t i = 0; i < num_branches; ++i) {
        if (!branches_[i]) {
            auto branch = std::make_shared<HMGSNode>(get_state());
            branch->set_depth(1);
            branches_[i] = branch;
            branch_qualities_[i] = 0.5;
            branch_last_access_[i] = std::chrono::steady_clock::now();
            branch_simulation_counts_[i] = 0;
        }
    }
}

void HMGSRootNode::clear_branches() {
    std::lock_guard<std::mutex> lock(mutex_);
    branches_.clear();
    branch_qualities_.clear();
    branch_last_access_.clear();
    branch_simulation_counts_.clear();
}

double HMGSRootNode::get_merge_threshold() const {
    return merge_threshold_.load(std::memory_order_acquire);
}

void HMGSRootNode::set_merge_threshold(double threshold) {
    merge_threshold_.store(std::max(0.0, std::min(1.0, threshold)), std::memory_order_release);
}

size_t HMGSRootNode::get_branch_update_count() const {
    return branch_update_count_.load(std::memory_order_acquire);
}

void HMGSRootNode::set_branch_update_count(size_t count) {
    branch_update_count_.store(count, std::memory_order_release);
}

void HMGSRootNode::increment_branch_update_count() {
    branch_update_count_.fetch_add(1, std::memory_order_release);
}

size_t HMGSRootNode::get_merge_check_count() const {
    return merge_check_count_.load(std::memory_order_acquire);
}

void HMGSRootNode::set_merge_check_count(size_t count) {
    merge_check_count_.store(count, std::memory_order_release);
}

void HMGSRootNode::increment_merge_check_count() {
    merge_check_count_.fetch_add(1, std::memory_order_release);
}

bool HMGSRootNode::is_root() const {
    return true;
}

double HMGSRootNode::get_branch_quality(size_t index) const {
    if (index >= branch_qualities_.size()) return 0.0;
    return branch_qualities_[index];
}

void HMGSRootNode::set_branch_quality(size_t index, double quality) {
    if (index >= branch_qualities_.size()) return;
    branch_qualities_[index] = std::max(0.0, std::min(1.0, quality));
}

void HMGSRootNode::update_branch_quality(size_t index, double delta) {
    if (index >= branch_qualities_.size()) return;
    double current = branch_qualities_[index];
    double new_value = std::max(0.0, std::min(1.0, current + delta));
    branch_qualities_[index] = new_value;
}

std::chrono::steady_clock::time_point HMGSRootNode::get_branch_last_access(size_t index) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (index >= branch_last_access_.size()) return std::chrono::steady_clock::now();
    return branch_last_access_[index];
}

void HMGSRootNode::update_branch_last_access(size_t index) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (index < branch_last_access_.size()) {
        branch_last_access_[index] = std::chrono::steady_clock::now();
    }
}

size_t HMGSRootNode::get_branch_simulation_count(size_t index) const {
    if (index >= branch_simulation_counts_.size()) return 0;
    return branch_simulation_counts_[index];
}

void HMGSRootNode::increment_branch_simulation_count(size_t index) {
    if (index >= branch_simulation_counts_.size()) return;
    branch_simulation_counts_[index]++;
}

std::shared_ptr<HMGSRootNode> HMGSRootNode::clone_root() const {
    auto new_root = std::make_shared<HMGSRootNode>(get_state());
    
    // 复制基本属性
    new_root->visit_count_.store(visit_count_.load(std::memory_order_acquire), std::memory_order_release);
    new_root->total_value_.store(total_value_.load(std::memory_order_acquire), std::memory_order_release);
    new_root->merge_prob_.store(merge_prob_.load(std::memory_order_acquire), std::memory_order_release);
    new_root->merged_into_prob_.store(merged_into_prob_.load(std::memory_order_acquire), std::memory_order_release);
    new_root->continuous_unmerged_.store(continuous_unmerged_.load(std::memory_order_acquire), std::memory_order_release);
    new_root->novelty_score_.store(novelty_score_.load(std::memory_order_acquire), std::memory_order_release);
    new_root->backtrack_count_.store(backtrack_count_.load(std::memory_order_acquire), std::memory_order_release);
    new_root->uncertainty_.store(uncertainty_.load(std::memory_order_acquire), std::memory_order_release);
    new_root->depth_.store(depth_.load(std::memory_order_acquire), std::memory_order_release);
    new_root->expanded_.store(expanded_.load(std::memory_order_acquire), std::memory_order_release);
    new_root->replica_consistency_.store(replica_consistency_.load(std::memory_order_acquire), std::memory_order_release);
    
    // 复制根节点特定属性
    new_root->merge_threshold_.store(merge_threshold_.load(std::memory_order_acquire), std::memory_order_release);
    new_root->branch_update_count_.store(branch_update_count_.load(std::memory_order_acquire), std::memory_order_release);
    new_root->merge_check_count_.store(merge_check_count_.load(std::memory_order_acquire), std::memory_order_release);
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 复制向量数据
    new_root->state_encoding_ = state_encoding_;
    new_root->moves_ = moves_;
    new_root->prior_probs_ = prior_probs_;
    new_root->last_visit_time_ = last_visit_time_;
    
    // 复制分支数据
    new_root->branches_.resize(branches_.size());
    new_root->branch_qualities_.resize(branch_qualities_.size());
    new_root->branch_last_access_.resize(branch_last_access_.size());
    new_root->branch_simulation_counts_.resize(branch_simulation_counts_.size());
    
    for (size_t i = 0; i < branches_.size(); ++i) {
        if (branches_[i]) {
            new_root->branches_[i] = branches_[i]->clone();
        }
        new_root->branch_qualities_[i] = branch_qualities_[i];
        new_root->branch_last_access_[i] = branch_last_access_[i];
        new_root->branch_simulation_counts_[i] = branch_simulation_counts_[i];
    }
    
    return new_root;
}

} // namespace algorithms
} // namespace spacego