#pragma once

#include "network/algorithms/hmgs_dp.h"
#include "go/board/go_board_cuda.h"
#include <memory>
#include <vector>
#include <queue>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <random>
#include <string>
#include <functional>
#include <unordered_map>

namespace spacego {
namespace go {
namespace search {

struct HMGS_Node {
    int move_x = -1;
    int move_y = -1;
    double value = 0.0;
    std::atomic<int> visits{0};
    HMGS_Node* parent = nullptr;
    std::vector<HMGS_Node*> children;
    std::string board_state;
    GoBoardCPU board;  // 节点对应棋盘状态（直传棋盘对象，避免每次模拟字符串序列化开销）
    uint64_t zobrist_key = 0;  // 换位表key（棋子布局hash+行棋方），用于状态合并
    float prior_prob = 0.0f;
    int merge_count = 0;
    float merge_probability = 0.0f;
    std::atomic<int> virtual_loss{0};
    bool expanded = false;

    HMGS_Node() = default;
    explicit HMGS_Node(HMGS_Node* p) : parent(p) {}

    double get_average_value() const {
        int v = visits.load();
        return v > 0 ? value / v : 0.0;
    }

    HMGS_Node* add_child(int x, int y, float prior = 1.0f) {
        HMGS_Node* child = new HMGS_Node(this);
        child->move_x = x;
        child->move_y = y;
        child->prior_prob = prior;
        children.push_back(child);
        return child;
    }

    // 添加对已有节点的引用（换位表状态合并：相同局面共享同一节点，形成DAG）
    void add_child_ref(HMGS_Node* child) {
        children.push_back(child);
    }

    static void delete_tree(HMGS_Node* node) {
        if (!node) return;
        for (auto* c : node->children) delete_tree(c);
        delete node;
    }
};

class LockFreeQueue {
public:
    LockFreeQueue();
    ~LockFreeQueue();
    void enqueue(HMGS_Node* node);
    HMGS_Node* dequeue();
    bool is_empty() const;
    size_t size_approx() const;

private:
    struct Node {
        HMGS_Node* data;
        std::atomic<Node*> next;
        Node(HMGS_Node* d) : data(d), next(nullptr) {}
    };
    std::atomic<Node*> head_;
    std::atomic<Node*> tail_;
    std::atomic<size_t> size_;
};

class WorkStealingQueue {
public:
    WorkStealingQueue(size_t capacity = 1024);
    ~WorkStealingQueue();
    void push(HMGS_Node* node);
    HMGS_Node* pop();
    HMGS_Node* steal();
    bool is_empty() const;
    size_t size() const;

private:
    std::vector<HMGS_Node*> buffer_;
    std::atomic<size_t> top_;
    std::atomic<size_t> bottom_;
    size_t capacity_;
    mutable std::mutex steal_mutex_;
};

class HMGSDP_Optimized {
public:
    HMGSDP_Optimized();
    virtual ~HMGSDP_Optimized();

    virtual void initialize(const std::string& model_path = "");

    virtual std::pair<int, int> search(const std::string& board_state,
                               int max_iterations = 10000,
                               double exploration_weight = 1.0);

    struct ParallelSearchConfig {
        size_t num_workers = 8;
        size_t batch_size = 64;
        bool use_work_stealing = true;
        bool use_batch_inference = true;
        int max_simulations_per_second = 0;
    };

    void set_parallel_config(const ParallelSearchConfig& config);
    ParallelSearchConfig get_parallel_config() const;

    struct SearchStatistics {
        size_t total_simulations = 0;
        size_t batch_count = 0;
        double average_batch_size = 0.0;
        double work_steal_count = 0.0;
        double gpu_utilization = 0.0;
        double simulations_per_second = 0.0;
        double search_time_ms = 0.0;
        std::vector<double> worker_utilizations;
    };

    SearchStatistics get_search_statistics() const;
    void reset_search_statistics();

    // ==================== 神经网络评估注入 ====================
    // 类型别名：策略先验评估器（返回362维策略分布）
    // 接收棋盘对象引用（避免字符串序列化，提升搜索评估吞吐）
    using PolicyPriorFn = std::function<std::vector<float>(
        const GoBoardCPU& board, int current_player)>;
    // 类型别名：价值评估器（返回局面价值，范围[-1,1]）
    using ValueFn = std::function<float(const GoBoardCPU& board, int current_player)>;

    // 注入神经网络策略/价值评估器（AlphaZero式搜索的核心）
    // 设置后：expand_node使用模型策略作为先验，simulate使用模型价值
    void set_neural_evaluator(PolicyPriorFn policy_fn, ValueFn value_fn);

    // ==================== 批量评估注入 ====================
    // 批量评估器：一次前向评估多个叶子节点（策略分布+价值）
    // 输入：一组棋盘状态与其当前行棋方，输出：每个状态的(策略分布[362], 价值)
    // 相比逐节点单样本推理，批量推理可显著提升GPU利用率（自对弈数据生成的关键提速）
    using BatchEvaluatorFn = std::function<std::vector<std::pair<std::vector<float>, float>>(
        const std::vector<GoBoardCPU>& boards, const std::vector<int>& players)>;

    // 设置批量评估器后，search()主循环按batch收集叶子节点统一推理
    void set_batch_evaluator(BatchEvaluatorFn eval_fn);

    // 获取根节点子节点的访问次数分布（归一化为362维策略）
    // 用于生成自对弈训练目标（policy_target = 搜索访问次数分布）
    std::vector<float> get_root_policy(int board_size = 19) const;

    // 获取根节点价值估计（根节点平均价值）
    double get_root_value() const;

protected:
    virtual double simulate(HMGS_Node* node, const std::string& state);
    virtual HMGS_Node* expand_node(HMGS_Node* node, const std::string& state);
    virtual void backpropagate(HMGS_Node* node, double reward);

    // 使用给定策略先验展开节点（批量推理路径：先验来自批量评估结果）
    void expand_node_with_prior(HMGS_Node* node, const std::vector<float>& prior);

private:
    void parallel_search_worker(size_t worker_id);

    HMGS_Node* select_node_for_expansion(HMGS_Node* root, size_t worker_id);
    float calculate_move_prior(int x, int y, const GoBoardCPU& board, StoneColor player);

    // 神经网络评估器（可选）
    PolicyPriorFn policy_fn_;
    ValueFn value_fn_;
    BatchEvaluatorFn batch_eval_fn_;  // 批量评估器（优先级最高，设置后搜索走批量路径）

    // 换位表：zobrist key → 节点（HMGS-DP状态合并核心，相同局面共享节点避免重复评估）
    // 换位表持有全部节点所有权（DAG下多parent共享，删除时遍历表，避免double free）
    std::unordered_map<uint64_t, HMGS_Node*> tt_;

    ParallelSearchConfig parallel_config_;

    std::vector<std::unique_ptr<WorkStealingQueue>> worker_queues_;
    std::vector<std::thread> worker_threads_;

    std::atomic<bool> search_running_{false};
    std::atomic<size_t> current_iteration_{0};
    std::atomic<size_t> total_simulations_{0};
    std::atomic<size_t> batch_count_{0};
    std::atomic<size_t> work_steal_count_{0};

    mutable std::mutex stats_mutex_;
    SearchStatistics current_stats_;

    std::mt19937 rng_;
    std::uniform_int_distribution<size_t> worker_dist_;

    HMGS_Node* root_node_ = nullptr;
    std::string root_state_;
    double exploration_weight_ = 1.414;
    int max_iterations_ = 10000;

    std::atomic<size_t> active_workers_{0};
    std::atomic<int> ready_workers_{0};
    std::condition_variable start_cv_;
    std::mutex start_mutex_;
};

class TranspositionTableOptimized {
public:
    struct Entry {
        uint64_t hash = 0;
        float value = 0.0f;
        uint64_t visits = 0;
        int depth = 0;
        enum class Bound { EXACT, LOWER, UPPER } bound = Bound::EXACT;
    };

    TranspositionTableOptimized(size_t size_mb = 1024);
    ~TranspositionTableOptimized();

    void store(uint64_t hash, float value, int depth, Entry::Bound bound);
    bool probe(uint64_t hash, Entry& entry) const;
    void clear();
    size_t size() const { return table_size_; }
    size_t usage() const;

private:
    size_t table_size_;
    std::vector<Entry> table_;
    std::atomic<size_t> entry_count_{0};
    size_t index(uint64_t hash) const;
};

}
}
}
