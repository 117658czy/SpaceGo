#include "go/search/hmgs_dp_optimized.h"
#include "go/search/spatial_topology_accelerator.h"
#ifdef USE_CUDA
#include "go/search/hmgs_dp_cuda.h"
#endif
#include <algorithm>
#include <cmath>
#include <numeric>
#include <chrono>
#include <unordered_set>
#include "framework/logger.h"

namespace spacego {
namespace go {
namespace search {

namespace {
// 换位表key：棋子布局hash + 行棋方（相同布局但轮换方视为不同局面，避免误合并）
uint64_t compute_tt_key(const GoBoardCPU& board) {
    uint64_t h = static_cast<uint64_t>(board.hash());
    return h ^ (board.get_current_player() == StoneColor::WHITE ? 0x9E3779B97F4A7C15ULL : 0ULL);
}

// 动态计算pass先验概率
// 固定0.01导致对局几乎从不选pass→全部400步截断→胜负标签错误。
// 策略：基础先验0.05，随剩余空点减少而显著提升（接近终局时pass应成为主要着法）。
// 空点统计基于节点棋盘（361点），与对局步数无关，可同时用于单节点/批量展开路径。
// 重要修复：19路对局在250步左右收束时空点通常仍有130~170（模型一直在填低价值点），
// 阈值必须覆盖该区间，否则搜索分布中pass占比仍极低→训练目标(policy_target)的pass分量不足→
// 模型学不到pass→对局全靠硬编码的250步兜底强制收束（步数恒定253，数据单一，棋力停滞）。
// 现改为空点<220即开始线性抬升（已进入官子/终局阶段），空点=0时pass先验≈0.60，
// 使搜索在终局阶段将大量访问分配给pass，训练目标含充分pass信号，模型学会自然终局。
float compute_pass_prior(const GoBoardCPU& board, const std::vector<float>& prior) {
    float base = 0.05f;
    if (prior.size() >= 362) {
        base = std::max(base, prior[361]);  // 模型输出pass先验作下界
    }
    // 统计剩余空点数
    int empty = 0;
    for (int i = 0; i < 361; ++i) {
        auto [x, y] = GoBoardCPU::index_to_xy(i);
        if (board.is_empty(x, y)) empty++;
    }
    // 空点<=220时线性抬升pass先验，空点=0时达到0.60
    if (empty < 220) {
        float boost = (220 - empty) / 220.0f * 0.55f;
        base = std::min(0.6f, base + boost);
    }
    return base;
}
}

LockFreeQueue::LockFreeQueue()
    : head_(nullptr), tail_(nullptr), size_(0) {
    Node* dummy = new Node(nullptr);
    head_.store(dummy);
    tail_.store(dummy);
}

LockFreeQueue::~LockFreeQueue() {
    while (Node* n = head_.load()) {
        head_.store(n->next.load());
        delete n;
    }
}

void LockFreeQueue::enqueue(HMGS_Node* data) {
    Node* nn = new Node(data);
    Node* ot;
    while (true) {
        ot = tail_.load();
        Node* nxt = ot->next.load();
        if (ot == tail_.load()) {
            if (!nxt) {
                if (ot->next.compare_exchange_weak(nxt, nn)) break;
            } else {
                tail_.compare_exchange_weak(ot, nxt);
            }
        }
    }
    tail_.compare_exchange_weak(ot, nn);
    size_++;
}

HMGS_Node* LockFreeQueue::dequeue() {
    Node* oh;
    while (true) {
        oh = head_.load();
        Node* ot = tail_.load();
        Node* nxt = oh->next.load();
        if (oh == head_.load()) {
            if (oh == ot) {
                if (!nxt) return nullptr;
                tail_.compare_exchange_weak(ot, nxt);
            } else {
                HMGS_Node* d = nxt->data;
                if (head_.compare_exchange_weak(oh, nxt)) { delete oh; size_--; return d; }
            }
        }
    }
}

bool LockFreeQueue::is_empty() const { return size_.load() == 0; }
size_t LockFreeQueue::size_approx() const { return size_.load(); }

WorkStealingQueue::WorkStealingQueue(size_t cap)
    : capacity_(cap), top_(0), bottom_(0) { buffer_.resize(cap); }
WorkStealingQueue::~WorkStealingQueue() {}

void WorkStealingQueue::push(HMGS_Node* node) {
    size_t b = bottom_.load(std::memory_order_relaxed);
    buffer_[b % capacity_] = node;
    bottom_.store(b + 1, std::memory_order_release);
}

HMGS_Node* WorkStealingQueue::pop() {
    size_t b = bottom_.load(std::memory_order_relaxed);
    b--;
    bottom_.store(b, std::memory_order_relaxed);
    size_t t = top_.load(std::memory_order_acquire);
    if (t <= b) {
        HMGS_Node* node = buffer_[b % capacity_];
        if (t != b) return node;
        if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_release, std::memory_order_relaxed)) {
            bottom_.store(b + 1, std::memory_order_relaxed);
            return nullptr;
        }
        bottom_.store(b + 1, std::memory_order_relaxed);
        return node;
    } else {
        bottom_.store(b + 1, std::memory_order_relaxed);
        return nullptr;
    }
}

HMGS_Node* WorkStealingQueue::steal() {
    std::lock_guard<std::mutex> lock(steal_mutex_);
    size_t t = top_.load(std::memory_order_acquire);
    size_t b = bottom_.load(std::memory_order_acquire);
    if (t < b) {
        HMGS_Node* node = buffer_[t % capacity_];
        if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_release, std::memory_order_relaxed))
            return nullptr;
        return node;
    }
    return nullptr;
}

bool WorkStealingQueue::is_empty() const {
    return top_.load(std::memory_order_relaxed) >= bottom_.load(std::memory_order_relaxed);
}
size_t WorkStealingQueue::size() const {
    return bottom_.load(std::memory_order_relaxed) - top_.load(std::memory_order_relaxed);
}

TranspositionTableOptimized::TranspositionTableOptimized(size_t size_mb) : entry_count_(0) {
    table_size_ = (size_mb * 1024 * 1024) / sizeof(Entry);
    table_.resize(table_size_);
}
TranspositionTableOptimized::~TranspositionTableOptimized() {}

size_t TranspositionTableOptimized::index(uint64_t hash) const { return hash % table_size_; }

void TranspositionTableOptimized::store(uint64_t hash, float value, int depth, Entry::Bound bound) {
    size_t idx = index(hash);
    Entry e; e.hash = hash; e.value = value; e.depth = depth; e.bound = bound; e.visits = 1;
    if (table_[idx].hash == 0 || depth >= table_[idx].depth) {
        table_[idx] = e;
        if (table_[idx].hash == 0) entry_count_++;
    }
}

bool TranspositionTableOptimized::probe(uint64_t hash, Entry& entry) const {
    size_t idx = index(hash);
    if (table_[idx].hash == hash) { entry = table_[idx]; return true; }
    return false;
}

void TranspositionTableOptimized::clear() {
    for (auto& e : table_) e = Entry();
    entry_count_ = 0;
}
size_t TranspositionTableOptimized::usage() const { return entry_count_.load(); }

HMGSDP_Optimized::HMGSDP_Optimized()
    : rng_(std::random_device{}()), worker_dist_(0, 7) {}

HMGSDP_Optimized::~HMGSDP_Optimized() {
    search_running_ = false;
    start_cv_.notify_all();
    for (auto& t : worker_threads_) if (t.joinable()) t.join();
    // 遍历换位表删除：状态合并后树为DAG（多parent共享节点），递归delete_tree会double free
    for (auto& [key, node] : tt_) delete node;
    tt_.clear();
    root_node_ = nullptr;
}

void HMGSDP_Optimized::initialize(const std::string&) {
    // 初始化搜索配置，无需加载模型（当前使用规则评估）
}

void HMGSDP_Optimized::set_parallel_config(const ParallelSearchConfig& config) {
    parallel_config_ = config;
    worker_dist_ = std::uniform_int_distribution<size_t>(0, config.num_workers > 0 ? config.num_workers - 1 : 0);
}

void HMGSDP_Optimized::set_neural_evaluator(PolicyPriorFn policy_fn, ValueFn value_fn) {
    policy_fn_ = std::move(policy_fn);
    value_fn_ = std::move(value_fn);
}

void HMGSDP_Optimized::set_batch_evaluator(BatchEvaluatorFn eval_fn) {
    batch_eval_fn_ = std::move(eval_fn);
}

std::vector<float> HMGSDP_Optimized::get_root_policy(int board_size) const {
    // 从根节点子节点的访问次数生成归一化策略分布（AlphaZero式训练目标）
    std::vector<float> policy(static_cast<size_t>(board_size * board_size + 1), 0.0f);
    if (!root_node_) return policy;

    int total_visits = 0;
    for (auto* child : root_node_->children) {
        if (!child) continue;
        // move_x == -1 表示pass节点，映射到索引361
        if (child->move_x < 0 || child->move_y < 0) {
            policy[static_cast<size_t>(board_size * board_size)] +=
                static_cast<float>(child->visits);
        } else {
            int idx = GoBoardCPU::xy_to_index(child->move_x, child->move_y);
            if (idx >= 0 && idx < board_size * board_size) {
                policy[static_cast<size_t>(idx)] = static_cast<float>(child->visits);
            }
        }
        total_visits += child->visits;
    }

    // 归一化为概率分布（pass索引361的访问次数也被统计）
    if (total_visits > 0) {
        for (auto& p : policy) {
            p /= static_cast<float>(total_visits);
        }
    }
    return policy;
}

double HMGSDP_Optimized::get_root_value() const {
    if (!root_node_) return 0.0;
    int v = root_node_->visits.load();
    return v > 0 ? root_node_->value / v : 0.0;
}

HMGSDP_Optimized::ParallelSearchConfig HMGSDP_Optimized::get_parallel_config() const {
    return parallel_config_;
}

std::pair<int, int> HMGSDP_Optimized::search(
    const std::string& board_state, int max_iterations, double exploration_weight) {

    auto t0 = std::chrono::steady_clock::now();
    root_state_ = board_state;
    max_iterations_ = max_iterations;
    exploration_weight_ = exploration_weight;

    // 清理上一棵搜索树（遍历换位表删除，DAG安全）；重置统计
    for (auto& [key, node] : tt_) delete node;
    tt_.clear();
    root_node_ = new HMGS_Node();
    root_node_->board_state = board_state;
    // 根节点解析棋盘对象（仅一次），搜索树内部节点直接持有GoBoardCPU，避免字符串往返
    root_node_->board = GoBoardCPU::from_string(board_state);
    root_node_->zobrist_key = compute_tt_key(root_node_->board);
    tt_[root_node_->zobrist_key] = root_node_;

    current_iteration_ = 0;
    total_simulations_ = 0;
    batch_count_ = 0;
    work_steal_count_ = 0;
    search_running_ = true;

    // 迭代搜索：按batch收集叶子节点→批量推理→展开+回传
    // 批量推理复用GPU并行能力，显著降低自对弈数据生成中NN评估的墙钟时间
    const size_t batch_size = std::max<size_t>(1, parallel_config_.batch_size);
    for (int i = 0; i < max_iterations && search_running_; ) {
        // 1. 选择一批叶子节点（沿UCT路径下潜到未展开节点）
        // 状态合并形成DAG后，同一节点可能被多条路径选中，用set去重避免重复评估
        std::vector<HMGS_Node*> leaves;
        std::unordered_set<HMGS_Node*> seen;
        leaves.reserve(batch_size);
        for (size_t j = 0; j < batch_size && i < max_iterations && search_running_; ++j, ++i) {
            HMGS_Node* node = select_node_for_expansion(root_node_, 0);
            if (!node) break;
            if (!seen.insert(node).second) continue;  // 换位合并导致的重复节点，跳过

            // 终局规则结算检测（打破"从不pass→学不会pass→从不pass"循环）：
            // 历史教训：叶子评估永远用神经网络猜测pass后价值→搜索给pass访问少→
            //   policy_target中pass分量低→模型学不会自然终局→对局恒253步靠250步兜底。
            // 修复：连续两个pass（当前节点是pass且父节点也是pass）= 真终局，
            //   直接按中国规则面积计点结算真实胜负，给pass路径注入可靠终局信号。
            // 注意：不能按"空点少"判定——19路终局时空点含双方领地仍较多(130+)，
            //   空点阈值会使规则结算几乎永不触发，只有连续两pass才是规则上的终局。
            bool rule_terminal = false;
            double rule_value = 0.0;
            if (node->move_x < 0 && node->parent && node->parent->move_x < 0) {
                rule_terminal = true;
                // 中国规则面积计点判胜负（黑贴7.5目）
                auto [black_final, white_final] = node->board.compute_area_score(7.5f);
                float outcome = (black_final > white_final) ? 1.0f :
                                ((white_final > black_final) ? -1.0f : 0.0f);
                // 从当前玩家视角转换（pass后已切换到对方行棋）
                StoneColor cur = node->board.get_current_player();
                rule_value = (cur == StoneColor::BLACK) ? outcome : -outcome;
                // 终局节点无需展开，直接回传真实胜负并跳过批量评估
                backpropagate(node, rule_value);
                total_simulations_++;
                current_iteration_++;
                continue;
            }

            leaves.push_back(node);
        }
        if (leaves.empty()) break;

        // 2. 批量评估叶子节点（一次前向同时输出策略先验与价值）
        std::vector<std::pair<std::vector<float>, float>> evals;
        if (batch_eval_fn_) {
            std::vector<GoBoardCPU> boards;
            std::vector<int> players;
            boards.reserve(leaves.size());
            players.reserve(leaves.size());
            for (auto* lf : leaves) {
                boards.push_back(lf->board);
                players.push_back(static_cast<int>(lf->board.get_current_player()));
            }
            evals = batch_eval_fn_(boards, players);
            batch_count_++;
        }

        // 3. 逐叶子展开（用批量先验）并回传价值
        for (size_t k = 0; k < leaves.size(); ++k) {
            HMGS_Node* node = leaves[k];
            double value = 0.0;
            if (k < evals.size() && !evals[k].first.empty()) {
                // 批量路径：一次评估同时提供先验与价值，无需再调用单节点回调
                expand_node_with_prior(node, evals[k].first);
                value = static_cast<double>(evals[k].second);
            } else {
                // 回退：无批量评估器或评估失败时使用单节点路径
                std::string node_state = node->board_state.empty() ? root_state_ : node->board_state;
                expand_node(node, node_state);
                value = simulate(node, node_state);
            }
            backpropagate(node, value);
            total_simulations_++;
            current_iteration_++;
        }
    }

    // 选择最佳着法
    int best_x = -1, best_y = -1;
    int best_visits = -1;
    for (auto* child : root_node_->children) {
        if (child && child->visits > best_visits) {
            best_visits = child->visits;
            best_x = child->move_x;
            best_y = child->move_y;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::lock_guard<std::mutex> lock(stats_mutex_);
    current_stats_.total_simulations = total_simulations_.load();
    current_stats_.search_time_ms = ms;
    if (ms > 0) current_stats_.simulations_per_second = total_simulations_.load() * 1000.0 / ms;

    return {best_x, best_y};
}

HMGSDP_Optimized::SearchStatistics HMGSDP_Optimized::get_search_statistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return current_stats_;
}

void HMGSDP_Optimized::reset_search_statistics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    current_stats_ = SearchStatistics();
}

void HMGSDP_Optimized::parallel_search_worker(size_t) {
    // 简化版：单线程搜索不需要worker
}

HMGS_Node* HMGSDP_Optimized::select_node_for_expansion(HMGS_Node* root, size_t) {
    if (!root) return nullptr;
    HMGS_Node* node = root;
    // 防御：状态合并形成DAG后，围棋循环局面（打劫等）可能导致换位表复用节点
    // 成为当前路径祖先从而成环。环存在时此循环将无限下潜挂死进程，
    // 故设深度上限（正常对局下潜深度远小于该值，不影响搜索质量）
    constexpr int kMaxDescentDepth = 1024;
    int depth = 0;
    while (!node->children.empty() && depth++ < kMaxDescentDepth) {
        HMGS_Node* best = nullptr;
        double best_score = -1e18;
        for (auto* child : node->children) {
            if (!child) continue;
            int v = child->visits;
            double exp = child->value / std::max(v, 1) +
                exploration_weight_ * std::sqrt(std::log(std::max(node->visits.load(), 1) + 1) / (v + 1));
            if (exp > best_score) { best_score = exp; best = child; }
        }
        if (!best) break;
        node = best;
    }
    return node;
}

double HMGSDP_Optimized::simulate(HMGS_Node* node, const std::string& state) {
    // 优先使用神经网络价值评估（AlphaZero式：模型评估代替随机rollout）
    if (value_fn_) {
        try {
            // 使用节点持有棋盘对象（避免from_string解析开销）
            GoBoardCPU board = node->board;
            StoneColor player = board.get_current_player();
            float v = value_fn_(board, static_cast<int>(player));
            // 价值从当前玩家视角返回，与UCT中价值一致
            return static_cast<double>(v);
        } catch (...) {
            return 0.0;
        }
    }

    // 回退到启发式评估（material + territory）
    try {
        GoBoardCPU board = node->board;

        StoneColor player = board.get_current_player();
        auto black = board.get_all_stones(StoneColor::BLACK);
        auto white = board.get_all_stones(StoneColor::WHITE);

        float material = static_cast<float>(black.size()) - static_cast<float>(white.size());

        float territory = 0.0f;
        constexpr int BSZ = 19;
        constexpr float CW = 1.5f, EP = 0.8f;

        for (const auto& [bx, by] : black) {
            float dc = std::sqrt(std::pow(bx - BSZ/2.0f, 2) + std::pow(by - BSZ/2.0f, 2));
            float w = CW / (1.0f + dc * 0.1f);
            if (bx <= 2 || bx >= 16 || by <= 2 || by >= 16) w *= EP;
            territory += w;
        }
        for (const auto& [wx, wy] : white) {
            float dc = std::sqrt(std::pow(wx - BSZ/2.0f, 2) + std::pow(wy - BSZ/2.0f, 2));
            float w = CW / (1.0f + dc * 0.1f);
            if (wx <= 2 || wx >= 16 || wy <= 2 || wy >= 16) w *= EP;
            territory -= w;
        }

        double score = std::clamp((material * 10.0 + territory) / 100.0, -1.0, 1.0);
        return player == StoneColor::WHITE ? -score : score;
    } catch (...) {
        return 0.0;
    }
}

HMGS_Node* HMGSDP_Optimized::expand_node(HMGS_Node* node, const std::string& state) {
    if (!node || node->expanded) return node;
    try {
        // 使用节点持有棋盘对象（根节点在search()中解析，子节点在扩展时直接派生）
        GoBoardCPU board = node->board;
        StoneColor player = board.get_current_player();

        // 神经网络策略先验（可选）：一次推理获得全部合法着法的先验概率
        std::unordered_map<int, float> prior_map;
        if (policy_fn_) {
            try {
                auto prior = policy_fn_(board, static_cast<int>(player));
                // prior为362维分布（索引0-360棋盘位置，361为pass）
                if (prior.size() >= 361) {
                    for (int i = 0; i < 361; ++i) {
                        if (prior[static_cast<size_t>(i)] > 0.0f) {
                            auto [x, y] = GoBoardCPU::index_to_xy(i);
                            prior_map[i] = prior[static_cast<size_t>(i)];
                        }
                    }
                }
            } catch (...) {
                // 策略推理失败时回退到启发式先验
            }
        }

        auto moves = board.get_legal_moves(player);

        // 按先验排序取top-K：控制分支因子，避免为每个合法着法都make_move（开局约350个，
        // 低先验着法被搜索选中的概率极低，截断后可显著降低每次模拟的CPU开销）
        // 参考KataGo/AlphaZero的做法：策略先验低的着法不展开
        std::vector<std::pair<float, std::pair<int, int>>> candidates;
        candidates.reserve(moves.size());
        for (const auto& [x, y] : moves) {
            // 优先使用神经网络策略先验，否则用启发式
            float prior = 0.01f;
            int idx = GoBoardCPU::xy_to_index(x, y);
            auto it = prior_map.find(idx);
            if (it != prior_map.end()) {
                prior = std::max(0.01f, it->second);
            } else {
                prior = calculate_move_prior(x, y, board, player);
            }
            candidates.emplace_back(prior, std::make_pair(x, y));
        }
        // 降序排列（高先验优先），取前kMaxBranch个
        std::sort(candidates.begin(), candidates.end(),
                  [](const std::pair<float, std::pair<int, int>>& a,
                     const std::pair<float, std::pair<int, int>>& b) {
                      return a.first > b.first;
                  });
        const size_t kMaxBranch = 64;
        size_t branch = std::min(kMaxBranch, candidates.size());
        for (size_t i = 0; i < branch; ++i) {
            const auto& [x, y] = candidates[i].second;
            HMGS_Node* child = node->add_child(x, y, candidates[i].first);
            // 直接派生子棋盘并落子（快速落子：候选着法已通过合法性/劫检查，无需重复校验）
            GoBoardCPU cb = board;
            cb.make_move_fast(x, y, player);
            child->board = cb;
        }

        // 添加pass节点（move_x=-1, move_y=-1表示pass）
        // pass保证搜索树可覆盖终局路径，避免对局因无pass选项而无限延续
        {
            std::vector<float> prior;
            if (policy_fn_) {
                try {
                    prior = policy_fn_(board, static_cast<int>(player));
                } catch (...) {
                    // 策略推理失败时使用空先验（compute_pass_prior内部兜底）
                }
            }
            float pass_prior = compute_pass_prior(board, prior);
            HMGS_Node* pass_child = node->add_child(-1, -1, pass_prior);
            // pass后轮到对方，棋盘布局不变但行棋方切换（与批量路径expand_node_with_prior保持一致，
            // 否则后续展开会用错误的行棋方评估局面）
            GoBoardCPU pb = board;
            pb.set_current_player(opponent_color(player));
            pass_child->board = pb;
        }
        node->expanded = true;
    } catch (...) {}
    return node;
}

// 使用给定策略先验展开节点（批量推理路径专用）
// 先验来自批量评估结果（一次前向同时获得策略分布与价值），
// 与expand_node的单节点推理路径共享top-K截断逻辑
void HMGSDP_Optimized::expand_node_with_prior(HMGS_Node* node, const std::vector<float>& prior) {
    if (!node || node->expanded) return;
    try {
        // 使用节点持有棋盘对象（根节点在search()中解析，子节点在扩展时直接派生）
        GoBoardCPU board = node->board;
        StoneColor player = board.get_current_player();

        // 从批量先验构建prior_map（索引0-360棋盘位置）
        std::unordered_map<int, float> prior_map;
        if (prior.size() >= 361) {
            for (int i = 0; i < 361; ++i) {
                if (prior[static_cast<size_t>(i)] > 0.0f) {
                    prior_map[i] = prior[static_cast<size_t>(i)];
                }
            }
        }

        auto moves = board.get_legal_moves(player);

        // 按先验排序取top-K：控制分支因子，避免为每个合法着法都make_move
        std::vector<std::pair<float, std::pair<int, int>>> candidates;
        candidates.reserve(moves.size());
        for (const auto& [x, y] : moves) {
            // 优先使用神经网络策略先验，否则用启发式
            float p = 0.01f;
            int idx = GoBoardCPU::xy_to_index(x, y);
            auto it = prior_map.find(idx);
            if (it != prior_map.end()) {
                p = std::max(0.01f, it->second);
            } else {
                p = calculate_move_prior(x, y, board, player);
            }
            candidates.emplace_back(p, std::make_pair(x, y));
        }
        // 降序排列（高先验优先），取前kMaxBranch个
        std::sort(candidates.begin(), candidates.end(),
                  [](const std::pair<float, std::pair<int, int>>& a,
                     const std::pair<float, std::pair<int, int>>& b) {
                      return a.first > b.first;
                  });
        const size_t kMaxBranch = 64;
        size_t branch = std::min(kMaxBranch, candidates.size());
        for (size_t i = 0; i < branch; ++i) {
            const auto& [x, y] = candidates[i].second;
            // 直接派生子棋盘并落子（快速落子：候选着法已通过合法性/劫检查，无需重复校验）
            GoBoardCPU cb = board;
            cb.make_move_fast(x, y, player);
            // 换位表状态合并：相同局面复用已有节点（共享统计与子树，避免重复评估）
            uint64_t key = compute_tt_key(cb);
            auto tt_it = tt_.find(key);
            if (tt_it != tt_.end()) {
                if (tt_it->second != node) {
                    node->add_child_ref(tt_it->second);
                }
            } else {
                HMGS_Node* child = node->add_child(x, y, candidates[i].first);
                child->board = cb;
                child->zobrist_key = key;
                tt_[key] = child;
            }
        }

        // 添加pass节点（move_x=-1, move_y=-1表示pass）
        float pass_prior = compute_pass_prior(board, prior);
        // pass后轮到对方，棋盘布局不变但行棋方切换（否则后续展开会用错误的行棋方）
        GoBoardCPU pb = board;
        pb.set_current_player(opponent_color(player));
        uint64_t pass_key = compute_tt_key(pb);
        auto pass_tt_it = tt_.find(pass_key);
        if (pass_tt_it != tt_.end()) {
            if (pass_tt_it->second != node) {
                node->add_child_ref(pass_tt_it->second);
            }
        } else {
            HMGS_Node* pass_child = node->add_child(-1, -1, pass_prior);
            pass_child->board = pb;
            pass_child->zobrist_key = pass_key;
            tt_[pass_key] = pass_child;
        }

        node->expanded = true;
    } catch (...) {}
}

void HMGSDP_Optimized::backpropagate(HMGS_Node* node, double reward) {
    // 防御：状态合并成环时parent链可能无限回环（打劫/循环局面），
    // 设步数上限强制退出，避免沿环无限回传挂死进程（正常对局深度远小于上限）
    constexpr int kMaxBackpropDepth = 1024;
    int depth = 0;
    while (node && depth++ < kMaxBackpropDepth) {
        node->value += reward;
        node->visits++;
        node = node->parent;
        // 每向上一层视角翻转（叶子价值为叶子当前玩家视角，父节点视角相反）
        // 保证所有祖先节点的value累加都统一到"各自当前玩家视角"，避免黑/白价值混叠
        reward = -reward;
    }
}

float HMGSDP_Optimized::calculate_move_prior(int x, int y, const GoBoardCPU& board, StoneColor) {
    constexpr int BSZ = 19;
    float prior = 1.0f;
    float cx = (BSZ - 1) / 2.0f, cy = (BSZ - 1) / 2.0f;
    float dc = std::sqrt(std::pow(x - cx, 2) + std::pow(y - cy, 2));
    prior *= (1.5f - dc / (std::sqrt(2.0f) * BSZ / 2.0f) * 0.5f);

    const int dx[] = {1,-1,0,0}, dy[] = {0,0,1,-1};
    int adj = 0;
    for (int d = 0; d < 4; ++d) {
        int nx = x+dx[d], ny = y+dy[d];
        if (nx >= 0 && nx < BSZ && ny >= 0 && ny < BSZ && !board.is_empty(nx, ny)) adj++;
    }
    if (adj > 0) prior *= (1.0f + adj * 0.3f);

    constexpr int sp[][2] = {{3,3},{3,9},{3,15},{9,3},{9,9},{9,15},{15,3},{15,9},{15,15}};
    for (auto& [sx,sy] : sp) if (x==sx && y==sy) { prior *= 1.3f; break; }

    return std::clamp(prior, 0.01f, 10.0f);
}

#ifdef USE_CUDA
// ==================== HMGSDP_CUDA 拓扑加速器集成 ====================

void HMGSDP_CUDA::set_topology_accelerator(
    std::shared_ptr<SpatialTopologyAccelerator> accelerator) {
    topology_accelerator_ = accelerator;
    if (accelerator) {
        LOG_INFO("HMGS-DP CUDA已绑定空间拓扑加速器");
    }
}

std::shared_ptr<SpatialTopologyAccelerator> 
HMGSDP_CUDA::get_topology_accelerator() const {
    return topology_accelerator_;
}

#endif // USE_CUDA

}  // namespace search
}  // namespace go
}  // namespace spacego
