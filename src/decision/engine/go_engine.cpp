#define NOMINMAX
#include "go/engine/go_engine.h"
#include "framework/logger.h"
#include <chrono>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <thread>
#include <condition_variable>
#include <deque>
#include <queue>
#include <set>
#include <regex>
#include <cctype>

#ifdef USE_LIBTORCH
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#endif

// =============================================================================
// 文件级匿名命名空间 - 存放后台思考线程变量、SGF辅助函数和pass追踪
// =============================================================================
namespace {
    // ---- 后台思考(Pondering)相关变量 ----
    // 后台思考线程对象，同一时刻只有一个引擎在思考
    std::thread         g_pondering_thread;
    std::mutex          g_pondering_mutex;
    std::atomic<bool>   g_pondering_active{false};
    std::condition_variable g_pondering_cv;
    // 当前正在后台思考的引擎指针（用于安全回调）
    spacego::go::engine::GoEngine* g_pondering_engine = nullptr;

#ifdef USE_LIBTORCH
    // 神经网络推理互斥锁：HMGS-DP多worker并发调用评估回调时串行化前向，
    // 避免同一模型实例并发推理导致GPU内核竞争/状态错乱
    std::mutex g_nn_infer_mutex;
#endif

    // ---- 连续虚着(Pass)追踪 ----
    // 用于is_game_over()检测，记录当前连续pass次数
    // 注意：落子（非pass）会重置此计数器
    int g_consecutive_passes = 0;

    // ---- SGF格式辅助函数 ----
    // 将棋盘内部坐标(row,col)转换为SGF坐标字符串（小写字母）
    // 格式: 列(a-s) + 行(a-s)，例如(0,0) -> "aa", (18,18) -> "ss"
    std::string coord_to_sgf(int x, int y, int /*board_size*/) {
        if (x < 0 || y < 0) return "";  // pass着法用空串表示
        std::string result;
        result += static_cast<char>('a' + y);  // SGF列在前
        result += static_cast<char>('a' + x);  // SGF行在后
        return result;
    }

    // 将SGF坐标字符串解析为棋盘内部坐标(row,col)
    // 输入格式: "ab" -> (1,0), ""或"tt" -> (-1,-1)表示pass
    std::pair<int, int> sgf_to_coord(const std::string& coord) {
        if (coord.empty() || coord.size() < 2) return {-1, -1};
        int y = coord[0] - 'a';
        int x = coord[1] - 'a';
        // 坐标范围校验：0-18对应19路棋盘
        if (x < 0 || x > 18 || y < 0 || y > 18) return {-1, -1};
        return {x, y};
    }

    // 将SGF属性值中的特殊字符进行转义处理
    // SGF规范要求: 右方括号需转义, 反斜杠需转义
    std::string sgf_escape(const std::string& text) {
        std::string result;
        result.reserve(text.size() + 8);
        for (char ch : text) {
            if (ch == '\\') result += "\\\\";
            else if (ch == ']') result += "\\]";
            else result += ch;
        }
        return result;
    }

    // 从SGF token中去除转义字符
    std::string sgf_unescape(const std::string& text) {
        std::string result;
        result.reserve(text.size());
        for (size_t i = 0; i < text.size(); ++i) {
            if (text[i] == '\\' && i + 1 < text.size()) {
                result += text[++i];  // 取转义后的下一个字符
            } else {
                result += text[i];
            }
        }
        return result;
    }
}  // 匿名命名空间

namespace spacego {
namespace go {
namespace engine {

// 构造函数
GoEngine::GoEngine() 
    : board_(nullptr)
    , ca_moe_(nullptr)
    , hmgs_dp_(nullptr)
    , state_(EngineState::IDLE)
    , initialized_(false)
    , stop_requested_(false) {
}

GoEngine::GoEngine(const EngineConfig& config)
    : board_(nullptr)
    , ca_moe_(nullptr)
    , hmgs_dp_(nullptr)
    , config_(config)
    , state_(EngineState::IDLE)
    , initialized_(false)
    , stop_requested_(false) {
}

// 析构函数
GoEngine::~GoEngine() {
    shutdown();
}

// 移动构造函数
GoEngine::GoEngine(GoEngine&& other) noexcept
    : board_(std::move(other.board_))
    , ca_moe_(std::move(other.ca_moe_))
    , hmgs_dp_(std::move(other.hmgs_dp_))
    , config_(std::move(other.config_))
    , state_(other.state_)
    , initialized_(other.initialized_.load())
    , stop_requested_(other.stop_requested_.load())
    , stats_(std::move(other.stats_))
    , progress_callback_(std::move(other.progress_callback_))
    , thinking_callback_(std::move(other.thinking_callback_)) {
    other.initialized_ = false;
}

// 移动赋值运算符
GoEngine& GoEngine::operator=(GoEngine&& other) noexcept {
    if (this != &other) {
        board_ = std::move(other.board_);
        ca_moe_ = std::move(other.ca_moe_);
        hmgs_dp_ = std::move(other.hmgs_dp_);
        config_ = std::move(other.config_);
        state_ = other.state_;
        initialized_ = other.initialized_.load();
        stop_requested_ = other.stop_requested_.load();
        stats_ = std::move(other.stats_);
        progress_callback_ = std::move(other.progress_callback_);
        thinking_callback_ = std::move(other.thinking_callback_);
        other.initialized_ = false;
    }
    return *this;
}

// 初始化引擎
bool GoEngine::initialize() {
    if (initialized_) {
        return true;
    }
    
    try {
        // 初始化棋盘
        board_ = std::make_unique<GoBoardCPU>();
        
        // 初始化HMGS-DP搜索引擎
        hmgs_dp_ = std::make_unique<search::HMGSDP_Optimized>();
        
#ifdef USE_LIBTORCH
        // 神经网络引擎路径：model_path指向LibTorch训练模型(.pt)时加载CA-MOE，
        // 通过set_neural_evaluator把策略/价值注入HMGS-DP（AlphaZero式决策）
        bool is_pt_model = config_.model_path.size() > 3 &&
            config_.model_path.compare(config_.model_path.size() - 3, 3, ".pt") == 0;
        if (is_pt_model && init_neural_engine()) {
            // 搜索迭代数：nn引擎用nn_simulations（默认800），GTP可通过--simulations覆盖
            config_.max_iterations = config_.nn_simulations;
            log_thinking("神经网络引擎初始化成功 (CA-MOE + HMGS-DP)");
        } else {
            // 旧版CAMOE路径（手写权重模型或无模型）
            ca_moe_ = std::make_shared<CAMOE>();
            hmgs_dp_->initialize(config_.model_path);
            if (!config_.model_path.empty() && !is_pt_model) {
                if (!load_model(config_.model_path)) {
                    log_thinking("警告: 模型加载失败，使用默认配置");
                }
            } else if (is_pt_model) {
                log_thinking("警告: 神经网络引擎初始化失败，回退规则评估模式");
                hmgs_dp_->initialize("");
            }
        }
#else
        ca_moe_ = std::make_shared<CAMOE>();
        hmgs_dp_->initialize(config_.model_path);
        if (!config_.model_path.empty()) {
            if (!load_model(config_.model_path)) {
                log_thinking("警告: 模型加载失败，使用默认配置");
            }
        }
#endif
        
        initialized_ = true;
        state_ = EngineState::IDLE;
        log_thinking("引擎初始化成功 (HMGS-DP + CA-MOE)");
        return true;
        
    } catch (const std::exception& e) {
        log_thinking(std::string("引擎初始化失败: ") + e.what());
        return false;
    }
}

bool GoEngine::initialize(const EngineConfig& config) {
    config_ = config;
    return initialize();
}

// 关闭引擎
void GoEngine::shutdown() {
    if (!initialized_) {
        return;
    }
    
    stop_search();
    hmgs_dp_.reset();  // 释放HMGS-DP搜索引擎
    board_.reset();
    ca_moe_.reset();
    initialized_ = false;
    state_ = EngineState::IDLE;
    log_thinking("引擎已关闭");
}

// 清空棋盘
void GoEngine::clear_board() {
    if (!initialized_) {
        throw std::runtime_error("引擎未初始化");
    }
    board_ = std::make_unique<GoBoardCPU>();
    board_->set_current_player(StoneColor::BLACK);
    g_consecutive_passes = 0;  // 重置连续pass计数
    log_thinking("棋盘已清空");
}

// 设置棋盘大小
bool GoEngine::set_board_size(int size) {
    if (size != 9 && size != 13 && size != 19) {
        log_thinking("不支持的棋盘大小: " + std::to_string(size));
        return false;
    }
    
    config_.board_size = size;
    clear_board();
    log_thinking("棋盘大小设置为: " + std::to_string(size));
    return true;
}

// 落子
bool GoEngine::play_move(int x, int y, StoneColor color) {
    if (!initialized_) {
        throw std::runtime_error("引擎未初始化");
    }
    
    if (!is_valid_coordinate(x, y)) {
        log_thinking("无效坐标: (" + std::to_string(x) + ", " + std::to_string(y) + ")");
        return false;
    }
    
    if (!is_legal_move(x, y, color)) {
        log_thinking("非法着法: (" + std::to_string(x) + ", " + std::to_string(y) + ")");
        return false;
    }
    
    auto result = board_->make_move(x, y, color);
    if (!result.success) {
        return false;
    }
    
    // 切换玩家
    board_->set_current_player(color == StoneColor::BLACK ? StoneColor::WHITE : StoneColor::BLACK);
    
    // 落子后重置连续pass计数（非pass着法打断连续虚着）
    g_consecutive_passes = 0;
    
    std::stringstream ss;
    ss << "落子: " << color_to_string(color) << " " << move_to_string(x, y);
    if (!result.captured_positions.empty()) {
        ss << " (提子: " << result.captured_positions.size() << "枚)";
    }
    log_thinking(ss.str());
    
    return true;
}

bool GoEngine::play_move(const std::string& move, StoneColor color) {
    if (move == "pass" || move == "PASS" || move == "Pass") {
        // 虚着：不改变棋盘状态，仅切换玩家并累加连续pass计数
        board_->set_current_player(color == StoneColor::BLACK ? StoneColor::WHITE : StoneColor::BLACK);
        g_consecutive_passes++;  // 累加连续虚着计数（用于is_game_over检测）
        log_thinking("虚着 (Pass) - 连续pass: " + std::to_string(g_consecutive_passes));
        return true;
    }
    
    auto [x, y] = string_to_move(move);
    return play_move(x, y, color);
}

// 悔棋
bool GoEngine::undo_move() {
    if (!initialized_) {
        throw std::runtime_error("引擎未初始化");
    }
    
    bool result = board_->undo_move();
    if (result) {
        g_consecutive_passes = 0;  // 悔棋后重置pass计数，恢复可继续对局状态
        log_thinking("悔棋成功");
    } else {
        log_thinking("悔棋失败: 无历史记录");
    }
    return result;
}

// 获取当前玩家
StoneColor GoEngine::get_current_player() const {
    if (!board_) {
        return StoneColor::BLACK;
    }
    return board_->get_current_player();
}

// 设置当前玩家
void GoEngine::set_current_player(StoneColor color) {
    if (board_) {
        board_->set_current_player(color);
    }
}

// 生成最佳着法
SearchResult GoEngine::genmove(StoneColor color) {
    return genmove_with_time(color, config_.time_limit_ms);
}

SearchResult GoEngine::genmove() {
    return genmove(get_current_player());
}

// 生成最佳着法（带时间限制）
SearchResult GoEngine::genmove_with_time(StoneColor color, int time_ms) {
    if (!initialized_) {
        throw std::runtime_error("引擎未初始化");
    }
    
    state_ = EngineState::THINKING;
    stop_requested_ = false;
    
    log_thinking("开始思考...");
    
    auto result = do_search(config_.max_iterations, time_ms);
    
    // 搜索返回的着法必须能真实落在棋盘上；否则（典型场景：搜索内部棋盘由
    // to_string/from_string 反序列化重建，丢失劫信息 ko_position_，返回了劫点）
    // 直接返回会导致下游GTP对手拒绝该着、双方棋盘失同步、对局崩溃。
    // 历史教训：play_move失败被静默吞掉曾让genmove反复返回非法着N19。
    if (result.move_x >= 0 && result.move_y >= 0) {
        if (!play_move(result.move_x, result.move_y, color)) {
            log_thinking("搜索返回的着法无法落子(" + result.move_string +
                         ")，回退选择真实合法着法");
            result = fallback_legal_move(color);
            if (result.move_x >= 0 && result.move_y >= 0) {
                play_move(result.move_x, result.move_y, color);
            }
        }
    }
    
    state_ = EngineState::IDLE;
    update_stats(result);
    
    std::stringstream ss;
    ss << "最佳着法: " << result.move_string 
       << " 胜率: " << (result.win_probability * 100) << "%"
       << " 访问次数: " << result.visits
       << " 用时: " << result.search_time_ms << "ms";
    log_thinking(ss.str());
    
    return result;
}

// 快速生成着法
SearchResult GoEngine::genmove_quick(StoneColor color) {
    int quick_iterations = config_.max_iterations / 10;
    if (quick_iterations < 100) {
        quick_iterations = 100;
    }
    
    auto saved_iterations = config_.max_iterations;
    config_.max_iterations = quick_iterations;
    
    auto result = genmove_with_time(color, config_.time_limit_ms / 5);
    
    config_.max_iterations = saved_iterations;
    return result;
}

// 回退选择合法着法：在真实棋盘（含劫信息）的合法着中，按NN策略先验选最优。
// 优先级：NN策略 > CA-MOE先验 > 启发式（先手角点）；无合法着时返回pass。
SearchResult GoEngine::fallback_legal_move(StoneColor color) {
    SearchResult result;
    result.move_x = -1;
    result.move_y = -1;
    result.move_string = "pass";
    result.win_probability = 0.5f;

    // 真实棋盘的合法着法（内部已含劫/自杀/占位过滤，比搜索内部棋盘可靠）
    auto legal_moves = get_legal_moves(color);
    if (legal_moves.empty()) {
        log_thinking("回退: 无合法着法，pass");
        return result;
    }

    int best_idx = -1;
    float best_prob = -1.0f;
    float fallback_value = 0.0f;
#ifdef USE_LIBTORCH
    std::vector<float> nn_policy;
    float nn_value = 0.0f;
    if (nn_evaluate(nn_policy, nn_value)) {
        for (const auto& [x, y] : legal_moves) {
            int idx = x * config_.board_size + y;
            if (idx >= 0 && idx < static_cast<int>(nn_policy.size()) &&
                nn_policy[static_cast<size_t>(idx)] > best_prob) {
                best_prob = nn_policy[static_cast<size_t>(idx)];
                best_idx = idx;
            }
        }
        fallback_value = nn_value;
    } else
#endif
    if (ca_moe_) {
        auto output = ca_moe_->forward(*board_);
        for (const auto& [x, y] : legal_moves) {
            int idx = x * config_.board_size + y;
            if (idx >= 0 && idx < static_cast<int>(output.prior_probs.size())) {
                if (output.prior_probs[idx] > best_prob) {
                    best_prob = output.prior_probs[idx];
                    best_idx = idx;
                }
            }
        }
        fallback_value = output.value;
    }

    if (best_idx < 0) {
        // 无策略可用：退回启发式，优先角/边星位（简单兜底）
        const auto& [x, y] = legal_moves[0];
        best_idx = x * config_.board_size + y;
        log_thinking("回退: 无策略先验，采用首个合法着");
    }

    result.move_x = best_idx / config_.board_size;
    result.move_y = best_idx % config_.board_size;
    result.move_string = move_to_string(result.move_x, result.move_y);
    result.win_probability = (fallback_value + 1.0f) / 2.0f;
    result.visits = 1;
    log_thinking("回退选点: " + result.move_string);
    return result;
}

// 获取合法着法列表
std::vector<std::pair<int, int>> GoEngine::get_legal_moves(StoneColor color) const {
    if (!board_) {
        return {};
    }
    return board_->get_legal_moves(color);
}

// 检查着法是否合法
bool GoEngine::is_legal_move(int x, int y, StoneColor color) const {
    if (!board_) {
        return false;
    }
    return board_->is_legal_move(x, y, color);
}

bool GoEngine::is_legal_move(const std::string& move, StoneColor color) const {
    if (move == "pass" || move == "PASS" || move == "Pass") {
        return true;
    }
    auto [x, y] = string_to_move(move);
    return is_legal_move(x, y, color);
}

// 分析当前局面
SearchResult GoEngine::analyze(int max_iterations) {
    if (!initialized_) {
        throw std::runtime_error("引擎未初始化");
    }
    
    state_ = EngineState::ANALYZING;
    stop_requested_ = false;
    
    int iterations = max_iterations > 0 ? max_iterations : config_.max_iterations;
    auto result = do_search(iterations, config_.time_limit_ms);
    
    state_ = EngineState::IDLE;
    return result;
}

// 分析特定着法
float GoEngine::analyze_move(int x, int y) {
    if (!initialized_ || !is_valid_coordinate(x, y)) {
        return 0.0f;
    }
    
    // 评估该着法：神经网络后端优先，否则旧版CA-MOE
#ifdef USE_LIBTORCH
    std::vector<float> nn_policy;
    float nn_value = 0.0f;
    if (nn_evaluate(nn_policy, nn_value)) {
        int idx = x * config_.board_size + y;
        if (idx >= 0 && idx < static_cast<int>(nn_policy.size())) {
            return nn_policy[static_cast<size_t>(idx)];
        }
        return 0.0f;
    }
#endif
    auto output = ca_moe_->forward(*board_);
    
    // 计算该位置的概率
    int idx = x * config_.board_size + y;
    if (idx >= 0 && idx < static_cast<int>(output.prior_probs.size())) {
        return output.prior_probs[idx];
    }
    
    return 0.0f;
}

// 获取着法概率分布
std::vector<std::pair<std::string, float>> GoEngine::get_move_probabilities() {
    if (!initialized_) {
        return {};
    }
    
    // 获取策略分布：神经网络后端优先，否则旧版CA-MOE
    std::vector<float> policy_dist;
#ifdef USE_LIBTORCH
    float nn_value = 0.0f;
    if (!nn_evaluate(policy_dist, nn_value)) policy_dist.clear();
#endif
    if (policy_dist.empty() && ca_moe_) {
        auto output = ca_moe_->forward(*board_);
        policy_dist.assign(output.prior_probs.begin(), output.prior_probs.end());
    }
    std::vector<std::pair<std::string, float>> result;
    
    for (int i = 0; i < config_.board_size * config_.board_size; ++i) {
        int x = i / config_.board_size;
        int y = i % config_.board_size;
        
        if (i < static_cast<int>(policy_dist.size()) &&
            is_legal_move(x, y, get_current_player())) {
            result.emplace_back(move_to_string(x, y), policy_dist[static_cast<size_t>(i)]);
        }
    }
    
    // 按概率排序
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    return result;
}

// 检查游戏是否结束
// 判定规则：连续两次虚着(Pass) -> 对局结束
// 其他结束条件（如一方认输、超时等）由上层协议处理
bool GoEngine::is_game_over() const {
    // 检测连续两次pass：围棋规则规定双方连续pass则对局结束
    // g_consecutive_passes由play_move("pass")累加、play_move(stone)重置
    return g_consecutive_passes >= 2;
}

// 计算领地
// 使用BFS洪泛算法计算黑白双方的领地：
// 1. 对于每个空点，BFS找到连通空区域
// 2. 检查区域仅与黑棋/白棋/双方/无子接触
// 3. 仅与一方接触的空区域归该方领地
// 4. 提子数直接计入领地
std::pair<int, int> GoEngine::count_territory() const {
    if (!board_) {
        return {0, 0};
    }
    
    const int size = config_.board_size;
    const int dx[4] = {1, -1, 0, 0};
    const int dy[4] = {0, 0, 1, -1};
    
    // 访问标记数组：标记已处理过的交叉点
    std::vector<bool> visited(size * size, false);
    
    int black_territory = 0;
    int white_territory = 0;
    
    // 遍历棋盘所有交叉点
    for (int x = 0; x < size; ++x) {
        for (int y = 0; y < size; ++y) {
            int idx = x * size + y;
            
            // 如果该点非空或已访问，跳过
            if (!board_->is_empty(x, y) || visited[idx]) {
                continue;
            }
            
            // BFS洪泛：遍历当前连通空区域的所有点
            std::vector<int> region;
            std::deque<int> queue;
            queue.push_back(idx);
            visited[idx] = true;
            
            while (!queue.empty()) {
                int cur = queue.front();
                queue.pop_front();
                region.push_back(cur);
                
                int cx = cur / size;
                int cy = cur % size;
                
                // 检查四个相邻方向
                for (int dir = 0; dir < 4; ++dir) {
                    int nx = cx + dx[dir];
                    int ny = cy + dy[dir];
                    int nidx = nx * size + ny;
                    
                    // 边界检查
                    if (nx < 0 || nx >= size || ny < 0 || ny >= size) continue;
                    if (visited[nidx]) continue;
                    
                    // 只遍历空点（遇到棋子则记录邻接关系，不加入队列）
                    if (board_->is_empty(nx, ny)) {
                        visited[nidx] = true;
                        queue.push_back(nidx);
                    }
                }
            }
            
            // 统计该空区域与哪些颜色的棋子邻接
            bool touches_black = false;
            bool touches_white = false;
            
            for (int pt : region) {
                int rx = pt / size;
                int ry = pt % size;
                
                for (int dir = 0; dir < 4; ++dir) {
                    int nx = rx + dx[dir];
                    int ny = ry + dy[dir];
                    if (nx < 0 || nx >= size || ny < 0 || ny >= size) continue;
                    
                    if (board_->is_black(nx, ny)) {
                        touches_black = true;
                    } else if (board_->is_white(nx, ny)) {
                        touches_white = true;
                    }
                }
            }
            
            // 领地归属判定：
            // - 仅接触黑棋 -> 黑方领地
            // - 仅接触白棋 -> 白方领地
            // - 同时接触双方或无接触 -> 中立区域，不计入任何一方
            if (touches_black && !touches_white) {
                black_territory += static_cast<int>(region.size());
            } else if (touches_white && !touches_black) {
                white_territory += static_cast<int>(region.size());
            }
            // 双方都接触或都不接触 -> 中立，不计入
        }
    }
    
    // 加上提子数（提子直接算入得分）
    black_territory += board_->get_capture_count(StoneColor::BLACK);
    white_territory += board_->get_capture_count(StoneColor::WHITE);
    
    return {black_territory, white_territory};
}

// 估算胜率
float GoEngine::estimate_win_rate(StoneColor color) const {
    if (!initialized_) {
        return 0.5f;
    }
    
    float value = 0.0f;
#ifdef USE_LIBTORCH
    // 神经网络后端优先（模型价值为黑棋视角）
    std::vector<float> nn_policy;
    if (nn_evaluate(nn_policy, value)) {
        if (color == StoneColor::WHITE) {
            value = -value;
        }
        return (value + 1.0f) / 2.0f;
    }
#endif
    if (!ca_moe_) {
        return 0.5f;
    }
    
    auto output = ca_moe_->forward(*board_);
    value = output.value;
    
    // value是黑棋胜率，如果是白棋需要反转
    if (color == StoneColor::WHITE) {
        value = -value;
    }
    
    // 转换为0-1范围
    return (value + 1.0f) / 2.0f;
}

// 设置配置
void GoEngine::set_config(const EngineConfig& config) {
    config_ = config;
}

// 设置时间限制
void GoEngine::set_time_limit(int time_ms) {
    config_.time_limit_ms = time_ms;
}

// 设置最大搜索次数
void GoEngine::set_max_iterations(int iterations) {
    config_.max_iterations = iterations;
}

// 设置探索权重
void GoEngine::set_exploration_weight(float weight) {
    config_.exploration_weight = weight;
}

// 获取统计信息
EngineStats GoEngine::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

// 重置统计信息
void GoEngine::reset_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = EngineStats();
}

// 加载模型
bool GoEngine::load_model(const std::string& path) {
    if (!ca_moe_) {
        return false;
    }
    
    bool result = ca_moe_->load_model(path);
    if (result) {
        config_.model_path = path;
        log_thinking("模型加载成功: " + path);
    } else {
        log_thinking("模型加载失败: " + path);
    }
    
    return result;
}

#ifdef USE_LIBTORCH
// 初始化神经网络引擎后端：加载LibTorch训练的CA-MOE模型（elo_gated等产出的.pt），
// 并把策略/价值评估器绑定到HMGS-DP搜索（AlphaZero式：expand用策略先验、simulate用价值）。
bool GoEngine::init_neural_engine() {
    try {
        // 创建CA-MOE专家网络并加载checkpoint（8专家19路；hidden_dim尝试256，失败回退128）
        nn_model_ = std::make_shared<neural::CA_MOEExpertSystem>(8, config_.board_size, 0.1f, 256);
        try {
            (*nn_model_)->load(config_.model_path);
        } catch (...) {
            nn_model_ = std::make_shared<neural::CA_MOEExpertSystem>(8, config_.board_size, 0.1f, 128);
            (*nn_model_)->load(config_.model_path);
        }
        log_thinking("CA-MOE模型加载成功: " + config_.model_path);

        // 设备迁移与推理模式（eval下BN/训练层稳定）
        torch::Device dev = (config_.use_cuda && torch::cuda::is_available())
            ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
        (*nn_model_)->to(dev);
        (*nn_model_)->set_eval_mode();

        // 初始化搜索器（空模型路径：评估完全由神经网络回调提供）
        hmgs_dp_->initialize("");
        search::HMGSDP_Optimized::ParallelSearchConfig ps;
        ps.num_workers = 1;
        ps.batch_size = 128;
        ps.use_batch_inference = false;
        hmgs_dp_->set_parallel_config(ps);

        // 策略先验回调：模型单次前向输出362维分布（搜索expand_node使用）
        auto policy_cb = [this](const GoBoardCPU& board, int current_player) {
            std::vector<float> out(362, 1.0f / 362.0f);
            try {
                std::vector<float> raw(17 * 19 * 19, 0.0f);
                board.encode_to_tensor(raw.data(), 17);
                auto state = torch::from_blob(raw.data(), {1, 17, 19, 19}, torch::kFloat32).clone();
                if (config_.use_cuda && torch::cuda::is_available()) state = state.to(torch::kCUDA);
                torch::NoGradGuard ng;
                std::lock_guard<std::mutex> lock(g_nn_infer_mutex);
                auto result = (*nn_model_)->forward(state);
                auto policy = result.final_policy.squeeze(0).to(torch::kCPU).contiguous();
                const float* p = policy.data_ptr<float>();
                float sum = 0.0f;
                for (int i = 0; i < 362; ++i) {
                    out[static_cast<size_t>(i)] = std::max(p[i], 0.0f);
                    sum += out[static_cast<size_t>(i)];
                }
                if (sum > 1e-8f) for (auto& v : out) v /= sum;
            } catch (...) {}
            return out;
        };

        // 价值回调：当前玩家视角[-1,1]（搜索simulate使用）
        auto value_cb = [this](const GoBoardCPU& board, int current_player) {
            try {
                std::vector<float> raw(17 * 19 * 19, 0.0f);
                board.encode_to_tensor(raw.data(), 17);
                auto state = torch::from_blob(raw.data(), {1, 17, 19, 19}, torch::kFloat32).clone();
                if (config_.use_cuda && torch::cuda::is_available()) state = state.to(torch::kCUDA);
                torch::NoGradGuard ng;
                std::lock_guard<std::mutex> lock(g_nn_infer_mutex);
                auto result = (*nn_model_)->forward(state);
                float v = result.final_value.to(torch::kCPU).item<float>();
                if (current_player == -1) v = -v;  // 模型价值为黑棋视角，-1=白需反转
                return std::clamp(v, -1.0f, 1.0f);
            } catch (...) { return 0.0f; }
        };

        // 批量评估回调：一次前向评估一批叶子（策略+价值），提升GPU利用率
        auto batch_cb = [this](const std::vector<GoBoardCPU>& boards,
                               const std::vector<int>& players) {
            std::vector<std::pair<std::vector<float>, float>> results;
            results.reserve(boards.size());
            if (boards.empty()) return results;
            try {
                std::vector<float> raw(boards.size() * 17 * 19 * 19, 0.0f);
                for (size_t i = 0; i < boards.size(); ++i) {
                    boards[i].encode_to_tensor(raw.data() + i * 17 * 19 * 19, 17);
                }
                auto state = torch::from_blob(
                    raw.data(), {static_cast<long>(boards.size()), 17, 19, 19},
                    torch::kFloat32).clone();
                if (config_.use_cuda && torch::cuda::is_available()) state = state.to(torch::kCUDA);
                torch::NoGradGuard ng;
                std::lock_guard<std::mutex> lock(g_nn_infer_mutex);
                auto result = (*nn_model_)->forward(state);
                auto pb = result.final_policy.to(torch::kCPU).contiguous();
                auto vb = result.final_value.to(torch::kCPU).contiguous();
                const float* pp = pb.data_ptr<float>();
                const float* vp = vb.data_ptr<float>();
                for (size_t i = 0; i < boards.size(); ++i) {
                    std::vector<float> out(362, 0.0f);
                    float sum = 0.0f;
                    for (int j = 0; j < 362; ++j) {
                        out[static_cast<size_t>(j)] = std::max(pp[i * 362 + j], 0.0f);
                        sum += out[static_cast<size_t>(j)];
                    }
                    if (sum > 1e-8f) for (auto& v : out) v /= sum;
                    float val = vp[i];
                    if (players[i] == -1) val = -val;
                    results.emplace_back(std::move(out), std::clamp(val, -1.0f, 1.0f));
                }
            } catch (...) {
                for (size_t i = 0; i < boards.size(); ++i) {
                    results.emplace_back(std::vector<float>(362, 1.0f / 362.0f), 0.0f);
                }
            }
            return results;
        };

        hmgs_dp_->set_batch_evaluator(batch_cb);
        hmgs_dp_->set_neural_evaluator(policy_cb, value_cb);
        // 用stderr输出初始化状态（GTP协议stdout只允许命令响应，stderr供人查看）
        std::cerr << "[NN引擎] CA-MOE模型加载成功: " << config_.model_path
                  << " | 搜索模拟=" << config_.nn_simulations << std::endl;
        return true;
    } catch (const std::exception& e) {
        log_thinking(std::string("神经网络引擎初始化异常: ") + e.what());
        std::cerr << "[NN引擎] 加载失败: " << e.what() << std::endl;
        return false;
    }
}

// 神经网络单次前向评估（成员棋盘board_，策略362+价值），供分析/胜率接口使用
bool GoEngine::nn_evaluate(std::vector<float>& policy_out, float& value_out) const {
    if (!nn_model_ || !board_) return false;
    try {
        std::vector<float> raw(17 * 19 * 19, 0.0f);
        board_->encode_to_tensor(raw.data(), 17);
        auto state = torch::from_blob(raw.data(), {1, 17, 19, 19}, torch::kFloat32).clone();
        if (config_.use_cuda && torch::cuda::is_available()) state = state.to(torch::kCUDA);
        torch::NoGradGuard ng;
        std::lock_guard<std::mutex> lock(g_nn_infer_mutex);
        auto result = (*nn_model_)->forward(state);
        policy_out.assign(362, 0.0f);
        auto policy = result.final_policy.squeeze(0).to(torch::kCPU).contiguous();
        const float* p = policy.data_ptr<float>();
        for (int i = 0; i < 362; ++i) {
            policy_out[static_cast<size_t>(i)] = std::max(p[i], 0.0f);
        }
        value_out = result.final_value.to(torch::kCPU).item<float>();
        return true;
    } catch (...) { return false; }
}
#endif

// 检查模型是否已加载
bool GoEngine::is_model_loaded() const {
    return !config_.model_path.empty();
}

// 棋盘状态字符串
std::string GoEngine::get_board_string() const {
    if (!board_) {
        return "";
    }
    return board_->to_string();
}

// SGF格式输出
// 生成完整的SGF（Smart Game Format）格式对局记录：
// - 包含棋局头部信息（GM, FF, SZ, DT, PB, PW, KM, RU）
// - 使用AB/AW标签输出当前棋盘所有棋子（作为局面快照）
// - 包含提子数、当前手数、当前玩家等注释信息
// 注意：由于棋盘内部历史不可直接访问，采用棋子快照方式生成完整合法的SGF
std::string GoEngine::to_sgf() const {
    std::stringstream ss;
    const int size = config_.board_size;
    
    // ---- SGF头部信息 ----
    // GM[1] = 围棋, FF[4] = SGF版本4
    ss << "(;GM[1]FF[4]SZ[" << size << "]\n";
    // 日期（当前系统日期）
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_now;
#ifdef _WIN32
    localtime_s(&tm_now, &time_t_now);
#else
    localtime_r(&time_t_now, &tm_now);
#endif
    ss << "DT[" << (tm_now.tm_year + 1900) << "-"
       << (tm_now.tm_mon + 1 < 10 ? "0" : "") << (tm_now.tm_mon + 1) << "-"
       << (tm_now.tm_mday < 10 ? "0" : "") << tm_now.tm_mday << "]\n";
    // 对局者信息
    ss << "PB[Black]PW[White]\n";
    // 贴目（默认中国规则贴7.5目）
    ss << "KM[7.5]\n";
    // 规则
    ss << "RU[Chinese]\n";
    // 引擎信息
    ss << "AN[SpaceGo]\n";
    
    // ---- 棋盘局面快照 ----
    // 收集所有黑棋位置，输出AB（Add Black）标签
    auto black_stones = board_->get_all_stones(StoneColor::BLACK);
    if (!black_stones.empty()) {
        ss << "AB";
        for (const auto& [x, y] : black_stones) {
            ss << "[" << coord_to_sgf(x, y, size) << "]";
        }
        ss << "\n";
    }
    
    // 收集所有白棋位置，输出AW（Add White）标签
    auto white_stones = board_->get_all_stones(StoneColor::WHITE);
    if (!white_stones.empty()) {
        ss << "AW";
        for (const auto& [x, y] : white_stones) {
            ss << "[" << coord_to_sgf(x, y, size) << "]";
        }
        ss << "\n";
    }
    
    // ---- 附加信息（注释形式） ----
    // 当前玩家和手数
    ss << "PL[" << (board_->get_current_player() == StoneColor::BLACK ? "B" : "W") << "]\n";
    
    // 提子统计
    int black_caps = board_->get_capture_count(StoneColor::BLACK);
    int white_caps = board_->get_capture_count(StoneColor::WHITE);
    if (black_caps > 0 || white_caps > 0) {
        ss << "C[Captures: Black=" << black_caps << " White=" << white_caps << "]\n";
    }
    
    // 手数与引擎信息
    ss << "C[MoveCount=" << board_->get_move_count()
       << " Engine=SpaceGo]\n";
    
    ss << ")\n";
    return ss.str();
}

// 从SGF加载
// 解析SGF格式并还原对局状态：
// - 解析SZ（棋盘大小）、AB/AW（初始棋子布局）、B/W（着法序列）
// - 依次执行所有着法以还原完整对局
// - 支持多分支SGF（仅取主分支）
// - 支持pass着法（空坐标或tt）
bool GoEngine::load_from_sgf(const std::string& sgf) {
    if (!initialized_) {
        throw std::runtime_error("引擎未初始化");
    }
    
    // ---- 第一步：清空当前棋盘 ----
    clear_board();
    
    // 查找SGF内容范围（跳过外层可能的多余字符）
    size_t start_pos = sgf.find('(');
    if (start_pos == std::string::npos) {
        log_thinking("SGF解析失败: 未找到'('起始标记");
        return false;
    }
    
    size_t end_pos = sgf.find_last_of(')');
    if (end_pos == std::string::npos || end_pos <= start_pos) {
        log_thinking("SGF解析失败: 未找到')'结束标记");
        return false;
    }
    
    std::string content = sgf.substr(start_pos + 1, end_pos - start_pos - 1);
    
    // ---- 第二步：解析SGF属性 ----
    // SGF格式: ;PROP[value][value]...PROP[value]...
    // 顶层节点以';'开头，子节点嵌套在括号中
    int board_size = 19;  // 默认19路
    std::vector<std::pair<std::string, StoneColor>> setup_stones;  // 初始布局棋子
    std::vector<std::pair<std::string, StoneColor>> moves;          // 着法序列
    
    size_t pos = 0;
    bool in_header = true;  // 是否还在头节点（第一个节点）
    
    while (pos < content.size()) {
        // 跳过空白字符
        while (pos < content.size() && 
               (content[pos] == ' ' || content[pos] == '\n' || 
                content[pos] == '\r' || content[pos] == '\t')) {
            ++pos;
        }
        
        if (pos >= content.size()) break;
        
        // 遇到';' -> 新节点开始
        if (content[pos] == ';') {
            ++pos;
            in_header = false;
            continue;
        }
        
        // 遇到'(' -> 进入子分支（跳过，仅处理主分支）
        if (content[pos] == '(') {
            // 查找匹配的')'并跳过整个子树
            int depth = 1;
            ++pos;
            while (pos < content.size() && depth > 0) {
                if (content[pos] == '(') ++depth;
                else if (content[pos] == ')') --depth;
                ++pos;
            }
            continue;
        }
        
        // 遇到')' -> 节点结束
        if (content[pos] == ')') {
            ++pos;
            continue;
        }
        
        // ---- 解析属性名 ----
        // 属性名由连续大写字母组成
        std::string prop_name;
        while (pos < content.size() && std::isupper(static_cast<unsigned char>(content[pos]))) {
            prop_name += content[pos];
            ++pos;
        }
        
        if (prop_name.empty()) {
            ++pos;
            continue;
        }
        
        // ---- 解析属性值 ----
        // 每个属性可以有多个值，每值用[...]包围
        std::vector<std::string> prop_values;
        while (pos < content.size() && content[pos] == '[') {
            ++pos;  // 跳过'['
            std::string value;
            while (pos < content.size() && content[pos] != ']') {
                // 处理转义字符
                if (content[pos] == '\\' && pos + 1 < content.size()) {
                    ++pos;  // 跳过'\\'
                    value += content[pos];
                } else {
                    value += content[pos];
                }
                ++pos;
            }
            if (pos < content.size()) ++pos;  // 跳过']'
            prop_values.push_back(value);
        }
        
        // ---- 根据属性名处理数据 ----
        if (prop_name == "SZ" && !prop_values.empty()) {
            // 棋盘大小
            try {
                board_size = std::stoi(prop_values[0]);
            } catch (...) {
                board_size = 19;
            }
        } else if (prop_name == "AB") {
            // 初始黑棋布局
            for (const auto& val : prop_values) {
                setup_stones.emplace_back(val, StoneColor::BLACK);
            }
        } else if (prop_name == "AW") {
            // 初始白棋布局
            for (const auto& val : prop_values) {
                setup_stones.emplace_back(val, StoneColor::WHITE);
            }
        } else if (prop_name == "B" && !prop_values.empty()) {
            // 黑棋着法
            moves.emplace_back(prop_values[0], StoneColor::BLACK);
        } else if (prop_name == "W" && !prop_values.empty()) {
            // 白棋着法
            moves.emplace_back(prop_values[0], StoneColor::WHITE);
        }
        // 忽略其他属性（C, KM, RE, PB, PW等）
    }
    
    // ---- 第三步：设置棋盘大小 ----
    if (board_size != config_.board_size) {
        if (!set_board_size(board_size)) {
            log_thinking("SGF加载失败: 不支持的棋盘大小 " + std::to_string(board_size));
            return false;
        }
    }
    
    // ---- 第四步：布置初始棋子（AB/AW） ----
    for (const auto& [coord_str, color] : setup_stones) {
        auto [x, y] = sgf_to_coord(coord_str);
        if (x >= 0 && y >= 0 && x < board_size && y < board_size) {
            board_->set_stone(x, y, color);
        }
    }
    
    // ---- 第五步：按顺序执行所有着法 ----
    // 预留一个临时变量用于记录最后一手玩家
    StoneColor last_move_color = StoneColor::BLACK;
    int moves_played = 0;
    
    for (const auto& [coord_str, color] : moves) {
        last_move_color = color;
        
        // 检测pass着法：SGF中用空字符串或"tt"表示
        if (coord_str.empty() || coord_str == "tt") {
            // pass -> 切换玩家即可
            board_->set_current_player(color == StoneColor::BLACK ? 
                                       StoneColor::WHITE : StoneColor::BLACK);
            g_consecutive_passes++;
            moves_played++;
            continue;
        }
        
        auto [x, y] = sgf_to_coord(coord_str);
        
        // 坐标有效性检查
        if (x < 0 || y < 0 || x >= board_size || y >= board_size) {
            log_thinking("SGF解析警告: 无效坐标 '" + coord_str + 
                         "' (着法#" + std::to_string(moves_played + 1) + ")，已跳过");
            continue;
        }
        
        // 合法性检查（跳过非法着法，容忍不规范的SGF）
        if (!board_->is_legal_move(x, y, color)) {
            log_thinking("SGF解析警告: 非法着法 " + coord_str + 
                         " (着法#" + std::to_string(moves_played + 1) + ")，已跳过");
            continue;
        }
        
        // 执行落子
        auto result = board_->make_move(x, y, color);
        if (result.success) {
            // 落子后切换玩家并重置pass计数
            board_->set_current_player(color == StoneColor::BLACK ? 
                                       StoneColor::WHITE : StoneColor::BLACK);
            g_consecutive_passes = 0;
            moves_played++;
        }
    }
    
    // ---- 第六步：设置正确的当前玩家 ----
    // 根据最后一手棋确定当前玩家
    if (moves_played > 0) {
        // 最后一手之后已自动切换，当前玩家正确
        // 若setup_stones非空但无moves，需要确认玩家
        board_->set_move_count(board_->get_move_count() + moves_played);
    }
    
    log_thinking("SGF加载成功: 棋盘" + std::to_string(board_size) + 
                 "路, 初始棋子" + std::to_string(setup_stones.size()) + 
                 "枚, 着法" + std::to_string(moves_played) + "手");
    
    return true;
}

// 着法坐标转换
std::string GoEngine::move_to_string(int x, int y) {
    if (x < 0 || y < 0) return "pass";
    int board_size = 19;
    std::string result;
    // 列标签：A-H, J-T（跳过I，符合GTP协议标准）
    if (y >= 8) {
        result += static_cast<char>('A' + y + 1);  // 跳过I
    } else {
        result += static_cast<char>('A' + y);
    }
    
    // 行号：从1到board_size（从底部开始编号）
    result += std::to_string(board_size - x);
    return result;
}

std::pair<int, int> GoEngine::string_to_move(const std::string& move) {
    if (move.empty() || move == "pass") return {-1, -1};
    int board_size = 19;
    
    char col = std::toupper(move[0]);
    if (col < 'A' || col > 'T') {  // GTP协议最大列T（19路棋盘）
        return {-1, -1};
    }
    
    int y = col - 'A';
    if (col > 'I') {
        y--;  // 跳过字母I
    }
    
    try {
        int row = std::stoi(move.substr(1));
        
        // 验证行号范围
        if (row < 1 || row > board_size) {
            return {-1, -1};
        }
        
        int x = board_size - row;
        
        // 验证坐标范围
        if (x < 0 || x >= board_size || y < 0 || y >= board_size) {
            return {-1, -1};
        }
        
        return {x, y};
    } catch (...) {
        return {-1, -1};
    }
}

// 颜色转换
StoneColor GoEngine::string_to_color(const std::string& color_str) {
    if (color_str == "black" || color_str == "Black" || color_str == "BLACK" || color_str == "B") {
        return StoneColor::BLACK;
    }
    return StoneColor::WHITE;
}

std::string GoEngine::color_to_string(StoneColor color) {
    return color == StoneColor::BLACK ? "Black" : "White";
}

// 设置进度回调
void GoEngine::set_progress_callback(ProgressCallback callback) {
    progress_callback_ = std::move(callback);
}

// 设置思考回调
void GoEngine::set_thinking_callback(ThinkingCallback callback) {
    thinking_callback_ = std::move(callback);
}

// 后台思考
// 启动后台思考线程：在对手思考期间，我方引擎持续分析局面
// 线程循环调用analyze()进行轻量级搜索，直到stop_pondering()被调用
void GoEngine::start_pondering() {
    if (state_ != EngineState::IDLE || !initialized_) {
        return;  // 仅在空闲且已初始化时启动后台思考
    }
    
    // 如果已有后台思考线程在运行，先停止它
    if (g_pondering_active.load()) {
        stop_pondering();
    }
    
    // 设置后台思考状态
    state_ = EngineState::PONDERING;
    g_pondering_active.store(true);
    g_pondering_engine = this;
    stop_requested_ = false;
    
    // 启动后台思考线程
    // 线程函数：周期性调用analyze()进行局面分析
    // 每轮使用较少的迭代次数（500次）以保持响应性
    g_pondering_thread = std::thread([this]() {
        log_thinking("后台思考线程已启动");
        
        while (g_pondering_active.load()) {
            // 在每轮分析前重置为PONDERING状态
            // analyze()内部会将状态改为ANALYZING->IDLE，这里统一恢复
            state_ = EngineState::PONDERING;
            
            try {
                // 使用analyze进行轻量级局面分析
                // 每次500次迭代，保持低延迟以便快速响应stop信号
                this->analyze(500);
            } catch (const std::exception& e) {
                // 分析过程中的异常不应导致线程崩溃
                log_thinking(std::string("后台思考异常: ") + e.what());
                
                // 短暂休眠后重试，避免快速循环消耗CPU
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            
            // 检查stop信号（在analyze调用之间检查，确保及时响应）
            if (!g_pondering_active.load()) {
                break;
            }
            
            // 短暂让步，避免CPU空转
            // 在analyze之间留出微小间隔，确保对stop信号响应及时
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        
        // 线程退出前将状态恢复为IDLE
        state_ = EngineState::IDLE;
        log_thinking("后台思考线程已退出");
    });
}

void GoEngine::stop_pondering() {
    if (state_ != EngineState::PONDERING) {
        return;  // 不在后台思考状态，无需停止
    }
    
    // 第一步：设置停止标志，通知线程退出
    g_pondering_active.store(false);
    
    // 第二步：中断可能正在进行的搜索
    // stop_search()设置stop_requested_标志，do_search内部会检查并提前退出
    stop_search();
    
    // 第三步：等待后台思考线程安全退出
    if (g_pondering_thread.joinable()) {
        g_pondering_thread.join();
    }
    
    // 第四步：清理状态
    g_pondering_engine = nullptr;
    state_ = EngineState::IDLE;
    stop_requested_ = false;
    
    log_thinking("后台思考已停止");
}

bool GoEngine::is_pondering() const {
    return state_ == EngineState::PONDERING;
}

// 停止当前搜索
void GoEngine::stop_search() {
    stop_requested_ = true;
}

// 获取引擎名称
std::string GoEngine::get_engine_name() {
    return "SpaceGo";
}

// 获取引擎版本
std::string GoEngine::get_engine_version() {
    return "1.0.0";
}

// 获取引擎信息
std::string GoEngine::get_engine_info() {
    std::stringstream ss;
    ss << get_engine_name() << " " << get_engine_version() << "\n";
    ss << "HMGS-DP搜索算法 + CA-MOE元决策系统\n";
#if SPACEGO_CUDA_ENABLED
    ss << "CUDA加速: 启用\n";
#else
    ss << "CUDA加速: 禁用 (CPU模式)\n";
#endif
    return ss.str();
}

// 执行搜索 - 使用简化版HMGS-DP算法进行完整迭代搜索
SearchResult GoEngine::do_search(int max_iterations, int time_limit_ms) {
    SearchResult result;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 设置引擎状态
    state_ = EngineState::THINKING;
    stop_requested_ = false;
    
    try {
        // 获取当前棋盘状态字符串
        std::string board_state = board_->to_string();
        
        log_thinking("开始HMGS-DP搜索 (最大迭代: " + std::to_string(max_iterations) + 
                     ", 时间限制: " + std::to_string(time_limit_ms) + "ms)");
        
        // 使用HMGS-DP进行迭代搜索（非MCTS，使用规则评估+CA-MOE）
        // HMGS-DP内部会执行以下流程：
        // 1. 初始化根节点
        // 2. 循环执行max_iterations次：
        //    a. 选择节点 (UCT或简化选择策略)
        //    b. 扩展叶子节点 (生成合法子节点)
        //    c. 评估节点 (使用simulate()规则评估)
        //    d. 反向传播价值
        // 3. 返回访问次数最多的子节点的着法
        
        auto [best_x, best_y] = hmgs_dp_->search(
            board_state,
            max_iterations,
            config_.exploration_weight
        );
        
        // 计算搜索耗时
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        // 填充搜索结果
        if (best_x >= 0 && best_y >= 0) {
            result.move_x = best_x;
            result.move_y = best_y;
            result.move_string = move_to_string(best_x, best_y);
            result.visits = max_iterations;  // HMGS-DP完成的迭代次数
            
            // 获取该着法的胜率估计（神经网络优先，否则旧版CA-MOE）
            result.win_probability = 0.5f;
#ifdef USE_LIBTORCH
            std::vector<float> nn_policy;
            float nn_value = 0.0f;
            if (nn_evaluate(nn_policy, nn_value)) {
                int idx = best_x * config_.board_size + best_y;
                if (idx >= 0 && idx < static_cast<int>(nn_policy.size()) &&
                    nn_policy[static_cast<size_t>(idx)] > 0.0f) {
                    result.win_probability = nn_policy[static_cast<size_t>(idx)];
                } else {
                    result.win_probability = (nn_value + 1.0f) / 2.0f;
                }
            } else
#endif
            if (ca_moe_) {
                auto ca_moe_output = ca_moe_->forward(*board_);
                int idx = best_x * config_.board_size + best_y;
                if (idx >= 0 && idx < static_cast<int>(ca_moe_output.prior_probs.size())) {
                    // 将先验概率转换为胜率估计（简化处理）
                    result.win_probability = ca_moe_output.prior_probs[idx];
                } else {
                    // 如果无法从CA-MOE获取，使用value字段
                    result.win_probability = (ca_moe_output.value + 1.0f) / 2.0f;
                }
            }
            
            log_thinking("HMGS-DP搜索完成: " + result.move_string + 
                        " (" + std::to_string(result.move_x) + "," + 
                        std::to_string(result.move_y) + ")" +
                        " 胜率: " + std::to_string(result.win_probability * 100).substr(0, 4) + "%" +
                        " 耗时: " + std::to_string(elapsed_ms).substr(0, 5) + "ms");
        } else {
            // 无合法着法，返回pass
            result.move_x = -1;
            result.move_y = -1;
            result.move_string = "pass";
            result.win_probability = 0.5f;
            result.visits = max_iterations;
            
            log_thinking("HMGS-DP搜索完成: pass (无合法着法)");
        }
        
        result.search_time_ms = elapsed_ms;
        
    } catch (const std::exception& e) {
        log_thinking(std::string("HMGS-DP搜索异常: ") + e.what());
        
        // 降级：如果HMGS-DP失败，回退到单次评估（神经网络优先，否则旧版CA-MOE）
        log_thinking("降级到单次评估模式");
        
        auto legal_moves = get_legal_moves(get_current_player());
        int best_idx = -1;
        float best_prob = -1.0f;
        float fallback_value = 0.0f;
#ifdef USE_LIBTORCH
        std::vector<float> nn_policy;
        float nn_value = 0.0f;
        if (nn_evaluate(nn_policy, nn_value)) {
            for (const auto& [x, y] : legal_moves) {
                int idx = x * config_.board_size + y;
                if (idx >= 0 && idx < static_cast<int>(nn_policy.size()) &&
                    nn_policy[static_cast<size_t>(idx)] > best_prob) {
                    best_prob = nn_policy[static_cast<size_t>(idx)];
                    best_idx = idx;
                }
            }
            fallback_value = nn_value;
        } else
#endif
        if (ca_moe_) {
            auto output = ca_moe_->forward(*board_);
            for (const auto& [x, y] : legal_moves) {
                int idx = x * config_.board_size + y;
                if (idx >= 0 && idx < static_cast<int>(output.prior_probs.size())) {
                    if (output.prior_probs[idx] > best_prob) {
                        best_prob = output.prior_probs[idx];
                        best_idx = idx;
                    }
                }
            }
            fallback_value = output.value;
        }
        
        if (best_idx >= 0) {
            result.move_x = best_idx / config_.board_size;
            result.move_y = best_idx % config_.board_size;
            result.move_string = move_to_string(result.move_x, result.move_y);
            result.win_probability = (fallback_value + 1.0f) / 2.0f;
            result.visits = 1;  // 单次评估
        } else {
            result.move_x = -1;
            result.move_y = -1;
            result.move_string = "pass";
            result.win_probability = 0.5f;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        result.search_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    
    // 恢复引擎状态
    state_ = EngineState::IDLE;
    
    return result;
}

// 更新统计信息
void GoEngine::update_stats(const SearchResult& result) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.total_moves++;
    stats_.total_searches++;
    stats_.total_time_ms += result.search_time_ms;
    stats_.average_time_per_move_ms = stats_.total_time_ms / stats_.total_moves;
}

// 记录思考信息
void GoEngine::log_thinking(const std::string& message) {
    if (config_.use_logging && thinking_callback_) {
        thinking_callback_(message);
    }
    
    auto& logger = framework::Logger::instance();
    logger.info("[Engine] " + message);
}

// 检查着法有效性
bool GoEngine::check_move_valid(int x, int y, StoneColor color) const {
    return is_valid_coordinate(x, y) && is_legal_move(x, y, color);
}

// 坐标边界检查
bool GoEngine::is_valid_coordinate(int x, int y) const {
    return x >= 0 && x < config_.board_size && y >= 0 && y < config_.board_size;
}

// 工厂函数
std::unique_ptr<GoEngine> create_engine() {
    return std::make_unique<GoEngine>();
}

std::unique_ptr<GoEngine> create_engine(const EngineConfig& config) {
    return std::make_unique<GoEngine>(config);
}

}  // namespace engine
}  // namespace go
}  // namespace spacego
