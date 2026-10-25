#pragma once

#include "config/platform_config.h"
#include "go/board/go_board_cuda.h"
#include "go/search/ca_moe.h"
#include "go/search/hmgs_dp_optimized.h"
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <mutex>
#include <atomic>

#ifdef USE_LIBTORCH
#include "go/neural/ca_moe_expert.h"
#endif

namespace spacego {
namespace go {
namespace engine {

// 引擎配置结构
struct EngineConfig {
    int board_size = config::BOARD_SIZE;
    int max_iterations = config::DEFAULT_MAX_ITERATIONS;
    float exploration_weight = config::DEFAULT_EXPLORATION_WEIGHT;
    int time_limit_ms = config::DEFAULT_TIME_LIMIT_MS;
    bool use_cuda = SPACEGO_CUDA_ENABLED;
    bool use_logging = true;
    std::string model_path;
    // 神经网络引擎搜索模拟数（--model 指定LibTorch训练模型时生效，
    // 对应HMGS-DP的迭代次数；越大棋力越强但单步更慢）
    int nn_simulations = 800;
    
    EngineConfig() = default;
};

// 搜索结果结构
struct SearchResult {
    int move_x;
    int move_y;
    float win_probability;
    int visits;
    double search_time_ms;
    std::vector<std::pair<int, int>> principal_variation;
    std::string move_string;
    
    SearchResult() : move_x(-1), move_y(-1), win_probability(0.0f),
                     visits(0), search_time_ms(0.0) {}
};

// 引擎状态枚举
enum class EngineState {
    IDLE,
    THINKING,
    PONDERING,
    ANALYZING
};

// 引擎统计信息
struct EngineStats {
    size_t total_moves;
    size_t total_searches;
    double total_time_ms;
    double average_time_per_move_ms;
    size_t cache_hits;
    size_t cache_misses;
    
    EngineStats() : total_moves(0), total_searches(0),
                    total_time_ms(0.0), average_time_per_move_ms(0.0),
                    cache_hits(0), cache_misses(0) {}
};

// 围棋引擎核心类 - 封装所有AI组件
class GoEngine {
public:
    // 构造函数和析构函数
    GoEngine();
    explicit GoEngine(const EngineConfig& config);
    ~GoEngine();
    
    // 禁止拷贝
    GoEngine(const GoEngine&) = delete;
    GoEngine& operator=(const GoEngine&) = delete;
    
    // 允许移动
    GoEngine(GoEngine&& other) noexcept;
    GoEngine& operator=(GoEngine&& other) noexcept;
    
    // ==================== 引擎生命周期 ====================
    
    // 初始化引擎
    bool initialize();
    bool initialize(const EngineConfig& config);
    
    // 关闭引擎
    void shutdown();
    
    // 检查是否已初始化
    bool is_initialized() const { return initialized_; }
    
    // ==================== 棋盘操作 ====================
    
    // 清空棋盘
    void clear_board();
    
    // 设置棋盘大小
    bool set_board_size(int size);
    
    // 获取棋盘大小
    int get_board_size() const { return config_.board_size; }
    
    // 落子
    bool play_move(int x, int y, StoneColor color);
    bool play_move(const std::string& move, StoneColor color);
    
    // 悔棋
    bool undo_move();
    
    // 获取当前棋盘状态
    const GoBoardCPU& get_board() const { return *board_; }
    
    // 获取当前玩家
    StoneColor get_current_player() const;
    
    // 设置当前玩家
    void set_current_player(StoneColor color);
    
    // ==================== 着法生成与搜索 ====================
    
    // 生成最佳着法
    SearchResult genmove(StoneColor color);
    SearchResult genmove();
    
    // 生成最佳着法（带时间限制）
    SearchResult genmove_with_time(StoneColor color, int time_ms);
    
    // 快速生成着法（减少搜索次数）
    SearchResult genmove_quick(StoneColor color);

    // 回退选择合法着法：搜索返回的着法无法落在真实棋盘时（如搜索内部棋盘
    // 未序列化劫信息而返回劫点），按NN策略在真实合法着中选最优，保证
    // genmove永远返回合法着（否则下游GTP对手拒绝后双方棋盘失同步）。
    SearchResult fallback_legal_move(StoneColor color);
    
    // 获取合法着法列表
    std::vector<std::pair<int, int>> get_legal_moves(StoneColor color) const;
    
    // 检查着法是否合法
    bool is_legal_move(int x, int y, StoneColor color) const;
    bool is_legal_move(const std::string& move, StoneColor color) const;
    
    // ==================== 分析功能 ====================
    
    // 分析当前局面
    SearchResult analyze(int max_iterations = 0);
    
    // 分析特定着法
    float analyze_move(int x, int y);
    
    // 获取着法概率分布
    std::vector<std::pair<std::string, float>> get_move_probabilities();
    
    // ==================== 游戏状态 ====================
    
    // 检查游戏是否结束
    bool is_game_over() const;
    
    // 计算领地
    std::pair<int, int> count_territory() const;
    
    // 估算胜率
    float estimate_win_rate(StoneColor color) const;
    
    // ==================== 引擎配置 ====================
    
    // 获取配置
    const EngineConfig& get_config() const { return config_; }
    
    // 设置配置
    void set_config(const EngineConfig& config);
    
    // 设置时间限制
    void set_time_limit(int time_ms);
    
    // 设置最大搜索次数
    void set_max_iterations(int iterations);
    
    // 设置探索权重
    void set_exploration_weight(float weight);
    
    // ==================== 状态查询 ====================
    
    // 获取引擎状态
    EngineState get_state() const { return state_; }
    
    // 获取统计信息
    EngineStats get_stats() const;
    
    // 重置统计信息
    void reset_stats();
    
    // ==================== 模型管理 ====================
    
    // 加载模型
    bool load_model(const std::string& path);
    
    // 检查模型是否已加载
    bool is_model_loaded() const;
    
    // ==================== 输入输出 ====================
    
    // 棋盘状态字符串
    std::string get_board_string() const;
    
    // SGF格式输出
    std::string to_sgf() const;
    
    // 从SGF加载
    bool load_from_sgf(const std::string& sgf);
    
    // 着法坐标转换
    static std::string move_to_string(int x, int y);
    static std::pair<int, int> string_to_move(const std::string& move);
    
    // 颜色转换
    static StoneColor string_to_color(const std::string& color_str);
    static std::string color_to_string(StoneColor color);
    
    // ==================== 回调设置 ====================
    
    // 设置搜索进度回调
    using ProgressCallback = std::function<void(int, int, float)>;
    void set_progress_callback(ProgressCallback callback);
    
    // 设置思考回调
    using ThinkingCallback = std::function<void(const std::string&)>;
    void set_thinking_callback(ThinkingCallback callback);
    
    // ==================== 高级功能 ====================
    
    // 后台思考
    void start_pondering();
    void stop_pondering();
    bool is_pondering() const;
    
    // 停止当前搜索
    void stop_search();
    
    // 获取引擎名称和版本
    static std::string get_engine_name();
    static std::string get_engine_version();
    static std::string get_engine_info();
    
private:
    // 内部组件
    std::unique_ptr<GoBoardCPU> board_;
    std::shared_ptr<CAMOE> ca_moe_;
    std::unique_ptr<search::HMGSDP_Optimized> hmgs_dp_;  // HMGS-DP搜索引擎
#ifdef USE_LIBTORCH
    // 神经网络引擎后端：加载LibTorch训练的CA-MOE模型（如elo_gated/elo_best.pt），
    // 通过set_neural_evaluator把策略/价值注入HMGS-DP搜索（AlphaZero式决策）
    std::shared_ptr<neural::CA_MOEExpertSystem> nn_model_;
#endif
    
    // 配置和状态
    EngineConfig config_;
    EngineState state_;
    std::atomic<bool> initialized_;
    std::atomic<bool> stop_requested_;
    
    // 统计信息
    mutable EngineStats stats_;
    mutable std::mutex stats_mutex_;
    
    // 回调函数
    ProgressCallback progress_callback_;
    ThinkingCallback thinking_callback_;
    
    // 内部方法
    SearchResult do_search(int max_iterations, int time_limit_ms);
    void update_stats(const SearchResult& result);
    void log_thinking(const std::string& message);
#ifdef USE_LIBTORCH
    // 初始化神经网络引擎后端：加载CA-MOE模型并绑定HMGS-DP评估器
    bool init_neural_engine();
    // 神经网络单次前向评估（策略362+价值，黑棋视角），返回false表示无nn后端
    bool nn_evaluate(std::vector<float>& policy_out, float& value_out) const;
#endif
    
    // 着法合法性检查
    bool check_move_valid(int x, int y, StoneColor color) const;
    
    // 坐标边界检查
    bool is_valid_coordinate(int x, int y) const;
};

// 引擎工厂函数
std::unique_ptr<GoEngine> create_engine();
std::unique_ptr<GoEngine> create_engine(const EngineConfig& config);

}  // namespace engine
}  // namespace go
}  // namespace spacego
