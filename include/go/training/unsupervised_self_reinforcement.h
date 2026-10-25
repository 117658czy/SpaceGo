#pragma once

#ifdef USE_LIBTORCH

#include <torch/torch.h>
#include <torch/optim/schedulers/lr_scheduler.h>
#include <vector>
#include <memory>
#include <string>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <queue>
#include <functional>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <future>
#include <unordered_map>

#include "go/neural/ca_moe_expert.h"
#include "go/neural/neural_network_cuda.h"
#include "go/board/go_board_cuda.h"
#include "go/training/experience_buffer.h"
#include "go/training/experience_collector.h"
#include "go/training/multi_dim_evaluator.h"
#include "go/search/ca_moe_enhanced_integration.h"
#include "framework/logger.h"
#include "framework/performance_monitor.h"
#include "framework/amp_grad_scaler.h"

namespace spacego {
namespace go {

// 前向声明：CudaStreamRAII定义在go/neural/neural_network_cuda.h中
#ifdef USE_CUDA
namespace neural { class CudaStreamRAII; }
#endif

namespace training {

// ==================== 配置结构体 ====================
struct SelfPlayConfig {
    int num_workers = 8;                    // 并行Worker数量（增大以加速数据生成）
    int simulations_per_move = 1600;        // 每步模拟次数（增大以提升自弈数据质量）
    int batch_size = 64;                    // 批量推理大小（增大以提升GPU利用率）
    float temperature = 1.0f;               // 初始温度参数
    float temperature_decay = 0.995f;       // 温度衰减率（每局衰减，非每步）
    float min_temperature = 0.0f;           // 最小温度（允许后期完全贪婪）
    float dirichlet_alpha = 0.03f;          // Dirichlet噪声alpha（适配19x19棋盘）
    float dirichlet_weight = 0.25f;         // Dirichlet噪声权重
    float virtual_loss = 3.0f;              // 虚拟损失（用于并行MCTS）
    bool use_cuda = true;                   // 是否使用CUDA加速
    int cuda_device_id = 0;                 // CUDA设备ID
    bool use_mixed_precision = true;        // 使用混合精度推理
    int max_game_length = 600;              // 最大对局长度（增大以允许更完整的对局）
    float resign_threshold = -0.9f;          // 认输阈值
    int seed = 42;                           // 随机种子
    // ===== SGF棋谱保存配置 =====
    bool save_sgf_games = true;              // 是否保存SGF棋谱（供Sabaki等工具打开）
    std::string sgf_output_dir = "./sgf";   // SGF棋谱输出目录
    int sgf_save_interval = 10;              // 每N局保存一次棋谱（避免磁盘占用过大）
    bool prestore_augmentation = true;       // 生成时预存8种D4对称版本（取代训练时增强），false则退回训练时增强
    // ===== 多元评价 + GAE 配置 =====
    bool enable_multi_dim_eval = true;       // 启用多元评价系统（16项量规）
    bool enable_gae = true;                  // 启用GAE时序差分学习
};

// 训练配置
struct TrainingConfig {
    int batch_size = 512;                   // 训练批次大小（增大以获得更稳定的梯度）
    int games_per_iteration = 100;          // 每迭代对局数（快速迭代模式）
    int num_epochs_per_iteration = 1;       // 每迭代轮数（快速迭代模式）
    int generation_train_ratio = 3;         // 生成:训练健康比例（每生成N局训练1个epoch，控制数据利用率）
    float learning_rate = 3e-4f;            // 初始学习率（降低以防止过拟合）
    float lr_decay_rate = 0.95f;            // 学习率衰减率
    float weight_decay = 5e-4f;             // 权重衰减（增大以增强L2正则化）
    float policy_loss_weight = 1.0f;        // 策略损失权重
    float value_loss_weight = 1.0f;         // 价值损失权重
    float gating_loss_weight = 0.5f;        // 门控损失权重
    float advantage_loss_weight = 0.5f;     // GAE优势加权的策略损失权重
    float entropy_regularization = 0.05f;   // 熵正则化系数（增大以鼓励策略多样性）
    // ===== GAE/TD配置 =====
    GAEConfig gae_config;                   // GAE参数配置
    bool use_mixed_precision = true;        // 使用混合精度训练
    bool gradient_clipping = true;          // 梯度裁剪
    float max_gradient_norm = 3.0f;         // 最大梯度范数（收紧以防止梯度爆炸）
    int window_size = 500000;               // 经验窗口大小（非渐进模式使用，限制50万条，动态收缩）
    // ===== 渐进式经验池扩张配置 =====
    bool progressive_expansion = true;      // 启用渐进式扩张（false=传统固定大小）
    int initial_window_size = 50000;        // 初始经验池容量（小池快速启动）
    int max_window_size = 500000;           // 最大经验池容量（最终扩张上限）
    float expansion_factor = 2.0f;          // 每次扩张倍数（2.0=翻倍）
    int expansion_interval = 10;            // 每N次迭代检查是否扩张
    int reanalyze_interval = 10;            // 重分析间隔
    std::string optimizer_type = "adamw";   // 优化器类型: adam, sgd, adamw（使用adamw以更好支持weight_decay）
    // ===== 防过拟合参数 =====
    float label_smoothing = 0.1f;           // Label Smoothing系数（平滑策略目标分布）
    bool use_data_augmentation = true;      // 是否使用8-fold棋盘对称性数据增强
    float dropout_rate = 0.1f;              // Dropout比率（在全连接层后添加Dropout）
    float validation_ratio = 0.1f;          // 验证集比例（留出10%数据用于验证）
};

// 评估配置
struct EvaluationConfig {
    int evaluation_interval = 1;            // 评估间隔（每次迭代都评估）
    int num_evaluation_games = 10;          // 评估对局数（快速迭代模式）
    float exploration_factor = 0.0f;        // 评估时探索因子（通常为0）
    bool use_baseline_model = true;         // 与基准模型对比
    int max_concurrent_games = 1;           // 最大并发评估游戏数（单线程避免GPU竞争）
    int max_eval_game_length = 200;         // 评估对局最大步数（防止无限对弈）
    float komi = 7.5f;                     // 贴目（中国规则白方贴7.5目）
};

// 完整训练系统配置
struct UnsupervisedTrainingConfig {
    SelfPlayConfig self_play;
    TrainingConfig training;
    EvaluationConfig evaluation;
    
    int total_iterations = 10000;           // 总迭代次数
    int checkpoint_interval = 10;           // 检查点保存间隔（更频繁保存）
    int log_interval = 10;                  // 日志打印间隔
    std::string checkpoint_dir = "./checkpoints";
    std::string log_dir = "./logs";
    std::string model_name = "spacego_unsupervised";
    int seed = 42;                          // 随机种子
    bool enable_profiling = false;          // 启用性能剖析
    int num_early_stopping_iterations = 20; // 早停检查窗口（收紧以更早检测过拟合）
    float min_improvement_threshold = 0.001f; // 最小改进阈值
    int validation_early_stop_patience = 5;  // 验证集损失早停耐心值（连续N次验证损失未改善则判定过拟合）
};

// ==================== 数据结构 ====================

// 自我对弈生成的单步数据
struct SelfPlayStep {
    torch::Tensor state;                    // 当前状态特征 [17, 19, 19]
    torch::Tensor policy_target;            // 策略目标（MCTS访问次数）[362]
    float value_target;                     // 价值目标（最终结果；GAE模式下=TD目标回报）
    float root_value;                       // 根节点价值估计
    torch::Tensor gating_weights;           // 门控权重 [num_experts]
    std::vector<int> expert_indices;        // 参与的专家索引
    int action_taken;                       // 实际采取的动作
    float temperature_at_move;              // 该步的温度
    int move_number;                        // 步数编号
    // ===== 多元评价（每步即时评估） =====
    MultiDimEvaluation multi_dim_eval;      // 该步后的多元评价结果
    float step_reward;                      // 即时奖励（多元评价差值）
    // ===== GAE/TD字段（游戏结束后填充） =====
    float gae_advantage;                    // GAE优势值
    float td_return;                        // TD目标回报
    bool has_gae_signal;                    // 是否已计算GAE信号
};

// 一局完整对局数据
struct GameData {
    int game_id;                            // 对局ID
    std::vector<SelfPlayStep> steps;        // 所有步骤
    float outcome;                          // 最终结果 (+1黑胜, -1白胜, 0平)
    int game_length;                        // 对局长度
    float total_inference_time_ms;          // 总推理时间
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;
    
    // 序列化支持
    std::vector<uint8_t> serialize() const;
    static GameData deserialize(const std::vector<uint8_t>& data);
};

// 训练统计信息
struct TrainingMetrics {
    int iteration = 0;
    int total_games_generated = 0;
    int total_training_steps = 0;
    
    // 损失值
    float avg_policy_loss = 0.0f;
    float avg_value_loss = 0.0f;
    float avg_gating_loss = 0.0f;
    float avg_total_loss = 0.0f;
    
    // 验证集损失（用于检测过拟合）
    float val_policy_loss = 0.0f;
    float val_value_loss = 0.0f;
    float val_total_loss = 0.0f;
    bool has_validation = false;           // 是否有验证集数据
    
    // 性能指标
    float current_elo = 1200.0f;
    float elo_change = 0.0f;
    float win_rate_vs_baseline = 0.5f;
    
    // 效率指标
    float games_per_second = 0.0f;
    float steps_per_second = 0.0f;
    float gpu_utilization = 0.0f;
    float memory_usage_mb = 0.0f;
    
    // 时间统计
    float self_play_time_sec = 0.0f;
    float training_time_sec = 0.0f;
    float evaluation_time_sec = 0.0f;
    float total_time_sec = 0.0f;
    
    // 序列化
    std::string to_string() const;
    static TrainingMetrics from_string(const std::string& str);
};

// ==================== 推理收集器（批量打包Worker推理请求） ====================

// 推理收集器：将多个Worker的独立推理请求打包为批量推理，提升GPU利用率
class InferenceCollector {
public:
    // 推理任务：包含输入状态和结果promise
    struct InferenceTask {
        torch::Tensor state;                                         // 棋盘状态 [17, 19, 19]
        std::promise<neural::CA_MOEInferenceResult> result_promise;  // 结果promise
    };

    // 构造函数：model=共享模型, batch_size=批量大小(默认64), timeout_ms=超时(默认100ms)
    InferenceCollector(std::shared_ptr<neural::CA_MOEExpertSystem> model,
                       int batch_size = 64, int timeout_ms = 100);

    ~InferenceCollector();

    // 禁止拷贝
    InferenceCollector(const InferenceCollector&) = delete;
    InferenceCollector& operator=(const InferenceCollector&) = delete;

    // 提交一个棋盘状态用于推理，返回future结果
    std::future<neural::CA_MOEInferenceResult> submit(torch::Tensor state);

    // 启动后台处理线程
    void start();

    // 停止后台线程
    void stop();

    // 获取统计信息
    struct CollectorStats {
        int total_tasks_processed = 0;
        int total_batches = 0;
        double avg_batch_size = 0.0;
        double avg_latency_ms = 0.0;
    };
    CollectorStats get_stats() const;

private:
    std::shared_ptr<neural::CA_MOEExpertSystem> model_;
    int batch_size_;
    int timeout_ms_;

    // 任务队列
    std::queue<std::shared_ptr<InferenceTask>> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    // 后台处理线程
    std::thread process_thread_;
    std::atomic<bool> running_{false};

    // 统计
    mutable std::mutex stats_mutex_;
    int total_tasks_ = 0;
    int total_batches_ = 0;
    double total_batch_sum_ = 0.0;
    double total_latency_ms_ = 0.0;

    // 处理循环
    void process_loop();

    // 执行批量推理
    void process_batch(std::vector<std::shared_ptr<InferenceTask>>& tasks);
};

// ==================== 自我对弈生成器 ====================

class SelfPlayGenerator {
public:
    SelfPlayGenerator(int worker_id,
                      const SelfPlayConfig& config,
                      std::shared_ptr<neural::CA_MOEExpertSystem> model,
                      std::shared_ptr<InferenceCollector> collector = nullptr);
    
    ~SelfPlayGenerator();
    
    // 初始化CUDA环境
    bool initialize();
    
    // 生成一局自对弈
    GameData generate_game(int game_id);
    
    // 批量生成多局游戏
    std::vector<GameData> generate_games(int start_id, int count);
    
    // 设置温度
    void set_temperature(float temp) { current_temperature = temp; }
    float get_temperature() const { return current_temperature; }
    
    // 获取统计信息
    struct GeneratorStats {
        int games_completed = 0;
        int total_moves = 0;
        double total_inference_time_ms = 0.0;
        double avg_inference_time_ms = 0.0;
    };
    GeneratorStats get_stats() const;
    void reset_stats();
    
    // 停止生成
    void stop() { running = false; }
    bool is_running() const { return running; }
    
    int get_worker_id() const { return worker_id; }

private:
    int worker_id;
    SelfPlayConfig config;
    std::shared_ptr<neural::CA_MOEExpertSystem> model;
    std::shared_ptr<InferenceCollector> inference_collector_;  // 推理收集器（可选，用于批量推理）
    std::shared_ptr<search::CA_MOEEnhancedIntegration> search_integration;
    std::unique_ptr<search::HMGSDP_Optimized> hmgs_searcher_;  // HMGS-DP搜索器（生成访问次数分布作训练目标）
    // NN评估缓存：状态字符串 → (策略分布[362], 价值)
    // 作用：expand_node与simulate对同一节点连续调用时合并为单次前向推理，
    //       使每步搜索的NN推理次数减半（AlphaZero网络一次前向同时输出策略与价值）
    std::unordered_map<std::string, std::pair<std::vector<float>, float>> nn_eval_cache_;
    
    std::atomic<bool> running;
    float current_temperature;
    std::mt19937 rng;

    // 动态对战统计（跨步状态，用于丰富即时奖励信号）
    // 胜率保持时间：连续保持胜率估计>阈值(0.55)的步数，奖励"稳守优势"行为
    int win_hold_steps_ = 0;
    float last_win_rate_ = 0.5f;
    // 策略执行率：实际落子是否为搜索访问分布的top-1（模型"说到做到"程度）
    int policy_exec_hits_ = 0;
    int policy_exec_total_ = 0;
    
    GeneratorStats stats;
    mutable std::mutex stats_mutex;
    
    // 内部方法
    // 选择动作：根据策略和温度采样，使用legal_mask过滤非法落子点
    // legal_mask: 大小362的布尔数组，true表示该位置合法（索引0-360为棋盘位置，361为pass）
    int select_action(const torch::Tensor& policy, float temperature,
                      const std::vector<bool>& legal_mask);
    torch::Tensor add_dirichlet_noise(const torch::Tensor& policy);
    torch::Tensor normalize_policy(const torch::Tensor& mcts_visits);
    void apply_temperature_decay();
    std::vector<int> get_top_k_experts(const torch::Tensor& gating_weights, int top_k);
    // 构建合法着法mask：索引0-360为棋盘位置，361为pass（始终合法）
    std::vector<bool> build_legal_mask(const GoBoardCPU& board) const;
    // 使用HMGS-DP搜索生成策略分布（访问次数归一化）与根价值
    // 返回 (策略分布[362], 根价值)
    std::pair<std::vector<float>, float> search_policy(
        const std::string& board_state, int current_player);
    // 初始化HMGS-DP搜索器（注入神经网络评估回调）
    void initialize_hmgs_search();
    // 保存SGF棋谱到文件（供Sabaki等工具打开）
    void save_sgf(const GameData& game) const;
    // 生成8种D4对称变换版本（棋盘对称群：4次旋转×2次翻转）
    // 返回8个(state, policy)对，每种对应一个对称变换
    // state: [17, 19, 19], policy: [362]
    std::vector<std::pair<torch::Tensor, torch::Tensor>> generate_d4_symmetries(
        const torch::Tensor& state, const torch::Tensor& policy);
    // ===== 多元评价 + GAE =====
    // 对当前棋盘做多元评价
    MultiDimEvaluation evaluate_board(const GoBoardCPU& board, StoneColor perspective);
    // 游戏结束后计算GAE并填充到所有步骤
    void compute_gae_for_game(GameData& game);
    
    // CUDA相关
    torch::Device device;
#if defined(USE_CUDA)
    std::unique_ptr<neural::CudaStreamRAII> cuda_stream;  // CUDA流RAII包装器
#endif
    bool use_fp16;
    
    // ===== 多元评价器（每个Worker独立实例） =====
    std::unique_ptr<MultiDimEvaluator> multi_dim_evaluator;
    std::unique_ptr<GAECalculator> gae_calculator;
};

// ==================== 高效经验回放系统 ====================

class EfficientReplayBuffer {
public:
    EfficientReplayBuffer(int capacity, int num_experts = 8);
    ~EfficientReplayBuffer();
    
    // 添加游戏数据
    void add_game(const GameData& game);
    
    // 批量添加
    void add_games(const std::vector<GameData>& games);
    
    // 采样训练批次
    struct TrainingBatch {
        torch::Tensor states;               // [batch_size, 17, 19, 19]
        torch::Tensor target_policies;       // [batch_size, 362]
        torch::Tensor target_values;         // [batch_size]
        torch::Tensor gating_weights;        // [batch_size, num_experts]
        torch::Tensor advantages;            // [batch_size] GAE优势值
        torch::Tensor td_returns;            // [batch_size] TD目标回报
        std::vector<int> expert_indices;     // 专家索引
        std::vector<float> importance_weights; // 重要性权重（用于PER）
        std::vector<int> buffer_indices;     // 缓冲区索引（用于更新优先级）
    };
    
    TrainingBatch sample_batch(int batch_size, const std::string& strategy = "uniform");
    
    // 更新优先级（用于PER）
    void update_priorities(const std::vector<int>& indices, 
                           const std::vector<float>& priorities);
    
    // 动态扩容经验池（渐进式扩张时使用）
    bool expand_capacity(int new_capacity);

    // 初始化GPU预装存储（在initialize时调用，检查显存并决定模式）
    bool init_gpu_storage(torch::Device device, int capacity, int num_experts);
    
    // 获取状态
    int size() const;
    int capacity() const { return capacity_; }
    bool is_ready() const { return size() >= min_samples_to_train; }
    int get_min_samples() const { return min_samples_to_train; }
    void set_min_samples(int n) { min_samples_to_train = n; }
    
    // 清空
    void clear();
    
    // 持久化
    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    int capacity_;
    int num_experts_;
    int min_samples_to_train = 1000;
    
    // 使用环形缓冲区存储（CPU模式，GPU预装模式时不再使用）
    struct BufferEntry {
        torch::Tensor state;
        torch::Tensor target_policy;
        float target_value;
        torch::Tensor gating_weights;
        std::vector<int> expert_indices;
        float priority;
        float advantage;           // GAE优势
        float td_return;           // TD目标回报
    };
    
    std::vector<BufferEntry> buffer;
    int next_index = 0;
    int current_size = 0;
    
    // GPU预装经验存储（将数据以GPU大矩阵形式存储，采样时直接index_select）
    bool use_gpu_storage_ = false;
    torch::Device storage_device_{torch::kCPU};
    
    // GPU大矩阵存储：所有经验数据预装在一张大矩阵中
    // 维度：[capacity, ...]
    torch::Tensor gpu_states_;           // [capacity, 17, 19, 19]
    torch::Tensor gpu_policies_;         // [capacity, 362]
    torch::Tensor gpu_values_;           // [capacity]
    torch::Tensor gpu_gating_weights_;   // [capacity, num_experts]
    torch::Tensor gpu_advantages_;       // [capacity]
    torch::Tensor gpu_td_returns_;       // [capacity]
    // expert_indices和priority仍然用CPU存储（数据量小）
    std::vector<std::vector<int>> expert_indices_list_;
    std::vector<float> priority_list_;
    
    // 私有辅助方法：分配GPU大矩阵
    bool init_gpu_storage_matrices(torch::Device device, int capacity, int num_experts);
    void fallback_to_cpu_storage();
    
    mutable std::mutex buffer_mutex;
    
    // 优先级经验回放（PER）
    bool use_prioritized_replay = true;
    float alpha = 0.6f;                     // 优先级指数
    float beta = 0.4f;                      // 重要性采样指数
    float beta_increment = 0.001f;          // beta递增
    float epsilon = 1e-6f;                  // 避免零优先级
    
    // SumTree for PER
    class SumTreeImpl;
    std::unique_ptr<SumTreeImpl> sum_tree;
    
    float max_priority = 1.0f;
    std::mt19937 rng;
};

// ==================== 混合精度训练器 ====================

class MixedPrecisionTrainer {
public:
    MixedPrecisionTrainer(std::shared_ptr<neural::CA_MOEExpertSystem> model,
                          const TrainingConfig& config);
    
    ~MixedPrecisionTrainer();
    
    // 初始化优化器和调度器
    void setup_optimizer();
    
    // 单次训练步骤
    struct TrainStepResult {
        float policy_loss;
        float value_loss;
        float gating_loss;
        float total_loss;
        float grad_norm;
        float learning_rate;
        float elapsed_time_ms;
    };
    
    TrainStepResult train_step(const EfficientReplayBuffer::TrainingBatch& batch);
    
    // 训练一个epoch
    struct EpochResult {
        float avg_policy_loss;
        float avg_value_loss;
        float avg_gating_loss;
        float avg_total_loss;
        int num_steps;
        float epoch_time_sec;
    };
    
    EpochResult train_epoch(EfficientReplayBuffer& replay_buffer);
    
    // 验证集损失评估（不更新梯度，用于检测过拟合）
    EpochResult evaluate_validation(EfficientReplayBuffer& replay_buffer, int val_batch_count = 5);
    
    // 学习率调整
    void adjust_learning_rate(float new_lr);
    void step_lr_scheduler(int iteration);
    
    // 梯度操作
    void zero_grad();
    
    // 模型操作
    void save_model(const std::string& path);
    void load_model(const std::string& path);

    // 对手风格条件化训练开关：
    // 二倍体(diploid)训练开启（默认true）——门控学会"看对手棋风分派专家"；
    // ELO门控在线训练(elo-gated)关闭——单模型M vs 固定Gen0，推理时不注入对手画像，
    // 若训练时注入随机风格会造成train/eval不一致（门控学了无用的条件分支）。
    void set_opponent_style_conditioning(bool enable) { opponent_style_conditioning_ = enable; }
    bool get_opponent_style_conditioning() const { return opponent_style_conditioning_; }

    // 获取当前学习率
    float get_learning_rate() const;

private:
    std::shared_ptr<neural::CA_MOEExpertSystem> model;
    TrainingConfig config;
    // 对手风格条件化训练开关（默认开启，见set_opponent_style_conditioning注释）
    bool opponent_style_conditioning_ = true;
    
    // 优化器
    std::unique_ptr<torch::optim::Optimizer> optimizer;
    
    // 学习率调度器
    std::unique_ptr<framework::CosineAnnealingLRScheduler> lr_scheduler;
    
    // 混合精度训练（AMP）
    std::unique_ptr<framework::AMPGradScaler> scaler;
    bool use_amp;
    
    // 设备
    torch::Device device;
    
    // 统计
    int total_steps_trained = 0;
    float current_lr;
    
    // 损失计算结果
    struct LossResult {
        float policy_loss = 0.0f;
        float value_loss = 0.0f;
        float gating_loss = 0.0f;
        float total_loss = 0.0f;
    };
    
    // 统一计算训练损失（train_step和evaluate_validation共用）
    // 返回：total_loss张量（用于反向传播） + 浮点损失值
    // 门控损失 = KL(门控‖均匀) + 负载均衡 + logits L2正则（防坍缩关键：
    //   softmax输出饱和时KL梯度≈0，对logits直接L2正则梯度始终存在）
    std::pair<torch::Tensor, LossResult> compute_losses(
        const torch::Tensor& pred_policy,
        const torch::Tensor& pred_value,
        const torch::Tensor& pred_gating,
        const torch::Tensor& target_policy,
        const torch::Tensor& target_value,
        const std::vector<float>& importance_weights = {},
        const torch::Tensor& advantages = {},
        const torch::Tensor& pred_gating_logits = {});
};

// ==================== ELO评估系统 ====================

class ELOEvaluator {
public:
    ELOEvaluator(std::shared_ptr<neural::CA_MOEExpertSystem> current_model,
                 std::shared_ptr<neural::CA_MOEExpertSystem> baseline_model,
                 const EvaluationConfig& config);
    
    // 执行评估
    struct EvaluationResult {
        int games_played;
        int wins;
        int losses;
        int draws;
        float win_rate;
        float elo_rating;
        float elo_change;
        float avg_game_length;
        float confidence_interval_lower;
        float confidence_interval_upper;
        std::chrono::duration<float> evaluation_duration;
    };
    
    EvaluationResult evaluate();
    
    // 单局对战：真实对弈，交替使用current_model和baseline_model
    float play_single_game(bool current_first);
    
    // 计算ELO（标准ELO公式，K=32）
    float calculate_elo(float win_rate, float k_factor = 32.0f);
    static float expected_win_rate(float elo_a, float elo_b);
    
    // 获取历史
    std::vector<EvaluationResult> get_history() const { return history; }
    
    // 设置训练进度（0.0-1.0），用于ELO评估时衡量模型强度
    void set_training_progress(float progress) { training_progress = progress; }
    // 设置评估对弈使用的推理设备（CPU或CUDA）
    void set_eval_device(torch::Device dev) { eval_device = dev; }

private:
    std::shared_ptr<neural::CA_MOEExpertSystem> current_model;
    std::shared_ptr<neural::CA_MOEExpertSystem> baseline_model;
    EvaluationConfig config;
    
    float current_elo = 1200.0f;
    float baseline_elo = 1200.0f;           // 基准模型ELO（固定为初始值）
    float training_progress = 0.0f;  // 训练进度，用于衡量模型强度提升
    std::vector<EvaluationResult> history;
    std::mt19937 rng;
    
    // 评估对弈中使用的推理设备
    torch::Device eval_device = torch::Device(torch::kCPU);
    // 推理互斥锁：保护多线程环境下的forward调用
    mutable std::mutex eval_inference_mutex;
};

// ==================== 主训练管理器 ====================

class UnsupervisedSelfReinforcementTrainer {
public:
    explicit UnsupervisedSelfReinforcementTrainer(const UnsupervisedTrainingConfig& config);
    
    ~UnsupervisedSelfReinforcementTrainer();
    
    // 初始化所有组件
    bool initialize();
    
    // 开始完整训练流程
    void start_training();
    
    // 停止训练
    void stop_training();
    
    // 单步迭代（用于调试或自定义控制）
    TrainingMetrics run_single_iteration();
    
    // 获取当前状态
    bool is_training() const { return training.load(); }
    int get_current_iteration() const { return current_iteration.load(); }
    TrainingMetrics get_latest_metrics() const { return latest_metrics; }
    
    // 获取组件（用于高级用法）
    std::vector<std::shared_ptr<SelfPlayGenerator>>& get_generators() { return generators; }
    std::shared_ptr<EfficientReplayBuffer> get_replay_buffer() { return replay_buffer; }
    std::shared_ptr<MixedPrecisionTrainer> get_trainer() { return trainer; }
    std::shared_ptr<ELOEvaluator> get_evaluator() { return evaluator; }
    std::shared_ptr<neural::CA_MOEExpertSystem> get_current_model() { return current_model; }
    std::shared_ptr<neural::CA_MOEExpertSystem> get_best_model() { return best_model; }
    
    // 检查点管理
    void save_checkpoint(const std::string& path = "");
    void load_checkpoint(const std::string& path);
    std::string get_latest_checkpoint_path() const;
    
    // 日志和报告
    void generate_training_report(const std::string& output_path);
    std::vector<TrainingMetrics> get_training_history() const { return training_history; }

private:
    UnsupervisedTrainingConfig config;
    
    // 核心模型
    std::shared_ptr<neural::CA_MOEExpertSystem> current_model;
    std::shared_ptr<neural::CA_MOEExpertSystem> best_model;
    std::shared_ptr<neural::CA_MOEExpertSystem> baseline_model;
    
    // 核心组件
    std::vector<std::shared_ptr<SelfPlayGenerator>> generators;
    std::shared_ptr<EfficientReplayBuffer> replay_buffer;
    std::shared_ptr<MixedPrecisionTrainer> trainer;
    std::shared_ptr<ELOEvaluator> evaluator;
    std::shared_ptr<InferenceCollector> inference_collector_;  // 推理收集器（批量打包推理请求）
    
    // 线程控制
    std::atomic<bool> training{false};
    std::thread training_thread;
    std::vector<std::thread> self_play_threads;
    
    // 状态变量
    std::atomic<int> current_iteration{0};
    std::atomic<int> total_games{0};
    std::atomic<int> total_steps{0};
    TrainingMetrics latest_metrics;
    std::vector<TrainingMetrics> training_history;
    
    // 最佳模型跟踪
    float best_elo = 1200.0f;
    int iterations_without_improvement = 0;
    
    // 验证集损失跟踪（用于过拟合检测）
    float best_val_loss = std::numeric_limits<float>::max();  // 历史最佳验证集总损失
    int iterations_without_val_improvement = 0;               // 验证集损失连续未改善的迭代次数
    float latest_val_policy_loss = 0.0f;                       // 最近一次验证集策略损失
    float latest_val_value_loss = 0.0f;                        // 最近一次验证集价值损失
    float latest_val_total_loss = 0.0f;                        // 最近一次验证集总损失
    bool has_latest_validation = false;                        // 是否已有验证集评估结果
    
    // 训练损失跟踪（从run_training_phase传递到run_single_iteration的metrics）
    float latest_train_policy_loss = 0.0f;                     // 最近一次训练epoch策略损失
    float latest_train_value_loss = 0.0f;                      // 最近一次训练epoch价值损失
    float latest_train_gating_loss = 0.0f;                     // 最近一次训练epoch门控损失
    float latest_train_total_loss = 0.0f;                      // 最近一次训练epoch总损失
    
    // 同步机制
    std::mutex model_mutex;
    std::condition_variable model_updated_cv;
    
    // 日志
    framework::Logger* logger;
    std::unique_ptr<framework::PerformanceMonitor> perf_monitor;
    
    // 内部方法
    void training_loop();
    void run_self_play_phase();
    void run_training_phase();
    void run_evaluation_phase();
    void update_models_if_needed();
    bool should_stop_early() const;
    
    // 日志输出
    void log_iteration_start(int iteration);
    void log_iteration_complete(const TrainingMetrics& metrics);
    void log_training_progress(const TrainingMetrics& metrics);
    
    // 文件工具
    void ensure_directory_exists(const std::string& path);
    std::string timestamp_string() const;
};

} // namespace training
} // namespace go
} // namespace spacego
#endif  // USE_LIBTORCH
