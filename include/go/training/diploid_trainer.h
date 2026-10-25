#pragma once

#ifdef USE_LIBTORCH

#include <torch/torch.h>
#include <vector>
#include <memory>
#include <string>
#include <atomic>
#include <random>
#include <mutex>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <filesystem>

#include "go/board/go_board_cuda.h"
#include "go/neural/ca_moe_expert.h"
#include "go/search/hmgs_dp_optimized.h"
#include "go/training/unsupervised_self_reinforcement.h"

namespace spacego {
namespace go {
namespace training {

// ==================== 二倍体训练配置（生产级） ====================

struct DiploidConfig {
    // === 模型配置 ===
    int num_experts = 8;                  // CA-MOE专家数量
    int board_size = 19;
    float dropout_rate = 0.1f;            // Dropout比率
    int hidden_dim = 256;                 // 模型通道宽度（长时间模式保持大模型256；小模型需重新预训练对应基线）

    // === 自对弈配置 ===
    int num_self_play_workers = 1;        // 自对弈Worker数（GPU已饱和时多worker无收益且叠加线程开销，保持单worker）
    int games_per_gen = 100;              // 每代自对弈局数（生产级）
    int simulations_per_move = 300;       // HMGS-DP搜索模拟数（用于自对弈数据生成）
    int eval_simulations = 300;           // 竞争对弈/ELO评估搜索模拟数（评估与训练一致用搜索，300兼顾评估质量与时长）
    int max_game_length = 400;            // 最大对局长度

    // === 分化配置 ===
    int seed_a = 42;                      // 模型A随机种子
    int seed_b = 123;                     // 模型B随机种子
    float temp_decay_a = 0.92f;           // A温度衰减率（快降→更贪婪）
    float temp_decay_b = 0.97f;           // B温度衰减率（慢降→更多探索）
    float perturbation_std = 0.005f;      // 选择后参数扰动标准差

    // === 训练配置 ===
    float learning_rate = 3e-4f;          // 学习率
    float lr_decay_rate = 0.95f;          // 学习率衰减
    float weight_decay = 5e-4f;           // 权重衰减（L2正则化）
    int training_epochs_per_gen = 3;      // 每代训练轮数
    int batch_size = 128;                 // 训练批次大小（256时FP32激活峰值≈5.8GB打满6GB上限，训练撞顶变慢；128激活减半，峰值≈4GB不撞顶）
    float policy_loss_weight = 1.0f;      // 策略损失权重
    float value_loss_weight = 1.0f;       // 价值损失权重
    float gating_loss_weight = 10.0f;     // 门控损失权重（防专家坍缩：2.0仍压不住"赢家通吃"坍缩[0,0,0,1,0,0,0,0]；10.0时坍缩惩罚=6.08×10≈60远大于策略损失5，健康时0.5×10=5轻微）
    int replay_buffer_capacity = 15000;  // 经验池容量（50局≈13000步规模，数据优先策略下每代数据量翻倍）
    bool progressive_expansion = true;    // 渐进式经验池扩张
    int initial_buffer_size = 5000;       // 初始经验池大小
    float validation_ratio = 0.1f;        // 验证集比例
    bool use_mixed_precision = true;       // 混合精度训练

    // === 竞争配置 ===
    int competition_games = 40;           // 竞争对弈局数（生产级）
    float komi = 7.5f;                   // 贴目
    float win_rate_threshold = 0.55f;     // 胜率阈值（>55%判胜）

    // === ELO门控（长时间模式：必须保证ELO"大幅"上涨） ===
    // 单代接受所需的最低ELO增量：只有超出历史最佳 >= 该阈值才接受，
    // 低于阈值一律回滚，保证每次接受都是实打实的大幅提升，过滤±35 Elo评估噪声。
    float elo_min_improvement = 30.0f;   // 单代接受的最小ELO增量（>评估噪声，才计为大幅上涨）

    // === 进化配置 ===
    int max_generations = 50;             // 最大代数
    int no_improve_patience = 5;          // 连续无提升停止代数
    int gen0_pretrain_epochs = 6;        // Gen0预训练轮数（双倍训练）

    // === 输出配置 ===
    std::string output_dir;
    std::string checkpoint_dir;
    std::string log_dir;
    int checkpoint_interval = 1;          // 每N代保存检查点（1=每代保存，供ELO门控回滚到上一代）
    int seed = 42;                        // 全局随机种子
};

// ==================== 代训练指标 ====================

struct DiploidGenerationMetrics {
    int generation = 0;

    // 模型A训练指标
    float a_policy_loss = 0.0f;
    float a_value_loss = 0.0f;
    float a_gating_loss = 0.0f;
    float a_total_loss = 0.0f;

    // 模型B训练指标
    float b_policy_loss = 0.0f;
    float b_value_loss = 0.0f;
    float b_gating_loss = 0.0f;
    float b_total_loss = 0.0f;

    // 验证集指标
    float a_val_loss = 0.0f;
    float b_val_loss = 0.0f;

    // 竞争结果（对Gen0基线：a_win_rate=A对Gen0胜率，b_win_rate=B对Gen0胜率）
    int a_wins = 0;
    int b_wins = 0;
    int draws = 0;
    float a_win_rate = 0.0f;
    float b_win_rate = 0.0f;   // B对Gen0基线胜率（用于消除自对弈坍塌假胜者）

    // 分化指标
    float parameter_distance = 0.0f;

    // 进化信息
    char winner = '-';
    float best_elo = 0.0f;
    bool accepted = false;               // 本代是否被ELO门控接受（ELO创历史新高）

    // 数据生成统计
    int a_games_generated = 0;
    int b_games_generated = 0;

    // 耗时
    float elapsed_minutes = 0.0f;
};

// ==================== 生产级二倍体训练器 ====================

class DiploidTrainer {
public:
    explicit DiploidTrainer(const DiploidConfig& config);
    ~DiploidTrainer();

    // 禁止拷贝和移动（unique_ptr成员+不完整类型限制）
    DiploidTrainer(const DiploidTrainer&) = delete;
    DiploidTrainer& operator=(const DiploidTrainer&) = delete;
    DiploidTrainer(DiploidTrainer&&) = delete;
    DiploidTrainer& operator=(DiploidTrainer&&) = delete;

    // 运行完整训练流程
    void run();

    // 获取各代指标
    const std::vector<DiploidGenerationMetrics>& get_history() const { return history_; }

private:
    // === 初始化 ===
    bool initialize();

    // === 模型参数操作 ===
    // 复制CA-MOE模型参数
    void copy_model_parameters(
        std::shared_ptr<neural::CA_MOEExpertSystem> src,
        std::shared_ptr<neural::CA_MOEExpertSystem> dst);

    // 添加高斯噪声到所有参数
    void add_parameter_noise(
        std::shared_ptr<neural::CA_MOEExpertSystem> model,
        float stddev);

    // 计算参数距离 ||M_a - M_b|| / ||M_a||
    float compute_parameter_distance(
        std::shared_ptr<neural::CA_MOEExpertSystem> model_a,
        std::shared_ptr<neural::CA_MOEExpertSystem> model_b);

    // === 自对弈数据生成 ===
    // 为指定模型生成自对弈数据
    std::vector<GameData> generate_self_play_data(
        std::shared_ptr<neural::CA_MOEExpertSystem> model,
        int num_games,
        int seed,
        float temperature_decay);

    // === 训练 ===
    // 训练模型一个世代
    std::tuple<float, float, float, float> train_model_generation(
        std::shared_ptr<neural::CA_MOEExpertSystem> model,
        std::shared_ptr<EfficientReplayBuffer>& buffer,
        std::shared_ptr<MixedPrecisionTrainer>& trainer);

    // 创建训练组件
    void setup_training_components(
        std::shared_ptr<neural::CA_MOEExpertSystem> model,
        std::shared_ptr<EfficientReplayBuffer>& buffer,
        std::shared_ptr<MixedPrecisionTrainer>& trainer);

    // === 竞争对弈 ===
    // 运行A vs B竞争对弈（搜索对弈，与自对弈一致评估真实棋力）
    void run_competition(
        std::shared_ptr<neural::CA_MOEExpertSystem> model_a,
        std::shared_ptr<neural::CA_MOEExpertSystem> model_b,
        DiploidGenerationMetrics& metrics);

    // 单局对战（HMGS-DP搜索选动作）
    float play_competition_game(
        std::shared_ptr<neural::CA_MOEExpertSystem> black_model,
        std::shared_ptr<neural::CA_MOEExpertSystem> white_model,
        std::mt19937& rng);

    // === 评估搜索器 ===
    // 初始化竞争对弈/ELO评估用的HMGS-DP搜索器（批量推理绑定模型）
    void init_eval_searcher();
    // 用指定模型对当前局面执行搜索，返回(访问分布[362], 根价值)
    std::pair<std::vector<float>, float> eval_search(
        std::shared_ptr<neural::CA_MOEExpertSystem> model,
        const GoBoardCPU& board);

    // === 进化循环 ===
    void run_generation_zero();
    DiploidGenerationMetrics run_generation(int gen);

    // === ELO计算 ===
    float compute_elo_change(float win_rate, int num_games);

    // === 检查点管理 ===
    void save_checkpoint(
        std::shared_ptr<neural::CA_MOEExpertSystem> model_a,
        std::shared_ptr<neural::CA_MOEExpertSystem> model_b,
        int generation);
    void save_model(
        std::shared_ptr<neural::CA_MOEExpertSystem> model,
        const std::string& name);

    // === 输出 ===
    void print_table_header();
    void print_separator();
    void print_generation_metrics(const DiploidGenerationMetrics& m);

    // === 配置 ===
    DiploidConfig config_;
    torch::Device device_;

    // === 模型（二倍体对） ===
    std::shared_ptr<neural::CA_MOEExpertSystem> model_a_;
    std::shared_ptr<neural::CA_MOEExpertSystem> model_b_;
    std::shared_ptr<neural::CA_MOEExpertSystem> gen0_model_;

    // === 评估搜索器（竞争对弈/ELO评估共用，与自对弈搜索逻辑一致） ===
    std::shared_ptr<search::HMGSDP_Optimized> eval_searcher_;

    // === 训练组件（每代重建） ===
    std::shared_ptr<EfficientReplayBuffer> replay_a_;
    std::shared_ptr<EfficientReplayBuffer> replay_b_;
    std::shared_ptr<MixedPrecisionTrainer> trainer_a_;
    std::shared_ptr<MixedPrecisionTrainer> trainer_b_;

    // === 随机数生成器 ===
    std::mt19937 global_rng_;

    // === 统计 ===
    std::vector<DiploidGenerationMetrics> history_;
    float best_elo_ = 0.0f;
    float peak_elo_ = 0.0f;   // 历史最高ELO（用于"新一代更好则替换"的最佳模型保存判定）
    int best_gen_ = 0;
    int resumed_gen_ = -1;   // 恢复的代数，-1表示全新开始

    // === 时间记录 ===
    std::chrono::steady_clock::time_point start_time_;
};

} // namespace training
} // namespace go
} // namespace spacego

#endif // USE_LIBTORCH
