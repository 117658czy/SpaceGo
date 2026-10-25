#pragma once

#ifdef USE_LIBTORCH

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <chrono>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <functional>

#include "go/training/param_registry.h"
#include "go/training/unsupervised_self_reinforcement.h"

namespace spacego {
namespace go {
namespace training {

// ==================== 自拟合调参器 ====================
//
// 核心思路：
// 1. 生成少量棋局 → 小经验池（~5000条）
// 2. 用当前参数训练一次 → 记录损失 L_before
// 3. 自动更换参数
// 4. 用同一经验池再训练 → 记录损失 L_after
// 5. 计算变化率 improvement = (L_before - L_after) / L_before
// 6. 如果 improvement > fp8_precision：保留该参数，继续调下一个
// 7. 如果 improvement <= fp8_precision：该参数已收敛
// 8. 全部训练参数收敛后 → 用收敛网络评估搜索参数好坏
//
// FP8 E4M3精度：3位尾数，对损失值~6精度约0.5，对~0.1精度约0.008
// 收敛阈值：绝对改善 < 当前损失 × 0.1%（近似fp8可发现精度）

// 单次调参试验的记录
struct TuneTrial {
    std::string param_name;         // 调整的参数名
    ParamValue old_value;           // 旧值
    ParamValue new_value;           // 新值
    float loss_before;              // 调参前损失
    float loss_after;               // 调参后损失
    float improvement;              // 改善率 (loss_before - loss_after) / loss_before
    bool accepted;                  // 是否接受此变更
    std::string direction;          // "up" / "down" / "reset"
    float elapsed_sec;              // 本次试验耗时
};

// 参数调优状态
struct ParamTuneState {
    std::string name;               // 参数名
    bool converged = false;         // 是否已收敛
    int consecutive_no_improve = 0; // 连续无改善次数
    int total_trials = 0;           // 总试验次数
    float best_improvement = 0.0f;  // 最佳单次改善率
    ParamValue best_value;          // 最佳值
    float last_explored_delta = 0.0f; // 上次探索的步长比例
};

// 自拟合调参器
class AutoTuner {
public:
    // 构造函数：传入基础训练配置
    explicit AutoTuner(const UnsupervisedTrainingConfig& base_config);

    ~AutoTuner() = default;

    // 启动自动调参主循环
    void run();

    // 从外部设置初始参数（可选）
    void set_initial_param(const std::string& name, const ParamValue& value);

    // 设置输出目录
    void set_output_dir(const std::string& dir);

    // 获取调参历史记录
    const std::vector<TuneTrial>& get_history() const { return history_; }

    // 获取当前最佳配置
    UnsupervisedTrainingConfig get_best_config() const;

private:
    UnsupervisedTrainingConfig base_config_;   // 基础配置（会被不断修改）
    std::string output_dir_ = "./auto_tune_output";

    // 核心组件（在initialize中创建）
    std::shared_ptr<neural::CA_MOEExpertSystem> model_;
    std::shared_ptr<MixedPrecisionTrainer> trainer_;
    std::shared_ptr<EfficientReplayBuffer> replay_buffer_;
    std::vector<std::unique_ptr<SelfPlayGenerator>> generators_;

    // 调参状态
    std::vector<ParamTuneState> training_param_states_;  // 训练参数调优状态
    std::vector<ParamTuneState> search_param_states_;    // 搜索参数调优状态
    std::vector<TuneTrial> history_;                     // 历史记录

    // 随机数生成器
    std::mt19937 rng_;

    // FP8精度阈值：相对改善低于此值视为收敛
    // FP8 E4M3有3位尾数，相对精度约2^-3 ≈ 12.5%，但实际应更精细
    // 这里用0.1%作为收敛判据（对loss~6来说绝对精度≈0.006）
    static constexpr float FP8_RELATIVE_THRESHOLD = 0.001f;

    // 每次同池重训最大步数
    static constexpr int RETRAIN_STEPS = 50;

    // 连续无改善最大次数（超过此数认为参数收敛）
    static constexpr int MAX_NO_IMPROVE = 3;

    // ===== 核心流程 =====

    // 初始化所有组件（模型、训练器、经验池等）
    bool initialize();

    // 生成少量棋局填充经验池
    // num_games: 要生成的局数（通常2-3局）
    void generate_mini_games(int num_games);

    // 用当前参数在经验池上训练指定步数，返回平均总损失
    float train_on_pool(int steps);

    // 在经验池上评估当前模型损失（不更新权重）
    float evaluate_on_pool();

    // 同池重训：更换参数后在同一经验池上训练并测量变化率
    TuneTrial retrain_with_param(const std::string& param_name,
                                  const ParamValue& new_value,
                                  float baseline_loss);

    // ===== 参数调整策略 =====

    // 为指定参数生成一个探索值（force_direction: 0=随机, 1=增大, -1=减小）
    ParamValue explore_new_value(const std::string& param_name,
                                  const ParamValue& current_value,
                                  float current_loss,
                                  int force_direction = 0);

    // 计算float参数的探索步长（对数尺度）
    float compute_log_step(const std::string& param_name, float current_value);

    // 计算int参数的探索步长
    int compute_int_step(const std::string& param_name, int current_value);

    // ===== Phase 1: 训练参数调优 =====

    // 获取可调训练参数列表
    std::vector<std::string> get_tunable_training_params() const;

    // 对单个训练参数进行坐标下降调优
    bool tune_single_training_param(ParamTuneState& state);

    // Phase 1 主循环：依次调优所有训练参数
    void run_training_params_phase();

    // ===== Phase 2: 搜索参数评估 =====

    // 获取可调搜索参数列表
    std::vector<std::string> get_tunable_search_params() const;

    // 用收敛网络评估搜索参数好坏
    // 生成几局棋，通过策略损失/价值损失/ELO评估搜索质量
    float evaluate_search_param(const std::string& param_name,
                                 const ParamValue& test_value);

    // Phase 2 主循环：评估搜索参数
    void run_search_params_phase();

    // ===== 工具函数 =====

    // 保存当前模型快照（用于回滚）
    void save_model_snapshot();

    // 回滚到上次快照
    void rollback_model();

    // 将当前ParamRegistry同步到config
    void sync_registry_to_config();

    // 从config同步到ParamRegistry
    void sync_config_to_registry();

    // 打印当前调参状态摘要
    void print_status(const std::string& phase, int current, int total) const;

    // 保存调参报告
    void save_report() const;

    // 重置优化器（参数变化后需要重建）
    void reset_optimizer();

    // 获取参数的float值
    float get_float_param(const std::string& name) const;

    // 获取参数的int值
    int get_int_param(const std::string& name) const;

    // 设置参数并记录
    void set_param(const std::string& name, const ParamValue& value);

    // 模型快照存储
    std::vector<torch::Tensor> model_snapshot_params_;
    std::vector<torch::Tensor> model_snapshot_buffers_;
};

} // namespace training
} // namespace go
} // namespace spacego

#endif // USE_LIBTORCH