#include "go/training/auto_tuner.h"

#ifdef USE_LIBTORCH

#include "framework/amp_grad_scaler.h"
#include "go/board/go_board_cuda.h"
#include "go/neural/ca_moe_expert.h"
#include "go/search/ca_moe_enhanced_integration.h"
#include "framework/logger.h"

#include <numeric>
#include <filesystem>
#include <algorithm>
#include <cstdlib>

namespace spacego {
namespace go {
namespace training {

// ==================== 构造函数 ====================

AutoTuner::AutoTuner(const UnsupervisedTrainingConfig& base_config)
    : base_config_(base_config)
    , rng_(base_config.seed)
{
}

// ==================== 初始化 ====================

bool AutoTuner::initialize() {
    auto& logger = framework::Logger::instance();
    logger.info("AutoTuner: 初始化自拟合调参器...");

    // 初始化ParamRegistry
    initialize_param_registry();
    sync_config_to_registry();

    // 1. 创建CA-MOE模型
    std::cout << "[AutoTuner] 创建CA-MOE混合专家网络..." << std::endl;
    model_ = std::make_shared<neural::CA_MOEExpertSystem>(8, 19);
    (*model_)->set_train_mode();
    if (torch::cuda::is_available() && base_config_.self_play.use_cuda) {
        (*model_)->to(torch::kCUDA);
        std::cout << "      模型已加载到GPU" << std::endl;
    }
    std::cout << "      完成！" << std::endl;

    // 2. 创建经验池（小容量，快速迭代）
    std::cout << "[AutoTuner] 创建经验回放缓冲区..." << std::endl;
    replay_buffer_ = std::make_shared<EfficientReplayBuffer>(10000); // 小池：1万条
    std::cout << "      容量: 10000 条经验" << std::endl;
    std::cout << "      完成！" << std::endl;

    // 3. 创建训练器
    std::cout << "[AutoTuner] 创建训练器..." << std::endl;
    trainer_ = std::make_shared<MixedPrecisionTrainer>(model_, base_config_.training);
    trainer_->setup_optimizer();
    std::cout << "      完成！" << std::endl;

    // 4. 创建自我对弈生成器（只用2个Worker，快速生成）
    std::cout << "[AutoTuner] 创建自我对弈生成器..." << std::endl;
    int num_workers = std::min(2, base_config_.self_play.num_workers);
    for (int i = 0; i < num_workers; ++i) {
        auto gen = std::make_unique<SelfPlayGenerator>(
            i, base_config_.self_play, model_);
        generators_.push_back(std::move(gen));
        std::cout << "      Worker-" << i << " 初始化完成" << std::endl;
    }
    std::cout << "      完成！共创建 " << num_workers << " 个Worker" << std::endl;

    // 5. 准备输出目录
    std::filesystem::create_directories(output_dir_);
    std::filesystem::create_directories(output_dir_ + "/checkpoints");
    std::filesystem::create_directories(output_dir_ + "/logs");

    // 6. 初始化调优状态
    auto training_params = get_tunable_training_params();
    for (const auto& name : training_params) {
        ParamTuneState state;
        state.name = name;
        state.best_value = ParamRegistry::instance().get(name);
        training_param_states_.push_back(std::move(state));
    }

    auto search_params = get_tunable_search_params();
    for (const auto& name : search_params) {
        ParamTuneState state;
        state.name = name;
        state.best_value = ParamRegistry::instance().get(name);
        search_param_states_.push_back(std::move(state));
    }

    std::cout << "\n[AutoTuner] 初始化完成！" << std::endl;
    std::cout << "  可调训练参数: " << training_param_states_.size() << " 个" << std::endl;
    std::cout << "  可调搜索参数: " << search_param_states_.size() << " 个" << std::endl;
    std::cout << "  FP8收敛阈值: " << FP8_RELATIVE_THRESHOLD * 100 << "%" << std::endl;
    std::cout << "  同池重训步数: " << RETRAIN_STEPS << std::endl;

    return true;
}

// ==================== 核心流程 ====================

void AutoTuner::generate_mini_games(int num_games) {
    // 生成少量棋局填充经验池
    int games_per_worker = std::max(1, num_games / static_cast<int>(generators_.size()));
    int total_generated = 0;

    for (size_t i = 0; i < generators_.size(); ++i) {
        int start_id = total_generated;
        auto games = generators_[i]->generate_games(start_id, games_per_worker);

        if (!games.empty()) {
            replay_buffer_->add_games(games);
            total_generated += static_cast<int>(games.size());
        }
    }
}

float AutoTuner::train_on_pool(int steps) {
    // 在经验池上训练指定步数
    // 返回无权重损失（policy+value+gating的简单求和），避免改变loss权重时总损失虚假下降
    if (!replay_buffer_->is_ready()) {
        return 999.0f;
    }

    float unweighted_loss = 0.0f;
    int valid_steps = 0;

    for (int step = 0; step < steps; ++step) {
        auto batch = replay_buffer_->sample_batch(base_config_.training.batch_size);
        if (batch.states.size(0) == 0) continue;

        auto result = trainer_->train_step(batch);
        // 使用无权重求和：policy_loss + value_loss + gating_loss
        // 这样改变loss权重不会虚假地降低衡量指标
        unweighted_loss += (result.policy_loss + result.value_loss + result.gating_loss);
        valid_steps++;
    }

    return valid_steps > 0 ? unweighted_loss / valid_steps : 999.0f;
}

float AutoTuner::evaluate_on_pool() {
    // 在经验池上评估无权重损失（不更新权重）
    if (!replay_buffer_->is_ready()) {
        return 999.0f;
    }

    auto val_result = trainer_->evaluate_validation(*replay_buffer_);
    // 返回无权重求和，与train_on_pool保持一致
    return val_result.avg_policy_loss + val_result.avg_value_loss + val_result.avg_gating_loss;
}

TuneTrial AutoTuner::retrain_with_param(const std::string& param_name,
                                          const ParamValue& new_value,
                                          float baseline_loss) {
    TuneTrial trial;
    trial.param_name = param_name;
    trial.old_value = ParamRegistry::instance().get(param_name);
    trial.new_value = new_value;
    trial.loss_before = baseline_loss;
    trial.direction = "explore";
    trial.accepted = false;
    trial.improvement = 0.0f;

    auto start_time = std::chrono::steady_clock::now();

    try {
        // 1. 保存当前模型快照
        save_model_snapshot();

        // 2. 应用新参数
        set_param(param_name, new_value);
        sync_registry_to_config();

        // 如果是训练参数（如loss权重、lr等），需要重置优化器
        reset_optimizer();

        // 3. 用同一经验池训练
        trial.loss_after = train_on_pool(RETRAIN_STEPS);

        // 4. 计算改善率
        if (baseline_loss > 1e-6f) {
            trial.improvement = (baseline_loss - trial.loss_after) / baseline_loss;
        } else {
            trial.improvement = 0.0f;
        }

        // 5. 判断是否接受此变更
        trial.accepted = trial.improvement > 0.0f;

        if (!trial.accepted) {
            // 回滚模型和参数
            rollback_model();
            set_param(param_name, trial.old_value);
            sync_registry_to_config();
            reset_optimizer();
        }
    } catch (const std::exception& e) {
        // CUDA/训练异常时强制回滚
        std::cerr << "      [警告] retrain异常: " << e.what() << std::endl;
        trial.accepted = false;
        trial.loss_after = baseline_loss;
        trial.improvement = 0.0f;
        try {
            rollback_model();
            set_param(param_name, trial.old_value);
            sync_registry_to_config();
            reset_optimizer();
        } catch (...) {
            // 回滚也失败，只能继续
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    trial.elapsed_sec = std::chrono::duration<float>(end_time - start_time).count();

    return trial;
}

// ==================== 参数调整策略 ====================

ParamValue AutoTuner::explore_new_value(const std::string& param_name,
                                          const ParamValue& current_value,
                                          float current_loss,
                                          int force_direction) {
    auto* meta = ParamRegistry::instance().get_meta(param_name);
    if (!meta) return current_value;

    // 根据参数类型选择不同的探索策略
    if (std::holds_alternative<float>(current_value)) {
        float cur = std::get<float>(current_value);
        float step = compute_log_step(param_name, cur);

        // 对数尺度探索：乘或除一个因子
        // 对于损失高的情况用较大步长，损失低时用较小步长
        float adaptive_factor = 1.0f;
        if (current_loss > 6.0f) {
            adaptive_factor = 2.0f;  // 损失很高，大胆探索
        } else if (current_loss > 4.0f) {
            adaptive_factor = 1.5f;
        } else if (current_loss > 2.0f) {
            adaptive_factor = 1.3f;  // 中等损失
        } else {
            adaptive_factor = 1.1f;  // 损失较低，谨慎探索
        }

        float min_val = std::get<float>(meta->min_value);
        float max_val = std::get<float>(meta->max_value);

        // 确定方向：1=增大, -1=减小, 0=随机
        int direction = force_direction;
        if (direction == 0) {
            std::uniform_int_distribution<int> dir_dist(0, 1);
            direction = dir_dist(rng_) == 0 ? 1 : -1;
        }

        float new_val;
        if (direction > 0) {
            new_val = cur * (1.0f + step * adaptive_factor);
        } else {
            new_val = cur / (1.0f + step * adaptive_factor);
        }

        // 限制范围
        new_val = std::max(min_val, std::min(max_val, new_val));
        return new_val;

    } else if (std::holds_alternative<int>(current_value)) {
        int cur = std::get<int>(current_value);
        int step = compute_int_step(param_name, cur);
        int min_val = std::get<int>(meta->min_value);
        int max_val = std::get<int>(meta->max_value);

        int direction = force_direction;
        if (direction == 0) {
            std::uniform_int_distribution<int> dir_dist(0, 1);
            direction = dir_dist(rng_) == 0 ? 1 : -1;
        }

        int new_val;
        if (direction > 0) {
            new_val = cur + step;
        } else {
            new_val = cur - step;
        }

        new_val = std::max(min_val, std::min(max_val, new_val));
        return new_val;

    } else if (std::holds_alternative<bool>(current_value)) {
        // 布尔参数：取反
        return !std::get<bool>(current_value);
    }

    return current_value;
}

float AutoTuner::compute_log_step(const std::string& param_name, float current_value) {
    // 对数尺度的步长：小值用相对步长，大值用绝对步长
    if (current_value < 0.01f) {
        return 0.5f;  // 很小的值：50%步长
    } else if (current_value < 0.1f) {
        return 0.3f;  // 小值：30%步长
    } else if (current_value < 1.0f) {
        return 0.2f;  // 中等值：20%步长
    } else {
        return 0.1f;  // 大值：10%步长
    }
}

int AutoTuner::compute_int_step(const std::string& param_name, int current_value) {
    if (current_value < 10) return 1;
    if (current_value < 100) return std::max(1, current_value / 10);
    return std::max(1, current_value / 5);
}

// ==================== Phase 1: 训练参数调优 ====================

std::vector<std::string> AutoTuner::get_tunable_training_params() const {
    // 可调训练参数（按优先级排序：先调影响最大的）
    return {
        "training.value_loss_weight",       // 价值损失权重（上次训练问题最大）
        "training.learning_rate",           // 学习率
        "training.policy_loss_weight",      // 策略损失权重
        "training.gating_loss_weight",      // 门控损失权重
        "training.dropout_rate",            // Dropout率
        "training.weight_decay",            // 权重衰减
        "training.label_smoothing",         // 标签平滑
        "training.entropy_regularization",  // 熵正则化
        "training.max_gradient_norm",       // 梯度裁剪阈值
    };
}

bool AutoTuner::tune_single_training_param(ParamTuneState& state) {
    if (state.converged) return true;

    ParamValue current_val = ParamRegistry::instance().get(state.name);
    float current_loss = evaluate_on_pool();

    // 如果经验池不足，先生成棋局
    if (current_loss > 990.0f) {
        std::cout << "    经验池不足，生成棋局..." << std::endl;
        generate_mini_games(2);
        current_loss = train_on_pool(RETRAIN_STEPS); // 先训练一下建立基线
    }

    // 第一次探索：随机方向
    ParamValue new_val = explore_new_value(state.name, current_val, current_loss, 0);

    // 如果新值和旧值相同，跳过
    if (new_val == current_val) {
        state.consecutive_no_improve++;
        if (state.consecutive_no_improve >= MAX_NO_IMPROVE) {
            state.converged = true;
            return true;
        }
        return false;
    }

    // 同池重训
    auto trial = retrain_with_param(state.name, new_val, current_loss);
    history_.push_back(trial);

    state.total_trials++;

    // 打印试验结果
    std::cout << "    [" << state.name << "] ";
    std::visit([](const auto& v) { std::cout << v; }, trial.old_value);
    std::cout << " → ";
    std::visit([](const auto& v) { std::cout << v; }, trial.new_value);
    std::cout << " | 损失: " << std::fixed << std::setprecision(4)
              << trial.loss_before << " → " << trial.loss_after
              << " | 改善: " << (trial.improvement * 100) << "%"
              << " | " << (trial.accepted ? "✓ 接受" : "✗ 回滚")
              << " | " << trial.elapsed_sec << "s"
              << std::endl;

    if (trial.accepted) {
        state.consecutive_no_improve = 0;
        if (trial.improvement > state.best_improvement) {
            state.best_improvement = trial.improvement;
            state.best_value = new_val;
        }
        // 记住成功的方向
        state.last_explored_delta = trial.improvement;
    } else {
        state.consecutive_no_improve++;

        // 第一次失败后，尝试反方向探索（如果当前损失足够高值得继续）
        if (current_loss > 1.5f && state.consecutive_no_improve < MAX_NO_IMPROVE) {
            // 判断第一次尝试的方向，反方向再试
            float old_f = 0, new_f = 0;
            if (std::holds_alternative<float>(trial.old_value) &&
                std::holds_alternative<float>(trial.new_value)) {
                old_f = std::get<float>(trial.old_value);
                new_f = std::get<float>(trial.new_value);
            }
            int reverse_dir = (new_f > old_f) ? -1 : 1;

            ParamValue reverse_val = explore_new_value(state.name, current_val, current_loss, reverse_dir);
            if (reverse_val != current_val && reverse_val != new_val) {
                auto trial2 = retrain_with_param(state.name, reverse_val, current_loss);
                history_.push_back(trial2);
                state.total_trials++;

                std::cout << "    [" << state.name << "] ";
                std::visit([](const auto& v) { std::cout << v; }, trial2.old_value);
                std::cout << " → ";
                std::visit([](const auto& v) { std::cout << v; }, trial2.new_value);
                std::cout << " | 损失: " << std::fixed << std::setprecision(4)
                          << trial2.loss_before << " → " << trial2.loss_after
                          << " | 改善: " << (trial2.improvement * 100) << "%"
                          << " | " << (trial2.accepted ? "✓ 接受" : "✗ 回滚")
                          << " | (反方向)"
                          << std::endl;

                if (trial2.accepted) {
                    state.consecutive_no_improve = 0;
                    if (trial2.improvement > state.best_improvement) {
                        state.best_improvement = trial2.improvement;
                        state.best_value = reverse_val;
                    }
                } else {
                    state.consecutive_no_improve++;
                }
            }
        }
    }

    // 检查收敛：连续多次无改善 或 改善率低于FP8精度
    bool no_improve_converged = state.consecutive_no_improve >= MAX_NO_IMPROVE;
    bool precision_converged = (trial.accepted &&
        trial.improvement < FP8_RELATIVE_THRESHOLD);

    if (no_improve_converged || precision_converged) {
        state.converged = true;
        std::cout << "    [" << state.name << "] 已收敛";
        if (precision_converged) {
            std::cout << "（改善率 " << (trial.improvement * 100)
                      << "% < FP8阈值 " << (FP8_RELATIVE_THRESHOLD * 100) << "%）";
        }
        std::cout << " → 最优值: ";
        std::visit([](const auto& v) { std::cout << v; }, state.best_value);
        std::cout << std::endl;
        return true;
    }

    return false;
}

void AutoTuner::run_training_params_phase() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Phase 1: 训练参数自拟合调优" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "策略：同池重训 → 测变化率 → 自动换参" << std::endl;
    std::cout << "收敛条件：改善率 < " << (FP8_RELATIVE_THRESHOLD * 100)
              << "% 或连续" << MAX_NO_IMPROVE << "次无改善" << std::endl;
    std::cout << std::endl;

    // 先生成初始棋局
    std::cout << "[Phase 1] 生成初始棋局（2局）..." << std::endl;
    generate_mini_games(2);
    std::cout << "  经验池大小: " << replay_buffer_->size() << std::endl;

    // 先训练建立基线
    std::cout << "[Phase 1] 建立基线损失..." << std::endl;
    float baseline_loss = train_on_pool(RETRAIN_STEPS);
    std::cout << "  基线总损失: " << std::fixed << std::setprecision(4)
              << baseline_loss << std::endl;

    int total_params = static_cast<int>(training_param_states_.size());
    int round = 0;
    int max_rounds = 15; // 最多15轮全局调优（充分收敛）
    int consecutive_no_improve_rounds = 0; // 连续无改善轮数

    while (round < max_rounds) {
        round++;
        std::cout << "\n--- Phase 1, 第 " << round << "/" << max_rounds
                  << " 轮 ---" << std::endl;

        bool all_converged = true;
        bool any_improved_this_round = false;
        int current_idx = 0;

        for (auto& state : training_param_states_) {
            current_idx++;
            print_status("Phase 1", current_idx, total_params);

            if (!state.converged) {
                all_converged = false;

                // 记录本轮改善情况
                float prev_best = state.best_improvement;

                // 调参（内部已包含双方向探索）
                tune_single_training_param(state);

                // 检查本轮是否有改善
                if (state.best_improvement > prev_best + FP8_RELATIVE_THRESHOLD) {
                    any_improved_this_round = true;
                }
            }
        }

        // 连续无改善计数
        if (any_improved_this_round) {
            consecutive_no_improve_rounds = 0;
        } else {
            consecutive_no_improve_rounds++;
        }

        // 每轮结束后重新生成棋局并训练
        if (!all_converged) {
            std::cout << "\n[Phase 1] 第" << round << "轮完成，重新生成棋局..." << std::endl;
            // 清空经验池，重新生成
            replay_buffer_ = std::make_shared<EfficientReplayBuffer>(10000);
            generate_mini_games(2);
            baseline_loss = train_on_pool(RETRAIN_STEPS);
            std::cout << "  新基线损失: " << std::fixed << std::setprecision(4)
                      << baseline_loss << std::endl;

            // 重置连续无改善计数（新数据上重新评估）
            for (auto& state : training_param_states_) {
                if (!state.converged) {
                    state.consecutive_no_improve = 0;
                }
            }
        }

        if (all_converged) {
            std::cout << "\n[Phase 1] 所有训练参数已收敛！" << std::endl;
            break;
        }

        // 连续2轮无任何改善，提前停止
        if (consecutive_no_improve_rounds >= 2) {
            std::cout << "\n[Phase 1] 连续" << consecutive_no_improve_rounds
                      << "轮无任何参数改善，提前停止调优" << std::endl;
            break;
        }
    }

    // 打印Phase 1结果
    std::cout << "\n========================================" << std::endl;
    std::cout << "Phase 1 完成！训练参数调优结果：" << std::endl;
    std::cout << "========================================" << std::endl;
    for (const auto& state : training_param_states_) {
        std::cout << "  " << state.name << " = ";
        std::visit([](const auto& v) { std::cout << v; }, state.best_value);
        std::cout << " (试验" << state.total_trials << "次";
        if (state.converged) std::cout << ", 已收敛";
        std::cout << ")" << std::endl;
    }
}

// ==================== Phase 2: 搜索参数评估 ====================

std::vector<std::string> AutoTuner::get_tunable_search_params() const {
    return {
        "self_play.simulations_per_move",  // 每步模拟次数
        "self_play.dirichlet_alpha",       // Dirichlet噪声
        "self_play.dirichlet_weight",      // 噪声权重
        "self_play.temperature",           // 温度
        "self_play.virtual_loss",          // 虚拟损失
    };
}

float AutoTuner::evaluate_search_param(const std::string& param_name,
                                        const ParamValue& test_value) {
    // 用收敛网络评估搜索参数好坏
    // 方法：设置搜索参数，生成2局棋，测量策略损失
    // 策略损失越低说明搜索质量越好

    ParamValue old_value = ParamRegistry::instance().get(param_name);
    set_param(param_name, test_value);
    sync_registry_to_config();

    // 临时创建一个小经验池来收集新棋局
    auto eval_buffer = std::make_shared<EfficientReplayBuffer>(5000);

    // 生成2局棋
    for (size_t i = 0; i < generators_.size(); ++i) {
        int games_per_w = 1;
        auto games = generators_[i]->generate_games(0, games_per_w);
        if (!games.empty()) {
            eval_buffer->add_games(games);
        }
    }

    // 在这个经验池上评估损失（不训练，只测损失）
    float eval_loss = 999.0f;
    if (eval_buffer->is_ready()) {
        auto val_result = trainer_->evaluate_validation(*eval_buffer);
        eval_loss = val_result.avg_policy_loss; // 用策略损失评估搜索质量
    }

    // 恢复原参数
    set_param(param_name, old_value);
    sync_registry_to_config();

    return eval_loss;
}

void AutoTuner::run_search_params_phase() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Phase 2: 搜索参数评估" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "策略：用收敛网络评估不同搜索参数的棋局质量" << std::endl;
    std::cout << std::endl;

    int total_params = static_cast<int>(search_param_states_.size());

    for (int idx = 0; idx < total_params; ++idx) {
        auto& state = search_param_states_[idx];
        print_status("Phase 2", idx + 1, total_params);

        auto* meta = ParamRegistry::instance().get_meta(state.name);
        if (!meta) continue;

        std::cout << "\n  评估搜索参数: " << state.name << std::endl;

        // 测试当前值
        ParamValue current_val = ParamRegistry::instance().get(state.name);
        float current_loss = evaluate_search_param(state.name, current_val);
        std::cout << "    当前值: ";
        std::visit([](const auto& v) { std::cout << v; }, current_val);
        std::cout << " → 策略损失: " << std::fixed << std::setprecision(4)
                  << current_loss << std::endl;

        // 测试推荐值
        float best_loss = current_loss;
        state.best_value = current_val;

        for (const auto& rec_val : meta->recommended_values) {
            if (rec_val == current_val) continue;

            float test_loss = evaluate_search_param(state.name, rec_val);
            std::cout << "    测试值: ";
            std::visit([](const auto& v) { std::cout << v; }, rec_val);
            std::cout << " → 策略损失: " << std::fixed << std::setprecision(4)
                      << test_loss << std::endl;

            if (test_loss < best_loss) {
                best_loss = test_loss;
                state.best_value = rec_val;
            }

            state.total_trials++;
        }

        // 应用最佳搜索参数
        if (state.best_value != current_val) {
            set_param(state.name, state.best_value);
            sync_registry_to_config();
            std::cout << "    ★ 更新为: ";
            std::visit([](const auto& v) { std::cout << v; }, state.best_value);
            std::cout << " (策略损失: " << best_loss << ")" << std::endl;
        } else {
            std::cout << "    当前值已是最优" << std::endl;
        }

        state.converged = true;
    }

    // 打印Phase 2结果
    std::cout << "\n========================================" << std::endl;
    std::cout << "Phase 2 完成！搜索参数评估结果：" << std::endl;
    std::cout << "========================================" << std::endl;
    for (const auto& state : search_param_states_) {
        std::cout << "  " << state.name << " = ";
        std::visit([](const auto& v) { std::cout << v; }, state.best_value);
        std::cout << std::endl;
    }
}

// ==================== 主循环 ====================

void AutoTuner::run() {
    std::cout << "\n╔══════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     SpaceGo 自拟合调参器 (AutoTuner)       ║" << std::endl;
    std::cout << "║     同池重训·变化率检测·自动换参            ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════╝" << std::endl;

    auto total_start = std::chrono::steady_clock::now();

    if (!initialize()) {
        std::cerr << "[AutoTuner] 初始化失败！" << std::endl;
        return;
    }

    // Phase 1: 训练参数自拟合（带异常保护，崩溃前保存结果）
    try {
        run_training_params_phase();
    } catch (const std::exception& e) {
        std::cerr << "\n[AutoTuner] Phase 1 异常: " << e.what() << std::endl;
        std::cerr << "[AutoTuner] 保存当前进度并继续..." << std::endl;
    }

    // Phase 2: 搜索参数评估（带异常保护）
    try {
        run_search_params_phase();
    } catch (const std::exception& e) {
        std::cerr << "\n[AutoTuner] Phase 2 异常: " << e.what() << std::endl;
        std::cerr << "[AutoTuner] 跳过Phase 2，使用默认搜索参数" << std::endl;
    }

    auto total_end = std::chrono::steady_clock::now();
    float total_sec = std::chrono::duration<float>(total_end - total_start).count();

    // 保存最终配置和报告
    std::cout << "\n========================================" << std::endl;
    std::cout << "自拟合调参完成！" << std::endl;
    std::cout << "总耗时: " << static_cast<int>(total_sec) / 60 << "m"
              << static_cast<int>(total_sec) % 60 << "s" << std::endl;
    std::cout << "总试验次数: " << history_.size() << std::endl;
    std::cout << "========================================" << std::endl;

    // 将所有调优后的最优值写回ParamRegistry
    auto& reg = ParamRegistry::instance();
    for (const auto& state : training_param_states_) {
        reg.set(state.name, state.best_value);
    }
    for (const auto& state : search_param_states_) {
        reg.set(state.name, state.best_value);
    }

    // 保存最优配置
    reg.save_to_file(output_dir_ + "/best_params.txt");
    std::cout << "最优参数已保存到: " << output_dir_ << "/best_params.txt" << std::endl;

    // 保存调参报告
    save_report();
}

// ==================== 工具函数 ====================

void AutoTuner::save_model_snapshot() {
    model_snapshot_params_.clear();
    model_snapshot_buffers_.clear();

    // 保存模型参数和缓冲区
    for (const auto& p : (*model_)->parameters()) {
        model_snapshot_params_.push_back(p.clone());
    }
    for (const auto& b : (*model_)->buffers()) {
        model_snapshot_buffers_.push_back(b.clone());
    }
}

void AutoTuner::rollback_model() {
    // 恢复模型参数
    auto params = (*model_)->parameters();
    for (size_t i = 0; i < model_snapshot_params_.size() && i < params.size(); ++i) {
        params[i].data().copy_(model_snapshot_params_[i]);
    }
    auto buffers = (*model_)->buffers();
    for (size_t i = 0; i < model_snapshot_buffers_.size() && i < buffers.size(); ++i) {
        buffers[i].data().copy_(model_snapshot_buffers_[i]);
    }
}

void AutoTuner::sync_registry_to_config() {
    ParamRegistry::instance().apply_to_config(base_config_);
}

void AutoTuner::sync_config_to_registry() {
    ParamRegistry::instance().load_from_config(base_config_);
}

void AutoTuner::print_status(const std::string& phase, int current, int total) const {
    std::cout << "\n[" << phase << " " << current << "/" << total << "] ";
}

void AutoTuner::save_report() const {
    std::string path = output_dir_ + "/logs/auto_tune_report.txt";
    std::ofstream file(path);
    if (!file.is_open()) return;

    file << "# SpaceGo 自拟合调参报告\n";
    file << "# 生成时间: " << __DATE__ << " " << __TIME__ << "\n\n";

    file << "## Phase 1: 训练参数调优结果\n\n";
    for (const auto& state : training_param_states_) {
        file << state.name << " = ";
        std::visit([&file](const auto& v) { file << v; }, state.best_value);
        file << " (试验" << state.total_trials << "次, ";
        if (state.converged) file << "已收敛";
        else file << "未收敛";
        file << ")\n";
    }

    file << "\n## Phase 2: 搜索参数评估结果\n\n";
    for (const auto& state : search_param_states_) {
        file << state.name << " = ";
        std::visit([&file](const auto& v) { file << v; }, state.best_value);
        file << " (试验" << state.total_trials << "次)\n";
    }

    file << "\n## 详细试验历史\n\n";
    for (size_t i = 0; i < history_.size(); ++i) {
        const auto& trial = history_[i];
        file << "#" << (i + 1) << " " << trial.param_name << ": ";
        std::visit([&file](const auto& v) { file << v; }, trial.old_value);
        file << " → ";
        std::visit([&file](const auto& v) { file << v; }, trial.new_value);
        file << " | 损失: " << trial.loss_before << " → " << trial.loss_after
             << " | 改善: " << (trial.improvement * 100) << "%"
             << " | " << (trial.accepted ? "接受" : "回滚")
             << " | " << trial.elapsed_sec << "s\n";
    }

    file.close();
    std::cout << "调参报告已保存到: " << path << std::endl;
}

void AutoTuner::reset_optimizer() {
    // 重建优化器（参数变化后learning_rate等可能变了）
    trainer_ = std::make_shared<MixedPrecisionTrainer>(model_, base_config_.training);
    trainer_->setup_optimizer();
}

float AutoTuner::get_float_param(const std::string& name) const {
    auto val = ParamRegistry::instance().get(name);
    if (auto* p = std::get_if<float>(&val)) return *p;
    return 0.0f;
}

int AutoTuner::get_int_param(const std::string& name) const {
    auto val = ParamRegistry::instance().get(name);
    if (auto* p = std::get_if<int>(&val)) return *p;
    return 0;
}

void AutoTuner::set_param(const std::string& name, const ParamValue& value) {
    ParamRegistry::instance().set(name, value);
}

void AutoTuner::set_initial_param(const std::string& name, const ParamValue& value) {
    set_param(name, value);
}

void AutoTuner::set_output_dir(const std::string& dir) {
    output_dir_ = dir;
}

UnsupervisedTrainingConfig AutoTuner::get_best_config() const {
    UnsupervisedTrainingConfig config = base_config_;
    ParamRegistry::instance().apply_to_config(config);
    return config;
}

} // namespace training
} // namespace go
} // namespace spacego

#endif // USE_LIBTORCH