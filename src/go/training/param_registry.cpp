#include "go/training/param_registry.h"
#include "go/training/unsupervised_self_reinforcement.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace spacego {
namespace go {
namespace training {

ParamRegistry& ParamRegistry::instance() {
    static ParamRegistry registry;
    return registry;
}

void ParamRegistry::register_param(const std::string& name, const std::string& group,
                                    const std::string& description,
                                    const ParamValue& default_value,
                                    const ParamValue& min_value,
                                    const ParamValue& max_value,
                                    const std::vector<ParamValue>& recommended_values,
                                    bool tunable) {
    ParamMeta meta;
    meta.name = name;
    meta.group = group;
    meta.description = description;
    meta.default_value = default_value;
    meta.min_value = min_value;
    meta.max_value = max_value;
    meta.recommended_values = recommended_values;
    meta.tunable = tunable;
    metas_[name] = meta;
    // 如果尚未设置值，则用默认值初始化
    if (values_.find(name) == values_.end()) {
        values_[name] = default_value;
    }
}

bool ParamRegistry::set(const std::string& name, const ParamValue& value) {
    auto it = metas_.find(name);
    if (it == metas_.end()) return false;
    // 检查范围约束
    if (!it->second.is_in_range(value)) {
        std::cerr << "[ParamRegistry] 参数 " << name << " 的值超出范围" << std::endl;
        return false;
    }
    values_[name] = value;
    notify_change(name, value);
    return true;
}

ParamValue ParamRegistry::get(const std::string& name) const {
    auto it = values_.find(name);
    if (it != values_.end()) return it->second;
    auto mit = metas_.find(name);
    if (mit != metas_.end()) return mit->second.default_value;
    return 0; // 未注册参数返回0
}

const ParamMeta* ParamRegistry::get_meta(const std::string& name) const {
    auto it = metas_.find(name);
    if (it != metas_.end()) return &it->second;
    return nullptr;
}

void ParamRegistry::on_change(ParamChangeCallback callback) {
    callbacks_.push_back(std::move(callback));
}

void ParamRegistry::notify_change(const std::string& name, const ParamValue& value) {
    for (auto& cb : callbacks_) {
        cb(name, value);
    }
}

void ParamRegistry::apply_to_config(UnsupervisedTrainingConfig& config) const {
    // 自我对弈配置
    auto try_set_int = [&](const std::string& name, int& target) {
        auto v = get(name);
        if (auto* p = std::get_if<int>(&v)) target = *p;
    };
    auto try_set_float = [&](const std::string& name, float& target) {
        auto v = get(name);
        if (auto* p = std::get_if<float>(&v)) target = *p;
    };
    auto try_set_bool = [&](const std::string& name, bool& target) {
        auto v = get(name);
        if (auto* p = std::get_if<bool>(&v)) target = *p;
    };
    auto try_set_string = [&](const std::string& name, std::string& target) {
        auto v = get(name);
        if (auto* p = std::get_if<std::string>(&v)) target = *p;
    };

    // === self_play ===
    try_set_int("self_play.num_workers", config.self_play.num_workers);
    try_set_int("self_play.simulations_per_move", config.self_play.simulations_per_move);
    try_set_int("self_play.batch_size", config.self_play.batch_size);
    try_set_float("self_play.temperature", config.self_play.temperature);
    try_set_float("self_play.temperature_decay", config.self_play.temperature_decay);
    try_set_float("self_play.min_temperature", config.self_play.min_temperature);
    try_set_float("self_play.dirichlet_alpha", config.self_play.dirichlet_alpha);
    try_set_float("self_play.dirichlet_weight", config.self_play.dirichlet_weight);
    try_set_float("self_play.virtual_loss", config.self_play.virtual_loss);
    try_set_bool("self_play.use_cuda", config.self_play.use_cuda);
    try_set_int("self_play.cuda_device_id", config.self_play.cuda_device_id);
    try_set_bool("self_play.use_mixed_precision", config.self_play.use_mixed_precision);
    try_set_int("self_play.max_game_length", config.self_play.max_game_length);
    try_set_float("self_play.resign_threshold", config.self_play.resign_threshold);
    try_set_int("self_play.seed", config.self_play.seed);
    try_set_bool("self_play.save_sgf_games", config.self_play.save_sgf_games);
    try_set_int("self_play.sgf_save_interval", config.self_play.sgf_save_interval);
    try_set_bool("self_play.prestore_augmentation", config.self_play.prestore_augmentation);

    // === training ===
    try_set_int("training.batch_size", config.training.batch_size);
    try_set_int("training.games_per_iteration", config.training.games_per_iteration);
    try_set_int("training.num_epochs_per_iteration", config.training.num_epochs_per_iteration);
    try_set_int("training.generation_train_ratio", config.training.generation_train_ratio);
    try_set_float("training.learning_rate", config.training.learning_rate);
    try_set_float("training.lr_decay_rate", config.training.lr_decay_rate);
    try_set_float("training.weight_decay", config.training.weight_decay);
    try_set_float("training.policy_loss_weight", config.training.policy_loss_weight);
    try_set_float("training.value_loss_weight", config.training.value_loss_weight);
    try_set_float("training.gating_loss_weight", config.training.gating_loss_weight);
    try_set_float("training.entropy_regularization", config.training.entropy_regularization);
    try_set_bool("training.use_mixed_precision", config.training.use_mixed_precision);
    try_set_bool("training.gradient_clipping", config.training.gradient_clipping);
    try_set_float("training.max_gradient_norm", config.training.max_gradient_norm);
    try_set_int("training.window_size", config.training.window_size);
    try_set_bool("training.progressive_expansion", config.training.progressive_expansion);
    try_set_int("training.initial_window_size", config.training.initial_window_size);
    try_set_int("training.max_window_size", config.training.max_window_size);
    try_set_float("training.expansion_factor", config.training.expansion_factor);
    try_set_int("training.expansion_interval", config.training.expansion_interval);
    try_set_string("training.optimizer_type", config.training.optimizer_type);
    try_set_float("training.label_smoothing", config.training.label_smoothing);
    try_set_bool("training.use_data_augmentation", config.training.use_data_augmentation);
    try_set_float("training.dropout_rate", config.training.dropout_rate);
    try_set_float("training.validation_ratio", config.training.validation_ratio);

    // === evaluation ===
    try_set_int("evaluation.evaluation_interval", config.evaluation.evaluation_interval);
    try_set_int("evaluation.num_evaluation_games", config.evaluation.num_evaluation_games);
    try_set_float("evaluation.exploration_factor", config.evaluation.exploration_factor);
    try_set_bool("evaluation.use_baseline_model", config.evaluation.use_baseline_model);
    try_set_int("evaluation.max_concurrent_games", config.evaluation.max_concurrent_games);
    try_set_int("evaluation.max_eval_game_length", config.evaluation.max_eval_game_length);
    try_set_float("evaluation.komi", config.evaluation.komi);

    // === 顶层配置 ===
    try_set_int("total_iterations", config.total_iterations);
    try_set_int("checkpoint_interval", config.checkpoint_interval);
    try_set_int("log_interval", config.log_interval);
    try_set_string("checkpoint_dir", config.checkpoint_dir);
    try_set_string("log_dir", config.log_dir);
    try_set_string("model_name", config.model_name);
    try_set_int("seed", config.seed);
    try_set_int("num_early_stopping_iterations", config.num_early_stopping_iterations);
    try_set_float("min_improvement_threshold", config.min_improvement_threshold);
    try_set_int("validation_early_stop_patience", config.validation_early_stop_patience);
}

void ParamRegistry::load_from_config(const UnsupervisedTrainingConfig& config) {
    // === self_play ===
    set("self_play.num_workers", config.self_play.num_workers);
    set("self_play.simulations_per_move", config.self_play.simulations_per_move);
    set("self_play.batch_size", config.self_play.batch_size);
    set("self_play.temperature", config.self_play.temperature);
    set("self_play.temperature_decay", config.self_play.temperature_decay);
    set("self_play.min_temperature", config.self_play.min_temperature);
    set("self_play.dirichlet_alpha", config.self_play.dirichlet_alpha);
    set("self_play.dirichlet_weight", config.self_play.dirichlet_weight);
    set("self_play.virtual_loss", config.self_play.virtual_loss);
    set("self_play.use_cuda", config.self_play.use_cuda);
    set("self_play.cuda_device_id", config.self_play.cuda_device_id);
    set("self_play.use_mixed_precision", config.self_play.use_mixed_precision);
    set("self_play.max_game_length", config.self_play.max_game_length);
    set("self_play.resign_threshold", config.self_play.resign_threshold);
    set("self_play.seed", config.self_play.seed);
    set("self_play.save_sgf_games", config.self_play.save_sgf_games);
    set("self_play.sgf_save_interval", config.self_play.sgf_save_interval);
    set("self_play.prestore_augmentation", config.self_play.prestore_augmentation);

    // === training ===
    set("training.batch_size", config.training.batch_size);
    set("training.games_per_iteration", config.training.games_per_iteration);
    set("training.num_epochs_per_iteration", config.training.num_epochs_per_iteration);
    set("training.generation_train_ratio", config.training.generation_train_ratio);
    set("training.learning_rate", config.training.learning_rate);
    set("training.lr_decay_rate", config.training.lr_decay_rate);
    set("training.weight_decay", config.training.weight_decay);
    set("training.policy_loss_weight", config.training.policy_loss_weight);
    set("training.value_loss_weight", config.training.value_loss_weight);
    set("training.gating_loss_weight", config.training.gating_loss_weight);
    set("training.entropy_regularization", config.training.entropy_regularization);
    set("training.use_mixed_precision", config.training.use_mixed_precision);
    set("training.gradient_clipping", config.training.gradient_clipping);
    set("training.max_gradient_norm", config.training.max_gradient_norm);
    set("training.window_size", config.training.window_size);
    set("training.progressive_expansion", config.training.progressive_expansion);
    set("training.initial_window_size", config.training.initial_window_size);
    set("training.max_window_size", config.training.max_window_size);
    set("training.expansion_factor", config.training.expansion_factor);
    set("training.expansion_interval", config.training.expansion_interval);
    set("training.optimizer_type", config.training.optimizer_type);
    set("training.label_smoothing", config.training.label_smoothing);
    set("training.use_data_augmentation", config.training.use_data_augmentation);
    set("training.dropout_rate", config.training.dropout_rate);
    set("training.validation_ratio", config.training.validation_ratio);

    // === evaluation ===
    set("evaluation.evaluation_interval", config.evaluation.evaluation_interval);
    set("evaluation.num_evaluation_games", config.evaluation.num_evaluation_games);
    set("evaluation.exploration_factor", config.evaluation.exploration_factor);
    set("evaluation.use_baseline_model", config.evaluation.use_baseline_model);
    set("evaluation.max_concurrent_games", config.evaluation.max_concurrent_games);
    set("evaluation.max_eval_game_length", config.evaluation.max_eval_game_length);
    set("evaluation.komi", config.evaluation.komi);

    // === 顶层配置 ===
    set("total_iterations", config.total_iterations);
    set("checkpoint_interval", config.checkpoint_interval);
    set("log_interval", config.log_interval);
    set("checkpoint_dir", config.checkpoint_dir);
    set("log_dir", config.log_dir);
    set("model_name", config.model_name);
    set("seed", config.seed);
    set("num_early_stopping_iterations", config.num_early_stopping_iterations);
    set("min_improvement_threshold", config.min_improvement_threshold);
    set("validation_early_stop_patience", config.validation_early_stop_patience);
}

bool ParamRegistry::save_to_file(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) return false;

    file << "# SpaceGo 维空弈 - 参数配置文件\n";
    file << "# 由ParamRegistry自动生成\n\n";

    // 按分组输出
    auto groups = get_all_groups();
    for (const auto& group : groups) {
        file << "# ========== " << group << " ==========\n";
        auto names = get_names_by_group(group);
        for (const auto& name : names) {
            auto it = metas_.find(name);
            auto vit = values_.find(name);
            if (it == metas_.end() || vit == values_.end()) continue;

            file << "# " << it->second.description << "\n";
            // 输出值
            std::visit([&file](const auto& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, bool>) {
                    file << (v ? "true" : "false");
                } else if constexpr (std::is_same_v<T, float>) {
                    file << std::setprecision(8) << v;
                } else {
                    file << v;
                }
            }, vit->second);
            file << "\n";

            // 输出参数名=值
            file << name << "=";
            std::visit([&file](const auto& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, bool>) {
                    file << (v ? "true" : "false");
                } else if constexpr (std::is_same_v<T, float>) {
                    file << std::setprecision(8) << v;
                } else {
                    file << v;
                }
            }, vit->second);
            file << "\n\n";
        }
    }

    file.close();
    return true;
}

bool ParamRegistry::load_from_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return false;

    std::string line;
    while (std::getline(file, line)) {
        // 跳过注释和空行
        if (line.empty() || line[0] == '#') continue;

        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;

        std::string key = line.substr(0, eq_pos);
        std::string value = line.substr(eq_pos + 1);

        // 去除首尾空白
        auto trim = [](std::string& s) {
            s.erase(0, s.find_first_not_of(" \t"));
            s.erase(s.find_last_not_of(" \t") + 1);
        };
        trim(key);
        trim(value);

        // 查找参数元信息，确定类型后设置值
        auto it = metas_.find(key);
        if (it == metas_.end()) continue;

        const auto& meta = it->second;
        ParamValue new_val;

        // 根据默认值类型推断参数类型并解析
        if (std::holds_alternative<int>(meta.default_value)) {
            new_val = std::stoi(value);
        } else if (std::holds_alternative<float>(meta.default_value)) {
            new_val = std::stof(value);
        } else if (std::holds_alternative<bool>(meta.default_value)) {
            new_val = (value == "true" || value == "1");
        } else {
            new_val = value;
        }

        set(key, new_val);
    }

    file.close();
    return true;
}

std::vector<std::string> ParamRegistry::get_all_names() const {
    std::vector<std::string> names;
    for (const auto& [k, _] : metas_) {
        names.push_back(k);
    }
    return names;
}

std::vector<std::string> ParamRegistry::get_names_by_group(const std::string& group) const {
    std::vector<std::string> names;
    for (const auto& [k, v] : metas_) {
        if (v.group == group) names.push_back(k);
    }
    return names;
}

std::vector<std::string> ParamRegistry::get_all_groups() const {
    std::vector<std::string> groups;
    for (const auto& [_, v] : metas_) {
        if (std::find(groups.begin(), groups.end(), v.group) == groups.end()) {
            groups.push_back(v.group);
        }
    }
    return groups;
}

void ParamRegistry::print_all() const {
    auto groups = get_all_groups();
    for (const auto& group : groups) {
        print_group(group);
    }
}

void ParamRegistry::print_group(const std::string& group) const {
    std::cout << "\n=== " << group << " ===\n";
    auto names = get_names_by_group(group);

    // 计算最大参数名宽度
    size_t max_name_width = 0;
    for (const auto& name : names) {
        max_name_width = std::max(max_name_width, name.size());
    }
    max_name_width = std::min(max_name_width, size_t(40));

    for (const auto& name : names) {
        auto it = metas_.find(name);
        auto vit = values_.find(name);
        if (it == metas_.end() || vit == values_.end()) continue;

        const auto& meta = it->second;
        std::cout << "  " << std::left << std::setw(static_cast<int>(max_name_width) + 2)
                  << name << "= ";

        // 输出当前值
        std::visit([](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, bool>) {
                std::cout << (v ? "true" : "false");
            } else if constexpr (std::is_same_v<T, float>) {
                std::cout << std::fixed << std::setprecision(6) << v;
            } else {
                std::cout << v;
            }
        }, vit->second);

        // 输出范围和描述
        bool has_range = false;
        if (std::holds_alternative<int>(meta.min_value) && std::holds_alternative<int>(meta.max_value)) {
            int mn = std::get<int>(meta.min_value);
            int mx = std::get<int>(meta.max_value);
            if (mn != mx) {
                std::cout << "  [" << mn << "~" << mx << "]";
                has_range = true;
            }
        } else if (std::holds_alternative<float>(meta.min_value) && std::holds_alternative<float>(meta.max_value)) {
            float mn = std::get<float>(meta.min_value);
            float mx = std::get<float>(meta.max_value);
            if (std::abs(mn - mx) > 1e-9f) {
                std::cout << "  [" << mn << "~" << mx << "]";
                has_range = true;
            }
        }

        std::cout << "  # " << meta.description;
        if (!meta.tunable) std::cout << " [不可调]";
        std::cout << "\n";
    }
}

std::vector<ParamRegistry::TuneSuggestion> ParamRegistry::suggest(
    float policy_loss, float value_loss, float gating_loss,
    float val_policy_loss, float val_value_loss,
    float elo, int iteration) const {

    std::vector<TuneSuggestion> suggestions;

    // 计算训练-验证差值（过拟合指标）
    float policy_gap = val_policy_loss - policy_loss;  // 正值=验证更差=过拟合
    float value_gap = val_value_loss - value_loss;

    // === 策略损失过高 → 调整搜索和策略相关参数 ===
    if (policy_loss > 6.0f) {
        // 策略损失高说明搜索质量不足
        suggestions.push_back({"self_play.simulations_per_move",
            get("self_play.simulations_per_move"),
            std::min(std::get<int>(get("self_play.simulations_per_move")) + 400, 3200),
            "策略损失过高(" + std::to_string(policy_loss) + ")，增加搜索次数提升数据质量"});
        suggestions.push_back({"self_play.dirichlet_alpha",
            get("self_play.dirichlet_alpha"),
            std::get<float>(get("self_play.dirichlet_alpha")) * 1.5f,
            "策略损失过高，增加Dirichlet噪声提升探索"});
    }

    // === 策略损失中等 → 微调温度和噪声 ===
    if (policy_loss > 4.0f && policy_loss <= 6.0f) {
        suggestions.push_back({"self_play.temperature",
            get("self_play.temperature"),
            std::min(std::get<float>(get("self_play.temperature")) + 0.1f, 2.0f),
            "策略损失中等，适当提高温度增加探索多样性"});
    }

    // === 过拟合检测 → 增加正则化 ===
    if (policy_gap > 0.5f || value_gap > 0.1f) {
        float current_dropout = std::get<float>(get("training.dropout_rate"));
        if (current_dropout < 0.3f) {
            suggestions.push_back({"training.dropout_rate",
                get("training.dropout_rate"),
                current_dropout + 0.05f,
                "验证损失显著高于训练损失，增大Dropout防过拟合"});
        }
        float current_l2 = std::get<float>(get("training.weight_decay"));
        if (current_l2 < 0.01f) {
            suggestions.push_back({"training.weight_decay",
                get("training.weight_decay"),
                current_l2 * 2.0f,
                "检测到过拟合趋势，增大权重衰减"});
        }
        float current_label_smooth = std::get<float>(get("training.label_smoothing"));
        if (current_label_smooth < 0.2f) {
            suggestions.push_back({"training.label_smoothing",
                get("training.label_smoothing"),
                current_label_smooth + 0.05f,
                "检测到过拟合趋势，增大Label Smoothing"});
        }
    }

    // === 价值损失过高 → 调整价值损失权重和搜索深度 ===
    if (value_loss > 0.5f) {
        suggestions.push_back({"training.value_loss_weight",
            get("training.value_loss_weight"),
            std::get<float>(get("training.value_loss_weight")) * 1.5f,
            "价值损失过高(" + std::to_string(value_loss) + ")，增大价值损失权重"});
    }

    // === 门控损失异常 → 调整门控权重和熵正则 ===
    if (std::abs(gating_loss) > 1.0f) {
        suggestions.push_back({"training.gating_loss_weight",
            get("training.gating_loss_weight"),
            std::get<float>(get("training.gating_loss_weight")) * 0.8f,
            "门控损失异常(" + std::to_string(gating_loss) + ")，降低门控权重稳定训练"});
    }

    // === 学习率衰减 ===
    if (iteration > 0 && iteration % 50 == 0) {
        float current_lr = std::get<float>(get("training.learning_rate"));
        if (current_lr > 1e-5f) {
            suggestions.push_back({"training.learning_rate",
                get("training.learning_rate"),
                current_lr * std::get<float>(get("training.lr_decay_rate")),
                "已达" + std::to_string(iteration) + "次迭代，按衰减率降低学习率"});
        }
    }

    // === 损失下降缓慢 → 增大学习率 ===
    if (policy_loss < 3.0f && value_loss < 0.2f && iteration > 20) {
        float current_entropy = std::get<float>(get("training.entropy_regularization"));
        if (current_entropy < 0.1f) {
            suggestions.push_back({"training.entropy_regularization",
                get("training.entropy_regularization"),
                current_entropy + 0.01f,
                "损失已较低，增加熵正则化避免策略过早收敛"});
        }
    }

    return suggestions;
}

void ParamRegistry::reset_defaults() {
    for (const auto& [name, meta] : metas_) {
        values_[name] = meta.default_value;
    }
}

bool ParamRegistry::reset_default(const std::string& name) {
    auto it = metas_.find(name);
    if (it == metas_.end()) return false;
    values_[name] = it->second.default_value;
    return true;
}

// ==================== 初始化所有参数注册 ====================
// 此函数在程序启动时调用，将所有可调参数注册到ParamRegistry
void initialize_param_registry() {
    auto& reg = ParamRegistry::instance();

    // ===== self_play 分组 =====
    reg.register_param("self_play.num_workers", "self_play", "并行Worker数量",
                       8, 1, 32, {1, 4, 8, 16});
    reg.register_param("self_play.simulations_per_move", "self_play", "每步MCTS模拟次数",
                       1600, 100, 10000, {400, 800, 1600, 3200});
    reg.register_param("self_play.batch_size", "self_play", "批量推理大小",
                       64, 1, 512, {16, 32, 64, 128});
    reg.register_param("self_play.temperature", "self_play", "初始温度参数",
                       1.0f, 0.0f, 5.0f, {0.5f, 1.0f, 1.5f, 2.0f});
    reg.register_param("self_play.temperature_decay", "self_play", "温度衰减率（每局）",
                       0.995f, 0.9f, 1.0f, {0.99f, 0.995f, 0.999f});
    reg.register_param("self_play.min_temperature", "self_play", "最小温度",
                       0.0f, 0.0f, 1.0f, {0.0f, 0.1f, 0.25f});
    reg.register_param("self_play.dirichlet_alpha", "self_play", "Dirichlet噪声alpha",
                       0.03f, 0.001f, 1.0f, {0.03f, 0.1f, 0.25f, 0.5f});
    reg.register_param("self_play.dirichlet_weight", "self_play", "Dirichlet噪声权重",
                       0.25f, 0.0f, 1.0f, {0.1f, 0.25f, 0.5f});
    reg.register_param("self_play.virtual_loss", "self_play", "虚拟损失（并行MCTS）",
                       3.0f, 0.0f, 10.0f, {1.0f, 3.0f, 5.0f});
    reg.register_param("self_play.use_cuda", "self_play", "是否使用CUDA加速",
                       true, false, true);
    reg.register_param("self_play.cuda_device_id", "self_play", "CUDA设备ID",
                       0, 0, 7);
    reg.register_param("self_play.use_mixed_precision", "self_play", "使用混合精度推理",
                       true, false, true);
    reg.register_param("self_play.max_game_length", "self_play", "最大对局长度",
                       600, 100, 2000, {300, 500, 600, 800});
    reg.register_param("self_play.resign_threshold", "self_play", "认输阈值",
                       -0.9f, -1.0f, 0.0f, {-0.95f, -0.9f, -0.8f});
    reg.register_param("self_play.seed", "self_play", "随机种子",
                       42, 0, 999999);
    reg.register_param("self_play.save_sgf_games", "self_play", "是否保存SGF棋谱",
                       true, false, true);
    reg.register_param("self_play.sgf_save_interval", "self_play", "每N局保存一次棋谱",
                       10, 1, 1000);
    reg.register_param("self_play.prestore_augmentation", "self_play", "预存8种对称版本",
                       true, false, true);

    // ===== training 分组 =====
    reg.register_param("training.batch_size", "training", "训练批次大小",
                       512, 16, 4096, {64, 128, 256, 512, 1024});
    reg.register_param("training.games_per_iteration", "training", "每迭代对局数",
                       100, 1, 10000, {10, 50, 100, 500});
    reg.register_param("training.num_epochs_per_iteration", "training", "每迭代训练轮数",
                       1, 1, 50, {1, 3, 5, 10});
    reg.register_param("training.generation_train_ratio", "training", "生成:训练健康比例",
                       3, 1, 10, {2, 3, 5});
    reg.register_param("training.learning_rate", "training", "初始学习率",
                       3e-4f, 1e-6f, 0.1f, {1e-4f, 3e-4f, 1e-3f, 3e-3f});
    reg.register_param("training.lr_decay_rate", "training", "学习率衰减率",
                       0.95f, 0.5f, 1.0f, {0.9f, 0.95f, 0.99f});
    reg.register_param("training.weight_decay", "training", "权重衰减（L2正则化）",
                       5e-4f, 0.0f, 0.1f, {1e-4f, 5e-4f, 1e-3f, 5e-3f});
    reg.register_param("training.policy_loss_weight", "training", "策略损失权重",
                       1.0f, 0.0f, 10.0f, {0.5f, 1.0f, 2.0f});
    reg.register_param("training.value_loss_weight", "training", "价值损失权重",
                       1.0f, 0.0f, 10.0f, {0.5f, 1.0f, 2.0f});
    reg.register_param("training.gating_loss_weight", "training", "门控损失权重",
                       0.5f, 0.0f, 5.0f, {0.1f, 0.5f, 1.0f});
    reg.register_param("training.entropy_regularization", "training", "熵正则化系数",
                       0.05f, 0.0f, 1.0f, {0.01f, 0.05f, 0.1f, 0.2f});
    reg.register_param("training.use_mixed_precision", "training", "使用混合精度训练",
                       true, false, true);
    reg.register_param("training.gradient_clipping", "training", "梯度裁剪",
                       true, false, true);
    reg.register_param("training.max_gradient_norm", "training", "最大梯度范数",
                       3.0f, 0.1f, 100.0f, {1.0f, 3.0f, 5.0f, 10.0f});
    reg.register_param("training.window_size", "training", "经验窗口大小（非渐进模式）",
                       500000, 10000, 5000000);
    reg.register_param("training.progressive_expansion", "training", "启用渐进式经验池扩张",
                       true, false, true);
    reg.register_param("training.initial_window_size", "training", "初始经验池容量",
                       50000, 1000, 500000, {10000, 50000, 100000});
    reg.register_param("training.max_window_size", "training", "最大经验池容量",
                       500000, 50000, 5000000, {100000, 500000, 1000000});
    reg.register_param("training.expansion_factor", "training", "经验池扩张倍数",
                       2.0f, 1.1f, 5.0f, {1.5f, 2.0f, 3.0f});
    reg.register_param("training.expansion_interval", "training", "扩张检查间隔（迭代）",
                       10, 1, 100, {5, 10, 20});
    reg.register_param("training.optimizer_type", "training", "优化器类型",
                       std::string("adamw"), std::string("adam"), std::string("adamw"),
                       {std::string("adam"), std::string("sgd"), std::string("adamw")});
    reg.register_param("training.label_smoothing", "training", "Label Smoothing系数",
                       0.1f, 0.0f, 0.5f, {0.0f, 0.05f, 0.1f, 0.2f});
    reg.register_param("training.use_data_augmentation", "training", "8-fold对称数据增强",
                       true, false, true);
    reg.register_param("training.dropout_rate", "training", "Dropout比率",
                       0.1f, 0.0f, 0.5f, {0.0f, 0.1f, 0.2f, 0.3f});
    reg.register_param("training.validation_ratio", "training", "验证集比例",
                       0.1f, 0.0f, 0.3f, {0.05f, 0.1f, 0.2f});

    // ===== evaluation 分组 =====
    reg.register_param("evaluation.evaluation_interval", "evaluation", "评估间隔（迭代）",
                       1, 1, 100, {1, 5, 10, 20});
    reg.register_param("evaluation.num_evaluation_games", "evaluation", "评估对局数",
                       10, 1, 1000, {10, 20, 50, 100});
    reg.register_param("evaluation.exploration_factor", "evaluation", "评估时探索因子",
                       0.0f, 0.0f, 1.0f, {0.0f, 0.1f, 0.25f});
    reg.register_param("evaluation.use_baseline_model", "evaluation", "与基准模型对比",
                       true, false, true);
    reg.register_param("evaluation.max_concurrent_games", "evaluation", "最大并发评估数",
                       1, 1, 16);
    reg.register_param("evaluation.max_eval_game_length", "evaluation", "评估对局最大步数",
                       200, 50, 600, {100, 200, 400});
    reg.register_param("evaluation.komi", "evaluation", "贴目（中国规则）",
                       7.5f, 0.0f, 10.0f, {6.5f, 7.5f});

    // ===== 顶层配置 =====
    reg.register_param("total_iterations", "general", "总迭代次数",
                       10000, 1, 1000000, {10, 100, 1000, 10000});
    reg.register_param("checkpoint_interval", "general", "检查点保存间隔",
                       10, 1, 1000, {5, 10, 50, 100});
    reg.register_param("log_interval", "general", "日志打印间隔",
                       10, 1, 1000);
    reg.register_param("checkpoint_dir", "general", "检查点目录",
                       std::string("./checkpoints"), std::string(""), std::string(""));
    reg.register_param("log_dir", "general", "日志目录",
                       std::string("./logs"), std::string(""), std::string(""));
    reg.register_param("model_name", "general", "模型名称",
                       std::string("spacego_unsupervised"), std::string(""), std::string(""));
    reg.register_param("seed", "general", "全局随机种子",
                       42, 0, 999999);
    reg.register_param("num_early_stopping_iterations", "general", "早停检查窗口",
                       20, 1, 100, {10, 20, 50});
    reg.register_param("min_improvement_threshold", "general", "最小改进阈值",
                       0.001f, 0.0001f, 0.1f, {0.001f, 0.005f, 0.01f});
    reg.register_param("validation_early_stop_patience", "general", "验证早停耐心值",
                       5, 1, 50, {3, 5, 10, 20});
}

} // namespace training
} // namespace go
} // namespace spacego