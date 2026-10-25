#ifdef USE_LIBTORCH

#include "go/training/cli_tuner.h"
#include "go/training/unsupervised_self_reinforcement.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cctype>

namespace spacego {
namespace go {
namespace training {

CLITuner::CLITuner()
    : registry_(ParamRegistry::instance()) {
    // 初始化参数注册表
    initialize_param_registry();
    // 从默认配置初始化
    sync_config_from_registry();
}

void CLITuner::run() {
    print_banner();

    std::string line;
    while (true) {
        print_prompt();
        if (!std::getline(std::cin, line)) break;

        // 去除首尾空白
        while (!line.empty() && std::isspace(line.front())) line.erase(line.begin());
        while (!line.empty() && std::isspace(line.back())) line.pop_back();
        if (line.empty()) continue;

        // 解析命令和参数
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;

        // 将命令转为小写
        std::string cmd_lower = cmd;
        std::transform(cmd_lower.begin(), cmd_lower.end(), cmd_lower.begin(), ::tolower);

        if (cmd_lower == "quit" || cmd_lower == "exit" || cmd_lower == "q") {
            std::cout << "退出调参界面。\n";
            break;
        }
        else if (cmd_lower == "help" || cmd_lower == "h" || cmd_lower == "?") {
            cmd_help();
        }
        else if (cmd_lower == "list" || cmd_lower == "ls" || cmd_lower == "l") {
            std::string group;
            iss >> group;
            cmd_list(group);
        }
        else if (cmd_lower == "get" || cmd_lower == "g") {
            std::string name;
            iss >> name;
            cmd_get(name);
        }
        else if (cmd_lower == "set" || cmd_lower == "s") {
            std::string name, value;
            iss >> name;
            std::getline(iss, value);
            // 去除value前导空白
            while (!value.empty() && std::isspace(value.front())) value.erase(value.begin());
            while (!value.empty() && std::isspace(value.back())) value.pop_back();
            cmd_set(name, value);
        }
        else if (cmd_lower == "reset") {
            std::string name;
            iss >> name;
            if (name.empty()) {
                cmd_reset_all();
            } else {
                cmd_reset(name);
            }
        }
        else if (cmd_lower == "save") {
            std::string path;
            iss >> path;
            cmd_save(path);
        }
        else if (cmd_lower == "load") {
            std::string path;
            iss >> path;
            cmd_load(path);
        }
        else if (cmd_lower == "suggest" || cmd_lower == "auto") {
            cmd_suggest();
        }
        else if (cmd_lower == "apply") {
            cmd_apply_suggest();
        }
        else if (cmd_lower == "train" || cmd_lower == "t") {
            cmd_train();
        }
        else if (cmd_lower == "quick" || cmd_lower == "qt") {
            cmd_train_quick();
        }
        else if (cmd_lower == "config" || cmd_lower == "show") {
            cmd_show_config();
        }
        else if (cmd_lower == "diff" || cmd_lower == "d") {
            cmd_diff();
        }
        else if (cmd_lower == "search" || cmd_lower == "find") {
            std::string keyword;
            iss >> keyword;
            cmd_search(keyword);
        }
        else {
            std::cout << "未知命令: " << cmd << "  输入 'help' 查看可用命令\n";
        }
    }
}

void CLITuner::set_train_callback(TrainCallback callback) {
    train_callback_ = std::move(callback);
}

void CLITuner::update_metrics(float policy_loss, float value_loss, float gating_loss,
                               float val_policy_loss, float val_value_loss,
                               float elo, int iteration) {
    last_policy_loss_ = policy_loss;
    last_value_loss_ = value_loss;
    last_gating_loss_ = gating_loss;
    last_val_policy_loss_ = val_policy_loss;
    last_val_value_loss_ = val_value_loss;
    last_elo_ = elo;
    last_iteration_ = iteration;
}

void CLITuner::cmd_help() {
    std::cout << "\n╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║             SpaceGo 维空弈 - CLI调参界面                ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";
    std::cout << "参数查看:\n";
    std::cout << "  list [group]         列出所有/指定分组的参数 (简写: ls)\n";
    std::cout << "  get <name>           查看指定参数详情 (简写: g)\n";
    std::cout << "  search <keyword>     搜索包含关键词的参数 (简写: find)\n";
    std::cout << "  config               显示当前完整配置 (简写: show)\n";
    std::cout << "  diff                 显示与默认值的差异 (简写: d)\n\n";
    std::cout << "参数修改:\n";
    std::cout << "  set <name> <value>   设置参数值 (简写: s)\n";
    std::cout << "  reset [name]         重置指定/所有参数为默认值\n\n";
    std::cout << "配置持久化:\n";
    std::cout << "  save [path]          保存当前配置到文件\n";
    std::cout << "  load <path>          从文件加载配置\n\n";
    std::cout << "智能调参:\n";
    std::cout << "  suggest              根据训练指标给出调参建议 (简写: auto)\n";
    std::cout << "  apply                应用所有建议\n\n";
    std::cout << "训练启动:\n";
    std::cout << "  train                使用当前配置启动完整训练 (简写: t)\n";
    std::cout << "  quick                快速训练模式（5迭代×5局）(简写: qt)\n\n";
    std::cout << "其他:\n";
    std::cout << "  help                 显示此帮助信息 (简写: h)\n";
    std::cout << "  quit                 退出调参界面 (简写: q)\n\n";
    std::cout << "参数分组: self_play, training, evaluation, general\n\n";
    std::cout << "示例:\n";
    std::cout << "  set self_play.temperature 1.5    # 设置温度为1.5\n";
    std::cout << "  set training.learning_rate 0.001 # 设置学习率为0.001\n";
    std::cout << "  list training                    # 列出训练组参数\n";
    std::cout << "  suggest                          # 查看调参建议\n\n";
}

void CLITuner::cmd_list(const std::string& group) {
    if (group.empty()) {
        registry_.print_all();
    } else {
        registry_.print_group(group);
    }
}

void CLITuner::cmd_get(const std::string& name) {
    if (name.empty()) {
        std::cout << "用法: get <参数名>\n";
        return;
    }
    auto* meta = registry_.get_meta(name);
    if (!meta) {
        std::cout << "未找到参数: " << name << "\n";
        std::cout << "使用 'list' 查看所有参数，或 'search <关键词>' 搜索\n";
        return;
    }
    auto val = registry_.get(name);

    std::cout << "\n  参数: " << name << "\n";
    std::cout << "  描述: " << meta->description << "\n";
    std::cout << "  当前: " << value_to_string(val) << "\n";
    std::cout << "  默认: " << value_to_string(meta->default_value) << "\n";
    std::cout << "  分组: " << meta->group << "\n";
    std::cout << "  可调: " << (meta->tunable ? "是" : "否") << "\n";

    // 范围
    if (std::holds_alternative<int>(meta->min_value)) {
        int mn = std::get<int>(meta->min_value);
        int mx = std::get<int>(meta->max_value);
        if (mn != mx) std::cout << "  范围: [" << mn << ", " << mx << "]\n";
    } else if (std::holds_alternative<float>(meta->min_value)) {
        float mn = std::get<float>(meta->min_value);
        float mx = std::get<float>(meta->max_value);
        if (std::abs(mn - mx) > 1e-9f) std::cout << "  范围: [" << mn << ", " << mx << "]\n";
    }

    // 推荐值
    if (!meta->recommended_values.empty()) {
        std::cout << "  推荐: ";
        for (size_t i = 0; i < meta->recommended_values.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << value_to_string(meta->recommended_values[i]);
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void CLITuner::cmd_set(const std::string& name, const std::string& value) {
    if (name.empty() || value.empty()) {
        std::cout << "用法: set <参数名> <值>\n";
        std::cout << "示例: set self_play.temperature 1.5\n";
        return;
    }

    auto* meta = registry_.get_meta(name);
    if (!meta) {
        std::cout << "未找到参数: " << name << "\n";
        return;
    }
    if (!meta->tunable) {
        std::cout << "参数 " << name << " 不可在运行时修改\n";
        return;
    }

    // 根据默认值类型解析输入值
    ParamValue new_val;
    try {
        if (std::holds_alternative<int>(meta->default_value)) {
            new_val = std::stoi(value);
        } else if (std::holds_alternative<float>(meta->default_value)) {
            new_val = std::stof(value);
        } else if (std::holds_alternative<bool>(meta->default_value)) {
            new_val = (value == "true" || value == "1" || value == "yes");
        } else {
            new_val = value;
        }
    } catch (const std::exception& e) {
        std::cout << "解析值失败: " << e.what() << "\n";
        return;
    }

    if (registry_.set(name, new_val)) {
        std::cout << "  " << name << " = " << value_to_string(new_val) << "  ✓\n";
        sync_config_from_registry();
    } else {
        std::cout << "  设置失败（值可能超出范围）\n";
    }
}

void CLITuner::cmd_reset(const std::string& name) {
    if (registry_.reset_default(name)) {
        std::cout << "  " << name << " 已重置为默认值: "
                  << value_to_string(registry_.get(name)) << "\n";
        sync_config_from_registry();
    } else {
        std::cout << "未找到参数: " << name << "\n";
    }
}

void CLITuner::cmd_reset_all() {
    std::cout << "确认重置所有参数为默认值? (y/N): ";
    std::string confirm;
    std::getline(std::cin, confirm);
    if (confirm == "y" || confirm == "Y") {
        registry_.reset_defaults();
        sync_config_from_registry();
        std::cout << "  所有参数已重置为默认值 ✓\n";
    } else {
        std::cout << "  已取消\n";
    }
}

void CLITuner::cmd_save(const std::string& path) {
    std::string save_path = path.empty() ? "spacego_params.txt" : path;
    if (registry_.save_to_file(save_path)) {
        std::cout << "  配置已保存到: " << save_path << " ✓\n";
    } else {
        std::cout << "  保存失败\n";
    }
}

void CLITuner::cmd_load(const std::string& path) {
    if (path.empty()) {
        std::cout << "用法: load <配置文件路径>\n";
        return;
    }
    if (registry_.load_from_file(path)) {
        sync_config_from_registry();
        std::cout << "  配置已从 " << path << " 加载 ✓\n";
    } else {
        std::cout << "  加载失败（文件不存在或格式错误）\n";
    }
}

void CLITuner::cmd_suggest() {
    auto suggestions = registry_.suggest(
        last_policy_loss_, last_value_loss_, last_gating_loss_,
        last_val_policy_loss_, last_val_value_loss_,
        last_elo_, last_iteration_);

    if (suggestions.empty()) {
        std::cout << "\n  当前无调参建议。请先运行训练获取指标数据。\n\n";
        return;
    }

    std::cout << "\n╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║                   调参建议                              ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";

    for (size_t i = 0; i < suggestions.size(); ++i) {
        const auto& s = suggestions[i];
        std::cout << "  [" << (i + 1) << "] " << s.param_name << "\n";
        std::cout << "      当前: " << value_to_string(s.current_value)
                  << " → 建议: " << value_to_string(s.suggested_value) << "\n";
        std::cout << "      理由: " << s.reason << "\n\n";
    }
    std::cout << "  使用 'apply' 应用所有建议\n\n";
}

void CLITuner::cmd_apply_suggest() {
    auto suggestions = registry_.suggest(
        last_policy_loss_, last_value_loss_, last_gating_loss_,
        last_val_policy_loss_, last_val_value_loss_,
        last_elo_, last_iteration_);

    if (suggestions.empty()) {
        std::cout << "  当前无调参建议。\n";
        return;
    }

    int applied = 0;
    for (const auto& s : suggestions) {
        if (registry_.set(s.param_name, s.suggested_value)) {
            std::cout << "  " << s.param_name << ": "
                      << value_to_string(s.current_value) << " → "
                      << value_to_string(s.suggested_value) << " ✓\n";
            applied++;
        }
    }
    if (applied > 0) {
        sync_config_from_registry();
        std::cout << "  已应用 " << applied << " 条建议\n";
    }
}

void CLITuner::cmd_train() {
    sync_config_from_registry();
    if (train_callback_) {
        std::cout << "\n  使用当前配置启动训练...\n\n";
        int result = train_callback_(active_config_);
        if (result != 0) {
            std::cout << "\n  训练异常退出（代码: " << result << "）\n";
        }
    } else {
        std::cout << "  训练回调未设置，无法启动训练\n";
    }
}

void CLITuner::cmd_train_quick() {
    sync_config_from_registry();
    // 快速训练模式：5迭代×5局
    active_config_.total_iterations = 5;
    active_config_.training.games_per_iteration = 5;
    active_config_.evaluation.num_evaluation_games = 3;
    active_config_.checkpoint_interval = 5;
    std::cout << "\n  快速训练模式: 5迭代 × 5局 × 3评估局\n\n";
    if (train_callback_) {
        int result = train_callback_(active_config_);
        if (result != 0) {
            std::cout << "\n  训练异常退出（代码: " << result << "）\n";
        }
    } else {
        std::cout << "  训练回调未设置，无法启动训练\n";
    }
}

void CLITuner::cmd_show_config() {
    sync_config_from_registry();
    const auto& c = active_config_;

    std::cout << "\n╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║                当前训练配置                             ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";

    std::cout << "=== 通用 ===\n";
    std::cout << "  总迭代次数: " << c.total_iterations << "\n";
    std::cout << "  检查点间隔: " << c.checkpoint_interval << "\n";
    std::cout << "  日志间隔: " << c.log_interval << "\n";
    std::cout << "  模型名称: " << c.model_name << "\n";
    std::cout << "  随机种子: " << c.seed << "\n\n";

    std::cout << "=== 自我对弈 ===\n";
    std::cout << "  Workers: " << c.self_play.num_workers << "\n";
    std::cout << "  每步模拟: " << c.self_play.simulations_per_move << "\n";
    std::cout << "  温度: " << c.self_play.temperature
              << " (衰减: " << c.self_play.temperature_decay
              << ", 最低: " << c.self_play.min_temperature << ")\n";
    std::cout << "  Dirichlet: alpha=" << c.self_play.dirichlet_alpha
              << ", weight=" << c.self_play.dirichlet_weight << "\n";
    std::cout << "  最大局长: " << c.self_play.max_game_length << "\n";
    std::cout << "  CUDA: " << (c.self_play.use_cuda ? "启用" : "禁用") << "\n\n";

    std::cout << "=== 训练 ===\n";
    std::cout << "  学习率: " << c.training.learning_rate
              << " (衰减率: " << c.training.lr_decay_rate << ")\n";
    std::cout << "  批次大小: " << c.training.batch_size << "\n";
    std::cout << "  每迭代局数: " << c.training.games_per_iteration << "\n";
    std::cout << "  优化器: " << c.training.optimizer_type << "\n";
    std::cout << "  损失权重: 策略=" << c.training.policy_loss_weight
              << ", 价值=" << c.training.value_loss_weight
              << ", 门控=" << c.training.gating_loss_weight << "\n";
    std::cout << "  权重衰减: " << c.training.weight_decay << "\n";
    std::cout << "  Dropout: " << c.training.dropout_rate << "\n";
    std::cout << "  Label Smoothing: " << c.training.label_smoothing << "\n";
    std::cout << "  熵正则: " << c.training.entropy_regularization << "\n";
    std::cout << "  渐进扩张: " << (c.training.progressive_expansion ? "启用" : "禁用");
    if (c.training.progressive_expansion) {
        std::cout << " (" << c.training.initial_window_size
                  << " → " << c.training.max_window_size
                  << ", ×" << c.training.expansion_factor
                  << ", 每" << c.training.expansion_interval << "迭代)\n";
    } else {
        std::cout << " (固定" << c.training.window_size << ")\n";
    }
    std::cout << "\n";

    std::cout << "=== 评估 ===\n";
    std::cout << "  评估间隔: 每" << c.evaluation.evaluation_interval << "次迭代\n";
    std::cout << "  评估局数: " << c.evaluation.num_evaluation_games << "\n";
    std::cout << "  贴目: " << c.evaluation.komi << "\n\n";

    // 显示最近的训练指标
    if (last_iteration_ > 0) {
        std::cout << "=== 最近训练指标 (迭代" << last_iteration_ << ") ===\n";
        std::cout << "  策略损失: " << last_policy_loss_ << "\n";
        std::cout << "  价值损失: " << last_value_loss_ << "\n";
        std::cout << "  门控损失: " << last_gating_loss_ << "\n";
        std::cout << "  验证策略: " << last_val_policy_loss_ << "\n";
        std::cout << "  验证价值: " << last_val_value_loss_ << "\n";
        std::cout << "  ELO: " << last_elo_ << "\n\n";
    }
}

void CLITuner::cmd_diff() {
    auto names = registry_.get_all_names();
    bool found = false;

    std::cout << "\n=== 与默认值的差异 ===\n";
    for (const auto& name : names) {
        auto* meta = registry_.get_meta(name);
        if (!meta) continue;
        auto current = registry_.get(name);
        // 比较当前值与默认值是否不同
        bool different = false;
        if (std::holds_alternative<int>(current)) {
            different = std::get<int>(current) != std::get<int>(meta->default_value);
        } else if (std::holds_alternative<float>(current)) {
            different = std::abs(std::get<float>(current) - std::get<float>(meta->default_value)) > 1e-9f;
        } else if (std::holds_alternative<bool>(current)) {
            different = std::get<bool>(current) != std::get<bool>(meta->default_value);
        } else if (std::holds_alternative<std::string>(current)) {
            different = std::get<std::string>(current) != std::get<std::string>(meta->default_value);
        }

        if (different) {
            found = true;
            std::cout << "  " << name << ": "
                      << value_to_string(meta->default_value) << " → "
                      << value_to_string(current) << "\n";
        }
    }
    if (!found) {
        std::cout << "  所有参数均为默认值\n";
    }
    std::cout << "\n";
}

void CLITuner::cmd_search(const std::string& keyword) {
    if (keyword.empty()) {
        std::cout << "用法: search <关键词>\n";
        return;
    }

    auto names = registry_.get_all_names();
    std::string kw_lower = keyword;
    std::transform(kw_lower.begin(), kw_lower.end(), kw_lower.begin(), ::tolower);

    bool found = false;
    for (const auto& name : names) {
        auto* meta = registry_.get_meta(name);
        if (!meta) continue;

        std::string name_lower = name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
        std::string desc_lower = meta->description;
        std::transform(desc_lower.begin(), desc_lower.end(), desc_lower.begin(), ::tolower);

        if (name_lower.find(kw_lower) != std::string::npos ||
            desc_lower.find(kw_lower) != std::string::npos) {
            auto val = registry_.get(name);
            std::cout << "  " << std::left << std::setw(40) << name
                      << "= " << value_to_string(val)
                      << "  # " << meta->description << "\n";
            found = true;
        }
    }
    if (!found) {
        std::cout << "  未找到匹配 '" << keyword << "' 的参数\n";
    }
}

void CLITuner::print_banner() {
    std::cout << "\n╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║     SpaceGo 维空弈 - 交互式调参与训练系统               ║\n";
    std::cout << "║     Interactive Parameter Tuning & Training CLI         ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";
    std::cout << "\n  输入 'help' 查看可用命令\n";
    std::cout << "  输入 'list' 查看所有可调参数\n";
    std::cout << "  输入 'suggest' 获取智能调参建议\n";
    std::cout << "  输入 'train' 启动训练\n\n";
}

void CLITuner::print_prompt() {
    std::cout << "spacego-tune> ";
    std::cout.flush();
}

std::string CLITuner::value_to_string(const ParamValue& val) const {
    std::string result;
    std::visit([&result](const auto& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, bool>) {
            result = v ? "true" : "false";
        } else if constexpr (std::is_same_v<T, float>) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(6) << v;
            result = oss.str();
        } else if constexpr (std::is_same_v<T, std::string>) {
            result = v;
        } else {
            result = std::to_string(v);
        }
    }, val);
    return result;
}

void CLITuner::sync_config_from_registry() {
    registry_.apply_to_config(active_config_);
}

} // namespace training
} // namespace go
} // namespace spacego

#endif // USE_LIBTORCH