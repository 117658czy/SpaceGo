#pragma once

#ifdef USE_LIBTORCH

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include "go/training/param_registry.h"
#include "go/training/unsupervised_self_reinforcement.h"

namespace spacego {
namespace go {
namespace training {

// CLI调参界面：交互式命令行，支持查看/修改参数、自动建议、保存/加载配置、启动训练
class CLITuner {
public:
    CLITuner();
    ~CLITuner() = default;

    // 启动CLI交互循环
    void run();

    // 设置训练回调（在CLI中触发训练时调用）
    using TrainCallback = std::function<int(const UnsupervisedTrainingConfig&)>;
    void set_train_callback(TrainCallback callback);

    // 从外部更新训练指标（用于自动调参建议）
    void update_metrics(float policy_loss, float value_loss, float gating_loss,
                        float val_policy_loss, float val_value_loss,
                        float elo, int iteration);

private:
    ParamRegistry& registry_;
    TrainCallback train_callback_;

    // 当前训练指标
    float last_policy_loss_ = 0.0f;
    float last_value_loss_ = 0.0f;
    float last_gating_loss_ = 0.0f;
    float last_val_policy_loss_ = 0.0f;
    float last_val_value_loss_ = 0.0f;
    float last_elo_ = 1200.0f;
    int last_iteration_ = 0;

    // 当前活跃的配置（可随时修改然后启动训练）
    UnsupervisedTrainingConfig active_config_;

    // 命令处理函数
    void cmd_help();
    void cmd_list(const std::string& group);
    void cmd_get(const std::string& name);
    void cmd_set(const std::string& name, const std::string& value);
    void cmd_reset(const std::string& name);
    void cmd_reset_all();
    void cmd_save(const std::string& path);
    void cmd_load(const std::string& path);
    void cmd_suggest();
    void cmd_apply_suggest();
    void cmd_train();
    void cmd_train_quick(); // 快速训练模式（少量迭代）
    void cmd_show_config();
    void cmd_diff(); // 显示与默认值的差异
    void cmd_search(const std::string& keyword); // 搜索参数

    // 辅助函数
    void print_banner();
    void print_prompt();
    std::string value_to_string(const ParamValue& val) const;
    void sync_config_from_registry();
};

} // namespace training
} // namespace go
} // namespace spacego

#endif // USE_LIBTORCH