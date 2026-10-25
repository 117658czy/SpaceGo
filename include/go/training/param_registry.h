#pragma once

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <variant>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <type_traits>

namespace spacego {
namespace go {
namespace training {

// 前向声明（避免循环include，实际定义在unsupervised_self_reinforcement.h中）
struct UnsupervisedTrainingConfig;

// 参数值类型：支持int、float、bool、string四种
using ParamValue = std::variant<int, float, bool, std::string>;

// 参数元信息：名称、分组、描述、范围约束、推荐值
struct ParamMeta {
    std::string name;           // 参数名（如 "self_play.temperature"）
    std::string group;          // 分组（如 "self_play"、"training"、"evaluation"）
    std::string description;    // 中文描述
    ParamValue default_value;   // 默认值
    ParamValue min_value;       // 最小值（仅数值类型有效）
    ParamValue max_value;       // 最大值（仅数值类型有效）
    std::vector<ParamValue> recommended_values; // 推荐值列表（用于调参建议）
    bool tunable;               // 是否可调（某些结构参数不可在运行时修改）

    // 判断当前值是否在合法范围内
    bool is_in_range(const ParamValue& val) const {
        if (auto* iv = std::get_if<int>(&val)) {
            if (auto* mn = std::get_if<int>(&min_value))
                if (*iv < *mn) return false;
            if (auto* mx = std::get_if<int>(&max_value))
                if (*iv > *mx) return false;
        } else if (auto* fv = std::get_if<float>(&val)) {
            if (auto* mn = std::get_if<float>(&min_value))
                if (*fv < *mn) return false;
            if (auto* mx = std::get_if<float>(&max_value))
                if (*fv > *mx) return false;
        }
        return true;
    }
};

// 参数变更回调类型
using ParamChangeCallback = std::function<void(const std::string&, const ParamValue&)>;

// ==================== 参数注册表 ====================
// 统一管理所有训练/搜索/网络可调参数，支持：
// - 注册参数（带元信息和范围约束）
// - 运行时读写参数值
// - 参数变更回调通知
// - 保存/加载配置到文件
// - 自动调参建议
class ParamRegistry {
public:
    static ParamRegistry& instance();

    // 注册一个可调参数（带完整元信息）
    void register_param(const std::string& name, const std::string& group,
                        const std::string& description,
                        const ParamValue& default_value,
                        const ParamValue& min_value,
                        const ParamValue& max_value,
                        const std::vector<ParamValue>& recommended_values = {},
                        bool tunable = true);

    // 设置参数值（触发回调）
    bool set(const std::string& name, const ParamValue& value);

    // 获取参数值
    ParamValue get(const std::string& name) const;

    // 获取参数元信息
    const ParamMeta* get_meta(const std::string& name) const;

    // 注册参数变更回调
    void on_change(ParamChangeCallback callback);

    // 将当前所有参数值写入UnsupervisedTrainingConfig
    // （由调用方根据参数名映射到对应配置字段）
    void apply_to_config(struct UnsupervisedTrainingConfig& config) const;

    // 从UnsupervisedTrainingConfig加载所有参数值
    void load_from_config(const struct UnsupervisedTrainingConfig& config);

    // 保存参数到文件（可被后续加载）
    bool save_to_file(const std::string& path) const;

    // 从文件加载参数
    bool load_from_file(const std::string& path);

    // 获取所有参数名
    std::vector<std::string> get_all_names() const;

    // 获取指定分组的参数名
    std::vector<std::string> get_names_by_group(const std::string& group) const;

    // 获取所有分组名
    std::vector<std::string> get_all_groups() const;

    // 打印所有参数（格式化表格）
    void print_all() const;

    // 打印指定分组的参数
    void print_group(const std::string& group) const;

    // 自动调参：根据当前训练指标给出建议
    struct TuneSuggestion {
        std::string param_name;
        ParamValue current_value;
        ParamValue suggested_value;
        std::string reason; // 调参理由
    };
    std::vector<TuneSuggestion> suggest(
        float policy_loss, float value_loss, float gating_loss,
        float val_policy_loss, float val_value_loss,
        float elo, int iteration) const;

    // 重置所有参数为默认值
    void reset_defaults();

    // 重置指定参数为默认值
    bool reset_default(const std::string& name);

private:
    ParamRegistry() = default;
    std::map<std::string, ParamMeta> metas_;       // 参数元信息
    std::map<std::string, ParamValue> values_;     // 当前参数值
    std::vector<ParamChangeCallback> callbacks_;   // 变更回调列表

    void notify_change(const std::string& name, const ParamValue& value);
};

// 初始化所有参数注册到ParamRegistry（在程序启动时调用）
void initialize_param_registry();

} // namespace training
} // namespace go
} // namespace spacego