#pragma once

#ifdef USE_LIBTORCH

#include <vector>
#include <memory>
#include <torch/torch.h>
#include "go/training/experience_buffer.h"
#include "go/neural/ca_moe_expert.h"

namespace spacego {
namespace go {
namespace training {

// 搜索过程中的单步数据
struct SearchStepData {
    torch::Tensor state;          // 当前状态特征
    torch::Tensor next_state;     // 下一状态特征
    int action;                    // 选择的动作
    float value_current;           // 当前状态价值估计
    float value_next;              // 下一状态价值估计
    torch::Tensor gating_weights;  // 门控权重
    std::vector<int> expert_indices;  // 参与专家索引
};

// 一局对局的完整经验
struct GameExperience {
    std::vector<SearchStepData> steps;  // 对局的所有步
    int game_id;                          // 对局ID
    float final_outcome;                  // 最终结果（+1胜，-1负，0平局）
};

// 经验收集器
class ExperienceCollector {
public:
    ExperienceCollector(int num_experts = 8);
    
    // 设置经验池管理器
    void set_experience_manager(std::shared_ptr<MultiExpertExperienceManager> manager);
    
    // 设置CA-MOE系统（用于获取专家信息）
    void set_ca_moe_system(std::shared_ptr<neural::CA_MOEExpertSystem> system);
    
    // 开始新的一局
    void start_game(int game_id);
    
    // 记录搜索中的一步
    void record_step(const SearchStepData& step);
    
    // 结束一局并处理经验
    void end_game(float final_outcome);
    
    // 计算奖励（基于胜率变化）
    float compute_reward(float value_current, float value_next, float final_outcome, int step_idx);
    
    // 奖励归因：将门控权重分配给专家
    std::vector<std::vector<int>> assign_experts(const std::vector<torch::Tensor>& gating_weights, int top_k = 2);
    
    // 将收集的经验添加到经验池
    void flush_to_buffer();
    
    // 获取当前收集的经验数量
    int get_collected_count() const;
    
    // 清空收集的经验
    void clear_collected();
    
    // 获取最近一局的经验（用于调试）
    std::optional<GameExperience> get_latest_game() const;
    
private:
    int num_experts;
    std::shared_ptr<MultiExpertExperienceManager> experience_manager;
    std::shared_ptr<neural::CA_MOEExpertSystem> ca_moe_system;
    
    std::vector<GameExperience> collected_games;
    std::optional<GameExperience> current_game;
    
    mutable std::mutex collector_mutex;
};

// HMGS-DP搜索回调集成 - 用于从搜索树收集数据
class HMGSDP_SearchCallback {
public:
    HMGSDP_SearchCallback(std::shared_ptr<ExperienceCollector> collector);
    
    // 在节点选择时调用
    void on_node_select(const torch::Tensor& state, 
                       const neural::CA_MOEInferenceResult& inference_result,
                       int selected_action);
    
    // 在价值备份时调用
    void on_value_backup(float value_current, float value_next);
    
    // 获取当前收集的步数
    int get_current_steps() const { return current_steps; }
    
    // 重置步数计数
    void reset_steps() { current_steps = 0; }
    
private:
    std::shared_ptr<ExperienceCollector> collector;
    std::vector<SearchStepData> pending_steps;
    int current_steps;
    mutable std::mutex callback_mutex;
};

} // namespace training
} // namespace go
} // namespace spacego
#endif  // USE_LIBTORCH
