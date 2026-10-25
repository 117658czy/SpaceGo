#include "go/training/experience_collector.h"

#ifdef USE_LIBTORCH

#include <algorithm>
#include <cmath>
#include <optional>

namespace spacego {
namespace go {
namespace training {

// ExperienceCollector实现
ExperienceCollector::ExperienceCollector(int num_experts)
    : num_experts(num_experts) {
}

void ExperienceCollector::set_experience_manager(std::shared_ptr<MultiExpertExperienceManager> manager) {
    std::lock_guard<std::mutex> lock(collector_mutex);
    experience_manager = manager;
}

void ExperienceCollector::set_ca_moe_system(std::shared_ptr<neural::CA_MOEExpertSystem> system) {
    std::lock_guard<std::mutex> lock(collector_mutex);
    ca_moe_system = system;
}

void ExperienceCollector::start_game(int game_id) {
    std::lock_guard<std::mutex> lock(collector_mutex);
    current_game = GameExperience();
    current_game->game_id = game_id;
    current_game->final_outcome = 0.0f;
}

void ExperienceCollector::record_step(const SearchStepData& step) {
    std::lock_guard<std::mutex> lock(collector_mutex);
    if (current_game) {
        current_game->steps.push_back(step);
    }
}

void ExperienceCollector::end_game(float final_outcome) {
    std::lock_guard<std::mutex> lock(collector_mutex);
    if (current_game) {
        current_game->final_outcome = final_outcome;
        collected_games.push_back(*current_game);
        current_game = std::nullopt;
    }
}

float ExperienceCollector::compute_reward(float value_current, float value_next, 
                                          float final_outcome, int step_idx) {
    // 基础奖励：胜率变化
    float reward = value_next - value_current;
    
    // 如果是最后一步，加上最终结果
    if (step_idx >= 0) {
        reward += 0.1f * final_outcome;
    }
    
    return reward;
}

std::vector<std::vector<int>> ExperienceCollector::assign_experts(
    const std::vector<torch::Tensor>& gating_weights, int top_k) {
    
    std::vector<std::vector<int>> all_assignments;
    
    for (const auto& weights : gating_weights) {
        std::vector<std::pair<float, int>> weight_idx_pairs;
        auto cpu_weights = weights.to(torch::kCPU);
        
        for (int j = 0; j < num_experts; ++j) {
            weight_idx_pairs.emplace_back(-cpu_weights[j].item<float>(), j);
        }
        std::sort(weight_idx_pairs.begin(), weight_idx_pairs.end());
        
        std::vector<int> assignment;
        for (int j = 0; j < top_k && j < num_experts; ++j) {
            assignment.push_back(weight_idx_pairs[j].second);
        }
        all_assignments.push_back(assignment);
    }
    
    return all_assignments;
}

void ExperienceCollector::flush_to_buffer() {
    std::lock_guard<std::mutex> lock(collector_mutex);
    
    if (!experience_manager || collected_games.empty()) {
        return;
    }
    
    for (const auto& game : collected_games) {
        std::vector<Experience> experiences;
        std::vector<torch::Tensor> all_gating_weights;
        std::vector<std::vector<float>> all_expert_weights;
        
        for (size_t i = 0; i < game.steps.size(); ++i) {
            const auto& step = game.steps[i];
            
            Experience exp;
            exp.state = step.state;
            exp.action = step.action;
            exp.value_current = step.value_current;
            exp.value_next = step.value_next;
            exp.expert_indices = step.expert_indices;
            
            float next_value = (i + 1 < game.steps.size()) ? 
                               game.steps[i + 1].value_current : game.final_outcome;
            exp.reward = compute_reward(step.value_current, next_value, game.final_outcome, i);
            
            if (!step.next_state.defined() && i + 1 < game.steps.size()) {
                exp.next_state = game.steps[i + 1].state;
            } else {
                exp.next_state = step.next_state;
            }
            
            experiences.push_back(exp);
            all_gating_weights.push_back(step.gating_weights);
            
            std::vector<float> weights;
            for (int idx : step.expert_indices) {
                weights.push_back(step.gating_weights[idx].item<float>());
            }
            all_expert_weights.push_back(weights);
        }
        
        auto assignments = assign_experts(all_gating_weights, 2);
        
        experience_manager->add_experiences_with_attribution(
            experiences, assignments, all_expert_weights);
    }
    
    collected_games.clear();
}

int ExperienceCollector::get_collected_count() const {
    std::lock_guard<std::mutex> lock(collector_mutex);
    int count = 0;
    for (const auto& game : collected_games) {
        count += (int)game.steps.size();
    }
    return count;
}

void ExperienceCollector::clear_collected() {
    std::lock_guard<std::mutex> lock(collector_mutex);
    collected_games.clear();
    current_game = std::nullopt;
}

std::optional<GameExperience> ExperienceCollector::get_latest_game() const {
    std::lock_guard<std::mutex> lock(collector_mutex);
    if (!collected_games.empty()) {
        return collected_games.back();
    }
    return std::nullopt;
}

// HMGSDP_SearchCallback实现
HMGSDP_SearchCallback::HMGSDP_SearchCallback(std::shared_ptr<ExperienceCollector> collector)
    : collector(collector), current_steps(0) {
}

void HMGSDP_SearchCallback::on_node_select(const torch::Tensor& state,
                                            const neural::CA_MOEInferenceResult& inference_result,
                                            int selected_action) {
    std::lock_guard<std::mutex> lock(callback_mutex);
    
    SearchStepData step;
    step.state = state;
    step.next_state = state;
    step.action = selected_action;
    step.gating_weights = inference_result.gating_weights;
    
    if (!inference_result.expert_values.empty()) {
        step.value_current = inference_result.final_value.item<float>();
    }
    
    std::vector<std::pair<float, int>> weight_idx_pairs;
    auto cpu_weights = inference_result.gating_weights.to(torch::kCPU);
    
    for (int j = 0; j < cpu_weights.size(1); ++j) {
        weight_idx_pairs.emplace_back(-cpu_weights[0][j].item<float>(), j);
    }
    std::sort(weight_idx_pairs.begin(), weight_idx_pairs.end());
    
    int top_k = 2;
    for (int j = 0; j < top_k && j < (int)weight_idx_pairs.size(); ++j) {
        step.expert_indices.push_back(weight_idx_pairs[j].second);
    }
    
    pending_steps.push_back(step);
    current_steps++;
}

void HMGSDP_SearchCallback::on_value_backup(float value_current, float value_next) {
    std::lock_guard<std::mutex> lock(callback_mutex);
    
    if (!pending_steps.empty()) {
        pending_steps.back().value_current = value_current;
        pending_steps.back().value_next = value_next;
    }
}

} // namespace training
} // namespace go
} // namespace spacego

#endif  // USE_LIBTORCH
