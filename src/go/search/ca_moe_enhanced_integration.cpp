#include "go/search/ca_moe_enhanced_integration.h"

#ifdef USE_LIBTORCH

#include "go/board/go_board_cuda.h"
#include <chrono>
#include <algorithm>
#include <sstream>
#include <optional>
#include <iomanip>

namespace spacego {
namespace go {
namespace search {

CA_MOEEnhancedIntegration::CA_MOEEnhancedIntegration()
    : training_mode(false), game_in_progress(false) {
    stats_.expert_usage.resize(8, 0);
}

void CA_MOEEnhancedIntegration::set_ca_moe_system(std::shared_ptr<neural::CA_MOEExpertSystem> system) {
    std::lock_guard<std::mutex> lock(integration_mutex);
    ca_moe_system = system;
    if (system) {
        stats_.expert_usage.resize((*system)->get_num_experts(), 0);
    }
}

void CA_MOEEnhancedIntegration::set_experience_collector(std::shared_ptr<training::ExperienceCollector> collector) {
    std::lock_guard<std::mutex> lock(integration_mutex);
    experience_collector = collector;
}

void CA_MOEEnhancedIntegration::update_cache(const std::string& key,
                                          const neural::CA_MOEInferenceResult& value) const {
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        lru_list_.remove(key);
    }

    if (cache_.size() >= CACHE_SIZE) {
        auto oldest = lru_list_.back();
        lru_list_.pop_back();
        cache_.erase(oldest);
    }

    cache_[key] = value;
    lru_list_.push_front(key);
}

std::optional<neural::CA_MOEInferenceResult> CA_MOEEnhancedIntegration::get_from_cache(
    const std::string& key) const {

    auto it = cache_.find(key);
    if (it == cache_.end()) {
        return std::nullopt;
    }

    lru_list_.remove(key);
    lru_list_.push_front(key);

    return it->second;
}

torch::Tensor CA_MOEEnhancedIntegration::state_str_to_tensor(const std::string& state_str) const {
    auto tensor = torch::zeros({17, 19, 19}, torch::dtype(torch::kFloat32));

    if (state_str.empty()) {
        return tensor;
    }

    char current_player = 'B';
    std::string board_str = state_str;

    size_t sep_pos = state_str.find('|');
    if (sep_pos != std::string::npos && sep_pos > 0) {
        current_player = state_str[0];
        board_str = state_str.substr(sep_pos + 1);
    }

    char self_char = (current_player == 'B' || current_player == 'b') ? 'B' : 'W';
    char opp_char = (self_char == 'B') ? 'W' : 'B';
    char self_lower = (self_char == 'B') ? 'b' : 'w';
    char opp_lower = (opp_char == 'B') ? 'b' : 'w';

    auto tensor_accessor = tensor.accessor<float, 3>();

    for (int row = 0; row < 19; ++row) {
        for (int col = 0; col < 19; ++col) {
            int idx = row * 19 + col;

            if (idx >= static_cast<int>(board_str.size())) {
                continue;
            }

            char stone = board_str[idx];

            if (stone == self_char || stone == self_lower) {
                tensor_accessor[0][row][col] = 1.0f;
            }

            if (stone == self_char || stone == self_lower) {
                tensor_accessor[1][row][col] = 1.0f;
            }

            if (stone == opp_char || stone == opp_lower) {
                tensor_accessor[2][row][col] = 1.0f;
            }

            if (stone == opp_char || stone == opp_lower) {
                tensor_accessor[3][row][col] = 1.0f;
            }

            if (stone == self_char || stone == self_lower) {
                tensor_accessor[4][row][col] = 1.0f;
            } else if (stone == opp_char || stone == opp_lower) {
                tensor_accessor[4][row][col] = -1.0f;
            }

            if (stone == self_char || stone == self_lower ||
                stone == opp_char || stone == opp_lower) {
                tensor_accessor[5][row][col] = 1.0f;
            }
        }
    }

    float color_constant = (current_player == 'B' || current_player == 'b') ? 1.0f : 0.0f;
    for (int row = 0; row < 19; ++row) {
        for (int col = 0; col < 19; ++col) {
            tensor_accessor[16][row][col] = color_constant;
        }
    }

    return tensor;
}

namespace {
    std::string tensor_to_cache_key(const torch::Tensor& tensor) {
        if (!tensor.defined()) {
            return "";
        }

        auto cpu_tensor = tensor.to(torch::kCPU).contiguous();
        auto data_ptr = cpu_tensor.data_ptr<float>();
        auto num_elements = cpu_tensor.numel();

        uint64_t hash = 5381;
        size_t step = std::max<size_t>(1, num_elements / 64);
        for (size_t i = 0; i < num_elements; i += step) {
            hash = ((hash << 5) + hash) + static_cast<uint64_t>(
                static_cast<int64_t>(data_ptr[i] * 1000.0f));
        }

        std::stringstream ss;
        ss << std::hex << std::setfill('0') << std::setw(16) << hash;
        return ss.str();
    }
}

neural::CA_MOEInferenceResult CA_MOEEnhancedIntegration::infer_full(const torch::Tensor& state) {
    std::string cache_key = tensor_to_cache_key(state);

    if (auto cached = get_from_cache(cache_key)) {
        stats_.cache_hits++;
        stats_.total_calls++;
        return *cached;
    }

    std::lock_guard<std::mutex> lock(integration_mutex);

    if (!ca_moe_system) {
        stats_.total_calls++;
        return neural::CA_MOEInferenceResult();
    }

    auto start = std::chrono::high_resolution_clock::now();

    auto batch_state = state.unsqueeze(0);

    neural::CA_MOEInferenceResult result = (*ca_moe_system)->forward(batch_state);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    stats_.total_calls++;
    stats_.avg_inference_time_ms = (stats_.avg_inference_time_ms * static_cast<double>(stats_.total_calls - 1)
                                   + elapsed_ms) / static_cast<double>(stats_.total_calls);

    if (result.gating_weights.defined()) {
        auto weights_cpu = result.gating_weights.to(torch::kCPU).contiguous();
        auto weights_accessor = weights_cpu.accessor<float, 2>();
        int num_experts = static_cast<int>(weights_accessor.size(1));
        if (stats_.expert_usage.size() < static_cast<size_t>(num_experts)) {
            stats_.expert_usage.resize(num_experts, 0);
        }
        for (int e = 0; e < num_experts; ++e) {
            if (weights_accessor[0][e] > 0.1f) {
                stats_.expert_usage[e]++;
            }
        }
    }

    inference_times_.push_back(elapsed_ms);

    update_cache(cache_key, result);

    if (training_mode && experience_collector && game_in_progress) {
        training::SearchStepData step;
        step.state = state;
        step.next_state = state;
        step.action = -1;
        step.gating_weights = result.gating_weights;

        if (result.final_value.defined() && result.final_value.numel() > 0) {
            step.value_current = result.final_value.item<float>();
        }

        if (result.gating_weights.defined()) {
            auto cpu_weights = result.gating_weights.to(torch::kCPU);
            std::vector<std::pair<float, int>> weight_idx_pairs;
            for (int j = 0; j < cpu_weights.size(1); ++j) {
                weight_idx_pairs.emplace_back(-cpu_weights[0][j].item<float>(), j);
            }
            std::sort(weight_idx_pairs.begin(), weight_idx_pairs.end());
            int top_k = 2;
            for (int j = 0; j < top_k && j < static_cast<int>(weight_idx_pairs.size()); ++j) {
                step.expert_indices.push_back(weight_idx_pairs[j].second);
            }
        }

        experience_collector->record_step(step);
    }

    return result;
}

std::vector<neural::CA_MOEInferenceResult> CA_MOEEnhancedIntegration::batch_infer_full(
    const std::vector<torch::Tensor>& states) {

    std::vector<neural::CA_MOEInferenceResult> results;
    results.reserve(states.size());

    std::vector<torch::Tensor> uncached_states;
    std::vector<size_t> uncached_indices;
    std::vector<std::string> all_cache_keys;
    all_cache_keys.reserve(states.size());

    for (size_t i = 0; i < states.size(); ++i) {
        std::string key = tensor_to_cache_key(states[i]);
        all_cache_keys.push_back(key);

        if (auto cached = get_from_cache(key)) {
            results.push_back(*cached);
            stats_.cache_hits++;
        } else {
            results.push_back(neural::CA_MOEInferenceResult());
            uncached_states.push_back(states[i]);
            uncached_indices.push_back(i);
        }
    }
    stats_.total_calls += states.size();

    if (!uncached_states.empty()) {
        std::lock_guard<std::mutex> lock(integration_mutex);

        if (ca_moe_system) {
            auto start = std::chrono::high_resolution_clock::now();

            auto batch = torch::stack(uncached_states, 0);

            neural::CA_MOEInferenceResult batch_result = (*ca_moe_system)->forward(batch);

            auto end = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

            double total_uncached = static_cast<double>(uncached_states.size());
            stats_.avg_inference_time_ms = (stats_.avg_inference_time_ms *
                                           static_cast<double>(stats_.total_calls - uncached_states.size())
                                           + elapsed_ms) / static_cast<double>(stats_.total_calls);

            if (batch_result.gating_weights.defined()) {
                auto weights_cpu = batch_result.gating_weights.to(torch::kCPU).contiguous();
                auto weights_accessor = weights_cpu.accessor<float, 2>();
                int batch_count = static_cast<int>(weights_accessor.size(0));
                int num_experts = static_cast<int>(weights_accessor.size(1));
                if (stats_.expert_usage.size() < static_cast<size_t>(num_experts)) {
                    stats_.expert_usage.resize(num_experts, 0);
                }
                for (int b = 0; b < batch_count; ++b) {
                    for (int e = 0; e < num_experts; ++e) {
                        if (weights_accessor[b][e] > 0.1f) {
                            stats_.expert_usage[e]++;
                        }
                    }
                }
            }

            for (size_t i = 0; i < uncached_states.size(); ++i) {
                neural::CA_MOEInferenceResult single_result;

                if (batch_result.final_policy.defined() &&
                    batch_result.final_policy.size(0) > static_cast<int64_t>(i)) {
                    single_result.final_policy = batch_result.final_policy[i].clone();
                }

                if (batch_result.final_value.defined() &&
                    batch_result.final_value.size(0) > static_cast<int64_t>(i)) {
                    single_result.final_value = batch_result.final_value[i].clone();
                }

                if (batch_result.gating_weights.defined() &&
                    batch_result.gating_weights.size(0) > static_cast<int64_t>(i)) {
                    single_result.gating_weights = batch_result.gating_weights[i].clone();
                }

                single_result.expert_policies.resize(batch_result.expert_policies.size());
                for (size_t e = 0; e < batch_result.expert_policies.size(); ++e) {
                    if (batch_result.expert_policies[e].defined() &&
                        batch_result.expert_policies[e].size(0) > static_cast<int64_t>(i)) {
                        single_result.expert_policies[e] = batch_result.expert_policies[e][i].clone();
                    }
                }

                single_result.expert_values.resize(batch_result.expert_values.size());
                for (size_t e = 0; e < batch_result.expert_values.size(); ++e) {
                    if (batch_result.expert_values[e].defined() &&
                        batch_result.expert_values[e].size(0) > static_cast<int64_t>(i)) {
                        single_result.expert_values[e] = batch_result.expert_values[e][i].clone();
                    }
                }

                size_t original_idx = uncached_indices[i];
                results[original_idx] = single_result;
                update_cache(all_cache_keys[original_idx], single_result);
            }

            inference_times_.push_back(elapsed_ms);
        }
    }

    return results;
}

std::pair<std::vector<double>, double> CA_MOEEnhancedIntegration::infer(const std::string& state_str) {
    torch::Tensor tensor = state_str_to_tensor(state_str);

    neural::CA_MOEInferenceResult result = infer_full(tensor);

    std::vector<double> policy_probs;
    if (result.final_policy.defined() && result.final_policy.numel() > 0) {
        auto policy_cpu = result.final_policy.to(torch::kCPU).contiguous();
        auto policy_ptr = policy_cpu.data_ptr<float>();
        size_t policy_size = static_cast<size_t>(policy_cpu.numel());
        policy_probs.resize(policy_size, 0.0);
        for (size_t i = 0; i < policy_size; ++i) {
            policy_probs[i] = static_cast<double>(policy_ptr[i]);
        }
    }

    double value = 0.0;
    if (result.final_value.defined() && result.final_value.numel() > 0) {
        auto value_cpu = result.final_value.to(torch::kCPU).contiguous();
        value = static_cast<double>(value_cpu.data_ptr<float>()[0]);
    }

    return {policy_probs, value};
}

void CA_MOEEnhancedIntegration::set_training_mode(bool enabled) {
    std::lock_guard<std::mutex> lock(integration_mutex);
    training_mode = enabled;
    if (ca_moe_system) {
        if (enabled) {
            (*ca_moe_system)->set_train_mode();
        } else {
            (*ca_moe_system)->set_eval_mode();
        }
    }
}

void CA_MOEEnhancedIntegration::start_game(int game_id) {
    std::lock_guard<std::mutex> lock(integration_mutex);
    game_in_progress = true;
    if (experience_collector) {
        experience_collector->start_game(game_id);
    }
}

void CA_MOEEnhancedIntegration::end_game(float final_outcome) {
    std::lock_guard<std::mutex> lock(integration_mutex);
    game_in_progress = false;
    if (experience_collector) {
        experience_collector->end_game(final_outcome);
    }
}

CA_MOEEnhancedIntegration::PerformanceStats CA_MOEEnhancedIntegration::get_performance_stats() const {
    std::lock_guard<std::mutex> lock(integration_mutex);
    return stats_;
}

void CA_MOEEnhancedIntegration::reset_stats() {
    std::lock_guard<std::mutex> lock(integration_mutex);
    stats_ = PerformanceStats();
    if (ca_moe_system) {
        stats_.expert_usage.resize((*ca_moe_system)->get_num_experts(), 0);
    } else {
        stats_.expert_usage.resize(8, 0);
    }
    inference_times_.clear();
}

} // namespace search
} // namespace go
} // namespace spacego

#endif // USE_LIBTORCH
