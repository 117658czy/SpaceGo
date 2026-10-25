#pragma once

#ifdef USE_LIBTORCH

#include "go/neural/ca_moe_expert.h"
#include "go/training/experience_collector.h"
#include <memory>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <list>
#include <optional>

namespace spacego {
namespace go {
namespace search {

class CA_MOEEnhancedIntegration {
public:
    CA_MOEEnhancedIntegration();

    void set_ca_moe_system(std::shared_ptr<neural::CA_MOEExpertSystem> system);

    void set_experience_collector(std::shared_ptr<training::ExperienceCollector> collector);

    std::shared_ptr<neural::CA_MOEExpertSystem> get_ca_moe_system() const { return ca_moe_system; }

    std::shared_ptr<training::ExperienceCollector> get_experience_collector() const { return experience_collector; }

    neural::CA_MOEInferenceResult infer_full(const torch::Tensor& state);

    std::vector<neural::CA_MOEInferenceResult> batch_infer_full(
        const std::vector<torch::Tensor>& states);

    std::pair<std::vector<double>, double> infer(const std::string& state_str);

    void set_training_mode(bool enabled);
    bool is_training_mode() const { return training_mode; }

    void start_game(int game_id);
    void end_game(float final_outcome);

    struct PerformanceStats {
        size_t total_calls = 0;
        size_t cache_hits = 0;
        double avg_inference_time_ms = 0.0;
        std::vector<size_t> expert_usage;
    };

    PerformanceStats get_performance_stats() const;
    void reset_stats();

    torch::Tensor state_str_to_tensor(const std::string& state_str) const;

private:
    std::shared_ptr<neural::CA_MOEExpertSystem> ca_moe_system;
    std::shared_ptr<training::ExperienceCollector> experience_collector;

    bool training_mode;
    bool game_in_progress;

    mutable std::mutex integration_mutex;

    static constexpr size_t CACHE_SIZE = 100;
    mutable std::unordered_map<std::string, neural::CA_MOEInferenceResult> cache_;
    mutable std::list<std::string> lru_list_;

    void update_cache(const std::string& key, const neural::CA_MOEInferenceResult& value) const;
    std::optional<neural::CA_MOEInferenceResult> get_from_cache(const std::string& key) const;

    PerformanceStats stats_;
    std::vector<double> inference_times_;
};

} // namespace search
} // namespace go
} // namespace spacego

#endif // USE_LIBTORCH
