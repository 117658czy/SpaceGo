#include "go/training/supervised_pretrain.h"

#ifdef USE_LIBTORCH

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <sstream>

namespace spacego {
namespace go {
namespace training {

// KGSDataLoader实现
KGSDataLoader::KGSDataLoader() {
}

bool KGSDataLoader::load_from_directory(const std::string& directory) {
    std::lock_guard<std::mutex> lock(loader_mutex);
    samples.clear();
    return true;
}

bool KGSDataLoader::load_from_file(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(loader_mutex);
    return true;
}

std::vector<SupervisedSample> KGSDataLoader::convert_to_samples() {
    std::lock_guard<std::mutex> lock(loader_mutex);
    return samples;
}

std::vector<SupervisedSample> KGSDataLoader::get_batch(size_t start_idx, size_t batch_size) {
    std::lock_guard<std::mutex> lock(loader_mutex);
    
    std::vector<SupervisedSample> batch;
    for (size_t i = start_idx; i < start_idx + batch_size && i < samples.size(); ++i) {
        batch.push_back(samples[i]);
    }
    return batch;
}

void KGSDataLoader::shuffle() {
    std::lock_guard<std::mutex> lock(loader_mutex);
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(samples.begin(), samples.end(), g);
}

void KGSDataLoader::clear() {
    std::lock_guard<std::mutex> lock(loader_mutex);
    samples.clear();
}

// SupervisedPretrainer实现
SupervisedPretrainer::SupervisedPretrainer(const SupervisedPretrainConfig& config)
    : config(config), current_epoch(0), current_step(0) {
}

void SupervisedPretrainer::set_ca_moe_system(std::shared_ptr<neural::CA_MOEExpertSystem> system) {
    std::lock_guard<std::mutex> lock(pretrain_mutex);
    ca_moe_system = system;
    
    if (ca_moe_system) {
        std::vector<torch::Tensor> all_params;
        
        if (config.train_backbone) {
            auto backbone = (*ca_moe_system)->get_backbone();
            for (const auto& p : backbone->parameters()) {
                all_params.push_back(p);
            }
        }
        
        if (config.train_experts) {
            auto policy_heads = (*ca_moe_system)->get_policy_heads();
            auto value_heads = (*ca_moe_system)->get_value_heads();
            for (auto& head : policy_heads) {
                for (const auto& p : head->parameters()) {
                    all_params.push_back(p);
                }
            }
            for (auto& head : value_heads) {
                for (const auto& p : head->parameters()) {
                    all_params.push_back(p);
                }
            }
        }
        
        if (config.train_gating) {
            auto gating = (*ca_moe_system)->get_gating_network();
            for (const auto& p : gating->parameters()) {
                all_params.push_back(p);
            }
        }
        
        optimizer = std::unique_ptr<torch::optim::Adam>(
            new torch::optim::Adam(all_params, torch::optim::AdamOptions(config.learning_rate)
                .weight_decay(config.weight_decay)));
    }
}

void SupervisedPretrainer::set_data_loader(std::shared_ptr<KGSDataLoader> loader) {
    std::lock_guard<std::mutex> lock(pretrain_mutex);
    data_loader = loader;
    
    if (data_loader) {
        auto all_samples = data_loader->convert_to_samples();
        split_data();
    }
}

void SupervisedPretrainer::split_data() {
    if (!data_loader) {
        return;
    }
    
    auto all_samples = data_loader->convert_to_samples();
    size_t validation_size = (size_t)(all_samples.size() * config.validation_split);
    
    training_data.clear();
    validation_data.clear();
    
    for (size_t i = 0; i < all_samples.size(); ++i) {
        if (i < validation_size) {
            validation_data.push_back(all_samples[i]);
        } else {
            training_data.push_back(all_samples[i]);
        }
    }
}

torch::Tensor SupervisedPretrainer::compute_policy_loss(const std::vector<SupervisedSample>& batch) {
    std::lock_guard<std::mutex> lock(pretrain_mutex);
    
    if (!ca_moe_system || batch.empty()) {
        return torch::tensor(0.0f);
    }
    
    std::vector<torch::Tensor> states;
    std::vector<torch::Tensor> target_policies;
    
    for (const auto& sample : batch) {
        states.push_back(sample.state.unsqueeze(0));
        
        std::vector<float> policy_floats(sample.target_policy.begin(), sample.target_policy.end());
        target_policies.push_back(torch::tensor(policy_floats).unsqueeze(0));
    }
    
    auto batch_states = torch::cat(states, 0);
    auto batch_targets = torch::cat(target_policies, 0);
    
    auto result = (*ca_moe_system)->forward(batch_states);
    
    if (!result.final_policy.defined()) {
        return torch::tensor(0.0f);
    }
    
    auto loss = -(batch_targets * result.final_policy).sum(1).mean();
    return loss;
}

torch::Tensor SupervisedPretrainer::compute_value_loss(const std::vector<SupervisedSample>& batch) {
    std::lock_guard<std::mutex> lock(pretrain_mutex);
    
    if (!ca_moe_system || batch.empty()) {
        return torch::tensor(0.0f);
    }
    
    std::vector<torch::Tensor> states;
    std::vector<float> target_values;
    
    for (const auto& sample : batch) {
        states.push_back(sample.state.unsqueeze(0));
        target_values.push_back(sample.target_value);
    }
    
    auto batch_states = torch::cat(states, 0);
    auto batch_targets = torch::tensor(target_values);
    
    auto result = (*ca_moe_system)->forward(batch_states);
    
    if (!result.final_value.defined()) {
        return torch::tensor(0.0f);
    }
    
    auto loss = torch::mse_loss(result.final_value.squeeze(1), batch_targets);
    return loss;
}

SupervisedPretrainStats SupervisedPretrainer::train_epoch() {
    std::lock_guard<std::mutex> lock(pretrain_mutex);
    
    SupervisedPretrainStats stats;
    stats.epoch = current_epoch;
    stats.step = current_step;
    
    if (!optimizer || training_data.empty()) {
        return stats;
    }
    
    float total_policy_loss = 0.0f;
    float total_value_loss = 0.0f;
    int num_batches = 0;
    
    for (size_t i = 0; i < training_data.size(); i += config.batch_size) {
        std::vector<SupervisedSample> batch;
        for (size_t j = i; j < i + config.batch_size && j < training_data.size(); ++j) {
            batch.push_back(training_data[j]);
        }
        
        if (batch.empty()) {
            continue;
        }
        
        optimizer->zero_grad();
        
        auto policy_loss = compute_policy_loss(batch);
        auto value_loss = compute_value_loss(batch);
        auto total_loss = policy_loss + value_loss;
        
        total_loss.backward();
        optimizer->step();
        
        total_policy_loss += policy_loss.item<float>();
        total_value_loss += value_loss.item<float>();
        num_batches++;
        current_step++;
    }
    
    if (num_batches > 0) {
        stats.policy_loss = total_policy_loss / num_batches;
        stats.value_loss = total_value_loss / num_batches;
        stats.total_loss = stats.policy_loss + stats.value_loss;
    }
    
    training_history.push_back(stats);
    current_epoch++;
    
    return stats;
}

SupervisedPretrainStats SupervisedPretrainer::validate() {
    std::lock_guard<std::mutex> lock(pretrain_mutex);
    
    SupervisedPretrainStats stats;
    stats.epoch = current_epoch;
    stats.step = current_step;
    
    if (!ca_moe_system || validation_data.empty()) {
        return stats;
    }
    
    float total_policy_loss = 0.0f;
    float total_value_loss = 0.0f;
    float total_policy_accuracy = 0.0f;
    float total_value_mae = 0.0f;
    int num_batches = 0;
    
    for (size_t i = 0; i < validation_data.size(); i += config.batch_size) {
        std::vector<SupervisedSample> batch;
        for (size_t j = i; j < i + config.batch_size && j < validation_data.size(); ++j) {
            batch.push_back(validation_data[j]);
        }
        
        if (batch.empty()) {
            continue;
        }
        
        auto policy_loss = compute_policy_loss(batch);
        auto value_loss = compute_value_loss(batch);
        
        total_policy_loss += policy_loss.item<float>();
        total_value_loss += value_loss.item<float>();
        num_batches++;
    }
    
    if (num_batches > 0) {
        stats.policy_loss = total_policy_loss / num_batches;
        stats.value_loss = total_value_loss / num_batches;
        stats.total_loss = stats.policy_loss + stats.value_loss;
        stats.policy_accuracy = total_policy_accuracy / num_batches;
        stats.value_mae = total_value_mae / num_batches;
    }
    
    return stats;
}

void SupervisedPretrainer::train() {
    std::lock_guard<std::mutex> lock(pretrain_mutex);
    
    training_history.clear();
    current_epoch = 0;
    current_step = 0;
    
    for (int epoch = 0; epoch < config.num_epochs; ++epoch) {
        auto train_stats = train_epoch();
        auto val_stats = validate();
        
        if (optimizer) {
            float lr = config.learning_rate;
            if (current_step < config.warmup_steps) {
                lr = config.learning_rate * (float)current_step / config.warmup_steps;
            }
            for (auto& param_group : optimizer->param_groups()) {
                param_group.options().set_lr(lr);
            }
        }
    }
}

void SupervisedPretrainer::save_checkpoint(const std::string& path) {
    std::lock_guard<std::mutex> lock(pretrain_mutex);
    if (ca_moe_system) {
        (*ca_moe_system)->save(path);
    }
}

void SupervisedPretrainer::load_checkpoint(const std::string& path) {
    std::lock_guard<std::mutex> lock(pretrain_mutex);
    if (ca_moe_system) {
        (*ca_moe_system)->load(path);
    }
}

std::vector<SupervisedPretrainStats> SupervisedPretrainer::get_training_history() const {
    std::lock_guard<std::mutex> lock(pretrain_mutex);
    return training_history;
}

void SupervisedPretrainer::generate_report(const std::string& output_path) {
    std::lock_guard<std::mutex> lock(pretrain_mutex);
    
    std::ofstream file(output_path);
    if (!file.is_open()) {
        return;
    }
    
    file << "监督预训练验收报告\n";
    file << "===================\n\n";
    file << "训练轮数: " << current_epoch << "\n";
    file << "训练步数: " << current_step << "\n";
    file << "训练样本数: " << training_data.size() << "\n";
    file << "验证样本数: " << validation_data.size() << "\n";
    
    if (!training_history.empty()) {
        auto& last_stats = training_history.back();
        file << "最终策略损失: " << last_stats.policy_loss << "\n";
        file << "最终价值损失: " << last_stats.value_loss << "\n";
        file << "最终总损失: " << last_stats.total_loss << "\n";
    }
    
    file.close();
}

void SupervisedPretrainer::reset() {
    std::lock_guard<std::mutex> lock(pretrain_mutex);
    current_epoch = 0;
    current_step = 0;
    training_history.clear();
    training_data.clear();
    validation_data.clear();
}

} // namespace training
} // namespace go
} // namespace spacego

#endif  // USE_LIBTORCH
