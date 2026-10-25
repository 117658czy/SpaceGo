#include "go/training/experience_buffer.h"

#ifdef USE_LIBTORCH

#include <algorithm>
#include <cmath>
#include <cstring>

namespace spacego {
namespace go {
namespace training {

// SumTree实现
PrioritizedExperiencePool::SumTree::SumTree(int capacity) 
    : capacity(capacity), count(0) {
    int tree_size = 1;
    while (tree_size < capacity) {
        tree_size *= 2;
    }
    tree.resize(2 * tree_size, 0.0f);
    data.resize(capacity);
}

void PrioritizedExperiencePool::SumTree::propagate(int idx, float change) {
    while (idx > 1) {
        idx /= 2;
        tree[idx] += change;
    }
}

int PrioritizedExperiencePool::SumTree::retrieve(int idx, float s) {
    if (idx >= (int)tree.size() / 2) {
        return idx - (int)tree.size() / 2;
    }
    int left = 2 * idx;
    if (s <= tree[left]) {
        return retrieve(left, s);
    } else {
        return retrieve(left + 1, s - tree[left]);
    }
}

void PrioritizedExperiencePool::SumTree::update(int idx, float priority) {
    idx += (int)tree.size() / 2;
    float change = priority - tree[idx];
    tree[idx] = priority;
    propagate(idx, change);
}

void PrioritizedExperiencePool::SumTree::add(float priority, const Experience& exp) {
    int idx = count % capacity;
    data[idx] = exp;
    update(idx, priority);
    count++;
}

std::pair<int, Experience> PrioritizedExperiencePool::SumTree::get(float s) {
    int idx = retrieve(1, s);
    return {idx, data[idx]};
}

float PrioritizedExperiencePool::SumTree::get_total_priority() const {
    return tree[1];
}

int PrioritizedExperiencePool::SumTree::get_size() const {
    return std::min(count, capacity);
}

// PrioritizedExperiencePool实现
PrioritizedExperiencePool::PrioritizedExperiencePool(int capacity, float alpha, float beta)
    : capacity(capacity), alpha(alpha), beta(beta), rng(std::random_device{}()) {
    sum_tree = std::make_unique<SumTree>(capacity);
}

void PrioritizedExperiencePool::add(const Experience& exp) {
    std::lock_guard<std::mutex> lock(mutex);
    
    float priority = 1.0f;
    if (sum_tree->get_size() > 0) {
        float max_p = sum_tree->get_total_priority() / sum_tree->get_size();
        priority = std::max(std::abs(exp.reward), max_p);
    }
    priority = std::pow(priority, alpha);
    
    sum_tree->add(priority, exp);
}

std::vector<Experience> PrioritizedExperiencePool::sample(int batch_size) {
    std::lock_guard<std::mutex> lock(mutex);
    
    std::vector<Experience> batch;
    int size = sum_tree->get_size();
    if (size == 0) {
        return batch;
    }
    
    float total_p = sum_tree->get_total_priority();
    std::uniform_real_distribution<float> dist(0.0f, total_p);
    
    for (int i = 0; i < batch_size && i < size; ++i) {
        float s = dist(rng);
        auto [idx, exp] = sum_tree->get(s);
        batch.push_back(exp);
    }
    
    return batch;
}

void PrioritizedExperiencePool::update_priorities(const std::vector<int>& indices, 
                                                 const std::vector<float>& priorities) {
    std::lock_guard<std::mutex> lock(mutex);
    
    for (size_t i = 0; i < indices.size() && i < priorities.size(); ++i) {
        float p = std::pow(std::abs(priorities[i]), alpha);
        sum_tree->update(indices[i], p);
    }
}

int PrioritizedExperiencePool::size() const {
    std::lock_guard<std::mutex> lock(mutex);
    return sum_tree->get_size();
}

void PrioritizedExperiencePool::clear() {
    std::lock_guard<std::mutex> lock(mutex);
    sum_tree = std::make_unique<SumTree>(capacity);
}

std::vector<uint8_t> PrioritizedExperiencePool::serialize() const {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<uint8_t> data;
    data.reserve(1024 * 1024);
    return data;
}

void PrioritizedExperiencePool::deserialize(const std::vector<uint8_t>& data) {
    std::lock_guard<std::mutex> lock(mutex);
}

// MultiExpertExperienceManager实现
MultiExpertExperienceManager::MultiExpertExperienceManager(int num_experts, int capacity_per_expert)
    : num_experts(num_experts) {
    for (int i = 0; i < num_experts; ++i) {
        expert_pools.push_back(std::make_unique<PrioritizedExperiencePool>(capacity_per_expert));
    }
}

void MultiExpertExperienceManager::add_experience(int expert_idx, const Experience& exp) {
    std::lock_guard<std::mutex> lock(manager_mutex);
    if (expert_idx >= 0 && expert_idx < num_experts) {
        expert_pools[expert_idx]->add(exp);
    }
}

void MultiExpertExperienceManager::add_experiences_with_attribution(
    const std::vector<Experience>& experiences,
    const std::vector<std::vector<int>>& expert_assignments,
    const std::vector<std::vector<float>>& expert_weights) {
    
    std::lock_guard<std::mutex> lock(manager_mutex);
    
    for (size_t i = 0; i < experiences.size(); ++i) {
        const auto& exp = experiences[i];
        const auto& assignments = expert_assignments[i];
        const auto& weights = expert_weights[i];
        
        for (size_t j = 0; j < assignments.size(); ++j) {
            int expert_idx = assignments[j];
            float weight = weights[j];
            
            Experience attributed_exp = exp;
            attributed_exp.expert_indices = {expert_idx};
            attributed_exp.expert_weights = {weight};
            attributed_exp.reward *= weight;
            
            if (expert_idx >= 0 && expert_idx < num_experts) {
                expert_pools[expert_idx]->add(attributed_exp);
            }
        }
    }
}

std::vector<Experience> MultiExpertExperienceManager::sample_from_expert(int expert_idx, int batch_size) {
    std::lock_guard<std::mutex> lock(manager_mutex);
    if (expert_idx >= 0 && expert_idx < num_experts) {
        return expert_pools[expert_idx]->sample(batch_size);
    }
    return {};
}

std::vector<std::vector<Experience>> MultiExpertExperienceManager::sample_all_experts(int batch_size_per_expert) {
    std::lock_guard<std::mutex> lock(manager_mutex);
    
    std::vector<std::vector<Experience>> all_samples;
    for (int i = 0; i < num_experts; ++i) {
        all_samples.push_back(expert_pools[i]->sample(batch_size_per_expert));
    }
    return all_samples;
}

PrioritizedExperiencePool* MultiExpertExperienceManager::get_expert_pool(int expert_idx) {
    std::lock_guard<std::mutex> lock(manager_mutex);
    if (expert_idx >= 0 && expert_idx < num_experts) {
        return expert_pools[expert_idx].get();
    }
    return nullptr;
}

std::vector<int> MultiExpertExperienceManager::get_pool_sizes() const {
    std::lock_guard<std::mutex> lock(manager_mutex);
    
    std::vector<int> sizes;
    for (int i = 0; i < num_experts; ++i) {
        sizes.push_back(expert_pools[i]->size());
    }
    return sizes;
}

void MultiExpertExperienceManager::clear_all() {
    std::lock_guard<std::mutex> lock(manager_mutex);
    
    for (int i = 0; i < num_experts; ++i) {
        expert_pools[i]->clear();
    }
}

std::vector<uint8_t> MultiExpertExperienceManager::serialize_expert_pool(int expert_idx) const {
    std::lock_guard<std::mutex> lock(manager_mutex);
    if (expert_idx >= 0 && expert_idx < num_experts) {
        return expert_pools[expert_idx]->serialize();
    }
    return {};
}

void MultiExpertExperienceManager::deserialize_expert_pool(int expert_idx, const std::vector<uint8_t>& data) {
    std::lock_guard<std::mutex> lock(manager_mutex);
    if (expert_idx >= 0 && expert_idx < num_experts) {
        expert_pools[expert_idx]->deserialize(data);
    }
}

} // namespace training
} // namespace go
} // namespace spacego

#endif  // USE_LIBTORCH
