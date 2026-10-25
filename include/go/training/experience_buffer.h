#pragma once

#ifdef USE_LIBTORCH

#include <vector>
#include <queue>
#include <memory>
#include <utility>
#include <torch/torch.h>
#include <mutex>
#include <atomic>
#include <cstdint>
#include <algorithm>
#include <random>
#include <stdexcept>

namespace spacego {
namespace go {
namespace training {

// 单条经验数据
struct Experience {
    torch::Tensor state;          // 状态特征 (17, 19, 19)
    int action;                    // 动作索引
    float reward;                  // 奖励
    torch::Tensor next_state;     // 下一状态
    std::vector<int> expert_indices;  // 参与专家索引
    std::vector<float> expert_weights; // 专家权重
    float value_current;           // 当前状态价值
    float value_next;              // 下一状态价值
};

// 优先级经验池（单个专家）
class PrioritizedExperiencePool {
public:
    PrioritizedExperiencePool(int capacity = 5000, float alpha = 0.6f, float beta = 0.4f);
    
    // 添加经验
    void add(const Experience& exp);
    
    // 批量采样
    std::vector<Experience> sample(int batch_size);
    
    // 更新优先级
    void update_priorities(const std::vector<int>& indices, 
                          const std::vector<float>& priorities);
    
    // 获取池大小
    int size() const;
    
    // 清空池
    void clear();
    
    // 序列化/反序列化（用于分布式传输）
    std::vector<uint8_t> serialize() const;
    void deserialize(const std::vector<uint8_t>& data);
    
private:
    // SumTree用于高效优先级采样
    class SumTree {
    public:
        SumTree(int capacity);
        void update(int idx, float priority);
        void add(float priority, const Experience& exp);
        std::pair<int, Experience> get(float s);
        float get_total_priority() const;
        int get_size() const;
        
    private:
        int capacity;
        std::vector<float> tree;
        std::vector<Experience> data;
        int count;
        void propagate(int idx, float change);
        int retrieve(int idx, float s);
    };
    
    std::unique_ptr<SumTree> sum_tree;
    int capacity;
    float alpha;
    float beta;
    mutable std::mutex mutex;
    std::mt19937 rng;
};

// 多专家经验池管理器
class MultiExpertExperienceManager {
public:
    MultiExpertExperienceManager(int num_experts = 8, int capacity_per_expert = 5000);
    
    // 为专家添加经验
    void add_experience(int expert_idx, const Experience& exp);
    
    // 批量为所有专家添加经验（含归因）
    void add_experiences_with_attribution(
        const std::vector<Experience>& experiences,
        const std::vector<std::vector<int>>& expert_assignments,
        const std::vector<std::vector<float>>& expert_weights);
    
    // 从指定专家池采样
    std::vector<Experience> sample_from_expert(int expert_idx, int batch_size);
    
    // 从所有专家池批量采样
    std::vector<std::vector<Experience>> sample_all_experts(int batch_size_per_expert);
    
    // 获取专家池
    PrioritizedExperiencePool* get_expert_pool(int expert_idx);
    
    // 统计信息
    std::vector<int> get_pool_sizes() const;
    
    // 清空所有池
    void clear_all();
    
    // 分布式训练支持：同步经验池
    std::vector<uint8_t> serialize_expert_pool(int expert_idx) const;
    void deserialize_expert_pool(int expert_idx, const std::vector<uint8_t>& data);
    
    // 获取专家数量
    int get_num_experts() const { return num_experts; }
    
private:
    std::vector<std::unique_ptr<PrioritizedExperiencePool>> expert_pools;
    int num_experts;
    mutable std::mutex manager_mutex;
};

} // namespace training
} // namespace go
} // namespace spacego
#endif  // USE_LIBTORCH
