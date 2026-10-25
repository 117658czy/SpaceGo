#pragma once

#ifdef USE_LIBTORCH

#include <torch/torch.h>
#include <vector>
#include <memory>
#include <string>
#include "go/neural/ca_moe_expert.h"

namespace spacego {
namespace go {
namespace training {

// 监督预训练配置
struct SupervisedPretrainConfig {
    float learning_rate = 1e-3f;
    int batch_size = 32;
    int num_epochs = 100;
    float weight_decay = 1e-4f;
    int warmup_steps = 1000;
    float validation_split = 0.1f;
    bool train_backbone = true;
    bool train_experts = true;
    bool train_gating = true;
};

// 监督预训练数据样本
struct SupervisedSample {
    torch::Tensor state;
    std::vector<double> target_policy;
    float target_value;
};

// 监督预训练统计
struct SupervisedPretrainStats {
    float policy_loss;
    float value_loss;
    float total_loss;
    float policy_accuracy;
    float value_mae;
    int epoch;
    int step;
};

// KGS棋谱数据加载器
class KGSDataLoader {
public:
    KGSDataLoader();
    
    // 从目录加载棋谱
    bool load_from_directory(const std::string& directory);
    
    // 从单个文件加载棋谱
    bool load_from_file(const std::string& filepath);
    
    // 转换为训练样本
    std::vector<SupervisedSample> convert_to_samples();
    
    // 获取样本数量
    size_t get_sample_count() const { return samples.size(); }
    
    // 获取批次
    std::vector<SupervisedSample> get_batch(size_t start_idx, size_t batch_size);
    
    // 打乱数据
    void shuffle();
    
    // 清空数据
    void clear();
    
private:
    std::vector<SupervisedSample> samples;
    mutable std::mutex loader_mutex;
};

// 监督预训练器
class SupervisedPretrainer {
public:
    SupervisedPretrainer(const SupervisedPretrainConfig& config);
    
    // 设置CA-MOE系统
    void set_ca_moe_system(std::shared_ptr<neural::CA_MOEExpertSystem> system);
    
    // 设置数据加载器
    void set_data_loader(std::shared_ptr<KGSDataLoader> loader);
    
    // 训练一个epoch
    SupervisedPretrainStats train_epoch();
    
    // 验证
    SupervisedPretrainStats validate();
    
    // 完整训练
    void train();
    
    // 计算策略损失
    torch::Tensor compute_policy_loss(const std::vector<SupervisedSample>& batch);
    
    // 计算价值损失
    torch::Tensor compute_value_loss(const std::vector<SupervisedSample>& batch);
    
    // 保存检查点
    void save_checkpoint(const std::string& path);
    
    // 加载检查点
    void load_checkpoint(const std::string& path);
    
    // 获取当前epoch
    int get_current_epoch() const { return current_epoch; }
    
    // 获取当前step
    int get_current_step() const { return current_step; }
    
    // 获取训练历史
    std::vector<SupervisedPretrainStats> get_training_history() const;
    
    // 生成验收报告
    void generate_report(const std::string& output_path);
    
    // 重置训练状态
    void reset();
    
private:
    SupervisedPretrainConfig config;
    std::shared_ptr<neural::CA_MOEExpertSystem> ca_moe_system;
    std::shared_ptr<KGSDataLoader> data_loader;
    
    std::unique_ptr<torch::optim::Adam> optimizer;
    std::vector<SupervisedPretrainStats> training_history;
    
    int current_epoch;
    int current_step;
    
    std::vector<SupervisedSample> training_data;
    std::vector<SupervisedSample> validation_data;
    
    void split_data();
    
    mutable std::mutex pretrain_mutex;
};

// 模拟数据生成器，用于无棋谱文件时的监督预训练
// 生成随机的围棋棋盘状态和策略/价值标签
class MockDataGenerator {
public:
    MockDataGenerator() = default;
    
    // 生成指定数量的模拟训练样本
    // count: 样本数量
    // 返回包含随机棋盘状态、策略分布和价值标签的样本集合
    std::vector<SupervisedSample> generate_samples(int count) {
        std::vector<SupervisedSample> samples;
        samples.reserve(count);
        
        // 固定随机种子保证可复现性
        torch::manual_seed(42);
        
        for (int i = 0; i < count; ++i) {
            SupervisedSample sample;
            
            // 生成随机棋盘状态：17通道x19x19
            // 通道0: 黑子位置, 通道1: 白子位置, 通道2-15: 气/ liberty等特征, 通道16: 当前玩家
            sample.state = torch::randn({17, 19, 19});
            
            // 生成随机策略分布（362个位置的概率：361个棋盘位置+1个pass，归一化）
            auto policy_logits = torch::randn({362});
            auto policy_probs = torch::softmax(policy_logits, 0);
            
            sample.target_policy.resize(362);
            for (int j = 0; j < 362; ++j) {
                sample.target_policy[j] = policy_probs[j].item<double>();
            }
            
            // 生成随机价值标签（-1到1之间）
            sample.target_value = static_cast<float>(
                torch::rand({1}).item<float>() * 2.0 - 1.0);
            
            samples.push_back(std::move(sample));
        }
        
        return samples;
    }
};

} // namespace training
} // namespace go
} // namespace spacego
#endif  // USE_LIBTORCH
