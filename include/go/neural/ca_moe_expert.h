#pragma once

#ifdef USE_LIBTORCH

#include <torch/torch.h>
#include <vector>
#include <memory>
#include <string>
#include <mutex>

namespace spacego {
namespace go {
namespace neural {

// 共享主干网络：5~10层残差卷积
// 注意：使用shared_ptr<Impl>代替ModuleHolder成员，避免MSVC过度实例化默认构造函数
class SharedBackboneImpl : public torch::nn::Module {
public:
    SharedBackboneImpl(int input_channels = 17, int num_blocks = 10, int hidden_dim = 256, float dropout_rate = 0.0f);

    torch::Tensor forward(torch::Tensor x);

private:
    std::shared_ptr<torch::nn::Conv2dImpl> input_conv;
    std::shared_ptr<torch::nn::BatchNorm2dImpl> input_bn;
    torch::nn::Sequential residual_blocks{nullptr};
    std::shared_ptr<torch::nn::DropoutImpl> dropout;  // 防过拟合Dropout层
    int hidden_dim;
};
TORCH_MODULE(SharedBackbone);

// 专家策略头
class ExpertPolicyHeadImpl : public torch::nn::Module {
public:
    ExpertPolicyHeadImpl(int input_dim = 256, int board_size = 19, float dropout_rate = 0.0f);

    torch::Tensor forward(torch::Tensor features);

private:
    std::shared_ptr<torch::nn::Conv2dImpl> conv;
    std::shared_ptr<torch::nn::LinearImpl> fc;
    std::shared_ptr<torch::nn::DropoutImpl> dropout;  // 全连接层后Dropout
    int board_size;
};
TORCH_MODULE(ExpertPolicyHead);

// 专家价值头
class ExpertValueHeadImpl : public torch::nn::Module {
public:
    ExpertValueHeadImpl(int input_dim = 256, float dropout_rate = 0.0f);

    torch::Tensor forward(torch::Tensor features);

private:
    std::shared_ptr<torch::nn::Conv2dImpl> conv;
    std::shared_ptr<torch::nn::LinearImpl> fc1;
    std::shared_ptr<torch::nn::LinearImpl> fc2;
    std::shared_ptr<torch::nn::DropoutImpl> dropout;  // 全连接层后Dropout
};
TORCH_MODULE(ExpertValueHead);

// 门控网络
class GatingNetworkImpl : public torch::nn::Module {
public:
    GatingNetworkImpl(int input_dim = 256, int num_experts = 8, float dropout_rate = 0.0f);

    torch::Tensor forward(torch::Tensor features);

    // 返回 (softmax权重, logits)：供训练时对logits施加L2正则防坍缩
    std::pair<torch::Tensor, torch::Tensor> forward_with_logits(torch::Tensor features);

    // ==================== 对手风格条件化（专家分化核心机制） ====================
    // 围棋实战思维：面对不同棋风的对手，应选择不同对策/专家。
    // 设置当前"对手风格"向量（5维：[攻击, 防守, 均衡, 实地, 厚势]），
    // 前向时对门控logits叠加 style_projection(风格) 偏置，让门控按对手风格分派专家。
    // 训练时注入多样化风格（原型池采样）→ 门控学会条件分化；
    // 竞争对弈时注入真实对手画像 → 推理时真正"看对手下棋"。
    // 未设置（或 clear 后）→ 无条件门控，行为与旧版完全一致（兼容自对弈/旧checkpoint）。
    void set_opponent_style(const std::vector<float>& style);
    void clear_opponent_style();

    // 兼容加载门控网络：
    // torch::nn::Module::load 是严格模式——旧checkpoint的gating_network归档中没有
    // style_projection子模块时直接抛异常。改为逐组件 try_read 加载：
    // 新checkpoint（含style_projection）正常加载；旧checkpoint缺失style_projection时
    // 保持零初始化（无偏置，行为与旧版一致），fc1/fc2/dropout 正常恢复。
    void load_compat(torch::serialize::InputArchive& archive);

private:
    std::shared_ptr<torch::nn::LinearImpl> fc1;
    std::shared_ptr<torch::nn::LinearImpl> fc2;
    std::shared_ptr<torch::nn::DropoutImpl> dropout;  // 全连接层后Dropout
    // 对手风格投影层：5维风格向量 → num_experts 门控偏置。
    // 权重零初始化：初始无偏置（logits不变化），随训练学习"风格→专家"映射，
    // 不破坏已有棋力（新参数从零开始，梯度只增不减地引导分化）。
    std::shared_ptr<torch::nn::LinearImpl> style_projection;
    // 当前对手风格向量（CPU [5] tensor，未设置时为未定义 tensor）
    torch::Tensor opponent_style_;
    int num_experts;
    // softmax温度：>1软化门控分布，防止logits饱和成one-hot导致专家坍缩。
    // 历史教训：温度4.0无效——softmax(x/T)的导数含1/T，所有门控梯度被缩小4倍，
    //          反而削弱了KL正则的对抗力，门控仍坍缩成[0,0,0,1,0,0,0,0]。
    // 温度1.0（不软化）：门控分化完全交给KL正则+负载均衡损失（权重10.0），
    //          训练时坍缩惩罚大（6.08×10=60），健康时惩罚小（0.5×10=5）。
    float softmax_temperature = 1.0f;
};
TORCH_MODULE(GatingNetwork);

// CA-MOE推理结果
struct CA_MOEInferenceResult {
    std::vector<torch::Tensor> expert_policies;  // 每个专家的策略分布 [batch_size, 362]
    std::vector<torch::Tensor> expert_values;     // 每个专家的价值估计 [batch_size, 1]
    torch::Tensor gating_weights;                  // 门控权重 [batch_size, num_experts]
    torch::Tensor gating_logits;                   // 门控logits（softmax前）[batch_size, num_experts]
    torch::Tensor final_policy;                    // 最终加权策略 [batch_size, 362]
    torch::Tensor final_value;                     // 最终加权价值 [batch_size, 1]
};

// CA-MOE混合专家系统
class CA_MOEExpertSystemImpl : public torch::nn::Module {
public:
    // hidden_dim：主干/专家头/门控共享的通道宽度，决定模型参数量。
    // 默认256为原生产模型；缩小(如128)可显著减参——门控fc1(hidden_dim*361→512)与残差卷积是参数大头，
    // 专家数保持不变，供回滚(train)模式启用"小模型"快速迭代；后续多小模型可按HMGS-DP思路合并参数。
    CA_MOEExpertSystemImpl(int num_experts = 8, int board_size = 19,
                           float dropout_rate = 0.0f, int hidden_dim = 256);

    // 前向推理
    CA_MOEInferenceResult forward(torch::Tensor batch_states);

    // 批量推理
    std::vector<CA_MOEInferenceResult> batch_forward(
        const std::vector<torch::Tensor>& states);

    // 模型保存/加载
    void save(const std::string& path);
    void load(const std::string& path);

    // 获取专家参数（用于分布式训练）
    std::vector<torch::Tensor> get_expert_parameters(int expert_idx);
    void set_expert_parameters(int expert_idx, const std::vector<torch::Tensor>& params);

    // 获取/设置全局参数
    std::vector<torch::Tensor> get_shared_parameters();
    void set_shared_parameters(const std::vector<torch::Tensor>& params);

    // 获取专家数量
    int get_num_experts() const { return num_experts; }

    // 设置为评估模式
    void set_eval_mode();

    // 设置为训练模式
    void set_train_mode();

    // 获取网络组件（使用shared_ptr<Impl>代替ModuleHolder返回类型）
    std::shared_ptr<SharedBackboneImpl> get_backbone() const { return backbone; }
    const std::vector<std::shared_ptr<ExpertPolicyHeadImpl>>& get_policy_heads() const { return policy_heads; }
    const std::vector<std::shared_ptr<ExpertValueHeadImpl>>& get_value_heads() const { return value_heads; }
    std::shared_ptr<GatingNetworkImpl> get_gating_network() const { return gating_network; }

    // ==================== 对手风格条件化（对外接口） ====================
    // 设置/清除整个专家系统的对手风格（转发给门控网络，带推理互斥锁保护）。
    // 竞争对弈中由训练器按对手实时画像调用；自对弈/未设置时不注入，保持无条件。
    void set_opponent_style(const std::vector<float>& style);
    void clear_opponent_style();

private:
    // 使用shared_ptr<Impl>代替ModuleHolder成员变量，避免MSVC实例化默认构造函数
    std::shared_ptr<SharedBackboneImpl> backbone;
    std::vector<std::shared_ptr<ExpertPolicyHeadImpl>> policy_heads;
    std::vector<std::shared_ptr<ExpertValueHeadImpl>> value_heads;
    std::shared_ptr<GatingNetworkImpl> gating_network;
    int num_experts;
    int board_size;
    int hidden_dim;   // 模型通道宽度（用于保存/加载时校验，防256/128通道checkpoint混用）
    bool is_training;
    // 推理互斥锁：保护多线程环境下的forward调用
    // 多个SelfPlayGenerator worker共享同一模型时，需要序列化推理调用
    mutable std::mutex inference_mutex_;
};
TORCH_MODULE(CA_MOEExpertSystem);

} // namespace neural
} // namespace go
} // namespace spacego
#endif  // USE_LIBTORCH
