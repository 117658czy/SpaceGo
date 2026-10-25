#include "go/neural/ca_moe_expert.h"

#ifdef USE_LIBTORCH

#include "go/neural/opponent_style_profiler.h"
#include "go/neural/residual_block.h"
#include "framework/logger.h"
#include <torch/torch.h>
#include <stdexcept>
#include <fstream>
#include <algorithm>
#include <random>

namespace spacego {
namespace go {
namespace neural {

// SharedBackbone实现
// 使用make_shared<Impl>直接创建，避免MSVC实例化ModuleHolder默认构造函数
SharedBackboneImpl::SharedBackboneImpl(int input_channels, int num_blocks, int hidden_dim, float dropout_rate)
    : hidden_dim(hidden_dim) {

    // 输入卷积层：直接创建Conv2dImpl，不经过ModuleHolder包装
    input_conv = std::make_shared<torch::nn::Conv2dImpl>(
        torch::nn::Conv2dOptions(input_channels, hidden_dim, 3).padding(1).bias(false));
    register_module("input_conv", input_conv);

    input_bn = std::make_shared<torch::nn::BatchNorm2dImpl>(hidden_dim);
    register_module("input_bn", input_bn);

    // 残差块序列
    residual_blocks = register_module("residual_blocks", torch::nn::Sequential());
    for (int i = 0; i < num_blocks; ++i) {
        residual_blocks->push_back(ResidualBlock(hidden_dim));
    }

    // Dropout层（仅在dropout_rate > 0时创建）
    if (dropout_rate > 0.0f) {
        dropout = std::make_shared<torch::nn::DropoutImpl>(
            torch::nn::DropoutOptions().p(dropout_rate));
        register_module("dropout", dropout);
    }
}

torch::Tensor SharedBackboneImpl::forward(torch::Tensor x) {
    x = torch::relu(input_bn->forward(input_conv->forward(x)));
    x = residual_blocks->forward(x);
    // 训练模式下应用Dropout（eval模式下Dropout自动变为identity）
    if (dropout) {
        x = dropout->forward(x);
    }
    return x;
}

// ExpertPolicyHead实现
ExpertPolicyHeadImpl::ExpertPolicyHeadImpl(int input_dim, int board_size, float dropout_rate)
    : board_size(board_size) {

    int action_size = board_size * board_size + 1;

    conv = std::make_shared<torch::nn::Conv2dImpl>(
        torch::nn::Conv2dOptions(input_dim, 2, 1).bias(false));
    register_module("conv", conv);

    fc = std::make_shared<torch::nn::LinearImpl>(2 * board_size * board_size, action_size);
    register_module("fc", fc);

    // 全连接层后Dropout（防止策略头过拟合）
    if (dropout_rate > 0.0f) {
        dropout = std::make_shared<torch::nn::DropoutImpl>(
            torch::nn::DropoutOptions().p(dropout_rate));
        register_module("dropout", dropout);
    }
}

torch::Tensor ExpertPolicyHeadImpl::forward(torch::Tensor features) {
    auto x = conv->forward(features);
    x = x.flatten(1);
    // 在全连接层前应用Dropout（训练模式下生效）
    if (dropout) {
        x = dropout->forward(x);
    }
    x = fc->forward(x);
    x = torch::log_softmax(x, 1);
    return x;
}

// ExpertValueHead实现
ExpertValueHeadImpl::ExpertValueHeadImpl(int input_dim, float dropout_rate) {
    conv = std::make_shared<torch::nn::Conv2dImpl>(
        torch::nn::Conv2dOptions(input_dim, 1, 1).bias(false));
    register_module("conv", conv);

    fc1 = std::make_shared<torch::nn::LinearImpl>(1 * 19 * 19, 256);
    register_module("fc1", fc1);

    fc2 = std::make_shared<torch::nn::LinearImpl>(256, 1);
    register_module("fc2", fc2);

    // 全连接层后Dropout（防止价值头过拟合）
    if (dropout_rate > 0.0f) {
        dropout = std::make_shared<torch::nn::DropoutImpl>(
            torch::nn::DropoutOptions().p(dropout_rate));
        register_module("dropout", dropout);
    }
}

torch::Tensor ExpertValueHeadImpl::forward(torch::Tensor features) {
    auto x = conv->forward(features);
    x = x.flatten(1);
    x = torch::relu(fc1->forward(x));
    // 在最后一层前应用Dropout（训练模式下生效）
    if (dropout) {
        x = dropout->forward(x);
    }
    x = torch::tanh(fc2->forward(x));
    return x;
}

// GatingNetwork实现
GatingNetworkImpl::GatingNetworkImpl(int input_dim, int num_experts, float dropout_rate)
    : num_experts(num_experts) {

    int flattened_size = input_dim * 19 * 19;

    fc1 = std::make_shared<torch::nn::LinearImpl>(flattened_size, 512);
    register_module("fc1", fc1);

    fc2 = std::make_shared<torch::nn::LinearImpl>(512, num_experts);
    register_module("fc2", fc2);

    // 对手风格投影层：5维风格向量 → num_experts 门控偏置。
    // 权重零初始化（bias 也为零）：初始 logits 不变化，行为与旧版一致；
    // 训练中随"风格→专家"信号缓慢学习，实现按对手棋风分派专家的分化。
    style_projection = std::make_shared<torch::nn::LinearImpl>(
        torch::nn::LinearOptions(5, num_experts).bias(true));
    register_module("style_projection", style_projection);
    torch::NoGradGuard no_grad_init;
    style_projection->weight.zero_();
    style_projection->bias.zero_();

    // 全连接层后Dropout（防止门控网络过拟合）
    if (dropout_rate > 0.0f) {
        dropout = std::make_shared<torch::nn::DropoutImpl>(
            torch::nn::DropoutOptions().p(dropout_rate));
        register_module("dropout", dropout);
    }
}

torch::Tensor GatingNetworkImpl::forward(torch::Tensor features) {
    auto x = features.flatten(1);
    x = torch::relu(fc1->forward(x));
    // 在最后一层前应用Dropout（训练模式下生效）
    if (dropout) {
        x = dropout->forward(x);
    }
    x = fc2->forward(x);
    // 温度软化：logits除以温度后softmax，>1时分布更均匀
    // 防止门控网络快速坍缩成one-hot（单个专家独占）
    x = torch::softmax(x / softmax_temperature, 1);
    return x;
}

// 门控前向：返回logits（softmax前）与软化权重，供训练时对logits施加L2正则
// 历史教训：仅对softmax输出施加KL/熵正则无效——门控坍缩成one-hot后，
// softmax雅可比在饱和区≈0，正则梯度传不回logits（"赢家通吃"梯度畅通无阻）。
// 对logits直接加L2正则，梯度始终存在，从源头限制logits幅度防坍缩。
std::pair<torch::Tensor, torch::Tensor> GatingNetworkImpl::forward_with_logits(
    torch::Tensor features) {
    auto x = features.flatten(1);
    auto pre_act = fc1->forward(x);
    x = torch::relu(pre_act);
    // 源头尺度归一化（无参数实现，不新增checkpoint键 → 旧checkpoint可直接加载）：
    //   fc1有47M参数（占模型76%）、输入flatten后达92416维，实测ReLU输出尺度约100，
    //   经fc2（512→8）累加后原始logits幅度冲到±2000。这对门控分化有两重伤害：
    //     ① 输出端标准化虽能恢复softmax可用性，但"局面间差异 ÷ 绝对幅度"被压得极小
    //        （实测门控batch标准差仅3e-3，等于没在按局面分化）；
    //     ② gating_loss的L2正则只拿得到clamp后的值(≤2)，对真实幅度毫无约束力。
    //   放在ReLU之后而非之前逐样本归一化：ReLU输出非负、均值>0，不会像放在ReLU之前
    //   那样因零均值把一半神经元结构性截成0。此步把fc2输入压到O(1)，从源头掐断
    //   "特征尺度→logits幅度"的放大链条。
    x = (x - x.mean(1, true)) / (x.std(1, false, true) + 1e-5f);
    if (dropout) {
        x = dropout->forward(x);
    }
    auto raw_logits = fc2->forward(x);
    // 诊断：门控内部状态，监控"源头尺度归一化"是否把logits幅度压回可控区间。
    //   · fc1正激活比例≈0 → ReLU全灭（fc1占总参数76%，会永久停更）；
    //   · |raw_logits-bias|均值 = fc2输出幅度（归一化前）。fc2输入已压到O(1)，
    //     健康值应落在O(1)~O(10)；若仍是1e3量级说明fc2权重膨胀未收敛（L2正则正在拉回）。
    //   · fc2权重L2：直接观察权重是否在缩小（源头修复的收敛指标）。
    if (this->is_training()) {
        static int gating_inner_diag = 0;
        if ((++gating_inner_diag % 50) == 1) {
            float alive = (pre_act > 0).to(torch::kFloat32).mean().item<float>();
            float spread = (raw_logits - fc2->bias).abs().mean().item<float>();
            float w2_norm = fc2->weight.norm().item<float>();
            std::cout << "[门控内部] fc1正激活比例=" << alive
                      << " | |raw_logits-bias|均值=" << spread
                      << " | fc2权重L2=" << w2_norm
                      << " | 输入特征|均值|=" << features.abs().mean().item<float>()
                      << std::endl;
        }
    }
    // 输出端标准化：逐样本零均值单位方差（只依赖本样本的跨专家维度，与batch内其他
    //   样本无关，故推理行为确定）。此处的尺度已被源头归一化压小（fc2输入由~100降到
    //   O(1)，输出幅度同比例回落），所以除以它不会再像从前（除以2e+03）那样把局面间的
    //   相对差异一并抹平——这正是门控能按局面分化的信息基础。
    //   注：此处不对raw_logits做截断——PyTorch的clamp反向对越界元素置0梯度，一旦截断，
    //   L2正则就再也拉不回膨胀的fc2权重，这是它此前失效的第二重原因。
    auto logits = (raw_logits - raw_logits.mean(1, true)) /
                  (raw_logits.std(1, false, true) + 1e-5f);
    // 钳制（标准化之后的纯安全网）：n=8时z分数上界为√(n-1)=2.65，±3永不触发，
    //   仅用于兜住异常放大。历史教训：clamp导致坍缩的根因是原始幅度±2000被整段截平
    //   （±10时5专家0.2/3专家0；±2时5专家0.1978/3专家0.0036），现已由源头归一化解决。
    constexpr float kGatingLogitClamp = 3.0f;
    logits = logits.clamp(-kGatingLogitClamp, kGatingLogitClamp);
    // 对手风格条件偏置：若已设置对手风格，则叠加 style_projection(风格)。
    // 风格向量为 CPU [5]，unsqueeze(0) → [1,5] 广播到 [N,5] → 投影后加到 [N,8] logits。
    // 零初始化 → 初始偏置≈0，随训练学习"对手棋风→专家偏好"映射。
    // 放在标准化之后：偏置幅度O(1)即可被softmax感知（源头归一化已消除爆炸尺度，
    //   否则任何偏置都会被±2000的原始logits淹没）。
    if (opponent_style_.defined() && opponent_style_.numel() > 0) {
        auto style_bias = style_projection->forward(
            opponent_style_.unsqueeze(0).to(logits.device()));
        logits = logits + style_bias;
        // 风格偏置叠加后再次钳制，防止投影层梯度把logits推出界（同上界）
        logits = logits.clamp(-kGatingLogitClamp, kGatingLogitClamp);
    }
    auto weights = torch::softmax(logits / softmax_temperature, 1);
    // 返回标准化后的logits供gating_loss的L2正则使用：源头归一化已把fc2输入从~100压到
    //   O(1)，其输出幅度随之同比例下降（实测spread由2e+03回落），故L2在此尺度上温和有效；
    //   无需返回未归一化的原始值（那会让L2项瞬间飙到1e4量级、压垮梯度裁剪后的其他梯度）。
    return {weights, logits};
}

// 设置当前对手风格向量（5维：[攻击, 防守, 均衡, 实地, 厚势]）
// 训练/竞争前由训练器调用；数值截断到 [0,1] 防异常输入。
void GatingNetworkImpl::set_opponent_style(const std::vector<float>& style) {
    if (style.size() != 5) {
        throw std::invalid_argument("对手风格向量必须为5维 (攻击,防守,均衡,实地,厚势)");
    }
    std::vector<float> clamped(5);
    for (int i = 0; i < 5; ++i) {
        clamped[i] = std::min(1.0f, std::max(0.0f, style[i]));
    }
    opponent_style_ = torch::from_blob(clamped.data(), {5}, torch::kFloat32).clone();
}

// 清除对手风格 → 门控恢复无条件（自对弈生成数据 / 对局结束时调用）
void GatingNetworkImpl::clear_opponent_style() {
    opponent_style_ = torch::Tensor();
}

// 兼容加载门控网络（逐组件 try_read，兼容新旧checkpoint格式）
// 详见头文件注释：Module::load 严格模式在旧checkpoint缺失style_projection时抛异常，
// 此处对每个子模块 try_read——缺失的保持当前（零初始化）值，存在的正常恢复。
void GatingNetworkImpl::load_compat(torch::serialize::InputArchive& archive) {
    torch::serialize::InputArchive sub;
    if (archive.try_read("fc1", sub)) {
        fc1->load(sub);
    }
    if (archive.try_read("fc2", sub)) {
        fc2->load(sub);
    }
    if (dropout) {
        torch::serialize::InputArchive drop_archive;
        if (archive.try_read("dropout", drop_archive)) {
            dropout->load(drop_archive);
        }
    }
    if (style_projection) {
        torch::serialize::InputArchive sp_archive;
        if (archive.try_read("style_projection", sp_archive)) {
            style_projection->load(sp_archive);
        }
    }
}

// CA_MOEExpertSystemImpl实现
CA_MOEExpertSystemImpl::CA_MOEExpertSystemImpl(int num_experts, int board_size,
                                               float dropout_rate, int hidden_dim)
    : num_experts(num_experts), board_size(board_size),
      hidden_dim(hidden_dim), is_training(false) {

    // 创建共享主干网络，直接使用shared_ptr<Impl>，通道宽度由hidden_dim决定（参数大头）
    auto backbone_impl = std::make_shared<SharedBackboneImpl>(17, 10, hidden_dim, dropout_rate);
    register_module("backbone", backbone_impl);
    backbone = backbone_impl;

    // 创建专家策略头和价值头，输入维度=主干输出通道数(即hidden_dim)
    for (int i = 0; i < num_experts; ++i) {
        auto policy_head = std::make_shared<ExpertPolicyHeadImpl>(hidden_dim, board_size, dropout_rate);
        register_module("policy_head_" + std::to_string(i), policy_head);
        policy_heads.push_back(policy_head);

        auto value_head = std::make_shared<ExpertValueHeadImpl>(hidden_dim, dropout_rate);
        register_module("value_head_" + std::to_string(i), value_head);
        value_heads.push_back(value_head);
    }

    // 创建门控网络，输入维度=主干输出通道数(即hidden_dim)
    auto gating_impl = std::make_shared<GatingNetworkImpl>(hidden_dim, num_experts, dropout_rate);
    register_module("gating_network", gating_impl);
    gating_network = gating_impl;
}

CA_MOEInferenceResult CA_MOEExpertSystemImpl::forward(torch::Tensor batch_states) {
    // 加锁保护：多线程环境下序列化推理调用
    // 多个SelfPlayGenerator worker共享同一模型时，需要防止并发forward导致的内存竞争
    std::lock_guard<std::mutex> lock(inference_mutex_);

    CA_MOEInferenceResult result;

    if (!batch_states.defined()) {
        throw std::runtime_error("forward: batch_states未定义");
    }

    // 确保当前线程的CUDA context已初始化
    // 如果batch_states在GPU上，当前线程必须有正确的CUDA context才能操作
    if (batch_states.is_cuda() && torch::cuda::is_available()) {
        // 通过c10::cuda::CUDACachingAllocator的lazy初始化机制
        // 执行一个简单的CUDA操作来触发当前线程的CUDA context初始化
        static thread_local bool cuda_ctx_init = false;
        if (!cuda_ctx_init) {
            // 在当前设备上创建一个临时tensor来初始化CUDA context
            auto _ = torch::zeros({1}, batch_states.device());
            cuda_ctx_init = true;
        }
    }

    auto features = backbone->forward(batch_states);
    // 门控前向同时返回logits（softmax前），供训练时L2正则防坍缩
    // （历史教训：仅对softmax输出做KL/熵正则，门控one-hot后梯度被softmax雅可比
    //   饱和区阻断，正则完全失效；对logits直接L2正则梯度始终存在）
    auto gating_pair = gating_network->forward_with_logits(features);
    result.gating_weights = gating_pair.first;
    result.gating_logits = gating_pair.second;

    for (int i = 0; i < num_experts; ++i) {
        result.expert_policies.push_back(policy_heads[i]->forward(features));
        result.expert_values.push_back(value_heads[i]->forward(features));
    }

    auto batch_size = batch_states.size(0);
    auto action_size = board_size * board_size + 1;

    result.final_policy = torch::zeros({batch_size, action_size}, batch_states.options());
    result.final_value = torch::zeros({batch_size, 1}, batch_states.options());

    for (int i = 0; i < num_experts; ++i) {
        auto weight = result.gating_weights.slice(1, i, i + 1).unsqueeze(2);
        result.final_policy += result.expert_policies[i].exp() * weight.squeeze(2);
        result.final_value += result.expert_values[i] * weight.squeeze(2);
    }

    result.final_policy = result.final_policy / result.final_policy.sum(1, true);

    // 推理模式（eval）下将tensor显式clone到CPU：
    // 确保返回的tensor独立于GPU存储，避免CUDA caching allocator回收导致的存储失效
    // 同时避免跨线程CUDA context问题（返回的tensor全部在CPU上，后续操作不需要CUDA context）
    // 训练模式下不clone到CPU：需要保留GPU上的计算图以支持backward()梯度反传
    if (!is_training) {
        result.final_policy = result.final_policy.to(torch::kCPU).clone();
        result.final_value = result.final_value.to(torch::kCPU).clone();
        result.gating_weights = result.gating_weights.to(torch::kCPU).clone();
        for (auto& p : result.expert_policies) {
            p = p.to(torch::kCPU).clone();
        }
        for (auto& v : result.expert_values) {
            v = v.to(torch::kCPU).clone();
        }
    }

    return result;
}

std::vector<CA_MOEInferenceResult> CA_MOEExpertSystemImpl::batch_forward(
    const std::vector<torch::Tensor>& states) {

    // 使用torch::stack替代逐个unsqueeze+cat，减少中间临时tensor分配
    auto batch_tensor = torch::stack(states, 0);

    auto batch_result = forward(batch_tensor);

    // 使用clone()创建独立tensor，避免索引视图(view)依赖batch_result的底层存储
    // batch_result在函数返回后被销毁，视图会变为dangling pointer导致崩溃
    std::vector<CA_MOEInferenceResult> results;
    results.reserve(states.size());
    for (size_t i = 0; i < states.size(); ++i) {
        CA_MOEInferenceResult single_result;
        single_result.expert_policies.reserve(num_experts);
        single_result.expert_values.reserve(num_experts);
        for (int j = 0; j < num_experts; ++j) {
            single_result.expert_policies.push_back(batch_result.expert_policies[j][i].clone());
            single_result.expert_values.push_back(batch_result.expert_values[j][i].clone());
        }
        single_result.gating_weights = batch_result.gating_weights[i].clone();
        single_result.final_policy = batch_result.final_policy[i].clone();
        single_result.final_value = batch_result.final_value[i].clone();
        results.push_back(std::move(single_result));
    }

    return results;
}

// ==================== 对手风格条件化（对外接口实现） ====================
// 竞争对弈时由训练器注入对方实时画像；带推理互斥锁，防止与并发推理竞争。
void CA_MOEExpertSystemImpl::set_opponent_style(const std::vector<float>& style) {
    std::lock_guard<std::mutex> lock(inference_mutex_);
    gating_network->set_opponent_style(style);
}

void CA_MOEExpertSystemImpl::clear_opponent_style() {
    std::lock_guard<std::mutex> lock(inference_mutex_);
    gating_network->clear_opponent_style();
}

void CA_MOEExpertSystemImpl::save(const std::string& path) {
    try {
        // 检查路径有效性
        if (path.empty()) {
            throw std::invalid_argument("模型保存路径不能为空");
        }

        // 使用OutputArchive保存所有组件
        torch::serialize::OutputArchive archive;

        // 保存主干网络
        torch::serialize::OutputArchive backbone_archive;
        backbone->save(backbone_archive);
        archive.write("backbone", backbone_archive);

        // 保存门控网络
        torch::serialize::OutputArchive gating_archive;
        gating_network->save(gating_archive);
        archive.write("gating_network", gating_archive);

        // 保存所有专家策略头和价值头
        for (int i = 0; i < num_experts; ++i) {
            torch::serialize::OutputArchive policy_archive;
            policy_heads[i]->save(policy_archive);
            archive.write("policy_head_" + std::to_string(i), policy_archive);

            torch::serialize::OutputArchive value_archive;
            value_heads[i]->save(value_archive);
            archive.write("value_head_" + std::to_string(i), value_archive);
        }

        // 保存配置参数
        archive.write("num_experts", torch::IValue(num_experts));
        archive.write("board_size", torch::IValue(board_size));
        archive.write("hidden_dim", torch::IValue(hidden_dim));
        archive.write("is_training", torch::IValue(is_training));

        // 保存到文件
        archive.save_to(path);

        LOG_INFO("CA_MOEExpertSystem模型已成功保存到: %s (专家数: %d, 棋盘大小: %d)",
                 path.c_str(), num_experts, board_size);

    } catch (const std::invalid_argument& e) {
        LOG_ERROR("模型保存参数错误: %s", e.what());
        throw;
    } catch (const torch::Error& e) {
        LOG_ERROR("PyTorch序列化错误: %s", e.what());
        throw;
    } catch (const std::exception& e) {
        LOG_ERROR("保存模型时发生未知错误: %s", e.what());
        throw;
    }
}

void CA_MOEExpertSystemImpl::load(const std::string& path) {
    try {
        // 检查路径有效性
        if (path.empty()) {
            throw std::invalid_argument("模型加载路径不能为空");
        }

        // 检查文件是否存在
        std::ifstream file(path);
        if (!file.good()) {
            throw std::runtime_error("模型文件不存在或无法访问: " + path);
        }
        file.close();

        // 使用InputArchive加载数据
        torch::serialize::InputArchive archive;
        archive.load_from(path);

        // 先读取并校验配置参数（含hidden_dim），再加载子模块：
        // 防"256通道checkpoint配128小模型"时子模块load对shape不匹配静默resize破坏模型。
        torch::IValue num_experts_ivalue, board_size_ivalue, is_training_ivalue;
        archive.read("num_experts", num_experts_ivalue);
        archive.read("board_size", board_size_ivalue);
        archive.read("is_training", is_training_ivalue);

        int64_t saved_num_experts = num_experts_ivalue.toInt();
        int64_t saved_board_size = board_size_ivalue.toInt();
        bool saved_is_training = is_training_ivalue.toBool();

        // hidden_dim：旧checkpoint无此字段，默认按256处理（历史checkpoint均为256通道）
        int64_t saved_hidden_dim = 256;
        torch::IValue hidden_dim_ivalue;
        if (archive.try_read("hidden_dim", hidden_dim_ivalue)) {
            saved_hidden_dim = hidden_dim_ivalue.toInt();
        }

        if (saved_num_experts != num_experts) {
            throw std::runtime_error(
                "专家数量不匹配: 文件中保存的专家数=" +
                std::to_string(saved_num_experts) +
                ", 当前模型配置的专家数=" + std::to_string(num_experts));
        }
        if (saved_board_size != board_size) {
            throw std::runtime_error(
                "棋盘大小不匹配: 文件中保存的棋盘大小=" +
                std::to_string(saved_board_size) +
                ", 当前模型配置的棋盘大小=" + std::to_string(board_size));
        }
        if (saved_hidden_dim != hidden_dim) {
            throw std::runtime_error(
                "通道数(hidden_dim)不匹配: 文件中保存的通道数=" +
                std::to_string(saved_hidden_dim) +
                ", 当前模型配置的通道数=" + std::to_string(hidden_dim));
        }

        // 加载主干网络
        torch::serialize::InputArchive backbone_archive;
        archive.read("backbone", backbone_archive);
        backbone->load(backbone_archive);

        // 加载门控网络（兼容新旧checkpoint：旧版无style_projection，保持零初始化）
        torch::serialize::InputArchive gating_archive;
        archive.read("gating_network", gating_archive);
        gating_network->load_compat(gating_archive);

        // 加载所有专家头参数
        for (int i = 0; i < num_experts; ++i) {
            torch::serialize::InputArchive policy_archive;
            archive.read("policy_head_" + std::to_string(i), policy_archive);
            policy_heads[i]->load(policy_archive);

            torch::serialize::InputArchive value_archive;
            archive.read("value_head_" + std::to_string(i), value_archive);
            value_heads[i]->load(value_archive);
        }

        // 恢复训练模式
        is_training = saved_is_training;
        if (is_training) {
            set_train_mode();
        } else {
            set_eval_mode();
        }

        LOG_INFO("CA_MOEExpertSystem模型已成功从 %s 加载 (专家数: %d, 棋盘大小: %d, 训练模式: %s)",
                 path.c_str(), num_experts, board_size, is_training ? "是" : "否");

    } catch (const std::invalid_argument& e) {
        LOG_ERROR("模型加载参数错误: %s", e.what());
        throw;
    } catch (const std::runtime_error& e) {
        LOG_ERROR("模型加载运行时错误: %s", e.what());
        throw;
    } catch (const torch::Error& e) {
        LOG_ERROR("PyTorch反序列化错误: %s", e.what());
        throw;
    } catch (const std::exception& e) {
        LOG_ERROR("加载模型时发生未知错误: %s", e.what());
        throw;
    }
}

std::vector<torch::Tensor> CA_MOEExpertSystemImpl::get_expert_parameters(int expert_idx) {
    if (expert_idx < 0 || expert_idx >= num_experts) {
        throw std::out_of_range("Expert index out of range");
    }

    // 直接返回当前设备的tensor，不做CPU迁移（避免GPU→CPU→GPU开销和Sandbox杀进程）
    std::vector<torch::Tensor> params;
    for (const auto& p : policy_heads[expert_idx]->parameters()) {
        params.push_back(p);
    }
    for (const auto& p : value_heads[expert_idx]->parameters()) {
        params.push_back(p);
    }
    return params;
}

void CA_MOEExpertSystemImpl::set_expert_parameters(int expert_idx, const std::vector<torch::Tensor>& params) {
    if (expert_idx < 0 || expert_idx >= num_experts) {
        throw std::out_of_range("Expert index out of range");
    }

    auto policy_params = policy_heads[expert_idx]->parameters();
    auto value_params = value_heads[expert_idx]->parameters();

    size_t offset = 0;
    for (size_t i = 0; i < policy_params.size(); ++i) {
        policy_params[i].data().copy_(params[offset++]);
    }
    for (size_t i = 0; i < value_params.size(); ++i) {
        value_params[i].data().copy_(params[offset++]);
    }
}

std::vector<torch::Tensor> CA_MOEExpertSystemImpl::get_shared_parameters() {
    // 直接返回当前设备的tensor，不做CPU迁移（避免GPU→CPU→GPU开销和Sandbox杀进程）
    std::vector<torch::Tensor> params;
    for (const auto& p : backbone->parameters()) {
        params.push_back(p);
    }
    for (const auto& p : gating_network->parameters()) {
        params.push_back(p);
    }
    return params;
}

void CA_MOEExpertSystemImpl::set_shared_parameters(const std::vector<torch::Tensor>& params) {
    auto backbone_params = backbone->parameters();
    auto gating_params = gating_network->parameters();

    size_t offset = 0;
    for (size_t i = 0; i < backbone_params.size(); ++i) {
        backbone_params[i].data().copy_(params[offset++]);
    }
    for (size_t i = 0; i < gating_params.size(); ++i) {
        gating_params[i].data().copy_(params[offset++]);
    }
}

void CA_MOEExpertSystemImpl::set_eval_mode() {
    is_training = false;
    this->eval();
}

void CA_MOEExpertSystemImpl::set_train_mode() {
    is_training = true;
    this->train();
}

// ==================== 对手画像采集器实现 ====================
// 统计规则见 opponent_style_profiler.h 顶部注释（线位判实地/厚势，邻接判攻击/防守/均衡）

void OpponentStyleProfiler::record_move(int edge_dist, bool has_self, bool has_opp) {
    total_moves_++;
    // 线位特征：第1~3线（边缘距离≤2）→ 实地；第4线及以上 → 厚势
    if (edge_dist <= 2) {
        territorial_count_++;
    } else {
        influence_count_++;
    }
    // 邻接特征：贴近对方 → 攻击；仅贴近己方 → 防守；双方皆无 → 均衡
    if (has_opp) {
        aggressive_count_++;
    } else if (has_self) {
        defensive_count_++;
    } else {
        balanced_count_++;
    }
}

OpponentStyleVector OpponentStyleProfiler::get_style() const {
    OpponentStyleVector style;
    if (total_moves_ <= 0) {
        // 无样本时返回均衡原型，避免零向量导致门控偏置无效
        style = prototype_styles()[2];
        return style;
    }
    float inv = 1.0f / static_cast<float>(total_moves_);
    style.aggressive  = static_cast<float>(aggressive_count_) * inv;
    style.defensive   = static_cast<float>(defensive_count_) * inv;
    style.balanced    = static_cast<float>(balanced_count_) * inv;
    style.territorial = static_cast<float>(territorial_count_) * inv;
    style.influence   = static_cast<float>(influence_count_) * inv;
    return style;
}

void OpponentStyleProfiler::reset() {
    aggressive_count_ = 0;
    defensive_count_ = 0;
    balanced_count_ = 0;
    territorial_count_ = 0;
    influence_count_ = 0;
    total_moves_ = 0;
}

// 五种棋风原型：极攻击 / 极防守 / 均衡 / 极实地 / 极厚势
const std::array<OpponentStyleVector, 5>& OpponentStyleProfiler::prototype_styles() {
    static const std::array<OpponentStyleVector, 5> prototypes = {{
        {0.9f, 0.1f, 0.3f, 0.4f, 0.5f},  // 极攻击：贴身作战为主
        {0.1f, 0.9f, 0.3f, 0.5f, 0.4f},  // 极防守：补强稳健为主
        {0.5f, 0.5f, 0.5f, 0.5f, 0.5f},  // 均衡：各风格并重
        {0.3f, 0.4f, 0.2f, 0.9f, 0.1f},  // 极实地：低线抢地为主
        {0.4f, 0.3f, 0.2f, 0.1f, 0.9f},  // 极厚势：高线张势为主
    }};
    return prototypes;
}

// 训练增强：随机选一个原型风格，每维叠加 ±0.15 噪声，模拟"同一局面面对不同棋风对手"
OpponentStyleVector OpponentStyleProfiler::sample_training_style(std::mt19937& rng) {
    static const std::array<OpponentStyleVector, 5>& prototypes = prototype_styles();
    std::uniform_int_distribution<int> pick(0, static_cast<int>(prototypes.size()) - 1);
    std::uniform_real_distribution<float> noise(-0.15f, 0.15f);

    OpponentStyleVector style = prototypes[pick(rng)];
    style.aggressive   += noise(rng);
    style.defensive    += noise(rng);
    style.balanced     += noise(rng);
    style.territorial  += noise(rng);
    style.influence    += noise(rng);
    style.clamp01();  // 截断到 [0,1]
    return style;
}

} // namespace neural
} // namespace go
} // namespace spacego

#endif  // USE_LIBTORCH
