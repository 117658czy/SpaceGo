#include "go/training/diploid_trainer.h"

#ifdef USE_LIBTORCH

#include "go/neural/ca_moe_expert.h"
#include "go/neural/opponent_style_profiler.h"
#include "go/training/unsupervised_self_reinforcement.h"

#ifdef USE_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

#include <filesystem>
#include <unordered_map>
#include <future>

// 竞争对弈推理互斥锁
namespace spacego {
namespace go {
namespace training {
namespace {
    std::mutex eval_inference_mutex;
}
} } }

namespace spacego {
namespace go {
namespace training {

// ==================== 构造与初始化 ====================

DiploidTrainer::DiploidTrainer(const DiploidConfig& config)
    : config_(config), device_(torch::kCPU), global_rng_(config.seed) {
    if (torch::cuda::is_available()) {
        device_ = torch::Device(torch::kCUDA, 0);
    }
    // 设置默认输出目录
    if (config_.output_dir.empty()) {
        const_cast<DiploidConfig&>(config_).output_dir = "./training_output/diploid";
    }
    if (config_.checkpoint_dir.empty()) {
        const_cast<DiploidConfig&>(config_).checkpoint_dir = config_.output_dir + "/checkpoints";
    }
    if (config_.log_dir.empty()) {
        const_cast<DiploidConfig&>(config_).log_dir = config_.output_dir + "/logs";
    }
}

DiploidTrainer::~DiploidTrainer() {
    // 显式定义析构函数，确保在.cpp中销毁EfficientReplayBuffer（需要SumTreeImpl完整类型）
    replay_a_.reset();
    replay_b_.reset();
    trainer_a_.reset();
    trainer_b_.reset();
}

bool DiploidTrainer::initialize() {
    try {
        std::filesystem::create_directories(config_.output_dir);
        std::filesystem::create_directories(config_.checkpoint_dir);
        std::filesystem::create_directories(config_.log_dir);

        // 创建三个CA-MOE模型实例（8专家，19×19棋盘，通道宽度由hidden_dim控制）
        model_a_ = std::make_shared<neural::CA_MOEExpertSystem>(
            config_.num_experts, config_.board_size, config_.dropout_rate, config_.hidden_dim);
        model_b_ = std::make_shared<neural::CA_MOEExpertSystem>(
            config_.num_experts, config_.board_size, config_.dropout_rate, config_.hidden_dim);
        gen0_model_ = std::make_shared<neural::CA_MOEExpertSystem>(
            config_.num_experts, config_.board_size, config_.dropout_rate, config_.hidden_dim);

        // 移动到设备
        // 注意：gen0_model_保持CPU常驻！它是ELO评估基线，只在每代末尾的ELO竞争时
        // 临时迁移到GPU（见run_generation末尾）。训练阶段三模型同驻GPU会挤占显存，
        // 导致经验池回退CPU存储模式（显著拖慢训练），故gen0默认留在CPU。
        (*model_a_)->to(device_);
        (*model_b_)->to(device_);
        (*model_a_)->set_eval_mode();
        (*model_b_)->set_eval_mode();
        (*gen0_model_)->set_eval_mode();

        // 尝试从已有checkpoint恢复模型和ELO
        resumed_gen_ = -1;
        // 优先查找最终的A模型（训练完成后的保存）
        std::string final_a_path = config_.checkpoint_dir + "/diploid_final_a.pt";
        std::string gen0_path = config_.checkpoint_dir + "/gen0_baseline.pt";

        if (std::filesystem::exists(final_a_path)) {
            std::cout << "[恢复] 发现最终模型: " << final_a_path << std::endl;
            try {
                (*model_a_)->load(final_a_path);
                (*model_a_)->to(device_);
                (*model_a_)->set_eval_mode();
                // B模型也加载（如果存在）
                std::string final_b_path = config_.checkpoint_dir + "/diploid_final_b.pt";
                if (std::filesystem::exists(final_b_path)) {
                    (*model_b_)->load(final_b_path);
                    (*model_b_)->to(device_);
                    (*model_b_)->set_eval_mode();
                } else {
                    copy_model_parameters(model_a_, model_b_);
                }
                resumed_gen_ = 0; // 标记为已恢复
                std::cout << "[恢复] 成功加载最终模型" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[警告] 加载最终模型失败: " << e.what() << "，从头开始" << std::endl;
            }
        } else {
            // 查找最新的代际checkpoint
            int latest_gen = -1;
            for (int g = config_.max_generations; g >= 1; --g) {
                std::string path_a = config_.checkpoint_dir + "/diploid_a_gen" + std::to_string(g) + ".pt";
                if (std::filesystem::exists(path_a)) {
                    latest_gen = g;
                    break;
                }
            }
            if (latest_gen > 0) {
                std::string path_a = config_.checkpoint_dir + "/diploid_a_gen" + std::to_string(latest_gen) + ".pt";
                std::string path_b = config_.checkpoint_dir + "/diploid_b_gen" + std::to_string(latest_gen) + ".pt";
                std::cout << "[恢复] 发现第" << latest_gen << "代checkpoint" << std::endl;
                try {
                    (*model_a_)->load(path_a);
                    (*model_a_)->to(device_);
                    (*model_a_)->set_eval_mode();
                    if (std::filesystem::exists(path_b)) {
                        (*model_b_)->load(path_b);
                        (*model_b_)->to(device_);
                        (*model_b_)->set_eval_mode();
                    } else {
                        copy_model_parameters(model_a_, model_b_);
                    }
                    resumed_gen_ = latest_gen;
                    std::cout << "[恢复] 成功加载第" << latest_gen << "代模型" << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "[警告] 加载第" << latest_gen << "代模型失败: " << e.what() << "，从头开始" << std::endl;
                    resumed_gen_ = -1;
                }
            }
        }

        // 恢复gen0基线模型（用于ELO计算）
        // gen0保持CPU常驻（ELO基线按需迁移GPU，见run_generation末尾）
        if (resumed_gen_ >= 0 && std::filesystem::exists(gen0_path)) {
            try {
                (*gen0_model_)->load(gen0_path);
                (*gen0_model_)->set_eval_mode();
                std::cout << "[恢复] 加载Gen0基线模型" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[警告] 加载Gen0基线失败: " << e.what() << std::endl;
            }
        }

        // 恢复ELO历史（从CSV文件）
        if (resumed_gen_ >= 0) {
            std::string csv_path = config_.output_dir + "/diploid_training_history.csv";
            if (std::filesystem::exists(csv_path)) {
                std::ifstream csv_file(csv_path);
                std::string line;
                best_elo_ = 0.0f;
                best_gen_ = 0;
                // 跳过表头
                std::getline(csv_file, line);
                while (std::getline(csv_file, line)) {
                    if (line.empty()) continue;
                    // 解析CSV行的最后一个ELO字段
                    std::istringstream iss(line);
                    std::string token;
                    std::vector<std::string> fields;
                    while (std::getline(iss, token, ',')) {
                        fields.push_back(token);
                    }
                    // 字段: generation,...,winner,elo,minutes
                    // ELO位于索引15（CSV表头：...param_distance,winner,elo,minutes）
                    if (fields.size() >= 17) {
                        float elo = std::stof(fields[15]);
                        int gen_val = std::stoi(fields[0]);
                        if (elo > best_elo_) {
                            best_elo_ = elo;
                            best_gen_ = gen_val;
                        }
                    }
                }
                std::cout << "[恢复] 历史最佳ELO: " << best_elo_ << " (第" << best_gen_ << "代)" << std::endl;
            }
            // 历史最高ELO作为峰值起点（最佳模型替换判定的基准）
            peak_elo_ = best_elo_;
        }

        // 统计参数数量
        int total_params = 0;
        for (const auto& p : (*model_a_)->parameters()) {
            total_params += static_cast<int>(p.numel());
        }

        std::cout << "\n========================================" << std::endl;
        std::cout << "  二倍体分化竞争训练系统（生产级）" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "  模型: CA-MOE (" << config_.num_experts << "专家)" << std::endl;
        std::cout << "  参数量: " << (total_params / 1000) << "K" << std::endl;
        std::cout << "  设备: " << (device_.is_cuda() ? "CUDA" : "CPU") << std::endl;
        if (device_.is_cuda()) {
            auto props = at::cuda::getDeviceProperties(0);
            std::cout << "  GPU: " << props->name
                      << " (" << (props->totalGlobalMem / 1024 / 1024 / 1024) << "GB)" << std::endl;
        }
        std::cout << "  每代游戏: " << config_.games_per_gen << std::endl;
        std::cout << "  竞争游戏: " << config_.competition_games << std::endl;
        std::cout << "  经验池容量: " << config_.replay_buffer_capacity << std::endl;
        std::cout << "  输出目录: " << config_.output_dir << std::endl;
        if (resumed_gen_ >= 0) {
            std::cout << "  [恢复模式] 从已有checkpoint继续" << std::endl;
        }
        std::cout << "========================================\n" << std::endl;

        // 初始化评估搜索器（竞争对弈/ELO评估用HMGS-DP搜索，与自对弈训练目标一致）
        init_eval_searcher();

        return true;
    } catch (const std::exception& e) {
        std::cerr << "[错误] 初始化失败: " << e.what() << std::endl;
        return false;
    }
}

// ==================== 模型参数操作 ====================

void DiploidTrainer::copy_model_parameters(
    std::shared_ptr<neural::CA_MOEExpertSystem> src,
    std::shared_ptr<neural::CA_MOEExpertSystem> dst) {
    torch::NoGradGuard no_grad;
    // 跳过自复制（src==dst时无需操作）
    if (src == dst) {
        return;
    }

    // 复制共享参数（backbone + gating）— 已在当前设备上，无需CPU迁移
    auto shared_params = (*src)->get_shared_parameters();
    (*dst)->set_shared_parameters(shared_params);

    // 复制每个专家参数
    int num_experts = (*src)->get_num_experts();
    for (int i = 0; i < num_experts; ++i) {
        auto expert_params = (*src)->get_expert_parameters(i);
        (*dst)->set_expert_parameters(i, expert_params);
    }

    // 复制buffers（如BatchNorm的running_mean），用unordered_map优化匹配
    auto src_bufs = (*src)->named_buffers();
    std::unordered_map<std::string, torch::Tensor> src_buf_map;
    for (auto& sb : src_bufs) {
        src_buf_map[sb.key()] = sb.value();
    }
    auto dst_bufs = (*dst)->named_buffers();
    for (auto& db : dst_bufs) {
        auto it = src_buf_map.find(db.key());
        if (it != src_buf_map.end()) {
            db.value().copy_(it->second);
        }
    }
}

void DiploidTrainer::add_parameter_noise(
    std::shared_ptr<neural::CA_MOEExpertSystem> model, float stddev) {
    torch::NoGradGuard no_grad;
    // 直接在GPU上添加噪声，不做CPU迁移（避免Sandbox杀进程）
    for (auto& p : (*model)->parameters()) {
        if (!p.requires_grad()) continue;
        auto noise = torch::randn_like(p) * stddev;
        p.add_(noise);
    }
}

float DiploidTrainer::compute_parameter_distance(
    std::shared_ptr<neural::CA_MOEExpertSystem> model_a,
    std::shared_ptr<neural::CA_MOEExpertSystem> model_b) {
    torch::NoGradGuard no_grad;
    float norm_self_sq = 0.0f;
    float norm_diff_sq = 0.0f;

    // 比较共享参数
    auto shared_a = (*model_a)->get_shared_parameters();
    auto shared_b = (*model_b)->get_shared_parameters();
    for (size_t i = 0; i < shared_a.size() && i < shared_b.size(); ++i) {
        auto diff = shared_a[i] - shared_b[i];
        norm_diff_sq += diff.pow(2).sum().item<float>();
        norm_self_sq += shared_a[i].pow(2).sum().item<float>();
    }

    // 比较专家参数
    int num_experts = (*model_a)->get_num_experts();
    for (int e = 0; e < num_experts; ++e) {
        auto expert_a = (*model_a)->get_expert_parameters(e);
        auto expert_b = (*model_b)->get_expert_parameters(e);
        for (size_t i = 0; i < expert_a.size() && i < expert_b.size(); ++i) {
            auto diff = expert_a[i] - expert_b[i];
            norm_diff_sq += diff.pow(2).sum().item<float>();
            norm_self_sq += expert_a[i].pow(2).sum().item<float>();
        }
    }

    if (norm_self_sq < 1e-10f) return 0.0f;
    return std::sqrt(norm_diff_sq / norm_self_sq);
}

// ==================== 自对弈数据生成 ====================

std::vector<GameData> DiploidTrainer::generate_self_play_data(
    std::shared_ptr<neural::CA_MOEExpertSystem> model,
    int num_games, int seed, float temperature_decay) {
    // 配置SelfPlay
    SelfPlayConfig sp_config;
    sp_config.num_workers = config_.num_self_play_workers;
    sp_config.simulations_per_move = config_.simulations_per_move;
    sp_config.batch_size = 64;
    sp_config.temperature = 1.0f;
    sp_config.temperature_decay = temperature_decay;
    sp_config.min_temperature = 0.1f;
    sp_config.dirichlet_alpha = 0.03f;
    sp_config.dirichlet_weight = 0.25f;
    sp_config.max_game_length = config_.max_game_length;
    sp_config.use_cuda = device_.is_cuda();
    sp_config.use_mixed_precision = config_.use_mixed_precision;
    sp_config.seed = seed;
    sp_config.save_sgf_games = false;
    // 修复"数据增强溢出"：预存8对称增强使每代数据量膨胀8倍（40局×253步×8≈8.1万步），
    // 远超经验池容量15000。环形缓冲覆盖后仅最后约7局数据保留，前32局训练数据全丢，
    // 且8倍膨胀让每代CPU峰值内存达~2GB（Windows堆释放后RSS不回落→内存随代次累积）。
    // 关闭预存增强后：40局×253步≈1万步 < 15000容量，40局数据全部入池，
    // 数据覆盖更完整，内存峰值降至~250MB。（GAE计算按stride=1自动兼容）
    sp_config.prestore_augmentation = false;
    sp_config.enable_multi_dim_eval = true;
    sp_config.enable_gae = true;

    // 多Worker并行自对弈：利用多核CPU并行搜索（GPU推理由全局互斥锁串行化）。
    // 每Worker独立SelfPlayGenerator（独立搜索树/评估缓存/多元评价器），共享同一模型。
    // GPU已饱和时并行收益来自CPU侧（select/expand/特征编码）被多核摊平。
    int num_workers = std::max(1, config_.num_self_play_workers);
    if (num_workers <= 1 || num_games <= 0) {
        // 单Worker路径（兼容并行关闭的情况）
        SelfPlayGenerator generator(0, sp_config, model, nullptr);
        if (!generator.initialize()) {
            std::cerr << "[错误] SelfPlayGenerator初始化失败" << std::endl;
            return {};
        }
        return generator.generate_games(seed * 1000, num_games);
    }

    // 将num_games均分给各Worker（余数分配给前extra个Worker）
    std::vector<std::future<std::vector<GameData>>> futures;
    int base = num_games / num_workers;
    int extra = num_games % num_workers;
    for (int w = 0; w < num_workers; ++w) {
        int cnt = base + (w < extra ? 1 : 0);
        if (cnt <= 0) continue;
        futures.push_back(std::async(std::launch::async,
            [this, model, sp_config, seed, w, cnt]() {
                SelfPlayGenerator generator(w, sp_config, model, nullptr);
                if (!generator.initialize()) {
                    std::cerr << "[错误] Worker-" << w << " SelfPlayGenerator初始化失败" << std::endl;
                    return std::vector<GameData>{};
                }
                // 各Worker使用不同起始ID段，避免游戏ID冲突
                return generator.generate_games(seed * 1000 + w * 100000, cnt);
            }));
    }

    // 合并各Worker生成的数据（顺序不影响训练，经验池随机采样）
    std::vector<GameData> all_games;
    for (auto& f : futures) {
        auto games = f.get();
        all_games.insert(all_games.end(), games.begin(), games.end());
    }
    return all_games;
}

// ==================== 训练组件 ====================

void DiploidTrainer::setup_training_components(
    std::shared_ptr<neural::CA_MOEExpertSystem> model,
    std::shared_ptr<EfficientReplayBuffer>& buffer,
    std::shared_ptr<MixedPrecisionTrainer>& trainer) {
    // 确保模型在GPU上且参数requires_grad=true
    (*model)->to(device_);
    for (auto& p : (*model)->parameters()) {
        p.set_requires_grad(true);
    }

    // 配置训练参数
    TrainingConfig train_config;
    train_config.batch_size = config_.batch_size;
    train_config.learning_rate = config_.learning_rate;
    train_config.lr_decay_rate = config_.lr_decay_rate;
    train_config.weight_decay = config_.weight_decay;
    train_config.policy_loss_weight = config_.policy_loss_weight;
    train_config.value_loss_weight = config_.value_loss_weight;
    train_config.gating_loss_weight = config_.gating_loss_weight;
    train_config.use_mixed_precision = config_.use_mixed_precision;
    train_config.gradient_clipping = true;
    train_config.max_gradient_norm = 3.0f;
    train_config.window_size = config_.replay_buffer_capacity;
    train_config.progressive_expansion = config_.progressive_expansion;
    train_config.initial_window_size = config_.initial_buffer_size;
    train_config.max_window_size = config_.replay_buffer_capacity;
    train_config.expansion_factor = 2.0f;
    train_config.expansion_interval = 10;
    train_config.dropout_rate = config_.dropout_rate;
    train_config.label_smoothing = 0.1f;
    train_config.use_data_augmentation = true;
    train_config.validation_ratio = config_.validation_ratio;
    // 二倍体训练使用较大T_max避免余弦退火过早将学习率降到0
    // 每代约62步×3epochs=186步，设T_max=5000足够
    train_config.num_epochs_per_iteration = 50;

    // 经验池：复用已有buffer并清空，避免每代重建导致GPU显存累积不释放
    if (buffer) {
        buffer->clear();
        std::cout << "[经验池] 清空已有经验池，准备加载当代数据" << std::endl;
    } else {
        // 首次创建经验池
        buffer = std::make_shared<EfficientReplayBuffer>(
            config_.initial_buffer_size, config_.num_experts);
        if (device_.is_cuda()) {
            buffer->init_gpu_storage(device_, config_.replay_buffer_capacity, config_.num_experts);
        }
        std::cout << "[经验池] 创建新经验池 (容量=" << config_.replay_buffer_capacity << ")" << std::endl;
    }

    // 每代重建训练器（因为优化器状态与模型参数绑定，模型参数变化后需重建）
    trainer = std::make_shared<MixedPrecisionTrainer>(model, train_config);
    trainer->setup_optimizer();
    // 打印验证学习率是否正确设置
    std::cout << "[验证] 训练器初始学习率: " << std::setprecision(6) << trainer->get_learning_rate() << std::endl;
}

std::tuple<float, float, float, float> DiploidTrainer::train_model_generation(
    std::shared_ptr<neural::CA_MOEExpertSystem> model,
    std::shared_ptr<EfficientReplayBuffer>& buffer,
    std::shared_ptr<MixedPrecisionTrainer>& trainer) {
    if (buffer->size() < buffer->get_min_samples()) {
        return {0.0f, 0.0f, 0.0f, 0.0f};
    }

    (*model)->set_train_mode();

    float avg_policy = 0.0f, avg_value = 0.0f, avg_gating = 0.0f, avg_total = 0.0f;
    int epochs = config_.training_epochs_per_gen;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto result = trainer->train_epoch(*buffer);
        avg_policy += result.avg_policy_loss;
        avg_value += result.avg_value_loss;
        avg_gating += result.avg_gating_loss;
        avg_total += result.avg_total_loss;
    }

    avg_policy /= epochs;
    avg_value /= epochs;
    avg_gating /= epochs;
    avg_total /= epochs;

    // 验证集评估
    float val_total = 0.0f;
    if (config_.validation_ratio > 0 && buffer->size() > 100) {
        auto val_result = trainer->evaluate_validation(*buffer, 3);
        val_total = val_result.avg_total_loss;
    }

    (*model)->set_eval_mode();

    // 训练结束：清除训练期间注入的对手风格（train_step内50%概率注入的条件信号），
    // 确保模型回归无条件推理——后续自对弈数据生成/推理不受训练期残留风格影响。
    (*model)->clear_opponent_style();

    return {avg_policy, avg_value, avg_gating, avg_total};
}

// ==================== 评估搜索器 ====================

// 竞争对弈/ELO评估使用HMGS-DP搜索（与自对弈训练目标一致），
// 修复原评估"直接取模型原始策略采样"导致的训练-评估mismatch：
// 模型学的是搜索访问分布（强策略），而原评估测的是无搜索的原始策略（弱），
// 导致策略损失持续下降但ELO不涨——评估根本没有测量模型真正学到的棋力。
void DiploidTrainer::init_eval_searcher() {
    eval_searcher_ = std::make_shared<search::HMGSDP_Optimized>();
    search::HMGSDP_Optimized::ParallelSearchConfig ps_config;
    ps_config.num_workers = 1;
    ps_config.batch_size = 64;
    ps_config.use_work_stealing = false;
    ps_config.use_batch_inference = true;
    eval_searcher_->set_parallel_config(ps_config);
}

// 用指定模型对当前局面执行HMGS-DP搜索，返回(访问次数归一化分布[362], 根价值)
// 竞争对弈黑/白模型不同，每次调用将批量评估器重新绑定到当前走子方模型
std::pair<std::vector<float>, float> DiploidTrainer::eval_search(
    std::shared_ptr<neural::CA_MOEExpertSystem> model, const GoBoardCPU& board) {
    std::vector<float> uniform(362, 1.0f / 362.0f);
    if (!eval_searcher_ || !model) return {uniform, 0.0f};
    try {
        // 批量评估器：与SelfPlayGenerator的batch回调一致（模型推理+策略归一化+价值视角转换）
        auto batch_cb = [this, model](const std::vector<GoBoardCPU>& boards,
                                      const std::vector<int>& players) {
            std::vector<std::pair<std::vector<float>, float>> results;
            if (boards.empty()) return results;
            try {
                int n = static_cast<int>(boards.size());
                torch::Tensor states = torch::zeros({n, 17, 19, 19}, torch::kFloat32);
                for (int i = 0; i < n; ++i) {
                    boards[i].encode_to_tensor(states[i].data_ptr<float>(), 17);
                }
                if (device_.is_cuda()) states = states.to(device_);
                torch::NoGradGuard no_grad;
                neural::CA_MOEInferenceResult infer_result;
                {
                    // 锁内推理：与竞争对弈单局推理共用锁，避免与并发推理冲突
                    std::lock_guard<std::mutex> lock(eval_inference_mutex);
                    infer_result = (*model)->forward(states);
                }
                auto policy_batch = infer_result.final_policy.to(torch::kCPU).contiguous();
                auto value_batch = infer_result.final_value.to(torch::kCPU).contiguous();
                const float* policy_ptr = policy_batch.data_ptr<float>();
                const float* value_ptr = value_batch.data_ptr<float>();
                for (int i = 0; i < n; ++i) {
                    std::vector<float> out(362, 0.0f);
                    float sum = 0.0f;
                    for (int j = 0; j < 362; ++j) {
                        out[static_cast<size_t>(j)] = std::max(policy_ptr[i * 362 + j], 0.0f);
                        sum += out[static_cast<size_t>(j)];
                    }
                    if (sum > 1e-8f) {
                        for (auto& v : out) v /= sum;
                    }
                    float val = value_ptr[i];
                    // final_value为黑棋视角，转换为行棋方视角
                    // players: static_cast<int>(StoneColor)，BLACK=1 / WHITE=2
                    if (is_white_player(players[i])) {
                        val = -val;
                    }
                    results.emplace_back(std::move(out), std::clamp(val, -1.0f, 1.0f));
                }
            } catch (...) {
                for (size_t i = 0; i < boards.size(); ++i) {
                    results.emplace_back(std::vector<float>(362, 1.0f / 362.0f), 0.0f);
                }
            }
            return results;
        };
        eval_searcher_->set_batch_evaluator(batch_cb);

        std::string board_str = board.to_string();
        int player = (board.get_current_player() == StoneColor::BLACK) ? 1 : -1;
        eval_searcher_->search(board_str, config_.eval_simulations, 1.414);

        auto policy = eval_searcher_->get_root_policy(19);
        float root_value = static_cast<float>(eval_searcher_->get_root_value());
        float sum = 0.0f;
        for (auto& p : policy) sum += p;
        if (sum < 1e-6f) return {uniform, root_value};
        return {policy, root_value};
    } catch (...) {
        return {uniform, 0.0f};
    }
}

// ==================== 竞争对弈 ====================

float DiploidTrainer::play_competition_game(
    std::shared_ptr<neural::CA_MOEExpertSystem> black_model,
    std::shared_ptr<neural::CA_MOEExpertSystem> white_model,
    std::mt19937& rng) {
    GoBoardCPU board;
    int consecutive_passes = 0;
    float temperature = 0.5f;  // 竞争时使用较低温度

    // ==================== 对手画像采集（专家分化的实战应用） ====================
    // 围棋思维：看对手的棋风下棋——进攻型对手要防守稳健，实地型对手要抢厚势。
    // 双方各维护一个画像采集器，统计各自落子风格；
    // 轮到某方思考前，把"对方"实时画像注入本方门控（对手风格条件化）。
    neural::OpponentStyleProfiler black_profiler;
    neural::OpponentStyleProfiler white_profiler;

    // 记录落子风格特征到落子方画像：
    // 1) 线位（边缘距离）判实地/厚势；2) 邻接窗口（曼哈顿距离≤2，排除落子点本身）
    //    判攻击/防守/均衡。详见 OpponentStyleProfiler 注释。
    auto record_style_move = [](neural::OpponentStyleProfiler& profiler,
                                int x, int y, StoneColor player, const GoBoardCPU& bd) {
        int edge_dist = std::min({x, y, 18 - x, 18 - y});
        bool has_self = false;
        bool has_opp = false;
        for (int dx = -2; dx <= 2; ++dx) {
            for (int dy = -2; dy <= 2; ++dy) {
                if (dx == 0 && dy == 0) continue;            // 排除落子点本身
                if (std::abs(dx) + std::abs(dy) > 2) continue; // 曼哈顿距离≤2窗口
                int nx = x + dx;
                int ny = y + dy;
                if (nx < 0 || nx > 18 || ny < 0 || ny > 18) continue;
                StoneColor stone = bd.get_stone(nx, ny);
                if (stone == player) {
                    has_self = true;
                } else if (stone != StoneColor::EMPTY) {
                    has_opp = true;
                }
            }
        }
        profiler.record_move(edge_dist, has_self, has_opp);
    };

    while (consecutive_passes < 2 && board.get_move_count() < config_.max_game_length) {
        StoneColor current_player = board.get_current_player();
        auto& current_model = (current_player == StoneColor::BLACK) ? black_model : white_model;

        // 对手风格注入：当前走子方感知"对方"的实时画像（黑感知白，白感知黑）
        // 画像随对局推进不断更新——开局画像稀疏时近似均衡原型，中盘/终局逐渐清晰。
        neural::OpponentStyleVector opp_style =
            (current_player == StoneColor::BLACK) ? white_profiler.get_style()
                                                  : black_profiler.get_style();
        auto opp_arr = opp_style.to_array();
        // 注意：current_model 为 shared_ptr<CA_MOEExpertSystem>(ModuleHolder)，
        // 需先解引用再 -> 才能访问 Impl 的 set_opponent_style（ModuleHolder 的
        // operator-> 不会被 shared_ptr 的 -> 自动链式调用）
        (*current_model)->set_opponent_style(
            std::vector<float>(opp_arr.begin(), opp_arr.end()));

        // HMGS-DP搜索评估（与自对弈训练目标一致）：用搜索访问分布替代原始策略
        // 修复训练-评估mismatch：模型学的是搜索分布（强策略），
        // 原评估直接取模型原始策略采样（无搜索弱策略），导致ELO测不到真实棋力
        auto [search_policy_vec, root_value] = eval_search(current_model, board);
        torch::Tensor policy = torch::from_blob(
            const_cast<float*>(search_policy_vec.data()), {362}, torch::kFloat32).clone();

        // 构建合法动作掩码
        auto legal_moves = board.get_legal_moves(current_player);
        std::vector<bool> legal_mask(362, false);
        for (auto& m : legal_moves) {
            int idx = GoBoardCPU::xy_to_index(m.first, m.second);
            legal_mask[idx] = true;
        }
        legal_mask[361] = true;  // pass总是合法

        // 终局强制收束：与自对弈一致，超过300步直接pass、250~300步高权重引导pass，
        // 避免竞争对弈400步截断（截断盘面的面积计分失真，ELO评估失去意义）
        int action = 361;  // 默认pass（超过300步强制pass时直接采用）
        if (board.get_move_count() <= 300) {
            if (board.get_move_count() > 250) {
                int empty_count = 0;
                for (int i = 0; i < 361; ++i) {
                    auto [x, y] = GoBoardCPU::index_to_xy(i);
                    if (board.is_empty(x, y)) empty_count++;
                }
                float pass_min = (empty_count <= 10) ? 0.9f : ((empty_count <= 40) ? 0.7f : 0.5f);
                policy = policy.clone();
                policy[361] = std::max(policy[361].item<float>(), pass_min);
            }

            // 温度采样选择动作
            auto policy_acc = policy.accessor<float, 1>();
            std::vector<float> weights(362, 0.0f);
            float weight_sum = 0.0f;
            for (int i = 0; i < 362; ++i) {
                if (legal_mask[i] && policy_acc[i] > 0) {
                    weights[i] = std::pow(policy_acc[i], 1.0f / temperature);
                    weight_sum += weights[i];
                }
            }

            if (weight_sum > 0) {
                std::uniform_real_distribution<float> dist(0, weight_sum);
                float r = dist(rng);
                float cumsum = 0.0f;
                action = 361;
                for (int i = 0; i < 362; ++i) {
                    cumsum += weights[i];
                    if (r < cumsum) { action = i; break; }
                }
            }
        }

        // 执行动作
        if (action == 361) {
            board.set_current_player(
                (current_player == StoneColor::BLACK) ? StoneColor::WHITE : StoneColor::BLACK);
            consecutive_passes++;
        } else {
            auto pos = GoBoardCPU::index_to_xy(action);
            auto result = board.make_move(pos.first, pos.second, current_player);
            if (!result.success) {
                action = 361;  // 非法动作fallback到pass
                board.set_current_player(
                    (current_player == StoneColor::BLACK) ? StoneColor::WHITE : StoneColor::BLACK);
                consecutive_passes++;
            } else {
                // 落子成功：把该手风格特征记录到落子方画像（供对方后续感知）
                if (current_player == StoneColor::BLACK) {
                    record_style_move(black_profiler, pos.first, pos.second, current_player, board);
                } else {
                    record_style_move(white_profiler, pos.first, pos.second, current_player, board);
                }
                consecutive_passes = 0;
            }
        }
    }

    // 对局结束：清除双方模型的对手风格，防止画像残留影响后续对局与自对弈推理
    (*black_model)->clear_opponent_style();
    (*white_model)->clear_opponent_style();

    // 计分：中国规则面积计点（棋子 + 领地，黑方贴7.5目）
    // 原实现只数棋子不计领地，会导致模型学习"多下子"而非"围空"，
    // 现改为完整的领地计算，使胜负判定符合真实围棋规则
    auto [black_score, white_score] = board.compute_area_score(config_.komi);

    if (black_score > white_score) return 1.0f;
    if (white_score > black_score) return -1.0f;
    return 0.0f;
}

void DiploidTrainer::run_competition(
    std::shared_ptr<neural::CA_MOEExpertSystem> model_a,
    std::shared_ptr<neural::CA_MOEExpertSystem> model_b,
    DiploidGenerationMetrics& metrics) {
    int half_games = config_.competition_games / 2;

    // 前一半：A执黑，B执白
    for (int i = 0; i < half_games; ++i) {
        float result = play_competition_game(model_a, model_b, global_rng_);
        if (result > 0) metrics.a_wins++;
        else if (result < 0) metrics.b_wins++;
        else metrics.draws++;
    }

    // 后一半：B执黑，A执白（消除先手优势）
    for (int i = 0; i < half_games; ++i) {
        float result = play_competition_game(model_b, model_a, global_rng_);
        if (result < 0) metrics.a_wins++;  // A执白胜 = A胜
        else if (result > 0) metrics.b_wins++;
        else metrics.draws++;
    }

    int total = metrics.a_wins + metrics.b_wins + metrics.draws;
    metrics.a_win_rate = (total > 0) ? static_cast<float>(metrics.a_wins) / total : 0.0f;

    if (metrics.a_win_rate > config_.win_rate_threshold) {
        metrics.winner = 'A';
    } else if ((1.0f - metrics.a_win_rate) > config_.win_rate_threshold) {
        metrics.winner = 'B';
    } else {
        metrics.winner = '-';
    }
}

// ==================== ELO计算 ====================

float DiploidTrainer::compute_elo_change(float win_rate, int num_games) {
    const float K = 32.0f;
    float actual = win_rate;
    float expected = 0.5f;
    return K * (actual - expected);
}

// ==================== 进化循环 ====================

void DiploidTrainer::run_generation_zero() {
    std::cout << "\n========== 第0代 (Gen0): 基础模型预训练 ==========\n" << std::endl;

    // 自对弈
    std::cout << "[Gen0] 自对弈 " << config_.games_per_gen << " 局..." << std::endl;
    auto games = generate_self_play_data(model_a_, config_.games_per_gen, config_.seed, 0.95f);
    std::cout << "  生成 " << games.size() << " 局";

    if (!games.empty()) {
        float avg_len = 0;
        for (auto& g : games) avg_len += g.game_length;
        std::cout << ", 平均长度: " << (avg_len / games.size()) << " 步";
    }
    std::cout << std::endl;

    // 训练
    // 重置输出流格式：自对弈日志使用fixed+precision(1)后未恢复，会导致此处learning_rate
    // 显示为0.0（实为0.001）。显式恢复defaultfloat避免误读。
    std::cout << std::defaultfloat << std::setprecision(6);
    std::cout << "[Gen0] 训练 (epochs=" << config_.gen0_pretrain_epochs
              << ", lr=" << config_.learning_rate << ")..." << std::endl;

    setup_training_components(model_a_, replay_a_, trainer_a_);
    replay_a_->add_games(games);

    // Gen0双倍训练轮数
    int gen0_epochs = config_.gen0_pretrain_epochs;
    (*model_a_)->set_train_mode();
    float avg_policy = 0, avg_value = 0, avg_gating = 0;
    for (int e = 0; e < gen0_epochs; ++e) {
        auto result = trainer_a_->train_epoch(*replay_a_);
        avg_policy += result.avg_policy_loss;
        avg_value += result.avg_value_loss;
        avg_gating += result.avg_gating_loss;
        std::cout << "  Epoch " << (e + 1) << "/" << gen0_epochs
                  << " | 策略: " << std::fixed << std::setprecision(4) << result.avg_policy_loss
                  << " | 价值: " << result.avg_value_loss
                  << " | 门控: " << result.avg_gating_loss << std::endl;
    }
    (*model_a_)->set_eval_mode();

    // Gen0预训练结束：清除训练期注入的对手风格条件，恢复无条件推理
    (*model_a_)->clear_opponent_style();

    avg_policy /= gen0_epochs;
    avg_value /= gen0_epochs;
    avg_gating /= gen0_epochs;
    std::cout << "  Gen0平均 | 策略: " << avg_policy << " | 价值: " << avg_value
              << " | 总: " << (avg_policy + avg_value) << std::endl;

    // 保存Gen0作为ELO基准
    // gen0默认CPU常驻，此处临时迁移GPU完成参数复制+保存（model_a_在GPU上，跨设备直接复制是隐式慢路径）
    (*gen0_model_)->to(device_);
    (*gen0_model_)->set_eval_mode();
    copy_model_parameters(model_a_, gen0_model_);
    (*gen0_model_)->set_eval_mode();
    save_model(gen0_model_, "gen0_baseline");
    // 复制完成后移回CPU释放显存（后续每代ELO竞争前再临时迁移）
    (*gen0_model_)->to(torch::kCPU);
    (*gen0_model_)->set_eval_mode();

    // 创建二倍体副本
    std::cout << "[Gen0] 创建二倍体..." << std::endl;
    copy_model_parameters(model_a_, model_b_);

    // 初始分化
    add_parameter_noise(model_a_, config_.perturbation_std);
    add_parameter_noise(model_b_, config_.perturbation_std);

    float init_dist = compute_parameter_distance(model_a_, model_b_);
    std::cout << "  初始参数距离: " << std::fixed << std::setprecision(3)
              << (init_dist * 100) << "%" << std::endl;
}

DiploidGenerationMetrics DiploidTrainer::run_generation(int gen) {
    DiploidGenerationMetrics m;
    m.generation = gen;

    std::cout << "\n---------- 第" << gen << "代 ----------\n" << std::endl;

    // ===== A: 自对弈 + 训练 =====
    // 显存管理：A自对弈/训练期间model_b_完全用不到（竞争在A/B训练之后），
    // 临时移到CPU释放约0.25GB显存，确保A阶段经验池GPU存储模式不因显存不足回退CPU
    (*model_b_)->to(torch::kCPU);
    (*model_b_)->set_eval_mode();
    std::cout << "[Gen" << gen << "-A] 自对弈 (温度衰减=" << config_.temp_decay_a
              << ")..." << std::endl;
    auto games_a = generate_self_play_data(model_a_, config_.games_per_gen,
        config_.seed_a + gen * 100, config_.temp_decay_a);
    m.a_games_generated = static_cast<int>(games_a.size());

    // 显存管理：自对弈逐局推理会留下大量PyTorch缓存块（实测可达5GB+），
    // 若训练前不清理，训练阶段可用显存仅剩0.2~0.6GB：
    //   - 新建经验池时回退CPU存储模式（显著拖慢训练）
    //   - 或AMP训练在内存压力下反复分配释放（训练耗时放大一个数量级）
    // 实测清理后训练阶段可用显存恢复到2.4GB+，训练耗时从55分钟降至几分钟
#ifdef USE_CUDA
    c10::cuda::CUDACachingAllocator::emptyCache();
#endif
    std::cout << "[Gen" << gen << "-A] 训练..." << std::endl;
    setup_training_components(model_a_, replay_a_, trainer_a_);
    replay_a_->add_games(games_a);
    auto [a_pl, a_vl, a_gl, a_tl] = train_model_generation(model_a_, replay_a_, trainer_a_);
    m.a_policy_loss = a_pl;
    m.a_value_loss = a_vl;
    m.a_gating_loss = a_gl;
    m.a_total_loss = a_tl;

    std::cout << "  A损失 | 策略: " << std::fixed << std::setprecision(4) << a_pl
              << " | 价值: " << a_vl << " | 门控: " << a_gl << std::endl;

    // 显存管理：释放A的训练组件（GPU预装经验池+Adam优化器状态），
    // 否则B训练/竞争时显存不足会回退CPU存储模式（显著拖慢训练）
    replay_a_.reset();
    trainer_a_.reset();
    // 强制释放自对弈数据：swap清空vector容量比clear()+shrink_to_fit()更彻底。
    // Windows CRT堆对大块内存free后不立即归还OS（任务管理器RSS不回落），
    // 显式swap确保容量立即还给分配器，避免长时训练中内存随代次累积。
    std::vector<GameData>().swap(games_a);
#ifdef USE_CUDA
    c10::cuda::CUDACachingAllocator::emptyCache();
#endif

    // ===== B: 自对弈 + 训练 =====
    // 显存管理：B自对弈/训练期间model_a_完全用不到（竞争在B训练之后），
    // 临时移到CPU释放约0.25GB显存；同时将A阶段移走的model_b_移回GPU供B自对弈
    (*model_a_)->to(torch::kCPU);
    (*model_a_)->set_eval_mode();
    (*model_b_)->to(device_);
    (*model_b_)->set_eval_mode();
    std::cout << "[Gen" << gen << "-B] 自对弈 (温度衰减=" << config_.temp_decay_b
              << ")..." << std::endl;
    auto games_b = generate_self_play_data(model_b_, config_.games_per_gen,
        config_.seed_b + gen * 100, config_.temp_decay_b);
    m.b_games_generated = static_cast<int>(games_b.size());

    // 显存管理：同A自对弈，训练前清理自对弈推理缓存，保证B训练阶段显存充足
#ifdef USE_CUDA
    c10::cuda::CUDACachingAllocator::emptyCache();
#endif
    std::cout << "[Gen" << gen << "-B] 训练..." << std::endl;
    setup_training_components(model_b_, replay_b_, trainer_b_);
    replay_b_->add_games(games_b);
    auto [b_pl, b_vl, b_gl, b_tl] = train_model_generation(model_b_, replay_b_, trainer_b_);
    m.b_policy_loss = b_pl;
    m.b_value_loss = b_vl;
    m.b_gating_loss = b_gl;
    m.b_total_loss = b_tl;

    std::cout << "  B损失 | 策略: " << std::fixed << std::setprecision(4) << b_pl
              << " | 价值: " << b_vl << " | 门控: " << b_gl << std::endl;

    // 显存管理：释放B的训练组件，为竞争对弈与下一代的模型操作腾出显存
    replay_b_.reset();
    trainer_b_.reset();
    // 同A侧：swap强制释放B侧自对弈数据（Windows CRT堆free后RSS不回落）
    std::vector<GameData>().swap(games_b);
#ifdef USE_CUDA
    c10::cuda::CUDACachingAllocator::emptyCache();
#endif

    // 竞争对弈需要model_a_参与，将A模型移回GPU
    (*model_a_)->to(device_);
    (*model_a_)->set_eval_mode();

    // ===== 竞争对弈（对Gen0基线，消除自对弈坍塌假胜者） =====
    // 历史教训：A vs B 自对弈竞争会产生"针对对方的共谋策略"假胜者——
    //   Gen3中B击败A(24:56)但对Gen0基线仅ELO 17.4，而A对Gen0有127；
    //   自对弈坍塌让模型学会对付自己而非变强。胜者判定必须以固定基线
    //   Gen0的胜率为准（真实棋力），而非A/B互搏胜负（相对棋力）。
    // 评估局数不变：原A vs B(80局)+胜者vs Gen0(80局)=160局，
    //   现A vs Gen0(80局)+B vs Gen0(80局)=160局，一次评估完成胜者判定+ELO。
    std::cout << "[Gen" << gen << "] A/B 分别对 Gen0 竞争 (各" << config_.competition_games
              << "局)..." << std::endl;
    // gen0默认CPU常驻（不挤占训练阶段显存），此处临时迁移GPU执行竞争评估
    (*gen0_model_)->to(device_);
    (*gen0_model_)->set_eval_mode();

    DiploidGenerationMetrics metrics_a;  // A vs Gen0
    metrics_a.generation = gen;
    run_competition(model_a_, gen0_model_, metrics_a);
    DiploidGenerationMetrics metrics_b;  // B vs Gen0
    metrics_b.generation = gen;
    run_competition(model_b_, gen0_model_, metrics_b);

    // ELO竞争结束，移回CPU释放显存，为下一代的A自对弈/训练腾出空间
    (*gen0_model_)->to(torch::kCPU);
    (*gen0_model_)->set_eval_mode();
#ifdef USE_CUDA
    c10::cuda::CUDACachingAllocator::emptyCache();
#endif

    // 汇总：A/B 对 Gen0 的胜率分别记录
    m.a_wins = metrics_a.a_wins;
    m.a_win_rate = metrics_a.a_win_rate;
    m.b_wins = metrics_b.a_wins;  // 语义：B 对 Gen0 的胜局数
    m.b_win_rate = metrics_b.a_win_rate;
    m.draws = 0;

    std::cout << "  结果: A对Gen0=" << m.a_wins << "胜"
              << " | A胜率=" << std::fixed << std::setprecision(1)
              << (m.a_win_rate * 100) << "%"
              << " | B对Gen0=" << m.b_wins << "胜"
              << " | B胜率=" << (m.b_win_rate * 100) << "%" << std::defaultfloat << std::endl;

    // 胜者判定：对Gen0胜率高者胜（超过阈值才判胜，否则无显著胜者）
    // 以真实棋力（对固定基线胜率）为进化选择依据，杜绝自对弈坍塌假胜者
    if (m.a_win_rate > config_.win_rate_threshold &&
        m.a_win_rate >= m.b_win_rate) {
        m.winner = 'A';
    } else if (m.b_win_rate > config_.win_rate_threshold &&
               m.b_win_rate > m.a_win_rate) {
        m.winner = 'B';
    } else {
        m.winner = '-';
    }

    // 显存管理：竞争对弈逐局forward会产生大量PyTorch缓存块，
    // 若不清理会随代数累积，导致下一代训练阶段显存不足回退CPU存储模式
#ifdef USE_CUDA
    c10::cuda::CUDACachingAllocator::emptyCache();
#endif

    // 参数距离
    m.parameter_distance = compute_parameter_distance(model_a_, model_b_);
    std::cout << "  参数距离: " << std::setprecision(2)
              << (m.parameter_distance * 100) << "%" << std::setprecision(4) << std::endl;

    // ===== 进化选择 =====
    if (m.winner != '-') {
        std::shared_ptr<neural::CA_MOEExpertSystem> winner =
            (m.winner == 'A') ? model_a_ : model_b_;

        // 胜者复制到双方
        copy_model_parameters(winner, model_a_);
        copy_model_parameters(winner, model_b_);

        // 添加扰动确保分化起点不同
        add_parameter_noise(model_a_, config_.perturbation_std);
        add_parameter_noise(model_b_, config_.perturbation_std);
    } else {
        // 无明显胜者，轻微扰动促进进一步分化
        add_parameter_noise(model_a_, config_.perturbation_std * 0.3f);
        add_parameter_noise(model_b_, config_.perturbation_std * 0.3f);
    }

    // ELO评估：直接用胜者（或对Gen0胜率较高者）对Gen0的胜率计算
    // （原逻辑：先A vs B竞争判定胜者，再用胜者对Gen0评估ELO——两阶段评估；
    //   现逻辑：A/B分别对Gen0评估，胜者判定与ELO基于同一组对局，避免额外误差）
    float baseline_win_rate = (m.winner == 'B') ? m.b_win_rate : m.a_win_rate;
    // 标准ELO公式（对数映射）：ElO差 = -400 * log10((1-p)/p)，p为对Gen0胜率
    // 原线性近似 K*(p-0.5) 在胜率偏离0.5时严重失真（如47.5%实际对应约-17.6而非-0.8）
    // 每代直接报告本代原始ELO（不做EMA平滑：平滑会在首代取原始值后产生"稳步下滑"假象，
    // 且掩盖真实胜率波动；ELO评价体系应以每代原始值透明反映真实棋力变化）
    float win_rate_clamped = std::clamp(baseline_win_rate, 0.01f, 0.99f);  // 防log(0)
    float current_elo = -400.0f * std::log10((1.0f - win_rate_clamped) / win_rate_clamped);

    // ===== ELO门控（用户指定：把ELO门控思想并入二倍体，且要求"大幅"上涨才接受） =====
    // 唯一接受指标是ELO：本代ELO必须超出历史最佳 >= elo_min_improvement（默认+30）才接受，
    // 否则回滚到上一代checkpoint，从机制上杜绝"越学越差"和"微涨噪声"——失败代自动回滚，best永不倒退。
    // 历史教训：早期二倍体每代无条件覆盖best_elo_，ELO从168.4一路跌到17.4；
    // 门控保证CSV的ELO列单调不降，且每次上涨都是实打实的大幅提升（过滤±35 Elo评估噪声）。
    float old_best = best_elo_;
    if (current_elo >= best_elo_ + config_.elo_min_improvement) {
        best_elo_ = current_elo;
        m.accepted = true;
        std::cout << "  ✓ ELO门控接受（ELO " << std::fixed << std::setprecision(1)
                  << current_elo << " 较历史最佳 " << old_best
                  << " 上涨≥+" << config_.elo_min_improvement << "，大幅提升）"
                  << std::defaultfloat << std::endl;
    } else {
        // 回滚：A/B参数恢复为上一代checkpoint（load方式，绕开copy_在1660 Ti上的崩溃）
        std::string prev_a = config_.checkpoint_dir + "/diploid_a_gen" + std::to_string(gen - 1) + ".pt";
        std::string prev_b = config_.checkpoint_dir + "/diploid_b_gen" + std::to_string(gen - 1) + ".pt";
        bool rolled_back = false;
        if (std::filesystem::exists(prev_a) && std::filesystem::exists(prev_b)) {
            try {
                (*model_a_)->load(prev_a);
                (*model_a_)->to(device_);
                (*model_a_)->set_eval_mode();
                (*model_b_)->load(prev_b);
                (*model_b_)->to(device_);
                (*model_b_)->set_eval_mode();
                rolled_back = true;
                std::cout << "  ✗ ELO门控回滚（ELO " << std::fixed << std::setprecision(1)
                          << current_elo << " 未达最佳+" << config_.elo_min_improvement
                          << " 的涨幅门槛，恢复第" << (gen - 1) << "代模型）"
                          << std::defaultfloat << std::endl;
            } catch (...) {
                std::cerr << "  ELO门控回滚失败（load异常），保留本代模型" << std::endl;
            }
        }
        if (!rolled_back) {
            // 无上一代checkpoint（如gen=1）：无法回滚，至少记录不倒退
            std::cout << "  ✗ ELO门控未接受（ELO " << std::fixed << std::setprecision(1)
                      << current_elo << " 未达最佳+" << config_.elo_min_improvement
                      << " 的涨幅门槛），无上一代checkpoint可回滚，保留本代"
                      << std::defaultfloat << std::endl;
            best_elo_ = std::max(best_elo_, current_elo);
        }
    }

    m.best_elo = best_elo_;
    return m;
}

// ==================== 检查点 ====================

void DiploidTrainer::save_model(
    std::shared_ptr<neural::CA_MOEExpertSystem> model, const std::string& name) {
    (*model)->set_eval_mode();
    std::string path = config_.checkpoint_dir + "/" + name + ".pt";
    try {
        // 尝试直接保存（LibTorch支持GPU tensor直接序列化）
        (*model)->save(path);
    } catch (...) {
        // 如果直接保存失败，回退到CPU中转
        auto orig_device = (*model)->parameters().begin()->device();
        (*model)->to(torch::kCPU);
        (*model)->save(path);
        (*model)->to(orig_device);
    }
    std::cout << "  模型已保存: " << path << std::endl;
}

void DiploidTrainer::save_checkpoint(
    std::shared_ptr<neural::CA_MOEExpertSystem> model_a,
    std::shared_ptr<neural::CA_MOEExpertSystem> model_b,
    int generation) {
    save_model(model_a, "diploid_a_gen" + std::to_string(generation));
    save_model(model_b, "diploid_b_gen" + std::to_string(generation));

    // 保存代指标
    std::string metrics_path = config_.output_dir + "/generation_metrics.txt";
    std::ofstream file(metrics_path, std::ios::app);
    if (file.is_open() && !history_.empty()) {
        auto& m = history_.back();
        file << m.generation << ","
             << m.a_policy_loss << "," << m.a_value_loss << "," << m.a_total_loss << ","
             << m.b_policy_loss << "," << m.b_value_loss << "," << m.b_total_loss << ","
             << m.a_wins << "," << m.b_wins << "," << m.draws << ","
             << m.a_win_rate << "," << m.parameter_distance << ","
             << m.winner << "," << m.best_elo << "," << m.elapsed_minutes << "\n";
        file.close();
    }

    // 同步保存ELO历史CSV（追加模式，供中断恢复时读取历史最佳ELO）
    // 原实现只在训练正常结束时一次性写入，导致中断后ELO历史丢失
    std::string csv_path = config_.output_dir + "/diploid_training_history.csv";
    bool csv_exists = std::filesystem::exists(csv_path);
    std::ofstream csv(csv_path, std::ios::app);
    if (csv.is_open()) {
        // 首次创建时写入表头
        if (!csv_exists) {
            csv << "generation,a_policy_loss,a_value_loss,a_gating_loss,a_total_loss,"
                << "b_policy_loss,b_value_loss,b_gating_loss,b_total_loss,"
                << "a_wins,b_wins,draws,a_win_rate,param_distance,winner,elo,minutes\n";
        }
        if (!history_.empty()) {
            auto& m = history_.back();
            csv << m.generation << ","
                << m.a_policy_loss << "," << m.a_value_loss << "," << m.a_gating_loss << "," << m.a_total_loss << ","
                << m.b_policy_loss << "," << m.b_value_loss << "," << m.b_gating_loss << "," << m.b_total_loss << ","
                << m.a_wins << "," << m.b_wins << "," << m.draws << ","
                << m.a_win_rate << "," << m.parameter_distance << ","
                << m.winner << "," << m.best_elo << "," << m.elapsed_minutes << "\n";
        }
        csv.close();
    }
}

// ==================== 输出 ====================

void DiploidTrainer::print_separator() {
    std::cout << "+------+------------+------------+------------+------------+"
              << "------+------+------+-------+------+------+" << std::endl;
}

void DiploidTrainer::print_table_header() {
    print_separator();
    std::cout << "|  Gen | A总损失    | B总损失    | A策略损失  | B策略损失  |"
              << " A胜  | B胜  | 平局 |胜率%  |距离% | ELO  |" << std::endl;
    print_separator();
}

void DiploidTrainer::print_generation_metrics(const DiploidGenerationMetrics& m) {
    if (m.generation == 0) {
        std::cout << "| " << std::setw(4) << m.generation
                  << " | " << std::setw(10) << "-"
                  << " | " << std::setw(10) << "-"
                  << " | " << std::setw(10) << "-"
                  << " | " << std::setw(10) << "-"
                  << " | " << std::setw(4) << "-"
                  << " | " << std::setw(4) << "-"
                  << " | " << std::setw(4) << "-"
                  << " | " << std::setw(5) << "-"
                  << " | " << std::setprecision(2) << std::setw(4) << (m.parameter_distance * 100)
                  << " | " << std::setw(4) << "0"
                  << " |" << std::setprecision(4) << std::endl;
    } else {
        std::cout << "| " << std::setw(4) << m.generation
                  << " | " << std::fixed << std::setprecision(3) << std::setw(10) << m.a_total_loss
                  << " | " << std::setw(10) << m.b_total_loss
                  << " | " << std::setw(10) << m.a_policy_loss
                  << " | " << std::setw(10) << m.b_policy_loss
                  << " | " << std::setw(4) << m.a_wins
                  << " | " << std::setw(4) << m.b_wins
                  << " | " << std::setw(4) << m.draws
                  << " | " << std::setprecision(1) << std::setw(4) << (m.a_win_rate * 100)
                  << " | " << std::setprecision(2) << std::setw(4) << (m.parameter_distance * 100)
                  << " | " << std::setprecision(1) << std::setw(4) << m.best_elo
                  << " |" << std::setprecision(4) << std::endl;
    }
}

// ==================== 主循环 ====================

void DiploidTrainer::run() {
    if (!initialize()) {
        std::cerr << "[错误] 二倍体训练器初始化失败" << std::endl;
        return;
    }

    start_time_ = std::chrono::steady_clock::now();

    if (resumed_gen_ < 0) {
        // 全新训练：执行Gen0预训练
        run_generation_zero();

        // 评估初始模型（竞争对弈）
        std::cout << "\n[Gen0] 初始A vs B竞争评估..." << std::endl;
        DiploidGenerationMetrics gen0_metrics;
        gen0_metrics.generation = 0;
        gen0_metrics.winner = '-';
        gen0_metrics.parameter_distance = compute_parameter_distance(model_a_, model_b_);
        run_competition(model_a_, model_b_, gen0_metrics);
        std::cout << "  Gen0竞争: A=" << gen0_metrics.a_wins << " B=" << gen0_metrics.b_wins
                  << " 平=" << gen0_metrics.draws
                  << " | A胜率=" << (gen0_metrics.a_win_rate * 100) << "%" << std::endl;

        print_table_header();
        print_generation_metrics(gen0_metrics);
        history_.push_back(gen0_metrics);
    } else {
        // 从checkpoint恢复：跳过Gen0预训练，直接进入进化循环
        std::cout << "\n[恢复模式] 跳过Gen0预训练，直接从第" << (resumed_gen_ + 1) << "代继续" << std::endl;

        // 检查gen0_model_是否有效（ELO计算需要）
        bool has_gen0 = false;
        for (const auto& p : (*gen0_model_)->parameters()) {
            if (p.norm().item<float>() > 0.01f) { has_gen0 = true; break; }
        }
        if (!has_gen0) {
            std::cout << "[警告] Gen0基线模型为空，使用当前model_a_作为基线" << std::endl;
            // 临时迁移GPU复制后移回CPU（gen0默认CPU常驻）
            (*gen0_model_)->to(device_);
            (*gen0_model_)->set_eval_mode();
            copy_model_parameters(model_a_, gen0_model_);
            (*gen0_model_)->set_eval_mode();
            (*gen0_model_)->to(torch::kCPU);
        }

        print_table_header();
    }

    // 进化循环：从恢复代数的下一代理开始（全新训练从1开始）
    int no_improve = 0;
    int start_gen = (resumed_gen_ >= 0) ? resumed_gen_ + 1 : 1;
    for (int gen = start_gen; gen <= config_.max_generations; ++gen) {
        auto m = run_generation(gen);

        auto now = std::chrono::steady_clock::now();
        m.elapsed_minutes = std::chrono::duration<float>(now - start_time_).count() / 60.0f;

        print_generation_metrics(m);
        history_.push_back(m);

        // 检查点：每代保存"当前"模型（覆盖式，满足"每代一次"），
        // 每checkpoint_interval代另存带代号checkpoint（用于中断恢复）
        save_model(model_a_, "diploid_a_current");
        save_model(model_b_, "diploid_b_current");
        if (gen % config_.checkpoint_interval == 0) {
            save_checkpoint(model_a_, model_b_, gen);
        }

        // 最佳模型：新一代ELO创历史新高时替换保存（"新的比上一代好就替换"）
        if (m.best_elo > peak_elo_) {
            peak_elo_ = m.best_elo;
            auto best_model = (m.winner == 'B') ? model_b_ : model_a_;
            save_model(best_model, "diploid_best");
            best_gen_ = gen;
            std::cout << "  [最佳] 新峰值ELO " << std::fixed << std::setprecision(1)
                      << m.best_elo << "，已更新最佳模型" << std::endl;
        }

        // 早停检查
        if (m.winner != '-') {
            no_improve = 0;
        } else {
            no_improve++;
            std::cout << "  [提示] 无显著胜者 (" << no_improve
                      << "/" << config_.no_improve_patience << ")" << std::endl;
        }

        if (no_improve >= config_.no_improve_patience) {
            std::cout << "\n  [早停] 连续 " << no_improve << " 代无显著胜者\n" << std::endl;
            break;
        }
    }

    print_separator();

    // 最终保存
    save_model(model_a_, "diploid_final_a");
    save_model(model_b_, "diploid_final_b");

    // 最终报告
    auto total_time = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - start_time_).count();

    std::cout << "\n========================================" << std::endl;
    std::cout << "  二倍体训练完成！" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  总代数: " << (history_.size() - 1) << std::endl;
    std::cout << "  总耗时: " << std::fixed << std::setprecision(1)
              << (total_time / 60.0f) << " 分钟" << std::endl;
    std::cout << "  最佳ELO变化: " << best_elo_ << std::endl;
    if (best_gen_ > 0) {
        std::cout << "  最佳代数: " << best_gen_ << std::endl;
    }

    // 验证指标
    if (history_.size() > 1) {
        auto& last = history_.back();
        std::cout << "\n  最终指标:" << std::endl;
        std::cout << "    A策略损失: " << std::fixed << std::setprecision(4)
                  << last.a_policy_loss << std::endl;
        std::cout << "    B策略损失: " << last.b_policy_loss << std::endl;
        float best_policy = std::min(last.a_policy_loss, last.b_policy_loss);
        if (best_policy < 5.0f) {
            std::cout << "    [PASS] 策略损失 < 5.0（学到非均匀策略）" << std::endl;
        }
        if (last.a_win_rate > 0.55f || last.a_win_rate < 0.45f) {
            std::cout << "    [PASS] 胜率偏离50%"
                      << " (A=" << (last.a_win_rate * 100) << "%)" << std::endl;
        }
        if (best_elo_ > 0) {
            std::cout << "    [PASS] ELO正向增长: +" << best_elo_ << std::endl;
        }
    }

    // 保存训练指标CSV
    std::string csv_path = config_.output_dir + "/diploid_training_history.csv";
    std::ofstream csv(csv_path);
    csv << "generation,a_policy_loss,a_value_loss,a_gating_loss,a_total_loss,"
        << "b_policy_loss,b_value_loss,b_gating_loss,b_total_loss,"
        << "a_wins,b_wins,draws,a_win_rate,param_distance,winner,elo,minutes\n";
    for (auto& m : history_) {
        csv << m.generation << ","
            << m.a_policy_loss << "," << m.a_value_loss << "," << m.a_gating_loss << "," << m.a_total_loss << ","
            << m.b_policy_loss << "," << m.b_value_loss << "," << m.b_gating_loss << "," << m.b_total_loss << ","
            << m.a_wins << "," << m.b_wins << "," << m.draws << ","
            << m.a_win_rate << "," << m.parameter_distance << ","
            << m.winner << "," << m.best_elo << "," << m.elapsed_minutes << "\n";
    }
    csv.close();
    std::cout << "\n  训练历史已保存: " << csv_path << std::endl;
}

} // namespace training
} // namespace go
} // namespace spacego

#endif // USE_LIBTORCH
