#include "go/training/unsupervised_self_reinforcement.h"

#ifdef USE_LIBTORCH

#include "framework/amp_grad_scaler.h"
#include "go/board/go_board_cuda.h"
#include "go/neural/ca_moe_expert.h"
#include "go/neural/opponent_style_profiler.h"
#include "go/search/hmgs_dp_optimized.h"
#include <numeric>
#include <cmath>
#include <filesystem>
#include <chrono>
#include <future>
#include <iomanip>
#include <sstream>
#include <random>

namespace spacego {
namespace go {
namespace training {

namespace {
// 全局推理互斥锁：多worker并行自对弈时共享同一模型实例，
// 用锁串行化GPU前向（GPU已饱和，串行推理不损失吞吐；CPU搜索逻辑并行摊平）
// 注：模型forward内部已有推理锁，此处额外保护autocast开关状态与初始化操作
std::mutex g_selfplay_infer_mutex;

// 锁内执行模型推理：多worker共享模型时串行化GPU前向
// 实测FP16 autocast在1660 Ti+62M小模型上是负优化（FP16转换/内核启动开销>tensor core收益），
// 故保持FP32推理；autocast开关已回滚移除
neural::CA_MOEInferenceResult locked_forward(
    std::shared_ptr<neural::CA_MOEExpertSystem> model, const torch::Tensor& state) {
    std::lock_guard<std::mutex> lock(g_selfplay_infer_mutex);
    return (*model)->forward(state);
}

std::vector<uint8_t> save_tensor_to_bytes(const torch::Tensor& tensor) {
    std::ostringstream oss;
    torch::save(std::vector<torch::Tensor>{tensor}, oss);
    std::string str = oss.str();
    return std::vector<uint8_t>(str.begin(), str.end());
}

torch::Tensor load_tensor_from_bytes(const std::vector<uint8_t>& bytes) {
    std::string str(bytes.begin(), bytes.end());
    std::istringstream iss(str);
    std::vector<torch::Tensor> tensors;
    torch::load(tensors, iss);
    return tensors.empty() ? torch::Tensor() : tensors[0];
}
}

// ==================== GameData 序列化实现 ====================

std::vector<uint8_t> GameData::serialize() const {
    std::ostringstream oss;
    
    // 写入基本信息
    oss.write(reinterpret_cast<const char*>(&game_id), sizeof(game_id));
    oss.write(reinterpret_cast<const char*>(&outcome), sizeof(outcome));
    oss.write(reinterpret_cast<const char*>(&game_length), sizeof(game_length));
    oss.write(reinterpret_cast<const char*>(&total_inference_time_ms), 
              sizeof(total_inference_time_ms));
    
    // 写入步数数量
    size_t num_steps = steps.size();
    oss.write(reinterpret_cast<const char*>(&num_steps), sizeof(num_steps));
    
    // 序列化每一步（简化版本：只保存关键数据）
    for (const auto& step : steps) {
        int action = step.action_taken;
        float value = step.value_target;
        float root_val = step.root_value;
        int move_num = step.move_number;
        float temp = step.temperature_at_move;
        
        oss.write(reinterpret_cast<const char*>(&action), sizeof(action));
        oss.write(reinterpret_cast<const char*>(&value), sizeof(value));
        oss.write(reinterpret_cast<const char*>(&root_val), sizeof(root_val));
        oss.write(reinterpret_cast<const char*>(&move_num), sizeof(move_num));
        oss.write(reinterpret_cast<const char*>(&temp), sizeof(temp));
        
        // 保存状态张量（使用torch的序列化）
        // 注意：实际实现中应该更高效地处理张量序列化
        auto state_bytes = save_tensor_to_bytes(step.state);
        size_t state_size = state_bytes.size();
        oss.write(reinterpret_cast<const char*>(&state_size), sizeof(state_size));
        oss.write(reinterpret_cast<const char*>(state_bytes.data()), state_size);
        
        // 保存策略目标
        auto policy_bytes = save_tensor_to_bytes(step.policy_target);
        size_t policy_size = policy_bytes.size();
        oss.write(reinterpret_cast<const char*>(&policy_size), sizeof(policy_size));
        oss.write(reinterpret_cast<const char*>(policy_bytes.data()), policy_size);
        
        // 保存门控权重
        auto gating_bytes = save_tensor_to_bytes(step.gating_weights);
        size_t gating_size = gating_bytes.size();
        oss.write(reinterpret_cast<const char*>(&gating_size), sizeof(gating_size));
        oss.write(reinterpret_cast<const char*>(gating_bytes.data()), gating_size);
        
        // 专家索引
        size_t num_experts = step.expert_indices.size();
        oss.write(reinterpret_cast<const char*>(&num_experts), sizeof(num_experts));
        for (int idx : step.expert_indices) {
            oss.write(reinterpret_cast<const char*>(&idx), sizeof(idx));
        }
    }
    
    std::string str = oss.str();
    return std::vector<uint8_t>(str.begin(), str.end());
}

GameData GameData::deserialize(const std::vector<uint8_t>& data) {
    GameData game;
    std::istringstream iss(std::string(data.begin(), data.end()));
    
    iss.read(reinterpret_cast<char*>(&game.game_id), sizeof(game.game_id));
    iss.read(reinterpret_cast<char*>(&game.outcome), sizeof(game.outcome));
    iss.read(reinterpret_cast<char*>(&game.game_length), sizeof(game.game_length));
    iss.read(reinterpret_cast<char*>(&game.total_inference_time_ms), 
             sizeof(game.total_inference_time_ms));
    
    size_t num_steps = 0;
    iss.read(reinterpret_cast<char*>(&num_steps), sizeof(num_steps));
    
    for (size_t i = 0; i < num_steps; ++i) {
        SelfPlayStep step;
        
        iss.read(reinterpret_cast<char*>(&step.action_taken), sizeof(step.action_taken));
        iss.read(reinterpret_cast<char*>(&step.value_target), sizeof(step.value_target));
        iss.read(reinterpret_cast<char*>(&step.root_value), sizeof(step.root_value));
        iss.read(reinterpret_cast<char*>(&step.move_number), sizeof(step.move_number));
        iss.read(reinterpret_cast<char*>(&step.temperature_at_move), 
                 sizeof(step.temperature_at_move));
        
        // 反序列化状态张量
        size_t state_size = 0;
        iss.read(reinterpret_cast<char*>(&state_size), sizeof(state_size));
        std::vector<uint8_t> state_bytes(state_size);
        iss.read(reinterpret_cast<char*>(state_bytes.data()), state_size);
        step.state = load_tensor_from_bytes(state_bytes);
        
        // 反序列化策略目标
        size_t policy_size = 0;
        iss.read(reinterpret_cast<char*>(&policy_size), sizeof(policy_size));
        std::vector<uint8_t> policy_bytes(policy_size);
        iss.read(reinterpret_cast<char*>(policy_bytes.data()), policy_size);
        step.policy_target = load_tensor_from_bytes(policy_bytes);
        
        // 反序列化门控权重
        size_t gating_size = 0;
        iss.read(reinterpret_cast<char*>(&gating_size), sizeof(gating_size));
        std::vector<uint8_t> gating_bytes(gating_size);
        iss.read(reinterpret_cast<char*>(gating_bytes.data()), gating_size);
        step.gating_weights = load_tensor_from_bytes(gating_bytes);
        
        // 专家索引
        size_t num_experts = 0;
        iss.read(reinterpret_cast<char*>(&num_experts), sizeof(num_experts));
        step.expert_indices.resize(num_experts);
        for (size_t j = 0; j < num_experts; ++j) {
            iss.read(reinterpret_cast<char*>(&step.expert_indices[j]), 
                     sizeof(step.expert_indices[j]));
        }
        
        game.steps.push_back(std::move(step));
    }
    
    return game;
}

// ==================== TrainingMetrics 实现 ====================

std::string TrainingMetrics::to_string() const {
    std::ostringstream oss;
    oss << iteration << " "
        << total_games_generated << " "
        << total_training_steps << " "
        << avg_policy_loss << " "
        << avg_value_loss << " "
        << avg_gating_loss << " "
        << avg_total_loss << " "
        << current_elo << " "
        << elo_change << " "
        << win_rate_vs_baseline << " "
        << games_per_second << " "
        << self_play_time_sec << " "
        << training_time_sec << " "
        << evaluation_time_sec;
    return oss.str();
}

TrainingMetrics TrainingMetrics::from_string(const std::string& str) {
    TrainingMetrics metrics;
    std::istringstream iss(str);
    iss >> metrics.iteration
        >> metrics.total_games_generated
        >> metrics.total_training_steps
        >> metrics.avg_policy_loss
        >> metrics.avg_value_loss
        >> metrics.avg_gating_loss
        >> metrics.avg_total_loss
        >> metrics.current_elo
        >> metrics.elo_change
        >> metrics.win_rate_vs_baseline
        >> metrics.games_per_second
        >> metrics.self_play_time_sec
        >> metrics.training_time_sec
        >> metrics.evaluation_time_sec;
    return metrics;
}

// ==================== InferenceCollector 实现 ====================

InferenceCollector::InferenceCollector(
    std::shared_ptr<neural::CA_MOEExpertSystem> model,
    int batch_size, int timeout_ms)
    : model_(std::move(model))
    , batch_size_(batch_size)
    , timeout_ms_(timeout_ms)
{
}

InferenceCollector::~InferenceCollector() {
    stop();
}

std::future<neural::CA_MOEInferenceResult> InferenceCollector::submit(torch::Tensor state) {
    // 创建推理任务，state已经是GPU上的[17,19,19]张量
    auto task = std::make_shared<InferenceTask>();
    task->state = std::move(state);
    auto future = task->result_promise.get_future();

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        task_queue_.push(std::move(task));
    }
    queue_cv_.notify_one();

    return future;
}

void InferenceCollector::start() {
    running_ = true;
    process_thread_ = std::thread(&InferenceCollector::process_loop, this);
}

void InferenceCollector::stop() {
    if (!running_.exchange(false)) {
        return;  // 已经停止
    }
    queue_cv_.notify_all();
    if (process_thread_.joinable()) {
        process_thread_.join();
    }
}

void InferenceCollector::process_loop() {
    while (running_) {
        std::vector<std::shared_ptr<InferenceTask>> batch_tasks;
        batch_tasks.reserve(batch_size_);

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);

            // 等待队列有任务或超时
            // 累积任务直到达到batch_size或等待超时
            auto timeout = std::chrono::milliseconds(timeout_ms_);
            auto deadline = std::chrono::steady_clock::now() + timeout;

            // 循环收集任务：在达到batch_size或超时前不断尝试获取
            while (running_ && static_cast<int>(batch_tasks.size()) < batch_size_) {
                // 使用wait_until等待新任务到达
                bool got_task = queue_cv_.wait_until(lock, deadline, [this] {
                    return !task_queue_.empty() || !running_;
                });

                if (!running_) {
                    break;
                }

                if (!got_task && task_queue_.empty()) {
                    // 超时且队列为空
                    break;
                }

                // 取出所有当前可用任务（不限制个数，保证公平性）
                while (!task_queue_.empty() && static_cast<int>(batch_tasks.size()) < batch_size_) {
                    batch_tasks.push_back(std::move(task_queue_.front()));
                    task_queue_.pop();
                }

                // 如果已经收集了足够任务，退出循环
                if (static_cast<int>(batch_tasks.size()) >= batch_size_) {
                    break;
                }
            }
        }

        // 处理收集到的任务
        if (!batch_tasks.empty()) {
            process_batch(batch_tasks);
        }
    }
}

void InferenceCollector::process_batch(std::vector<std::shared_ptr<InferenceTask>>& tasks) {
    int N = static_cast<int>(tasks.size());
    if (N == 0) return;

    auto batch_start = std::chrono::high_resolution_clock::now();

    try {
        // 检查所有task的state是否有效
        for (int i = 0; i < N; ++i) {
            if (!tasks[i]->state.defined()) {
                std::cerr << "[InferenceCollector] 任务" << i << " state未定义！" << std::endl;
                tasks[i]->result_promise.set_exception(
                    std::make_exception_ptr(std::runtime_error("state tensor未定义")));
                tasks.erase(tasks.begin() + i);
                --i;
                --N;
            }
        }
        if (N == 0) return;

        // 将所有任务的状态tensor stack成batch: [N, 17, 19, 19]
        std::vector<torch::Tensor> states_vec;
        states_vec.reserve(N);
        for (auto& task : tasks) {
            states_vec.push_back(task->state);
        }
        auto batch_states = torch::stack(states_vec);  // [N, 17, 19, 19]

        // 检查batch_states有效性
        if (!batch_states.defined() || batch_states.numel() == 0) {
            throw std::runtime_error("batch_states未定义或为空");
        }

        // 批量推理（禁用梯度）
        torch::NoGradGuard no_grad;
        auto batch_result = (*model_)->forward(batch_states);

        // 检查推理结果有效性
        if (!batch_result.final_policy.defined()) {
            throw std::runtime_error("forward返回final_policy未定义");
        }
        if (!batch_result.final_value.defined()) {
            throw std::runtime_error("forward返回final_value未定义");
        }
        if (!batch_result.gating_weights.defined()) {
            throw std::runtime_error("forward返回gating_weights未定义");
        }

        // 将批量结果分发给各个任务
        // batch_result.final_policy: [N, 362]
        // batch_result.final_value:  [N, 1]
        // batch_result.gating_weights: [N, num_experts]
        // 注意：必须使用clone()创建独立tensor，不能使用索引视图(view)
        // 因为视图依赖于batch_result的底层存储，batch_result销毁后视图会失效
        for (int i = 0; i < N; ++i) {
            neural::CA_MOEInferenceResult result;
            result.final_policy = batch_result.final_policy[i].clone();   // [362]
            result.final_value = batch_result.final_value[i].clone();     // [1]
            result.gating_weights = batch_result.gating_weights[i].clone(); // [num_experts]
            tasks[i]->result_promise.set_value(std::move(result));
        }
    } catch (const std::exception& e) {
        // 推理异常时，将所有任务设置为异常状态
        for (auto& task : tasks) {
            try {
                task->result_promise.set_exception(
                    std::make_exception_ptr(
                        std::runtime_error("批量推理失败: " + std::string(e.what()))));
            } catch (...) {
                // promise可能已经被设置，忽略
            }
        }
    }

    auto batch_end = std::chrono::high_resolution_clock::now();
    double latency_ms = std::chrono::duration<double, std::milli>(batch_end - batch_start).count();

    // 更新统计信息
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        total_tasks_ += N;
        total_batches_ += 1;
        total_batch_sum_ += static_cast<double>(N);
        total_latency_ms_ += latency_ms;
    }
}

InferenceCollector::CollectorStats InferenceCollector::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    CollectorStats s;
    s.total_tasks_processed = total_tasks_;
    s.total_batches = total_batches_;
    s.avg_batch_size = total_batches_ > 0 ? total_batch_sum_ / total_batches_ : 0.0;
    s.avg_latency_ms = total_batches_ > 0 ? total_latency_ms_ / total_batches_ : 0.0;
    return s;
}

// ==================== SelfPlayGenerator 实现 ====================

SelfPlayGenerator::SelfPlayGenerator(int worker_id, 
                                     const SelfPlayConfig& config,
                                     std::shared_ptr<neural::CA_MOEExpertSystem> model,
                                     std::shared_ptr<InferenceCollector> collector)
    : worker_id(worker_id)
    , config(config)
    , model(model)
    , inference_collector_(std::move(collector))
    , running(true)
    , current_temperature(config.temperature)
    , rng(config.seed + worker_id)
    , device(config.use_cuda && torch::cuda::is_available() ? 
             torch::Device(torch::kCUDA, config.cuda_device_id) : 
             torch::Device(torch::kCPU))
{
    use_fp16 = config.use_mixed_precision && device.is_cuda();
    // 初始化多元评价器（每个Worker独立实例，避免锁竞争）
    multi_dim_evaluator = std::make_unique<MultiDimEvaluator>();
    // 初始化GAE计算器（使用训练配置中的GAE参数在initialize中设置）
    gae_calculator = std::make_unique<GAECalculator>();
}

SelfPlayGenerator::~SelfPlayGenerator() {
    stop();
}

bool SelfPlayGenerator::initialize() {
    try {
        if (!model) {
            std::cerr << "[SelfPlayGenerator-" << worker_id << "] 错误: 模型为空" << std::endl;
            return false;
        }
        
        // 设置模型为评估模式（推理时不需要梯度）
        // 多worker共享模型：状态切换与设备迁移需串行化，避免并发竞争
        {
            std::lock_guard<std::mutex> lock(g_selfplay_infer_mutex);
            (*model)->set_eval_mode();
            if (device.is_cuda()) {
                (*model)->to(device);
            }
        }
        
        if (device.is_cuda()) {
            // 创建CUDA流以支持异步执行（RAII包装器自动管理生命周期）
#ifdef USE_CUDA
            cuda_stream = std::make_unique<neural::CudaStreamRAII>(cudaStreamNonBlocking);
#endif

            std::cout << "[SelfPlayGenerator-" << worker_id << "] 初始化成功" << std::endl;
            std::cout << "  设备: CUDA:" << config.cuda_device_id << std::endl;
            std::cout << "  混合精度: " << (use_fp16 ? "启用" : "禁用") << std::endl;
            std::cout << "  每步模拟: " << config.simulations_per_move << std::endl;
        } else {
            std::cout << "[SelfPlayGenerator-" << worker_id << "] 初始化成功 (CPU模式)" << std::endl;
        }

        // 初始化HMGS-DP搜索器（神经网络策略/价值评估注入）
        initialize_hmgs_search();
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[SelfPlayGenerator-" << worker_id << "] 初始化失败: " 
                  << e.what() << std::endl;
        return false;
    }
}

// 初始化HMGS-DP搜索器：注入神经网络策略先验与价值评估回调
// 使自对弈每步通过搜索（访问次数分布）生成高质量训练目标
void SelfPlayGenerator::initialize_hmgs_search() {
    try {
        hmgs_searcher_ = std::make_unique<search::HMGSDP_Optimized>();
        hmgs_searcher_->initialize("");

        // 配置并行搜索（单线程搜索，批量推理batch=128减少GPU内核启动次数）
        search::HMGSDP_Optimized::ParallelSearchConfig ps_config;
        ps_config.num_workers = 1;
        ps_config.batch_size = 128;
        ps_config.use_batch_inference = false;
        hmgs_searcher_->set_parallel_config(ps_config);

        // 策略先验回调：对给定棋盘状态做一次模型推理，返回362维策略分布
        // 合并推理：AlphaZero网络一次前向同时输出策略与价值，
        // 将value一并缓存，供simulate的value_cb直接命中，避免第二次推理
        // 接收棋盘对象引用（搜索树节点直传，避免字符串序列化开销）
        auto policy_cb = [this](const GoBoardCPU& board, int current_player) {
            try {
                // 缓存key：棋盘状态字符串（每节点仅一次序列化，用于命中value缓存）
                std::string board_state = board.to_string();
                // 缓存命中：同一搜索中expand_node与simulate对同一节点连续调用
                auto cache_it = nn_eval_cache_.find(board_state);
                if (cache_it != nn_eval_cache_.end()) {
                    return cache_it->second.first;
                }

                std::vector<float> raw(17 * 19 * 19, 0.0f);
                board.encode_to_tensor(raw.data(), 17);
                auto state = torch::from_blob(raw.data(), {1, 17, 19, 19}, torch::kFloat32).clone();
                if (device.is_cuda()) {
                    state = state.to(device);
                }
                torch::NoGradGuard no_grad;
                auto result = locked_forward(model, state);
                auto policy = result.final_policy.squeeze(0).to(torch::kCPU);
                auto policy_data = policy.contiguous().data_ptr<float>();
                std::vector<float> out(362, 0.0f);
                float sum = 0.0f;
                for (int i = 0; i < 362; ++i) {
                    out[static_cast<size_t>(i)] = std::max(policy_data[i], 0.0f);
                    sum += out[static_cast<size_t>(i)];
                }
                // 归一化为概率分布（避免全零导致除零）
                if (sum > 1e-8f) {
                    for (auto& p : out) p /= sum;
                }

                // 同步缓存价值（黑棋视角 → 当前玩家视角），供value_cb直接命中
                float value = result.final_value.to(torch::kCPU).item<float>();
                if (current_player == -1) {
                    value = -value;
                }
                nn_eval_cache_[board_state] = {out, std::clamp(value, -1.0f, 1.0f)};

                return out;
            } catch (...) {
                // 推理失败时返回均匀分布兜底
                return std::vector<float>(362, 1.0f / 362.0f);
            }
        };

        // 价值回调：对给定棋盘状态做一次模型推理，返回当前玩家视角的价值[-1,1]
        // 优先命中policy_cb推理时的缓存（单次前向同时输出策略与价值）
        auto value_cb = [this](const GoBoardCPU& board, int current_player) {
            try {
                // 缓存key与policy_cb一致（同一节点状态字符串）
                std::string board_state = board.to_string();
                // 缓存命中：expand_node已在本步搜索中推理过该状态
                auto cache_it = nn_eval_cache_.find(board_state);
                if (cache_it != nn_eval_cache_.end()) {
                    return cache_it->second.second;
                }

                std::vector<float> raw(17 * 19 * 19, 0.0f);
                board.encode_to_tensor(raw.data(), 17);
                auto state = torch::from_blob(raw.data(), {1, 17, 19, 19}, torch::kFloat32).clone();
                if (device.is_cuda()) {
                    state = state.to(device);
                }
                torch::NoGradGuard no_grad;
                auto result = locked_forward(model, state);
                float value = result.final_value.to(torch::kCPU).item<float>();
                // final_value为黑棋视角（训练约定），转换为当前玩家视角
                // current_player: 1=黑, -1=白（GoBoardCPU的StoneColor约定）
                if (current_player == -1) {
                    value = -value;
                }
                value = std::clamp(value, -1.0f, 1.0f);
                // 单独推理的价值也缓存（策略分布用均匀分布占位，仅作完整性）
                nn_eval_cache_[board_state] = {std::vector<float>(362, 1.0f / 362.0f), value};
                return value;
            } catch (...) {
                return 0.0f;
            }
        };

        // 批量评估回调：一次前向评估一批叶子节点（策略+价值），大幅提升GPU利用率
        // 搜索主循环按batch_size收集叶子节点后统一调用（自对弈数据生成的关键提速）
        auto batch_cb = [this](const std::vector<GoBoardCPU>& boards,
                               const std::vector<int>& players) {
            std::vector<std::pair<std::vector<float>, float>> results;
            results.reserve(boards.size());
            if (boards.empty()) return results;
            try {
                // 构建批量输入 [N, 17, 19, 19]
                std::vector<float> raw(boards.size() * 17 * 19 * 19, 0.0f);
                for (size_t i = 0; i < boards.size(); ++i) {
                    boards[i].encode_to_tensor(raw.data() + i * 17 * 19 * 19, 17);
                }
                auto state = torch::from_blob(
                    raw.data(), {static_cast<long>(boards.size()), 17, 19, 19},
                    torch::kFloat32).clone();
                if (device.is_cuda()) {
                    state = state.to(device);
                }
                torch::NoGradGuard no_grad;
                auto result = locked_forward(model, state);
                // 批量输出：策略 [N,362]，价值 [N]
                auto policy_batch = result.final_policy.to(torch::kCPU).contiguous();
                auto value_batch = result.final_value.to(torch::kCPU).contiguous();
                const float* policy_ptr = policy_batch.data_ptr<float>();
                const float* value_ptr = value_batch.data_ptr<float>();
                for (size_t i = 0; i < boards.size(); ++i) {
                    std::vector<float> out(362, 0.0f);
                    float sum = 0.0f;
                    for (int j = 0; j < 362; ++j) {
                        out[static_cast<size_t>(j)] = std::max(policy_ptr[i * 362 + j], 0.0f);
                        sum += out[static_cast<size_t>(j)];
                    }
                    // 归一化为概率分布（避免全零导致除零）
                    if (sum > 1e-8f) {
                        for (auto& v : out) v /= sum;
                    }
                    float val = value_ptr[i];
                    // final_value为黑棋视角（训练约定），转换为当前玩家视角
                    // players: 1=黑, -1=白（GoBoardCPU的StoneColor约定）
                    if (players[i] == -1) {
                        val = -val;
                    }
                    results.emplace_back(std::move(out), std::clamp(val, -1.0f, 1.0f));
                }
            } catch (...) {
                // 推理失败：返回均匀策略+零价值兜底（与单节点回调行为一致）
                for (size_t i = 0; i < boards.size(); ++i) {
                    results.emplace_back(std::vector<float>(362, 1.0f / 362.0f), 0.0f);
                }
            }
            return results;
        };

        hmgs_searcher_->set_batch_evaluator(batch_cb);
        hmgs_searcher_->set_neural_evaluator(policy_cb, value_cb);
        std::cout << "[SelfPlayGenerator-" << worker_id << "] HMGS-DP搜索器初始化完成 (模拟数="
                  << config.simulations_per_move << ")" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[SelfPlayGenerator-" << worker_id << "] HMGS-DP搜索器初始化失败: "
                  << e.what() << std::endl;
        hmgs_searcher_.reset();
    }
}

// 使用HMGS-DP搜索生成策略分布（访问次数归一化）与根价值
// board_state: 棋盘状态字符串（361字符）
// current_player: 当前行棋方（1=黑, -1=白）
// 返回: (策略分布[362], 根价值)
std::pair<std::vector<float>, float> SelfPlayGenerator::search_policy(
    const std::string& board_state, int current_player) {
    std::vector<float> uniform(362, 1.0f / 362.0f);
    if (!hmgs_searcher_) {
        return {uniform, 0.0f};
    }

    try {
        // 清空NN评估缓存：缓存仅在本步搜索内有效（跨步状态不同，防止无界增长）
        nn_eval_cache_.clear();

        // 执行HMGS-DP搜索：模拟次数由config.simulations_per_move控制
        // 探索权重1.414（标准PUCT探索常数）
        hmgs_searcher_->search(board_state, config.simulations_per_move, 1.414);

        // 从搜索树根节点子节点访问次数生成策略分布（AlphaZero式训练目标）
        auto policy = hmgs_searcher_->get_root_policy(19);
        float root_value = static_cast<float>(hmgs_searcher_->get_root_value());

        // 检查策略是否全零（搜索未产生有效访问），兜底为均匀分布
        float sum = 0.0f;
        for (auto& p : policy) sum += p;
        if (sum < 1e-6f) {
            return {uniform, root_value};
        }
        return {policy, root_value};
    } catch (const std::exception& e) {
        std::cerr << "[SelfPlayGenerator-" << worker_id << "] 搜索失败: "
                  << e.what() << "，使用均匀分布" << std::endl;
        return {uniform, 0.0f};
    }
}

GameData SelfPlayGenerator::generate_game(int game_id) {
    GameData game;
    game.game_id = game_id;
    game.start_time = std::chrono::system_clock::now();
    double total_inference_time = 0.0;

    // 在当前线程中初始化CUDA context（如果使用GPU）
    // generate_game可能在非主线程中调用（training_thread），
    // 必须在调用forward前确保当前线程有正确的CUDA context
    static thread_local bool cuda_ctx_initialized = false;
    if (!cuda_ctx_initialized && device.is_cuda()) {
#ifdef USE_CUDA
        cudaSetDevice(config.cuda_device_id);
        torch::cuda::synchronize();
#endif
        cuda_ctx_initialized = true;
    }

    try {
        // 初始化棋盘状态（19x19围棋棋盘）
        // 使用位棋盘表示优化内存访问
        GoBoardCPU board;
        
        int move_num = 0;
        bool game_ended = false;
        int consecutive_passes = 0;  // 连续pass计数（双方各pass一次则游戏结束）
        // 每局重置动态对战统计（跨步状态不跨局累积）
        win_hold_steps_ = 0;
        last_win_rate_ = 0.5f;
        policy_exec_hits_ = 0;
        policy_exec_total_ = 0;
        
        while (!game_ended && move_num < config.max_game_length) {
            auto step_start = std::chrono::high_resolution_clock::now();
            
            // 1. 提取当前状态特征（17个平面 x 19 x 19）
            // 使用encode_to_tensor将棋盘编码为张量（复用预分配buffer避免每步分配）
            torch::Tensor current_state = torch::empty({17, 19, 19});
            board.encode_to_tensor(current_state.data_ptr<float>(), 17);
            
            // 2. 将状态移到GPU并准备批量推理
            if (device.is_cuda()) {
                current_state = current_state.to(device);
                // 注意：不将输入转换为FP16，因为模型权重是FP32
                // 混合精度训练时由AMPGradScaler处理梯度缩放，推理时保持FP32
            }
            
            // 3. 使用HMGS-DP搜索生成策略（训练目标）与动作
            // AlphaZero范式：搜索访问次数分布作为policy_target，替代模型自身输出
            // （模型自身输出作目标会导致"自我蒸馏"死循环，策略损失恒≈log(362)无法下降）
            auto inference_start = std::chrono::high_resolution_clock::now();

            torch::NoGradGuard no_grad;

            // 执行HMGS-DP搜索：获取访问次数归一化策略分布与根价值
            std::string board_str = board.to_string();
            int player_idx = (board.get_current_player() == StoneColor::BLACK) ? 1 : -1;
            auto [search_policy_vec, search_root_value] = search_policy(board_str, player_idx);

            // 将搜索策略分布转为张量 [362]
            torch::Tensor search_policy_tensor = torch::from_blob(
                const_cast<float*>(search_policy_vec.data()),
                {1, 362}, torch::kFloat32).clone();
            torch::Tensor mcts_policy = search_policy_tensor.squeeze(0);
            float root_value = search_root_value;

            auto inference_end = std::chrono::high_resolution_clock::now();
            auto inference_duration = std::chrono::duration<float, std::milli>(
                inference_end - inference_start).count();
            total_inference_time += inference_duration;

            // 4. 保存纯净策略（搜索访问次数分布，即训练目标）
            torch::Tensor pure_policy = mcts_policy.clone();

            // 5. 添加Dirichlet噪声（仅用于动作选择的探索，不影响训练目标）
            torch::Tensor selection_policy = mcts_policy;
            if (move_num < 30) {
                selection_policy = add_dirichlet_noise(mcts_policy);
            }

            // 6. 选择动作（基于搜索后的策略分布）
            std::vector<bool> legal_mask = build_legal_mask(board);
            int selected_action;
            // 终局强制收束：对局超过300步（远超19路正常约250步终局）时直接pass，
            // 保证连续两次pass自然终局，杜绝400步截断导致的胜负标签错误。
            // 250~300步区间先以高权重引导pass（温度较高时概率采样可能仍不选pass）
            if (move_num > 300) {
                selected_action = 361;  // 强制pass，触发连续pass终局
            } else if (move_num > 250) {
                // 引导阶段：按空点数提升pass权重（空点越少越接近终局）
                int empty_count = 0;
                for (int i = 0; i < 361; ++i) {
                    auto [x, y] = GoBoardCPU::index_to_xy(i);
                    if (board.is_empty(x, y)) empty_count++;
                }
                float pass_min = (empty_count <= 10) ? 0.9f : ((empty_count <= 40) ? 0.7f : 0.5f);
                selection_policy = selection_policy.clone();
                selection_policy[361] = std::max(selection_policy[361].item<float>(), pass_min);
                selected_action = select_action(selection_policy, current_temperature, legal_mask);
            } else {
                selected_action = select_action(selection_policy, current_temperature, legal_mask);
            }

            // 7.5 多元评价：落子前评估对方棋盘状态
            MultiDimEvaluation eval_before;
            if (config.enable_multi_dim_eval) {
                StoneColor move_player = board.get_current_player();
                eval_before = multi_dim_evaluator->evaluate(board, move_player);
            }

            // 8. 在棋盘上执行动作
            StoneColor player = board.get_current_player();
            
            if (selected_action == 361) {
                consecutive_passes++;
                if (consecutive_passes >= 2) {
                    game_ended = true;
                }
                board.set_current_player(opponent_color(player));
                board.set_move_count(board.get_move_count() + 1);
            } else {
                consecutive_passes = 0;
                auto [action_x, action_y] = GoBoardCPU::index_to_xy(selected_action);
                MoveResult move_result = board.make_move(action_x, action_y, player);
                
                if (!move_result.success) {
                    std::cerr << "[警告] Worker-" << worker_id 
                              << " 游戏" << game_id << " 步骤" << move_num 
                              << ": 无效动作 " << selected_action 
                              << "（legal_mask过滤后仍失败）" << std::endl;
                    break;
                }
            }

            // 8.5 多元评价：落子后评估己方棋盘状态，计算即时奖励
            MultiDimEvaluation eval_after;
            if (config.enable_multi_dim_eval) {
                eval_after = multi_dim_evaluator->evaluate(board, player);
            }

            // 9. 记录数据
            SelfPlayStep step;
            step.state = current_state.cpu();
            step.policy_target = normalize_policy(pure_policy).cpu();
            step.root_value = root_value;
            // 门控权重与专家索引：对当前局面做一次推理获取（用于训练门控网络）
            // 搜索内部已有推理，但此处单独获取gating以保持数据完整性
            {
                torch::Tensor gating_weights;
                try {
                    auto batch_state = current_state.unsqueeze(0);
                    auto result = locked_forward(model, batch_state);
                    gating_weights = result.gating_weights.to(torch::kCPU).squeeze(0);
                } catch (...) {
                    // 推理失败时使用均匀门控兜底
                    int num_experts = (*model)->get_num_experts();
                    gating_weights = torch::ones({num_experts}) / static_cast<float>(num_experts);
                }
                step.gating_weights = gating_weights;
                step.expert_indices = get_top_k_experts(gating_weights, 3);
            }
            step.action_taken = selected_action;
            step.temperature_at_move = current_temperature;
            step.move_number = move_num;
            step.value_target = 0.0f;
            // 多元评价数据
            step.multi_dim_eval = eval_after;
            // 即时奖励 = 多元评价评分奖励 + 基础差值 + 动态对战信号
            // 1) 评分奖励（用户指定的经典AI招式）：多元评价得分越高奖励越多。
            //    以当前玩家视角的综合评分与胜率估计为"表现分"，领先越多即时奖励越大，
            //    让模型每一步都追求"把局面下得更好"（而不仅是终局胜负）。
            //    综合评分0.5=均势：得分越高（>0.5）说明当前玩家越有利；
            //    胜率估计0.5=均势：>0.55视为领先。两者合成"表现分"再缩放。
            float composite_after = eval_after.composite_score;  // [0,1]，0.5=均势
            float win_rate_after = eval_after.metrics[(int)EvalDimension::WIN_RATE_ESTIMATE].score;
            // 表现分 = (综合评分偏离均势) + (胜率偏离均势)，范围约[-0.5,0.5]→×0.4缩放
            float perf_score = (composite_after - 0.5f) * 0.4f + (win_rate_after - 0.5f) * 0.4f;
            float score_reward = config.enable_multi_dim_eval ?
                std::max(-0.2f, std::min(0.2f, perf_score)) : 0.0f;  // 裁剪防爆炸
            // 2) 基础奖励：综合评分差 + 每步胜率变化（MultiDimEvaluator内合成）
            float base_reward = config.enable_multi_dim_eval ?
                multi_dim_evaluator->compute_step_reward(eval_before, eval_after, player) : 0.0f;
            // 3) 胜率保持时间奖励：连续保持胜率估计>0.55的步数越多奖励越高，
            //    鼓励模型"稳守优势"而非大起大落；每步累加、封顶防爆炸
            if (win_rate_after > 0.55f) {
                win_hold_steps_++;
            } else {
                win_hold_steps_ = 0;
            }
            float hold_reward = std::min(win_hold_steps_, 20) * 0.005f;  // 封顶 20*0.005=0.1
            // 4) 策略执行率奖励：实际落子是否等于搜索访问分布的top-1（模型"说到做到"）
            int top1_action = static_cast<int>(pure_policy.argmax().item<int>());
            bool exec_hit = (selected_action == top1_action);
            policy_exec_total_++;
            if (exec_hit) policy_exec_hits_++;
            float exec_reward = exec_hit ? 0.02f : 0.0f;  // 温和正向激励

            step.step_reward = score_reward + base_reward + hold_reward + exec_reward;
            step.has_gae_signal = false;
            step.gae_advantage = 0.0f;
            step.td_return = 0.0f;
            game.steps.push_back(std::move(step));

            // 生成时预存8种D4对称版本（数据增强）
            if (config.prestore_augmentation) {
                auto original_step = game.steps.back();
                auto symmetries = generate_d4_symmetries(
                    original_step.state, original_step.policy_target);

                for (int sym = 1; sym < 8; ++sym) {
                    SelfPlayStep aug_step;
                    aug_step.state = symmetries[sym].first;
                    aug_step.policy_target = symmetries[sym].second;
                    aug_step.value_target = 0.0f;
                    aug_step.root_value = original_step.root_value;
                    aug_step.gating_weights = original_step.gating_weights.clone();
                    aug_step.expert_indices = original_step.expert_indices;
                    aug_step.action_taken = original_step.action_taken;
                    aug_step.temperature_at_move = original_step.temperature_at_move;
                    aug_step.move_number = original_step.move_number;
                    // 复制多元评价和GAE字段（同一局面不同视角，评价值同源）
                    aug_step.multi_dim_eval = original_step.multi_dim_eval;
                    aug_step.step_reward = original_step.step_reward;
                    aug_step.gae_advantage = 0.0f;
                    aug_step.td_return = 0.0f;
                    aug_step.has_gae_signal = false;
                    game.steps.push_back(std::move(aug_step));
                }
            }
            // 9. 检查游戏是否结束（双方无合法落子时结束）
            if (!game_ended) {
                auto current_legal = board.get_legal_moves(board.get_current_player());
                if (current_legal.empty()) {
                    // 当前玩家无合法落子，检查对方
                    StoneColor opponent = opponent_color(board.get_current_player());
                    auto opponent_legal = board.get_legal_moves(opponent);
                    if (opponent_legal.empty()) {
                        // 双方都无合法落子，游戏结束
                        game_ended = true;
                    }
                }
            }
            
            move_num++;
            
            // 10. 温度衰减已移至每局结束后（非每步衰减）
            // 每步衰减会导致一局内温度下降过快，min_temperature截断后衰减失效
            // 现在温度在一局内保持恒定，每局结束后统一衰减
        }
        
        // 计算游戏结果：中国规则面积计点（棋子 + 领地 + 黑方贴7.5目）
        // 黑棋视角：+1表示黑胜，-1表示白胜，0表示平局
        auto [black_final, white_final] = board.compute_area_score(7.5f);
        if (black_final > white_final) {
            game.outcome = 1.0f;   // 黑胜
        } else if (white_final > black_final) {
            game.outcome = -1.0f;  // 白胜
        } else {
            game.outcome = 0.0f;  // 平局
        }
        
        // 计算GAE：将即时奖励和价值估计转化为每步优势信号
        if (config.enable_gae && config.enable_multi_dim_eval) {
            // 收集原始步骤的奖励和价值（跳过增强步骤）
            int stride = config.prestore_augmentation ? 8 : 1;
            std::vector<float> step_rewards;
            std::vector<float> step_values;
            std::vector<size_t> original_indices;  // 原始步骤在game.steps中的索引
            
            for (size_t i = 0; i < game.steps.size(); i += stride) {
                step_rewards.push_back(game.steps[i].step_reward);
                step_values.push_back(game.steps[i].root_value);
                original_indices.push_back(i);
            }
            
            if (!step_rewards.empty()) {
                // 终局真实胜负注入GAE（当前玩家视角）：修复价值目标自洽循环缺陷。
                // 原实现compute_gae只传启发式奖励和搜索root_value，game.outcome从不进入
                // 价值目标 → 价值头学的是"搜索值的自洽复制"而非真实胜负（价值损失虚低0.001、
                // ELO停滞）。现在最后一步的TD bootstrap = 真实胜负，经折扣向后传播。
                // game.outcome是黑棋视角(+1黑胜/-1白胜)，最后一步为黑棋(move_number偶数)则
                // 黑视角直接取，白棋则取反（与传统模式 value_target 的视角规则一致）。
                float terminal_value = 0.0f;
                size_t last_idx = original_indices.back();
                int last_move_number = game.steps[last_idx].move_number;
                terminal_value = (last_move_number % 2 == 0) ? game.outcome : -game.outcome;
                auto gae_result = gae_calculator->compute_gae(
                    step_rewards, step_values, terminal_value);
                
                // 填充GAE优势到所有步骤（原始+增强）
                for (size_t gi = 0; gi < original_indices.size(); ++gi) {
                    size_t base_idx = original_indices[gi];
                    // 将GAE信号复制到该原始步骤及其所有增强版本
                    for (int aug = 0; aug < stride && (base_idx + aug) < game.steps.size(); ++aug) {
                        auto& s = game.steps[base_idx + aug];
                        s.gae_advantage = gae_result.advantages[gi];
                        s.td_return = gae_result.returns[gi];
                        s.value_target = gae_result.returns[gi];  // TD目标作为价值目标
                        s.has_gae_signal = true;
                    }
                }
            }
        } else {
            // 传统模式：使用最终胜负作为价值目标
            for (auto& step : game.steps) {
                step.value_target = (step.move_number % 2 == 0) ? game.outcome : -game.outcome;
            }
        }
        
        game.game_length = move_num;
        game.end_time = std::chrono::system_clock::now();
        game.total_inference_time_ms = static_cast<float>(total_inference_time);
        
        // 保存SGF棋谱（按间隔保存，避免磁盘占用过大）
        if (config.save_sgf_games && (game_id % config.sgf_save_interval == 0)) {
            save_sgf(game);
        }
        
        // 更新统计信息
        {
            std::lock_guard<std::mutex> lock(stats_mutex);
            stats.games_completed++;
            stats.total_moves += move_num;
            stats.total_inference_time_ms += total_inference_time;
            stats.avg_inference_time_ms = stats.total_inference_time_ms / 
                                          std::max(1, stats.total_moves);
        }

        // 每100局打印一次动态对战指标（不影响训练，仅监控）
        if (game_id % 100 == 0) {
            float exec_rate = policy_exec_total_ > 0 ?
                (100.0f * policy_exec_hits_ / policy_exec_total_) : 0.0f;
            std::cout << "[动态指标] 局" << game_id
                      << " | 策略执行率: " << std::fixed << std::setprecision(1) << exec_rate << "%"
                      << " | 最长胜率保持: " << win_hold_steps_ << "步" << std::endl;
            std::cout << std::defaultfloat;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[SelfPlayGenerator-" << worker_id << "] 游戏生成异常: " 
                  << e.what() << std::endl;
        game.outcome = 0.0f;  // 标记为无效游戏
        game.game_length = 0;
    }
    
    return game;
}

// 对当前棋盘做多元评价（便捷封装）
MultiDimEvaluation SelfPlayGenerator::evaluate_board(const GoBoardCPU& board, StoneColor perspective) {
    if (!multi_dim_evaluator) return MultiDimEvaluation();
    return multi_dim_evaluator->evaluate(board, perspective);
}

// 游戏结束后计算GAE并填充到所有步骤（便捷封装，实际在generate_game中内联调用）
void SelfPlayGenerator::compute_gae_for_game(GameData& game) {
    if (!gae_calculator || !config.enable_gae) return;
    // GAE计算已在generate_game中直接执行，此方法为外部调用入口保留
}

std::vector<GameData> SelfPlayGenerator::generate_games(int start_id, int count) {
    std::vector<GameData> games;
    games.reserve(count);
    
    for (int i = 0; i < count && running; ++i) {
        auto g_start = std::chrono::steady_clock::now();
        GameData game = generate_game(start_id + i);
        auto g_end = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(g_end - g_start).count();
        if (game.game_length > 0) {  // 只添加有效游戏
            games.push_back(std::move(game));
        }
        // 局间进度打印：便于监控自对弈数据生成的实时速度与有效性
        std::cout << "[SelfPlayGenerator-" << worker_id << "] 第" << (i + 1) << "/" << count
                  << "局 | 步数=" << game.game_length
                  << " | 耗时=" << std::fixed << std::setprecision(1) << secs << "s"
                  << std::endl;
        // 每局结束后衰减温度（非每步衰减）
        // 温度在一局内保持恒定，保证策略一致性
        apply_temperature_decay();
    }
    
    return games;
}

std::vector<bool> SelfPlayGenerator::build_legal_mask(const GoBoardCPU& board) const {
    // 构建大小为362的合法着法mask
    // 索引0-360：棋盘位置（19x19=361），索引361：pass
    std::vector<bool> mask(362, false);
    
    // 获取当前玩家的所有合法落子点
    StoneColor player = board.get_current_player();
    auto legal_moves = board.get_legal_moves(player);
    
    // 标记合法落子点
    for (const auto& [x, y] : legal_moves) {
        int idx = GoBoardCPU::xy_to_index(x, y);
        if (idx >= 0 && idx < 361) {
            mask[idx] = true;
        }
    }
    
    // pass（索引361）始终合法
    mask[361] = true;
    
    return mask;
}

int SelfPlayGenerator::select_action(const torch::Tensor& policy, float temperature,
                                      const std::vector<bool>& legal_mask) {
    // 根据温度参数采样动作，使用legal_mask过滤非法落子点
    // 高温度 -> 更多探索（接近均匀分布）
    // 低温度 -> 更贪婪（选择最高概率的动作）
    // legal_mask: 大小362的布尔数组，true表示该位置合法
    
    auto policy_cpu = policy.to(torch::kCPU);
    auto policy_data = policy_cpu.contiguous().data_ptr<float>();
    int num_actions = policy.numel();
    
    // 确保legal_mask大小匹配
    int mask_size = static_cast<int>(legal_mask.size());
    if (mask_size < num_actions) {
        // mask不足时，做防御性处理：选择概率最高的动作
        int best_action = 0;
        float best_prob = policy_data[0];
        for (int i = 1; i < num_actions; ++i) {
            if (policy_data[i] > best_prob) {
                best_prob = policy_data[i];
                best_action = i;
            }
        }
        return best_action;
    }
    
    if (temperature <= 0.01f) {
        // 贪婪选择：在合法动作中选择概率最高的
        int best_action = 361;  // 默认pass
        float best_prob = -1.0f;
        for (int i = 0; i < num_actions; ++i) {
            if (legal_mask[i] && policy_data[i] > best_prob) {
                best_prob = policy_data[i];
                best_action = i;
            }
        }
        return best_action;
    } else {
        // 基于温度的概率采样，仅对合法动作采样
        // 应用温度：p_i ^ (1/T)，非法动作概率设为0
        std::vector<float> adjusted_probs(num_actions, 0.0f);
        float sum = 0.0f;
        
        for (int i = 0; i < num_actions; ++i) {
            if (legal_mask[i]) {
                // 仅对合法动作应用温度缩放
                adjusted_probs[i] = std::pow(std::max(policy_data[i], 0.001f), 1.0f / temperature);
                sum += adjusted_probs[i];
            }
            // 非法动作保持0.0f
        }
        
        // 如果所有合法动作概率之和为0（极端情况），选择pass
        if (sum <= 0.0f) {
            return 361;  // pass
        }
        
        // 归一化
        for (int i = 0; i < num_actions; ++i) {
            adjusted_probs[i] /= sum;
        }
        
        // 采样
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng);
        float cumulative = 0.0f;
        
        for (int i = 0; i < num_actions; ++i) {
            cumulative += adjusted_probs[i];
            if (r <= cumulative) {
                return i;
            }
        }
        
        // 兜底：返回pass
        return 361;
    }
}

void SelfPlayGenerator::save_sgf(const GameData& game) const {
    // 将游戏数据导出为SGF格式文件，可被Sabaki等围棋软件打开
    // SGF格式参考：https://www.red-bean.com/sgf/
    
    try {
        // 确保输出目录存在
        std::filesystem::create_directories(config.sgf_output_dir);
        
        // 构建文件名
        std::string filename = config.sgf_output_dir + "/game_" + 
                               std::to_string(game.game_id) + ".sgf";
        std::ofstream ofs(filename);
        if (!ofs.is_open()) {
            std::cerr << "[SelfPlayGenerator-" << worker_id 
                      << "] 无法创建SGF文件: " << filename << std::endl;
            return;
        }
        
        // SGF头部（FF[4]格式，19x19棋盘，中国规则贴目7.5）
        ofs << "(;GM[1]FF[4]CA[UTF-8]";
        ofs << "AP[SpaceGo:SelfPlay]";
        ofs << "SZ[19]KM[7.5]";
        ofs << "PB[SpaceGo-Black]PW[SpaceGo-White]";
        
        // 结果
        if (game.outcome > 0.5f) {
            ofs << "RE[B+R]";  // 黑胜（R表示不计目）
        } else if (game.outcome < -0.5f) {
            ofs << "RE[W+R]";  // 白胜
        } else {
            ofs << "RE[0]";    // 平局
        }
        
        // 对局长度
        ofs << "GC[GameLength=" << game.game_length << "]";
        ofs << "\n";
        
        // 写入每一步着法
        for (const auto& step : game.steps) {
            // SGF中黑棋为B，白棋为W
            // move_number为偶数时是黑棋（0,2,4...），奇数时是白棋
            char color = (step.move_number % 2 == 0) ? 'B' : 'W';
            
            if (step.action_taken == 361) {
                // Pass：SGF中用空坐标表示
                ofs << ";" << color << "[]";
            } else {
                // 正常落子：SGF坐标使用小写字母，a-s对应0-18
                auto [x, y] = GoBoardCPU::index_to_xy(step.action_taken);
                if (x >= 0 && x < 19 && y >= 0 && y < 19) {
                    char cx = 'a' + x;
                    char cy = 'a' + y;
                    ofs << ";" << color << "[" << cx << cy << "]";
                }
            }
        }
        
        ofs << ")\n";
        ofs.close();
    } catch (const std::exception& e) {
        std::cerr << "[SelfPlayGenerator-" << worker_id 
                  << "] SGF保存异常: " << e.what() << std::endl;
    }
}

torch::Tensor SelfPlayGenerator::add_dirichlet_noise(const torch::Tensor& policy) {
    int num_actions = policy.numel();

    // 生成Dirichlet分布的噪声
    std::gamma_distribution<float> gamma(config.dirichlet_alpha, 1.0f);
    std::vector<float> noise(num_actions);
    float noise_sum = 0.0f;

    for (int i = 0; i < num_actions; ++i) {
        noise[i] = gamma(rng);
        noise_sum += noise[i];
    }

    // 归一化噪声
    for (auto& n : noise) {
        n /= noise_sum;
    }

    // 注意：policy可能在GPU上，需要先移到CPU再访问数据指针
    // 直接访问GPU张量的data_ptr会导致ACCESS_VIOLATION
    // to(kCPU)+contiguous已产生新连续张量，直接用作noisy_policy，无需clone
    auto policy_cpu = policy.to(torch::kCPU).contiguous();
    auto noisy_policy = policy_cpu;  // 直接复用，消除冗余clone
    auto noisy_data = noisy_policy.data_ptr<float>();
    auto orig_data = policy_cpu.data_ptr<float>();

    for (int i = 0; i < num_actions; ++i) {
        noisy_data[i] = (1.0f - config.dirichlet_weight) * orig_data[i] +
                         config.dirichlet_weight * noise[i];
    }

    // 将结果移回原始设备
    return noisy_policy.to(policy.device());
}

torch::Tensor SelfPlayGenerator::normalize_policy(const torch::Tensor& mcts_visits) {
    // 归一化MCTS访问次数得到概率分布
    // 注意：输入可能在GPU上，需要先移到CPU再访问数据指针
    // to(kCPU)+contiguous已产生新连续张量，无需clone
    auto normalized = mcts_visits.to(torch::kCPU).contiguous();
    auto data = normalized.data_ptr<float>();
    int num_elements = normalized.numel();

    float sum = 0.0f;
    for (int i = 0; i < num_elements; ++i) {
        sum += data[i];
    }

    if (sum > 0.0f) {
        for (int i = 0; i < num_elements; ++i) {
            data[i] /= sum;
        }
    }

    return normalized;
}

void SelfPlayGenerator::apply_temperature_decay() {
    if (current_temperature > config.min_temperature) {
        current_temperature *= config.temperature_decay;
        current_temperature = std::max(current_temperature, config.min_temperature);
    }
}

std::vector<int> SelfPlayGenerator::get_top_k_experts(const torch::Tensor& gating_weights, 
                                                       int top_k) {
    auto weights_cpu = gating_weights.to(torch::kCPU);
    auto weights_data = weights_cpu.contiguous().data_ptr<float>();
    int num_experts = weights_cpu.numel();
    
    // 创建专家索引和权重对
    struct ExpertWeight {
        int index;
        float weight;
    };
    
    std::vector<ExpertWeight> experts(num_experts);
    for (int i = 0; i < num_experts; ++i) {
        experts[i] = {i, weights_data[i]};
    }
    
    // 按权重降序排序
    std::sort(experts.begin(), experts.end(), 
              [](const ExpertWeight& a, const ExpertWeight& b) {
                  return a.weight > b.weight;
              });
    
    // 返回top-k专家索引
    std::vector<int> result(top_k);
    for (int i = 0; i < top_k && i < static_cast<int>(experts.size()); ++i) {
        result[i] = experts[i].index;
    }
    
    return result;
}

std::vector<std::pair<torch::Tensor, torch::Tensor>> SelfPlayGenerator::generate_d4_symmetries(
    const torch::Tensor& state, const torch::Tensor& policy) {
    // state: [17, 19, 19] (CPU), policy: [362] (CPU)
    // 生成8种D4对称变换版本（棋盘对称群：4次旋转×2次翻转）
    // 返回8个(state, policy)对，sym=0为原始版本，sym=1-7为变换版本
    std::vector<std::pair<torch::Tensor, torch::Tensor>> result;
    result.reserve(8);
    int board_size = 19;
    int action_board_size = board_size * board_size;  // 361，棋盘部分，不含pass

    // 提取策略的棋盘部分 [362] -> [19, 19]，保留pass不变
    auto policy_board = policy.slice(0, 0, action_board_size).reshape({board_size, board_size});
    auto policy_pass = policy.slice(0, action_board_size, 362);  // [1]

    for (int sym = 0; sym < 8; ++sym) {
        torch::Tensor transformed_state;
        torch::Tensor transformed_policy_board;

        switch (sym) {
            case 0:  // 不变
                transformed_state = state.clone();
                transformed_policy_board = policy_board.clone();
                break;
            case 1:  // 水平翻转
                transformed_state = torch::flip(state, {2});
                transformed_policy_board = torch::flip(policy_board, {1});
                break;
            case 2:  // 垂直翻转
                transformed_state = torch::flip(state, {1});
                transformed_policy_board = torch::flip(policy_board, {0});
                break;
            case 3:  // 180度旋转
                transformed_state = torch::flip(state, {1, 2});
                transformed_policy_board = torch::flip(policy_board, {0, 1});
                break;
            case 4:  // 转置（主对角线翻转）
                transformed_state = state.transpose(1, 2);
                transformed_policy_board = policy_board.transpose(0, 1);
                break;
            case 5:  // 90度顺时针旋转
                transformed_state = torch::flip(state.transpose(1, 2), {1});
                transformed_policy_board = torch::flip(policy_board.transpose(0, 1), {0});
                break;
            case 6:  // 270度顺时针旋转
                transformed_state = torch::flip(state.transpose(1, 2), {2});
                transformed_policy_board = torch::flip(policy_board.transpose(0, 1), {1});
                break;
            case 7:  // 反对角线翻转
                transformed_state = torch::flip(state.transpose(1, 2), {1, 2});
                transformed_policy_board = torch::flip(policy_board.transpose(0, 1), {0, 1});
                break;
        }

        // 重组策略: [19, 19] -> [361] + pass -> [362]
        auto transformed_policy = torch::cat({
            transformed_policy_board.reshape({action_board_size}),
            policy_pass
        }, 0);

        result.emplace_back(transformed_state, transformed_policy);
    }

    return result;
}

SelfPlayGenerator::GeneratorStats SelfPlayGenerator::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex);
    return stats;
}

void SelfPlayGenerator::reset_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex);
    stats = GeneratorStats{};
}

// ==================== EfficientReplayBuffer 实现 ====================

class EfficientReplayBuffer::SumTreeImpl {
public:
    SumTreeImpl(int capacity) : capacity(capacity), size(0), write_pos(0) {
        tree.resize(2 * capacity);
        data.resize(capacity);
    }
    
    void update(int idx, float priority) {
        tree[idx + capacity] = priority;
        propagate(idx + capacity);
    }
    
    void add(float priority, BufferEntry entry) {
        data[write_pos] = std::move(entry);
        update(write_pos, priority);
        write_pos = (write_pos + 1) % capacity;
        size = std::min(size + 1, capacity);
    }
    
    std::pair<int, float> get(float s) {
        // 从根节点tree[1]开始检索，而不是tree[0]
        // tree[0]未使用，根节点在tree[1]
        int idx = retrieve(1, s);
        return {idx, tree[idx + capacity]};
    }
    
    float get_total_priority() const {
        // 根节点在tree[1]，不是tree[0]
        // propagate函数从parent>=1开始更新，tree[0]从未被赋值
        return tree[1];
    }
    
    int get_size() const { return size; }
    BufferEntry* get_data(int idx) { return &data[idx]; }

private:
    int capacity;
    int size;
    int write_pos;
    std::vector<float> tree;
    std::vector<BufferEntry> data;
    
    void propagate(int idx) {
        int parent = idx / 2;
        while (parent >= 1) {
            tree[parent] = tree[2 * parent] + tree[2 * parent + 1];
            parent /= 2;
        }
    }
    
    int retrieve(int idx, float s) {
        int left = 2 * idx;
        int right = left + 1;
        
        if (left >= static_cast<int>(tree.size())) {
            return idx - capacity;
        }
        
        if (s <= tree[left]) {
            return retrieve(left, s);
        } else {
            return retrieve(right, s - tree[left]);
        }
    }
};

EfficientReplayBuffer::EfficientReplayBuffer(int capacity, int num_experts)
    : capacity_(capacity)
    , num_experts_(num_experts)
    , buffer(capacity)
    , sum_tree(std::make_unique<SumTreeImpl>(capacity))
    , rng(std::random_device{}())
{
    // 默认初始化为CPU模式，由init_gpu_storage()决定是否升级为GPU模式
}

EfficientReplayBuffer::~EfficientReplayBuffer() = default;

// ==================== GPU预装存储方法 ====================

bool EfficientReplayBuffer::init_gpu_storage_matrices(torch::Device device, int capacity, int num_experts) {
    if (!device.is_cuda()) {
        std::cerr << "[EfficientReplayBuffer] GPU存储初始化失败：设备不是CUDA" << std::endl;
        return false;
    }
    
    // 检查可用显存
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    try {
        // 使用CUDA Runtime API查询显存信息（兼容所有libtorch版本）
        cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (err != cudaSuccess) {
            std::cerr << "[EfficientReplayBuffer] cudaMemGetInfo失败: "
                      << cudaGetErrorString(err) << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "[EfficientReplayBuffer] 无法查询显存信息: " << e.what() << std::endl;
        return false;
    }
    
    // 估算所需显存（每条目约26KB）
    // states: 17*19*19*4 = 24,548 字节
    // policies: 362*4 = 1,448 字节
    // values: 4 字节
    // gating_weights: num_experts*4 字节
    constexpr size_t kStatesPerEntryBytes = 17 * 19 * 19 * sizeof(float);
    constexpr size_t kPoliciesPerEntryBytes = 362 * sizeof(float);
    constexpr size_t kValuesPerEntryBytes = sizeof(float);
    size_t gating_per_entry_bytes = num_experts * sizeof(float);
    size_t per_entry_bytes = kStatesPerEntryBytes + kPoliciesPerEntryBytes
                           + kValuesPerEntryBytes + gating_per_entry_bytes;
    size_t total_needed = capacity * per_entry_bytes;
    
    // 预留显存余量给模型和临时张量
    // 实测：模型(~0.25GB)+Adam优化器状态(~0.5GB)+batch临时张量(~0.1GB) ≈ 0.85GB。
    // 原2GB预留过高，导致B训练阶段（A模型仍驻留GPU时）可用显存不足而回退CPU存储模式。
    // 此处压到1GB，配合训练循环中"B训练期间A模型暂移CPU"策略，保证GPU存储模式稳定启用。
    constexpr size_t kReservedBytes = 1ULL * 1024 * 1024 * 1024;
    
    double needed_gb = total_needed / (1024.0 * 1024.0 * 1024.0);
    double free_gb = free_bytes / (1024.0 * 1024.0 * 1024.0);
    double total_gb = total_bytes / (1024.0 * 1024.0 * 1024.0);
    
    std::cout << "[EfficientReplayBuffer] 显存状态：总量=" << total_gb << "GB, 可用="
              << free_gb << "GB, 需要=" << needed_gb << "GB (容量=" << capacity << ")" << std::endl;
    
    if (free_bytes < total_needed + kReservedBytes) {
        std::cout << "[EfficientReplayBuffer] 显存不足，回退到CPU存储模式（"
                  << "可用=" << free_gb << "GB, 需要≈" << (total_needed + kReservedBytes) / (1024.0 * 1024.0 * 1024.0)
                  << "GB）" << std::endl;
        return false;
    }
    
    // 分配GPU大矩阵
    try {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        
        // 使用empty跳过零初始化（后续copy_会覆盖写入，无需zero填充）
        gpu_states_ = torch::empty({capacity, 17, 19, 19}, options);
        gpu_policies_ = torch::empty({capacity, 362}, options);
        gpu_values_ = torch::empty({capacity}, options);
        gpu_gating_weights_ = torch::empty({capacity, num_experts}, options);
        gpu_advantages_ = torch::empty({capacity}, options);
        gpu_td_returns_ = torch::empty({capacity}, options);
        
        // 预分配CPU辅助容器
        expert_indices_list_.resize(capacity);
        priority_list_.resize(capacity, 0.0f);
        
        // 同步CUDA确保分配完成
        if (device.is_cuda()) {
            torch::cuda::synchronize();
        }
    } catch (const std::exception& e) {
        std::cerr << "[EfficientReplayBuffer] GPU矩阵分配失败: " << e.what() << std::endl;
        return false;
    }
    
    std::cout << "[EfficientReplayBuffer] GPU预装存储初始化成功！显存占用≈" << needed_gb << "GB" << std::endl;
    return true;
}

void EfficientReplayBuffer::fallback_to_cpu_storage() {
    use_gpu_storage_ = false;
    storage_device_ = torch::kCPU;
    // 释放GPU资源
    gpu_states_ = torch::Tensor();
    gpu_policies_ = torch::Tensor();
    gpu_values_ = torch::Tensor();
    gpu_gating_weights_ = torch::Tensor();
    expert_indices_list_.clear();
    priority_list_.clear();
    // 确保CPU buffer正常
    if (buffer.size() != static_cast<size_t>(capacity_)) {
        buffer.resize(capacity_);
    }
    next_index = 0;
    current_size = 0;
}

// ==================== 动态扩容方法 ====================

bool EfficientReplayBuffer::expand_capacity(int new_capacity) {
    std::lock_guard<std::mutex> lock(buffer_mutex);

    // 新容量必须大于当前容量
    if (new_capacity <= capacity_) {
        return false;
    }

    std::cout << "[EfficientReplayBuffer] 扩容: " << capacity_
              << " → " << new_capacity
              << " (当前数据量: " << current_size << ")" << std::endl;

    if (use_gpu_storage_) {
        // GPU模式：分配新的GPU大矩阵，拷贝已有数据，释放旧矩阵
        try {
            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(storage_device_);

            // 分配新矩阵
            auto new_states = torch::empty({new_capacity, 17, 19, 19}, options);
            auto new_policies = torch::empty({new_capacity, 362}, options);
            auto new_values = torch::empty({new_capacity}, options);
            auto new_gating = torch::empty({new_capacity, num_experts_}, options);

            // 拷贝已有数据到新矩阵
            if (current_size > 0) {
                auto indices = torch::arange(current_size,
                    torch::TensorOptions().dtype(torch::kLong).device(storage_device_));
                new_states.slice(0, 0, current_size).copy_(gpu_states_.index_select(0, indices));
                new_policies.slice(0, 0, current_size).copy_(gpu_policies_.index_select(0, indices));
                new_values.slice(0, 0, current_size).copy_(gpu_values_.index_select(0, indices));
                new_gating.slice(0, 0, current_size).copy_(gpu_gating_weights_.index_select(0, indices));
            }

            // 替换旧矩阵（旧矩阵自动释放）
            gpu_states_ = std::move(new_states);
            gpu_policies_ = std::move(new_policies);
            gpu_values_ = std::move(new_values);
            gpu_gating_weights_ = std::move(new_gating);

            // 扩展CPU辅助容器
            expert_indices_list_.resize(new_capacity);
            priority_list_.resize(new_capacity, 0.0f);

            // 同步CUDA确保数据完整
            if (storage_device_.is_cuda()) {
                torch::cuda::synchronize();
            }
        } catch (const std::exception& e) {
            std::cerr << "[EfficientReplayBuffer] GPU扩容失败，回退到CPU: "
                      << e.what() << std::endl;
            // GPU扩容失败时回退到CPU模式
            fallback_to_cpu_storage();
            // 以CPU模式继续扩容
            buffer.resize(new_capacity);
        }
    } else {
        // CPU模式：直接resize缓冲区
        buffer.resize(new_capacity);
    }

    // 重建SumTree（保留已有数据的优先级信息）
    capacity_ = new_capacity;
    sum_tree = std::make_unique<SumTreeImpl>(new_capacity);

    // 重新插入已有数据的优先级
    for (int i = 0; i < current_size; ++i) {
        float p;
        if (use_gpu_storage_) {
            p = priority_list_[i];
        } else {
            p = buffer[i].priority;
        }
        sum_tree->update(i, p > 0.0f ? p : max_priority);
    }

    // next_index需要调整：如果旧容量下next_index指向的位置在新容量范围内则保留
    // 否则重置到current_size位置（环形缓冲区逻辑）
    if (next_index >= current_size) {
        // 环形缓冲区尚未填满一圈，next_index有效
        // 保持next_index不变
    } else {
        // 已填满，next_index指向最旧数据，扩容后保持
    }

    std::cout << "[EfficientReplayBuffer] 扩容完成，新容量: " << capacity_ << std::endl;
    return true;
}

bool EfficientReplayBuffer::init_gpu_storage(torch::Device device, int capacity, int num_experts) {
    std::lock_guard<std::mutex> lock(buffer_mutex);
    
    if (!device.is_cuda() || !torch::cuda::is_available()) {
        std::cout << "[EfficientReplayBuffer] CUDA不可用，保持CPU存储模式" << std::endl;
        return false;
    }
    
    // 尝试分配GPU矩阵
    if (init_gpu_storage_matrices(device, capacity, num_experts)) {
        use_gpu_storage_ = true;
        storage_device_ = device;
        // 清空旧的CPU buffer以释放内存
        buffer.clear();
        buffer.shrink_to_fit();
        return true;
    }
    
    return false;
}

void EfficientReplayBuffer::add_game(const GameData& game) {
    // 锁内：仅预留索引，避免阻塞采样线程
    int start_idx;
    {
        std::lock_guard<std::mutex> lock(buffer_mutex);
        start_idx = next_index;
        int n = static_cast<int>(game.steps.size());
        next_index = (next_index + n) % capacity_;
        current_size = std::min(current_size + n, capacity_);
    }
    
    // 锁外：执行数据拷贝（GPU或CPU），不阻塞采样
    for (size_t i = 0; i < game.steps.size(); ++i) {
        int idx = (start_idx + static_cast<int>(i)) % capacity_;
        const auto& step = game.steps[i];
        
        if (use_gpu_storage_) {
            // GPU预装模式：直接拷贝到GPU大矩阵
            gpu_states_[idx].copy_(step.state.to(storage_device_, /*non_blocking=*/true));
            gpu_policies_[idx].copy_(step.policy_target.to(storage_device_, /*non_blocking=*/true));
            gpu_values_[idx].fill_(step.value_target);
            gpu_gating_weights_[idx].copy_(step.gating_weights.to(storage_device_, /*non_blocking=*/true));
            gpu_advantages_[idx].fill_(step.gae_advantage);
            gpu_td_returns_[idx].fill_(step.td_return);
            expert_indices_list_[idx] = step.expert_indices;
            priority_list_[idx] = max_priority;
            // sum_tree更新需要锁（与sample_batch共享）
            {
                std::lock_guard<std::mutex> lock(buffer_mutex);
                sum_tree->update(idx, max_priority);
            }
        } else {
            // CPU模式
            BufferEntry entry;
            entry.state = step.state;
            entry.target_policy = step.policy_target;
            entry.target_value = step.value_target;
            entry.gating_weights = step.gating_weights;
            entry.expert_indices = step.expert_indices;
            entry.priority = max_priority;
            entry.advantage = step.gae_advantage;
            entry.td_return = step.td_return;
            
            {
                std::lock_guard<std::mutex> lock(buffer_mutex);
                buffer[idx] = std::move(entry);
                sum_tree->update(idx, max_priority);
            }
        }
    }
}

void EfficientReplayBuffer::add_games(const std::vector<GameData>& games) {
    // 锁内：统计总步数，预留索引
    int start_idx;
    int total_steps = 0;
    {
        std::lock_guard<std::mutex> lock(buffer_mutex);
        start_idx = next_index;
        for (const auto& game : games) {
            total_steps += static_cast<int>(game.steps.size());
        }
        next_index = (next_index + total_steps) % capacity_;
        current_size = std::min(current_size + total_steps, capacity_);
    }
    
    // 锁外：逐条拷贝数据
    int step_offset = 0;
    for (const auto& game : games) {
        for (size_t i = 0; i < game.steps.size(); ++i) {
            int idx = (start_idx + step_offset) % capacity_;
            const auto& step = game.steps[i];
            
            if (use_gpu_storage_) {
                gpu_states_[idx].copy_(step.state.to(storage_device_, true));
                gpu_policies_[idx].copy_(step.policy_target.to(storage_device_, true));
                gpu_values_[idx].fill_(step.value_target);
                gpu_gating_weights_[idx].copy_(step.gating_weights.to(storage_device_, true));
                gpu_advantages_[idx].fill_(step.gae_advantage);
                gpu_td_returns_[idx].fill_(step.td_return);
                expert_indices_list_[idx] = step.expert_indices;
                priority_list_[idx] = max_priority;
                { std::lock_guard<std::mutex> lock(buffer_mutex);
                  sum_tree->update(idx, max_priority); }
            } else {
                BufferEntry entry;
                entry.state = step.state; entry.target_policy = step.policy_target;
                entry.target_value = step.value_target; entry.gating_weights = step.gating_weights;
                entry.expert_indices = step.expert_indices; entry.priority = max_priority;
                entry.advantage = step.gae_advantage;
                entry.td_return = step.td_return;
                { std::lock_guard<std::mutex> lock(buffer_mutex);
                  buffer[idx] = std::move(entry);
                  sum_tree->update(idx, max_priority); }
            }
            step_offset++;
        }
    }
}

EfficientReplayBuffer::TrainingBatch EfficientReplayBuffer::sample_batch(
    int batch_size, const std::string& strategy) {
    
    TrainingBatch batch;
    batch.expert_indices.reserve(batch_size);
    batch.importance_weights.reserve(batch_size);
    batch.buffer_indices.reserve(batch_size);
    
    std::lock_guard<std::mutex> lock(buffer_mutex);
    
    // 用于收集采样的索引（GPU和CPU模式共用采样逻辑）
    std::vector<int64_t> sampled_indices;
    sampled_indices.reserve(batch_size);
    std::vector<float> sampled_priorities;
    sampled_priorities.reserve(batch_size);
    
    if (strategy == "prioritized" && use_prioritized_replay && current_size > 0) {
        // 优先级经验回放采样
        float total_priority = sum_tree->get_total_priority();
        beta = std::min(1.0f, beta + beta_increment);
        
        // 防止total_priority为0导致分布参数无效
        if (total_priority <= 0.0f) {
            total_priority = static_cast<float>(current_size);
        }
        
        std::uniform_real_distribution<float> dist(0.0f, total_priority);
        
        for (int i = 0; i < batch_size; ++i) {
            float s = dist(rng);
            auto [idx, priority] = sum_tree->get(s);
            
            // 边界检查：确保idx在有效范围内
            if (idx < 0 || idx >= current_size) {
                // 回退到均匀随机采样
                std::uniform_int_distribution<int> fallback_dist(0, current_size - 1);
                idx = fallback_dist(rng);
                priority = max_priority;
            }
            
            sampled_indices.push_back(idx);
            sampled_priorities.push_back(priority);
            batch.buffer_indices.push_back(idx);
            
            // 计算重要性采样权重
            float sampling_prob = priority / total_priority;
            float importance_weight = std::pow(current_size * sampling_prob + epsilon,
                                                -beta);
            batch.importance_weights.push_back(importance_weight);
        }
    } else {
        // 均匀随机采样
        std::uniform_int_distribution<int> dist(0, current_size - 1);
        
        for (int i = 0; i < batch_size; ++i) {
            int idx = dist(rng);
            sampled_indices.push_back(idx);
            sampled_priorities.push_back(1.0f);
            batch.buffer_indices.push_back(idx);
            batch.importance_weights.push_back(1.0f);
        }
    }
    
    if (use_gpu_storage_ && !sampled_indices.empty()) {
        // GPU预装模式：直接在GPU上通过index_select获取批量数据
        // 将索引转为GPU上的LongTensor，然后直接index_select获取batch
        auto indices_tensor = torch::tensor(sampled_indices, 
                                            torch::TensorOptions().dtype(torch::kLong).device(storage_device_));
        
        batch.states = gpu_states_.index_select(0, indices_tensor);
        batch.target_policies = gpu_policies_.index_select(0, indices_tensor);
        batch.target_values = gpu_values_.index_select(0, indices_tensor);
        batch.gating_weights = gpu_gating_weights_.index_select(0, indices_tensor);
        batch.advantages = gpu_advantages_.index_select(0, indices_tensor);
        batch.td_returns = gpu_td_returns_.index_select(0, indices_tensor);
        
        // 收集expert_indices（从CPU辅助容器中读取）
        for (int idx : sampled_indices) {
            const auto& expert_ids = expert_indices_list_[idx];
            batch.expert_indices.insert(batch.expert_indices.end(),
                                        expert_ids.begin(), expert_ids.end());
        }
    } else if (!sampled_indices.empty()) {
        // CPU模式：保持原有逻辑，逐条从buffer取出并用torch::stack合并
        std::vector<torch::Tensor> states_vec;
        std::vector<torch::Tensor> policies_vec;
        std::vector<torch::Tensor> values_vec;
        std::vector<torch::Tensor> gating_vec;
        std::vector<torch::Tensor> advantages_vec;
        std::vector<torch::Tensor> td_returns_vec;
        states_vec.reserve(batch_size);
        policies_vec.reserve(batch_size);
        values_vec.reserve(batch_size);
        gating_vec.reserve(batch_size);
        advantages_vec.reserve(batch_size);
        td_returns_vec.reserve(batch_size);
        
        for (int idx : sampled_indices) {
            const auto& entry = buffer[idx];
            states_vec.push_back(entry.state);
            policies_vec.push_back(entry.target_policy);
            values_vec.push_back(torch::tensor(entry.target_value));
            gating_vec.push_back(entry.gating_weights);
            advantages_vec.push_back(torch::tensor(entry.advantage));
            td_returns_vec.push_back(torch::tensor(entry.td_return));
            batch.expert_indices.insert(batch.expert_indices.end(),
                                        entry.expert_indices.begin(),
                                        entry.expert_indices.end());
        }
        
        batch.states = torch::stack(states_vec);
        batch.target_policies = torch::stack(policies_vec);
        batch.target_values = torch::stack(values_vec);
        batch.gating_weights = torch::stack(gating_vec);
        batch.advantages = torch::stack(advantages_vec);
        batch.td_returns = torch::stack(td_returns_vec);
    }
    
    return batch;
}

void EfficientReplayBuffer::update_priorities(const std::vector<int>& indices,
                                               const std::vector<float>& priorities) {
    std::lock_guard<std::mutex> lock(buffer_mutex);
    
    for (size_t i = 0; i < indices.size() && i < priorities.size(); ++i) {
        float priority = std::abs(priorities[i]) + epsilon;
        max_priority = std::max(max_priority, priority);
        sum_tree->update(indices[i], priority);
    }
}

int EfficientReplayBuffer::size() const {
    std::lock_guard<std::mutex> lock(buffer_mutex);
    return current_size;
}

void EfficientReplayBuffer::clear() {
    std::lock_guard<std::mutex> lock(buffer_mutex);
    
    if (use_gpu_storage_) {
        // GPU模式：将GPU大矩阵清零
        gpu_states_.zero_();
        gpu_policies_.zero_();
        gpu_values_.zero_();
        gpu_gating_weights_.zero_();
        // 清空CPU辅助容器
        for (auto& vec : expert_indices_list_) {
            vec.clear();
        }
        std::fill(priority_list_.begin(), priority_list_.end(), 0.0f);
    } else {
        // CPU模式：保持原有逻辑
        buffer.clear();
        buffer.resize(capacity_);
    }
    
    sum_tree.reset(new SumTreeImpl(capacity_));
    next_index = 0;
    current_size = 0;
}

void EfficientReplayBuffer::save(const std::string& path) const {
    std::ofstream ofs(path, std::ios::binary);
    
    // 简化的持久化：只保存基本元数据
    // 实际应用中可能需要更高效的格式
    int32_t version = 1;
    ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
    ofs.write(reinterpret_cast<const char*>(&capacity_), sizeof(capacity_));
    ofs.write(reinterpret_cast<const char*>(&current_size), sizeof(current_size));
    
    ofs.close();
}

void EfficientReplayBuffer::load(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    
    int32_t version = 0;
    ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
    
    if (version == 1) {
        ifs.read(reinterpret_cast<char*>(&capacity_), sizeof(capacity_));
        ifs.read(reinterpret_cast<char*>(&current_size), sizeof(current_size));
    }
    
    ifs.close();
}

// ==================== MixedPrecisionTrainer 实现 ====================

MixedPrecisionTrainer::MixedPrecisionTrainer(
    std::shared_ptr<neural::CA_MOEExpertSystem> model,
    const TrainingConfig& config)
    : model(model)
    , config(config)
    , use_amp(config.use_mixed_precision && torch::cuda::is_available())
    , device(torch::cuda::is_available() ? torch::Device(torch::kCUDA) : 
                                            torch::Device(torch::kCPU))
    , current_lr(config.learning_rate)
{
    if (use_amp) {
        scaler = std::make_unique<framework::AMPGradScaler>();
    }
}

MixedPrecisionTrainer::~MixedPrecisionTrainer() = default;

void MixedPrecisionTrainer::setup_optimizer() {
    // 收集模型参数
    std::vector<torch::Tensor> params;
    for (const auto& p : (*model)->parameters()) {
        if (p.requires_grad()) {
            params.push_back(p);
        }
    }
    
    // 根据配置创建优化器
    if (config.optimizer_type == "adam") {
        optimizer = std::make_unique<torch::optim::Adam>(
            params,
            torch::optim::AdamOptions(config.learning_rate)
                .weight_decay(config.weight_decay));
    } else if (config.optimizer_type == "adamw") {
        optimizer = std::make_unique<torch::optim::AdamW>(
            params,
            torch::optim::AdamWOptions(config.learning_rate)
                .weight_decay(config.weight_decay));
    } else {
        optimizer = std::make_unique<torch::optim::SGD>(
            params,
            torch::optim::SGDOptions(config.learning_rate)
                .momentum(0.9)
                .weight_decay(config.weight_decay));
    }
    
    // 创建学习率调度器（余弦退火）
    // T_max按epoch数设计（每epoch结束后才step一次lr_scheduler）
    // num_epochs_per_iteration * 10 给出合理的退火周期
    lr_scheduler = std::make_unique<framework::CosineAnnealingLRScheduler>(
        *optimizer, config.num_epochs_per_iteration * 10);
    
    std::cout << "[MixedPrecisionTrainer] 优化器初始化完成" << std::endl;
    std::cout << "  类型: " << config.optimizer_type << std::endl;
    std::cout << "  学习率: " << std::setprecision(6) << config.learning_rate << std::endl;
    std::cout << "  混合精度训练: " << (use_amp ? "启用" : "禁用") << std::endl;
}

std::pair<torch::Tensor, MixedPrecisionTrainer::LossResult> MixedPrecisionTrainer::compute_losses(
    const torch::Tensor& pred_policy,
    const torch::Tensor& pred_value,
    const torch::Tensor& pred_gating,
    const torch::Tensor& target_policy,
    const torch::Tensor& target_value,
    const std::vector<float>& importance_weights,
    const torch::Tensor& advantages,
    const torch::Tensor& pred_gating_logits) {
    
    LossResult result;
    
    // Label Smoothing处理：将目标分布与均匀分布混合，防止模型对标签过度自信
    auto policy_targets = target_policy;
    if (config.label_smoothing > 0.0f) {
        float eps = config.label_smoothing;
        int num_classes = target_policy.size(1);
        policy_targets = (1.0f - eps) * target_policy + eps / num_classes;
    }
    
    // 策略损失（交叉熵）
    auto policy_clamped = pred_policy.clamp(1e-8, 1.0);
    auto log_policy = policy_clamped.log();
    // 防发散（ELO门控在线训练实测）：模型把搜索最优动作压到1e-8时log=-18.4，
    //   标准交叉熵单样本贡献18×target权重，配合高lr一步冲爆梯度（策略损失飙6-8）。
    //   clip限制单样本贡献，收敛模型正常log_probs>-12不受影响。
    // 死区教训（2026-09-13实测）：下界原为-6（对应p=2.48e-3），而19路棋盘362个动作
    //   的均匀概率是1/362=2.76e-3（log=-5.895）——下界距均匀分布仅2%。模型只要对
    //   目标动作的概率略低于均匀即落入死区（clamp梯度恒0），策略头彻底学不动：
    //   实测裁剪前梯度总范数=1.5e-11、Δ参数L2≈0、策略损失恒5.92（均匀水平）且同一
    //   棋谱连训8遍逐遍完全相同。放宽到-12（p≥6.1e-6）让"偏弱于均匀"仍保有梯度，
    //   单样本最大贡献12×0.9=10.8，配合max_gradient_norm=3.0不会发散。
    log_policy = log_policy.clamp(-12.0, 0.0);
    auto policy_loss = -(policy_targets * log_policy).sum(1).mean();
    
    // 价值损失（MSE）- 使用tanh限制预测值
    auto predicted_values = torch::tanh(pred_value.squeeze(1));
    auto target_values_clamped = target_value.clamp(-1.0, 1.0);
    auto value_loss = torch::mse_loss(predicted_values, target_values_clamped);
    value_loss = torch::clamp(value_loss, 0.0, 10.0);
    
    // 门控损失（专家分化监督：KL正则 + 负载均衡 + logits L2正则）
    // 历史教训1：仅熵正则（0.05）→ one-hot坍缩，门控损失恒0.0000，MoE退化为单专家。
    // 历史教训2：自蒸馏交叉熵目标 → 梯度1e7被clip削掉，门控损失飙20失效。
    // 历史教训3：负载均衡(0.5) + 温度2.0 → 梯度被batch平均稀释(÷128)+clip，仍坍缩8.0。
    // 历史教训4：负载均衡 + 温度4.0 → 策略/价值梯度逐样本主导，仍坍缩8.0。
    // 历史教训5（根因）：KL(门控‖均匀)+负载均衡+温度1.0/权重10 → 诊断显示门控分布
    //   在2个epoch内从均匀坍缩成[0,0,0,0,0,0,0,1]。原因：softmax饱和——门控输出
    //   one-hot后，softmax雅可比在该区≈0，KL/熵正则的梯度传不回logits（策略/价值
    //   损失的"赢家通吃"梯度却畅通），正则完全失守。
    // 最终方案：对logits直接施加L2正则——梯度恒为2·logits/N，即使logits爆大也
    //   持续拉回，从源头限制幅度。配合KL(输出层)与负载均衡(batch级)实现分化。
    int num_experts = pred_gating.size(1);
    auto mean_gating = pred_gating.mean(0);  // [num_experts]，全batch平均路由概率
    auto load_balance = static_cast<float>(num_experts) *
                        (mean_gating * mean_gating).sum();
    // KL(pred ‖ uniform)：= -H(pred) + ln(N)，逐样本梯度，防单样本坍缩
    float uniform_log = std::log(1.0f / static_cast<float>(num_experts));
    auto gating_kl = (pred_gating * (pred_gating + 1e-8).log() -
                      pred_gating * uniform_log).sum(-1).mean();
    // logits L2正则：直接作用于门控logits（softmax前），防logits爆炸导致饱和坍缩
    torch::Tensor gating_logits_l2;
    if (pred_gating_logits.defined() && pred_gating_logits.numel() > 0) {
        gating_logits_l2 = pred_gating_logits.pow(2).mean();
    } else {
        gating_logits_l2 = torch::zeros({}, pred_gating.options());
    }
    // 组合：KL（逐样本防单样本坍缩）+ 负载均衡（batch级均匀使用）+ L2（源头限幅）
    // logits L2系数0.01：logits均值±2时损失≈0.04（轻微），±20时损失≈4（强拉回）
    torch::Tensor gating_loss = gating_kl + 0.5f * load_balance +
                                0.01f * gating_logits_l2;
    // 参考范围（8专家）：健康分化 ≈ 0.5~2.0，one-hot坍缩 ≈ 6.08+
    
    // 总损失 = 策略损失 + 价值损失 + 门控损失
    auto total_loss = config.policy_loss_weight * policy_loss +
                      config.value_loss_weight * value_loss +
                      config.gating_loss_weight * gating_loss;
    
    // GAE优势加权策略损失：用优势信号调整策略梯度方向
    // 优势>0时增强该动作概率，优势<0时压制
    if (advantages.defined() && advantages.numel() > 0 && config.advantage_loss_weight > 0.0f) {
        auto advantages_clamped = advantages.clamp(-5.0f, 5.0f);  // 防止极端优势值
        // 对目标策略中最高概率的动作施加优势导向的调整
        auto action_indices = target_policy.argmax(1);
        auto selected_log_probs = log_policy.gather(1, action_indices.unsqueeze(1)).squeeze(1);
        // 防发散（ELO门控在线训练实测）：模型把搜索最优动作压到1e-8时log≈-18.4，
        //   单样本损失贡献18.4×advantage(±5)=±92，一步更新即冲爆梯度（策略损失飙至17.8）。
        //   clip到[-6,0]（概率≥0.0025）限制单样本贡献≤30，收敛模型正常log_probs>-6不受影响。
        selected_log_probs = selected_log_probs.clamp(-6.0, 0.0);
        auto adv_weighted_policy_loss = -(selected_log_probs * advantages_clamped).mean();
        total_loss = total_loss + config.advantage_loss_weight * adv_weighted_policy_loss;
    }
    
    // 重要性采样权重（PER）
    if (!importance_weights.empty()) {
        auto importance_tensor = torch::tensor(importance_weights)
                                    .to(pred_policy.device())
                                    .unsqueeze(1);
        total_loss = total_loss * importance_tensor.mean();
    }
    
    result.policy_loss = policy_loss.item<float>();
    result.value_loss = value_loss.item<float>();
    result.gating_loss = gating_loss.item<float>();
    result.total_loss = total_loss.item<float>();
    
    return {total_loss, result};
}

MixedPrecisionTrainer::TrainStepResult MixedPrecisionTrainer::train_step(
    const EfficientReplayBuffer::TrainingBatch& batch) {
    
    TrainStepResult result{};
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 清零梯度
    optimizer->zero_grad();
    
    // 准备数据（如果数据已在GPU上则跳过传输，消除CPU-GPU传输瓶颈）
    auto states = batch.states.is_cuda() ? batch.states : batch.states.to(device);
    auto target_policies = batch.target_policies.is_cuda() ? batch.target_policies : batch.target_policies.to(device);
    auto target_values = batch.target_values.is_cuda() ? batch.target_values : batch.target_values.to(device);
    
    // 数据增强已改为生成时预存模式（SelfPlayGenerator::generate_game()中预存8种D4对称版本）
    // 训练时直接使用预存的增强数据，不再实时进行变换，节省训练时间

    // ==================== 对手风格条件化训练注入 ====================
    // 专家分化核心机制（围棋真实思维：看对手棋风调整对策）：
    // 以50%概率注入一个采样对手风格（原型池+噪声，见OpponentStyleProfiler），
    // 让门控学会"同一局面、面对不同棋风对手 → 分派不同专家"；
    // 剩余50%不注入（条件dropout）——保证模型在对手画像未知/自对弈等无条件
    // 场景下仍能正常发挥，且旧checkpoint（无风格投影层）行为完全不受影响。
    // 注：风格投影层零初始化，注入只提供"风格→专家"学习信号，不改已有棋力。
    // elo-gated模式通过set_opponent_style_conditioning(false)关闭：
    // 单模型对固定基线推理不注入画像，训练注入会造成train/eval不一致。
    if (opponent_style_conditioning_) {
        static thread_local std::mt19937 style_rng(static_cast<unsigned>(
            std::chrono::steady_clock::now().time_since_epoch().count()));
        static thread_local std::uniform_real_distribution<float> style_coin(0.0f, 1.0f);
        if (style_coin(style_rng) < 0.5f) {
            neural::OpponentStyleVector style =
                neural::OpponentStyleProfiler::sample_training_style(style_rng);
            auto style_arr = style.to_array();
            (*model)->set_opponent_style(
                std::vector<float>(style_arr.begin(), style_arr.end()));
        } else {
            (*model)->clear_opponent_style();
        }
    } else {
        // 无条件训练：确保模型不残留任何对手风格偏置
        (*model)->clear_opponent_style();
    }

    // 前向传播（统一FP32）
    // 重要修复：实测FP16 autocast训练（AMP）会破坏模型对弈能力——
    // Gen2/Gen3 AMP训练后对Gen0基线连续0胜（ELO=-798.3），且Pass能力崩坏（303步局67%）；
    // 同配置FP32训练（Gen1）ELO=127正常。1660 Ti无TensorCore，FP16无加速收益，
    // 故训练前向恢复纯FP32（GradScaler保留，FP32下scale/unscale数学等价无副作用）
    neural::CA_MOEInferenceResult inference_result;
    inference_result = (*model)->forward(states);
    
    // 调用统一损失计算函数
    auto advantages = batch.advantages.defined() ? 
        (batch.advantages.is_cuda() ? batch.advantages : batch.advantages.to(device)) :
        torch::Tensor();
    auto [total_loss, loss_result] = compute_losses(
        inference_result.final_policy,
        inference_result.final_value,
        inference_result.gating_weights,
        target_policies,
        target_values,
        batch.importance_weights,
        advantages,
        inference_result.gating_logits);

    // 诊断：每100个训练步打印一次门控分布，定位专家坍缩
    // （门控损失理论值：均匀=0.5，one-hot坍缩=6.08；若持续6+则门控未分化）
    static int gating_diag_step = 0;
    if ((++gating_diag_step % 100) == 1) {
        auto gm = inference_result.gating_weights.mean(0).detach().cpu();
        // batch内标准差：门控对"不同局面"的响应强度（饱和坍缩的判别指标）。
        //   ≈0 = 门控对所有局面都路由同一组专家（权重不随局面变化，fc1学不到东西）；
        //   ≥0.02 = 门控真正在按局面分化专家。配合均值(应趋于均匀1/8)一起看。
        auto gs = inference_result.gating_weights.std(0).detach().cpu();
        std::cout << "[门控诊断] batch均值=" << gm
                  << " | batch标准差=" << gs
                  << " | 门控损失=" << std::fixed << std::setprecision(4)
                  << loss_result.gating_loss << std::defaultfloat << std::endl;
    }
    
    // 反向传播
    if (use_amp) {
        scaler->scale(total_loss).backward();
        
        // 梯度裁剪
        if (config.gradient_clipping) {
            scaler->unscale_(*optimizer);
            torch::nn::utils::clip_grad_norm_((*model)->parameters(), 
                                              config.max_gradient_norm);
        }
        
        scaler->step(*optimizer);
        scaler->update();
    } else {
        total_loss.backward();
        
        // 梯度裁剪
        if (config.gradient_clipping) {
            // 诊断：裁剪前梯度总范数（与门控诊断同频打印）。
            //   范数≈0 → 梯度消失（策略log被clip进死区后梯度为0），权重不可能更新，
            //     这正是"同一棋谱连训8遍损失完全不变"的直接原因；
            //   范数极大 → 梯度爆炸，裁剪后步长被压缩。同时打印优化器实际lr，
            //   排查学习率调度器把lr归零导致step空转的可能。
            auto total_norm = torch::nn::utils::clip_grad_norm_(
                (*model)->parameters(), config.max_gradient_norm);
            if ((gating_diag_step % 100) == 1) {
                auto& param_groups = optimizer->param_groups();
                double lr_now = param_groups.empty() ? 0.0
                                                     : param_groups[0].options().get_lr();
                std::cout << "[梯度诊断] 裁剪前总范数=" << std::scientific
                          << static_cast<double>(total_norm) << std::defaultfloat
                          << " | 实际lr=" << lr_now << std::endl;
            }
        }
        
        optimizer->step();
    }
    
    // 记录结果
    result.policy_loss = loss_result.policy_loss;
    result.value_loss = loss_result.value_loss;
    result.gating_loss = loss_result.gating_loss;
    result.total_loss = loss_result.total_loss;
    result.learning_rate = current_lr;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.elapsed_time_ms = std::chrono::duration<float, std::milli>(
        end_time - start_time).count();
    
    total_steps_trained++;
    
    return result;
}

MixedPrecisionTrainer::EpochResult MixedPrecisionTrainer::train_epoch(
    EfficientReplayBuffer& replay_buffer) {
    
    EpochResult epoch_result{};
    auto epoch_start = std::chrono::high_resolution_clock::now();
    
    (*model)->set_train_mode();
    int num_steps = 0;
    
    // 计算每个epoch的步数
    int steps_per_epoch = replay_buffer.size() / config.batch_size;
    
    for (int step = 0; step < steps_per_epoch; ++step) {
        // 从经验池采样
        auto batch = replay_buffer.sample_batch(config.batch_size, "prioritized");
        
        if (batch.states.sizes()[0] == 0) {
            continue;
        }
        
        // 执行训练步骤
        auto step_result = train_step(batch);
        
        // 累积损失
        epoch_result.avg_policy_loss += step_result.policy_loss;
        epoch_result.avg_value_loss += step_result.value_loss;
        epoch_result.avg_gating_loss += step_result.gating_loss;
        epoch_result.avg_total_loss += step_result.total_loss;
        
        num_steps++;
        
        // 更新优先级（用于PER）
        std::vector<float> priorities(batch.buffer_indices.size());
        for (size_t i = 0; i < priorities.size(); ++i) {
            priorities[i] = step_result.total_loss;
        }
        replay_buffer.update_priorities(batch.buffer_indices, priorities);
    }

    // 学习率调度：每epoch结束后step一次，避免余弦退火过快将学习率降到0
    // 之前每batch step导致连续多代训练后lr退火到接近0
    lr_scheduler->step();
    current_lr = lr_scheduler->get_last_lr()[0];
    
    // 平均化损失
    if (num_steps > 0) {
        epoch_result.avg_policy_loss /= num_steps;
        epoch_result.avg_value_loss /= num_steps;
        epoch_result.avg_gating_loss /= num_steps;
        epoch_result.avg_total_loss /= num_steps;
    }
    
    epoch_result.num_steps = num_steps;
    
    auto epoch_end = std::chrono::high_resolution_clock::now();
    epoch_result.epoch_time_sec = std::chrono::duration<float>(
        epoch_end - epoch_start).count();
    
    return epoch_result;
}

MixedPrecisionTrainer::EpochResult MixedPrecisionTrainer::evaluate_validation(
    EfficientReplayBuffer& replay_buffer, int val_batch_count) {
    
    EpochResult result{};
    auto eval_start = std::chrono::high_resolution_clock::now();
    
    // 切换到评估模式（禁用Dropout和BatchNorm的更新）
    (*model)->set_eval_mode();
    
    int num_steps = 0;
    torch::NoGradGuard no_grad;  // 禁用梯度计算，节省内存和加速
    
    for (int step = 0; step < val_batch_count; ++step) {
        auto batch = replay_buffer.sample_batch(config.batch_size, "uniform");
        
        if (batch.states.sizes()[0] == 0) {
            continue;
        }
        
        // 准备数据（如果数据已在GPU上则跳过传输）
        auto states = batch.states.is_cuda() ? batch.states : batch.states.to(device);
        auto target_policies = batch.target_policies.is_cuda() ? batch.target_policies : batch.target_policies.to(device);
        auto target_values = batch.target_values.is_cuda() ? batch.target_values : batch.target_values.to(device);
        
        // 前向传播（不计算梯度）
        // eval模式下forward返回CPU tensor（避免跨线程CUDA context问题）
        // 验证阶段在单线程中运行，需要将结果移回GPU以与target_policies/target_values匹配
        auto inference_result = (*model)->forward(states);
        inference_result.final_policy = inference_result.final_policy.to(device);
        inference_result.final_value = inference_result.final_value.to(device);
        inference_result.gating_weights = inference_result.gating_weights.to(device);
        
        // 调用统一损失计算函数（验证集不应用重要性权重）
        auto [_, loss_result] = compute_losses(
            inference_result.final_policy,
            inference_result.final_value,
            inference_result.gating_weights,
            target_policies,
            target_values);
        
        result.avg_policy_loss += loss_result.policy_loss;
        result.avg_value_loss += loss_result.value_loss;
        result.avg_gating_loss += loss_result.gating_loss;
        result.avg_total_loss += loss_result.total_loss;
        num_steps++;
    }
    
    // 恢复训练模式
    (*model)->set_train_mode();
    
    // 平均化损失
    if (num_steps > 0) {
        result.avg_policy_loss /= num_steps;
        result.avg_value_loss /= num_steps;
        result.avg_gating_loss /= num_steps;
        result.avg_total_loss /= num_steps;
    }
    result.num_steps = num_steps;
    
    auto eval_end = std::chrono::high_resolution_clock::now();
    result.epoch_time_sec = std::chrono::duration<float>(eval_end - eval_start).count();
    
    return result;
}

void MixedPrecisionTrainer::adjust_learning_rate(float new_lr) {
    current_lr = new_lr;
    
    for (auto& param_group : optimizer->param_groups()) {
        static_cast<torch::optim::AdamOptions&>(param_group.options()).lr(new_lr);
    }
}

void MixedPrecisionTrainer::step_lr_scheduler(int iteration) {
    // 余弦退火学习率调度
    float progress = static_cast<float>(iteration) / 10000.0f;  // 总迭代次数
    float new_lr = config.learning_rate * (1.0f + std::cos(M_PI * progress)) / 2.0f;
    adjust_learning_rate(new_lr);
}

void MixedPrecisionTrainer::zero_grad() {
    optimizer->zero_grad();
}

void MixedPrecisionTrainer::save_model(const std::string& path) {
    (*model)->set_eval_mode();
    try {
        // 尝试直接保存（LibTorch支持GPU tensor直接序列化）
        (*model)->save(path);
    } catch (...) {
        // 如果直接保存失败，回退到CPU中转
        (*model)->to(torch::kCPU);
        (*model)->save(path);
        (*model)->to(device);
    }
}

void MixedPrecisionTrainer::load_model(const std::string& path) {
    (*model)->load(path);
    if (device.is_cuda()) {
        (*model)->to(device);
    }
}

float MixedPrecisionTrainer::get_learning_rate() const {
    return current_lr;
}

// ==================== ELOEvaluator 实现 ====================

ELOEvaluator::ELOEvaluator(
    std::shared_ptr<neural::CA_MOEExpertSystem> current_model,
    std::shared_ptr<neural::CA_MOEExpertSystem> baseline_model,
    const EvaluationConfig& config)
    : current_model(current_model)
    , baseline_model(baseline_model)
    , config(config)
    , rng(std::random_device{}())
{
}

ELOEvaluator::EvaluationResult ELOEvaluator::evaluate() {
    // 真实对弈评估：current_model vs baseline_model
    // 单线程执行，避免GPU资源竞争
    EvaluationResult result{};
    result.games_played = config.num_evaluation_games;
    
    auto eval_start = std::chrono::high_resolution_clock::now();
    
    int wins = 0, losses = 0, draws = 0;
    
    // 单线程顺序执行所有评估对局
    for (int g = 0; g < config.num_evaluation_games; ++g) {
        // 交替先后手：偶数局current_model执黑，奇数局执白
        bool current_first = (g % 2 == 0);
        float game_outcome = play_single_game(current_first);
        
        if (game_outcome > 0.5f) {
            wins++;   // 当前模型获胜
        } else if (game_outcome < -0.5f) {
            losses++; // 基准模型获胜
        } else {
            draws++;  // 平局
        }
    }
    
    // 计算结果
    result.wins = wins;
    result.losses = losses;
    result.draws = draws;
    // 实际胜率：胜+0.5*平（ELO标准计算方式）
    float score = static_cast<float>(wins) + 0.5f * static_cast<float>(draws);
    result.win_rate = score / std::max(1, result.games_played);
    result.avg_game_length = 0.0f;  // 真实对弈模式下暂不跟踪对局长度
    
    // 计算ELO变化（标准ELO公式）
    result.elo_rating = calculate_elo(result.win_rate);
    result.elo_change = result.elo_rating - current_elo;
    current_elo = result.elo_rating;
    
    // 计算置信区间（Wilson区间）
    float n = static_cast<float>(result.games_played);
    float p = result.win_rate;
    float z = 1.96f;  // 95%置信水平
    float denominator = 1.0f + z * z / n;
    result.confidence_interval_lower = (p + z * z / (2.0f * n) - 
                                        z * std::sqrt((p * (1.0f - p) + z * z / (4.0f * n)) / n)) / denominator;
    result.confidence_interval_upper = (p + z * z / (2.0f * n) + 
                                        z * std::sqrt((p * (1.0f - p) + z * z / (4.0f * n)) / n)) / denominator;
    
    auto eval_end = std::chrono::high_resolution_clock::now();
    result.evaluation_duration = std::chrono::duration_cast<std::chrono::duration<float>>(eval_end - eval_start);
    
    history.push_back(result);
    
    return result;
}

float ELOEvaluator::play_single_game(bool current_first) {
    // 真实对弈评估：current_model与baseline_model交替落子
    // current_first=true: current_model执黑，baseline_model执白
    // current_first=false: current_model执白，baseline_model执黑
    // 返回值：+1=当前模型胜，-1=基准模型胜，0=平局
    
    try {
        GoBoardCPU board;
        int consecutive_passes = 0;
        int max_moves = config.max_eval_game_length;
        
        for (int move_num = 0; move_num < max_moves; ++move_num) {
            StoneColor player = board.get_current_player();
            // 判断当前回合使用哪个模型
            // player==BLACK(偶数步)且current_first -> current_model
            // player==WHITE(奇数步)且current_first -> baseline_model
            // player==BLACK(偶数步)且!current_first -> baseline_model
            // player==WHITE(奇数步)且!current_first -> current_model
            bool is_black_turn = (player == StoneColor::BLACK);
            bool use_current = (is_black_turn == current_first);
            auto& model = use_current ? current_model : baseline_model;
            
            // 编码棋盘状态（复用预分配buffer避免每步分配）
            torch::Tensor state = torch::empty({17, 19, 19});
            board.encode_to_tensor(state.data_ptr<float>(), 17);
            
            // 移到推理设备
            torch::Tensor batch_state = state.unsqueeze(0);  // [1, 17, 19, 19]
            if (eval_device.is_cuda()) {
                batch_state = batch_state.to(eval_device);
            }
            
            // 神经网络推理
            torch::NoGradGuard no_grad;
            auto inference_result = (*model)->forward(batch_state);
            auto policy = inference_result.final_policy.squeeze(0);  // [362]
            
            // 移到CPU进行动作选择
            auto policy_cpu = policy.to(torch::kCPU);
            auto policy_data = policy_cpu.contiguous().data_ptr<float>();
            
            // 构建合法着法mask
            std::vector<bool> legal_mask(362, false);
            auto legal_moves = board.get_legal_moves(player);
            for (const auto& [x, y] : legal_moves) {
                int idx = GoBoardCPU::xy_to_index(x, y);
                if (idx >= 0 && idx < 361) {
                    legal_mask[idx] = true;
                }
            }
            legal_mask[361] = true;  // pass始终合法
            
            // 贪婪选择：在合法动作中选择概率最高的
            int best_action = 361;  // 默认pass
            float best_prob = -1.0f;
            for (int i = 0; i < 362; ++i) {
                if (legal_mask[i] && policy_data[i] > best_prob) {
                    best_prob = policy_data[i];
                    best_action = i;
                }
            }
            
            // 执行动作
            if (best_action == 361) {
                // Pass
                consecutive_passes++;
                if (consecutive_passes >= 2) {
                    break;  // 双方pass，游戏结束
                }
                board.set_current_player(opponent_color(player));
                board.set_move_count(board.get_move_count() + 1);
            } else {
                consecutive_passes = 0;
                auto [x, y] = GoBoardCPU::index_to_xy(best_action);
                board.make_move(x, y, player);
            }
        }
        
        // 计算结果：基于棋盘棋子数量（简化版中国规则面积计算）
        int black_stones = 0, white_stones = 0;
        for (int y = 0; y < 19; ++y) {
            for (int x = 0; x < 19; ++x) {
                if (board.is_black(x, y)) black_stones++;
                else if (board.is_white(x, y)) white_stones++;
            }
        }
        // 白方贴目
        float white_score = white_stones + config.komi;
        float black_score = static_cast<float>(black_stones);
        
        bool black_wins = black_score > white_score;
        bool is_draw = (black_score == white_score);
        
        if (is_draw) {
            return 0.0f;
        }
        
        // 判断当前模型是否获胜
        bool current_wins = (black_wins == current_first);
        return current_wins ? 1.0f : -1.0f;
        
    } catch (const std::exception& e) {
        std::cerr << "[ELOEvaluator] 评估对弈异常: " << e.what() << std::endl;
        return 0.0f;  // 异常时记为平局
    }
}

float ELOEvaluator::calculate_elo(float win_rate, float k_factor) {
    // 标准ELO公式：new_elo = old_elo + K * (actual - expected)
    // K=32为标准业余级别K因子
    // expected基于当前ELO与基准ELO的差值计算，随ELO增长自然递减
    // 这确保ELO增长有递减收益，反映真实棋力差距
    
    float expected = expected_win_rate(current_elo, baseline_elo);
    float elo_change = k_factor * (win_rate - expected);
    return current_elo + elo_change;
}

float ELOEvaluator::expected_win_rate(float elo_a, float elo_b) {
    // 预期胜率计算
    return 1.0f / (1.0f + std::pow(10.0f, (elo_b - elo_a) / 400.0f));
}

// ==================== UnsupervisedSelfReinforcementTrainer 主实现 ====================

UnsupervisedSelfReinforcementTrainer::UnsupervisedSelfReinforcementTrainer(
    const UnsupervisedTrainingConfig& config)
    : config(config)
{
    logger = &framework::Logger::instance();
    perf_monitor = std::make_unique<framework::PerformanceMonitor>();
}

UnsupervisedSelfReinforcementTrainer::~UnsupervisedSelfReinforcementTrainer() {
    stop_training();
}

bool UnsupervisedSelfReinforcementTrainer::initialize() {
    try {
        std::cout << "\n========================================" << std::endl;
        std::cout << "初始化无监督自强化训练系统" << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        // 1. 创建CA-MOE神经网络模型（传入dropout_rate防止过拟合）
        std::cout << "[1/6] 创建CA-MOE混合专家网络..." << std::endl;
        float dropout_rate = config.training.dropout_rate;
        current_model = std::make_shared<neural::CA_MOEExpertSystem>(8, 19, dropout_rate);
        best_model = std::make_shared<neural::CA_MOEExpertSystem>(8, 19, dropout_rate);
        baseline_model = std::make_shared<neural::CA_MOEExpertSystem>(8, 19, dropout_rate);
        
        // 复制初始权重到best和baseline模型
        // 注意：实际实现中需要深拷贝参数
        
        // 根据配置决定模型设备：只有config.self_play.use_cuda为true且CUDA可用时才使用GPU
        // 不能仅检查torch::cuda::is_available()，否则--no-cuda时模型仍在GPU上
        bool use_cuda = config.self_play.use_cuda && torch::cuda::is_available();
        if (use_cuda) {
            (*current_model)->to(torch::kCUDA);
            (*best_model)->to(torch::kCUDA);
            (*baseline_model)->to(torch::kCUDA);
            std::cout << "      模型已加载到GPU" << std::endl;
        }
        std::cout << "      完成！" << std::endl;

        // 1.5. 创建推理收集器（批量打包Worker推理请求，提升GPU利用率）
        // 注意：当前禁用InferenceCollector，直接使用同步推理
        // InferenceCollector存在tensor生命周期问题，后续修复后重新启用
        std::cout << "[1.5/6] 跳过推理收集器（使用同步推理模式）..." << std::endl;
        inference_collector_ = nullptr;  // 禁用InferenceCollector
        std::cout << "      完成！" << std::endl;

        // 2. 创建自我对弈生成器
        std::cout << "[2/6] 创建自我对弈生成器..." << std::endl;
        generators.reserve(config.self_play.num_workers);
        for (int i = 0; i < config.self_play.num_workers; ++i) {
            auto generator = std::make_shared<SelfPlayGenerator>(
                i, config.self_play, current_model, inference_collector_);
            
            if (generator->initialize()) {
                generators.push_back(generator);
                std::cout << "      Worker-" << i << " 初始化完成" << std::endl;
            }
        }
        std::cout << "      完成！共创建 " << generators.size() << " 个Worker" << std::endl;
        
        // 3. 创建经验回放缓冲区
        std::cout << "[3/6] 创建经验回放缓冲区..." << std::endl;
        // 渐进式扩张：使用初始容量创建小池快速启动
        // 非渐进模式：直接使用max_window_size（与原行为一致）
        int initial_buffer_cap = config.training.progressive_expansion
            ? config.training.initial_window_size
            : config.training.max_window_size;
        replay_buffer = std::make_shared<EfficientReplayBuffer>(
            initial_buffer_cap, 8);
        // 设置最小样本需求为批次大小的5倍，确保第一次迭代后即可开始训练
        // 每局游戏平均产生约24个样本，100局约2400个样本
        replay_buffer->set_min_samples(config.training.batch_size * 5);
        std::cout << "      容量: " << initial_buffer_cap << " 条经验" << std::endl;
        if (config.training.progressive_expansion) {
            std::cout << "      渐进扩张: 启用 (最大: " << config.training.max_window_size
                      << ", 倍数: " << config.training.expansion_factor
                      << ", 间隔: " << config.training.expansion_interval << " 迭代)" << std::endl;
        }
        std::cout << "      完成！" << std::endl;
        
        // 尝试启用GPU预装存储以消除CPU-GPU传输瓶颈
        // 注意：GPU显存紧张时可能导致tensor底层存储被回收，暂时禁用
        // if (torch::cuda::is_available()) {
        //     replay_buffer->init_gpu_storage(torch::Device(torch::kCUDA), 
        //                                     initial_buffer_cap, 8);
        // }
        
        // 4. 创建混合精度训练器
        std::cout << "[4/6] 创建混合精度训练器..." << std::endl;
        trainer = std::make_shared<MixedPrecisionTrainer>(
            current_model, config.training);
        trainer->setup_optimizer();
        std::cout << "      完成！" << std::endl;
        
        // 5. 创建ELO评估器
        std::cout << "[5/6] 创建ELO评估系统..." << std::endl;
        evaluator = std::make_shared<ELOEvaluator>(
            current_model, baseline_model, config.evaluation);
        // 设置评估对弈的推理设备（与模型所在设备一致）
        if (use_cuda) {
            evaluator->set_eval_device(torch::Device(torch::kCUDA));
        }
        std::cout << "      评估间隔: 每 " << config.evaluation.evaluation_interval 
                  << " 次迭代" << std::endl;
        std::cout << "      评估对局: 每次 " << config.evaluation.num_evaluation_games 
                  << " 局（真实对弈）" << std::endl;
        std::cout << "      完成！" << std::endl;
        
        // 6. 创建必要的目录
        std::cout << "[6/6] 准备工作目录..." << std::endl;
        ensure_directory_exists(config.checkpoint_dir);
        ensure_directory_exists(config.log_dir);
        // 创建SGF棋谱输出目录
        if (config.self_play.save_sgf_games) {
            ensure_directory_exists(config.self_play.sgf_output_dir);
            std::cout << "      SGF棋谱目录: " << config.self_play.sgf_output_dir << std::endl;
        }
        std::cout << "      检查点目录: " << config.checkpoint_dir << std::endl;
        std::cout << "      日志目录: " << config.log_dir << std::endl;
        std::cout << "      完成！" << std::endl;
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "初始化完成！" << std::endl;
        std::cout << "总迭代次数: " << config.total_iterations << std::endl;
        std::cout << "检查点间隔: 每 " << config.checkpoint_interval << " 次迭代" << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "\n[错误] 初始化失败: " << e.what() << std::endl;
        logger->fatal("训练系统初始化失败: " + std::string(e.what()));
        return false;
    }
}

void UnsupervisedSelfReinforcementTrainer::start_training() {
    if (training.exchange(true)) {
        std::cout << "[警告] 训练已在运行中" << std::endl;
        return;
    }
    
    std::cout << "\n开始无监督自强化训练..." << std::endl;
    std::cout << "开始时间: " << timestamp_string() << std::endl;
    
    training_thread = std::thread(&UnsupervisedSelfReinforcementTrainer::training_loop, this);
}

void UnsupervisedSelfReinforcementTrainer::stop_training() {
    if (!training.exchange(false)) {
        return;
    }
    
    std::cout << "\n正在停止训练..." << std::endl;
    
    // 停止所有生成器
    for (auto& gen : generators) {
        gen->stop();
    }

    // 停止推理收集器
    if (inference_collector_) {
        inference_collector_->stop();
        // 打印统计信息
        auto stats = inference_collector_->get_stats();
        std::cout << "[推理收集器] 总处理任务: " << stats.total_tasks_processed
                  << ", 总批次数: " << stats.total_batches
                  << ", 平均批量: " << stats.avg_batch_size
                  << ", 平均延迟: " << stats.avg_latency_ms << "ms" << std::endl;
    }
    
    // 等待训练线程结束
    if (training_thread.joinable()) {
        training_thread.join();
    }
    
    // 保存最终检查点
    save_checkpoint();
    
    std::cout << "训练已停止" << std::endl;
    std::cout << "总迭代次数: " << current_iteration.load() << std::endl;
    std::cout << "总对局数: " << total_games.load() << std::endl;
    std::cout << "最佳ELO: " << best_elo << std::endl;
}

TrainingMetrics UnsupervisedSelfReinforcementTrainer::run_single_iteration() {
    TrainingMetrics metrics{};
    metrics.iteration = current_iteration.load();
    
    auto iter_start = std::chrono::high_resolution_clock::now();
    
    // 阶段1：自我对弈数据生成
    auto sp_start = std::chrono::high_resolution_clock::now();
    run_self_play_phase();
    auto sp_end = std::chrono::high_resolution_clock::now();
    metrics.self_play_time_sec = std::chrono::duration<float>(sp_end - sp_start).count();
    
    // 渐进式经验池扩张：在自我对弈后、训练前检查是否需要扩容
    // 倍增扩张策略：每expansion_interval次迭代将经验池容量翻倍，直到达到max_window_size
    if (config.training.progressive_expansion) {
        int iteration = current_iteration.load();
        if (iteration > 0 && iteration % config.training.expansion_interval == 0) {
            int current_cap = replay_buffer->capacity();
            int max_cap = config.training.max_window_size;
            if (current_cap < max_cap) {
                int new_cap = std::min(
                    static_cast<int>(current_cap * config.training.expansion_factor),
                    max_cap);
                std::cout << "[扩张] 经验池扩张: " << current_cap
                          << " → " << new_cap << " (迭代 " << iteration << ")" << std::endl;
                replay_buffer->expand_capacity(new_cap);
            }
        }
    }
    
    // 阶段2：神经网络训练
    auto train_start = std::chrono::high_resolution_clock::now();
    run_training_phase();
    auto train_end = std::chrono::high_resolution_clock::now();
    metrics.training_time_sec = std::chrono::duration<float>(train_end - train_start).count();
    
    // 将训练损失填充到metrics中（从run_training_phase保存的成员变量读取）
    metrics.avg_policy_loss = latest_train_policy_loss;
    metrics.avg_value_loss = latest_train_value_loss;
    metrics.avg_gating_loss = latest_train_gating_loss;
    metrics.avg_total_loss = latest_train_total_loss;
    
    // 将验证集评估结果填充到metrics中
    if (has_latest_validation) {
        metrics.val_policy_loss = latest_val_policy_loss;
        metrics.val_value_loss = latest_val_value_loss;
        metrics.val_total_loss = latest_val_total_loss;
        metrics.has_validation = true;
    }
    
    // 阶段3：评估（如果到了评估间隔）
    // 使用 (iteration + 1) 确保第一次迭代后也进行评估
    if ((metrics.iteration + 1) % config.evaluation.evaluation_interval == 0) {
        auto eval_start = std::chrono::high_resolution_clock::now();
        // 切换模型到评估模式（关闭dropout和batchnorm训练行为）
        (*current_model)->eval();
        (*baseline_model)->eval();
        try {
            run_evaluation_phase();
        } catch (const std::exception& e) {
            std::cerr << "[评估异常] " << e.what() << std::endl;
            std::cout << "[评估] 评估阶段异常，跳过本次评估: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[评估异常] 未知异常" << std::endl;
            std::cout << "[评估] 评估阶段未知异常，跳过本次评估" << std::endl;
        }
        // 恢复模型到训练模式
        (*current_model)->train();
        (*baseline_model)->train();
        auto eval_end = std::chrono::high_resolution_clock::now();
        metrics.evaluation_time_sec = std::chrono::duration<float>(eval_end - eval_start).count();
    }
    
    auto iter_end = std::chrono::high_resolution_clock::now();
    metrics.total_time_sec = std::chrono::duration<float>(iter_end - iter_start).count();
    
    // 更新指标
    metrics.total_games_generated = total_games.load();
    metrics.total_training_steps = total_steps.load();
    metrics.current_elo = evaluator->get_history().empty() ? 
                          1200.0f : evaluator->get_history().back().elo_rating;
    
    // 计算效率指标
    if (metrics.total_time_sec > 0) {
        metrics.games_per_second = static_cast<float>(
            config.self_play.num_workers * config.training.games_per_iteration) / 
            metrics.self_play_time_sec;
    }
    
    latest_metrics = metrics;
    training_history.push_back(metrics);
    
    // 增加迭代计数
    current_iteration++;
    
    return metrics;
}

void UnsupervisedSelfReinforcementTrainer::training_loop() {
    // 在训练线程中初始化CUDA context
    // PyTorch CUDA context是线程本地的，模型在主线程创建并移到GPU，
    // 但训练线程必须先初始化自己的CUDA context才能正确访问GPU上的tensor
#ifdef USE_CUDA
    if (config.self_play.use_cuda && torch::cuda::is_available()) {
        // 调用cudaSetDevice初始化当前线程的CUDA context
        // 这样当前线程就能访问主线程创建的GPU tensor
        cudaSetDevice(config.self_play.cuda_device_id);
        // 确保CUDA context已就绪
        torch::cuda::synchronize();
    }
#endif

    while (training.load() && current_iteration.load() < config.total_iterations) {
        // 检查是否应该早停
        if (should_stop_early()) {
            std::cout << "\n[早停] 连续 " << iterations_without_improvement 
                      << " 次迭代无改进，停止训练" << std::endl;
            break;
        }

        // 执行单次迭代
        auto metrics = run_single_iteration();
        
        // 打印进度
        log_iteration_complete(metrics);
        
        // 定期保存检查点
        if (metrics.iteration > 0 && 
            metrics.iteration % config.checkpoint_interval == 0) {
            save_checkpoint();
        }
        
        // 定期打印详细日志
        if (metrics.iteration % config.log_interval == 0) {
            log_training_progress(metrics);
        }
    }
    
    // 训练完成
    training.store(false);
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "训练完成！" << std::endl;
    std::cout << "总迭代次数: " << current_iteration.load() << std::endl;
    std::cout << "总对局数: " << total_games.load() << std::endl;
    std::cout << "总训练步骤: " << total_steps.load() << std::endl;
    std::cout << "最终ELO: " << latest_metrics.current_elo << std::endl;
    std::cout << "最佳ELO: " << best_elo << std::endl;
    std::cout << "总耗时: " << latest_metrics.total_time_sec * current_iteration.load() 
              << " 秒" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 保存最终检查点和报告
    save_checkpoint();
    generate_training_report(config.log_dir + "/final_report.txt");
}

void UnsupervisedSelfReinforcementTrainer::run_self_play_phase() {
    int games_per_worker = config.training.games_per_iteration /
                           config.self_play.num_workers;

    // 顺序执行自我对弈（不使用std::async）
    // 原因：PyTorch CUDA context是线程本地的，std::async创建的新线程
    // 无法正确访问主线程的CUDA context，导致forward返回的tensor
    // 虽然defined()为true但底层存储已失效，任何操作都报"Expected a proper Tensor but got None"
    // 此外，inference_mutex_已经序列化了所有forward调用，std::async并无实际GPU并行收益
    // ELO评估（play_single_game）在主线程中执行同样的forward操作则完全正常
    std::vector<GameData> all_games;
    for (size_t i = 0; i < generators.size(); ++i) {
        int start_id = total_games.load() + static_cast<int>(i) * games_per_worker;
        auto games = generators[i]->generate_games(start_id, games_per_worker);
        all_games.insert(all_games.end(),
                        std::make_move_iterator(games.begin()),
                        std::make_move_iterator(games.end()));
    }

    // 将游戏数据添加到经验缓冲区
    if (!all_games.empty()) {
        replay_buffer->add_games(all_games);
        total_games.fetch_add(static_cast<int>(all_games.size()));
    }

    // 温度衰减已移至 generate_games() 中每局结束后执行
    // 此处不再重复衰减，避免双重衰减导致温度下降过快
}

void UnsupervisedSelfReinforcementTrainer::run_training_phase() {
    if (!replay_buffer->is_ready()) {
        std::cout << "[警告] 经验池未准备好（当前大小: " << replay_buffer->size()
                  << ", 最小需求: " << replay_buffer->get_min_samples() << "）" << std::endl;
        return;
    }

    std::cout << "[训练] 开始神经网络训练，经验池大小: " << replay_buffer->size() << std::endl;

    // 训练指定数量的epoch
    for (int epoch = 0; epoch < config.training.num_epochs_per_iteration; ++epoch) {
        std::cout << "[训练] Epoch " << (epoch + 1) << "/" << config.training.num_epochs_per_iteration
                  << "..." << std::flush;
        auto epoch_result = trainer->train_epoch(*replay_buffer);
        total_steps.fetch_add(epoch_result.num_steps);

        // 保存训练损失到成员变量，供run_single_iteration填充到metrics
        latest_train_policy_loss = epoch_result.avg_policy_loss;
        latest_train_value_loss = epoch_result.avg_value_loss;
        latest_train_gating_loss = epoch_result.avg_gating_loss;
        latest_train_total_loss = epoch_result.avg_total_loss;

        std::cout << " 完成 (步数: " << epoch_result.num_steps
                  << ", 策略损失: " << std::fixed << std::setprecision(6) << epoch_result.avg_policy_loss
                  << ", 价值损失: " << epoch_result.avg_value_loss
                  << ", 总损失: " << epoch_result.avg_total_loss << ")" << std::endl;

        // 更新学习率
        trainer->step_lr_scheduler(current_iteration.load());
    }

    // ===== 验证集评估（过拟合检测） =====
    // 训练完成后在验证集上评估损失，与训练损失对比以检测过拟合
    if (config.training.validation_ratio > 0.0f) {
        std::cout << "[验证] 评估验证集损失..." << std::flush;
        auto val_result = trainer->evaluate_validation(*replay_buffer);

        // 保存验证集损失到成员变量，供run_single_iteration读取
        latest_val_policy_loss = val_result.avg_policy_loss;
        latest_val_value_loss = val_result.avg_value_loss;
        latest_val_total_loss = val_result.avg_total_loss;
        has_latest_validation = true;

        std::cout << " 完成 (策略: " << std::fixed << std::setprecision(6) << val_result.avg_policy_loss
                  << ", 价值: " << val_result.avg_value_loss
                  << ", 总损失: " << val_result.avg_total_loss << ")" << std::endl;

        // 更新最佳验证集损失，跟踪连续未改善次数
        if (val_result.avg_total_loss < best_val_loss) {
            std::cout << "[验证] ★ 验证集损失改善 (" << std::fixed << std::setprecision(6)
                      << best_val_loss << " → " << val_result.avg_total_loss << ")" << std::endl;
            best_val_loss = val_result.avg_total_loss;
            iterations_without_val_improvement = 0;
        } else {
            iterations_without_val_improvement++;
            std::cout << "[验证] 验证集损失未改善 (连续 " << iterations_without_val_improvement
                      << "/" << config.validation_early_stop_patience << " 次)" << std::endl;

            // 过拟合预警：训练损失下降但验证损失上升
            if (iterations_without_val_improvement >= config.validation_early_stop_patience / 2) {
                std::cout << "[验证] ⚠ 警告：验证集损失持续未改善，可能存在过拟合风险" << std::endl;
            }
        }
    }
}

void UnsupervisedSelfReinforcementTrainer::run_evaluation_phase() {
    std::cout << "\n--- 评估阶段（真实对弈） ---" << std::endl;
    
    auto eval_result = evaluator->evaluate();
    
    std::cout << "评估结果:" << std::endl;
    std::cout << "  对局数: " << eval_result.games_played << "（真实对弈）" << std::endl;
    std::cout << "  胜/负/平: " << eval_result.wins << "/" 
              << eval_result.losses << "/" << eval_result.draws << std::endl;
    std::cout << "  胜率: " << std::fixed << std::setprecision(2) 
              << (eval_result.win_rate * 100.0f) << "%（含平局计0.5胜）" << std::endl;
    std::cout << "  ELO评分: " << std::fixed << std::setprecision(1) 
              << eval_result.elo_rating << std::endl;
    std::cout << "  ELO变化: " << (eval_result.elo_change >= 0 ? "+" : "") 
              << eval_result.elo_change << std::endl;
    std::cout << "  置信区间: [" << std::fixed << std::setprecision(2) 
              << (eval_result.confidence_interval_lower * 100.0f) << "%, "
              << (eval_result.confidence_interval_upper * 100.0f) << "%]" << std::endl;
    std::cout << "  评估耗时: " << std::fixed << std::setprecision(1) 
              << eval_result.evaluation_duration.count() << "秒" << std::endl;
    
    // 更新最佳模型
    if (eval_result.elo_rating > best_elo) {
        std::cout << "  ★ 新的最佳模型！（ELO提升 " 
                  << (eval_result.elo_rating - best_elo) << "）" << std::endl;
        
        std::lock_guard<std::mutex> lock(model_mutex);
        best_elo = eval_result.elo_rating;
        // 复制当前模型到best_model
        // 注意：实际实现中需要深拷贝
        iterations_without_improvement = 0;
    } else {
        iterations_without_improvement++;
    }
    
    // 更新baseline模型（定期）
    if (current_iteration.load() % (config.evaluation.evaluation_interval * 5) == 0) {
        std::cout << "  更新基准模型..." << std::endl;
        // 复制当前模型到baseline_model
    }
}

void UnsupervisedSelfReinforcementTrainer::update_models_if_needed() {
    // 可以在这里实现模型更新逻辑
    // 例如：定期同步worker模型等
}

bool UnsupervisedSelfReinforcementTrainer::should_stop_early() const {
    if (config.num_early_stopping_iterations <= 0) {
        return false;
    }
    
    // 检查ELO无改进
    if (iterations_without_improvement >= config.num_early_stopping_iterations) {
        return true;
    }
    
    // 检查验证集损失持续未改善（过拟合信号）
    // 当验证集损失连续多次未改善时，判定为过拟合，触发早停
    if (has_latest_validation && 
        iterations_without_val_improvement >= config.validation_early_stop_patience) {
        std::cout << "[早停] 验证集损失连续 " << iterations_without_val_improvement
                  << " 次迭代未改善，检测到过拟合趋势" << std::endl;
        return true;
    }
    
    return false;
}

void UnsupervisedSelfReinforcementTrainer::save_checkpoint(const std::string& path) {
    std::string checkpoint_path = path.empty() ? 
        get_latest_checkpoint_path() : path;
    
    try {
        std::cout << "\n保存检查点到: " << checkpoint_path << std::endl;
        
        // 保存当前模型
        trainer->save_model(checkpoint_path + "_current.pt");
        
        // 保存最佳模型（锁外执行GPU操作，锁仅保护模型指针访问）
        std::shared_ptr<neural::CA_MOEExpertSystem> model_to_save;
        {
            std::lock_guard<std::mutex> lock(model_mutex);
            model_to_save = best_model;  // shared_ptr拷贝，仅需短暂持锁
        }
        
        (*model_to_save)->set_eval_mode();
        if (torch::cuda::is_available()) {
            (*model_to_save)->to(torch::kCPU);  // 锁外GPU同步
        }
        (*model_to_save)->save(checkpoint_path + "_best.pt");
        if (torch::cuda::is_available()) {
            (*model_to_save)->to(torch::kCUDA);  // 锁外GPU同步
        }
        
        // 保存训练状态
        std::ofstream state_file(checkpoint_path + "_state.txt");
        state_file << "iteration=" << current_iteration.load() << "\n";
        state_file << "total_games=" << total_games.load() << "\n";
        state_file << "total_steps=" << total_steps.load() << "\n";
        state_file << "best_elo=" << best_elo << "\n";
        state_file << "iterations_without_improvement=" << iterations_without_improvement << "\n";
        state_file.close();
        
        // 保存训练历史
        std::ofstream history_file(checkpoint_path + "_history.csv");
        history_file << "iteration,games,steps,policy_loss,value_loss,gating_loss,"
                     << "total_loss,elo,elo_change,win_rate,self_play_time,"
                     << "training_time,evaluation_time\n";
        
        for (const auto& metrics : training_history) {
            history_file << metrics.iteration << ","
                        << metrics.total_games_generated << ","
                        << metrics.total_training_steps << ","
                        << metrics.avg_policy_loss << ","
                        << metrics.avg_value_loss << ","
                        << metrics.avg_gating_loss << ","
                        << metrics.avg_total_loss << ","
                        << metrics.current_elo << ","
                        << metrics.elo_change << ","
                        << metrics.win_rate_vs_baseline << ","
                        << metrics.self_play_time_sec << ","
                        << metrics.training_time_sec << ","
                        << metrics.evaluation_time_sec << "\n";
        }
        history_file.close();
        
        std::cout << "检查点保存完成" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[错误] 保存检查点失败: " << e.what() << std::endl;
        logger->error("保存检查点失败: " + std::string(e.what()));
    }
}

void UnsupervisedSelfReinforcementTrainer::load_checkpoint(const std::string& path) {
    try {
        std::cout << "加载检查点: " << path << std::endl;
        
        // 加载模型
        trainer->load_model(path + "_current.pt");
        (*best_model)->load(path + "_best.pt");
        
        // 加载训练状态
        std::ifstream state_file(path + "_state.txt");
        std::string line;
        while (std::getline(state_file, line)) {
            if (line.find("iteration=") == 0) {
                current_iteration.store(std::stoi(line.substr(10)));
            } else if (line.find("total_games=") == 0) {
                total_games.store(std::stoi(line.substr(11)));
            } else if (line.find("total_steps=") == 0) {
                total_steps.store(std::stoi(line.substr(12)));
            } else if (line.find("best_elo=") == 0) {
                best_elo = std::stof(line.substr(9));
            } else if (line.find("iterations_without_improvement=") == 0) {
                iterations_without_improvement = std::stoi(line.substr(30));
            }
        }
        state_file.close();
        
        std::cout << "检查点加载完成" << std::endl;
        std::cout << "  迭代: " << current_iteration.load() << std::endl;
        std::cout << "  最佳ELO: " << best_elo << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[错误] 加载检查点失败: " << e.what() << std::endl;
        throw;
    }
}

std::string UnsupervisedSelfReinforcementTrainer::get_latest_checkpoint_path() const {
    std::ostringstream oss;
    oss << config.checkpoint_dir << "/" << config.model_name 
        << "_iter_" << std::setw(6) << std::setfill('0') << current_iteration.load();
    return oss.str();
}

void UnsupervisedSelfReinforcementTrainer::log_iteration_start(int iteration) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "迭代 " << iteration << "/" << config.total_iterations << std::endl;
    std::cout << "时间: " << timestamp_string() << std::endl;
    std::cout << "========================================\n" << std::endl;
}

void UnsupervisedSelfReinforcementTrainer::log_iteration_complete(const TrainingMetrics& metrics) {
    std::cout << "[" << std::setw(6) << std::setfill('0') << metrics.iteration << "] ";
    std::cout << "游戏: " << std::setw(6) << metrics.total_games_generated << " | ";
    std::cout << "池: " << replay_buffer->size() << "/" << replay_buffer->capacity() << " | ";
    std::cout << "步骤: " << std::setw(8) << metrics.total_training_steps << " | ";
    
    if (metrics.avg_total_loss > 0) {
        std::cout << "损失: " << std::fixed << std::setprecision(4) 
                  << metrics.avg_total_loss << " | ";
        std::cout << "策略: " << metrics.avg_policy_loss << " | ";
        std::cout << "价值: " << metrics.avg_value_loss << " | ";
    }
    
    // 输出验证集损失，与训练损失对比以监控过拟合
    if (metrics.has_validation) {
        std::cout << "验证: " << std::fixed << std::setprecision(4) 
                  << metrics.val_total_loss << " | ";
        // 计算训练-验证损失差值，差值过大表示过拟合
        float loss_gap = metrics.val_total_loss - metrics.avg_total_loss;
        std::cout << "差值: " << std::showpos << loss_gap << std::noshowpos << " | ";
    }
    
    if (metrics.current_elo > 0) {
        std::cout << "ELO: " << std::fixed << std::setprecision(1) 
                  << metrics.current_elo;
        if (metrics.elo_change != 0) {
            std::cout << "(" << (metrics.elo_change >= 0 ? "+" : "") 
                      << metrics.elo_change << ")";
        }
    }
    
    std::cout << " | 耗时: " << std::fixed << std::setprecision(1) 
              << metrics.total_time_sec << "s" << std::endl;
}

void UnsupervisedSelfReinforcementTrainer::log_training_progress(const TrainingMetrics& metrics) {
    std::cout << "\n---------- 详细训练报告 ----------" << std::endl;
    std::cout << "迭代: " << metrics.iteration << std::endl;
    std::cout << "累计对局: " << metrics.total_games_generated << std::endl;
    std::cout << "累计训练步骤: " << metrics.total_training_steps << std::endl;
    std::cout << std::endl;
    
    std::cout << "损失值:" << std::endl;
    std::cout << "  策略损失: " << std::fixed << std::setprecision(6) 
              << metrics.avg_policy_loss << std::endl;
    std::cout << "  价值损失: " << metrics.avg_value_loss << std::endl;
    std::cout << "  门控损失: " << metrics.avg_gating_loss << std::endl;
    std::cout << "  总损失: " << metrics.avg_total_loss << std::endl;
    std::cout << std::endl;
    
    // 验证集损失报告（与训练损失对比，检测过拟合）
    if (metrics.has_validation) {
        std::cout << "验证集损失:" << std::endl;
        std::cout << "  策略损失: " << std::fixed << std::setprecision(6) 
                  << metrics.val_policy_loss << std::endl;
        std::cout << "  价值损失: " << metrics.val_value_loss << std::endl;
        std::cout << "  总损失: " << metrics.val_total_loss << std::endl;
        // 训练-验证损失差值：正值表示验证集损失高于训练集，差值越大过拟合越严重
        float loss_gap = metrics.val_total_loss - metrics.avg_total_loss;
        std::cout << "  训练-验证差值: " << std::showpos << loss_gap << std::noshowpos;
        if (loss_gap > 0.1f) {
            std::cout << " ⚠ 过拟合风险";
        } else if (loss_gap > 0.05f) {
            std::cout << " (轻微)";
        } else {
            std::cout << " (正常)";
        }
        std::cout << std::endl << std::endl;
    }
    
    std::cout << "性能指标:" << std::endl;
    std::cout << "  当前ELO: " << std::fixed << std::setprecision(1) 
              << metrics.current_elo << std::endl;
    std::cout << "  游戏速度: " << std::fixed << std::setprecision(2) 
              << metrics.games_per_second << " 游戏/秒" << std::endl;
    std::cout << std::endl;
    
    std::cout << "时间分配:" << std::endl;
    std::cout << "  自我对弈: " << std::fixed << std::setprecision(2) 
              << metrics.self_play_time_sec << "s (" 
              << (metrics.self_play_time_sec / metrics.total_time_sec * 100) << "%)" << std::endl;
    std::cout << "  神经网络训练: " << metrics.training_time_sec << "s (" 
              << (metrics.training_time_sec / metrics.total_time_sec * 100) << "%)" << std::endl;
    std::cout << "  模型评估: " << metrics.evaluation_time_sec << "s (" 
              << (metrics.evaluation_time_sec / metrics.total_time_sec * 100) << "%)" << std::endl;
    std::cout << "--------------------------------" << std::endl;
}

void UnsupervisedSelfReinforcementTrainer::generate_training_report(
    const std::string& output_path) {
    
    std::ofstream report(output_path);
    
    report << "SpaceGo 维空弈 - 无监督自强化训练报告\n";
    report << "==========================================\n\n";
    
    report << "训练配置:\n";
    report << "  总迭代次数: " << config.total_iterations << "\n";
    report << "  实际完成迭代: " << current_iteration.load() << "\n";
    report << "  自我对弈Workers: " << config.self_play.num_workers << "\n";
    report << "  每步模拟次数: " << config.self_play.simulations_per_move << "\n";
    report << "  训练批次大小: " << config.training.batch_size << "\n";
    report << "  经验窗口大小: " << config.training.window_size << "\n";
    report << "  混合精度训练: " << (config.training.use_mixed_precision ? "是" : "否") << "\n";
    report << "  防过拟合措施:\n";
    report << "    Label Smoothing: " << config.training.label_smoothing << "\n";
    report << "    Dropout比率: " << config.training.dropout_rate << "\n";
    report << "    数据增强: " << (config.training.use_data_augmentation ? "是(8-fold)" : "否") << "\n";
    report << "    Weight Decay: " << config.training.weight_decay << "\n";
    report << "    验证集比例: " << config.training.validation_ratio << "\n";
    report << "    验证集早停耐心: " << config.validation_early_stop_patience << "\n";
    report << "\n";
    
    report << "训练结果:\n";
    report << "  总对局数: " << total_games.load() << "\n";
    report << "  总训练步骤: " << total_steps.load() << "\n";
    report << "  最终ELO: " << latest_metrics.current_elo << "\n";
    report << "  最佳ELO: " << best_elo << "\n";
    report << "\n";
    
    if (!training_history.empty()) {
        auto& first = training_history.front();
        auto& last = training_history.back();
        
        report << "改进情况:\n";
        report << "  初始损失: " << first.avg_total_loss << "\n";
        report << "  最终损失: " << last.avg_total_loss << "\n";
        report << "  损失降低: " << ((first.avg_total_loss - last.avg_total_loss) / 
                                       first.avg_total_loss * 100) << "%\n";
        report << "  ELO提升: " << (last.current_elo - first.current_elo) << "\n";
        
        // 验证集损失对比（过拟合检测）
        if (last.has_validation) {
            report << "\n验证集损失:\n";
            report << "  最终验证损失: " << last.val_total_loss << "\n";
            report << "  最终训练损失: " << last.avg_total_loss << "\n";
            report << "  训练-验证差值: " << (last.val_total_loss - last.avg_total_loss) << "\n";
            report << "  最佳验证损失: " << best_val_loss << "\n";
            report << "  过拟合判定: ";
            if (last.val_total_loss - last.avg_total_loss > 0.1f) {
                report << "存在过拟合风险\n";
            } else {
                report << "正常\n";
            }
        }
        report << "\n";
    }
    
    report << "最佳模型路径: " << config.checkpoint_dir << "/" 
           << config.model_name << "_best.pt\n";
    
    report.close();
    
    std::cout << "训练报告已生成: " << output_path << std::endl;
}

void UnsupervisedSelfReinforcementTrainer::ensure_directory_exists(const std::string& path) {
    std::filesystem::create_directories(path);
}

std::string UnsupervisedSelfReinforcementTrainer::timestamp_string() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

} // namespace training
} // namespace go
} // namespace spacego

#endif  // USE_LIBTORCH
