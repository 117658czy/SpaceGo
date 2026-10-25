#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <atomic>
#include <thread>

#ifdef USE_LIBTORCH
#include <ATen/cuda/CUDAContext.h>
#endif

#include "framework/logger.h"
#include "go/neural/ca_moe_expert.h"
#include "go/training/supervised_pretrain.h"
#include "go/training/unsupervised_self_reinforcement.h"
#include "go/training/diploid_trainer.h"
#include "go/training/elo_gated_trainer.h"

using namespace spacego;
using namespace spacego::go;
using namespace spacego::go::neural;
using namespace spacego::go::training;

// 打印使用说明
void print_usage(const char* program_name) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "SpaceGo 维空弈 - AI训练系统" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    std::cout << "用法: " << program_name << " [选项]\n" << std::endl;
    
    std::cout << "训练模式:" << std::endl;
    std::cout << "  --mode=supervised      监督预训练模式（默认）" << std::endl;
    std::cout << "  --mode=unsupervised    无监督自强化训练模式" << std::endl;
    std::cout << "  --mode=diploid         二倍体分化竞争训练模式\n" << std::endl;
    
    std::cout << "通用选项:" << std::endl;
    std::cout << "  --help, -h             显示此帮助信息" << std::endl;
    std::cout << "  --config=<path>        配置文件路径" << std::endl;
    std::cout << "  --checkpoint=<path>    加载检查点继续训练" << std::endl;
    std::cout << "  --iterations=N         训练迭代次数（仅无监督模式）" << std::endl;
    std::cout << "  --workers=N            并行Worker数量（仅无监督模式）" << std::endl;
    std::cout << "  --games=N              每迭代对局数（仅无监督模式，默认100）" << std::endl;
    std::cout << "  --batch-size=N         训练批次大小（仅无监督模式，默认512）" << std::endl;
    std::cout << "  --gpu=N                指定GPU设备ID" << std::endl;
    std::cout << "  --no-cuda              禁用CUDA加速" << std::endl;
    std::cout << "  --no-mixed-precision   禁用混合精度训练" << std::endl;
    std::cout << "  --output-dir=<path>    输出目录" << std::endl;
    std::cout << "  --log-level=<level>    日志级别 (debug/info/warning/error)\n" << std::endl;
    
    std::cout << "示例:" << std::endl;
    std::cout << "  # 监督预训练（使用模拟数据）" << std::endl;
    std::cout << "  " << program_name << " --mode=supervised\n" << std::endl;
    std::cout << "  # 无监督自强化训练（使用4个GPU Worker）" << std::endl;
    std::cout << "  " << program_name << " --mode=unsupervised --workers=4 --iterations=1000\n" << std::endl;
    std::cout << "  # 从检查点恢复训练" << std::endl;
    std::cout << "  " << program_name << " --mode=unsupervised --checkpoint=./checkpoints/spacego_iter_001000\n" << std::endl;
}

// 打印训练进度信息
// stats: 训练统计信息
// is_validation: 是否为验证阶段
void print_progress(const SupervisedPretrainStats& stats, bool is_validation) {
    const char* phase = is_validation ? "验证" : "训练";
    std::cout << "[" << phase << "] Epoch " << (stats.epoch + 1)
              << " | Step " << stats.step
              << " | 策略损失: " << std::fixed << std::setprecision(6) << stats.policy_loss
              << " | 价值损失: " << stats.value_loss
              << " | 总损失: " << stats.total_loss
              << std::endl;
}

// 解析命令行参数
struct CommandLineArgs {
    std::string mode = "supervised";
    std::string config_path;
    std::string checkpoint_path;
    std::string output_dir = "./training_output";
    int iterations = 10000;
    int workers = 4;
    int games_per_iter = 0;  // 0表示使用默认值(100)
    int batch_size = 0;      // 0表示使用默认值(512)
    int gpu_device_id = 0;
    bool use_cuda = true;
    bool use_mixed_precision = true;
    std::string log_level = "info";
    bool show_help = false;
    // 二倍体训练参数
    int diploid_generations = 50;
    int diploid_simulations = 300;        // HMGS-DP每步模拟次数（合并推理后速度×2，300兼顾数据质量与训练吞吐）
    int diploid_self_play_games = 30;     // 每代自对弈局数
    int diploid_competition_games = 80;   // 竞争对弈局数（40局胜率误差约±50 Elo，提高到80降至±35 Elo）
    int diploid_epochs = 20;
    float diploid_lr = 1e-3f;
    // ELO门控训练参数（回滚模式：快速迭代，5~10分钟一代）
    int elo_rounds = 200;                  // 最大轮数
    int elo_simulations = 150;             // 每步模拟数（原300减半提速）
    int elo_games_per_round = 6;           // 每轮对弈学习局数（一局学一次，快速试错）
    int elo_eval_games = 24;               // 每轮评估ELO局数（≈2~3分钟，保证单代≤10分钟）
    float elo_temperature = 1.0f;          // 训练对弈温度（实测1.5探索致对弈全负→数据全乱局，
                                          //   分布漂移学坏回滚；1.0为标准AlphaZero探索）
    float elo_lr = 0.03f;                  // 高学习率
    int elo_hidden = 256;                  // 回滚模式M通道宽度（小模型可--elo-hidden=128减参）
    int elo_gen0_hidden = 256;             // 回滚模式Gen0基线通道宽度（默认256与大模型兼容；
                                           //   配M=128/基线=256形成"小模型挑战强基线"）
    std::string elo_start_checkpoint;      // 起点模型（Gen1峰值等）
    std::string elo_gen0_checkpoint;       // Gen0基线
    float elo_cumulative_start = 0.0f;     // 累积ELO起始值（扩容/换基线时延续上次累积水平）
    bool elo_fresh_start = false;          // 彻底从头：随机初始化M，跳过起点模型与自举基线
    // KataGo评估标尺（用户指定：用KataGo当评估器看进步，不要求赢它）
    bool elo_use_katago_eval = true;       // true=每轮1局vs KataGo看目差进步；false=旧自对弈评估
    std::string elo_katago_exe = "D:/KataGo/katago16/katago.exe";
    std::string elo_katago_model = "D:/KataGo/kata1-b18c384nbt.bin.gz";
    std::string elo_katago_config = "D:/KataGo/gtp.cfg";
    int elo_katago_visits = 400;           // KataGo基准强度（maxVisits，越大越强；模型弱先用低值）
    int elo_katago_selfplay_games = 4;     // 每轮教师自我对弈局数（蒸馏数据源，每局约300样本）
    int elo_katago_teacher_visits = 10;    // 教师自对弈强度（棋谱质量 vs 生成速度）
    int elo_katago_distill_epochs = 8;     // 教师棋谱每局训练遍数（数据复用，放大学习信号）
    // 按模式拆分单代训练时间：长时间(diploid)与回滚(elo-gated)各自独立的模型宽度/ELO门槛
    int diploid_hidden = 256;              // 长时间模式模型通道宽度
    float diploid_elo_min = 30.0f;         // 长时间模式单代接受所需最小ELO涨幅（大幅上涨才接受）
};

CommandLineArgs parse_args(int argc, char* argv[]) {
    CommandLineArgs args;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            args.show_help = true;
        } else if (arg.find("--mode=") == 0) {
            args.mode = arg.substr(7);
        } else if (arg.find("--config=") == 0) {
            args.config_path = arg.substr(9);
        } else if (arg.find("--checkpoint=") == 0) {
            args.checkpoint_path = arg.substr(13);
        } else if (arg.find("--iterations=") == 0) {
            args.iterations = std::stoi(arg.substr(13));
        } else if (arg.find("--workers=") == 0) {
            args.workers = std::stoi(arg.substr(10));
        } else if (arg.find("--games=") == 0) {
            args.games_per_iter = std::stoi(arg.substr(8));
        } else if (arg.find("--batch-size=") == 0) {
            args.batch_size = std::stoi(arg.substr(13));
        } else if (arg.find("--gpu=") == 0) {
            args.gpu_device_id = std::stoi(arg.substr(6));
        } else if (arg == "--no-cuda") {
            args.use_cuda = false;
        } else if (arg == "--no-mixed-precision") {
            args.use_mixed_precision = false;
        } else if (arg.find("--output-dir=") == 0) {
            args.output_dir = arg.substr(13);  // 跳过"--output-dir="共13字符
        } else if (arg.find("--log-level=") == 0) {
            args.log_level = arg.substr(12);   // 跳过"--log-level="共12字符
        } else if (arg.find("--diploid-generations=") == 0) {
            args.diploid_generations = std::stoi(arg.substr(22));
        } else if (arg.find("--diploid-simulations=") == 0) {
            args.diploid_simulations = std::stoi(arg.substr(22));
        } else if (arg.find("--diploid-games=") == 0) {
            args.diploid_self_play_games = std::stoi(arg.substr(16));
        } else if (arg.find("--diploid-compete=") == 0) {
            args.diploid_competition_games = std::stoi(arg.substr(18));
        } else if (arg.find("--diploid-epochs=") == 0) {
            args.diploid_epochs = std::stoi(arg.substr(17));
        } else if (arg.find("--diploid-lr=") == 0) {
            args.diploid_lr = std::stof(arg.substr(13));
        } else if (arg.find("--elo-rounds=") == 0) {
            args.elo_rounds = std::stoi(arg.substr(13));
        } else if (arg.find("--elo-games=") == 0) {
            args.elo_games_per_round = std::stoi(arg.substr(12));
        } else if (arg.find("--elo-eval-games=") == 0) {
            args.elo_eval_games = std::stoi(arg.substr(17));
        } else if (arg.find("--elo-temperature=") == 0) {
            args.elo_temperature = std::stof(arg.substr(18));  // 前缀"--elo-temperature="长18字符（含=），substr(18)跳过前缀取数值
        } else if (arg.find("--elo-lr=") == 0) {
            args.elo_lr = std::stof(arg.substr(9));
        } else if (arg.find("--elo-start=") == 0) {
            args.elo_start_checkpoint = arg.substr(12);
        } else if (arg.find("--elo-gen0=") == 0) {
            args.elo_gen0_checkpoint = arg.substr(11);
        } else if (arg.find("--elo-simulations=") == 0) {
            args.elo_simulations = std::stoi(arg.substr(18));
        } else if (arg.find("--elo-hidden=") == 0) {
            args.elo_hidden = std::stoi(arg.substr(13));
        } else if (arg.find("--elo-gen0-hidden=") == 0) {
            args.elo_gen0_hidden = std::stoi(arg.substr(18));
        } else if (arg.find("--elo-cumulative-start=") == 0) {
            // 前缀长度23：--elo-cumulative-start=共23字符，substr取等号后数值
            args.elo_cumulative_start = std::stof(arg.substr(23));
        } else if (arg == "--no-katago-eval") {
            // 回退旧自对弈评估（不用KataGo作标尺）
            args.elo_use_katago_eval = false;
        } else if (arg == "--elo-fresh") {
            // 彻底从头：随机初始化M，不加载起点模型/自举基线（起点模型塌缩时的唯一出路）
            args.elo_fresh_start = true;
        } else if (arg.find("--elo-katago-exe=") == 0) {
            args.elo_katago_exe = arg.substr(17);
        } else if (arg.find("--elo-katago-model=") == 0) {
            args.elo_katago_model = arg.substr(19);
        } else if (arg.find("--elo-katago-config=") == 0) {
            args.elo_katago_config = arg.substr(20);
        } else if (arg.find("--elo-katago-visits=") == 0) {
            args.elo_katago_visits = std::stoi(arg.substr(20));
        } else if (arg.find("--elo-katago-selfplay-games=") == 0) {
            args.elo_katago_selfplay_games = std::stoi(arg.substr(28));
        } else if (arg.find("--elo-katago-teacher-visits=") == 0) {
            args.elo_katago_teacher_visits = std::stoi(arg.substr(28));
        } else if (arg.find("--elo-katago-distill-epochs=") == 0) {
            args.elo_katago_distill_epochs = std::stoi(arg.substr(28));
        } else if (arg.find("--diploid-hidden=") == 0) {
            args.diploid_hidden = std::stoi(arg.substr(17));
        } else if (arg.find("--diploid-elo-min=") == 0) {
            args.diploid_elo_min = std::stof(arg.substr(18));
        }
    }
    
    return args;
}

// 从配置文件加载训练配置
UnsupervisedTrainingConfig load_config_from_file(const std::string& path) {
    UnsupervisedTrainingConfig config;
    
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[警告] 无法打开配置文件: " << path << "，使用默认配置" << std::endl;
        return config;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // 跳过注释和空行
        if (line.empty() || line[0] == '#') continue;
        
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;
        
        std::string key = line.substr(0, eq_pos);
        std::string value = line.substr(eq_pos + 1);
        
        // 去除首尾空白
        auto trim = [](std::string& s) {
            s.erase(0, s.find_first_not_of(" \t"));
            s.erase(s.find_last_not_of(" \t") + 1);
        };
        trim(key);
        trim(value);
        
        // 解析配置项
        if (key == "total_iterations")
            config.total_iterations = std::stoi(value);
        else if (key == "num_workers")
            config.self_play.num_workers = std::stoi(value);
        else if (key == "simulations_per_move")
            config.self_play.simulations_per_move = std::stoi(value);
        else if (key == "batch_size")
            config.training.batch_size = std::stoi(value);
        else if (key == "learning_rate")
            config.training.learning_rate = std::stof(value);
        else if (key == "window_size")
            config.training.window_size = std::stoi(value);
        else if (key == "progressive_expansion")
            config.training.progressive_expansion = (value == "true" || value == "1");
        else if (key == "initial_window_size")
            config.training.initial_window_size = std::stoi(value);
        else if (key == "max_window_size")
            config.training.max_window_size = std::stoi(value);
        else if (key == "expansion_factor")
            config.training.expansion_factor = std::stof(value);
        else if (key == "expansion_interval")
            config.training.expansion_interval = std::stoi(value);
        else if (key == "checkpoint_interval")
            config.checkpoint_interval = std::stoi(value);
        else if (key == "evaluation_interval")
            config.evaluation.evaluation_interval = std::stoi(value);
        else if (key == "num_evaluation_games")
            config.evaluation.num_evaluation_games = std::stoi(value);
        else if (key == "temperature")
            config.self_play.temperature = std::stof(value);
        else if (key == "temperature_decay")
            config.self_play.temperature_decay = std::stof(value);
        else if (key == "min_temperature")
            config.self_play.min_temperature = std::stof(value);
        else if (key == "dirichlet_alpha")
            config.self_play.dirichlet_alpha = std::stof(value);
        else if (key == "dirichlet_weight")
            config.self_play.dirichlet_weight = std::stof(value);
        else if (key == "use_cuda")
            config.self_play.use_cuda = (value == "true" || value == "1");
        else if (key == "use_mixed_precision")
            config.training.use_mixed_precision = (value == "true" || value == "1");
        else if (key == "checkpoint_dir")
            config.checkpoint_dir = value;
        else if (key == "log_dir")
            config.log_dir = value;
        else if (key == "model_name")
            config.model_name = value;
        else if (key == "seed")
            config.seed = std::stoi(value);
    }
    
    file.close();
    return config;
}

// 运行监督预训练模式
int run_supervised_training() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "启动监督预训练模式" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // 初始化日志系统
    auto& logger = framework::Logger::instance();
    logger.initialize("SpaceGo_Trainer_Supervised");
    logger.set_min_level(framework::LogLevel::INFO);
    
    try {
        // 检查CUDA是否可用
        if (torch::cuda::is_available()) {
            std::cout << "[信息] CUDA可用，将使用GPU加速" << std::endl;
            std::cout << "[信息] CUDA设备数量: " << torch::cuda::device_count() << std::endl;
            for (int i = 0; i < torch::cuda::device_count(); ++i) {
                torch::Device device(torch::kCUDA, i);
                auto props = at::cuda::getDeviceProperties(i);
                std::cout << "  GPU-" << i << ": " << props->name
                          << " (显存: " << (props->totalGlobalMem / 1024 / 1024 / 1024)
                          << "GB)" << std::endl;
            }
        } else {
            std::cout << "[警告] CUDA不可用，将使用CPU进行训练" << std::endl;
        }
        std::cout << std::endl;
        
        // 1. 创建CA-MOE系统
        std::cout << "[1/5] 创建CA-MOE混合专家系统..." << std::endl;
        auto ca_moe_system = std::make_shared<CA_MOEExpertSystemImpl>(8, 19);
        ca_moe_system->set_train_mode();
        
        if (torch::cuda::is_available()) {
            ca_moe_system->to(torch::kCUDA);
            std::cout << "      CA-MOE系统已加载到GPU" << std::endl;
        }
        std::cout << "      完成！" << std::endl;
        std::cout << std::endl;
        
        // 2. 创建模拟数据
        std::cout << "[2/5] 生成模拟训练数据..." << std::endl;
        MockDataGenerator generator;
        auto all_samples = generator.generate_samples(2000);
        
        size_t val_size = static_cast<size_t>(all_samples.size() * 0.2);
        std::vector<SupervisedSample> training_data(all_samples.begin() + val_size, 
                                                     all_samples.end());
        std::vector<SupervisedSample> validation_data(all_samples.begin(), 
                                                       all_samples.begin() + val_size);
        
        std::cout << "      训练样本: " << training_data.size() << std::endl;
        std::cout << "      验证样本: " << validation_data.size() << std::endl;
        std::cout << "      完成！" << std::endl;
        std::cout << std::endl;
        
        // 3. 配置训练参数
        std::cout << "[3/5] 配置训练参数..." << std::endl;
        float learning_rate = 1e-3f;
        int batch_size = 32;
        int num_epochs = 10;
        float weight_decay = 1e-4f;
        
        std::cout << "      学习率: " << learning_rate << std::endl;
        std::cout << "      批次大小: " << batch_size << std::endl;
        std::cout << "      训练轮数: " << num_epochs << std::endl;
        std::cout << "      完成！" << std::endl;
        std::cout << std::endl;
        
        // 4. 创建优化器
        std::cout << "[4/5] 创建优化器..." << std::endl;
        // CA_MOEExpertSystemImpl继承自torch::nn::Module，直接调用parameters()获取参数
        std::vector<torch::Tensor> params;
        for (const auto& p : ca_moe_system->parameters()) {
            params.push_back(p);
        }
        auto optimizer = torch::optim::Adam(params, 
            torch::optim::AdamOptions(learning_rate).weight_decay(weight_decay));
        std::cout << "      完成！" << std::endl;
        std::cout << std::endl;
        
        // 5. 开始训练
        std::cout << "[5/5] 开始训练..." << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        int current_step = 0;
        
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            std::cout << "--- Epoch " << (epoch + 1) << "/" << num_epochs << " ---" << std::endl;
            
            // 训练阶段
            ca_moe_system->set_train_mode();
            float total_train_policy_loss = 0.0f;
            float total_train_value_loss = 0.0f;
            int num_train_batches = 0;
            
            for (size_t i = 0; i < training_data.size(); i += batch_size) {
                optimizer.zero_grad();
                
                std::vector<torch::Tensor> states;
                std::vector<torch::Tensor> target_policies;
                std::vector<float> target_values;
                
                for (size_t j = i; j < i + batch_size && j < training_data.size(); ++j) {
                    auto& sample = training_data[j];
                    states.push_back(sample.state.unsqueeze(0));
                    
                    std::vector<float> policy_floats(sample.target_policy.begin(), 
                                                     sample.target_policy.end());
                    target_policies.push_back(torch::tensor(policy_floats).unsqueeze(0));
                    target_values.push_back(sample.target_value);
                }
                
                auto batch_states = torch::cat(states, 0);
                auto batch_target_policies = torch::cat(target_policies, 0);
                auto batch_target_values = torch::tensor(target_values);
                
                if (torch::cuda::is_available()) {
                    batch_states = batch_states.to(torch::kCUDA);
                    batch_target_policies = batch_target_policies.to(torch::kCUDA);
                    batch_target_values = batch_target_values.to(torch::kCUDA);
                }
                
                auto result = ca_moe_system->forward(batch_states);
                
                auto policy_loss = -(batch_target_policies * result.final_policy.log()).sum(1).mean();
                auto value_loss = torch::mse_loss(result.final_value.squeeze(1), batch_target_values);
                auto total_loss = policy_loss + value_loss;
                
                total_loss.backward();
                optimizer.step();
                
                total_train_policy_loss += policy_loss.item<float>();
                total_train_value_loss += value_loss.item<float>();
                num_train_batches++;
                current_step++;
            }
            
            // 打印训练统计
            if (num_train_batches > 0) {
                SupervisedPretrainStats train_stats;
                train_stats.epoch = epoch;
                train_stats.step = current_step;
                train_stats.policy_loss = total_train_policy_loss / num_train_batches;
                train_stats.value_loss = total_train_value_loss / num_train_batches;
                train_stats.total_loss = train_stats.policy_loss + train_stats.value_loss;
                print_progress(train_stats, false);
            }
            
            // 验证阶段
            ca_moe_system->set_eval_mode();
            float total_val_policy_loss = 0.0f;
            float total_val_value_loss = 0.0f;
            int num_val_batches = 0;
            
            {
                torch::NoGradGuard no_grad;
                
                for (size_t i = 0; i < validation_data.size(); i += batch_size) {
                    std::vector<torch::Tensor> states;
                    std::vector<torch::Tensor> target_policies;
                    std::vector<float> target_values;
                    
                    for (size_t j = i; j < i + batch_size && j < validation_data.size(); ++j) {
                        auto& sample = validation_data[j];
                        states.push_back(sample.state.unsqueeze(0));
                        
                        std::vector<float> policy_floats(sample.target_policy.begin(), 
                                                         sample.target_policy.end());
                        target_policies.push_back(torch::tensor(policy_floats).unsqueeze(0));
                        target_values.push_back(sample.target_value);
                    }
                    
                    auto batch_states = torch::cat(states, 0);
                    auto batch_target_policies = torch::cat(target_policies, 0);
                    auto batch_target_values = torch::tensor(target_values);
                    
                    if (torch::cuda::is_available()) {
                        batch_states = batch_states.to(torch::kCUDA);
                        batch_target_policies = batch_target_policies.to(torch::kCUDA);
                        batch_target_values = batch_target_values.to(torch::kCUDA);
                    }
                    
                    auto result = ca_moe_system->forward(batch_states);
                    
                    auto policy_loss = -(batch_target_policies * result.final_policy.log()).sum(1).mean();
                    auto value_loss = torch::mse_loss(result.final_value.squeeze(1), batch_target_values);
                    
                    total_val_policy_loss += policy_loss.item<float>();
                    total_val_value_loss += value_loss.item<float>();
                    num_val_batches++;
                }
            }
            
            if (num_val_batches > 0) {
                SupervisedPretrainStats val_stats;
                val_stats.epoch = epoch;
                val_stats.step = current_step;
                val_stats.policy_loss = total_val_policy_loss / num_val_batches;
                val_stats.value_loss = total_val_value_loss / num_val_batches;
                val_stats.total_loss = val_stats.policy_loss + val_stats.value_loss;
                print_progress(val_stats, true);
            }
            
            std::cout << std::endl;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        
        std::cout << "========================================" << std::endl;
        std::cout << "监督预训练完成！" << std::endl;
        std::cout << "总耗时: " << duration.count() << " 秒" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << std::endl;
        
        // 保存模型
        std::cout << "保存训练好的模型..." << std::endl;
        ca_moe_system->set_eval_mode();
        if (torch::cuda::is_available()) {
            ca_moe_system->to(torch::kCPU);
        }
        std::string model_path = "spacego_ca_moe_model_supervised.pt";
        ca_moe_system->save(model_path);
        std::cout << "模型已保存到: " << model_path << std::endl;
        std::cout << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        logger.fatal("训练过程中发生错误: " + std::string(e.what()));
        logger.shutdown();
        return 1;
    }
    
    std::cout << "监督预训练程序正常退出！" << std::endl;
    logger.shutdown();
    return 0;
}

// 运行无监督自强化训练模式
int run_unsupervised_training(const CommandLineArgs& args) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "启动无监督自强化训练模式" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // 初始化日志系统
    auto& logger = framework::Logger::instance();
    logger.initialize("SpaceGo_Trainer_Unsupervised");
    
    framework::LogLevel log_level = framework::LogLevel::INFO;
    if (args.log_level == "debug") log_level = framework::LogLevel::DEBUG;
    else if (args.log_level == "warning") log_level = framework::LogLevel::WARNING;
    else if (args.log_level == "error") log_level = framework::LogLevel::LVL_ERROR;
    logger.set_min_level(log_level);
    
    try {
        // 加载或创建配置
        UnsupervisedTrainingConfig config;
        
        if (!args.config_path.empty()) {
            std::cout << "从配置文件加载设置: " << args.config_path << std::endl;
            config = load_config_from_file(args.config_path);
        }
        
        // 命令行参数覆盖配置文件
        config.total_iterations = args.iterations;
        config.self_play.num_workers = args.workers;
        if (args.games_per_iter > 0) {
            config.training.games_per_iteration = args.games_per_iter;
        }
        if (args.batch_size > 0) {
            config.training.batch_size = args.batch_size;
        }
        config.self_play.cuda_device_id = args.gpu_device_id;
        config.self_play.use_cuda = args.use_cuda && torch::cuda::is_available();
        config.training.use_mixed_precision = args.use_mixed_precision;
        config.checkpoint_dir = args.output_dir + "/checkpoints";
        config.log_dir = args.output_dir + "/logs";
        
        // 打印环境信息
        std::cout << "\n--- 环境信息 ---" << std::endl;
        if (torch::cuda::is_available() && args.use_cuda) {
            std::cout << "CUDA状态: 可用" << std::endl;
            std::cout << "GPU数量: " << torch::cuda::device_count() << std::endl;
            std::cout << "使用GPU: GPU-" << args.gpu_device_id << std::endl;
            
            auto device_props = at::cuda::getDeviceProperties(args.gpu_device_id);
            std::cout << "GPU型号: " << device_props->name << std::endl;
            std::cout << "显存容量: " << (device_props->totalGlobalMem / 1024 / 1024 / 1024)
                      << " GB" << std::endl;
            std::cout << "计算能力: " << device_props->major << "." 
                      << device_props->minor << std::endl;
            std::cout << "混合精度: " << (config.training.use_mixed_precision ? "启用" : "禁用") 
                      << std::endl;
        } else {
            std::cout << "CUDA状态: 不可用或被禁用，将使用CPU" << std::endl;
            config.self_play.use_cuda = false;
            config.training.use_mixed_precision = false;
        }
        
        std::cout << "\n--- 训练配置 ---" << std::endl;
        std::cout << "总迭代次数: " << config.total_iterations << std::endl;
        std::cout << "并行Workers: " << config.self_play.num_workers << std::endl;
        std::cout << "每步模拟数: " << config.self_play.simulations_per_move << std::endl;
        std::cout << "训练批次大小: " << config.training.batch_size << std::endl;
        std::cout << "经验窗口大小: " << config.training.window_size << std::endl;
        if (config.training.progressive_expansion) {
            std::cout << "渐进式扩张: 启用" << std::endl;
            std::cout << "  初始容量: " << config.training.initial_window_size << std::endl;
            std::cout << "  最大容量: " << config.training.max_window_size << std::endl;
            std::cout << "  扩张倍数: " << config.training.expansion_factor << std::endl;
            std::cout << "  扩张间隔: 每 " << config.training.expansion_interval << " 次迭代" << std::endl;
        } else {
            std::cout << "渐进式扩张: 禁用（固定大小）" << std::endl;
        }
        std::cout << "学习率: " << config.training.learning_rate << std::endl;
        std::cout << "初始温度: " << config.self_play.temperature << std::endl;
        std::cout << "温度衰减率: " << config.self_play.temperature_decay << std::endl;
        std::cout << "评估间隔: 每 " << config.evaluation.evaluation_interval 
                  << " 次迭代" << std::endl;
        std::cout << "评估对局数: " << config.evaluation.num_evaluation_games << std::endl;
        std::cout << "检查点间隔: 每 " << config.checkpoint_interval << " 次迭代" << std::endl;
        std::cout << "输出目录: " << args.output_dir << std::endl;
        std::cout << std::endl;
        
        // 创建训练器
        auto trainer = std::make_unique<UnsupervisedSelfReinforcementTrainer>(config);
        
        // 初始化所有组件
        if (!trainer->initialize()) {
            std::cerr << "[错误] 训练系统初始化失败" << std::endl;
            return 1;
        }
        
        // 如果指定了检查点，则加载
        if (!args.checkpoint_path.empty()) {
            std::cout << "\n从检查点恢复训练..." << std::endl;
            trainer->load_checkpoint(args.checkpoint_path);
        }
        
        // 设置信号处理以支持优雅停止
        std::atomic<bool> stop_requested{false};
        
        // 注册Ctrl+C处理（简化版）
        // 实际实现中应该使用平台特定的信号处理
        
        // 开始训练
        trainer->start_training();

        // 训练启动时间，用于计算总耗时和预估剩余时间
        auto training_start_time = std::chrono::steady_clock::now();
        int last_displayed_iteration = -1;

        // 主线程等待训练完成，每2秒刷新一次进度条
        while (trainer->is_training()) {
            std::this_thread::sleep_for(std::chrono::seconds(2));

            if (stop_requested.load()) {
                std::cout << "\n收到停止请求..." << std::endl;
                trainer->stop_training();
                break;
            }

            // 获取当前训练状态
            auto metrics = trainer->get_latest_metrics();
            int current_iter = trainer->get_current_iteration();
            int total_iters = config.total_iterations;
            auto replay_buf = trainer->get_replay_buffer();

            // 只在有新迭代时更新进度（避免重复显示）
            if (current_iter == last_displayed_iteration) {
                // 即使迭代未更新，也显示"正在生成中"的提示
                auto now = std::chrono::steady_clock::now();
                auto elapsed_total = std::chrono::duration<float>(now - training_start_time).count();
                int elapsed_min = static_cast<int>(elapsed_total) / 60;
                int elapsed_sec = static_cast<int>(elapsed_total) % 60;
                // 使用\r回车覆盖当前行
                std::cout << "\r  [等待中] 迭代 " << current_iter << "/" << total_iters
                          << " | 池 " << replay_buf->size() << "/" << replay_buf->capacity()
                          << " | 耗时 " << elapsed_min << "m" << elapsed_sec << "s"
                          << std::flush;
                continue;
            }
            last_displayed_iteration = current_iter;

            // 计算进度百分比和进度条
            float progress = static_cast<float>(current_iter) / std::max(1, total_iters);
            int bar_width = 20;
            int filled = static_cast<int>(progress * bar_width);
            std::string bar(bar_width, ' ');
            for (int i = 0; i < bar_width; ++i) {
                bar[i] = (i < filled) ? '#' : '-';
            }

            // 计算耗时和预估剩余时间
            auto now = std::chrono::steady_clock::now();
            auto elapsed_total = std::chrono::duration<float>(now - training_start_time).count();
            int elapsed_min = static_cast<int>(elapsed_total) / 60;
            int elapsed_sec = static_cast<int>(elapsed_total) % 60;

            // 预估剩余时间：基于平均每迭代耗时
            float eta_sec = 0.0f;
            if (current_iter > 0 && progress < 1.0f) {
                float avg_iter_time = elapsed_total / current_iter;
                int remaining_iters = total_iters - current_iter;
                eta_sec = avg_iter_time * remaining_iters;
            }
            int eta_min = static_cast<int>(eta_sec) / 60;
            int eta_s = static_cast<int>(eta_sec) % 60;

            // 使用\r回车覆盖当前行，实现原地刷新
            std::cout << "\r  [" << bar << "] " << std::fixed << std::setprecision(1)
                      << (progress * 100.0f) << "% "
                      << "| 迭代 " << current_iter << "/" << total_iters
                      << " | 池 " << replay_buf->size() << "/" << replay_buf->capacity()
                      << " | 游戏 " << metrics.total_games_generated
                      << " | 损失 " << std::setprecision(4) << metrics.avg_total_loss
                      << " | ELO " << std::setprecision(1) << metrics.current_elo
                      << " | " << elapsed_min << "m" << elapsed_sec << "s"
                      << " ETA " << eta_min << "m" << eta_s << "s"
                      << std::flush;
        }

        // 训练结束后换行，避免覆盖最终输出
        std::cout << std::endl;
        
        // 打印最终报告
        auto final_metrics = trainer->get_latest_metrics();
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "最终训练报告" << std::endl;
        std::cout << "========================================\n" << std::endl;
        std::cout << "总迭代次数: " << final_metrics.iteration << std::endl;
        std::cout << "总对局数: " << final_metrics.total_games_generated << std::endl;
        std::cout << "总训练步骤: " << final_metrics.total_training_steps << std::endl;
        std::cout << "最终ELO评分: " << final_metrics.current_elo << std::endl;
        
        if (final_metrics.avg_total_loss > 0) {
            std::cout << "\n损失值:" << std::endl;
            std::cout << "  策略损失: " << std::fixed << std::setprecision(6) 
                      << final_metrics.avg_policy_loss << std::endl;
            std::cout << "  价值损失: " << final_metrics.avg_value_loss << std::endl;
            std::cout << "  门控损失: " << final_metrics.avg_gating_loss << std::endl;
            std::cout << "  总损失: " << final_metrics.avg_total_loss << std::endl;
        }
        
        std::cout << "\n最佳模型保存在: " << config.checkpoint_dir << "/" 
                  << config.model_name << "_best.pt" << std::endl;
        std::cout << "训练日志保存在: " << config.log_dir << "/" << std::endl;
        std::cout << "\n训练成功完成！" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n[错误] 无监督训练过程中发生异常: " << e.what() << std::endl;
        logger.fatal("无监督训练失败: " + std::string(e.what()));
        logger.shutdown();
        return 1;
    }
    
    logger.shutdown();
    return 0;
}

// 生成默认配置文件
void generate_default_config(const std::string& path) {
    std::ofstream file(path);
    
    file << "# SpaceGo 维空弈 - 无监督自强化训练配置文件\n";
    file << "# ============================================\n\n";
    
    file << "# === 训练基本配置 ===\n";
    file << "total_iterations=10000\n";
    file << "checkpoint_interval=20\n";
    file << "log_interval=10\n";
    file << "model_name=spacego_unsupervised\n";
    file << "seed=42\n\n";
    
    file << "# === 自我对弈配置 ===\n";
    file << "num_workers=4\n";
    file << "simulations_per_move=800\n";
    file << "batch_size=32\n";
    file << "temperature=1.0\n";
    file << "temperature_decay=0.995\n";
    file << "min_temperature=0.0\n";
    file << "dirichlet_alpha=0.03\n";
    file << "dirichlet_weight=0.25\n";
    file << "max_game_length=500\n";
    file << "resign_threshold=-0.9\n\n";
    
    file << "# === CUDA和性能配置 ===\n";
    file << "use_cuda=true\n";
    file << "use_mixed_precision=true\n";
    file << "cuda_device_id=0\n\n";
    
    file << "# === 训练超参数 ===\n";
    file << "learning_rate=0.001\n";
    file << "lr_decay_rate=0.95\n";
    file << "weight_decay=0.0001\n";
    file << "policy_loss_weight=1.0\n";
    file << "value_loss_weight=1.0\n";
    file << "gating_loss_weight=0.5\n";
    file << "entropy_regularization=0.01\n";
    file << "gradient_clipping=true\n";
    file << "max_gradient_norm=5.0\n";
    file << "window_size=1000000\n";
    file << "progressive_expansion=true\n";
    file << "initial_window_size=50000\n";
    file << "max_window_size=500000\n";
    file << "expansion_factor=2.0\n";
    file << "expansion_interval=10\n";
    file << "optimizer_type=adam\n\n";
    
    file << "# === 评估配置 ===\n";
    file << "evaluation_interval=50\n";
    file << "num_evaluation_games=100\n";
    file << "exploration_factor=0.0\n\n";
    
    file << "# === 目录配置 ===\n";
    file << "checkpoint_dir=./checkpoints\n";
    file << "log_dir=./logs\n\n";
    
    file.close();
    
    std::cout << "默认配置文件已生成: " << path << std::endl;
}

// 运行二倍体分化竞争训练模式
int run_diploid_training(const CommandLineArgs& args) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "启动二倍体分化竞争训练模式" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // 初始化日志
    auto& logger = framework::Logger::instance();
    logger.initialize("SpaceGo_Trainer_Diploid");
    logger.set_min_level(framework::LogLevel::INFO);

    if (!torch::cuda::is_available() || !args.use_cuda) {
        std::cout << "[警告] CUDA不可用或已禁用，将使用CPU（较慢）" << std::endl;
    } else {
        std::cout << "[信息] CUDA可用，使用GPU加速" << std::endl;
    }

    // 配置二倍体训练参数
    DiploidConfig config;
    config.simulations_per_move = args.diploid_simulations;
    config.games_per_gen = args.diploid_self_play_games;
    config.competition_games = args.diploid_competition_games;
    config.max_generations = args.diploid_generations;
    config.training_epochs_per_gen = args.diploid_epochs;
    config.learning_rate = args.diploid_lr;
    config.output_dir = args.output_dir + "/diploid";
    config.hidden_dim = args.diploid_hidden;
    config.elo_min_improvement = args.diploid_elo_min;

    std::cout << "\n--- 二倍体训练配置 ---" << std::endl;
    std::cout << "模型: CA-MOE (8专家, ~1.5M参数)" << std::endl;
    std::cout << "模型通道宽度: " << config.hidden_dim << std::endl;
    std::cout << "ELO接受门槛: +" << config.elo_min_improvement << "/代（大幅上涨才接受）" << std::endl;
    std::cout << "HMGS-DP搜索模拟数: " << config.simulations_per_move << std::endl;
    std::cout << "每代自对弈局数: " << config.games_per_gen << std::endl;
    std::cout << "竞争对弈局数: " << config.competition_games << std::endl;
    std::cout << "最大代数: " << config.max_generations << std::endl;
    std::cout << "每代训练轮数: " << config.training_epochs_per_gen << std::endl;
    std::cout << "学习率: " << config.learning_rate << std::endl;
    std::cout << "经验池容量: " << config.replay_buffer_capacity << std::endl;
    std::cout << "输出目录: " << config.output_dir << std::endl;
    std::cout << std::endl;

    // 创建并运行训练器
    DiploidTrainer trainer(config);
    trainer.run();

    std::cout << "\n二倍体训练程序正常退出！" << std::endl;
    logger.shutdown();
    return 0;
}

// 运行ELO门控在线训练模式（单模型 vs Gen0，ELO不涨回滚）
int run_elo_gated_training(const CommandLineArgs& args) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "启动ELO门控在线训练模式" << std::endl;
    std::cout << "========================================\n" << std::endl;

    auto& logger = framework::Logger::instance();
    logger.initialize("SpaceGo_Trainer_EloGated");
    logger.set_min_level(framework::LogLevel::INFO);

    if (!torch::cuda::is_available() || !args.use_cuda) {
        std::cout << "[警告] CUDA不可用或已禁用，将使用CPU（较慢）" << std::endl;
    } else {
        std::cout << "[信息] CUDA可用，使用GPU加速" << std::endl;
    }

    // 配置ELO门控训练参数
    EloGatedConfig config;
    config.output_dir = args.output_dir + "/elo_gated";
    config.checkpoint_dir = config.output_dir + "/checkpoints";
    config.max_rounds = args.elo_rounds;
    config.simulations_per_move = args.elo_simulations;
    config.games_per_round = args.elo_games_per_round;
    config.eval_games = args.elo_eval_games;
    config.temperature = args.elo_temperature;
    config.learning_rate = args.elo_lr;
    config.hidden_dim = args.elo_hidden;
    config.gen0_hidden_dim = args.elo_gen0_hidden;
    config.cumulative_elo_start = args.elo_cumulative_start;  // 扩容/换基线时延续累积ELO
    config.fresh_start = args.elo_fresh_start;  // 彻底从头：随机初始化M（不继承塌缩起点）
    // KataGo评估标尺配置（用户指定：用KataGo看进步，不要求赢它）
    config.use_katago_eval = args.elo_use_katago_eval;
    config.katago_exe = args.elo_katago_exe;
    config.katago_model = args.elo_katago_model;
    config.katago_config = args.elo_katago_config;
    config.katago_visits = args.elo_katago_visits;
    config.katago_selfplay_games = args.elo_katago_selfplay_games;
    config.katago_teacher_visits = args.elo_katago_teacher_visits;
    config.katago_distill_epochs = args.elo_katago_distill_epochs;
    // 默认起点：Gen1峰值 checkpoint；默认基线：Gen0
    config.start_checkpoint = args.elo_start_checkpoint.empty() ?
        config.checkpoint_dir + "/elo_best.pt" : args.elo_start_checkpoint;
    if (!std::filesystem::exists(config.start_checkpoint)) {
        config.start_checkpoint = args.elo_start_checkpoint.empty() ?
            "./training_output/diploid/checkpoints/diploid_a_gen1.pt" :
            args.elo_start_checkpoint;
    }
    config.gen0_checkpoint = args.elo_gen0_checkpoint.empty() ?
        "./training_output/diploid/checkpoints/gen0_baseline.pt" : args.elo_gen0_checkpoint;

    std::cout << "\n--- ELO门控训练配置 ---" << std::endl;
    std::cout << "模型: CA-MOE (8专家)" << std::endl;
    std::cout << "模型通道宽度: " << config.hidden_dim << std::endl;
    std::cout << "起点模型: "
              << (config.fresh_start ? std::string("随机初始化（--elo-fresh 彻底从头）")
                                     : config.start_checkpoint)
              << std::endl;
    std::cout << "Gen0基线: " << config.gen0_checkpoint << std::endl;
    std::cout << "HMGS-DP搜索模拟数: " << config.simulations_per_move << std::endl;
    std::cout << "每轮对弈学习局数: " << config.games_per_round << "（一局学一次）" << std::endl;
    if (config.use_katago_eval) {
        std::cout << "评估器: KataGo标尺（每轮1局，目差进步才接受）" << std::endl;
        std::cout << "KataGo: " << config.katago_exe << std::endl;
        std::cout << "KataGo权重: " << config.katago_model << std::endl;
        std::cout << "KataGo强度: " << config.katago_visits << " visits" << std::endl;
        std::cout << "蒸馏教师棋谱: 每轮" << config.katago_selfplay_games
                  << "局KataGo自我对弈（" << config.katago_teacher_visits << " visits）" << std::endl;
    } else {
        std::cout << "每轮评估局数: " << config.eval_games << "（自对弈）" << std::endl;
    }
    std::cout << "温度: " << config.temperature << " | 学习率: " << config.learning_rate << std::endl;
    std::cout << "最大轮数: " << config.max_rounds << std::endl;
    std::cout << "输出目录: " << config.output_dir << std::endl;
    std::cout << std::endl;

    // 创建并运行ELO门控训练器
    EloGatedTrainer trainer(config);
    if (!trainer.initialize()) {
        std::cerr << "ELO门控训练器初始化失败" << std::endl;
        logger.shutdown();
        return 1;
    }
    trainer.run();

    std::cout << "\nELO门控训练程序正常退出！" << std::endl;
    logger.shutdown();
    return 0;
}

int main(int argc, char* argv[]) {
    // 解析命令行参数
    auto args = parse_args(argc, argv);
    
    // 显示帮助信息
    if (args.show_help) {
        print_usage(argv[0]);
        return 0;
    }
    
    // 设置输出目录
    {
        std::filesystem::create_directories(args.output_dir);
    }
    
    std::cout << "╔══════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     SpaceGo 维空弈 - AI训练系统 v2.0       ║" << std::endl;
    std::cout << "║     Unsupervised Self-Reinforcement Training ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;
    
    // 根据模式选择训练方式
    if (args.mode == "unsupervised") {
        return run_unsupervised_training(args);
    } else if (args.mode == "diploid") {
        return run_diploid_training(args);
    } else if (args.mode == "elo-gated") {
        return run_elo_gated_training(args);
    } else if (args.mode == "generate-config") {
        generate_default_config(args.output_dir + "/unsupervised_config.txt");
        return 0;
    } else {
        // 默认运行监督预训练
        return run_supervised_training();
    }
}
