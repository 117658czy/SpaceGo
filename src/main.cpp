#include <iostream>
#include <memory>
#include <string>
#include <cstring>
#include <chrono>
#include <thread>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iomanip>

#include "framework/logger.h"
#include "go/engine/go_engine.h"
#include "network/protocol/gtp_protocol.h"

#ifdef USE_LIBTORCH
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include "go/training/unsupervised_self_reinforcement.h"
#include "go/training/param_registry.h"
#include "go/training/cli_tuner.h"
#include "go/training/auto_tuner.h"
#endif

#ifdef USE_GUI
// Windows头文件配置
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#define UNICODE
#define _UNICODE

// Windows头文件包含顺序非常重要
#include <windows.h>
#include <objbase.h>
#include <gdiplus.h>

// GDI+库链接
#pragma comment(lib, "gdiplus.lib")
#pragma comment(lib, "ole32.lib")

// GDI+命名空间
using namespace Gdiplus;

#include "components/main_window.h"
#include "utils/font.h"
#include "utils/color.h"
#include "core/animation.h"
#endif

using namespace spacego;

// 运行模式枚举
enum class RunMode {
    GTP,        // GTP协议模式（默认）
    GUI,        // 图形界面模式
    INTERACTIVE,// 交互式命令行模式
    TRAIN,      // 训练模式
    TUNE,       // 调参模式
    AUTO_TUNE   // 自拟合调参模式
};

// 打印帮助信息
void print_help(const char* program_name) {
    std::cout << "用法: " << program_name << " [选项]\n\n"
              << "运行模式:\n"
              << "  --gtp              以GTP协议模式运行（默认）\n"
              << "  --gui, -g          以图形界面模式运行\n"
              << "  --interactive      以交互式命令行模式运行\n"
              << "  --train            启动训练模式\n"
              << "  --tune             启动交互式调参模式\n"
              << "  --auto-tune        启动自拟合调参模式\n\n"
              << "训练选项 (--train 模式):\n"
              << "  --iterations=N     训练迭代次数（默认10000）\n"
              << "  --games=N          每迭代对局数（默认100）\n"
              << "  --batch-size=N     训练批次大小（默认512）\n"
              << "  --workers=N        并行Worker数量（默认8）\n"
              << "  --lr=<float>       学习率（默认0.0003）\n"
              << "  --simulations=N    每步MCTS模拟次数（默认1600）\n"
              << "  --config=<path>    配置文件路径\n"
              << "  --checkpoint=<path>从检查点恢复训练\n"
              << "  --no-cuda          禁用CUDA加速\n"
              << "  --output-dir=<path>输出目录（默认./training_output）\n\n"
              << "通用选项:\n"
              << "  --help, -h         显示帮助信息\n"
              << "  --version, -v      显示版本信息\n\n"
              << "示例:\n"
              << "  " << program_name << " --gtp\n"
              << "  " << program_name << " --train --iterations=100 --games=10\n"
              << "  " << program_name << " --tune\n"
              << std::endl;
}

// 打印版本信息
void print_version() {
    std::cout << go::engine::GoEngine::get_engine_info() << std::endl;
}

#ifdef USE_GUI
// 清理所有GUI单例资源
static void CleanupGUISingletons() {
    SpaceGo::GUI::FontManager* fontManager = SpaceGo::GUI::FontManager::GetInstance();
    if (fontManager) {
        delete fontManager;
    }

    SpaceGo::GUI::ColorManager* colorManager = SpaceGo::GUI::ColorManager::GetInstance();
    if (colorManager) {
        delete colorManager;
    }

    SpaceGo::GUI::AnimationManager* animManager = SpaceGo::GUI::AnimationManager::GetInstance();
    if (animManager) {
        delete animManager;
    }
}

int RunGUI() {
    Gdiplus::GdiplusStartupInput gdiplusStartupInput;
    ULONG_PTR gdiplusToken;
    Gdiplus::GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);

    SpaceGo::GUI::MainWindow* mainWindow = new SpaceGo::GUI::MainWindow();
    mainWindow->Show();
    int result = mainWindow->Run();

    delete mainWindow;

    // 清理所有单例资源
    CleanupGUISingletons();

    Gdiplus::GdiplusShutdown(gdiplusToken);

    return result;
}
#endif

// 运行 GTP 模式
// model_path: 神经网络模型(.pt)路径（--model=），加载后走CA-MOE神经网络引擎；
//             为空则用旧版规则评估模式
// simulations: HMGS-DP搜索迭代数（--simulations=），0表示用引擎默认值
int RunGTP(const std::string& model_path, int simulations) {
    auto& logger = framework::Logger::instance();
    logger.info("启动 GTP 协议模式...");

    // 创建引擎并初始化
    auto engine = std::make_shared<go::engine::GoEngine>();
    go::engine::EngineConfig engine_config;
    engine_config.model_path = model_path;
    if (simulations > 0) {
        engine_config.nn_simulations = simulations;
    }
    if (!engine->initialize(engine_config)) {
        logger.error("引擎初始化失败");
        return 1;
    }

    // 创建 GTP 协议处理器 (使用已创建的引擎)
    network::protocol::GTPProtocol gtp(engine);

    logger.info("GTP 协议处理器已就绪");
    logger.info("等待输入命令...");

    try {
        // 运行 GTP 主循环
        gtp.run(std::cin, std::cout);
    } catch (const std::exception& e) {
        logger.error(std::string("GTP 运行时错误：") + e.what());
        return 1;
    }

    logger.info("GTP 模式结束");
    return 0;
}

// 运行交互式模式
int RunInteractive() {
    auto& logger = framework::Logger::instance();
    logger.info("启动交互式模式...");

    // 创建引擎
    auto engine = std::make_shared<go::engine::GoEngine>();
    if (!engine->initialize()) {
        logger.error("引擎初始化失败");
        return 1;
    }

    std::cout << "\n========================================\n";
    std::cout << "SpaceGo 维空弈 - 交互式模式\n";
    std::cout << "========================================\n";
    std::cout << "输入 'help' 查看可用命令\n";
    std::cout << "输入 'quit' 退出\n";
    std::cout << "========================================\n\n";

    std::string command;
    while (true) {
        std::cout << "spacego> ";
        std::cout.flush();

        if (!std::getline(std::cin, command)) {
            break;
        }

        // 去除首尾空白
        while (!command.empty() && std::isspace(command.front())) {
            command.erase(command.begin());
        }
        while (!command.empty() && std::isspace(command.back())) {
            command.pop_back();
        }

        if (command.empty()) {
            continue;
        }

        // 解析命令
        if (command == "quit" || command == "exit") {
            std::cout << "再见！\n";
            break;
        }
        else if (command == "help") {
            std::cout << "\n可用命令:\n"
                      << "  help           - 显示帮助信息\n"
                      << "  version        - 显示版本信息\n"
                      << "  clear          - 清空棋盘\n"
                      << "  show           - 显示棋盘\n"
                      << "  play <color> <pos> - 落子 (如: play black D4)\n"
                      << "  genmove <color>    - 生成着法\n"
                      << "  undo           - 悔棋\n"
                      << "  analyze        - 分析当前局面\n"
                      << "  quit/exit      - 退出程序\n\n";
        }
        else if (command == "version") {
            std::cout << go::engine::GoEngine::get_engine_info() << "\n";
        }
        else if (command == "clear") {
            engine->clear_board();
            std::cout << "棋盘已清空\n";
        }
        else if (command == "show") {
            std::cout << "\n" << engine->get_board_string() << "\n";
        }
        else if (command.substr(0, 4) == "play") {
            // 解析play命令
            std::istringstream iss(command);
            std::string cmd, color_str, pos;
            iss >> cmd >> color_str >> pos;

            auto color = go::engine::GoEngine::string_to_color(color_str);
            if (engine->play_move(pos, color)) {
                std::cout << "落子成功: " << color_str << " " << pos << "\n";
            } else {
                std::cout << "落子失败: 非法着法\n";
            }
        }
        else if (command.substr(0, 7) == "genmove") {
            // 解析genmove命令
            std::istringstream iss(command);
            std::string cmd, color_str;
            iss >> cmd >> color_str;

            auto color = go::engine::GoEngine::string_to_color(color_str);
            auto result = engine->genmove(color);

            std::cout << "最佳着法: " << result.move_string
                      << " (胜率: " << (result.win_probability * 100) << "%)\n";
        }
        else if (command == "undo") {
            if (engine->undo_move()) {
                std::cout << "悔棋成功\n";
            } else {
                std::cout << "悔棋失败\n";
            }
        }
        else if (command == "analyze") {
            auto result = engine->analyze();
            std::cout << "分析结果: " << result.move_string
                      << " (胜率: " << (result.win_probability * 100)
                      << "%, 访问: " << result.visits << ")\n";

            // 显示前5个候选着法
            auto probs = engine->get_move_probabilities();
            std::cout << "候选着法:\n";
            for (size_t i = 0; i < probs.size() && i < 5; ++i) {
                std::cout << "  " << probs[i].first << ": "
                          << (probs[i].second * 100) << "%\n";
            }
        }
        else {
            std::cout << "未知命令: " << command << "\n";
            std::cout << "输入 'help' 查看可用命令\n";
        }
    }

    return 0;
}

#ifdef USE_LIBTORCH
// 运行训练模式
int RunTrain(int iterations, int games, int batch_size, int workers,
             float lr, int simulations,
             const std::string& config_path, const std::string& checkpoint_path,
             bool use_cuda, const std::string& output_dir,
             const std::string& log_level) {
    auto& logger = framework::Logger::instance();
    logger.initialize("SpaceGo_Train");

    framework::LogLevel log_lvl = framework::LogLevel::INFO;
    if (log_level == "debug") log_lvl = framework::LogLevel::DEBUG;
    else if (log_level == "warning") log_lvl = framework::LogLevel::WARNING;
    else if (log_level == "error") log_lvl = framework::LogLevel::LVL_ERROR;
    logger.set_min_level(log_lvl);

    try {
        // 加载或创建配置
        go::training::UnsupervisedTrainingConfig config;

        if (!config_path.empty()) {
            // 从配置文件加载
            std::ifstream file(config_path);
            if (file.is_open()) {
                // 使用ParamRegistry加载
                auto& reg = go::training::ParamRegistry::instance();
                go::training::initialize_param_registry();
                reg.load_from_file(config_path);
                reg.apply_to_config(config);
                std::cout << "从配置文件加载设置: " << config_path << std::endl;
            }
        }

        // 命令行参数覆盖配置
        config.total_iterations = iterations;
        config.self_play.num_workers = workers;
        config.training.games_per_iteration = games;
        config.training.batch_size = batch_size;
        config.training.learning_rate = lr;
        config.self_play.simulations_per_move = simulations;
        config.self_play.use_cuda = use_cuda && torch::cuda::is_available();
        config.checkpoint_dir = output_dir + "/checkpoints";
        config.log_dir = output_dir + "/logs";

        // 打印环境信息
        std::cout << "\n--- 环境信息 ---" << std::endl;
        if (torch::cuda::is_available() && use_cuda) {
            std::cout << "CUDA状态: 可用" << std::endl;
            std::cout << "GPU: " << at::cuda::getDeviceProperties(0)->name << std::endl;
            std::cout << "显存: " << (at::cuda::getDeviceProperties(0)->totalGlobalMem / 1024 / 1024 / 1024) << " GB" << std::endl;
        } else {
            std::cout << "CUDA状态: 不可用或被禁用" << std::endl;
            config.self_play.use_cuda = false;
        }

        std::cout << "\n--- 训练配置 ---" << std::endl;
        std::cout << "总迭代: " << config.total_iterations << std::endl;
        std::cout << "每迭代局数: " << config.training.games_per_iteration << std::endl;
        std::cout << "批次大小: " << config.training.batch_size << std::endl;
        std::cout << "学习率: " << config.training.learning_rate << std::endl;
        std::cout << "模拟次数: " << config.self_play.simulations_per_move << std::endl;
        std::cout << "Workers: " << config.self_play.num_workers << std::endl;
        std::cout << "渐进扩张: " << (config.training.progressive_expansion ? "启用" : "禁用") << std::endl;

        // 创建训练器
        auto trainer = std::make_unique<go::training::UnsupervisedSelfReinforcementTrainer>(config);

        if (!trainer->initialize()) {
            std::cerr << "[错误] 训练系统初始化失败" << std::endl;
            return 1;
        }

        // 加载检查点
        if (!checkpoint_path.empty()) {
            std::cout << "\n从检查点恢复: " << checkpoint_path << std::endl;
            trainer->load_checkpoint(checkpoint_path);
        }

        // 开始训练
        trainer->start_training();

        auto training_start_time = std::chrono::steady_clock::now();
        int last_displayed_iteration = -1;

        // 主线程等待训练完成，每2秒刷新进度
        while (trainer->is_training()) {
            std::this_thread::sleep_for(std::chrono::seconds(2));

            auto metrics = trainer->get_latest_metrics();
            int current_iter = trainer->get_current_iteration();
            int total_iters = config.total_iterations;
            auto replay_buf = trainer->get_replay_buffer();

            if (current_iter == last_displayed_iteration) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration<float>(now - training_start_time).count();
                int elapsed_min = static_cast<int>(elapsed) / 60;
                int elapsed_sec = static_cast<int>(elapsed) % 60;
                std::cout << "\r  [等待中] 迭代 " << current_iter << "/" << total_iters
                          << " | 池 " << replay_buf->size() << "/" << replay_buf->capacity()
                          << " | " << elapsed_min << "m" << elapsed_sec << "s"
                          << std::flush;
                continue;
            }
            last_displayed_iteration = current_iter;

            float progress = static_cast<float>(current_iter) / std::max(1, total_iters);
            int bar_width = 20;
            int filled = static_cast<int>(progress * bar_width);
            std::string bar(bar_width, ' ');
            for (int i = 0; i < bar_width; ++i) {
                bar[i] = (i < filled) ? '#' : '-';
            }

            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration<float>(now - training_start_time).count();
            int elapsed_min = static_cast<int>(elapsed) / 60;
            int elapsed_sec = static_cast<int>(elapsed) % 60;

            float eta_sec = 0.0f;
            if (current_iter > 0 && progress < 1.0f) {
                float avg_iter_time = elapsed / current_iter;
                eta_sec = avg_iter_time * (total_iters - current_iter);
            }
            int eta_min = static_cast<int>(eta_sec) / 60;
            int eta_s = static_cast<int>(eta_sec) % 60;

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

        std::cout << std::endl;

        // 打印最终报告
        auto final_metrics = trainer->get_latest_metrics();
        std::cout << "\n========================================" << std::endl;
        std::cout << "训练完成！" << std::endl;
        std::cout << "总迭代: " << final_metrics.iteration << std::endl;
        std::cout << "总对局: " << final_metrics.total_games_generated << std::endl;
        std::cout << "最终ELO: " << final_metrics.current_elo << std::endl;
        std::cout << "策略损失: " << final_metrics.avg_policy_loss << std::endl;
        std::cout << "价值损失: " << final_metrics.avg_value_loss << std::endl;
        std::cout << "========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\n[错误] 训练异常: " << e.what() << std::endl;
        logger.fatal("训练失败: " + std::string(e.what()));
        logger.shutdown();
        return 1;
    }

    logger.shutdown();
    return 0;
}

// 运行调参模式
int RunTune() {
    auto& logger = framework::Logger::instance();
    logger.initialize("SpaceGo_Tune");
    logger.set_min_level(framework::LogLevel::INFO);

    try {
        go::training::CLITuner tuner;

        // 设置训练回调：在调参界面中触发训练时调用
        tuner.set_train_callback([](const go::training::UnsupervisedTrainingConfig& config) -> int {
            auto& log = framework::Logger::instance();
            log.info("从调参界面启动训练...");

            try {
                auto trainer = std::make_unique<go::training::UnsupervisedSelfReinforcementTrainer>(config);
                if (!trainer->initialize()) {
                    std::cerr << "[错误] 训练初始化失败\n";
                    return 1;
                }

                trainer->start_training();

                // 等待训练完成
                while (trainer->is_training()) {
                    std::this_thread::sleep_for(std::chrono::seconds(2));
                }

                auto metrics = trainer->get_latest_metrics();
                std::cout << "\n训练完成！迭代=" << metrics.iteration
                          << " ELO=" << metrics.current_elo
                          << " 策略损失=" << metrics.avg_policy_loss
                          << " 价值损失=" << metrics.avg_value_loss << "\n\n";

                return 0;
            } catch (const std::exception& e) {
                std::cerr << "\n[训练异常] " << e.what() << "\n";
                return 1;
            }
        });

        // 启动CLI交互循环
        tuner.run();

    } catch (const std::exception& e) {
        std::cerr << "[错误] " << e.what() << "\n";
        logger.fatal("调参模式异常: " + std::string(e.what()));
        logger.shutdown();
        return 1;
    }

    logger.shutdown();
    return 0;
}

// 运行自拟合调参模式
int RunAutoTune(int train_iterations, int train_games, int train_batch_size,
                int train_workers, float train_lr, int train_simulations,
                const std::string& config_path, bool use_cuda,
                const std::string& output_dir) {
    auto& logger = framework::Logger::instance();
    logger.initialize("SpaceGo_AutoTune");
    logger.set_min_level(framework::LogLevel::INFO);

    try {
        // 创建基础训练配置
        go::training::UnsupervisedTrainingConfig config;

        if (!config_path.empty()) {
            auto& reg = go::training::ParamRegistry::instance();
            go::training::initialize_param_registry();
            reg.load_from_file(config_path);
            reg.apply_to_config(config);
            std::cout << "从配置文件加载设置: " << config_path << std::endl;
        }

        // 命令行参数覆盖
        config.total_iterations = train_iterations;
        config.self_play.num_workers = train_workers;
        config.training.games_per_iteration = train_games;
        config.training.batch_size = train_batch_size;
        config.training.learning_rate = train_lr;
        config.self_play.simulations_per_move = train_simulations;
        config.self_play.use_cuda = use_cuda && torch::cuda::is_available();

        // AutoTuner需要较小的配置，确保快速迭代
        config.self_play.simulations_per_move = std::min(train_simulations, 400);
        config.training.batch_size = std::min(train_batch_size, 256);
        config.self_play.num_workers = std::min(train_workers, 2);

        // 打印环境信息
        std::cout << "\n--- 自拟合调参环境 ---" << std::endl;
        if (torch::cuda::is_available() && use_cuda) {
            std::cout << "CUDA: 可用" << std::endl;
            std::cout << "GPU: " << at::cuda::getDeviceProperties(0)->name << std::endl;
        } else {
            std::cout << "CUDA: 不可用" << std::endl;
            config.self_play.use_cuda = false;
        }

        // 创建自拟合调参器
        go::training::AutoTuner tuner(config);
        tuner.set_output_dir(output_dir + "/auto_tune");

        // 启动自动调参
        tuner.run();

        // 获取最优配置
        auto best_config = tuner.get_best_config();
        std::cout << "\n========================================" << std::endl;
        std::cout << "自拟合调参完成！" << std::endl;
        std::cout << "最优参数已保存到: " << output_dir << "/auto_tune/best_params.txt" << std::endl;
        std::cout << "可用 --config=" << output_dir << "/auto_tune/best_params.txt 启动训练" << std::endl;
        std::cout << "========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\n[错误] 自拟合调参异常: " << e.what() << std::endl;
        logger.fatal("自拟合调参失败: " + std::string(e.what()));
        logger.shutdown();
        return 1;
    }

    logger.shutdown();
    return 0;
}
#endif // USE_LIBTORCH

int main(int argc, char* argv[]) {
    // ===== 第一阶段：快速解析模式参数，尽早分流 =====
    // 训练/调参模式需要跳过GTP/GUI初始化，避免与CUDA冲突
    RunMode mode = RunMode::GTP;

    // 训练相关参数
    int train_iterations = 10000;
    int train_games = 100;
    int train_batch_size = 512;
    int train_workers = 8;
    float train_lr = 3e-4f;
    int train_simulations = 1600;
    // GTP引擎参数（--model=加载神经网络模型，--simulations=设置搜索迭代数）
    std::string gtp_model_path;
    int gtp_simulations = 0;
    std::string config_path;
    std::string checkpoint_path;
    bool use_cuda = true;
    std::string output_dir = "./training_output";
    std::string log_level = "info";

    // 第一遍：只解析模式标志和训练参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--gtp") {
            mode = RunMode::GTP;
        }
        else if (arg == "--gui" || arg == "-g") {
            mode = RunMode::GUI;
        }
        else if (arg == "--interactive") {
            mode = RunMode::INTERACTIVE;
        }
        else if (arg == "--train") {
#ifdef USE_LIBTORCH
            mode = RunMode::TRAIN;
#else
            std::cerr << "错误: 训练模式需要LibTorch支持\n";
            return 1;
#endif
        }
        else if (arg == "--tune") {
#ifdef USE_LIBTORCH
            mode = RunMode::TUNE;
#else
            std::cerr << "错误: 调参模式需要LibTorch支持\n";
            return 1;
#endif
        }
        else if (arg == "--auto-tune") {
#ifdef USE_LIBTORCH
            mode = RunMode::AUTO_TUNE;
#else
            std::cerr << "错误: 自拟合调参模式需要LibTorch支持\n";
            return 1;
#endif
        }
        else if (arg.find("--iterations=") == 0) {
            train_iterations = std::stoi(arg.substr(13));
        }
        else if (arg.find("--games=") == 0) {
            train_games = std::stoi(arg.substr(8));
        }
        else if (arg.find("--batch-size=") == 0) {
            train_batch_size = std::stoi(arg.substr(13));
        }
        else if (arg.find("--workers=") == 0) {
            train_workers = std::stoi(arg.substr(10));
        }
        else if (arg.find("--lr=") == 0) {
            train_lr = std::stof(arg.substr(5));
        }
        else if (arg.find("--simulations=") == 0) {
            train_simulations = std::stoi(arg.substr(14));
            gtp_simulations = train_simulations;  // GTP模式复用同一参数
        }
        else if (arg.find("--model=") == 0) {
            gtp_model_path = arg.substr(8);  // GTP引擎神经网络模型路径(.pt)
        }
        else if (arg.find("--config=") == 0) {
            config_path = arg.substr(9);
        }
        else if (arg.find("--checkpoint=") == 0) {
            checkpoint_path = arg.substr(13);
        }
        else if (arg == "--no-cuda") {
            use_cuda = false;
        }
        else if (arg.find("--output-dir=") == 0) {
            output_dir = arg.substr(13);
        }
        else if (arg.find("--log-level=") == 0) {
            log_level = arg.substr(12);
        }
        else if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            return 0;
        }
        else if (arg == "--version" || arg == "-v") {
            print_version();
            return 0;
        }
        else {
            std::cerr << "未知参数: " << arg << "\n";
            print_help(argv[0]);
            return 1;
        }
    }

    // ===== 第二阶段：训练/调参模式直接分流，不初始化GTP组件 =====
#ifdef USE_LIBTORCH
    if (mode == RunMode::TRAIN) {
        // 训练模式：直接进入，不初始化GoEngine/GTP等组件
        // 禁用目录创建（Sandbox会阻止并杀掉进程）
        // std::filesystem::create_directories(output_dir);
        return RunTrain(train_iterations, train_games, train_batch_size,
                        train_workers, train_lr, train_simulations,
                        config_path, checkpoint_path, use_cuda, output_dir, log_level);
    }
    else if (mode == RunMode::TUNE) {
        // 调参模式：直接进入
        return RunTune();
    }
    else if (mode == RunMode::AUTO_TUNE) {
        // 自拟合调参模式：直接进入
        // 禁用目录创建（Sandbox会阻止并杀掉进程）
        // std::filesystem::create_directories(output_dir);
        return RunAutoTune(train_iterations, train_games, train_batch_size,
                           train_workers, train_lr, train_simulations,
                           config_path, use_cuda, output_dir);
    }
#endif

    // ===== 第三阶段：GTP/GUI/Interactive模式，初始化通用组件 =====
    auto& logger = framework::Logger::instance();
    logger.initialize("SpaceGo");
    logger.set_min_level(framework::LogLevel::INFO);

    logger.info("========================================");
    logger.info("SpaceGo 维空弈");
    logger.info("Version " + go::engine::GoEngine::get_engine_version());
    logger.info("========================================");

    // banner 输出到 stderr：GTP 模式下 stdout 必须只含协议响应，
    // 否则外部引擎对接（管道/GUI）会因 banner 混入而错位
    std::cerr << "╔══════════════════════════════════════════════╗" << std::endl;
    std::cerr << "║     SpaceGo 维空弈 - AI围棋引擎             ║" << std::endl;
    std::cerr << "╚══════════════════════════════════════════════╝" << std::endl;
    std::cerr << std::endl;

    int result = 0;

    try {
        switch (mode) {
            case RunMode::GTP:
                result = RunGTP(gtp_model_path, gtp_simulations);
                break;

            case RunMode::GUI:
#ifdef USE_GUI
                logger.info("启动GUI模式...");
                result = RunGUI();
#else
                logger.error("GUI模式未启用，请使用 --gtp 或 --interactive 模式");
                std::cerr << "错误: GUI模式未启用\n";
                result = 1;
#endif
                break;

            case RunMode::INTERACTIVE:
                result = RunInteractive();
                break;

            default:
                result = RunGTP(gtp_model_path, gtp_simulations);
                break;
        }

        logger.info("========================================");
        logger.info("SpaceGo 正常退出");

    } catch (const std::exception& e) {
        logger.fatal("致命错误: " + std::string(e.what()));
        std::cerr << "错误: " << e.what() << "\n";
        result = 1;
    }

    logger.shutdown();
    return result;
}
