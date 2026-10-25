#include "go/training/elo_gated_trainer.h"

#ifdef USE_LIBTORCH

#include "go/board/go_board_cuda.h"
#include "go/neural/ca_moe_expert.h"
#include "go/search/hmgs_dp_optimized.h"

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>

#ifdef _WIN32
// Windows 管道/进程 API（KataGo GTP 客户端）：NOMINMAX 避免与 torch 的 std::max 冲突
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#ifdef USE_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace spacego {
namespace go {
namespace training {

namespace {

// ==================== 匿名命名空间：对弈辅助函数 ====================
// 与SelfPlayGenerator中同名函数逻辑一致，但ELO门控训练器独立运行，
// 避免跨文件耦合（各翻译单元匿名命名空间互不冲突）。

// 全局推理互斥锁：M与Gen0搜索共享GPU，串行化前向推理避免内核竞争
std::mutex g_elo_infer_mutex;

// 对局贴目（中国规则；KataGo侧经"komi 7.5"设置，故final_score已含此贴目）。
// 集中定义避免多处硬编码不一致，同时供"目差贴目修正"使用（见play_game_vs_katago）。
constexpr float kKomi = 7.5f;

// 锁内执行模型推理：单次前向同时输出策略/价值/门控
neural::CA_MOEInferenceResult locked_forward(
    std::shared_ptr<neural::CA_MOEExpertSystem> model, const torch::Tensor& state) {
    std::lock_guard<std::mutex> lock(g_elo_infer_mutex);
    return (*model)->forward(state);
}

// 构建362维合法着法mask（0-360落子，361=pass）
std::vector<bool> build_legal_mask(const GoBoardCPU& board) {
    std::vector<bool> mask(362, false);
    StoneColor player = board.get_current_player();
    auto legal_moves = board.get_legal_moves(player);
    for (const auto& [x, y] : legal_moves) {
        int idx = GoBoardCPU::xy_to_index(x, y);
        if (idx >= 0 && idx < 361) {
            mask[idx] = true;
        }
    }
    mask[361] = true;  // pass始终合法
    return mask;
}

// ==================== KataGo GTP 客户端（外部评估基准引擎） ====================
// 通过 Windows 双向管道启动 katago.exe（GTP 协议），作为 ELO 评估对手。
// 关键点：
//   · stderr 重定向到 NUL——KataGo 日志/调优输出若灌满 stderr 管道会阻塞子进程
//   · 读响应直到空行——GTP 协议规定响应以空行（\n\n）结束
//   · -override-config maxVisits=N——训练侧动态控制 KataGo 基准强度
class KataGoGTPClient {
public:
    KataGoGTPClient() = default;
    ~KataGoGTPClient() { close(); }

    // 启动 katago.exe gtp -model <model> -config <config> -override-config maxVisits=N
    bool start(const std::string& exe, const std::string& model,
               const std::string& config, int visits) {
#ifdef _WIN32
        SECURITY_ATTRIBUTES sa{ sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE };
        HANDLE in_read = nullptr, in_write = nullptr, out_read = nullptr, out_write = nullptr;
        if (!CreatePipe(&in_read, &in_write, &sa, 0)) return false;
        if (!CreatePipe(&out_read, &out_write, &sa, 0)) {
            CloseHandle(in_read);
            CloseHandle(in_write);
            return false;
        }
        // 子进程只继承各自的使用端，父进程持有端关闭继承
        SetHandleInformation(in_write, HANDLE_FLAG_INHERIT, 0);
        SetHandleInformation(out_read, HANDLE_FLAG_INHERIT, 0);
        HANDLE null_handle = CreateFileA("NUL", GENERIC_WRITE, FILE_SHARE_WRITE,
                                         &sa, OPEN_EXISTING, 0, nullptr);

        std::string cmdline = "\"" + exe + "\" gtp -model \"" + model +
                              "\" -config \"" + config +
                              "\" -override-config maxVisits=" + std::to_string(visits);
        // CreateProcess 要求命令行缓冲区可写
        std::vector<char> cmd_buf(cmdline.begin(), cmdline.end());
        cmd_buf.push_back('\0');

        STARTUPINFOA si{ sizeof(STARTUPINFOA) };
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput = in_read;
        si.hStdOutput = out_write;
        si.hStdError = null_handle;
        PROCESS_INFORMATION pi{};
        BOOL ok = CreateProcessA(nullptr, cmd_buf.data(), nullptr, nullptr, TRUE,
                                 CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
        CloseHandle(in_read);
        CloseHandle(out_write);
        if (null_handle) CloseHandle(null_handle);
        if (!ok) {
            CloseHandle(in_write);
            CloseHandle(out_read);
            return false;
        }
        CloseHandle(pi.hThread);
        proc_handle_ = pi.hProcess;
        stdin_handle_ = in_write;    // 父进程写（发命令）
        stdout_handle_ = out_read;   // 父进程读（收响应）
        return true;
#else
        (void)exe; (void)model; (void)config; (void)visits;
        return false;
#endif
    }

    // 发送一条 GTP 命令并读取响应直到空行（去除尾部空行后返回）
    std::string send(const std::string& cmd) {
        std::lock_guard<std::mutex> lock(mutex_);
#ifdef _WIN32
        std::string line = cmd + "\n";
        DWORD written = 0;
        if (!WriteFile(stdin_handle_, line.data(), static_cast<DWORD>(line.size()),
                       &written, nullptr)) {
            return "";
        }
        FlushFileBuffers(stdin_handle_);
        std::string response;
        std::string current;
        char buf[512];
        while (true) {
            DWORD bytes = 0;
            if (!ReadFile(stdout_handle_, buf, sizeof(buf), &bytes, nullptr) || bytes == 0) {
                break;  // 子进程退出/管道关闭
            }
            for (DWORD i = 0; i < bytes; ++i) {
                char ch = buf[i];
                if (ch == '\n') {
                    if (current.empty() || current == "\r") {
                        return response;  // 空行 = 响应结束
                    }
                    if (!response.empty()) response += '\n';
                    response += current;
                    current.clear();
                } else if (ch != '\r') {
                    current += ch;
                }
            }
        }
        if (!current.empty()) {
            if (!response.empty()) response += '\n';
            response += current;
        }
        return response;
#else
        (void)cmd;
        return "";
#endif
    }

    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
#ifdef _WIN32
        if (proc_handle_) {
            std::string q = "quit\n";
            DWORD w = 0;
            if (stdin_handle_) WriteFile(stdin_handle_, q.data(),
                                         static_cast<DWORD>(q.size()), &w, nullptr);
            if (WaitForSingleObject(proc_handle_, 2000) == WAIT_TIMEOUT) {
                TerminateProcess(proc_handle_, 1);
            }
            CloseHandle(proc_handle_);
            proc_handle_ = nullptr;
        }
        if (stdin_handle_) { CloseHandle(stdin_handle_); stdin_handle_ = nullptr; }
        if (stdout_handle_) { CloseHandle(stdout_handle_); stdout_handle_ = nullptr; }
#endif
    }

    bool is_running() const { return proc_handle_ != nullptr; }

private:
#ifdef _WIN32
    HANDLE proc_handle_ = nullptr;
    HANDLE stdin_handle_ = nullptr;
    HANDLE stdout_handle_ = nullptr;
#endif
    std::mutex mutex_;
};

// GTP 坐标 <-> 棋盘索引（与 GoEngine::move_to_string 语义一致：列 A-H,J-T 跳过 I，
// 行号1=底部；内部 x=行(0=19路顶部)，y=列(0=A)）
std::string idx_to_gtp(int idx, int board_size = 19) {
    if (idx < 0) return "pass";
    int x = idx % board_size;    // x = 行(0=顶部)
    int y = idx / board_size;    // y = 列(0=A)
    std::string result;
    result += static_cast<char>('A' + y + (y >= 8 ? 1 : 0));  // 跳过 I
    result += std::to_string(board_size - x);
    return result;
}

// 解析 GTP 坐标（如 "D4"/"pass"）为棋盘索引；非法返回 -1
int gtp_to_idx(const std::string& mv, int board_size = 19) {
    if (mv.empty() || mv == "pass" || mv == "resign") return -1;
    char col = static_cast<char>(std::toupper(static_cast<unsigned char>(mv[0])));
    if (col < 'A' || col > 'T' || col == 'I') return -1;
    int y = col - 'A';
    if (col > 'I') y--;
    try {
        int row = std::stoi(mv.substr(1));
        if (row < 1 || row > board_size) return -1;
        int x = board_size - row;
        if (x < 0 || x >= board_size || y < 0 || y >= board_size) return -1;
        return y * board_size + x;  // index = y*size + x
    } catch (...) {
        return -1;
    }
}

// ==================== 通道迁移（双向：截取/扩展，快速ELO起步与扩容） ====================
// 将源模型权重适配到目标模型通道宽度，双向支持：
//   · 大→小（256→128）：逐维截取前缀通道（narrow），小模型继承大模型棋力快速起步
//   · 小→大（128→256）：逐维复制扩展（repeat_interleave），大模型继承小模型棋力
//     作为扩容起点——每个源通道复制成 factor 份（块复制），语义为"权重克隆"，
//     训练中逐步分化出更细的专家表示（突破128小模型容量上限）
//   · 按参数名逐一对应（src/dst结构同构，named_parameters key完全一致）
//   · 形状相同 → 直接复制（fc2门控输出层、style_projection等）
//   · buffers（BN running_mean/var）同样迁移，缺失key跳过
// 返回true表示至少完成一次有效迁移。
static bool migrate_model_channels(
    std::shared_ptr<neural::CA_MOEExpertSystem> src,   // 已加载的源模型（任意通道宽）
    std::shared_ptr<neural::CA_MOEExpertSystem> dst) {  // 目标模型（任意通道宽）
    if (!src || !dst) return false;
    torch::NoGradGuard no_grad;

    // 维度适配辅助：src_t → dst形状（逐维处理）
    //   · 源大 → narrow截取前缀；源小 → repeat_interleave复制扩展（要求dst为src整数倍）
    //   · 用narrow/repeat_interleave逐维操作（Tensor::index对vector<Slice>无重载）
    auto adapt_to_shape = [](const torch::Tensor& src_t,
                             const torch::Tensor& dst_t) -> torch::Tensor {
        torch::Tensor out = src_t;
        for (int d = 0; d < dst_t.dim(); ++d) {
            if (d >= src_t.dim()) return torch::Tensor();  // 维度不足，无法迁移
            if (src_t.size(d) > dst_t.size(d)) {
                out = out.narrow(d, 0, dst_t.size(d));  // 大→小：截取前缀
            } else if (src_t.size(d) < dst_t.size(d)) {
                // 小→大：复制扩展，要求目标维度是源维度的整数倍（如128→256为2倍）
                if (dst_t.size(d) % src_t.size(d) != 0) return torch::Tensor();
                int factor = dst_t.size(d) / src_t.size(d);
                out = out.repeat_interleave(factor, d);
            }
        }
        return out.contiguous();
    };

    bool migrated_any = false;

    // 参数迁移（权重/偏置）：named_parameters返回torch::OrderedDict，
    // 用contains/at按键查找（键为模块注册名层级路径，src/dst同构完全一致）
    auto src_params = (*src)->named_parameters();
    auto dst_params = (*dst)->named_parameters();
    for (auto& item : dst_params) {
        const std::string& name = item.key();
        if (!src_params.contains(name)) continue;  // dst独有参数（保持当前初始化值）
        // 注意：不能用.at()——"at"与torch的at::命名空间冲突（C2882），用operator[]
        auto& src_t = src_params[name];
        auto& dst_t = item.value();
        torch::Tensor src_adapted;
        if (src_t.sizes() == dst_t.sizes()) {
            src_adapted = src_t;
        } else {
            src_adapted = adapt_to_shape(src_t, dst_t);
        }
        if (!src_adapted.defined()) continue;
        dst_t.data().copy_(src_adapted);
        migrated_any = true;
    }

    // buffers迁移（BN running_mean/var/num_batches_tracked）
    auto src_bufs = (*src)->named_buffers();
    auto dst_bufs = (*dst)->named_buffers();
    for (auto& item : dst_bufs) {
        const std::string& name = item.key();
        if (!src_bufs.contains(name)) continue;
        auto& src_b = src_bufs[name];  // 同at冲突原因，用operator[]
        auto& dst_b = item.value();
        torch::Tensor src_adapted;
        if (src_b.sizes() == dst_b.sizes()) {
            src_adapted = src_b;
        } else {
            src_adapted = adapt_to_shape(src_b, dst_b);
        }
        if (!src_adapted.defined()) continue;
        dst_b.data().copy_(src_adapted);
        migrated_any = true;
    }

    return migrated_any;
}

// 用指定通道宽度尝试加载checkpoint到临时模型（成功返回非空，失败返回nullptr）
// 用于通道迁移：不知道源checkpoint的通道宽度时，逐一候选宽度试加载。
static std::shared_ptr<neural::CA_MOEExpertSystem> try_load_checkpoint(
    int width, int num_experts, int board_size, float dropout_rate,
    const std::string& path) {
    auto tmp = std::make_shared<neural::CA_MOEExpertSystem>(
        num_experts, board_size, dropout_rate, width);
    try {
        (*tmp)->load(path);
        return tmp;
    } catch (...) {
        return nullptr;
    }
}
// 基于温度采样动作（高温度→探索，低温度→贪心）
int select_action(const torch::Tensor& policy, float temperature,
                  const std::vector<bool>& legal_mask, std::mt19937& rng) {
    auto policy_cpu = policy.to(torch::kCPU);
    auto policy_data = policy_cpu.contiguous().data_ptr<float>();
    int num_actions = static_cast<int>(policy.numel());

    if (temperature <= 0.01f) {
        // 贪心选择：合法动作中概率最高
        int best_action = 361;
        float best_prob = -1.0f;
        for (int i = 0; i < num_actions; ++i) {
            if (legal_mask[i] && policy_data[i] > best_prob) {
                best_prob = policy_data[i];
                best_action = i;
            }
        }
        return best_action;
    }

    // 温度概率采样：p_i^(1/T)，仅合法动作
    std::vector<float> adjusted(num_actions, 0.0f);
    float sum = 0.0f;
    for (int i = 0; i < num_actions; ++i) {
        if (legal_mask[i]) {
            adjusted[i] = std::pow(std::max(policy_data[i], 0.001f), 1.0f / temperature);
            sum += adjusted[i];
        }
    }
    if (sum <= 0.0f) return 361;
    for (auto& v : adjusted) v /= sum;

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cumulative = 0.0f;
    for (int i = 0; i < num_actions; ++i) {
        cumulative += adjusted[i];
        if (r <= cumulative) return i;
    }
    return 361;
}

// 对策略分布加Dirichlet噪声（前30步探索，仅影响动作选择不影响训练目标）
torch::Tensor add_dirichlet_noise(const torch::Tensor& policy, std::mt19937& rng) {
    const float alpha = 0.3f;   // Dirichlet浓度（19x19围棋常用0.3）
    const float weight = 0.25f; // 噪声混合权重
    int num_actions = static_cast<int>(policy.numel());

    std::gamma_distribution<float> gamma(alpha, 1.0f);
    std::vector<float> noise(static_cast<size_t>(num_actions));
    float noise_sum = 0.0f;
    for (int i = 0; i < num_actions; ++i) {
        noise[static_cast<size_t>(i)] = gamma(rng);
        noise_sum += noise[static_cast<size_t>(i)];
    }
    if (noise_sum > 0.0f) {
        for (auto& n : noise) n /= noise_sum;
    }

    auto policy_cpu = policy.to(torch::kCPU).contiguous();
    auto noisy = policy_cpu;
    auto noisy_data = noisy.data_ptr<float>();
    auto orig_data = policy_cpu.data_ptr<float>();
    for (int i = 0; i < num_actions; ++i) {
        noisy_data[i] = (1.0f - weight) * orig_data[i] + weight * noise[static_cast<size_t>(i)];
    }
    return noisy.to(policy.device());
}

// 归一化搜索访问次数为概率分布（策略训练目标）
torch::Tensor normalize_policy(const torch::Tensor& mcts_visits) {
    auto normalized = mcts_visits.to(torch::kCPU).contiguous();
    auto data = normalized.data_ptr<float>();
    int num_elements = static_cast<int>(normalized.numel());
    float sum = 0.0f;
    for (int i = 0; i < num_elements; ++i) sum += data[i];
    if (sum > 0.0f) {
        for (int i = 0; i < num_elements; ++i) data[i] /= sum;
    }
    return normalized;
}

// 取门控权重前top-k专家索引
std::vector<int> get_top_k_experts(const torch::Tensor& gating_weights, int top_k) {
    auto weights_cpu = gating_weights.to(torch::kCPU);
    auto weights_data = weights_cpu.contiguous().data_ptr<float>();
    int num_experts = static_cast<int>(weights_cpu.numel());

    std::vector<std::pair<float, int>> experts;
    experts.reserve(static_cast<size_t>(num_experts));
    for (int i = 0; i < num_experts; ++i) {
        experts.emplace_back(weights_data[i], i);
    }
    std::sort(experts.begin(), experts.end(),
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                  return a.first > b.first;
              });

    std::vector<int> result;
    for (int i = 0; i < top_k && i < num_experts; ++i) {
        result.push_back(experts[static_cast<size_t>(i)].second);
    }
    return result;
}

} // namespace

// ==================== 构造与初始化 ====================

EloGatedTrainer::EloGatedTrainer(const EloGatedConfig& config)
    : config_(config), device_(torch::kCPU), rng_(config.seed) {
    if (torch::cuda::is_available()) {
        device_ = torch::Device(torch::kCUDA, 0);
    }
}

EloGatedTrainer::~EloGatedTrainer() {
    trainer_.reset();
    replay_.reset();
}

bool EloGatedTrainer::initialize() {
    try {
        // stdout无缓冲：管道重定向下默认全缓冲（4KB），崩溃会丢失最近输出，
        //   导致日志停留在"评估中 60/60"（真实崩溃点可能在其后若干行）。
        //   禁用缓冲让每行实时落盘，精确定位崩溃位置。
        std::setvbuf(stdout, nullptr, _IONBF, 0);
        std::setvbuf(stderr, nullptr, _IONBF, 0);

        std::filesystem::create_directories(config_.output_dir);
        std::filesystem::create_directories(config_.checkpoint_dir);

        // 创建学习模型M与固定基线Gen0（8专家，19x19，通道宽度分别由hidden_dim/gen0_hidden_dim控制）
        // M用hidden_dim（小模型快速迭代），Gen0用gen0_hidden_dim（默认256与强模型checkpoint兼容），
        // 支持"128小模型挑战256强基线"——M从迁移子集起步追赶，ELO相对强基线持续上涨。
        model_ = std::make_shared<neural::CA_MOEExpertSystem>(
            config_.num_experts, config_.board_size, config_.dropout_rate, config_.hidden_dim);
        gen0_ = std::make_shared<neural::CA_MOEExpertSystem>(
            config_.num_experts, config_.board_size, config_.dropout_rate, config_.gen0_hidden_dim);

        // [诊断] 打印实际参数量，确认hidden_dim是否真正生效（128→约30M，256→约62M）
        {
            int n = 0;
            for (const auto& p : (*model_)->parameters()) n += static_cast<int>(p.numel());
            std::cout << "[EloGated] M参数量=" << (n / 1000) << "K"
                      << " | hidden_dim=" << config_.hidden_dim
                      << " | experts=" << config_.num_experts << std::endl;
        }

        // M加载起点模型（如diploid Gen34峰值）：通道数不匹配（256起点配128小模型）时，
        // 回退策略（按优先级）：
        //   1) 通道迁移：临时加载256通道起点 → 逐层截取迁移到128（继承已训练棋力，
        //      快速ELO起步——用户目标"elo增长是唯一目标"的核心手段）；
        //   2) 同尺寸自举基线 gen0_h{hidden}.pt（上次运行自举保存的随机M冻结副本）；
        //   3) 随机初始化（保证"小模型开箱即用"）。
        bool start_loaded = false;
        if (config_.fresh_start) {
            // 彻底从头：起点模型已塌缩（梯度消失）时唯一出路——随机初始化M，
            // 不继承起点棋力，也不回退到同尺寸自举基线（gen0_h*.pt可能同为塌缩模型）
            std::cout << "[EloGated] 彻底从头：随机初始化M（跳过起点模型与自举基线）"
                      << std::endl;
        } else if (!config_.start_checkpoint.empty() && std::filesystem::exists(config_.start_checkpoint)) {
            try {
                std::cout << "[EloGated] 加载起点模型: " << config_.start_checkpoint << std::endl;
                (*model_)->load(config_.start_checkpoint);
                start_loaded = true;
            } catch (const std::exception& e) {
                std::cerr << "[EloGated] 起点模型加载失败（可能通道数不匹配）: "
                          << e.what() << std::endl;
                // 通道迁移尝试（双向）：从候选宽度加载源 → 适配迁移到目标宽度。
                // 覆盖"256起点→128小模型"(截取)与"128起点→256大模型"(扩展)两种情况
                int target_w = config_.hidden_dim;
                bool migrated = false;
                for (int cand_w : {128, 256}) {
                    if (cand_w == target_w) continue;
                    auto src_tmp = try_load_checkpoint(
                        cand_w, config_.num_experts, config_.board_size,
                        config_.dropout_rate, config_.start_checkpoint);
                    if (!src_tmp) continue;
                    if (migrate_model_channels(src_tmp, model_)) {
                        std::cout << "[EloGated] 通道迁移成功: " << cand_w
                                  << "通道 → " << target_w << "通道" << std::endl;
                        start_loaded = true;
                        migrated = true;
                        break;
                    }
                }
                if (!migrated) {
                    std::cerr << "[EloGated] 通道迁移未生效（无匹配宽度或参数不匹配），继续回退" << std::endl;
                }
            }
        } else if (config_.start_checkpoint.empty()) {
            std::cout << "[EloGated] 未指定起点模型路径" << std::endl;
        } else {
            std::cout << "[EloGated] 起点模型文件不存在: " << config_.start_checkpoint << std::endl;
        }
        if (!start_loaded && !config_.fresh_start) {
            std::string fallback = config_.checkpoint_dir + "/gen0_h" +
                                   std::to_string(config_.hidden_dim) + ".pt";
            if (std::filesystem::exists(fallback)) {
                try {
                    std::cout << "[EloGated] 回退起点: 加载同尺寸自举基线 " << fallback << std::endl;
                    (*model_)->load(fallback);
                    start_loaded = true;
                } catch (const std::exception& e) {
                    std::cerr << "[EloGated] 回退起点加载失败: " << e.what() << std::endl;
                }
            } else {
                std::cout << "[EloGated] 未找到同尺寸回退起点 " << fallback
                          << "，使用随机初始化" << std::endl;
            }
        }

        // Gen0固定基线：通道不匹配时同样尝试通道迁移；失败才"自举"冻结当前M。
        bool gen0_loaded = false;
        if (!config_.gen0_checkpoint.empty() && std::filesystem::exists(config_.gen0_checkpoint)) {
            try {
                std::cout << "[EloGated] 加载Gen0基线: " << config_.gen0_checkpoint << std::endl;
                (*gen0_)->load(config_.gen0_checkpoint);
                gen0_loaded = true;
            } catch (const std::exception& e) {
                std::cerr << "[EloGated] Gen0基线加载失败（可能通道数不匹配）: "
                          << e.what() << std::endl;
                // 通道迁移尝试（双向）：从候选宽度加载源 → 适配迁移到Gen0目标宽度。
                // 覆盖"256基线→128"(截取)与"128基线→256大模型"(扩展)两种情况
                int target_w = config_.gen0_hidden_dim;
                bool migrated = false;
                for (int cand_w : {128, 256}) {
                    if (cand_w == target_w) continue;
                    auto src_tmp = try_load_checkpoint(
                        cand_w, config_.num_experts, config_.board_size,
                        config_.dropout_rate, config_.gen0_checkpoint);
                    if (!src_tmp) continue;
                    if (migrate_model_channels(src_tmp, gen0_)) {
                        std::cout << "[EloGated] Gen0通道迁移成功: " << cand_w
                                  << "通道 → " << target_w << "通道" << std::endl;
                        gen0_loaded = true;
                        migrated = true;
                        break;
                    }
                }
                if (!migrated) {
                    std::cerr << "[EloGated] Gen0通道迁移未生效，回退自举" << std::endl;
                }
            }
        }
        if (!gen0_loaded) {
            // NoGradGuard + .data()：copy_是对requires_grad叶子张量的就地操作，走autograd会抛
            // "leaf Variable requires grad in-place"，需绕过autograd直接拷贝参数数据。
            torch::NoGradGuard no_grad;
            auto m_params = (*model_)->parameters();
            auto g_params = (*gen0_)->parameters();
            for (size_t i = 0; i < m_params.size() && i < g_params.size(); ++i) {
                g_params[i].data().copy_(m_params[i].data());
            }
            std::cout << "[EloGated] 未找到兼容Gen0基线，已自举：冻结当前M为基线（相对起点测ELO）" << std::endl;
            // 保存自举基线到按hidden_dim区分的路径，避免覆盖256通道的diploid基线
            std::string small_gen0 = config_.checkpoint_dir + "/gen0_h" +
                                     std::to_string(config_.hidden_dim) + ".pt";
            try { (*gen0_)->save(small_gen0); } catch (...) {}
        }

        // 设备迁移与模式设置（Gen0冻结，M在训练阶段切换train/eval）
        (*model_)->to(device_);
        (*gen0_)->to(device_);
        (*model_)->set_eval_mode();
        (*gen0_)->set_eval_mode();

        // 初始化多元评价器与GAE计算器（即时奖励 = 多元评价体系多拿分）
        multi_dim_evaluator_ = std::make_unique<MultiDimEvaluator>();
        gae_calculator_ = std::make_unique<GAECalculator>();

        // 初始化两个搜索器（M与Gen0各自绑定模型推理回调）
        init_searchers();

        // 初始化经验池（环形，容量8000，容纳约30局数据）
        replay_ = std::make_unique<EfficientReplayBuffer>(config_.replay_capacity, config_.num_experts);

        // 初始化训练器（高学习率0.03，禁用AMP——1660 Ti上FP16训练破坏模型的教训）
        TrainingConfig tconfig;
        tconfig.batch_size = config_.batch_size;
        tconfig.learning_rate = config_.learning_rate;
        tconfig.use_mixed_precision = false;
        trainer_ = std::make_unique<MixedPrecisionTrainer>(model_, tconfig);
        trainer_->setup_optimizer();
        // 对手风格条件化训练：CA-MOE专家分化的唯一有效信号（用户指定启用）。
        // 机制：train_step每步以50%概率注入采样对手风格（OpponentStyleProfiler原型池+噪声）
        //   → 门控学"风格→专家"映射，style_projection层获得梯度；另50%不注入（条件dropout）
        //   → 保证评估/对弈等"无条件"场景下模型仍正常发挥（评估走的就是这条无分支）。
        // 为何必须开：门控均匀正则(KL(pred‖uniform)+负载均衡)天然把路由推向"全局均匀"，
        //   而无条件训练下"按局面分化"带不来策略收益→门控收敛到常数路由（实测batch标准差
        //   恒0、8专家退化为固定5:3分裂、门控fc1占总参数76%却不再更新）。历史上熵正则/
        //   自蒸馏/负载均衡/温度/logits-L2五种方案均治不住，根因就是缺这个条件信号。
        trainer_->set_opponent_style_conditioning(true);

        // 高学习率风险提示：实测0.03门控L2正反馈爆炸、0.01策略损失飙至17.8，
        // 0.005+log_probs clip[-6,0] 为稳定组合；用户若显式传更大lr则提醒，
        // 避免"学坏→ELO全回滚→看似没涨"的假象（快速迭代目标下稳定优先）。
        if (config_.learning_rate > 0.01f) {
            std::cerr << "[EloGated] 警告: 学习率=" << config_.learning_rate
                      << " 大于稳定上限0.01（实测0.005稳定，0.03爆炸）。"
                      << "若ELO持续不涨请改 --elo-lr=0.005" << std::endl;
        }

        std::cout << "[EloGated] 初始化完成 | 设备="
                  << (device_.is_cuda() ? "CUDA" : "CPU")
                  << " | 学习率=" << config_.learning_rate
                  << " | 温度=" << config_.temperature
                  << " | 模拟=" << config_.simulations_per_move << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[EloGated] 初始化失败: " << e.what() << std::endl;
        return false;
    }
}

// ==================== 搜索器初始化（M与Gen0各一个） ====================

void EloGatedTrainer::init_searchers() {
    auto build_searcher = [this](std::shared_ptr<neural::CA_MOEExpertSystem> model,
                                 std::unique_ptr<search::HMGSDP_Optimized>& out_searcher) {
        out_searcher = std::make_unique<search::HMGSDP_Optimized>();
        out_searcher->initialize("");

        // 单线程搜索 + 批量推理（batch=128减少GPU内核启动次数）
        search::HMGSDP_Optimized::ParallelSearchConfig ps_config;
        ps_config.num_workers = 1;
        ps_config.batch_size = 128;
        ps_config.use_batch_inference = false;
        out_searcher->set_parallel_config(ps_config);

        // 策略先验回调：模型单次前向，同时缓存价值供value_cb命中
        auto policy_cb = [this, model](const GoBoardCPU& board, int current_player) {
            try {
                std::vector<float> raw(17 * 19 * 19, 0.0f);
                board.encode_to_tensor(raw.data(), 17);
                auto state = torch::from_blob(raw.data(), {1, 17, 19, 19}, torch::kFloat32).clone();
                if (device_.is_cuda()) state = state.to(device_);
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
                if (sum > 1e-8f) {
                    for (auto& p : out) p /= sum;
                }
                return out;
            } catch (...) {
                return std::vector<float>(362, 1.0f / 362.0f);
            }
        };

        // 价值回调：当前玩家视角 [-1,1]
        auto value_cb = [this, model](const GoBoardCPU& board, int current_player) {
            try {
                std::vector<float> raw(17 * 19 * 19, 0.0f);
                board.encode_to_tensor(raw.data(), 17);
                auto state = torch::from_blob(raw.data(), {1, 17, 19, 19}, torch::kFloat32).clone();
                if (device_.is_cuda()) state = state.to(device_);
                torch::NoGradGuard no_grad;
                auto result = locked_forward(model, state);
                float value = result.final_value.to(torch::kCPU).item<float>();
                // 模型价值为黑棋视角；白方行棋时取反，转成搜索需要的行棋方视角。
                // 原实现写 current_player == -1（WHITE实际是2，永不成立）→ 白方价值方向错误。
                if (is_white_player(current_player)) value = -value;
                return std::clamp(value, -1.0f, 1.0f);
            } catch (...) {
                return 0.0f;
            }
        };

        // 批量评估回调：一次前向评估一批叶子节点（策略+价值），提升GPU利用率
        auto batch_cb = [this, model](const std::vector<GoBoardCPU>& boards,
                                      const std::vector<int>& players) {
            std::vector<std::pair<std::vector<float>, float>> results;
            results.reserve(boards.size());
            if (boards.empty()) return results;
            try {
                std::vector<float> raw(boards.size() * 17 * 19 * 19, 0.0f);
                for (size_t i = 0; i < boards.size(); ++i) {
                    boards[i].encode_to_tensor(raw.data() + i * 17 * 19 * 19, 17);
                }
                auto state = torch::from_blob(
                    raw.data(), {static_cast<long>(boards.size()), 17, 19, 19},
                    torch::kFloat32).clone();
                if (device_.is_cuda()) state = state.to(device_);
                torch::NoGradGuard no_grad;
                auto result = locked_forward(model, state);
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
                    if (sum > 1e-8f) {
                        for (auto& v : out) v /= sum;
                    }
                    float val = value_ptr[i];
                    // 黑棋视角 → 行棋方视角（与单步value_cb同一约定）
                    if (is_white_player(players[i])) val = -val;
                    results.emplace_back(std::move(out), std::clamp(val, -1.0f, 1.0f));
                }
            } catch (...) {
                for (size_t i = 0; i < boards.size(); ++i) {
                    results.emplace_back(std::vector<float>(362, 1.0f / 362.0f), 0.0f);
                }
            }
            return results;
        };

        out_searcher->set_batch_evaluator(batch_cb);
        out_searcher->set_neural_evaluator(policy_cb, value_cb);
    };

    build_searcher(model_, searcher_m_);
    build_searcher(gen0_, searcher_g_);
    std::cout << "[EloGated] M与Gen0搜索器初始化完成 (模拟数="
              << config_.simulations_per_move << ")" << std::endl;
}

// ==================== 搜索策略封装 ====================

std::pair<std::vector<float>, float> EloGatedTrainer::search_policy(
    search::HMGSDP_Optimized* searcher, const std::string& board_state) {
    std::vector<float> uniform(362, 1.0f / 362.0f);
    if (!searcher) return {uniform, 0.0f};
    try {
        searcher->search(board_state, config_.simulations_per_move, 1.414);
        auto policy = searcher->get_root_policy(config_.board_size);
        float root_value = static_cast<float>(searcher->get_root_value());
        float sum = 0.0f;
        for (auto& p : policy) sum += p;
        if (sum < 1e-6f) return {uniform, root_value};
        return {policy, root_value};
    } catch (...) {
        // catch(...)而非catch(std::exception)：实测搜索器内部可能抛出非std异常
        //   （torch/cuda层），std::exception捕获不住会导致进程崩溃（0xE06D7363）
        std::cerr << "[EloGated] 搜索失败（异常），使用均匀分布" << std::endl;
        return {uniform, 0.0f};
    }
}

// ==================== 一局对战（M vs Gen0） ====================
// model_first=true: M执黑；false: M执白
// collect=true: 记录M每步训练数据（状态/搜索分布/门控/多元评分奖励）
// 返回值：M视角胜负 (+1胜 / -1负 / 0平)
// 训练对弈（collect=true）启用终局引导pass（move_num>250抬升pass权重），
// 评估（collect=false）禁用引导，让搜索自然决策，保证评估反映真实棋力。

float EloGatedTrainer::play_game_vs_gen0(int game_id, bool model_first,
                                         float temperature, bool collect,
                                         GameData* out_game) {
    GoBoardCPU board;
    int move_num = 0;
    bool game_ended = false;
    int consecutive_passes = 0;
    int win_hold_steps = 0;
    int policy_exec_total = 0;
    int policy_exec_hits = 0;

    GameData game;
    if (out_game) {
        game.game_id = game_id;
        game.start_time = std::chrono::system_clock::now();
    }

    while (!game_ended && move_num < config_.max_game_length) {
        StoneColor player = board.get_current_player();
        bool is_black = (player == StoneColor::BLACK);
        bool is_model_turn = (is_black == model_first);

        // 1. 编码当前状态
        torch::Tensor current_state = torch::empty({17, 19, 19});
        board.encode_to_tensor(current_state.data_ptr<float>(), 17);
        if (device_.is_cuda()) current_state = current_state.to(device_);

        // 2. HMGS-DP搜索生成访问次数分布（训练目标）与根价值
        torch::NoGradGuard no_grad;
        std::string board_str = board.to_string();
        auto* searcher = is_model_turn ? searcher_m_.get() : searcher_g_.get();
        auto [search_policy_vec, search_root_value] = search_policy(searcher, board_str);

        torch::Tensor search_policy_tensor = torch::from_blob(
            const_cast<float*>(search_policy_vec.data()),
            {1, 362}, torch::kFloat32).clone();
        torch::Tensor mcts_policy = search_policy_tensor.squeeze(0);

        // 3. 动作选择：M用高温度+前30步Dirichlet噪声探索，Gen0用自身温度
        torch::Tensor selection_policy = mcts_policy;
        float use_temp = is_model_turn ? temperature : config_.gen0_temperature;
        if (is_model_turn && move_num < 30) {
            selection_policy = add_dirichlet_noise(mcts_policy, rng_);
        }

        // 终局引导（训练与评估共用）：基于"剩余空点"驱动收束，配合连续pass终局结算。
        // 关键：只有官子接近收完（空点少）才抬升pass权重——空点多时绝不引导pass，
        // 让对局自然进行到官子尾声。原实现"move_num>250无条件抬0.5~0.9"在贪心评估下
        // 过早截断（255步官子未收完→计分失真→M=Gen34强模型评估仅50%）。
        // 空点驱动后：评估可用贪心(0.01)获得可复现、低噪声的ELO，严格门控才能生效。
        if (move_num > 300) {
            selection_policy = selection_policy.clone();
            selection_policy[361] = 1.0f;  // 超过300步兜底强制pass（防死循环/400截断）
        } else if (move_num > 250) {
            int empty_count = 0;
            for (int i = 0; i < 361; ++i) {
                auto [x, y] = GoBoardCPU::index_to_xy(i);
                if (board.is_empty(x, y)) empty_count++;
            }
            // 空点≤30（官子尾声）→ pass权重0.9；≤60 → 0.7；空点更多不抬（自然进行）
            float pass_min = 0.0f;
            if (empty_count <= 30) {
                pass_min = 0.9f;
            } else if (empty_count <= 60) {
                pass_min = 0.7f;
            }
            if (pass_min > 0.0f) {
                selection_policy = selection_policy.clone();
                selection_policy[361] = std::max(selection_policy[361].item<float>(), pass_min);
            }
        }

        std::vector<bool> legal_mask = build_legal_mask(board);
        int selected_action = select_action(selection_policy, use_temp, legal_mask, rng_);

        // 4. M步记录训练数据（多元评价在落子前评估，落子后计算即时奖励）
        MultiDimEvaluation eval_before;
        if (is_model_turn && out_game) {
            eval_before = multi_dim_evaluator_->evaluate(board, player);
        }

        // 5. 执行动作
        if (selected_action == 361) {
            consecutive_passes++;
            if (consecutive_passes >= 2) game_ended = true;
            board.set_current_player(opponent_color(player));
            board.set_move_count(board.get_move_count() + 1);
        } else {
            consecutive_passes = 0;
            auto [action_x, action_y] = GoBoardCPU::index_to_xy(selected_action);
            MoveResult move_result = board.make_move(action_x, action_y, player);
            if (!move_result.success) {
                std::cerr << "[EloGated] 无效动作 " << selected_action
                          << "（legal_mask过滤后仍失败），游戏" << game_id << std::endl;
                game_ended = true;
                break;
            }
        }

        // 6. M步记录（落子后评估己方局面，计算多元评分即时奖励）
        if (is_model_turn && out_game) {
            MultiDimEvaluation eval_after;
            if (config_.enable_multi_dim_eval) {
                eval_after = multi_dim_evaluator_->evaluate(board, player);
            }

            SelfPlayStep step;
            step.state = current_state.cpu();
            step.policy_target = normalize_policy(mcts_policy).cpu();  // 纯搜索访问分布
            step.root_value = search_root_value;
            // 门控权重与专家索引：对当前局面推理获取（训练门控网络）
            {
                torch::Tensor gating_weights;
                try {
                    auto batch_state = current_state.unsqueeze(0);
                    auto result = locked_forward(model_, batch_state);
                    gating_weights = result.gating_weights.to(torch::kCPU).squeeze(0);
                } catch (...) {
                    int num_experts = (*model_)->get_num_experts();
                    gating_weights = torch::ones({num_experts}) / static_cast<float>(num_experts);
                }
                step.gating_weights = gating_weights;
                step.expert_indices = get_top_k_experts(gating_weights, 3);
            }
            step.action_taken = selected_action;
            step.temperature_at_move = use_temp;
            step.move_number = move_num;
            step.value_target = 0.0f;
            step.multi_dim_eval = eval_after;

            // ===== 即时奖励：多元评价体系下多拿分（用户指定的经典AI招式） =====
            // 1) 评分奖励：综合评分与胜率估计偏离均势越大，奖励越多
            float composite_after = eval_after.composite_score;  // [0,1]，0.5=均势
            float win_rate_after = eval_after.metrics[(int)EvalDimension::WIN_RATE_ESTIMATE].score;
            float perf_score = (composite_after - 0.5f) * 0.4f + (win_rate_after - 0.5f) * 0.4f;
            float score_reward = config_.enable_multi_dim_eval ?
                std::max(-0.2f, std::min(0.2f, perf_score)) : 0.0f;  // 裁剪防爆炸
            // 2) 基础奖励：综合评分差 + 每步胜率变化（MultiDimEvaluator内合成）
            float base_reward = config_.enable_multi_dim_eval ?
                multi_dim_evaluator_->compute_step_reward(eval_before, eval_after, player) : 0.0f;
            // 3) 胜率保持时间奖励：连续保持优势的步数越多奖励越高（稳守优势）
            if (win_rate_after > 0.55f) {
                win_hold_steps++;
            } else {
                win_hold_steps = 0;
            }
            float hold_reward = std::min(win_hold_steps, 20) * 0.005f;
            // 4) 策略执行率奖励：实际落子=搜索访问分布top-1（模型说到做到）
            int top1_action = static_cast<int>(mcts_policy.argmax().item<int>());
            bool exec_hit = (selected_action == top1_action);
            policy_exec_total++;
            if (exec_hit) policy_exec_hits++;
            float exec_reward = exec_hit ? 0.02f : 0.0f;

            step.step_reward = score_reward + base_reward + hold_reward + exec_reward;
            step.has_gae_signal = false;
            step.gae_advantage = 0.0f;
            step.td_return = 0.0f;
            game.steps.push_back(std::move(step));
        }

        // 7. 无合法落子检查（双方均无合法着法时终局）
        if (!game_ended) {
            auto current_legal = board.get_legal_moves(board.get_current_player());
            if (current_legal.empty()) {
                auto opp_legal = board.get_legal_moves(opponent_color(board.get_current_player()));
                if (opp_legal.empty()) game_ended = true;
            }
        }

        move_num++;
    }

    // 终局结算：中国规则面积计点（棋子+领地+黑贴kKomi目），黑棋视角
    auto [black_final, white_final] = board.compute_area_score(kKomi);
    float outcome = 0.0f;
    if (black_final > white_final) outcome = 1.0f;
    else if (white_final > black_final) outcome = -1.0f;

    if (out_game) {
        game.outcome = outcome;
        game.game_length = move_num;
        game.end_time = std::chrono::system_clock::now();
        *out_game = std::move(game);
    }

    // 返回M视角胜负（M执黑则黑棋视角，M执白则取反）
    return model_first ? outcome : -outcome;
}

// ==================== M vs KataGo（KataGo作ELO评估标尺） ====================
// 与play_game_vs_gen0共用M侧决策逻辑（搜索+贪心选点+终局引导），但对手换成
// 外部KataGo进程（GTP管道）。评估专用：低温贪心、无Dirichlet探索、不记录训练数据。
// 棋盘始终在进程内维护（双方落子都应用）。
// 返回值 = 模型视角目差（正=模型赢N目，负=模型输N目）——KataGo是"标尺"，
//   衡量模型在KataGo手下能下成什么样：目差从-50缩到-20即进步，不要求赢。
float EloGatedTrainer::play_game_vs_katago(bool model_first, float temperature) {
    // 启动KataGo基准引擎（每次评估独立进程，用完即关，避免长驻占用显存/句柄）
    KataGoGTPClient kg;
    if (!kg.start(config_.katago_exe, config_.katago_model,
                  config_.katago_config, config_.katago_visits)) {
        std::cerr << "[EloGated] KataGo启动失败(" << config_.katago_exe
                  << ")，评估计平局" << std::endl;
        return 0.0f;
    }
    // GTP 握手 + 对局初始化
    kg.send("protocol_version");
    kg.send("boardsize 19");
    kg.send("komi " + std::to_string(kKomi));
    kg.send("clear_board");

    GoBoardCPU board;
    int move_num = 0;
    bool game_ended = false;
    int consecutive_passes = 0;
    int player_mismatch_count = 0;  // 诊断：搜索往返后行棋方不一致的步数（修复前白方回合恒不一致）

    while (!game_ended && move_num < config_.max_game_length) {
        StoneColor player = board.get_current_player();
        bool is_black = (player == StoneColor::BLACK);
        bool is_model_turn = (is_black == model_first);
        std::string color_name = is_black ? "black" : "white";

        if (is_model_turn) {
            // ---- M侧：搜索 + 贪心选点（与评估路径一致：低温、无Dirichlet） ----
            torch::Tensor current_state = torch::empty({17, 19, 19});
            board.encode_to_tensor(current_state.data_ptr<float>(), 17);
            if (device_.is_cuda()) current_state = current_state.to(device_);

            torch::NoGradGuard no_grad;
            std::string board_str = board.to_string();
            // 诊断：往返行棋方一致性。搜索器内部用 GoBoardCPU::from_string() 重建局面，
            //   修复前该往返丢失行棋方（恒为黑方）→ 白方回合100%不一致，
            //   模型拿到的是"轮到黑走"的局面特征，选点等于乱下。
            if (GoBoardCPU::from_string(board_str).get_current_player() !=
                board.get_current_player()) {
                player_mismatch_count++;
            }
            auto [search_policy_vec, search_root_value] =
                search_policy(searcher_m_.get(), board_str);
            (void)search_root_value;

            torch::Tensor mcts_policy = torch::from_blob(
                const_cast<float*>(search_policy_vec.data()),
                {1, 362}, torch::kFloat32).clone().squeeze(0);

            torch::Tensor selection_policy = mcts_policy;
            // 终局引导（与训练/自对弈评估一致）：空点驱动的pass提升 + 超长兜底
            if (move_num > 300) {
                selection_policy = selection_policy.clone();
                selection_policy[361] = 1.0f;
            } else if (move_num > 250) {
                int empty_count = 0;
                for (int i = 0; i < 361; ++i) {
                    auto [x, y] = GoBoardCPU::index_to_xy(i);
                    if (board.is_empty(x, y)) empty_count++;
                }
                float pass_min = 0.0f;
                if (empty_count <= 30) pass_min = 0.9f;
                else if (empty_count <= 60) pass_min = 0.7f;
                if (pass_min > 0.0f) {
                    selection_policy = selection_policy.clone();
                    selection_policy[361] = std::max(
                        selection_policy[361].item<float>(), pass_min);
                }
            }

            std::vector<bool> legal_mask = build_legal_mask(board);
            int selected_action = select_action(selection_policy, temperature,
                                                legal_mask, rng_);

            // 应用M落子到进程内棋盘，并同步给KataGo
            if (selected_action == 361) {
                consecutive_passes++;
                if (consecutive_passes >= 2) game_ended = true;
                board.set_current_player(opponent_color(player));
                board.set_move_count(board.get_move_count() + 1);
                kg.send("play " + color_name + " pass");
            } else {
                consecutive_passes = 0;
                auto [action_x, action_y] = GoBoardCPU::index_to_xy(selected_action);
                MoveResult move_result = board.make_move(action_x, action_y, player);
                if (!move_result.success) {
                    std::cerr << "[EloGated] M vs KataGo: M返回非法动作 "
                              << selected_action << "，评估计大负" << std::endl;
                    kg.send("quit");
                    return -64.0f;  // 大负目差（模型棋力异常）
                }
                kg.send("play " + color_name + " " + idx_to_gtp(selected_action));
            }
        } else {
            // ---- KataGo侧：GTP genmove ----
            std::string resp = kg.send("genmove " + color_name);
            // 解析响应首行 "= <move>" 或 "? <err>"
            std::string mv;
            if (!resp.empty() && resp[0] == '=') {
                mv = resp.substr(1);
                // 去掉前导空白与可能的尾部内容（取第一个token）
                size_t p = mv.find_first_not_of(" \t");
                if (p != std::string::npos) mv = mv.substr(p);
                size_t e = mv.find_first_of(" \t");
                if (e != std::string::npos) mv = mv.substr(0, e);
            }
            if (mv.empty() || mv == "resign") {
                // 异常/认输：直接判模型大胜（目差取一个明显大于正常对局的值）
                kg.send("quit");
                return model_first ? 64.0f : -64.0f;
            }
            // 应用KataGo落子到进程内棋盘
            if (mv == "pass") {
                consecutive_passes++;
                if (consecutive_passes >= 2) game_ended = true;
                board.set_current_player(opponent_color(player));
                board.set_move_count(board.get_move_count() + 1);
            } else {
                consecutive_passes = 0;
                int idx = gtp_to_idx(mv);
                if (idx < 0) {
                    std::cerr << "[EloGated] KataGo返回非法着法: " << mv
                              << "，评估计大负" << std::endl;
                    kg.send("quit");
                    return -64.0f;
                }
                auto [x, y] = GoBoardCPU::index_to_xy(idx);
                MoveResult move_result = board.make_move(x, y, player);
                if (!move_result.success) {
                    std::cerr << "[EloGated] KataGo着法(" << mv
                              << ")无法落子，评估计大负" << std::endl;
                    kg.send("quit");
                    return -64.0f;
                }
            }
        }

        // 无合法落子检查（双方都无合法着法时终局）
        if (!game_ended) {
            auto current_legal = board.get_legal_moves(board.get_current_player());
            if (current_legal.empty()) {
                auto opp_legal = board.get_legal_moves(
                    opponent_color(board.get_current_player()));
                if (opp_legal.empty()) game_ended = true;
            }
        }

        move_num++;
    }

    // 终局结算：优先用KataGo的final_score（400 visits的目差判断比进程内简化
    // 面积计分更准——死活/劫的处理更完整）；KataGo异常时回退进程内面积计分。
    float margin = 0.0f;  // 黑棋视角目差（正=黑赢N目）
    std::string fs = kg.send("final_score");
    if (!fs.empty() && fs[0] == '=') {
        std::string body = fs.substr(1);
        size_t p = body.find_first_not_of(" \t");
        if (p != std::string::npos) body = body.substr(p);
        if (body == "0" || body.empty()) {
            margin = 0.0f;
        } else if (!body.empty() && (body[0] == 'B' || body[0] == 'W')) {
            try {
                float m = std::stof(body.substr(2));
                margin = (body[0] == 'B') ? m : -m;
            } catch (...) {
                margin = 0.0f;
            }
        }
    }
    if (fs.empty() || fs[0] != '=') {
        // 回退：进程内中国规则面积计分（黑贴kKomi目，与KataGo路径同语义）
        auto [black_final, white_final] = board.compute_area_score(kKomi);
        margin = black_final - white_final;
    }
    // 诊断（定位"每轮目差恒定、上升曲线不真实"）：终局手数 + KataGo原始返回 +
    //   双方朴素面积（不含贴目）+ 盘上棋子数。
    //   饱和判据：手数已近终局、盘上仍有己方棋子，但己方面积≈0（全被吃/判死）→
    //   目差钉在地板（19路理论最差约-361），指标对棋力无分辨力，门控只能拿1目级
    //   噪声做接受/回滚决策——这正是"累积ELO在涨、真实棋力没动"的根因。
    {
        auto [raw_b, raw_w] = board.compute_area_score(0.0f);
        size_t m_stones = model_first
            ? board.get_all_stones(StoneColor::BLACK).size()
            : board.get_all_stones(StoneColor::WHITE).size();
        size_t o_stones = model_first
            ? board.get_all_stones(StoneColor::WHITE).size()
            : board.get_all_stones(StoneColor::BLACK).size();
        std::cout << "  [评估诊断] 终局手数=" << move_num
                  << " | KataGo返回=\"" << (fs.empty() ? std::string("(空)") : fs) << "\""
                  << " | 朴素面积(无贴目)黑=" << raw_b << " 白=" << raw_w
                  << " | 盘上棋子 M=" << m_stones << " 对手=" << o_stones
                  << " | 往返行棋方不一致=" << player_mismatch_count << "/" << move_num
                  << " | 终局原因=" << (game_ended ? "双方pass/无着" : "达最大手数")
                  << std::endl;
    }
    kg.send("quit");

    // 返回M视角目差，并做贴目修正（关键修复·实测10轮发现）。
    // margin是黑方最终得分（已含贴目）：margin = 原始目差 - kKomi。
    // 直接按执色取反会残留"±kKomi"的执色偏移——同一棋力执黑比执白天然低2×kKomi=15目：
    //   M执黑：返回 = (原始目差 - komi) = -W - 7.5
    //   M执白：返回 = -(原始目差 - komi) = -W + 7.5      （W = M不含贴目的原始落后目数）
    // 而门控用单一best_elo_跨执色比较（elo > best_elo_），于是白轮一旦设了best，
    //   后续黑轮天然差15目、永不可能超过——实测10轮中4个黑轮全部结构性回滚，
    //   2次"接受"里1次（-353.5）实为假接受（修正后-361比基线-357更差）。
    // 修正：加回贴目，得到"不含贴目的原始目差"N，使跨执色可比：
    //   N = 返回原值 + (M执黑 ? +kKomi : -kKomi)
    // 实测修正后黑白均值差由14.42目降到0.58目（≈0），验证该偏移正是贴目重复计入。
    float raw_margin = model_first ? margin : -margin;
    return raw_margin + (model_first ? kKomi : -kKomi);
}

// ==================== KataGo自我对弈棋谱（蒸馏教师数据源） ====================
// 两个KataGo客户端（黑/白）对弈产生高质量棋谱：每个局面的落子都是教师标签。
// 与"学评估对弈局"的本质区别——局面分布干净（双方都是强棋、正常对局进展），
// 模型学到的是"好局面下怎么下好棋"，而不是"怎么收拾自己乱下造成的烂摊子"。
// 返回GameData：steps=双方全部落子样本（state+one-hot标签），outcome=黑视角胜负。
GameData EloGatedTrainer::generate_katago_selfplay_game(int game_id) {
    GameData game;
    game.game_id = game_id;
    game.outcome = 0.0f;
    game.game_length = 0;
    game.start_time = std::chrono::system_clock::now();

    KataGoGTPClient kg_black, kg_white;
    if (!kg_black.start(config_.katago_exe, config_.katago_model,
                        config_.katago_config, config_.katago_teacher_visits) ||
        !kg_white.start(config_.katago_exe, config_.katago_model,
                        config_.katago_config, config_.katago_teacher_visits)) {
        std::cerr << "[EloGated] 教师自对弈：KataGo启动失败，本局跳过" << std::endl;
        return game;
    }
    // 两个引擎各自维护棋盘（通过互相 play 保持同步）
    for (KataGoGTPClient* p : {&kg_black, &kg_white}) {
        p->send("protocol_version");
        p->send("boardsize 19");
        p->send("komi " + std::to_string(kKomi));
        p->send("clear_board");
    }

    GoBoardCPU board;
    int move_num = 0;
    int consecutive_passes = 0;
    bool game_ended = false;
    bool abort = false;
    // 蒸馏样本门控/专家字段仅作形状占位——compute_losses的门控损失只作用于
    // 模型自身预测（无target gating），故无需逐步前向获取（省1200次/轮的推理）
    const int num_experts = (*model_)->get_num_experts();

    while (!game_ended && move_num < config_.max_game_length && !abort) {
        StoneColor player = board.get_current_player();
        bool black_turn = (player == StoneColor::BLACK);
        KataGoGTPClient& eng = black_turn ? kg_black : kg_white;
        KataGoGTPClient& opp = black_turn ? kg_white : kg_black;
        std::string color_name = black_turn ? "black" : "white";

        // 当前行棋方genmove
        std::string resp = eng.send("genmove " + color_name);
        std::string mv;
        if (!resp.empty() && resp[0] == '=') {
            mv = resp.substr(1);
            size_t p = mv.find_first_not_of(" \t");
            if (p != std::string::npos) mv = mv.substr(p);
            size_t e = mv.find_first_of(" \t");
            if (e != std::string::npos) mv = mv.substr(0, e);
        }
        if (mv.empty() || mv == "resign") {
            abort = true;  // 异常/认输：本局样本作废（不学习半局棋）
            break;
        }
        int action = (mv == "pass") ? 361 : gtp_to_idx(mv);
        if (action < 0) {
            abort = true;
            break;
        }

        // 教师样本：落子前的局面（行棋方=当前player）+ one-hot标签
        {
            torch::Tensor st = torch::empty({17, 19, 19});
            board.encode_to_tensor(st.data_ptr<float>(), 17);
            SelfPlayStep step;
            step.state = st;
            step.policy_target = torch::zeros({362});
            step.policy_target[action] = 1.0f;
            step.root_value = 0.0f;
            step.action_taken = action;
            step.temperature_at_move = 0.01f;
            step.move_number = move_num;
            step.value_target = 0.0f;  // 终局统一填充
            step.step_reward = 0.0f;
            step.has_gae_signal = false;
            step.gae_advantage = 0.0f;
            step.td_return = 0.0f;
            step.gating_weights = torch::ones({num_experts}) / static_cast<float>(num_experts);
            step.expert_indices = {0, 1, 2};
            game.steps.push_back(std::move(step));
        }

        // 应用落子 + 同步给对手引擎
        if (mv == "pass") {
            consecutive_passes++;
            if (consecutive_passes >= 2) game_ended = true;
            board.set_current_player(opponent_color(player));
            board.set_move_count(board.get_move_count() + 1);
            opp.send("play " + color_name + " pass");
        } else {
            consecutive_passes = 0;
            auto [x, y] = GoBoardCPU::index_to_xy(action);
            if (!board.make_move(x, y, player).success) {
                abort = true;
                break;
            }
            opp.send("play " + color_name + " " + idx_to_gtp(action));
        }

        // 无合法落子检查（双方均无合法着法时终局）
        if (!game_ended) {
            if (board.get_legal_moves(board.get_current_player()).empty() &&
                board.get_legal_moves(opponent_color(board.get_current_player())).empty()) {
                game_ended = true;
            }
        }
        move_num++;
    }

    // 终局结算：KataGo的final_score（黑视角目差）→ 胜负符号
    float margin = 0.0f;
    if (!abort) {
        std::string fs = kg_black.send("final_score");
        if (!fs.empty() && fs[0] == '=') {
            std::string body = fs.substr(1);
            size_t p = body.find_first_not_of(" \t");
            if (p != std::string::npos) body = body.substr(p);
            if (!body.empty() && (body[0] == 'B' || body[0] == 'W')) {
                try {
                    float m = std::stof(body.substr(2));
                    margin = (body[0] == 'B') ? m : -m;
                } catch (...) {
                    margin = 0.0f;
                }
            }
        } else {
            // 回退：进程内面积计分（黑贴kKomi目）
            auto [bf, wf] = board.compute_area_score(kKomi);
            margin = bf - wf;
        }
    }
    kg_black.send("quit");
    kg_white.send("quit");

    if (abort) {
        game.steps.clear();  // 半局样本作废
        return game;
    }

    // 价值填充：全工程统一为黑棋视角，所有步都用同一个终局符号（不再按行棋方翻转）
    float sign = (margin > 1e-4f) ? 1.0f : ((margin < -1e-4f) ? -1.0f : 0.0f);  // 黑视角
    for (auto& s : game.steps) {
        s.value_target = sign;
    }
    game.outcome = sign;
    game.game_length = move_num;
    game.end_time = std::chrono::system_clock::now();
    return game;
}

// ==================== GAE计算（对局终局真实胜负注入） ====================
// 只对M的步计算（steps仅含M步）。终局值从M视角注入最后一步的TD bootstrap，
// 打破"价值头自洽复制搜索值"的循环（价值目标 = 真实胜负经过折扣向后传播）。

void EloGatedTrainer::compute_gae_for_game(GameData& game) {
    if (game.steps.empty()) return;

    std::vector<float> step_rewards;
    std::vector<float> step_values;
    step_rewards.reserve(game.steps.size());
    step_values.reserve(game.steps.size());

    for (auto& s : game.steps) {
        step_rewards.push_back(s.step_reward);
        step_values.push_back(s.root_value);
    }

    // 终局真实胜负注入（M视角）：game.outcome是黑棋视角，
    // 若最后一步是黑棋(M move_number为偶数)则直接取，白棋则取反
    int last_move_number = game.steps.back().move_number;
    float terminal_value = (last_move_number % 2 == 0) ? game.outcome : -game.outcome;

    auto gae_result = gae_calculator_->compute_gae(step_rewards, step_values, terminal_value);

    for (size_t i = 0; i < game.steps.size(); ++i) {
        auto& s = game.steps[i];
        s.gae_advantage = gae_result.advantages[i];
        s.td_return = gae_result.returns[i];
        // GAE回报是行棋方视角；价值目标统一转成黑棋视角（偶数手=黑）
        s.value_target = (s.move_number % 2 == 0) ? gae_result.returns[i]
                                                  : -gae_result.returns[i];
        s.has_gae_signal = true;
    }
}

// ==================== 一局学一次 ====================
// 把该局M的步数据打包为batch，按子batch（64）切分多次小步训练。
// 实测单步200样本梯度方差大（高lr下模型大跳变→学坏回滚）；
// 拆小步梯度更平滑（等效小lr）+ 数据利用更充分（局内多次更新）。

void EloGatedTrainer::train_on_game(const GameData& game) {
    if (game.steps.empty()) return;

    // 打包该局所有M步为batch张量
    size_t n = game.steps.size();
    std::vector<torch::Tensor> state_list;
    std::vector<torch::Tensor> policy_list;
    std::vector<torch::Tensor> gating_list;
    std::vector<torch::Tensor> value_list;
    std::vector<torch::Tensor> adv_list;
    std::vector<int> expert_list;   // 扁平拼接：每样本top-k专家索引依次排列（k=3）
    std::vector<float> importance(n, 1.0f);

    state_list.reserve(n);
    policy_list.reserve(n);
    gating_list.reserve(n);
    value_list.reserve(n);
    adv_list.reserve(n);

    for (const auto& s : game.steps) {
        state_list.push_back(s.state);
        policy_list.push_back(s.policy_target);
        gating_list.push_back(s.gating_weights);
        value_list.push_back(torch::tensor(s.value_target));
        adv_list.push_back(torch::tensor(s.gae_advantage));
        // 扁平拼接（与EfficientReplayBuffer::sample_batch的expert_indices格式一致）
        expert_list.insert(expert_list.end(), s.expert_indices.begin(), s.expert_indices.end());
    }

    // 堆叠为batch [N, ...]
    auto states = torch::stack(state_list);
    auto target_policies = torch::stack(policy_list);
    auto target_values = torch::stack(value_list);
    auto gating_weights = torch::stack(gating_list);
    auto advantages = torch::stack(adv_list);

    // 移入训练设备（M所在设备）
    if (device_.is_cuda()) {
        states = states.to(device_);
        target_policies = target_policies.to(device_);
        target_values = target_values.to(device_);
        gating_weights = gating_weights.to(device_);
        advantages = advantages.to(device_);
    }

    // 子batch切分：每批64样本（一局200步约3-4次小步更新）
    const int sub_batch = 64;
    const int k_experts_per_sample = 3;  // get_top_k_experts(gating, 3)的k
    (*model_)->set_train_mode();

    // 诊断：训练前后全体参数L2范数，直接判定优化器是否真正改动了权重。
    //   实测现象：同一教师棋谱连训8遍，策略损失卡在5.95（≈均匀分布）且逐遍完全
    //   相同——若Δ范数≈0即为"权重没动"（梯度为0或被裁剪吃光），而非"学习率太小"。
    //   用float64累加防float32下大模型平方和溢出，取sqrt得L2。
    auto param_l2_norm = [&]() {
        auto acc = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat64).device(device_));
        for (const auto& p : (*model_)->parameters()) {
            acc += p.detach().to(torch::kFloat64).pow(2).sum();
        }
        return torch::sqrt(acc).item<double>();
    };
    double l2_before = param_l2_norm();

    float sum_policy = 0.0f, sum_value = 0.0f, sum_gating = 0.0f;
    int steps = 0;
    for (size_t start = 0; start < n; start += sub_batch) {
        size_t end = std::min(start + static_cast<size_t>(sub_batch), n);

        EfficientReplayBuffer::TrainingBatch batch;
        batch.states = states.slice(0, static_cast<long>(start), static_cast<long>(end));
        batch.target_policies = target_policies.slice(0, static_cast<long>(start), static_cast<long>(end));
        batch.target_values = target_values.slice(0, static_cast<long>(start), static_cast<long>(end));
        batch.gating_weights = gating_weights.slice(0, static_cast<long>(start), static_cast<long>(end));
        batch.advantages = advantages.slice(0, static_cast<long>(start), static_cast<long>(end));
        batch.td_returns = batch.target_values;  // GAE后value_target=TD回报
        // expert_indices对应切片：扁平索引区间 [start*k, end*k)
        std::vector<int> sub_experts(
            expert_list.begin() + static_cast<long>(start * k_experts_per_sample),
            expert_list.begin() + static_cast<long>(end * k_experts_per_sample));
        batch.expert_indices = std::move(sub_experts);
        std::vector<float> sub_importance(
            importance.begin() + static_cast<long>(start),
            importance.begin() + static_cast<long>(end));
        batch.importance_weights = std::move(sub_importance);

        auto result = trainer_->train_step(batch);
        sum_policy += result.policy_loss;
        sum_value += result.value_loss;
        sum_gating += result.gating_loss;
        steps++;
    }
    // 诊断：训练后参数L2（与训练前对比，Δ≈0即权重未被更新）
    double l2_after = param_l2_norm();
    (*model_)->set_eval_mode();
    // 训练结束清除对手风格（train_step无条件分支已clear，此处双保险防残留）
    (*model_)->clear_opponent_style();

    // 训练后强制CUDA同步：train_step的backward由autograd引擎异步线程执行，
    //   若该线程抛出异常（如参数异常值），主线程catch(...)无法捕获会导致进程
    //   直接terminate（0xE06D7363）。同步点把autograd异常拉回主线程统一处理。
#ifdef USE_CUDA
    torch::cuda::synchronize();
#endif

    std::cout << "  学1局 | 步数=" << game.steps.size()
              << " | 策略=" << std::fixed << std::setprecision(4)
              << (steps > 0 ? sum_policy / steps : 0.0f)
              << " | 价值=" << (steps > 0 ? sum_value / steps : 0.0f)
              << " | 门控=" << (steps > 0 ? sum_gating / steps : 0.0f)
              << " | 总=" << ((steps > 0 ? (sum_policy + sum_value + sum_gating) / steps : 0.0f))
              << " | ΔL2=" << std::scientific << std::setprecision(3)
              << (l2_after - l2_before)
              << std::defaultfloat << std::endl;
}

// ==================== ELO评估（M vs Gen0，搜索300 sims） ====================
// 与训练同款搜索（训练-评估一致原则），贪心决策。返回ELO值。
// 温度选择：贪心(0.01)可复现、噪声低——评估随机性只来自搜索本身，配合
// 空点驱动收束（官子收完才pass），计分准确。温度0.5采样评估实测波动大
// （24局胜率41.7%~91.7%），严格门控下被单次高值卡死。

float EloGatedTrainer::evaluate_elo(int games) {
    // KataGo评估模式（用户指定）：以KataGo为标尺，每轮1局定胜负，
    // 返回"模型视角目差"（正=赢KataGo，负=输N目）——用目差缩小衡量进步，
    // 而非要求一把赢过KataGo。
    if (config_.use_katago_eval) {
        auto eval_start = std::chrono::high_resolution_clock::now();
        // 每轮交替先后手，消除先手优势（round_为成员：初始评估=0执黑）
        bool model_first = (round_ % 2 == 0);
        float margin = 0.0f;
        try {
            margin = play_game_vs_katago(model_first, 0.01f);
        } catch (...) {
            // 评估异常：计大负（保守，不误判进步）
            std::cerr << "  KataGo评估局异常，计大负" << std::endl;
            margin = -64.0f;
        }
        auto eval_end = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration<float>(eval_end - eval_start).count();
        std::cout << "  评估1局 vs KataGo(" << config_.katago_visits << " visits)"
                  << (model_first ? " 执黑" : " 执白")
                  << " | 原始目差(已贴目修正)=" << std::fixed << std::setprecision(1)
                  << (margin >= 0.0f ? "+" : "") << margin << "目"
                  << " | 耗时=" << std::fixed << std::setprecision(0)
                  << elapsed << "s" << std::endl;
        // 恢复输出流默认格式，防 fixed/precision 残留污染后续打印（如累积ELO变3e+03）
        std::cout << std::defaultfloat << std::setprecision(6);
        return margin;
    }

    int wins = 0, draws = 0;
    auto eval_start = std::chrono::high_resolution_clock::now();

    // 评估时双方贪心（温度≈0）：选择最高概率动作，评估结果可复现、噪声低
    float orig_gen0_temp = config_.gen0_temperature;
    config_.gen0_temperature = 0.01f;

    for (int g = 0; g < games; ++g) {
        // 交替先后手消除先手优势
        bool model_first = (g % 2 == 0);
        float outcome = 0.0f;  // 异常时计平局
        try {
            outcome = play_game_vs_gen0(g, model_first, 0.01f, false, nullptr);
        } catch (...) {
            // 单局异常不影响评估：计平局继续（catch(...)兜住一切异常防崩溃）
            std::cerr << "  评估第" << g << "局异常，计平局" << std::endl;
        }
        if (outcome > 0.5f) wins++;
        else if (outcome > -0.5f && outcome < 0.5f) draws++;
        // 诊断：本局GPU操作后检查CUDA错误（定位0xE06D7363崩溃——疑似异步CUDA错误）
#ifdef USE_CUDA
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "  [诊断] 评估第" << g << "局后CUDA错误: "
                      << cudaGetErrorString(err) << std::endl;
            cudaGetLastError();  // 清除错误标志，防止在后续同步点重复抛出
        }
#endif
        // 进度打印：评估60局约10分钟无输出，分块打印避免进程被误判卡死
        if ((g + 1) % 10 == 0 || g == games - 1) {
            std::cout << "  评估中 " << (g + 1) << "/" << games << " | M胜=" << wins
                      << " 平=" << draws << std::endl;
            // 评估中定期清显存：60局搜索树分配/释放的缓存块累积易触发CUDA异步错误
            //   （实测崩溃点在评估收尾），每10局主动释放一次
#ifdef USE_CUDA
            c10::cuda::CUDACachingAllocator::emptyCache();
#endif
        }
    }

    config_.gen0_temperature = orig_gen0_temp;

    float score = static_cast<float>(wins) + 0.5f * static_cast<float>(draws);
    float win_rate = score / static_cast<float>(games);
    // 标准ELO公式：-400*log10((1-p)/p)，p为对Gen0胜率
    float win_rate_clamped = std::clamp(win_rate, 0.01f, 0.99f);
    float elo = -400.0f * std::log10((1.0f - win_rate_clamped) / win_rate_clamped);

    auto eval_end = std::chrono::high_resolution_clock::now();
    float elapsed = std::chrono::duration<float>(eval_end - eval_start).count();

    std::cout << "  评估 " << games << " 局 | M胜=" << wins
              << " 平=" << draws
              << " | 胜率=" << std::fixed << std::setprecision(1) << (win_rate * 100.0f) << "%"
              << " | ELO=" << std::setprecision(1) << elo
              << " | 耗时=" << std::setprecision(0) << elapsed << "s"
              << std::defaultfloat << std::endl;
    return elo;
}

// ==================== 参数快照与回滚（ELO门控核心） ====================

void EloGatedTrainer::snapshot_params() {
    param_snapshot_.clear();
    // 快照留在GPU（回滚为GPU→GPU同设备拷贝）：
    //   实测CPU快照方案（clone().to(kCPU)）崩溃于restore_params（0xE06D7363）——
    //   跨设备异步拷贝在1660 Ti上配合训练/评估的CUDA缓存复用会读取非法内存。
    //   快照250MB占用GPU，换取回滚的稳定性。
    for (const auto& p : (*model_)->parameters()) {
        param_snapshot_.push_back(p.detach().clone());
    }
    has_snapshot_ = true;
}

void EloGatedTrainer::restore_params() {
    // 主方案：重新加载elo_best.pt实现回滚。
    //   语义保证：轮初模型恒等于best——只有"接受"才更新best并立即保存，
    //   未接受则回滚回到best。因此回滚 = 重新加载best，彻底绕开copy_快照
    //   在1660 Ti上多次实测崩溃（0xE06D7363，跨设备/同设备copy_均触发）。
    //   torch::save/load是标准序列化（二倍体训练一直在用），路径稳定。
    std::string best_path = config_.checkpoint_dir + "/elo_best.pt";
    if (std::filesystem::exists(best_path)) {
        try {
            (*model_)->load(best_path);
            (*model_)->to(device_);
            (*model_)->set_eval_mode();
            return;
        } catch (...) {
            std::cerr << "  [恢复] load(best)失败，回退GPU快照copy_兜底" << std::endl;
        }
    } else {
        // 兜底前提示（best文件缺失时回滚依赖GPU快照）
        std::cerr << "  [恢复] 警告: best模型缺失(" << best_path << ")，使用GPU快照回滚" << std::endl;
    }
    // 兜底：GPU快照copy_（每个参数独立try，异常只跳过该参数）
    // 关键修复：对requires_grad叶子参数做in-place copy必须包NoGradGuard，
    //   否则autograd检测到"leaf Variable requires grad in-place"抛异常导致
    //   全部参数恢复失败（实测run5回滚完全失效、坏参数累积、ELO只降不升）。
    //   用.data()显式绕过autograd视图，只改底层数据不改计算图。
    if (!has_snapshot_) return;
    torch::NoGradGuard no_grad;
    auto params = (*model_)->parameters();
    for (size_t i = 0; i < params.size() && i < param_snapshot_.size(); ++i) {
        try {
            params[i].data().copy_(param_snapshot_[i].data());
        } catch (...) {
            std::cerr << "  [恢复] 参数" << i << " copy_异常，跳过" << std::endl;
        }
    }
#ifdef USE_CUDA
    try { torch::cuda::synchronize(); } catch (...) {}
#endif
}

// 共同进化：接受后把M（新best）参数复制给Gen0，使对手=最新自己。
// 这样下一轮M相对新Gen0的ELO从0附近重新爬升，累积ELO只增不减；
// 根治"M 100%碾压固定基线→ELO封顶(798.254)→永无法严格创新高"的死锁。
void EloGatedTrainer::promote_gen0() {
    torch::NoGradGuard no_grad;  // copy_是对requires_grad叶子参数的就地操作，需绕过autograd
    auto m_params = (*model_)->parameters();
    auto g_params = (*gen0_)->parameters();
    for (size_t i = 0; i < m_params.size() && i < g_params.size(); ++i) {
        try {
            g_params[i].data().copy_(m_params[i].data());
        } catch (...) {
            std::cerr << "  [提升] Gen0参数" << i << " copy_异常，跳过" << std::endl;
        }
    }
#ifdef USE_CUDA
    try { torch::cuda::synchronize(); } catch (...) {}
#endif
    std::cout << "  [提升] Gen0已进化到新best（对手=最新自己，相对ELO归零重爬）" << std::endl;
}

// 检查模型参数是否有NaN/Inf
// 高学习率下模型可能发散（策略损失飙到18+、门控L2正反馈），发散参数会让
// 后续任意数值操作抛异常导致进程崩溃（0xE06D7363未捕获C++异常）。
// 每局训练后检查，发现即回滚，防止崩溃并节省一次无意义评估。
bool EloGatedTrainer::params_are_finite() const {
    try {
        for (const auto& p : (*model_)->parameters()) {
            if (!p.detach().isfinite().all().item<bool>()) {
                return false;
            }
        }
        return true;
    } catch (...) {
        return false;  // 检查本身异常视为发散
    }
}

// ==================== 主循环 ====================

void EloGatedTrainer::run() {
    std::cout << "\n========================================\n"
              << "  ELO门控在线训练（M vs Gen0学习，KataGo评估标尺）\n"
              << "========================================\n"
              << "  每轮对弈学习: " << config_.games_per_round << " 局（一局学一次）\n"
              << (config_.use_katago_eval
                  ? "  每轮评估: 1局 vs KataGo（目差标尺，" + std::to_string(config_.katago_visits) + " visits）\n"
                    "  每轮蒸馏: " + std::to_string(config_.katago_selfplay_games) +
                    "局KataGo自我对弈棋谱（" + std::to_string(config_.katago_teacher_visits) + " visits）\n"
                  : "  每轮评估: " + std::to_string(config_.eval_games) + " 局（自对弈）\n")
              << "  初始温度: " << config_.temperature
              << " | 学习率: " << config_.learning_rate << "\n"
              << "  回滚指标: " << (config_.use_katago_eval ? "KataGo目差（严格改善才接受）"
                                                            : "ELO（唯一）")
              << "\n========================================\n" << std::endl;

    // 初始评估基线：KataGo模式下best_elo_ = 初始目差（负值），后续必须严格改善才接受
    std::cout << "[EloGated] 初始评估..." << std::endl;
    best_elo_ = evaluate_elo(config_.eval_games);
    // 重置输出流格式（evaluate_elo内部fixed+precision(1)遗留，防显示成2e+02科学计数法）
    std::cout << std::defaultfloat << std::setprecision(6);
    // 累积ELO = 起始继承值（不受初始评估影响：初始评估只设定best_elo_锚点，
    // 累积只随"接受"单调增加——保证"只进不退"的显示语义，扩容换基线时延续1982.9）
    cumulative_elo_ = config_.cumulative_elo_start;
    std::cout << "[EloGated] 初始标尺值=" << std::fixed << std::setprecision(1)
              << best_elo_ << " | 累积ELO=" << cumulative_elo_ << std::defaultfloat
              << "（历史最佳）\n" << std::endl;

    float temperature = config_.temperature;
    int accepted_rounds = 0;
    int rollback_rounds = 0;

    for (round_ = 0; round_ < config_.max_rounds; ++round_) {
        std::cout << "---------- 第" << (round_ + 1) << "轮 ----------" << std::endl;

        // 轮级异常兜底：任何未预期异常（含CUDA异步错误在评估收尾/emptyCache时抛出）
        //   都走回滚路径，保证进程存活（实测0xE06D7363崩溃点位于评估完成后的收尾，
        //   局部catch覆盖不到，需整轮兜底）
        try {
            // 清除上一轮遗留的CUDA错误标志：CUDA异步错误在同步点才抛出，
            //   残留错误会让下一轮首次GPU操作立即复现；主动清错防止累积
#ifdef USE_CUDA
            cudaGetLastError();
#endif

            // 1. 参数快照（回滚点）
            snapshot_params();

        // 2. 对弈学习：10局，每局后立即训练一次（一局学一次）
        float round_win_rate_sum = 0.0f;
        int round_games = 0;
        bool diverged = false;  // 训练发散标志（参数NaN时立即回滚，跳过评估）

        for (int g = 0; g < config_.games_per_round; ++g) {
            try {
                bool model_first = ((round_ + g) % 2 == 0);
                GameData game;
                auto game_start = std::chrono::steady_clock::now();
                float outcome = play_game_vs_gen0(
                    total_games_++, model_first, temperature, true, &game);
                auto game_end = std::chrono::steady_clock::now();
                double secs = std::chrono::duration<double>(game_end - game_start).count();

                if (outcome > 0.5f) round_win_rate_sum += 1.0f;
                else if (outcome > -0.5f && outcome < 0.5f) round_win_rate_sum += 0.5f;
                round_games++;

                std::cout << "  对弈 " << (g + 1) << "/" << config_.games_per_round
                          << " | 步数=" << game.game_length
                          << " | M" << (outcome > 0.5f ? "胜" : (outcome < -0.5f ? "负" : "平"))
                          << " | 耗时=" << std::fixed << std::setprecision(1) << secs << "s"
                          << std::defaultfloat << std::endl;

                // 一局学一次：该局数据GAE后立即训练
                if (!game.steps.empty()) {
                    compute_gae_for_game(game);
                    train_on_game(game);
                    // 诊断：训练后检查CUDA错误（train_step有GPU操作）
#ifdef USE_CUDA
                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess) {
                        std::cerr << "  [诊断] 训练后CUDA错误: "
                                  << cudaGetErrorString(err) << std::endl;
                        cudaGetLastError();
                    }
#endif
                    // NaN防护：训练后参数发散立即回滚，防崩溃
                    if (!params_are_finite()) {
                        diverged = true;
                        std::cout << "  ! 训练发散（参数NaN），终止本轮对弈" << std::endl;
                        break;
                    }
                }
            } catch (...) {
                // 异常兜底：对弈/训练异常也走回滚路径（catch(...)兜住非std异常防崩溃）
                std::cerr << "  对弈/训练异常，终止本轮对弈" << std::endl;
                diverged = true;
                break;
            }
        }

        float round_win_rate = (round_games > 0) ?
            round_win_rate_sum / static_cast<float>(round_games) : 0.0f;
        std::cout << "  对弈汇总: M胜率=" << std::fixed << std::setprecision(1)
                  << (round_win_rate * 100.0f) << "%" << std::defaultfloat << std::endl;

        if (diverged) {
            // 训练发散：回滚到轮初快照并重建优化器（重置Adam动量，防爆炸状态残留）
            restore_params();
            trainer_->setup_optimizer();
            std::cout << "  ✗ 回滚（训练发散，参数恢复至轮初快照）" << std::endl;
            std::cout << "  累计: 累积ELO=" << cumulative_elo_ << std::endl;
#ifdef USE_CUDA
            c10::cuda::CUDACachingAllocator::emptyCache();
#endif
            continue;  // 跳过本轮评估，直接进入下一轮
        }

        // 3. ELO评估
        std::cout << "  [标记] 进入评估..." << std::endl;
        float elo = 0.0f;
        try {
            elo = evaluate_elo(config_.eval_games);
        } catch (...) {
            // 评估异常：保守回滚（catch(...)兜住一切异常防崩溃）
            std::cerr << "  评估异常，回滚" << std::endl;
            restore_params();
            trainer_->setup_optimizer();
            std::cout << "  ✗ 回滚（评估异常）" << std::endl;
            continue;
        }
        std::cout << "  [标记] 评估完成 标尺值=" << elo << std::endl;

        // 4. 门控：严格只进不退。
        //    KataGo模式（用户指定）：KataGo是标尺，接受条件 = 目差严格改善
        //      （best_elo_ = 历史最佳目差）。模型从输几十目开始，目差缩小（-50→-20）
        //      即进步被接受；不要求一把赢过KataGo。
        //    自对弈模式（旧）：接受条件 = 相对当前Gen0必须为正（elo>0），
        //      保证累积ELO只增不减（负ELO误接受曾让累积下跌，见run16教训）。
        // 重置输出流格式：评估函数内部使用fixed+precision(1)后遗留defaultfloat+precision(1)，
        //   会导致显示成2e+02（科学计数法，看着像溢出，实为格式问题）
        std::cout << std::defaultfloat << std::setprecision(6);
        bool accepted = config_.use_katago_eval ? (elo > best_elo_) : (elo > 0.0f);
        if (accepted) {
            accepted_rounds++;
            // 累积ELO只增不减：
            //   KataGo模式：按目差改善量映射（1目≈20 ELO，下限+20）——进步即累积；
            //   自对弈模式：接受时elo>0，累加正值。
            float improvement = std::max(elo - best_elo_, 0.0f);
            cumulative_elo_ += config_.use_katago_eval
                ? std::max(improvement * 20.0f, 20.0f)
                : elo;
            best_elo_ = elo;
            // 保存最佳模型与代际模型
            (*model_)->set_eval_mode();
            (*model_)->save(config_.checkpoint_dir + "/elo_best.pt");
            (*model_)->save(config_.checkpoint_dir +
                            "/elo_round" + std::to_string(round_ + 1) + ".pt");
            std::cout << "  ✓ 接受（标尺值 " << std::fixed << std::setprecision(1) << best_elo_
                      << " > 历史最佳，累积ELO=" << cumulative_elo_
                      << "，更新并保存）" << std::defaultfloat << std::setprecision(6)
                      << std::endl;
            std::cout << "  [标记] 已接受并保存" << std::endl;
            // 共同进化：训练对弈对手Gen0提升到新best（仅影响训练数据生成，
            // 评估基准仍是KataGo；自对弈模式下相对基准归零重爬根治ELO封顶死锁）
            promote_gen0();
            if (!config_.use_katago_eval) {
                best_elo_ = 0.0f;  // 自对弈模式：新基准下M起点=Gen0拷贝，相对ELO从0重新爬升
            }
        } else {
            rollback_rounds++;
            std::cout << "  [标记] 进入回滚" << std::endl;
            restore_params();  // 回滚到轮初快照
            std::cout << "  [标记] 参数已恢复" << std::endl;
            // 重建优化器：回滚参数但保留Adam动量会延续爆炸状态，重置保证干净起点
            trainer_->setup_optimizer();
            std::cout << "  [标记] 优化器已重建" << std::endl;
            std::cout << "  ✗ 回滚（标尺值 " << std::fixed << std::setprecision(1) << elo
                      << std::defaultfloat
                      << " <= 历史最佳 " << std::fixed << std::setprecision(1) << best_elo_
                      << std::defaultfloat
                      << "，参数恢复至轮初快照）" << std::endl;
            std::cout << "  [标记] 已回滚" << std::endl;
        }

        // 6. 蒸馏学习KataGo教师棋谱（本轮新增：KataGo自我对弈产生干净局面分布）。
        //    与"学评估对弈局"的本质区别：教师自对弈的局面都是正常对局进展，
        //    模型学到"好局面下怎么下好棋"；评估局是弱模型乱下后的极端局面（无效）。
        //    放在门控之后：接受/回滚先定，再在结果参数上蒸馏教师棋理。
        if (config_.use_katago_eval && config_.katago_selfplay_games > 0) {
            std::cout << "  生成KataGo教师棋谱（" << config_.katago_selfplay_games
                      << "局 × " << config_.katago_teacher_visits << " visits，每局训练"
                      << config_.katago_distill_epochs << "遍）..." << std::endl;
            int teacher_steps = 0;
            for (int sg = 0; sg < config_.katago_selfplay_games; ++sg) {
                try {
                    GameData tg = generate_katago_selfplay_game(
                        200000 + round_ * 1000 + sg);
                    if (tg.steps.empty()) continue;
                    teacher_steps += static_cast<int>(tg.steps.size());
                    // 同一局棋谱训练多遍：数据量少时单遍更新远不够（策略损失不降），
                    // 重复训练把学习信号放大N倍，生成成本为零、每遍仅约1秒
                    for (int ep = 0; ep < config_.katago_distill_epochs; ++ep) {
                        train_on_game(tg);
                    }
                } catch (...) {
                    std::cerr << "  教师棋谱第" << (sg + 1) << "局异常，跳过" << std::endl;
                }
            }
            std::cout << "  蒸馏学习完成（累计" << teacher_steps << "个教师步 × "
                      << config_.katago_distill_epochs << "遍）" << std::endl;
        }

        // 5. 温度衰减（每轮）
        temperature = std::max(config_.min_temperature, temperature * config_.temperature_decay);

        std::cout << "  累计: 接受=" << accepted_rounds
                  << " 回滚=" << rollback_rounds
                  << " | 累积ELO=" << std::fixed << std::setprecision(1) << cumulative_elo_
                  << std::defaultfloat << std::setprecision(6) << std::endl;

        // 显存清理：评估后清空PyTorch缓存块，防止跨轮累积
#ifdef USE_CUDA
        c10::cuda::CUDACachingAllocator::emptyCache();
#endif
        } catch (...) {
            // 轮级异常兜底：回滚到轮初快照，继续下一轮（防止进程崩溃）
            restore_params();
            trainer_->setup_optimizer();
            std::cout << "  ✗ 回滚（轮级异常）" << std::endl;
            std::cout << "  累计: 最佳ELO=" << best_elo_ << std::endl;
        }
    }

    std::cout << "\n========================================\n"
              << "  ELO门控训练完成 | 累积ELO=" << std::fixed << std::setprecision(1)
              << cumulative_elo_ << std::defaultfloat << std::setprecision(6)
              << " | 接受=" << accepted_rounds
              << " 回滚=" << rollback_rounds << "\n"
              << "========================================\n" << std::endl;
}

} // namespace training
} // namespace go
} // namespace spacego

#endif // USE_LIBTORCH
