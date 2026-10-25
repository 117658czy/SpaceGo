#pragma once

#ifdef USE_LIBTORCH

#include <torch/torch.h>
#include <vector>
#include <memory>
#include <string>
#include <random>
#include <mutex>

#include "go/board/go_board_cuda.h"
#include "go/neural/ca_moe_expert.h"
#include "go/search/hmgs_dp_optimized.h"
#include "go/training/unsupervised_self_reinforcement.h"

namespace spacego {
namespace go {
namespace training {

// ==================== ELO门控在线训练配置 ====================
// 用户指定的经典AI训练框架：单模型 vs 固定基线Gen0，多元评价体系下多拿分，
// 高温度高学习率，一局学一次，学不好就回滚——唯一回滚指标是ELO。
// 从机制上杜绝"越学越差"：任何导致ELO不涨的更新都会被回滚。

struct EloGatedConfig {
    // === 模型配置 ===
    int num_experts = 8;                  // CA-MOE专家数量
    int board_size = 19;
    float dropout_rate = 0.1f;
    int hidden_dim = 256;                 // M通道宽度（小模型可--elo-hidden=128减参快速迭代）
    int gen0_hidden_dim = 256;            // Gen0基线通道宽度（默认256与大模型checkpoint兼容；
                                          //   可配128配小基线，或M=128/基线=256形成"小模型挑战
                                          //   强基线"——M从弱起步追赶，ELO持续上涨空间大）

    // === 路径 ===
    std::string checkpoint_dir = "./training_output/elo_gated/checkpoints";
    std::string output_dir = "./training_output/elo_gated";
    std::string start_checkpoint = "";    // 起点模型（如 Gen1 峰值）
    std::string gen0_checkpoint = "";     // 固定基线 Gen0

    // === 搜索配置 ===
    // 回滚模式时间预算（快速迭代）：与"长时间"二倍体模式拆开——回滚只管快速试错，
    // ELO不涨即回滚，因此压低模拟数与评估局数，把单代(单轮)压到5~10分钟。
    // 快速迭代档（128小模型）：80模拟/局≈3~4s，4局对弈+16局评估≈2~3分钟/轮，
    // 在"不拟合"前提下尽可能多迭代——评估局数不宜再降（<16局ELO噪声>±60，
    // 门控频繁误回滚反而拖慢上涨），单局搜索数再降则棋力信号太弱学不到棋。
    int simulations_per_move = 80;        // 对弈与评估的HMGS-DP模拟数（150→80：单局3.8s）
    int max_game_length = 400;

    // === 每轮在线学习配置 ===
    int games_per_round = 4;              // 每轮对弈学习局数（6→4：单轮更短，迭代更多）
    int eval_games = 24;                  // 每轮评估ELO的局数（16→24：降噪——16局ELO噪声±150，
                                          //   单次全胜(798)可封死后续所有评估，24局噪声±110更可信）
    int max_rounds = 200;                 // 最大轮数
    float temperature = 1.0f;             // 训练对弈温度（实测1.5探索导致对弈全负→数据全"乱局"，
                                          //   分布漂移：模型学乱局应对而非下好棋，ELO门控全回滚。
                                          //   1.0为标准AlphaZero探索，数据接近真实对局）
    float gen0_temperature = 1.0f;        // Gen0训练对弈温度（与M同温：分布一致才有学习价值；
                                          //   评估时固定贪心0.01不受此配置影响）
    float temperature_decay = 0.90f;      // 每轮温度衰减（0.92→0.90：快速迭代下更快收敛到贪心，
                                          //   减少探索浪费，提高对弈数据质量与评估稳定性）
    float min_temperature = 0.5f;         // 温度下限

    // === 训练配置（一局学一次，高学习率） ===
    float learning_rate = 0.005f;         // 高学习率（用户指定开大；实测0.03门控L2正反馈爆炸、
                                          //   0.01策略损失飙至17.8（adv加权log_probs无clip），
                                          //   0.005+log_probs clip[-6,0] 为实测稳定组合）
    int batch_size = 256;                 // 单局数据打包为batch训练
    int replay_capacity = 8000;           // 经验池容量（环形覆盖）
    bool use_mixed_precision = false;     // 禁用AMP（1660 Ti上FP16训练破坏模型的教训）

    // === ELO门控 ===
    // 严格门控（用户明确要求）：ELO只进不退，评估低于历史最佳即回滚，无弹性空间。
    // 评估可信度靠eval_games(24局)保证——噪声大时严格门控会误回滚，局数足够才可靠。
    float elo_threshold = 0.0f;           // 接受容差（固定0=必须严格创新高才接受）
    // 累积ELO起始值：共同进化+扩容时继承上次运行的累积水平（如128→256扩容后
    // 相对新Gen0的初始ELO≈0，但模型棋力继承自128 best，累积应从1982.9延续）
    float cumulative_elo_start = 0.0f;

    // === KataGo评估（用户指定：评估器换成KataGo，每轮1局定胜负） ===
    // 用外部KataGo引擎作为评估标尺（替代M vs Gen0自对弈）：
    // 每轮评估让当前模型与KataGo下1局，得到"模型视角目差"（负=输N目），
    // 门控接受条件 = 目差严格改善（不要求赢KataGo）。
    // 贴目修正（实测10轮后修复）：目差已剔除贴目，是"不含贴目的原始目差"。
    //   未修正时同一棋力执黑比执白天然低2×贴目=15目，而门控用单一最佳值跨执色比较，
    //   导致白轮设best后所有黑轮永远不可能超过（实测10轮4个黑轮全部结构性回滚）。
    // 注意：KataGo强度由katago_visits控制，调高标尺会延长"能接受的轮数"，
    //   模型强了再逐步提高基准强度。
    bool use_katago_eval = true;          // true=用KataGo评估；false=回退旧自对弈评估
    std::string katago_exe = "D:/KataGo/katago16/katago.exe";
    std::string katago_model = "D:/KataGo/kata1-b18c384nbt.bin.gz";
    std::string katago_config = "D:/KataGo/gtp.cfg";
    int katago_visits = 400;              // KataGo基准强度（评估标尺用，maxVisits）

    // === KataGo教师自我对弈（蒸馏数据源） ===
    // 每轮额外跑若干局KataGo自我对弈产生高质量棋谱，用这些局面蒸馏学习模型。
    // 与"学评估对弈局"的本质区别：教师自对弈的局面分布干净（双方都是强棋、
    // 正常对局进展），模型学到"好局面下怎么下好棋"；而评估局是弱模型乱下后
    // 的极端局面（学的是"怎么收拾自己造的烂摊子"，实测策略损失不降、无效）。
    int katago_selfplay_games = 4;        // 每轮教师自我对弈局数（每局约300样本）
    int katago_teacher_visits = 10;       // 教师自对弈强度（棋谱质量 vs 生成速度的权衡）
    // 教师棋谱训练遍数：每局教师棋谱重复训练多遍。实测只过一遍时策略损失恒定在
    // log(1/362)≈5.89（均匀分布水平）且逐遍完全相同——根因是策略log的clip下界(-6)
    // 距均匀概率(2.76e-3, log=-5.895)仅2%，目标动作概率略低于均匀即落入梯度死区
    // （见compute_losses注释）；下界放宽到-12后该现象消失。多遍训练把学习信号×N，
    // 代价仅每遍约1秒（生成成本为零）。
    int katago_distill_epochs = 8;

    // === 起点模型 ===
    // 彻底从头：跳过起点模型加载与同尺寸自举基线（gen0_h*.pt）回退，直接随机初始化M。
    // 用途：起点模型已塌缩（特征饱和→梯度消失，实测裁剪前梯度总范数1.5e-11、ΔL2≈0）
    //   时，在其上继续训练拿不到任何学习信号（损失恒定、权重不变），只能重新来过。
    //   Gen0固定基线不受影响（仍按gen0_checkpoint加载），训练对弈的数据来源不变。
    bool fresh_start = false;

    // === 多元评价 ===
    bool enable_multi_dim_eval = true;    // 多元评价体系作为即时奖励（必须多拿分）

    int seed = 42;
};

// ==================== ELO门控在线训练器 ====================

class EloGatedTrainer {
public:
    explicit EloGatedTrainer(const EloGatedConfig& config);
    ~EloGatedTrainer();

    // 初始化所有组件并加载起点模型
    bool initialize();

    // 主循环：对弈学习→ELO评估→门控回滚，直到最大轮数
    void run();

private:
    // ---- 组件 ----
    std::shared_ptr<neural::CA_MOEExpertSystem> model_;   // 学习模型M
    std::shared_ptr<neural::CA_MOEExpertSystem> gen0_;    // 固定基线Gen0（冻结）
    std::unique_ptr<search::HMGSDP_Optimized> searcher_m_;  // M的搜索器
    std::unique_ptr<search::HMGSDP_Optimized> searcher_g_;  // Gen0的搜索器
    std::unique_ptr<MultiDimEvaluator> multi_dim_evaluator_; // 多元评价（即时奖励来源）
    std::unique_ptr<GAECalculator> gae_calculator_;          // GAE时序差分
    std::unique_ptr<MixedPrecisionTrainer> trainer_;         // 一局学一次的优化器
    std::unique_ptr<EfficientReplayBuffer> replay_;          // 经验池

    // ---- 状态 ----
    EloGatedConfig config_;
    torch::Device device_;
    std::mt19937 rng_;
    // ELO门控状态（共同进化 + 累积ELO）：
    //   · best_elo_      ：相对当前Gen0的历史最佳（每次接受后Gen0提升到新best，
    //                       M起点=Gen0拷贝→相对基准归零，best_elo_重置为0重爬）
    //   · cumulative_elo_：累积ELO（只增不减，展示用主指标）。接受时累加"相对当前
    //                       Gen0的提升值"，Gen0每代变强一次，累积单调递增，永不封顶
    float best_elo_ = 0.0f;
    float cumulative_elo_ = 0.0f;
    int round_ = 0;
    int total_games_ = 0;
    std::mutex infer_mutex_;  // 共享模型推理互斥锁

    // ---- 参数快照（ELO门控回滚） ----
    // 每轮开始前克隆当前参数到CPU，ELO不涨则恢复
    std::vector<torch::Tensor> param_snapshot_;
    bool has_snapshot_ = false;

    // ---- 方法 ----
    // 初始化两个HMGS-DP搜索器（各自绑定模型推理回调）
    void init_searchers();
    // 对弈一局：M vs Gen0（交替先后手），返回M视角胜负；
    // collect=true且out_game非空时记录模型步训练数据到out_game
    float play_game_vs_gen0(int game_id, bool model_first, float temperature,
                            bool collect, GameData* out_game = nullptr);
    // 对弈一局：M vs KataGo（KataGo为外部GTP子进程），返回M视角目差。
    // 评估专用：M侧低温贪心决策，KataGo侧通过GTP管道交互
    float play_game_vs_katago(bool model_first, float temperature);
    // 生成一局KataGo自我对弈棋谱（蒸馏教师数据）：双方均为KataGo，
    // 每个局面的落子+终局胜负构成监督样本。局面分布干净=有效蒸馏
    GameData generate_katago_selfplay_game(int game_id);
    // 对一局模型步计算GAE（终局真实胜负注入）
    void compute_gae_for_game(GameData& game);
    // 一局学一次：把该局模型步打包为batch并训练一步
    void train_on_game(const GameData& game);
    // ELO评估：M vs Gen0打eval_games局（搜索300 sims，低温贪心）
    float evaluate_elo(int games);
    // 参数快照与回滚
    void snapshot_params();
    void restore_params();
    // 共同进化：把M（新best）参数复制给Gen0（对手=最新自己），
    // 让相对ELO归零重爬——累积ELO只增不减，根治"100%碾压固定基线→ELO封顶死锁"
    void promote_gen0();
    // 检查模型参数是否有NaN/Inf（训练发散时立即回滚，防止崩溃）
    bool params_are_finite() const;
    // 辅助：搜索策略（绑定指定搜索器与模型）
    std::pair<std::vector<float>, float> search_policy(
        search::HMGSDP_Optimized* searcher, const std::string& board_state);
};

} // namespace training
} // namespace go
} // namespace spacego

#endif // USE_LIBTORCH
