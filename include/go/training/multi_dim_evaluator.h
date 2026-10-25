#pragma once

#ifdef USE_LIBTORCH

#include <vector>
#include <string>
#include <cmath>
#include <array>
#include <unordered_set>
#include "go/board/go_board_cuda.h"

namespace spacego {
namespace go {
namespace training {

// ==================== 评价维度枚举 ====================

enum class EvalDimension : int {
    // 一、领地类
    ACTUAL_TERRITORY = 0,    // 实际领地
    CAPTURE_COUNT = 1,        // 提子数
    TERRITORY_DIFF = 2,       // 地盘差（含贴目）

    // 二、潜力类
    INFLUENCE = 3,            // 影响力/辐射
    DEVELOPMENT_SPACE = 4,    // 发展空间
    INFLUENCE_RATIO = 5,      // 势力范围比

    // 三、安全类
    GROUP_LIBERTIES = 6,      // 棋组气数
    WEAK_GROUPS = 7,          // 弱棋数
    CONNECTION_STRENGTH = 8,  // 连接强度
    LIFE_DEATH_STATUS = 9,    // 死活状态

    // 四、主动权类
    SENTE_RIGHT = 10,         // 先手权
    URGENT_POINTS = 11,       // 紧急点数
    THICKNESS_ADVANTAGE = 12, // 厚薄优势

    // 五、全局形态类
    GLOBAL_CONSISTENCY = 13,  // 全局一致性
    BORDER_OWNERSHIP = 14,    // 边界归属
    CENTER_CONTROL = 15,      // 天元控制

    // 六、动态对战类（每步胜率等）
    WIN_RATE_ESTIMATE = 16,      // 每步胜率估计（由领地差+贴目映射，最可靠信号）
    TERRITORY_CONTROL_RATIO = 17, // 领地控制比（己方领地/双方领地总量）
    CAPTURE_RACE_ADVANTAGE = 18, // 对杀优势（敌方弱棋多=己方有利，正向得分）

    NUM_DIMENSIONS = 19
};

// ==================== 单维度评分结构 ====================

struct MetricResult {
    std::string name;          // 量规名称
    float raw_value = 0.0f;    // 原始值（有量纲）
    float score = 0.5f;        // 归一化评分 [0, 1]，0.5=均势
    float weight = 0.0f;       // 权重
};

// 五大类汇总
struct CategorySummary {
    std::string name;
    std::vector<MetricResult> metrics;
    float category_score;      // 该大类综合评分 [0, 1]
};

// ==================== 完整多元评价结果 ====================

struct MultiDimEvaluation {
    // 五大类汇总
    CategorySummary territory;     // 一、领地类
    CategorySummary potential;     // 二、潜力类
    CategorySummary safety;        // 三、安全类
    CategorySummary initiative;    // 四、主动权类
    CategorySummary global_form;   // 五、全局形态类
    CategorySummary dynamic;       // 六、动态对战类（每步胜率等）

    // 综合评分 [0, 1]，0.5=均势
    float composite_score;

    // 所有19个量规的直接索引
    std::array<MetricResult, (int)EvalDimension::NUM_DIMENSIONS> metrics;

    // 从当前玩家视角的即时收益变化 [-1, 1]
    float step_reward;

    // 双方实际领地估算（用于计算领地差）
    float black_territory_estimate;
    float white_territory_estimate;
};

// ==================== 评价器配置 ====================

struct MultiDimEvalConfig {
    // === 五大类权重 ===
    float territory_weight = 0.25f;    // 领地类（当前收益最可靠）
    float potential_weight = 0.20f;    // 潜力类
    float safety_weight = 0.25f;       // 安全类（风险控制很重要）
    float initiative_weight = 0.15f;   // 主动权类
    float global_form_weight = 0.15f;  // 全局形态类
    float dynamic_weight = 0.0f;       // 动态对战类（胜率/领地控制/对杀，作为即时奖励主信号）

    // === 一类：领地参数 ===
    float komi = 7.5f;                 // 贴目

    // === 二类：潜力参数 ===
    int influence_radius = 5;          // 影响力最大辐射半径
    float influence_decay = 0.75f;     // 影响力每步衰减率

    // === 三类：安全参数 ===
    int danger_liberty = 1;            // 危险气数阈值
    int weak_liberty = 2;             // 弱棋气数阈值
    int safe_liberty = 3;             // 安全气数阈值
    int connection_gap = 2;           // 连接判定间距（曼哈顿距离）

    // === 四类：主动权参数 ===

    // === 五类：全局形态参数 ===

    // 奖励缩放
    float reward_scale = 0.1f;
};

// ==================== 影响力图（用于多次计算共享） ====================

struct InfluenceMap {
    std::array<std::array<float, 19>, 19> black_influence;  // 黑方影响力
    std::array<std::array<float, 19>, 19> white_influence;  // 白方影响力

    InfluenceMap() {
        for (int y = 0; y < 19; ++y) {
            for (int x = 0; x < 19; ++x) {
                black_influence[y][x] = 0.0f;
                white_influence[y][x] = 0.0f;
            }
        }
    }

    // 获取某点的净影响力（正值=黑优，负值=白优）
    float net_at(int x, int y) const {
        return black_influence[y][x] - white_influence[y][x];
    }
};

// ==================== 多元评价器 ====================

class MultiDimEvaluator {
public:
    MultiDimEvaluator(const MultiDimEvalConfig& config = MultiDimEvalConfig());

    // 对棋盘状态进行完整的16项多元评价
    MultiDimEvaluation evaluate(const GoBoardCPU& board, StoneColor current_player);

    // 计算两步之间的即时奖励
    float compute_step_reward(const MultiDimEvaluation& before,
                              const MultiDimEvaluation& after,
                              StoneColor current_player);

    // 更新配置
    void update_config(const MultiDimEvalConfig& config);
    const MultiDimEvalConfig& get_config() const { return config_; }

private:
    MultiDimEvalConfig config_;

    // === 影响力图构建（一次构建，多次查询） ===
    void build_influence_map(const GoBoardCPU& board, InfluenceMap& imap);

    // === 一、领地类：实际领地估算 ===
    // 使用洪水填充：从棋盘边缘和对方棋子开始搜索，被己方棋子包围且对方无法打入的空点归为己方领地
    float estimate_territory(const GoBoardCPU& board, StoneColor color);

    // === 三、安全类 ===

    // 分析所有棋组的气数分布
    struct GroupAnalysis {
        int total_groups = 0;          // 总棋组数
        int danger_groups = 0;         // 气数=1的危险棋组
        int weak_groups = 0;           // 气数≤2的弱棋组
        int safe_groups = 0;           // 气数≥3的安全棋组
        float avg_liberties = 0.0f;    // 平均气数
        float min_liberties = 99.0f;   // 最小气数
    };
    GroupAnalysis analyze_groups(const GoBoardCPU& board, StoneColor color);

    // 计算棋组之间的连接强度
    float evaluate_connection_strength(const GoBoardCPU& board, StoneColor color);

    // 评估死活状态：判断棋组是否有两只真眼
    float evaluate_life_death(const GoBoardCPU& board, StoneColor color);

    // === 四、主动权类 ===

    // 计算紧急点数：对方弱棋可追击点 + 己方弱棋需防守点
    int count_urgent_points(const GoBoardCPU& board, StoneColor current_player);

    // 评估厚薄优势：厚棋=无弱组且影响力强，薄棋=有弱组需防守
    float evaluate_thickness(const GoBoardCPU& board, StoneColor current_player,
                             const InfluenceMap& imap);

    // === 五、全局形态类 ===

    // 评估全局一致性：各区域发展方向是否协调（都取外势或都取实地）
    float evaluate_global_consistency(const GoBoardCPU& board, const InfluenceMap& imap);

    // 评估边界归属：四条边和四个角的归属倾向
    float evaluate_border_ownership(const GoBoardCPU& board, const InfluenceMap& imap);

    // 评估天元控制：中央区域的控制权
    float evaluate_center_control(const GoBoardCPU& board, const InfluenceMap& imap);

    // === 辅助方法 ===

    // 坐标是否在棋盘内
    bool in_bounds(int x, int y) const {
        return x >= 0 && x < 19 && y >= 0 && y < 19;
    }

    // 获取上下左右邻居
    std::vector<std::pair<int, int>> get_neighbors(int x, int y);

    // 获取对角方向
    std::vector<std::pair<int, int>> get_diagonals(int x, int y);

    // 曼哈顿距离
    int manhattan_dist(int x1, int y1, int x2, int y2);

    // 获取棋块（BFS泛洪）
    std::vector<std::pair<int, int>> get_group(const GoBoardCPU& board,
                                                int sx, int sy, StoneColor color);

    // 计算棋块气数
    int count_group_liberties(const GoBoardCPU& board,
                              const std::vector<std::pair<int, int>>& group);

    // 检测是否有两只真眼
    bool has_two_eyes(const GoBoardCPU& board,
                      const std::vector<std::pair<int, int>>& group);

    // 将原始值映射到[0,1]评分
    float to_score(float raw, float neutral, float scale);
};

// ==================== GAE计算器 ====================

struct GAEConfig {
    float gamma = 0.99f;             // 折扣因子
    float gae_lambda = 0.95f;        // GAE lambda
    bool normalize_advantage = true;
};

class GAECalculator {
public:
    GAECalculator(const GAEConfig& config = GAEConfig());

    struct GAEOutput {
        std::vector<float> advantages;
        std::vector<float> returns;
    };

    // 输入每步即时奖励 + 价值估计，输出GAE优势和TD回报
    // terminal_value：终局真实胜负（当前玩家视角，±1），用于最后一步的TD bootstrap。
    // 历史教训：原实现终局bootstrap恒为0（next_value=0），价值目标=搜索root_value的
    // 自洽循环，真实胜负game.outcome从未注入 → 价值损失虚低0.001、ELO停滞。
    GAEOutput compute_gae(
        const std::vector<float>& step_rewards,
        const std::vector<float>& step_values,
        float terminal_value = 0.0f);

    void update_config(const GAEConfig& config);
    const GAEConfig& get_config() const { return config_; }

private:
    GAEConfig config_;
};

} // namespace training
} // namespace go
} // namespace spacego

#endif // USE_LIBTORCH
