#include "go/training/multi_dim_evaluator.h"

#ifdef USE_LIBTORCH

#include <queue>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cmath>

namespace spacego {
namespace go {
namespace training {

// ==================== 辅助方法 ====================

std::vector<std::pair<int, int>> MultiDimEvaluator::get_neighbors(int x, int y) {
    std::vector<std::pair<int, int>> result;
    const int dx[] = {0, 0, -1, 1};
    const int dy[] = {-1, 1, 0, 0};
    for (int i = 0; i < 4; ++i) {
        int nx = x + dx[i], ny = y + dy[i];
        if (in_bounds(nx, ny)) result.emplace_back(nx, ny);
    }
    return result;
}

std::vector<std::pair<int, int>> MultiDimEvaluator::get_diagonals(int x, int y) {
    std::vector<std::pair<int, int>> result;
    const int dx[] = {-1, -1, 1, 1};
    const int dy[] = {-1, 1, -1, 1};
    for (int i = 0; i < 4; ++i) {
        int nx = x + dx[i], ny = y + dy[i];
        if (in_bounds(nx, ny)) result.emplace_back(nx, ny);
    }
    return result;
}

int MultiDimEvaluator::manhattan_dist(int x1, int y1, int x2, int y2) {
    return std::abs(x1 - x2) + std::abs(y1 - y2);
}

float MultiDimEvaluator::to_score(float raw, float neutral, float scale) {
    // sigmoid-like 映射到 [0,1]
    float x = (raw - neutral) / (scale + 1e-6f);
    return 1.0f / (1.0f + std::exp(-x));
}

// ==================== 构造 ====================

MultiDimEvaluator::MultiDimEvaluator(const MultiDimEvalConfig& config)
    : config_(config) {}

void MultiDimEvaluator::update_config(const MultiDimEvalConfig& config) {
    config_ = config;
}

// ==================== 影响力图构建 ====================

void MultiDimEvaluator::build_influence_map(const GoBoardCPU& board, InfluenceMap& imap) {
    // 为每颗棋子计算辐射影响力，距离衰减
    auto add_influence = [&](int cx, int cy, StoneColor color) {
        for (int y = 0; y < 19; ++y) {
            for (int x = 0; x < 19; ++x) {
                int dist = manhattan_dist(cx, cy, x, y);
                if (dist > config_.influence_radius) continue;
                if (dist == 0) continue;
                float val = std::pow(config_.influence_decay, float(dist));
                auto& target = (color == StoneColor::BLACK) ?
                    imap.black_influence : imap.white_influence;
                target[y][x] += val;
            }
        }
    };

    for (int y = 0; y < 19; ++y) {
        for (int x = 0; x < 19; ++x) {
            StoneColor s = board.get_stone(x, y);
            if (s == StoneColor::BLACK || s == StoneColor::WHITE) {
                add_influence(x, y, s);
            }
        }
    }
}

// ==================== 一、领地类 ====================

float MultiDimEvaluator::estimate_territory(const GoBoardCPU& board, StoneColor color) {
    // 使用简单的控制区域估算：
    // 一个空点如果在己方影响力下且对方影响力很弱，视为己方领地
    InfluenceMap imap;
    build_influence_map(board, imap);

    float territory = 0.0f;
    for (int y = 0; y < 19; ++y) {
        for (int x = 0; x < 19; ++x) {
            float net = imap.net_at(x, y);
            if (board.is_empty(x, y)) {
                // 空点：谁的影响力强算谁的
                if (color == StoneColor::BLACK) {
                    territory += (net > 0.1f) ? 1.0f : (net > -0.1f ? 0.5f : 0.0f);
                } else {
                    territory += (net < -0.1f) ? 1.0f : (net < 0.1f ? 0.5f : 0.0f);
                }
            } else if (board.get_stone(x, y) == color) {
                territory += 1.0f;  // 己方棋子肯定算
            }
        }
    }
    return territory;
}

// ==================== 三、安全类 ====================

MultiDimEvaluator::GroupAnalysis MultiDimEvaluator::analyze_groups(
    const GoBoardCPU& board, StoneColor color) {

    GroupAnalysis ga;
    bool visited[19][19] = {};

    for (int y = 0; y < 19; ++y) {
        for (int x = 0; x < 19; ++x) {
            if (visited[y][x]) continue;
            if (board.get_stone(x, y) != color) continue;

            auto group = get_group(board, x, y, color);
            for (auto& p : group) visited[p.second][p.first] = true;

            int libs = count_group_liberties(board, group);
            ga.total_groups++;

            if (libs == 0) {
                // 已被提，不计入
                ga.total_groups--;
            } else if (libs == 1) {
                ga.danger_groups++;
                ga.weak_groups++;
            } else if (libs <= config_.weak_liberty) {
                ga.weak_groups++;
            } else {
                ga.safe_groups++;
            }

            if (libs > 0) {
                ga.avg_liberties += libs;
                ga.min_liberties = std::min(ga.min_liberties, float(libs));
            }
        }
    }

    if (ga.total_groups > 0) {
        ga.avg_liberties /= ga.total_groups;
    } else {
        ga.avg_liberties = 0.0f;
        ga.min_liberties = 0.0f;
    }

    return ga;
}

float MultiDimEvaluator::evaluate_connection_strength(
    const GoBoardCPU& board, StoneColor color) {

    // 计算所有棋组两两之间的最短路径，间距≤2且有连通路=强连接
    // 简化：统计棋组数量作为间接指标，棋组越少=连接越强
    int group_count = 0;
    bool visited[19][19] = {};

    for (int y = 0; y < 19; ++y) {
        for (int x = 0; x < 19; ++x) {
            if (visited[y][x]) continue;
            if (board.get_stone(x, y) != color) continue;

            auto group = get_group(board, x, y, color);
            for (auto& p : group) visited[p.second][p.first] = true;
            group_count++;
        }
    }

    // 无棋子时返回均势
    if (group_count == 0) return 0.5f;

    // 棋组数越少连接越强：1组=完美, 2-3组=良好, 4-5=一般, >5=弱
    float score = 1.0f - std::min(float(group_count), 10.0f) / 10.0f;
    return 0.5f + 0.5f * (score - 0.5f) * 2.0f;
}

float MultiDimEvaluator::evaluate_life_death(
    const GoBoardCPU& board, StoneColor color) {

    // 判断所有棋组中活棋比例
    int alive_groups = 0;
    int total_groups = 0;
    bool visited[19][19] = {};

    for (int y = 0; y < 19; ++y) {
        for (int x = 0; x < 19; ++x) {
            if (visited[y][x]) continue;
            if (board.get_stone(x, y) != color) continue;

            auto group = get_group(board, x, y, color);
            for (auto& p : group) visited[p.second][p.first] = true;

            int libs = count_group_liberties(board, group);
            if (libs > 0) {
                total_groups++;
                if (has_two_eyes(board, group) || libs >= 4) {
                    alive_groups++;
                }
            }
        }
    }

    if (total_groups == 0) return 0.5f;
    float alive_ratio = float(alive_groups) / total_groups;
    return 0.5f + 0.5f * (alive_ratio - 0.5f) * 2.0f;
}

// ==================== 四、主动权类 ====================

int MultiDimEvaluator::count_urgent_points(
    const GoBoardCPU& board, StoneColor current_player) {

    StoneColor opponent = (current_player == StoneColor::BLACK) ?
                          StoneColor::WHITE : StoneColor::BLACK;

    // 统计弱棋的相邻空点作为紧急点
    std::vector<std::pair<int, int>> urgent;
    bool visited[19][19] = {};

    // 己方弱棋的防守点
    for (int y = 0; y < 19; ++y) {
        for (int x = 0; x < 19; ++x) {
            if (visited[y][x]) continue;
            if (board.get_stone(x, y) != current_player) continue;

            auto group = get_group(board, x, y, current_player);
            for (auto& p : group) visited[p.second][p.first] = true;

            int libs = count_group_liberties(board, group);
            if (libs > 0 && libs <= config_.weak_liberty) {
                // 该弱棋的最后一口气是需要紧急防守的点
                for (auto& p : group) {
                    for (auto& n : get_neighbors(p.first, p.second)) {
                        if (board.is_empty(n.first, n.second)) {
                            urgent.push_back(n);
                        }
                    }
                }
            }
        }
    }

    // 对方弱棋的攻击点
    std::memset(visited, 0, sizeof(visited));
    for (int y = 0; y < 19; ++y) {
        for (int x = 0; x < 19; ++x) {
            if (visited[y][x]) continue;
            if (board.get_stone(x, y) != opponent) continue;

            auto group = get_group(board, x, y, opponent);
            for (auto& p : group) visited[p.second][p.first] = true;

            int libs = count_group_liberties(board, group);
            if (libs > 0 && libs <= config_.weak_liberty) {
                for (auto& p : group) {
                    for (auto& n : get_neighbors(p.first, p.second)) {
                        if (board.is_empty(n.first, n.second)) {
                            urgent.push_back(n);
                        }
                    }
                }
            }
        }
    }

    // 去重
    std::sort(urgent.begin(), urgent.end());
    urgent.erase(std::unique(urgent.begin(), urgent.end()), urgent.end());

    return (int)urgent.size();
}

float MultiDimEvaluator::evaluate_thickness(
    const GoBoardCPU& board, StoneColor current_player,
    const InfluenceMap& imap) {

    StoneColor opponent = (current_player == StoneColor::BLACK) ?
                          StoneColor::WHITE : StoneColor::BLACK;

    // 厚棋 = 无弱组 + 影响力强
    auto ga_self = analyze_groups(board, current_player);
    auto ga_opp = analyze_groups(board, opponent);

    // 弱棋比率（越低越厚）
    float weak_ratio_self = (ga_self.total_groups > 0) ?
        float(ga_self.weak_groups) / ga_self.total_groups : 0.0f;
    float weak_ratio_opp = (ga_opp.total_groups > 0) ?
        float(ga_opp.weak_groups) / ga_opp.total_groups : 0.0f;

    // 影响力对比
    float influence_self = 0.0f, influence_opp = 0.0f;
    for (int y = 0; y < 19; ++y) {
        for (int x = 0; x < 19; ++x) {
            if (current_player == StoneColor::BLACK) {
                influence_self += imap.black_influence[y][x];
                influence_opp += imap.white_influence[y][x];
            } else {
                influence_self += imap.white_influence[y][x];
                influence_opp += imap.black_influence[y][x];
            }
        }
    }

    float inf_ratio = (influence_self + influence_opp > 0.0f) ?
        influence_self / (influence_self + influence_opp) : 0.5f;

    // 综合：弱棋少 + 影响力强 = 厚
    float weak_score = 1.0f - 0.5f * (weak_ratio_self - weak_ratio_opp + 1.0f);
    weak_score = std::max(0.0f, std::min(1.0f, weak_score));

    return 0.5f * weak_score + 0.5f * inf_ratio;
}

// ==================== 五、全局形态类 ====================

float MultiDimEvaluator::evaluate_global_consistency(
    const GoBoardCPU& board, const InfluenceMap& imap) {

    // 将棋盘分为4个象限，检查各象限影响力均值是否方向一致
    // 都取外势（中心影响力强）vs 都取实地（边角领地确定），一致性高分
    // 一个区域外势一个区域实地 = 低一致性

    struct Quadrant {
        float influence_self = 0.0f;
        float territory_self = 0.0f;
        int points = 0;
    };

    Quadrant quads[4];  // 左上、右上、左下、右下
    const int half = 9;

    for (int y = 0; y < 19; ++y) {
        for (int x = 0; x < 19; ++x) {
            int qi = (y < half ? 0 : 2) + (x < half ? 0 : 1);
            quads[qi].points++;
        }
    }

    // 计算每个象限的影响力占比
    std::vector<float> ratios;
    for (auto& q : quads) {
        if (q.points > 0) {
            ratios.push_back(q.influence_self / (q.influence_self + 0.01f));
        }
    }

    if (ratios.size() < 2) return 0.5f;

    // 计算方差：方差小 = 各区域方向一致
    float mean = 0.0f;
    for (float r : ratios) mean += r;
    mean /= ratios.size();

    float variance = 0.0f;
    for (float r : ratios) {
        variance += (r - mean) * (r - mean);
    }
    variance /= ratios.size();

    // 方差映射到 [0,1]，方差小=一致性高
    float consistency = std::exp(-variance * 10.0f);
    return 0.5f + 0.5f * (consistency - 0.5f) * 2.0f;
}

float MultiDimEvaluator::evaluate_border_ownership(
    const GoBoardCPU& board, const InfluenceMap& imap) {

    // 评估四条边(上y=0, 下y=18, 左x=0, 右x=18)和四角的归属
    float self_count = 0.0f, total = 0.0f;

    // 四条边
    auto eval_point = [&](int x, int y) {
        float net = imap.net_at(x, y);
        // 谁的影响力强算谁的
        if (net > 0.1f) self_count += 1.0f;
        else if (net < -0.1f) { /* opponent */  }
        else self_count += 0.5f;
        total += 1.0f;
    };

    for (int x = 0; x < 19; ++x) eval_point(x, 0);   // 上边
    for (int x = 0; x < 19; ++x) eval_point(x, 18);  // 下边
    for (int y = 1; y < 18; ++y) eval_point(0, y);   // 左边
    for (int y = 1; y < 18; ++y) eval_point(18, y);  // 右边

    if (total == 0.0f) return 0.5f;
    return self_count / total;
}

float MultiDimEvaluator::evaluate_center_control(
    const GoBoardCPU& board, const InfluenceMap& imap) {

    // 天元(9,9)为中心，评估3x3到7x7中腹区域的控制权
    float self_inf = 0.0f, opp_inf = 0.0f;
    int radius = 4;

    for (int y = 9 - radius; y <= 9 + radius; ++y) {
        for (int x = 9 - radius; x <= 9 + radius; ++x) {
            if (!in_bounds(x, y)) continue;
            float dist = manhattan_dist(x, y, 9, 9);
            if (dist > radius) continue;
            // 越靠近天元权重越大
            float weight = 1.0f - float(dist) / (radius + 1);
            self_inf += imap.black_influence[y][x] * weight;
            opp_inf += imap.white_influence[y][x] * weight;
        }
    }

    float total = self_inf + opp_inf;
    if (total < 0.01f) return 0.5f;
    return self_inf / total;
}

// ==================== 主评价入口 ====================

MultiDimEvaluation MultiDimEvaluator::evaluate(
    const GoBoardCPU& board, StoneColor current_player) {

    MultiDimEvaluation eval;
    StoneColor opponent = (current_player == StoneColor::BLACK) ?
                          StoneColor::WHITE : StoneColor::BLACK;

    // 构建影响力图（后续多个量规复用）
    InfluenceMap imap;
    build_influence_map(board, imap);

    // 计算双方领地
    eval.black_territory_estimate = estimate_territory(board, StoneColor::BLACK);
    eval.white_territory_estimate = estimate_territory(board, StoneColor::WHITE);

    // 安全分析
    auto ga_self = analyze_groups(board, current_player);
    auto ga_opp = analyze_groups(board, opponent);

    // 是否从当前玩家视角
    bool is_black = (current_player == StoneColor::BLACK);

    // === 一、领地类 ===
    {
        // 实际领地
        float terr_self = is_black ? eval.black_territory_estimate : eval.white_territory_estimate;
        float terr_opp = is_black ? eval.white_territory_estimate : eval.black_territory_estimate;
        float max_terr = 361.0f;
        float terr_score = to_score(terr_self, max_terr / 2.0f, max_terr / 4.0f);

        // 提子数
        int cap_self = board.get_capture_count(current_player);
        int cap_opp = board.get_capture_count(opponent);
        float cap_diff = float(cap_self - cap_opp);
        float cap_score = to_score(cap_diff, 0.0f, 5.0f);

        // 地盘差（含贴目）
        float terr_diff = is_black ?
            (eval.black_territory_estimate - eval.white_territory_estimate - config_.komi) :
            (eval.white_territory_estimate - eval.black_territory_estimate - config_.komi);
        float diff_score = to_score(terr_diff, 0.0f, 20.0f);

        eval.metrics[(int)EvalDimension::ACTUAL_TERRITORY] = {"实际领地", terr_self, terr_score, 0.4f};
        eval.metrics[(int)EvalDimension::CAPTURE_COUNT]     = {"提子数", float(cap_diff), cap_score, 0.3f};
        eval.metrics[(int)EvalDimension::TERRITORY_DIFF]    = {"地盘差", terr_diff, diff_score, 0.3f};

        eval.territory.name = "领地类";
        eval.territory.metrics = {
            eval.metrics[(int)EvalDimension::ACTUAL_TERRITORY],
            eval.metrics[(int)EvalDimension::CAPTURE_COUNT],
            eval.metrics[(int)EvalDimension::TERRITORY_DIFF]
        };
        eval.territory.category_score = 0.4f * terr_score + 0.3f * cap_score + 0.3f * diff_score;
    }

    // === 二、潜力类 ===
    {
        // 影响力：己方影响力图覆盖面积
        float self_inf = 0.0f, opp_inf = 0.0f;
        for (int y = 0; y < 19; ++y) {
            for (int x = 0; x < 19; ++x) {
                if (is_black) {
                    self_inf += imap.black_influence[y][x];
                    opp_inf += imap.white_influence[y][x];
                } else {
                    self_inf += imap.white_influence[y][x];
                    opp_inf += imap.black_influence[y][x];
                }
            }
        }
        float inf_total = self_inf + opp_inf;
        float inf_score = (inf_total > 0.01f) ? self_inf / inf_total : 0.5f;

        // 发展空间：己方影响力覆盖但尚未成空的区域
        float dev_space = 0.0f;
        for (int y = 0; y < 19; ++y) {
            for (int x = 0; x < 19; ++x) {
                if (!board.is_empty(x, y)) continue;
                float net = is_black ? imap.net_at(x, y) : -imap.net_at(x, y);
                if (net > 0.1f) {
                    dev_space += 1.0f;
                }
            }
        }
        float dev_score = to_score(dev_space, 30.0f, 30.0f);

        // 势力范围比（同影响力评分）
        float ratio_score = inf_score;

        eval.metrics[(int)EvalDimension::INFLUENCE]          = {"影响力", self_inf, inf_score, 0.35f};
        eval.metrics[(int)EvalDimension::DEVELOPMENT_SPACE]   = {"发展空间", dev_space, dev_score, 0.35f};
        eval.metrics[(int)EvalDimension::INFLUENCE_RATIO]     = {"势力范围比", inf_total, ratio_score, 0.3f};

        eval.potential.name = "潜力类";
        eval.potential.metrics = {
            eval.metrics[(int)EvalDimension::INFLUENCE],
            eval.metrics[(int)EvalDimension::DEVELOPMENT_SPACE],
            eval.metrics[(int)EvalDimension::INFLUENCE_RATIO]
        };
        eval.potential.category_score = 0.35f * inf_score + 0.35f * dev_score + 0.3f * ratio_score;
    }

    // === 三、安全类 ===
    {
        // 棋组气数：平均气数
        float lib_score = to_score(ga_self.avg_liberties, 4.0f, 3.0f);

        // 弱棋数：弱棋越少越好
        float weak_score = (ga_self.total_groups > 0) ?
            1.0f - float(ga_self.weak_groups) / std::max(ga_self.total_groups, 1) : 1.0f;

        // 连接强度
        float conn_score = evaluate_connection_strength(board, current_player);

        // 死活状态
        float life_score = evaluate_life_death(board, current_player);

        eval.metrics[(int)EvalDimension::GROUP_LIBERTIES]      = {"棋组气数", ga_self.avg_liberties, lib_score, 0.25f};
        eval.metrics[(int)EvalDimension::WEAK_GROUPS]          = {"弱棋数", float(ga_self.weak_groups), weak_score, 0.25f};
        eval.metrics[(int)EvalDimension::CONNECTION_STRENGTH]  = {"连接强度", 0.0f, conn_score, 0.25f};
        eval.metrics[(int)EvalDimension::LIFE_DEATH_STATUS]    = {"死活状态", 0.0f, life_score, 0.25f};

        eval.safety.name = "安全类";
        eval.safety.metrics = {
            eval.metrics[(int)EvalDimension::GROUP_LIBERTIES],
            eval.metrics[(int)EvalDimension::WEAK_GROUPS],
            eval.metrics[(int)EvalDimension::CONNECTION_STRENGTH],
            eval.metrics[(int)EvalDimension::LIFE_DEATH_STATUS]
        };
        eval.safety.category_score = 0.25f * (lib_score + weak_score + conn_score + life_score);
    }

    // === 四、主动权类 ===
    {
        // 先手权：基于紧急点数和厚薄综合判断
        int urgent_count = count_urgent_points(board, current_player);
        int opp_urgent = count_urgent_points(board, opponent);
        float sente_score = (urgent_count + opp_urgent > 0) ?
            float(opp_urgent) / (urgent_count + opp_urgent + 1) : 0.5f;

        // 紧急点数
        float urgent_score = to_score(float(opp_urgent - urgent_count), 0.0f, 5.0f);

        // 厚薄优势
        float thick_score = evaluate_thickness(board, current_player, imap);

        eval.metrics[(int)EvalDimension::SENTE_RIGHT]        = {"先手权", float(opp_urgent - urgent_count), sente_score, 0.35f};
        eval.metrics[(int)EvalDimension::URGENT_POINTS]      = {"紧急点数", float(urgent_count), urgent_score, 0.3f};
        eval.metrics[(int)EvalDimension::THICKNESS_ADVANTAGE] = {"厚薄优势", 0.0f, thick_score, 0.35f};

        eval.initiative.name = "主动权类";
        eval.initiative.metrics = {
            eval.metrics[(int)EvalDimension::SENTE_RIGHT],
            eval.metrics[(int)EvalDimension::URGENT_POINTS],
            eval.metrics[(int)EvalDimension::THICKNESS_ADVANTAGE]
        };
        eval.initiative.category_score = 0.35f * sente_score + 0.3f * urgent_score + 0.35f * thick_score;
    }

    // === 五、全局形态类 ===
    {
        float consistency = evaluate_global_consistency(board, imap);
        float border = evaluate_border_ownership(board, imap);
        float center = evaluate_center_control(board, imap);

        eval.metrics[(int)EvalDimension::GLOBAL_CONSISTENCY] = {"全局一致性", 0.0f, consistency, 0.35f};
        eval.metrics[(int)EvalDimension::BORDER_OWNERSHIP]    = {"边界归属", 0.0f, border, 0.35f};
        eval.metrics[(int)EvalDimension::CENTER_CONTROL]      = {"天元控制", 0.0f, center, 0.3f};

        eval.global_form.name = "全局形态类";
        eval.global_form.metrics = {
            eval.metrics[(int)EvalDimension::GLOBAL_CONSISTENCY],
            eval.metrics[(int)EvalDimension::BORDER_OWNERSHIP],
            eval.metrics[(int)EvalDimension::CENTER_CONTROL]
        };
        eval.global_form.category_score = 0.35f * consistency + 0.35f * border + 0.3f * center;
    }

    // === 六、动态对战类（每步胜率等即时信号） ===
    // 新增维度目的：弥补纯启发式五大类"与真实胜负脱节"的缺陷。
    // 领地差（含贴目）是终局胜负最可靠的在线近似，直接映射为每步胜率，
    // 让即时奖励携带真实棋局进程信息，与GAE终局bootstrap形成闭环。
    {
        // 当前玩家视角的领地差（含贴目）——与领地类中terr_diff口径一致
        float terr_diff = is_black ?
            (eval.black_territory_estimate - eval.white_territory_estimate - config_.komi) :
            (eval.white_territory_estimate - eval.black_territory_estimate - config_.komi);
        // 每步胜率估计：sigmoid映射，0.5=均势；领地差20目时胜率约88%
        float win_rate = 1.0f / (1.0f + std::exp(-terr_diff / 15.0f));
        float win_score = win_rate;  // 胜率本身即[0,1]评分

        // 领地控制比：己方领地占总领地比例
        float terr_self = is_black ? eval.black_territory_estimate : eval.white_territory_estimate;
        float terr_opp = is_black ? eval.white_territory_estimate : eval.black_territory_estimate;
        float control_ratio = (terr_self + 1e-6f) / (terr_self + terr_opp + 1e-6f);
        float control_score = std::max(0.0f, std::min(1.0f, control_ratio));

        // 对杀优势：敌方弱棋组多 = 己方有追杀机会，正向得分
        float race_diff = float(ga_opp.weak_groups) - float(ga_self.weak_groups);
        float race_score = to_score(race_diff, 0.0f, 2.0f);

        eval.metrics[(int)EvalDimension::WIN_RATE_ESTIMATE]      = {"每步胜率", terr_diff, win_score, 0.5f};
        eval.metrics[(int)EvalDimension::TERRITORY_CONTROL_RATIO] = {"领地控制比", control_ratio, control_score, 0.3f};
        eval.metrics[(int)EvalDimension::CAPTURE_RACE_ADVANTAGE]  = {"对杀优势", race_diff, race_score, 0.2f};

        eval.dynamic.name = "动态对战类";
        eval.dynamic.metrics = {
            eval.metrics[(int)EvalDimension::WIN_RATE_ESTIMATE],
            eval.metrics[(int)EvalDimension::TERRITORY_CONTROL_RATIO],
            eval.metrics[(int)EvalDimension::CAPTURE_RACE_ADVANTAGE]
        };
        eval.dynamic.category_score = 0.5f * win_score + 0.3f * control_score + 0.2f * race_score;
    }

    // === 综合评分（归一化，加入动态对战类） ===
    // 动态类权重默认0.0（不进composite，避免与即时奖励双重计入），
    // 供需要时通过config开启
    float total_weight = config_.territory_weight + config_.potential_weight +
                         config_.safety_weight + config_.initiative_weight +
                         config_.global_form_weight + config_.dynamic_weight;
    if (total_weight < 1e-6f) total_weight = 1.0f;
    eval.composite_score =
        (config_.territory_weight * eval.territory.category_score +
         config_.potential_weight * eval.potential.category_score +
         config_.safety_weight * eval.safety.category_score +
         config_.initiative_weight * eval.initiative.category_score +
         config_.global_form_weight * eval.global_form.category_score +
         config_.dynamic_weight * eval.dynamic.category_score) / total_weight;

    eval.step_reward = 0.0f;  // 由外部通过前后对比设置

    return eval;
}

// ==================== 即时奖励计算 ====================

float MultiDimEvaluator::compute_step_reward(
    const MultiDimEvaluation& before,
    const MultiDimEvaluation& after,
    StoneColor current_player) {

    // 计算综合评分的差值
    float delta_composite = after.composite_score - before.composite_score;

    // 每步胜率变化：这是与真实棋局进程最相关的即时信号
    // （胜率由领地差+贴目映射，一步棋若显著改善胜率，即时奖励为正）
    float before_win = before.metrics[(int)EvalDimension::WIN_RATE_ESTIMATE].score;
    float after_win = after.metrics[(int)EvalDimension::WIN_RATE_ESTIMATE].score;
    float delta_win_rate = after_win - before_win;

    // 综合评分差值 + 胜率变化加权合成即时奖励
    // reward_scale默认0.1：综合分差值本身较小，胜率变化提供更强梯度
    float reward = (delta_composite * 0.5f + delta_win_rate * 0.5f) *
                   config_.reward_scale * 10.0f;
    reward = std::max(-1.0f, std::min(1.0f, reward));

    return reward;
}

// ==================== 棋块分析辅助 ====================

std::vector<std::pair<int, int>> MultiDimEvaluator::get_group(
    const GoBoardCPU& board, int sx, int sy, StoneColor color) {

    std::vector<std::pair<int, int>> result;
    bool visited[19][19] = {};
    std::queue<std::pair<int, int>> q;
    q.push({sx, sy});
    visited[sy][sx] = true;

    while (!q.empty()) {
        auto [x, y] = q.front(); q.pop();
        result.emplace_back(x, y);

        for (auto& n : get_neighbors(x, y)) {
            if (!visited[n.second][n.first] &&
                board.get_stone(n.first, n.second) == color) {
                visited[n.second][n.first] = true;
                q.push(n);
            }
        }
    }
    return result;
}

int MultiDimEvaluator::count_group_liberties(
    const GoBoardCPU& board,
    const std::vector<std::pair<int, int>>& group) {

    bool counted[19][19] = {};
    int liberties = 0;

    for (auto& p : group) {
        for (auto& n : get_neighbors(p.first, p.second)) {
            if (board.is_empty(n.first, n.second) &&
                !counted[n.second][n.first]) {
                counted[n.second][n.first] = true;
                liberties++;
            }
        }
    }
    return liberties;
}

bool MultiDimEvaluator::has_two_eyes(
    const GoBoardCPU& board,
    const std::vector<std::pair<int, int>>& group) {

    // 在棋组内部和边界搜索独立的眼位空间
    // 一只眼 = 被己方棋子完全包围的空区域
    // 两只眼 = 两个独立的被包围空区域

    // 标记棋组占用的位置
    bool is_group[19][19] = {};
    for (auto& p : group) is_group[p.second][p.first] = true;

    // 搜索被棋组包围的空区域
    int eyes = 0;
    bool visited[19][19] = {};

    for (auto& p : group) {
        for (auto& n : get_neighbors(p.first, p.second)) {
            int nx = n.first, ny = n.second;
            if (!board.is_empty(nx, ny) || visited[ny][nx]) continue;

            // BFS搜索此空区域
            std::vector<std::pair<int, int>> empty_region;
            std::queue<std::pair<int, int>> q;
            q.push({nx, ny});
            visited[ny][nx] = true;
            bool fully_surrounded = true;

            while (!q.empty()) {
                auto [ex, ey] = q.front(); q.pop();
                empty_region.emplace_back(ex, ey);

                for (auto& nn : get_neighbors(ex, ey)) {
                    if (board.is_empty(nn.first, nn.second)) {
                        if (!visited[nn.second][nn.first]) {
                            visited[nn.second][nn.first] = true;
                            q.push(nn);
                        }
                    } else if (board.get_stone(nn.first, nn.second) !=
                               board.get_stone(p.first, p.second)) {
                        fully_surrounded = false;
                    }
                }
            }

            // 如果是被己方棋子完全包围的空区域 = 一只眼
            if (fully_surrounded && !empty_region.empty()) {
                eyes++;
            }
        }
    }

    return eyes >= 2;
}

// ==================== GAE 计算器 ====================

GAECalculator::GAECalculator(const GAEConfig& config) : config_(config) {}

void GAECalculator::update_config(const GAEConfig& config) {
    config_ = config;
}

GAECalculator::GAEOutput GAECalculator::compute_gae(
    const std::vector<float>& step_rewards,
    const std::vector<float>& step_values,
    float terminal_value) {

    int T = (int)step_rewards.size();
    GAEOutput output;
    output.advantages.resize(T);
    output.returns.resize(T);

    if (T == 0) return output;

    float gae = 0.0f;
    // 终局bootstrap值：注入真实胜负（当前玩家视角）而非0
    // 原实现next_value=0导致价值目标成为"搜索root_value的自洽复制"，
    // 真实胜负game.outcome从不进入价值头 → 价值损失虚低、ELO停滞。
    // 现在最后一步的TD误差 = reward + gamma*terminal_value - V(T-1)，
    // 通过折扣因子向后传播终局信息，价值头学会预测"谁能赢"。
    float next_value = terminal_value;

    for (int t = T - 1; t >= 0; --t) {
        // delta = r_t + gamma * V(t+1) - V(t)
        float delta = step_rewards[t] + config_.gamma * next_value - step_values[t];

        // GAE: A_t = delta + gamma * lambda * A_{t+1}
        gae = delta + config_.gamma * config_.gae_lambda * gae;
        output.advantages[t] = gae;

        // TD目标回报: G_t = A_t + V(t)
        output.returns[t] = output.advantages[t] + step_values[t];

        next_value = step_values[t];
    }

    // 标准化优势
    if (config_.normalize_advantage && T > 1) {
        float mean = 0.0f, var = 0.0f;
        for (float a : output.advantages) mean += a;
        mean /= T;
        for (float a : output.advantages) var += (a - mean) * (a - mean);
        var /= T;
        float std = std::sqrt(var + 1e-8f);

        for (int t = 0; t < T; ++t) {
            output.advantages[t] = (output.advantages[t] - mean) / std;
        }
    }

    return output;
}

} // namespace training
} // namespace go
} // namespace spacego

#endif // USE_LIBTORCH
