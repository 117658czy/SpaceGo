#pragma once

#include <array>
#include <cstdint>
#include <random>
#include <vector>

namespace spacego {
namespace go {
namespace neural {

// ==================== 对手风格向量 ====================
// 五维风格刻画（围棋实战语义，学棋者视角）：
//   aggressive  - 攻击性：贴近对方棋子作战（打入/紧逼/战斗），对手越爱贴身战斗此值越高
//   defensive   - 防守性：贴近己方棋子补强连接（长/虎/尖守），对手越稳健补棋此值越高
//   balanced    - 均衡性：远离双方棋子的中腹自由落子（布局展开/大场）
//   territorial - 实地性：低线落子（第1~3线，抢实地/守角/生根）
//   influence   - 厚势性：高线落子（第4线以上，张势/模样/外势）
// 各分量 ∈ [0,1]，由对局落子统计归一化得到；向量整体不要求归一（重心即棋风方向）。
struct OpponentStyleVector {
    float aggressive = 0.0f;
    float defensive = 0.0f;
    float balanced = 0.0f;
    float territorial = 0.0f;
    float influence = 0.0f;

    // 转成有序定长数组（供模型侧构造 tensor / 传参）
    std::array<float, 5> to_array() const {
        return {aggressive, defensive, balanced, territorial, influence};
    }

    // 全部分量截断到 [0,1]（统计计数比例天然在区间内，防噪声采样越界）
    void clamp01() {
        aggressive   = std::min(1.0f, std::max(0.0f, aggressive));
        defensive    = std::min(1.0f, std::max(0.0f, defensive));
        balanced     = std::min(1.0f, std::max(0.0f, balanced));
        territorial  = std::min(1.0f, std::max(0.0f, territorial));
        influence    = std::min(1.0f, std::max(0.0f, influence));
    }
};

// ==================== 对手画像采集器 ====================
// 增量统计对局中"对手"每一手落子的风格特征，生成实时对手画像。
// 用法（竞争对弈场景）：对局中每当对方落一子，调用 record_move()；
// 轮到本方思考前调用 get_style() 获取对方当前画像并注入本方门控。
// 特征统计规则（对应五种风格）：
//   1) 线位：边缘距离 = min(x, y, 18-x, 18-y)（0 = 第1线）。
//      边缘距离 ≤ 2（第1~3线）→ 实地倾向；≥ 3（第4线及以上）→ 厚势倾向。
//   2) 邻接：以落子点为中心、曼哈顿距离 ≤ 2 的窗口内，
//      存在对方棋子 → 攻击（贴身战斗/打入）；
//      仅存在己方棋子 → 防守（补强连接）；
//      双方皆无 → 均衡（中腹展开）。
class OpponentStyleProfiler {
public:
    // 记录对手的一手落子
    // x, y       : 落子坐标（19路棋盘 0~18）
    // has_self   : 落子点附近（窗口内）是否有己方（即统计对象自己的）棋子
    // has_opp    : 落子点附近（窗口内）是否有对方（即落子方对手的）棋子
    // edge_dist  : 落子点边缘距离（0=第1线，18=对侧，用于判线位）
    void record_move(int edge_dist, bool has_self, bool has_opp);

    // 获取归一化风格向量（各分量 = 对应特征手数 / 总手数，无手时返回均衡原型）
    OpponentStyleVector get_style() const;

    // 重置统计（新对局开始）
    void reset();

    // 已统计手数
    int get_move_count() const { return total_moves_; }

    // ==================== 训练用原型风格池 ====================
    // 训练时注入"多样化对手风格"增强门控的条件分化能力：
    // 从五种极端棋风原型中随机选一种，叠加小噪声，模拟面对不同风格对手。
    // 配合条件 dropout（一半概率不注入）使模型同时保持无条件棋力。

    // 五种棋风原型（围棋实战典型类型）
    static const std::array<OpponentStyleVector, 5>& prototype_styles();

    // 从原型池采样一个风格（随机原型 + 每维 ±0.15 噪声，截断到 [0,1]）
    // rng: 调用方提供的随机数生成器
    static OpponentStyleVector sample_training_style(std::mt19937& rng);

private:
    // 各风格特征命中次数
    int aggressive_count_ = 0;
    int defensive_count_ = 0;
    int balanced_count_ = 0;
    int territorial_count_ = 0;
    int influence_count_ = 0;
    int total_moves_ = 0;
};

} // namespace neural
} // namespace go
} // namespace spacego
