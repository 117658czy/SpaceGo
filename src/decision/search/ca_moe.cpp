#include "go/search/ca_moe.h"
#include <algorithm>
#include <random>
#include <numeric>
#include <cmath>
#include "framework/logger.h"

namespace spacego {
namespace go {

CAMOE::CAMOE() {
    initialize_default_parameters();
    initialize_network();
}

CAMOE::~CAMOE() {
    performance_history_.clear();
}

void CAMOE::initialize_default_parameters() {
    LOG_INFO("CA-MOE默认参数初始化完成");
}

void CAMOE::initialize_network() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    int input_size = 361;
    int hidden_size = 128;

    network_weights_.resize(input_size * hidden_size);
    for (auto& w : network_weights_) {
        w = dist(gen);
    }

    network_bias_.resize(hidden_size);
    for (auto& b : network_bias_) {
        b = dist(gen);
    }

    LOG_INFO("CA-MOE神经网络初始化完成");
}

LinkNeuronOutput CAMOE::forward(const GoBoardCPU& board) {
    LinkNeuronOutput output;

    const int BOARD_SIZE = 19;
    const int TOTAL_POSITIONS = BOARD_SIZE * BOARD_SIZE;

    // 使用启发式评估为每个棋盘位置生成先验概率
    output.prior_probs.resize(TOTAL_POSITIONS, 0.0f);

    float total_score = 0.0f;

    for (int x = 0; x < BOARD_SIZE; ++x) {
        for (int y = 0; y < BOARD_SIZE; ++y) {
            int idx = x * BOARD_SIZE + y;
            float score = 0.0f;

            // 已有棋子的位置概率为0
            if (!board.is_empty(x, y)) {
                output.prior_probs[idx] = 0.0f;
                continue;
            }

            // 非法着法概率为0
            if (!board.is_legal_move(x, y, board.get_current_player())) {
                output.prior_probs[idx] = 0.0f;
                continue;
            }

            score = 1.0f;

            // 距中心越近分数越高
            float center_x = (BOARD_SIZE - 1) / 2.0f;
            float center_y = (BOARD_SIZE - 1) / 2.0f;
            float dist_to_center = std::sqrt(
                std::pow(x - center_x, 2) + std::pow(y - center_y, 2));
            score *= (1.5f - dist_to_center / 13.0f);

            // 邻近有棋子的位置分数更高
            const int dx[] = {1, -1, 0, 0};
            const int dy[] = {0, 0, 1, -1};
            int adjacent_count = 0;
            for (int d = 0; d < 4; ++d) {
                int nx = x + dx[d];
                int ny = y + dy[d];
                if (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE) {
                    if (!board.is_empty(nx, ny)) {
                        adjacent_count++;
                    }
                }
            }
            if (adjacent_count > 0) {
                score *= (1.0f + adjacent_count * 0.25f);
            }

            // 星位加分
            const int star_points[][2] = {{3, 3}, {3, 9}, {3, 15},
                                       {9, 3}, {9, 9}, {9, 15},
                                       {15, 3}, {15, 9}, {15, 15}};
            for (const auto& [sx, sy] : star_points) {
                if (x == sx && y == sy) {
                    score *= 1.4f;
                    break;
                }
            }

            output.prior_probs[idx] = score;
            total_score += score;
        }
    }

    // 归一化为概率分布
    if (total_score > 0.0f) {
        for (auto& p : output.prior_probs) {
            p /= total_score;
        }
    }

    // 价值评估
    auto black_stones = board.get_all_stones(StoneColor::BLACK);
    auto white_stones = board.get_all_stones(StoneColor::WHITE);
    float material_diff = static_cast<float>(black_stones.size() - white_stones.size());
    float position_advantage = 0.0f;
    for (const auto& [bx, by] : black_stones) {
        float dist = std::sqrt(std::pow(bx - 9.0f, 2) + std::pow(by - 9.0f, 2));
        position_advantage += (13.5f - dist) / 13.5f;
    }
    for (const auto& [wx, wy] : white_stones) {
        float dist = std::sqrt(std::pow(wx - 9.0f, 2) + std::pow(wy - 9.0f, 2));
        position_advantage -= (13.5f - dist) / 13.5f;
    }
    float raw_value = (material_diff * 8.0f + position_advantage * 2.0f) / 100.0f;
    output.value = std::clamp(raw_value, -1.0f, 1.0f);

    // 状态编码
    output.state_encoding.resize(STATE_ENCODING_DIM);
    for (int i = 0; i < STATE_ENCODING_DIM; ++i) {
        output.state_encoding[i] = output.value * 0.5f;
    }

    // 不确定性
    int total_stones = static_cast<int>(black_stones.size() + white_stones.size());
    float complexity = static_cast<float>(total_stones) / TOTAL_POSITIONS;
    output.uncertainty = 1.0f - complexity;

    return output;
}

LinkNeuronOutput CAMOE::forward_simple(const GoBoardCPU& board) {
    LinkNeuronOutput output;

    const int BOARD_SIZE = 19;
    const int TOTAL_POSITIONS = BOARD_SIZE * BOARD_SIZE;

    // 使用启发式评估为每个棋盘位置生成先验概率
    output.prior_probs.resize(TOTAL_POSITIONS, 0.0f);

    float total_score = 0.0f;

    for (int x = 0; x < BOARD_SIZE; ++x) {
        for (int y = 0; y < BOARD_SIZE; ++y) {
            int idx = x * BOARD_SIZE + y;

            // 已有棋子或非法着法概率为0
            if (!board.is_empty(x, y) || !board.is_legal_move(x, y, board.get_current_player())) {
                output.prior_probs[idx] = 0.0f;
                continue;
            }

            float pos_score = 1.0f;

            // 距中心越近分数越高
            float center_x = (BOARD_SIZE - 1) / 2.0f;
            float center_y = (BOARD_SIZE - 1) / 2.0f;
            float dist_to_center = std::sqrt(
                std::pow(x - center_x, 2) + std::pow(y - center_y, 2));
            pos_score *= (1.5f - dist_to_center / 13.0f);

            // 邻近有棋子的位置分数更高
            const int dx[] = {1, -1, 0, 0};
            const int dy[] = {0, 0, 1, -1};
            int adjacent_count = 0;

            for (int d = 0; d < 4; ++d) {
                int nx = x + dx[d];
                int ny = y + dy[d];

                if (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE) {
                    if (!board.is_empty(nx, ny)) {
                        adjacent_count++;
                    }
                }
            }

            if (adjacent_count > 0) {
                pos_score *= (1.0f + adjacent_count * 0.25f);
            }

            // 星位加分
            const int star_points[][2] = {{3, 3}, {3, 9}, {3, 15},
                                       {9, 3}, {9, 9}, {9, 15},
                                       {15, 3}, {15, 9}, {15, 15}};

            for (const auto& [sx, sy] : star_points) {
                if (x == sx && y == sy) {
                    pos_score *= 1.4f;
                    break;
                }
            }

            output.prior_probs[idx] = pos_score;
            total_score += pos_score;
        }
    }

    // 归一化为概率分布
    if (total_score > 0.0f) {
        for (auto& p : output.prior_probs) {
            p /= total_score;
        }
    } else {
        // 没有合法着法时均匀分布
        float uniform = 1.0f / TOTAL_POSITIONS;
        for (auto& p : output.prior_probs) {
            p = uniform;
        }
    }

    auto black_stones = board.get_all_stones(StoneColor::BLACK);
    auto white_stones = board.get_all_stones(StoneColor::WHITE);

    float material_diff = static_cast<float>(black_stones.size() - white_stones.size());
    float position_advantage = 0.0f;

    for (const auto& [bx, by] : black_stones) {
        float dist = std::sqrt(std::pow(bx - 9.0f, 2) + std::pow(by - 9.0f, 2));
        position_advantage += (13.5f - dist) / 13.5f;
    }
    for (const auto& [wx, wy] : white_stones) {
        float dist = std::sqrt(std::pow(wx - 9.0f, 2) + std::pow(wy - 9.0f, 2));
        position_advantage -= (13.5f - dist) / 13.5f;
    }

    float raw_value = (material_diff * 8.0f + position_advantage * 2.0f) / 100.0f;
    output.value = std::clamp(raw_value, -1.0f, 1.0f);

    output.state_encoding.resize(STATE_ENCODING_DIM);
    for (size_t i = 0; i < STATE_ENCODING_DIM; ++i) {
        output.state_encoding[i] = output.value * 0.5f;
    }

    int total_stones = black_stones.size() + white_stones.size();
    float complexity = static_cast<float>(total_stones) / (BOARD_SIZE * BOARD_SIZE);
    output.uncertainty = std::clamp(complexity * 2.0f, 0.05f, 0.95f);

    return output;
}

bool CAMOE::save_model(const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("无法保存模型到: %s", path.c_str());
        return false;
    }

    size_t num_weights = network_weights_.size();
    file.write(reinterpret_cast<const char*>(&num_weights), sizeof(num_weights));
    file.write(reinterpret_cast<const char*>(network_weights_.data()),
                num_weights * sizeof(float));

    size_t num_bias = network_bias_.size();
    file.write(reinterpret_cast<const char*>(&num_bias), sizeof(num_bias));
    file.write(reinterpret_cast<const char*>(network_bias_.data()),
                num_bias * sizeof(float));

    file.close();
    LOG_INFO("模型已保存到: %s", path.c_str());
    return true;
}

bool CAMOE::load_model(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("无法加载模型: %s", path.c_str());
        return false;
    }

    size_t num_weights;
    file.read(reinterpret_cast<char*>(&num_weights), sizeof(num_weights));
    network_weights_.resize(num_weights);
    file.read(reinterpret_cast<char*>(network_weights_.data()),
              num_weights * sizeof(float));

    size_t num_bias;
    file.read(reinterpret_cast<char*>(&num_bias), sizeof(num_bias));
    network_bias_.resize(num_bias);
    file.read(reinterpret_cast<char*>(network_bias_.data()),
              num_bias * sizeof(float));

    file.close();
    LOG_INFO("模型已从: %s 加载", path.c_str());
    return true;
}

float CAMOE::sigmoid(float x) const {
    return 1.0f / (1.0f + std::exp(-std::clamp(x, -500.0f, 500.0f)));
}

float CAMOE::relu(float x) const {
    return std::max(0.0f, x);
}

float CAMOE::tanh_activation(float x) const {
    return std::tanh(x);
}

float CAMOE::cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) const {
    if (a.empty() || b.empty() || a.size() != b.size()) return 0.0f;

    float dot_product = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);

    if (norm_a < 0.0001f || norm_b < 0.0001f) return 0.0f;
    return dot_product / (norm_a * norm_b);
}

std::vector<float> CAMOE::apply_network(const std::vector<float>& input) {
    int input_size = static_cast<int>(input.size());
    int hidden_size = static_cast<int>(network_bias_.size());

    std::vector<float> output(hidden_size, 0.0f);
    for (int j = 0; j < hidden_size; ++j) {
        float sum = network_bias_[j];
        for (int i = 0; i < input_size; ++i) {
            int idx = i * hidden_size + j;
            if (idx < static_cast<int>(network_weights_.size())) {
                sum += input[i] * network_weights_[idx];
            }
        }
        output[j] = relu(sum);
    }
    return output;
}

std::vector<float> CAMOE::prepare_input(const GoBoardCPU& board) const {
    std::vector<float> input(361, 0.0f);

    auto black_stones = board.get_all_stones(StoneColor::BLACK);
    for (const auto& [x, y] : black_stones) {
        int idx = y * 19 + x;
        if (idx >= 0 && idx < 361) {
            input[idx] = 1.0f;
        }
    }

    auto white_stones = board.get_all_stones(StoneColor::WHITE);
    for (const auto& [x, y] : white_stones) {
        int idx = y * 19 + x;
        if (idx >= 0 && idx < 361) {
            input[idx] = -1.0f;
        }
    }

    return input;
}

std::shared_ptr<CAMOE> create_ca_moe() {
    return std::make_shared<CAMOE>();
}

} // namespace go
} // namespace spacego
