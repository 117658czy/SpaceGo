#include "go/board/go_board_cuda.h"
#include "go/board/zobrist_hash.h"
#include "framework/logger.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include <cstring>
#include <sstream>
#include <queue>
#include <set>
#include <algorithm>
#include <cctype>

namespace spacego {
namespace go {

// ============================================================================
// GoBoardCPU 实现 - 基于位棋盘的19路围棋棋盘CPU实现
// 使用3个位棋盘(black_bits_, white_bits_)表示棋子状态
// 支持落子、提子、劫争检测、眼位判断等围棋规则
// ============================================================================

GoBoardCPU::GoBoardCPU()
    : ko_position_(-1)
    , current_player_(StoneColor::BLACK)
    , move_count_(0)
    , zobrist_hash_(0) {
    std::memset(black_bits_, 0, sizeof(black_bits_));
    std::memset(white_bits_, 0, sizeof(white_bits_));
    capture_count_[0] = 0;
    capture_count_[1] = 0;
}

GoBoardCPU::GoBoardCPU(const std::string& state_str)
    : ko_position_(-1)
    , current_player_(StoneColor::BLACK)
    , move_count_(0)
    , zobrist_hash_(0) {
    std::memset(black_bits_, 0, sizeof(black_bits_));
    std::memset(white_bits_, 0, sizeof(white_bits_));
    capture_count_[0] = 0;
    capture_count_[1] = 0;

    for (int i = 0; i < TOTAL_POINTS && i < static_cast<int>(state_str.size()); ++i) {
        char ch = state_str[i];
        int x = i % GO_BOARD_SIZE;
        int y = i / GO_BOARD_SIZE;
        if (ch == 'B' || ch == 'b' || ch == '1') {
            set_stone(x, y, StoneColor::BLACK);
        } else if (ch == 'W' || ch == 'w' || ch == '2') {
            set_stone(x, y, StoneColor::WHITE);
        }
    }

    // 状态后缀解析 "<B|W>|<ko>|<moves>"（见to_string）——修复严重bug：
    //   原实现仅还原361字符棋盘，行棋方/劫点/手数全部丢失且 current_player_ 恒为BLACK。
    //   而 encode_to_tensor() 用通道2编码"轮到黑方行棋"，于是经 to_string→from_string
    //   往返后（搜索器、自对弈数据生成、评估全部走这条路径），模型与搜索树收到的
    //   每个局面都恒为"黑方行棋"：
    //     · 模型永远学不到"轮到谁走"（该通道在训练数据中是常量），执白时等于拿黑方的
    //       策略下白棋 → 实测棋力退化为随机（310手终局盘上仅剩1子，全部被吃）；
    //     · 搜索树根节点也总按黑方行棋展开，白方回合的策略目标是错的。
    //   旧格式（仅361字符）无后缀，保持原有默认值，向后兼容。
    if (static_cast<int>(state_str.size()) > TOTAL_POINTS) {
        char pc = state_str[TOTAL_POINTS];
        if (pc == 'B') current_player_ = StoneColor::BLACK;
        else if (pc == 'W') current_player_ = StoneColor::WHITE;

        size_t ko_sep = state_str.find('|', static_cast<size_t>(TOTAL_POINTS) + 1);
        if (ko_sep != std::string::npos) {
            size_t mv_sep = state_str.find('|', ko_sep + 1);
            try {
                if (mv_sep != std::string::npos) {
                    ko_position_ = std::stoi(
                        state_str.substr(ko_sep + 1, mv_sep - ko_sep - 1));
                    move_count_ = std::stoi(state_str.substr(mv_sep + 1));
                } else {
                    ko_position_ = std::stoi(state_str.substr(ko_sep + 1));
                }
            } catch (...) {
                // 后缀非法（异常输入）：保留默认值，不影响棋盘本身还原
            }
        }
    }
}

GoBoardCPU::GoBoardCPU(const uint64_t* black_bits, const uint64_t* white_bits)
    : ko_position_(-1)
    , current_player_(StoneColor::BLACK)
    , move_count_(0)
    , zobrist_hash_(0) {
    if (black_bits) {
        std::memcpy(black_bits_, black_bits, sizeof(black_bits_));
    } else {
        std::memset(black_bits_, 0, sizeof(black_bits_));
    }
    if (white_bits) {
        std::memcpy(white_bits_, white_bits, sizeof(white_bits_));
    } else {
        std::memset(white_bits_, 0, sizeof(white_bits_));
    }
    capture_count_[0] = 0;
    capture_count_[1] = 0;
}

int GoBoardCPU::xy_to_index(int x, int y) {
    return y * GO_BOARD_SIZE + x;
}

std::pair<int, int> GoBoardCPU::index_to_xy(int idx) {
    return {idx % GO_BOARD_SIZE, idx / GO_BOARD_SIZE};
}

StoneColor GoBoardCPU::get_stone(int x, int y) const {
    if (x < 0 || x >= GO_BOARD_SIZE || y < 0 || y >= GO_BOARD_SIZE) {
        return StoneColor::BORDER;
    }
    int idx = xy_to_index(x, y);
    int word = idx / 64;
    int bit = idx % 64;
    uint64_t mask = uint64_t(1) << bit;
    if (black_bits_[word] & mask) return StoneColor::BLACK;
    if (white_bits_[word] & mask) return StoneColor::WHITE;
    return StoneColor::EMPTY;
}

void GoBoardCPU::set_stone(int x, int y, StoneColor color) {
    if (x < 0 || x >= GO_BOARD_SIZE || y < 0 || y >= GO_BOARD_SIZE) return;
    int idx = xy_to_index(x, y);
    int word = idx / 64;
    int bit = idx % 64;
    uint64_t mask = uint64_t(1) << bit;
    black_bits_[word] &= ~mask;
    white_bits_[word] &= ~mask;
    if (color == StoneColor::BLACK) {
        black_bits_[word] |= mask;
    } else if (color == StoneColor::WHITE) {
        white_bits_[word] |= mask;
    }
}

bool GoBoardCPU::is_empty(int x, int y) const {
    return get_stone(x, y) == StoneColor::EMPTY;
}

bool GoBoardCPU::is_black(int x, int y) const {
    return get_stone(x, y) == StoneColor::BLACK;
}

bool GoBoardCPU::is_white(int x, int y) const {
    return get_stone(x, y) == StoneColor::WHITE;
}

void GoBoardCPU::find_connected_group(int x, int y, std::vector<bool>& visited,
                                       std::vector<std::pair<int, int>>& group) const {
    StoneColor color = get_stone(x, y);
    if (color == StoneColor::EMPTY || color == StoneColor::BORDER) return;

    std::queue<std::pair<int, int>> q;
    q.push({x, y});
    visited[xy_to_index(x, y)] = true;

    const int dx[] = {1, -1, 0, 0};
    const int dy[] = {0, 0, 1, -1};

    while (!q.empty()) {
        auto [cx, cy] = q.front();
        q.pop();
        group.push_back({cx, cy});

        for (int d = 0; d < 4; ++d) {
            int nx = cx + dx[d];
            int ny = cy + dy[d];
            if (nx < 0 || nx >= GO_BOARD_SIZE || ny < 0 || ny >= GO_BOARD_SIZE) continue;
            int nidx = xy_to_index(nx, ny);
            if (visited[nidx]) continue;
            if (get_stone(nx, ny) == color) {
                visited[nidx] = true;
                q.push({nx, ny});
            }
        }
    }
}

int GoBoardCPU::calculate_liberties(int x, int y) const {
    StoneColor color = get_stone(x, y);
    if (color == StoneColor::EMPTY || color == StoneColor::BORDER) return 0;

    std::set<int> liberties;
    std::vector<bool> visited(TOTAL_POINTS, false);
    std::queue<std::pair<int, int>> q;
    q.push({x, y});
    visited[xy_to_index(x, y)] = true;

    const int dx[] = {1, -1, 0, 0};
    const int dy[] = {0, 0, 1, -1};

    while (!q.empty()) {
        auto [cx, cy] = q.front();
        q.pop();

        for (int d = 0; d < 4; ++d) {
            int nx = cx + dx[d];
            int ny = cy + dy[d];
            if (nx < 0 || nx >= GO_BOARD_SIZE || ny < 0 || ny >= GO_BOARD_SIZE) continue;
            int nidx = xy_to_index(nx, ny);
            if (visited[nidx]) continue;
            StoneColor nc = get_stone(nx, ny);
            if (nc == StoneColor::EMPTY) {
                liberties.insert(nidx);
            } else if (nc == color) {
                visited[nidx] = true;
                q.push({nx, ny});
            }
        }
    }

    return static_cast<int>(liberties.size());
}

void GoBoardCPU::remove_group(int x, int y) {
    StoneColor color = get_stone(x, y);
    if (color == StoneColor::EMPTY || color == StoneColor::BORDER) return;

    std::vector<bool> visited(TOTAL_POINTS, false);
    std::vector<std::pair<int, int>> group;
    find_connected_group(x, y, visited, group);

    StoneColor opponent = opponent_color(color);
    int cap_idx = (opponent == StoneColor::BLACK) ? 0 : 1;

    for (auto& [gx, gy] : group) {
        set_stone(gx, gy, StoneColor::EMPTY);
        capture_count_[cap_idx]++;
    }
}

bool GoBoardCPU::would_capture(int x, int y, StoneColor player) const {
    StoneColor opponent = opponent_color(player);
    const int dx[] = {1, -1, 0, 0};
    const int dy[] = {0, 0, 1, -1};

    for (int d = 0; d < 4; ++d) {
        int nx = x + dx[d];
        int ny = y + dy[d];
        if (nx < 0 || nx >= GO_BOARD_SIZE || ny < 0 || ny >= GO_BOARD_SIZE) continue;
        if (get_stone(nx, ny) == opponent) {
            if (calculate_liberties(nx, ny) == 1) {
                return true;
            }
        }
    }
    return false;
}

bool GoBoardCPU::would_be_suicide(int x, int y, StoneColor player) const {
    if (would_capture(x, y, player)) return false;

    const int dx[] = {1, -1, 0, 0};
    const int dy[] = {0, 0, 1, -1};

    for (int d = 0; d < 4; ++d) {
        int nx = x + dx[d];
        int ny = y + dy[d];
        if (nx < 0 || nx >= GO_BOARD_SIZE || ny < 0 || ny >= GO_BOARD_SIZE) continue;
        StoneColor nc = get_stone(nx, ny);
        if (nc == StoneColor::EMPTY) return false;
        if (nc == player && calculate_liberties(nx, ny) > 1) return false;
    }
    return true;
}

bool GoBoardCPU::is_legal_move(int x, int y, StoneColor player) const {
    if (x < 0 || x >= GO_BOARD_SIZE || y < 0 || y >= GO_BOARD_SIZE) return false;
    if (!is_empty(x, y)) return false;
    if (is_ko_move(x, y, player)) return false;
    if (would_be_suicide(x, y, player)) return false;
    return true;
}

std::vector<std::pair<int, int>> GoBoardCPU::get_legal_moves(StoneColor player) const {
    std::vector<std::pair<int, int>> moves;
    moves.reserve(64);
    const int dx[] = {1, -1, 0, 0};
    const int dy[] = {0, 0, 1, -1};

    for (int y = 0; y < GO_BOARD_SIZE; ++y) {
        for (int x = 0; x < GO_BOARD_SIZE; ++x) {
            if (!is_empty(x, y)) continue;
            if (is_ko_move(x, y, player)) continue;

            // 快速预判（O(1)短路，语义与would_be_suicide完全等价）：
            // 邻接空点 → 落子后己方必有气，非自杀
            // 邻接己方棋子且气>1 → 与有气棋连接，非自杀
            // 仅"四周无空点、无有气己方"的罕见点（通常是贴对方气1棋）才做完整检查
            bool simple_legal = false;
            for (int d = 0; d < 4; ++d) {
                int nx = x + dx[d], ny = y + dy[d];
                if (nx < 0 || nx >= GO_BOARD_SIZE || ny < 0 || ny >= GO_BOARD_SIZE) continue;
                StoneColor nc = get_stone(nx, ny);
                if (nc == StoneColor::EMPTY) {
                    simple_legal = true;
                    break;
                }
                if (nc == player && calculate_liberties(nx, ny) > 1) {
                    simple_legal = true;
                    break;
                }
            }

            if (simple_legal) {
                moves.push_back({x, y});
            } else if (!would_be_suicide(x, y, player)) {
                // 罕见点（贴对方气1棋、角落等）：完整自杀检查
                moves.push_back({x, y});
            }
        }
    }
    return moves;
}

bool GoBoardCPU::is_ko_move(int x, int y, StoneColor player) const {
    if (ko_position_ < 0 || ko_position_ >= TOTAL_POINTS) return false;
    return xy_to_index(x, y) == ko_position_;
}

MoveResult GoBoardCPU::make_move(int x, int y, StoneColor player) {
    MoveResult result;

    if (!is_legal_move(x, y, player)) {
        return result;
    }

    // 合法性检查通过后复用快速落子逻辑（落子/提子/劫判定唯一实现）
    return make_move_fast(x, y, player);
}

MoveResult GoBoardCPU::make_move_fast(int x, int y, StoneColor player) {
    MoveResult result;

    HistoryEntry entry;
    entry.move = xy_to_index(x, y);
    entry.prev_ko = ko_position_;
    entry.prev_player = current_player_;
    entry.prev_move_count = move_count_;
    entry.prev_hash = zobrist_hash_;

    set_stone(x, y, player);

    StoneColor opponent = opponent_color(player);
    const int dx[] = {1, -1, 0, 0};
    const int dy[] = {0, 0, 1, -1};

    int captured_count = 0;
    int last_captured_pos = -1;

    for (int d = 0; d < 4; ++d) {
        int nx = x + dx[d];
        int ny = y + dy[d];
        if (nx < 0 || nx >= GO_BOARD_SIZE || ny < 0 || ny >= GO_BOARD_SIZE) continue;
        if (get_stone(nx, ny) == opponent) {
            if (calculate_liberties(nx, ny) == 0) {
                std::vector<bool> visited(TOTAL_POINTS, false);
                std::vector<std::pair<int, int>> group;
                find_connected_group(nx, ny, visited, group);
                for (auto& [gx, gy] : group) {
                    result.captured_positions.push_back(xy_to_index(gx, gy));
                    entry.captured.push_back(xy_to_index(gx, gy));
                    set_stone(gx, gy, StoneColor::EMPTY);
                    captured_count++;
                    last_captured_pos = xy_to_index(gx, gy);
                }
            }
        }
    }

    if (captured_count == 1) {
        int my_liberties = calculate_liberties(x, y);
        std::vector<bool> visited(TOTAL_POINTS, false);
        std::vector<std::pair<int, int>> group;
        find_connected_group(x, y, visited, group);
        if (group.size() == 1 && my_liberties == 1) {
            ko_position_ = last_captured_pos;
            result.is_ko = true;
            result.ko_position = last_captured_pos;
        } else {
            ko_position_ = -1;
        }
    } else {
        ko_position_ = -1;
    }

    current_player_ = opponent;
    move_count_++;
    result.success = true;

    history_.push_back(entry);

    return result;
}

bool GoBoardCPU::undo_move() {
    if (history_.empty()) return false;

    HistoryEntry entry = history_.back();
    history_.pop_back();

    int move_idx = entry.move;
    auto [mx, my] = index_to_xy(move_idx);
    set_stone(mx, my, StoneColor::EMPTY);

    StoneColor player_who_moved = (entry.prev_player == StoneColor::BLACK)
                                   ? StoneColor::BLACK : StoneColor::WHITE;
    for (int cap_pos : entry.captured) {
        auto [cx, cy] = index_to_xy(cap_pos);
        set_stone(cx, cy, opponent_color(player_who_moved));
    }

    ko_position_ = entry.prev_ko;
    current_player_ = entry.prev_player;
    move_count_ = entry.prev_move_count;
    zobrist_hash_ = entry.prev_hash;

    return true;
}

std::vector<int> GoBoardCPU::get_captured_stones(int x, int y, StoneColor player) const {
    std::vector<int> captured;
    StoneColor opponent = opponent_color(player);
    const int dx[] = {1, -1, 0, 0};
    const int dy[] = {0, 0, 1, -1};

    for (int d = 0; d < 4; ++d) {
        int nx = x + dx[d];
        int ny = y + dy[d];
        if (nx < 0 || nx >= GO_BOARD_SIZE || ny < 0 || ny >= GO_BOARD_SIZE) continue;
        if (get_stone(nx, ny) == opponent) {
            if (calculate_liberties(nx, ny) == 0) {
                std::vector<bool> visited(TOTAL_POINTS, false);
                std::vector<std::pair<int, int>> group;
                find_connected_group(nx, ny, visited, group);
                for (auto& [gx, gy] : group) {
                    captured.push_back(xy_to_index(gx, gy));
                }
            }
        }
    }
    return captured;
}

std::vector<std::pair<int, int>> GoBoardCPU::get_group(int x, int y) const {
    std::vector<bool> visited(TOTAL_POINTS, false);
    std::vector<std::pair<int, int>> group;
    find_connected_group(x, y, visited, group);
    return group;
}

std::string GoBoardCPU::to_string() const {
    std::string result;
    result.reserve(TOTAL_POINTS + 12);
    for (int y = 0; y < GO_BOARD_SIZE; ++y) {
        for (int x = 0; x < GO_BOARD_SIZE; ++x) {
            StoneColor c = get_stone(x, y);
            if (c == StoneColor::BLACK) result += 'B';
            else if (c == StoneColor::WHITE) result += 'W';
            else result += '.';
        }
    }
    // 附加状态后缀 "<B|W>|<ko>|<moves>"：把行棋方/劫点/手数一并序列化。
    // 必需性（严重bug修复）：encode_to_tensor()的通道2是"轮到黑方行棋"平面，
    //   current_player_ 若在往返中丢失，搜索与模型收到的每个局面都恒为"黑方行棋"
    //   （详见构造函数解析段的注释）；劫点则直接影响合法着法生成。
    result += (current_player_ == StoneColor::BLACK) ? 'B' : 'W';
    result += '|';
    result += std::to_string(ko_position_);
    result += '|';
    result += std::to_string(move_count_);
    return result;
}

GoBoardCPU GoBoardCPU::from_string(const std::string& state_str) {
    return GoBoardCPU(state_str);
}

bool GoBoardCPU::operator==(const GoBoardCPU& other) const {
    return std::memcmp(black_bits_, other.black_bits_, sizeof(black_bits_)) == 0 &&
           std::memcmp(white_bits_, other.white_bits_, sizeof(white_bits_)) == 0;
}

bool GoBoardCPU::operator!=(const GoBoardCPU& other) const {
    return !(*this == other);
}

size_t GoBoardCPU::hash() const {
    size_t h = 0;
    for (int i = 0; i < BITBOARD_WORDS; ++i) {
        h ^= std::hash<uint64_t>{}(black_bits_[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>{}(white_bits_[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
}

void GoBoardCPU::encode_to_tensor(float* output, int channels) const {
    if (!output) return;
    std::memset(output, 0, TOTAL_POINTS * channels * sizeof(float));
    for (int y = 0; y < GO_BOARD_SIZE; ++y) {
        for (int x = 0; x < GO_BOARD_SIZE; ++x) {
            int idx = y * GO_BOARD_SIZE + x;
            StoneColor c = get_stone(x, y);
            if (c == StoneColor::BLACK) {
                output[idx] = 1.0f;
            } else if (c == StoneColor::WHITE) {
                output[TOTAL_POINTS + idx] = 1.0f;
            }
            if (current_player_ == StoneColor::BLACK) {
                output[2 * TOTAL_POINTS + idx] = 1.0f;
            }
        }
    }
}

GoBoardCPU GoBoardCPU::decode_from_tensor(const float* input, int channels) {
    GoBoardCPU board;
    if (!input) return board;
    for (int y = 0; y < GO_BOARD_SIZE; ++y) {
        for (int x = 0; x < GO_BOARD_SIZE; ++x) {
            int idx = y * GO_BOARD_SIZE + x;
            if (input[idx] > 0.5f) {
                board.set_stone(x, y, StoneColor::BLACK);
            } else if (input[TOTAL_POINTS + idx] > 0.5f) {
                board.set_stone(x, y, StoneColor::WHITE);
            }
        }
    }
    if (input[2 * TOTAL_POINTS] > 0.5f) {
        board.current_player_ = StoneColor::BLACK;
    } else {
        board.current_player_ = StoneColor::WHITE;
    }
    return board;
}

int GoBoardCPU::get_capture_count(StoneColor player) const {
    if (player == StoneColor::BLACK) return capture_count_[0];
    if (player == StoneColor::WHITE) return capture_count_[1];
    return 0;
}

GoBoardCPU GoBoardCPU::copy() const {
    GoBoardCPU result;
    std::memcpy(result.black_bits_, black_bits_, sizeof(black_bits_));
    std::memcpy(result.white_bits_, white_bits_, sizeof(white_bits_));
    result.ko_position_ = ko_position_;
    result.current_player_ = current_player_;
    result.move_count_ = move_count_;
    result.capture_count_[0] = capture_count_[0];
    result.capture_count_[1] = capture_count_[1];
    result.zobrist_hash_ = zobrist_hash_;
    result.history_ = history_;
    return result;
}

std::vector<std::pair<int, int>> GoBoardCPU::get_all_stones(StoneColor color) const {
    std::vector<std::pair<int, int>> stones;
    for (int y = 0; y < GO_BOARD_SIZE; ++y) {
        for (int x = 0; x < GO_BOARD_SIZE; ++x) {
            if (get_stone(x, y) == color) {
                stones.push_back({x, y});
            }
        }
    }
    return stones;
}

// 中国规则面积计点（Area Scoring）
// 思路：
//   1. 遍历所有空点，用泛填充(flood fill)找出连通的空区域
//   2. 若某空区域仅与黑棋相邻 -> 归属黑方；仅与白棋相邻 -> 归属白方；
//      同时与黑白相邻（公共气/单官）-> 双方各得一半（标准面积计点简化）
//   3. 得分 = 己方棋子数 + 己方领地，黑方需贴出 komi 给白方
std::pair<float, float> GoBoardCPU::compute_area_score(float komi) const {
    std::vector<std::vector<bool>> visited(
        GO_BOARD_SIZE, std::vector<bool>(GO_BOARD_SIZE, false));

    float black_score = 0.0f;
    float white_score = 0.0f;

    // 棋子数
    for (int y = 0; y < GO_BOARD_SIZE; ++y) {
        for (int x = 0; x < GO_BOARD_SIZE; ++x) {
            if (is_black(x, y)) black_score += 1.0f;
            else if (is_white(x, y)) white_score += 1.0f;
        }
    }

    // 空区域归属判定（四个方向邻接）
    const int dx[4] = {1, -1, 0, 0};
    const int dy[4] = {0, 0, 1, -1};

    for (int y = 0; y < GO_BOARD_SIZE; ++y) {
        for (int x = 0; x < GO_BOARD_SIZE; ++x) {
            if (get_stone(x, y) != StoneColor::EMPTY || visited[static_cast<size_t>(y)][static_cast<size_t>(x)]) {
                continue;
            }

            // 泛填充当前空区域，同时收集邻接颜色
            std::queue<std::pair<int, int>> q;
            q.push({x, y});
            visited[static_cast<size_t>(y)][static_cast<size_t>(x)] = true;

            bool touches_black = false;
            bool touches_white = false;
            size_t region_size = 0;

            while (!q.empty()) {
                auto [cx, cy] = q.front();
                q.pop();
                region_size++;

                for (int d = 0; d < 4; ++d) {
                    int nx = cx + dx[d];
                    int ny = cy + dy[d];
                    if (nx < 0 || nx >= GO_BOARD_SIZE || ny < 0 || ny >= GO_BOARD_SIZE) {
                        continue;
                    }
                    StoneColor s = get_stone(nx, ny);
                    if (s == StoneColor::BLACK) {
                        touches_black = true;
                    } else if (s == StoneColor::WHITE) {
                        touches_white = true;
                    } else if (!visited[static_cast<size_t>(ny)][static_cast<size_t>(nx)]) {
                        visited[static_cast<size_t>(ny)][static_cast<size_t>(nx)] = true;
                        q.push({nx, ny});
                    }
                }
            }

            // 区域归属：仅邻接单色 -> 归属该方；同时邻接双方 -> 双方平分
            float region_val = static_cast<float>(region_size);
            if (touches_black && !touches_white) {
                black_score += region_val;
            } else if (touches_white && !touches_black) {
                white_score += region_val;
            } else if (touches_black && touches_white) {
                black_score += region_val * 0.5f;
                white_score += region_val * 0.5f;
            }
            // 空棋盘上无邻接颜色（不可能发生，围棋棋盘总有边界），忽略
        }
    }

    // 黑方贴目：黑分 - komi，白分不变（中国规则黑贴7.5目）
    black_score -= komi;
    return {black_score, white_score};
}

uint64_t GoBoardCPU::get_zobrist_hash() const {
    return zobrist_hash_;
}

void GoBoardCPU::update_zobrist_hash(int x, int y, StoneColor old_color, StoneColor new_color) {
    int idx = xy_to_index(x, y);
    if (old_color != StoneColor::EMPTY) {
        zobrist_hash_ ^= ZobristHash::get_hash(idx, old_color);
    }
    if (new_color != StoneColor::EMPTY) {
        zobrist_hash_ ^= ZobristHash::get_hash(idx, new_color);
    }
}

// ============================================================================
// GoBoardCUDA 实现 - CUDA加速的批量棋盘操作（条件编译）
// ============================================================================

#ifdef USE_CUDA

GoBoardCUDA::GoBoardCUDA()
    : batch_size_(0)
    , black_bits_device_(nullptr)
    , white_bits_device_(nullptr) {
}

GoBoardCUDA::GoBoardCUDA(size_t batch_size)
    : batch_size_(batch_size)
    , black_bits_device_(nullptr)
    , white_bits_device_(nullptr) {
    if (batch_size > 0) {
        size_t alloc_size = batch_size * BITBOARD_WORDS * sizeof(uint64_t);
        cudaError_t err;
        err = cudaMalloc(&black_bits_device_, alloc_size);
        if (err != cudaSuccess) {
            LOG_ERROR("CUDA内存分配失败(black_bits): %s", cudaGetErrorString(err));
            black_bits_device_ = nullptr;
        }
        err = cudaMalloc(&white_bits_device_, alloc_size);
        if (err != cudaSuccess) {
            LOG_ERROR("CUDA内存分配失败(white_bits): %s", cudaGetErrorString(err));
            white_bits_device_ = nullptr;
        }
    }
}

GoBoardCUDA::~GoBoardCUDA() {
    if (black_bits_device_) {
        cudaFree(black_bits_device_);
        black_bits_device_ = nullptr;
    }
    if (white_bits_device_) {
        cudaFree(white_bits_device_);
        white_bits_device_ = nullptr;
    }
}

GoBoardCUDA::GoBoardCUDA(GoBoardCUDA&& other) noexcept
    : batch_size_(other.batch_size_)
    , black_bits_device_(other.black_bits_device_)
    , white_bits_device_(other.white_bits_device_) {
    other.batch_size_ = 0;
    other.black_bits_device_ = nullptr;
    other.white_bits_device_ = nullptr;
}

GoBoardCUDA& GoBoardCUDA::operator=(GoBoardCUDA&& other) noexcept {
    if (this != &other) {
        if (black_bits_device_) cudaFree(black_bits_device_);
        if (white_bits_device_) cudaFree(white_bits_device_);

        batch_size_ = other.batch_size_;
        black_bits_device_ = other.black_bits_device_;
        white_bits_device_ = other.white_bits_device_;

        other.batch_size_ = 0;
        other.black_bits_device_ = nullptr;
        other.white_bits_device_ = nullptr;
    }
    return *this;
}

void GoBoardCUDA::upload_batch(const std::vector<GoBoardCPU>& boards) {
    if (boards.empty() || !black_bits_device_ || !white_bits_device_) return;

    size_t count = std::min(boards.size(), batch_size_);
    size_t bits_size = BITBOARD_WORDS * sizeof(uint64_t);

    std::vector<uint64_t> host_black(count * BITBOARD_WORDS, 0);
    std::vector<uint64_t> host_white(count * BITBOARD_WORDS, 0);

    for (size_t i = 0; i < count; ++i) {
        std::memcpy(host_black.data() + i * BITBOARD_WORDS, boards[i].get_black_bits(), bits_size);
        std::memcpy(host_white.data() + i * BITBOARD_WORDS, boards[i].get_white_bits(), bits_size);
    }

    cudaMemcpy(black_bits_device_, host_black.data(), count * bits_size, cudaMemcpyHostToDevice);
    cudaMemcpy(white_bits_device_, host_white.data(), count * bits_size, cudaMemcpyHostToDevice);
}

void GoBoardCUDA::download_batch(std::vector<GoBoardCPU>& boards) const {
    if (!black_bits_device_ || !white_bits_device_) return;

    size_t count = std::min(boards.size(), batch_size_);
    size_t bits_size = BITBOARD_WORDS * sizeof(uint64_t);

    std::vector<uint64_t> host_black(count * BITBOARD_WORDS, 0);
    std::vector<uint64_t> host_white(count * BITBOARD_WORDS, 0);

    cudaMemcpy(host_black.data(), black_bits_device_, count * bits_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_white.data(), white_bits_device_, count * bits_size, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < count; ++i) {
        boards[i] = GoBoardCPU(host_black.data() + i * BITBOARD_WORDS,
                                host_white.data() + i * BITBOARD_WORDS);
    }
}

size_t GoBoardCUDA::get_gpu_memory_usage() const {
    size_t usage = 0;
    if (black_bits_device_) usage += batch_size_ * BITBOARD_WORDS * sizeof(uint64_t);
    if (white_bits_device_) usage += batch_size_ * BITBOARD_WORDS * sizeof(uint64_t);
    return usage;
}

std::vector<int> GoBoardCUDA::batch_calculate_liberties() const {
    std::vector<int> results(batch_size_, 0);
    if (batch_size_ == 0 || !black_bits_device_ || !white_bits_device_) return results;

    cuda_kernels::launch_batch_calculate_liberties(
        black_bits_device_, white_bits_device_, results.data(), batch_size_);
    return results;
}

std::vector<int> GoBoardCUDA::batch_generate_legal_moves(StoneColor player) const {
    std::vector<int> results(batch_size_ * TOTAL_POINTS, 0);
    if (batch_size_ == 0 || !black_bits_device_ || !white_bits_device_) return results;

    cuda_kernels::launch_batch_generate_legal_moves(
        black_bits_device_, white_bits_device_, results.data(),
        static_cast<int>(player), batch_size_);
    return results;
}

std::vector<bool> GoBoardCUDA::batch_check_legal_moves(const std::vector<int>& moves,
                                                         StoneColor player) const {
    std::vector<bool> results(batch_size_ * moves.size(), false);
    if (batch_size_ == 0 || !black_bits_device_ || !white_bits_device_) return results;

    std::vector<int> h_moves = moves;
    // vector<bool>没有data()方法，使用vector<uint8_t>代替
    std::vector<uint8_t> h_results_uint(results.size(), 0);
    cuda_kernels::launch_batch_check_legal_moves(
        black_bits_device_, white_bits_device_, h_moves.data(), 
        reinterpret_cast<bool*>(h_results_uint.data()),
        static_cast<int>(player), batch_size_, static_cast<int>(moves.size()));
    // 转换回vector<bool>
    for (size_t i = 0; i < results.size(); ++i) {
        results[i] = (h_results_uint[i] != 0);
    }
    return results;
}

bool GoBoardCUDA::batch_submit_boards(const std::vector<void*>& boards,
                                        size_t cpu_count, size_t gpu_count) {
    return true;
}

size_t GoBoardCUDA::batch_submit_boards_auto(const std::vector<void*>& boards) {
    return 0;
}

// GoBoardCUDATools 实现
bool GoBoardCUDATools::initialize_cuda() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return false;
    }
    return true;
}

std::string GoBoardCUDATools::get_gpu_info() {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) return "GPU信息获取失败";
    return std::string(prop.name) + " (" + std::to_string(prop.totalGlobalMem / (1024*1024)) + " MB)";
}

double GoBoardCUDATools::benchmark_liberties_calculation(size_t batch_size, int iterations) {
    return 0.0;
}

double GoBoardCUDATools::benchmark_move_generation(size_t batch_size, int iterations) {
    return 0.0;
}

size_t GoBoardCUDATools::calculate_optimal_batch_size(size_t available_gpu_memory) {
    size_t per_board = 2 * BITBOARD_WORDS * sizeof(uint64_t);
    if (per_board == 0) return 1;
    return available_gpu_memory / per_board;
}

std::vector<std::string> GoBoardCUDATools::get_optimization_suggestions() {
    return {};
}

#endif // USE_CUDA

} // namespace go
} // namespace spacego
