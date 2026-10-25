#include "go/board/board_state.h"
#include "go/board/go_board_cuda.h"
#include "go/board/zobrist_hash.h"
#include "framework/logger.h"

#include <sstream>
#include <queue>
#include <set>
#include <cctype>

namespace spacego {
namespace go {

// ============================================================================
// BoardStateManager 类 - 封装 GoBoardCPU 的高层棋盘状态管理
// 提供着法历史、劫争检测、眼位检测、局面快照、死活判断等功能
// ============================================================================

class BoardStateManager {
public:
    BoardStateManager() {
        ZobristHash::initialize();
        board_ = std::make_unique<GoBoardCPU>();
        ko_state_.is_ko = false;
        LOG_INFO("BoardStateManager 初始化完成");
    }

    explicit BoardStateManager(int board_size) : board_size_(board_size) {
        ZobristHash::initialize();
        board_ = std::make_unique<GoBoardCPU>();
        ko_state_.is_ko = false;
        LOG_INFO("BoardStateManager 初始化完成，棋盘大小: %d", board_size);
    }

    // ========================================================================
    // 着法管理
    // ========================================================================

    bool push_move(int x, int y, StoneColor player) {
        if (x < 0 || x >= board_size_ || y < 0 || y >= board_size_) {
            return false;
        }

        int idx = GoBoardCPU::xy_to_index(x, y);
        int prev_ko = board_->get_ko_position();

        HistoryEntry entry;
        entry.move = static_cast<uint16_t>(idx);
        entry.prev_hash = board_->get_zobrist_hash();
        entry.prev_ko_position = static_cast<uint8_t>(prev_ko >= 0 ? prev_ko : 255);

        MoveResult result = board_->make_move(x, y, player);

        if (!result.success) {
            LOG_WARNING("着法 (%d, %d) 无效，玩家: %d", x, y, static_cast<int>(player));
            return false;
        }

        entry.captured_count = static_cast<uint8_t>(result.captured_positions.size());
        for (size_t i = 0; i < result.captured_positions.size() && i < 4; i++) {
            entry.captured_positions[i] = static_cast<uint16_t>(result.captured_positions[i]);
        }

        history_.push_back(entry);
        ko_state_.is_ko = result.is_ko;
        ko_state_.zobrist_hash = board_->get_zobrist_hash();
        ko_state_.capture_pos = result.ko_position;
        ko_state_.last_move = idx;

        LOG_DEBUG("着法 %d 成功落子 (%d, %d)，提子数: %zu，劫: %s",
                  static_cast<int>(history_.size()), x, y,
                  result.captured_positions.size(),
                  result.is_ko ? "是" : "否");

        return true;
    }

    bool pop_move() {
        if (history_.empty()) {
            LOG_WARNING("历史为空，无法撤销着法");
            return false;
        }

        board_->undo_move();
        history_.pop_back();

        if (!history_.empty()) {
            const auto& entry = history_.back();
            int prev_ko = static_cast<int>(entry.prev_ko_position);
            if (prev_ko == 255) {
                board_->set_ko_position(-1);
            } else {
                board_->set_ko_position(prev_ko);
            }
        } else {
            board_->set_ko_position(-1);
        }

        LOG_DEBUG("撤销着法成功，历史长度: %zu", history_.size());
        return true;
    }

    const std::vector<HistoryEntry>& get_history() const {
        return history_;
    }

    size_t get_history_size() const {
        return history_.size();
    }

    void clear_history() {
        history_.clear();
        board_ = std::make_unique<GoBoardCPU>();
        ko_state_ = KoState();
        LOG_INFO("历史记录已清除");
    }

    // ========================================================================
    // 劫争检测
    // ========================================================================

    bool is_ko() const {
        return board_->has_ko();
    }

    bool is_ko_move(int x, int y, StoneColor player) const {
        return board_->is_ko_move(x, y, player);
    }

    KoState get_ko_state() const {
        return ko_state_;
    }

    // ========================================================================
    // 眼位检测
    // 判断某位置是否为特定颜色的眼
    // 眼的条件:
    //   1. 位置为空
    //   2. 所有4个相邻交叉点都是该颜色棋子（角上或边上则全部检查到的都是）
    //   3. 对角线至少有3个是该颜色棋子（角/边需要更严格判断）
    // ========================================================================

    bool is_eye(int x, int y, StoneColor color) const {
        if (x < 0 || x >= board_size_ || y < 0 || y >= board_size_) {
            return false;
        }

        if (!board_->is_empty(x, y)) {
            return false;
        }

        StoneColor opponent = opponent_color(color);

        // 步骤1: 检查所有4个相邻方向是否都被己方控制
        static const int dirs[4][2] = {
            {0, 1}, {1, 0}, {0, -1}, {-1, 0}
        };

        for (int i = 0; i < 4; i++) {
            int nx = x + dirs[i][0];
            int ny = y + dirs[i][1];

            if (nx >= 0 && nx < board_size_ && ny >= 0 && ny < board_size_) {
                StoneColor adjacent = board_->get_stone(nx, ny);
                if (adjacent != color && adjacent != StoneColor::BORDER) {
                    return false;
                }
            }
        }

        // 步骤2: 检查对角线，要求该颜色棋子占据大多数
        static const int diag_dirs[4][2] = {
            {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
        };

        int friendly_diag = 0;
        int total_diag = 0;

        for (int i = 0; i < 4; i++) {
            int nx = x + diag_dirs[i][0];
            int ny = y + diag_dirs[i][1];

            if (nx >= 0 && nx < board_size_ && ny >= 0 && ny < board_size_) {
                total_diag++;
                StoneColor diag_stone = board_->get_stone(nx, ny);
                if (diag_stone == color) {
                    friendly_diag++;
                }
            }
        }

        // 边角的位置降低对角线要求
        int min_friendly = (total_diag >= 4) ? 3 : (total_diag >= 3) ? 2 : total_diag;

        return friendly_diag >= min_friendly;
    }

    // ========================================================================
    // 局面快照 - 保存和恢复完整棋盘状态
    // ========================================================================

    void save_snapshot(const std::string& name = "") {
        Snapshot snap;

        snap.name = name;
        snap.board_copy = board_->copy();
        snap.history_copy = history_;
        snap.ko_state_copy = ko_state_;

        snapshots_.push_back(std::move(snap));

        if (!name.empty()) {
            LOG_DEBUG("局面快照 \"%s\" 已保存，#%zu", name.c_str(), snapshots_.size());
        } else {
            LOG_DEBUG("局面快照已保存，#%zu", snapshots_.size());
        }
    }

    bool restore_snapshot(int index = -1) {
        if (snapshots_.empty()) {
            LOG_WARNING("没有可恢复的局面快照");
            return false;
        }

        if (index < 0 || index >= static_cast<int>(snapshots_.size())) {
            index = static_cast<int>(snapshots_.size()) - 1;
        }

        const auto& snap = snapshots_[index];
        *board_ = std::move(snap.board_copy.copy());
        history_ = snap.history_copy;
        ko_state_ = snap.ko_state_copy;

        LOG_INFO("已恢复局面快照 \"%s\" (索引: %d)", snap.name.c_str(), index);
        return true;
    }

    size_t get_snapshot_count() const {
        return snapshots_.size();
    }

    // ========================================================================
    // 死活判断
    // 使用简化版BFS检测：如果某块棋子拥有2个或以上的真眼则存活
    // ========================================================================

    bool is_alive(int x, int y) const {
        if (x < 0 || x >= board_size_ || y < 0 || y >= board_size_) {
            return false;
        }

        StoneColor stone = board_->get_stone(x, y);
        if (stone == StoneColor::EMPTY) {
            return false;
        }

        // 获获取整块棋子
        auto group = board_->get_group(x, y);
        if (group.empty()) {
            return false;
        }

        // 计算当前气数
        int liberties = board_->calculate_liberties(x, y);
        if (liberties >= 2) {
            return true;
        }

        // 查找眼位数量（该颜色棋子的空闲交叉点）
        // 统计该块棋子能够生成的"眼"
        std::set<std::pair<int, int>> eye_candidates;

        // 通过检查该块棋子的所有相邻空点
        for (const auto& pos : group) {
            static const int dirs[4][2] = {{0,1},{1,0},{0,-1},{-1,0}};
            for (int d = 0; d < 4; d++) {
                int nx = pos.first + dirs[d][0];
                int ny = pos.second + dirs[d][1];

                if (nx >= 0 && nx < board_size_ && ny >= 0 && ny < board_size_) {
                    if (board_->is_empty(nx, ny)) {
                        eye_candidates.insert({nx, ny});
                    }
                }
            }
        }

        // 统计真眼数量
        int true_eyes = 0;
        for (const auto& eye_pos : eye_candidates) {
            if (is_eye(eye_pos.first, eye_pos.second, stone)) {
                true_eyes++;
            }
        }

        return true_eyes >= 2;
    }

    // ========================================================================
    // 棋盘访问
    // ========================================================================

    GoBoardCPU* get_board() { return board_.get(); }
    const GoBoardCPU* get_board() const { return board_.get(); }

    StoneColor get_current_player() const {
        return board_->get_current_player();
    }

    int get_board_size() const { return board_size_; }
    void set_board_size(int size) {
        board_size_ = size;
        board_ = std::make_unique<GoBoardCPU>();
        history_.clear();
        snapshots_.clear();
        LOG_INFO("棋盘大小设置为 %d", size);
    }

private:
    struct Snapshot {
        std::string name;
        GoBoardCPU board_copy;
        std::vector<HistoryEntry> history_copy;
        KoState ko_state_copy;

        Snapshot() : board_copy() {}
    };

    std::unique_ptr<GoBoardCPU> board_;
    std::vector<HistoryEntry> history_;
    KoState ko_state_;
    int board_size_ = GO_BOARD_SIZE;
    std::vector<Snapshot> snapshots_;
};

// ============================================================================
// 辅助函数：着法字符串与坐标互转
// ============================================================================

// 列字母映射（围棋标准：跳过'I'以避免与'1'混淆）
static char column_to_letter(int col) {
    if (col >= 8) {
        return static_cast<char>('A' + col + 1);
    }
    return static_cast<char>('A' + col);
}

static int letter_to_column(char letter) {
    letter = static_cast<char>(std::toupper(static_cast<unsigned char>(letter)));
    int col = letter - 'A';
    if (col >= 9) {
        col--;
    }
    return col;
}

std::string move_to_string(int x, int y) {
    if (x < 0 || x >= GO_BOARD_SIZE || y < 0 || y >= GO_BOARD_SIZE) {
        return "pass";
    }

    std::string result;
    result += column_to_letter(x);
    int row = GO_BOARD_SIZE - y;
    if (row >= 10) {
        result += std::to_string(row);
    } else {
        result += static_cast<char>('0' + row);
    }
    return result;
}

bool string_to_move(const std::string& str, int& x, int& y) {
    if (str.empty() || str == "pass" || str == "PASS") {
        x = -1;
        y = -1;
        return true;
    }

    if (str.size() < 2) {
        return false;
    }

    x = letter_to_column(str[0]);
    if (x < 0 || x >= GO_BOARD_SIZE) {
        return false;
    }

    try {
        int row = std::stoi(str.substr(1));
        y = GO_BOARD_SIZE - row;
    } catch (...) {
        return false;
    }

    if (y < 0 || y >= GO_BOARD_SIZE) {
        return false;
    }

    return true;
}

int index_to_position(int index) {
    return index;
}

int position_to_index(int x, int y) {
    if (x < 0 || x >= GO_BOARD_SIZE || y < 0 || y >= GO_BOARD_SIZE) {
        return -1;
    }
    return GoBoardCPU::xy_to_index(x, y);
}

// ============================================================================
// 领地计算：Tromp-Taylor 规则简化版
// 使用泛填充(flood fill)来确定每个空点所属的领地
// 如果一块空区域仅被一种颜色的棋子包围，则该区域属于该颜色
// ============================================================================

struct TerritoryResult {
    int black_territory;
    int white_territory;
    std::vector<std::vector<int>> owner_map;
};

static TerritoryResult calculate_territory(const GoBoardCPU& board) {
    TerritoryResult result;
    result.black_territory = 0;
    result.white_territory = 0;

    const int size = GO_BOARD_SIZE;

    // owner_map: 0=未访问, 1=黑领地, 2=白领地, 3=中立, 4=实地棋子
    result.owner_map.assign(size, std::vector<int>(size, 0));

    // 将棋子位置标记为 -1 (非空区域)
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            StoneColor stone = board.get_stone(x, y);
            if (stone == StoneColor::BLACK) {
                result.owner_map[y][x] = -1;
            } else if (stone == StoneColor::WHITE) {
                result.owner_map[y][x] = -2;
            } else {
                result.owner_map[y][x] = 0;
            }
        }
    }

    // BFS泛填充每个空区域
    static const int dirs[4][2] = {{0,1},{1,0},{0,-1},{-1,0}};

    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            if (result.owner_map[y][x] != 0) {
                continue;
            }

            std::vector<std::pair<int, int>> region;
            std::queue<std::pair<int, int>> q;
            q.push({x, y});
            result.owner_map[y][x] = -3;

            bool touches_black = false;
            bool touches_white = false;

            while (!q.empty()) {
                auto [cx, cy] = q.front();
                q.pop();
                region.push_back({cx, cy});

                for (int d = 0; d < 4; d++) {
                    int nx = cx + dirs[d][0];
                    int ny = cy + dirs[d][1];

                    if (nx < 0 || nx >= size || ny < 0 || ny >= size) {
                        continue;
                    }

                    int val = result.owner_map[ny][nx];
                    if (val == -1) {
                        touches_black = true;
                    } else if (val == -2) {
                        touches_white = true;
                    } else if (val == 0) {
                        result.owner_map[ny][nx] = -3;
                        q.push({nx, ny});
                    }
                }
            }

            // 确定区域归属
            int owner = 0;
            if (touches_black && !touches_white) {
                owner = 1;
            } else if (touches_white && !touches_black) {
                owner = 2;
            } else {
                owner = 3;
            }

            for (const auto& [rx, ry] : region) {
                result.owner_map[ry][rx] = owner;
                if (owner == 1) {
                    result.black_territory++;
                } else if (owner == 2) {
                    result.white_territory++;
                }
            }
        }
    }

    return result;
}

// ============================================================================
// SGF 格式导入导出辅助函数
// ============================================================================

// SGF导出：将整局对弈导出为SGF字符串
std::string export_to_sgf(const BoardStateManager& manager) {
    std::ostringstream sgf;

    sgf << "(;GM[1]FF[4]SZ[" << manager.get_board_size()
        << "]KM[6.5]RU[Chinese]\n";

    const auto& history = manager.get_history();
    StoneColor current = StoneColor::BLACK;

    for (size_t i = 0; i < history.size(); i++) {
        const auto& entry = history[i];

        sgf << ";";

        if (current == StoneColor::BLACK) {
            sgf << "B[";
        } else {
            sgf << "W[";
        }

        int idx = static_cast<int>(entry.move);
        if (idx >= 0 && idx < TOTAL_POINTS) {
            auto [x, y] = GoBoardCPU::index_to_xy(idx);
            sgf << static_cast<char>('a' + x)
                << static_cast<char>('a' + y);
        } else {
            sgf << "tt";
        }

        sgf << "]\n";

        current = opponent_color(current);
    }

    sgf << ")\n";

    return sgf.str();
}

// SGF导入：从SGF字符串创建局面
struct SGFParseResult {
    bool success;
    std::string error_message;
    int move_count;
};

static SGFParseResult import_from_sgf(const std::string& sgf_content,
                                       BoardStateManager& manager) {
    SGFParseResult result;
    result.success = false;
    result.move_count = 0;

    if (sgf_content.empty()) {
        result.error_message = "SGF内容为空";
        return result;
    }

    manager.clear_history();

    StoneColor current = StoneColor::BLACK;
    int moves_parsed = 0;

    size_t pos = 0;
    while (pos < sgf_content.size()) {
        pos = sgf_content.find_first_of("BW", pos);
        if (pos == std::string::npos) break;

        char color_char = sgf_content[pos];
        bool is_black = (color_char == 'B');
        pos++;

        if (pos >= sgf_content.size() || sgf_content[pos] != '[') {
            continue;
        }
        pos++;

        size_t end = sgf_content.find(']', pos);
        if (end == std::string::npos) {
            break;
        }

        std::string move_str = sgf_content.substr(pos, end - pos);
        pos = end + 1;

        if (move_str.empty() || move_str == "tt") {
            current = opponent_color(current);
            moves_parsed++;
            continue;
        }

        if (move_str.size() >= 2) {
            int x = move_str[0] - 'a';
            int y = move_str[1] - 'a';

            if (x >= 0 && x < manager.get_board_size() &&
                y >= 0 && y < manager.get_board_size()) {

                StoneColor expected = is_black ? StoneColor::BLACK : StoneColor::WHITE;
                if (manager.push_move(x, y, expected)) {
                    moves_parsed++;
                }
            }
        }

        current = opponent_color(current);
    }

    result.success = (moves_parsed > 0);
    result.move_count = moves_parsed;

    if (!result.success) {
        result.error_message = "SGF解析失败：未找到有效着法";
    }

    return result;
}

// ============================================================================
// 棋盘状态工厂函数
// ============================================================================

BoardState create_initial_state() {
    BoardState state;
    return state;
}

BoardState create_state_from_board(const GoBoardCPU& board) {
    BoardState state;
    const uint64_t* black = board.get_black_bits();
    const uint64_t* white = board.get_white_bits();

    for (int i = 0; i < BITBOARD_WORDS; i++) {
        state.black_bits[i] = black[i];
        state.white_bits[i] = white[i];
    }

    state.zobrist_hash = board.get_zobrist_hash();
    state.move_count = static_cast<uint16_t>(board.get_move_count());
    state.current_player = static_cast<uint8_t>(board.get_current_player());

    int ko_pos = board.get_ko_position();
    state.ko_position = static_cast<uint8_t>(ko_pos >= 0 ? ko_pos : 255);

    state.capture_count[0] = static_cast<uint16_t>(board.get_capture_count(StoneColor::BLACK));
    state.capture_count[1] = static_cast<uint16_t>(board.get_capture_count(StoneColor::WHITE));

    return state;
}

bool apply_state_to_board(const BoardState& state, GoBoardCPU& board) {
    board = GoBoardCPU(state.black_bits, state.white_bits);
    board.set_current_player(static_cast<StoneColor>(state.current_player));
    board.set_move_count(state.move_count);

    uint8_t ko_pos = state.ko_position;
    board.set_ko_position(ko_pos != 255 ? static_cast<int>(ko_pos) : -1);

    return true;
}

} // namespace go
} // namespace spacego
