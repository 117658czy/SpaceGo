#pragma once

#include <cstdint>
#include <cstring>
#include <array>

namespace spacego {
namespace go {

constexpr int GO_BOARD_SIZE = 19;
constexpr int TOTAL_POINTS = GO_BOARD_SIZE * GO_BOARD_SIZE;
constexpr int BITBOARD_WORDS = (TOTAL_POINTS + 63) / 64;

enum class StoneColor : uint8_t {
    EMPTY = 0,
    BLACK = 1,
    WHITE = 2,
    BORDER = 3
};

inline StoneColor opponent_color(StoneColor color) {
    return (color == StoneColor::BLACK) ? StoneColor::WHITE : StoneColor::BLACK;
}

// 价值视角约定（全工程统一）：模型 final_value 为"黑棋视角"（+1=黑胜，-1=白胜）。
// 搜索把行棋方以 static_cast<int>(StoneColor) 传入回调（BLACK=1, WHITE=2）；
// 转成"行棋方视角"只需在白方行棋时取反。
// 集中到本函数，杜绝再写 current_player == -1 —— WHITE 的枚举值是 2，该判断永不成立，
// 会让白方价值从未被正确转换（实测留出集上黑方行棋命中4.7%、白方79.6%，即模型被当成白视角）。
inline bool is_white_player(int player) {
    return player == static_cast<int>(StoneColor::WHITE);
}

#pragma warning(push)
#pragma warning(disable: 4324)
struct alignas(16) BoardState {
    uint64_t black_bits[BITBOARD_WORDS];
    uint64_t white_bits[BITBOARD_WORDS];
    uint64_t zobrist_hash;
    uint16_t move_count;
    uint8_t current_player;
    uint8_t ko_position;
    uint16_t capture_count[2];
    uint8_t reserved[6];

    BoardState() {
        std::memset(black_bits, 0, sizeof(black_bits));
        std::memset(white_bits, 0, sizeof(white_bits));
        zobrist_hash = 0;
        move_count = 0;
        current_player = static_cast<uint8_t>(StoneColor::BLACK);
        ko_position = 255;
        capture_count[0] = 0;
        capture_count[1] = 0;
        std::memset(reserved, 0, sizeof(reserved));
    }

    BoardState(const BoardState& other) {
        std::memcpy(this, &other, sizeof(BoardState));
    }

    BoardState& operator=(const BoardState& other) {
        if (this != &other) {
            std::memcpy(this, &other, sizeof(BoardState));
        }
        return *this;
    }

    BoardState(BoardState&& other) noexcept {
        std::memcpy(this, &other, sizeof(BoardState));
    }

    BoardState& operator=(BoardState&& other) noexcept {
        if (this != &other) {
            std::memcpy(this, &other, sizeof(BoardState));
        }
        return *this;
    }

    bool operator==(const BoardState& other) const {
        return std::memcmp(black_bits, other.black_bits, sizeof(black_bits)) == 0 &&
               std::memcmp(white_bits, other.white_bits, sizeof(white_bits)) == 0;
    }

    bool operator!=(const BoardState& other) const {
        return !(*this == other);
    }

    static constexpr size_t size() { return sizeof(BoardState); }
};
#pragma warning(pop)

struct HistoryEntry {
    uint16_t move;
    uint8_t captured_count;
    uint16_t captured_positions[4];
    uint64_t prev_hash;
    uint8_t prev_ko_position;

    HistoryEntry() : move(0), captured_count(0), prev_hash(0), prev_ko_position(255) {
        captured_positions[0] = 0;
        captured_positions[1] = 0;
        captured_positions[2] = 0;
        captured_positions[3] = 0;
    }
};

struct KoState {
    uint64_t zobrist_hash;
    int capture_pos;
    int last_move;
    bool is_ko;

    KoState() : zobrist_hash(0), capture_pos(-1), last_move(-1), is_ko(false) {}
};

struct BatchBoardState {
    uint64_t* black_bits;
    uint64_t* white_bits;
    uint64_t* hashes;
    uint8_t* current_players;
    uint8_t* ko_positions;
    size_t batch_size;
};

}
}
