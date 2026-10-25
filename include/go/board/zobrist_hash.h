#pragma once

#include <cstdint>
#include "go/board/board_state.h"

namespace spacego {
namespace go {

class ZobristHash {
public:
    static void initialize(uint64_t seed = 0x5841F3D2A9B7E6C1ULL);
    
    static uint64_t get_hash(int position, StoneColor color) {
        if (position < 0 || position >= TOTAL_POINTS) return 0;
        int color_idx = static_cast<int>(color);
        if (color_idx < 0 || color_idx > 2) return 0;
        return table_[position][color_idx];
    }

    static uint64_t compute_initial_hash(const uint64_t* black_bits, const uint64_t* white_bits);

    static uint64_t update_hash(uint64_t current_hash, int position, 
                                StoneColor old_color, StoneColor new_color) {
        if (old_color != StoneColor::EMPTY) {
            current_hash ^= get_hash(position, old_color);
        }
        if (new_color != StoneColor::EMPTY) {
            current_hash ^= get_hash(position, new_color);
        }
        return current_hash;
    }

    static bool is_initialized() { return initialized_; }

private:
    static uint64_t table_[TOTAL_POINTS][3];
    static bool initialized_;
    
    static uint64_t splitmix64(uint64_t& state);
    static uint64_t xorshift128plus(uint64_t& s0, uint64_t& s1);
};

}
}
