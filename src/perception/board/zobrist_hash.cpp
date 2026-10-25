#include "go/board/zobrist_hash.h"
#include <random>
#include <cstring>

namespace spacego {
namespace go {

uint64_t ZobristHash::table_[TOTAL_POINTS][3];
bool ZobristHash::initialized_ = false;

uint64_t ZobristHash::splitmix64(uint64_t& state) {
    uint64_t result = state;
    state = state + 0x9E3779B97F4A7C15ULL;
    result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9ULL;
    result = (result ^ (result >> 27)) * 0x94D049BB133111EBULL;
    result = result ^ (result >> 31);
    return result;
}

uint64_t ZobristHash::xorshift128plus(uint64_t& s0, uint64_t& s1) {
    uint64_t result = s0 + s1;
    s1 ^= s0;
    s0 = ((s0 << 55) | (s0 >> 9)) ^ s1 ^ (s1 << 14);
    return result;
}

void ZobristHash::initialize(uint64_t seed) {
    if (initialized_) return;
    
    uint64_t state = seed;
    uint64_t s0 = splitmix64(state);
    uint64_t s1 = splitmix64(state);
    
    for (int pos = 0; pos < TOTAL_POINTS; ++pos) {
        for (int color = 0; color < 3; ++color) {
            table_[pos][color] = xorshift128plus(s0, s1);
        }
    }
    
    initialized_ = true;
}

uint64_t ZobristHash::compute_initial_hash(const uint64_t* black_bits, const uint64_t* white_bits) {
    if (!initialized_) {
        initialize();
    }
    
    uint64_t hash = 0;
    
    for (int pos = 0; pos < TOTAL_POINTS; ++pos) {
        int word_idx = pos / 64;
        int bit_idx = pos % 64;
        uint64_t mask = 1ULL << bit_idx;
        
        StoneColor color = StoneColor::EMPTY;
        if (black_bits[word_idx] & mask) {
            color = StoneColor::BLACK;
        } else if (white_bits[word_idx] & mask) {
            color = StoneColor::WHITE;
        }
        
        if (color != StoneColor::EMPTY) {
            hash ^= get_hash(pos, color);
        }
    }
    
    return hash;
}

}
}
