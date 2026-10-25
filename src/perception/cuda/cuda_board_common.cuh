#pragma once

#include "go/board/board_state.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace spacego {
namespace go {

#define CUDA_BOARD_SIZE GO_BOARD_SIZE
#define CUDA_TOTAL_POINTS TOTAL_POINTS
#define CUDA_BITBOARD_WORDS BITBOARD_WORDS

__device__ inline bool is_valid_coordinate(int x, int y, int board_size) {
    return x >= 0 && x < board_size && y >= 0 && y < board_size;
}

__device__ inline int xy_to_index_device(int x, int y, int board_size) {
    if (!is_valid_coordinate(x, y, board_size)) return -1;
    return y * board_size + x;
}

__device__ inline StoneColor get_stone_device(
    const uint64_t* __restrict__ black_bits,
    const uint64_t* __restrict__ white_bits,
    int board_idx, int x, int y,
    int board_size, int bitboard_words) {

    int idx = xy_to_index_device(x, y, board_size);
    if (idx == -1) return StoneColor::BORDER;

    int abs_idx = board_idx * bitboard_words + (idx / 64);
    int bit_idx = idx % 64;
    uint64_t mask = 1ULL << bit_idx;

    if (black_bits[abs_idx] & mask) {
        return StoneColor::BLACK;
    } else if (white_bits[abs_idx] & mask) {
        return StoneColor::WHITE;
    } else {
        return StoneColor::EMPTY;
    }
}

__device__ inline bool is_empty_device(
    const uint64_t* __restrict__ black_bits,
    const uint64_t* __restrict__ white_bits,
    int board_idx, int x, int y,
    int board_size, int bitboard_words) {

    return get_stone_device(black_bits, white_bits, board_idx, x, y, board_size, bitboard_words) == StoneColor::EMPTY;
}

} // namespace go
} // namespace spacego
