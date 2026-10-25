#include "cuda_board_common.cuh"

namespace spacego {
namespace go {

__device__ __constant__ const int d_go_board_size = CUDA_BOARD_SIZE;
__device__ __constant__ const int d_total_points = CUDA_TOTAL_POINTS;
__device__ __constant__ const int d_bitboard_words = CUDA_BITBOARD_WORDS;

__global__ void batch_calculate_liberties_kernel(
    const uint64_t* black_bits, const uint64_t* white_bits,
    int* liberty_counts, size_t batch_size) {

    int board_idx = blockIdx.x;
    if (board_idx >= batch_size) return;

    liberty_counts[board_idx] = 0;
}

__global__ void batch_check_legal_moves_kernel(
    const uint64_t* black_bits, const uint64_t* white_bits,
    const int* moves, bool* legal_flags,
    StoneColor player, size_t batch_size, size_t moves_per_board) {

    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= batch_size * moves_per_board) return;

    int board_idx = global_idx / moves_per_board;
    int move_idx = global_idx % moves_per_board;
    int move = moves[global_idx];

    if (move < 0 || move >= d_total_points) {
        legal_flags[global_idx] = false;
        return;
    }

    int x = move % d_go_board_size;
    int y = move / d_go_board_size;

    if (!is_empty_device(black_bits, white_bits, board_idx, x, y, d_go_board_size, d_bitboard_words)) {
        legal_flags[global_idx] = false;
        return;
    }

    legal_flags[global_idx] = true;
}

__global__ void batch_generate_legal_moves_kernel(
    const uint64_t* black_bits, const uint64_t* white_bits,
    int* legal_moves, int player, size_t batch_size) {

    int board_idx = blockIdx.x;
    if (board_idx >= batch_size) return;

    __shared__ int shared_moves[361];
    __shared__ int shared_move_count;

    if (threadIdx.x == 0) {
        shared_move_count = 0;
    }
    __syncthreads();

    for (int pos = threadIdx.x; pos < d_total_points; pos += blockDim.x) {
        int x = pos % d_go_board_size;
        int y = pos / d_go_board_size;

        if (is_empty_device(black_bits, white_bits, board_idx, x, y, d_go_board_size, d_bitboard_words)) {
            int idx = atomicAdd(&shared_move_count, 1);
            if (idx < d_total_points) {
                shared_moves[idx] = pos;
            }
        }
    }
    __syncthreads();

    if (threadIdx.x < shared_move_count) {
        legal_moves[board_idx * d_total_points + threadIdx.x] = shared_moves[threadIdx.x];
    }

    for (int i = threadIdx.x + shared_move_count; i < d_total_points; i += blockDim.x) {
        legal_moves[board_idx * d_total_points + i] = -1;
    }
}

namespace cuda_kernels {

void launch_batch_calculate_liberties(
    const uint64_t* black_bits, const uint64_t* white_bits,
    int* liberty_counts, size_t batch_size) {

    dim3 block_dim(64);
    dim3 grid_dim(static_cast<int>(batch_size));

    batch_calculate_liberties_kernel<<<grid_dim, block_dim>>>(
        black_bits, white_bits, liberty_counts, batch_size);

    cudaDeviceSynchronize();
}

void launch_batch_check_legal_moves(
    const uint64_t* black_bits, const uint64_t* white_bits,
    const int* moves, bool* legal_flags,
    int player, size_t batch_size, size_t moves_per_board) {

    int threads_per_block = 64;
    size_t total_work = batch_size * moves_per_board;
    int blocks = static_cast<int>((total_work + threads_per_block - 1) / threads_per_block);

    batch_check_legal_moves_kernel<<<blocks, threads_per_block>>>(
        black_bits, white_bits, moves, legal_flags,
        static_cast<StoneColor>(player), batch_size, moves_per_board);

    cudaDeviceSynchronize();
}

void launch_batch_generate_legal_moves(
    const uint64_t* black_bits, const uint64_t* white_bits,
    int* legal_moves, int player, size_t batch_size) {

    int threads_per_block = 64;
    dim3 block_dim(threads_per_block);
    dim3 grid_dim(static_cast<int>(batch_size));

    batch_generate_legal_moves_kernel<<<grid_dim, block_dim>>>(
        black_bits, white_bits, legal_moves, player, batch_size);

    cudaDeviceSynchronize();
}

}

} // namespace go
} // namespace spacego
