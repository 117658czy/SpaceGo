#pragma once

#include <cstddef>

// 平台配置头文件 - 用于 CPU/CUDA 条件编译

// 检测 CUDA 编译环境
#if defined(__CUDACC__) || defined(USE_CUDA)
    #define SPACEGO_CUDA_ENABLED 1
#else
    #define SPACEGO_CUDA_ENABLED 0
#endif

// 检测 LibTorch 环境
#ifdef USE_LIBTORCH
    #define SPACEGO_LIBTORCH_ENABLED 1
#else
    #define SPACEGO_LIBTORCH_ENABLED 0
#endif

// 检测异构计算环境
#ifdef USE_HETEROGENEOUS
    #define SPACEGO_HETEROGENEOUS_ENABLED 1
#else
    #define SPACEGO_HETEROGENEOUS_ENABLED 0
#endif

// 平台相关配置
#if SPACEGO_CUDA_ENABLED
    #include <cuda_runtime.h>
    
    // CUDA 错误检查宏
    #define CUDA_CHECK(call) \
        do { \
            cudaError_t err = call; \
            if (err != cudaSuccess) { \
                throw std::runtime_error(std::string("CUDA 错误：") + \
                    cudaGetErrorString(err) + " at " + __FILE__ + ":" + \
                    std::to_string(__LINE__)); \
            } \
        } while(0)
    
    // CUDA 设备同步宏
    #define CUDA_SYNC() CUDA_CHECK(cudaDeviceSynchronize())
#else
    // 非 CUDA 环境下的空宏
    #define CUDA_CHECK(call) ((void)0)
    #define CUDA_SYNC() ((void)0)
#endif

// 批处理配置
namespace spacego {
namespace config {

// 默认批次大小
constexpr size_t DEFAULT_BATCH_SIZE = 256;

// 棋盘配置
constexpr int BOARD_SIZE = 19;
constexpr int TOTAL_POINTS = BOARD_SIZE * BOARD_SIZE;
constexpr int BITBOARD_WORDS = (TOTAL_POINTS + 63) / 64;

// 搜索配置
constexpr int DEFAULT_MAX_ITERATIONS = 10000;
constexpr float DEFAULT_EXPLORATION_WEIGHT = 1.0f;

// GTP 配置
constexpr int GTP_DEFAULT_PORT = 12345;
constexpr size_t GTP_MAX_COMMAND_LENGTH = 4096;

// 超时配置
constexpr int DEFAULT_TIME_LIMIT_MS = 30000;  // 30 秒

}  // namespace config
}  // namespace spacego
