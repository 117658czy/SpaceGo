#pragma once

// ============================================================================
// 异构计算引擎 - 核心数据结构与类型定义
// 命名空间: spacego::go::heterogeneous
// 功能: 定义异构计算（CPU+GPU）调度所需的所有枚举、结构体和配置
// ============================================================================

#include "config/platform_config.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

// Windows API 需要先定义版本宏，确保使用最新API
// 注意：在恢复WIN32_LEAN_AND_MEAN状态，避免影响其他头文件
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#define SPACEGO_TEMP_DEFINED_LEAN_AND_MEAN
#endif
#include <windows.h>
#ifdef SPACEGO_TEMP_DEFINED_LEAN_AND_MEAN
#undef WIN32_LEAN_AND_MEAN
#undef SPACEGO_TEMP_DEFINED_LEAN_AND_MEAN
#endif
#endif

namespace spacego {
namespace go {
namespace heterogeneous {

// ============================================================================
// 1. ComputeDeviceType - 计算设备类型枚举
// ============================================================================

/**
 * @brief 标识异构计算系统中的设备类型
 *
 * 用于在CPU、GPU和自动选择三种模式间进行切换。
 * AUTO模式允许系统根据任务特征自动选择最优设备。
 */
enum class ComputeDeviceType : uint8_t {
    CPU = 0,   // CPU设备 - 适合串行密集、逻辑复杂、数据量小的任务
    GPU = 1,   // GPU设备 - 适合大规模并行计算、矩阵运算、神经网络推理
    AUTO = 2   // 自动选择 - 由异构调度器根据任务特征和设备状态动态决策
};

// ============================================================================
// 2. DeviceCapability - 设备能力描述结构体
// ============================================================================

/**
 * @brief 描述计算设备的硬件能力和特性
 *
 * 用于异构调度器进行设备选择决策。包含计算能力、内存带宽、
 * 存储容量以及精度支持等关键指标。
 *
 * CPU和GPU的能力通过各自的detect方法自动检测。
 */
struct DeviceCapability {
    // ---- 基本设备标识 ----
    ComputeDeviceType device_type;           // 设备类型（CPU/GPU/AUTO）

    // ---- 计算能力指标 ----
    int    compute_cores;                    // 计算核心数
                                             //   CPU: 逻辑核心数 (含超线程)
                                             //   GPU: 流多处理器(SM)数量
    size_t memory_bandwidth_gb_per_sec;      // 内存带宽 (GB/s)
                                             //   CPU: 主存带宽估算值
                                             //   GPU: HBM/GDDR 理论峰值带宽
    double compute_power_tflops;             // 理论峰值计算能力 (TFLOPS, FP32)

    // ---- 内存指标 ----
    size_t total_memory_mb;                  // 总物理内存 (MB)
    size_t available_memory_mb;              // 当前可用内存 (MB)
                                             //   CPU: 系统可用RAM
                                             //   GPU: 设备可用显存

    // ---- GPU专属指标 (CPU时置为0或false) ----
    int  max_threads_per_block;              // GPU每线程块最大线程数
                                             //   CPU: 固定为0（无此概念）
    bool supports_fp16;                      // 是否支持FP16（半精度）计算
                                             //   GPU: 计算能力 >= 5.3 时支持
                                             //   CPU: 可通过软件模拟，设为false
    bool supports_tensor_core;               // 是否支持Tensor Core加速
                                             //   GPU: 计算能力 >= 7.0 时支持
                                             //   CPU: 不支持，设为false

    // ---- 附加描述信息 ----
    std::string device_name;                 // 设备名称
                                             //   CPU: 如 "Intel Core i7-12700H"
                                             //   GPU: 如 "NVIDIA GeForce RTX 3060"

    // ========================================================================
    // 默认构造函数 - 将所有字段初始化为安全默认值
    // ========================================================================
    DeviceCapability()
        : device_type(ComputeDeviceType::CPU)
        , compute_cores(0)
        , memory_bandwidth_gb_per_sec(0)
        , compute_power_tflops(0.0)
        , total_memory_mb(0)
        , available_memory_mb(0)
        , max_threads_per_block(0)
        , supports_fp16(false)
        , supports_tensor_core(false)
        , device_name("未知设备")
    {
    }

    // ========================================================================
    // detect_cpu - 自动检测当前CPU硬件能力
    // ========================================================================
    /**
     * @brief 自动检测当前系统的CPU硬件能力
     *
     * 检测内容包括：
     *   1. 逻辑核心数（含超线程）- 通过 std::thread::hardware_concurrency()
     *   2. 总物理内存 - 通过 Windows GlobalMemoryStatusEx
     *   3. 可用内存 - 同上
     *   4. 内存带宽 - 根据DDR类型估算（DDR4-3200 约 25.6 GB/s/通道，
     *      双通道约 51.2 GB/s。由于无法精确检测，使用保守估算值）
     *   5. 理论FP32算力 - 粗略估算：
     *      核心数 * 基准频率 * 每周期FP32操作数(假设AVX2: 16 FLOP/cycle)
     *      由于频率动态变化，采用保守估算
     *
     * @return DeviceCapability 包含CPU能力信息的结构体
     */
    static DeviceCapability detect_cpu()
    {
        DeviceCapability cap;
        cap.device_type = ComputeDeviceType::CPU;

        // ---- 1. 检测逻辑核心数 ----
        cap.compute_cores = static_cast<int>(
            std::thread::hardware_concurrency()
        );
        // 如果检测失败（返回0），使用保守默认值
        if (cap.compute_cores <= 0) {
            cap.compute_cores = 4;
        }

        // ---- 2&3. 检测内存信息 ----
#ifdef _WIN32
        {
            MEMORYSTATUSEX mem_info;
            mem_info.dwLength = sizeof(MEMORYSTATUSEX);
            if (GlobalMemoryStatusEx(&mem_info)) {
                cap.total_memory_mb     = mem_info.ullTotalPhys     / (1024 * 1024);
                cap.available_memory_mb = mem_info.ullAvailPhys     / (1024 * 1024);
            } else {
                // API 调用失败时的保守默认值
                cap.total_memory_mb     = 8192;   // 假设8GB
                cap.available_memory_mb = 4096;   // 假设4GB可用
            }
        }
#else
        // 非Windows平台：使用保守默认值
        cap.total_memory_mb     = 8192;
        cap.available_memory_mb = 4096;
#endif

        // ---- 4. 估算内存带宽 ----
        // DDR4-3200 双通道: 25.6 GB/s * 2 = 51.2 GB/s
        // 由于无法精确检测，使用保守估算值
        cap.memory_bandwidth_gb_per_sec = 40;  // 保守估算 40 GB/s

        // ---- 5. 估算理论FP32算力 ----
        // 假设每个核心每个周期可执行16个FP32操作（AVX2 FMA: 8*2）
        // 假设基准频率约3.0 GHz（保守估算）
        // 算力 = 核心数 * 频率(GHz) * FLOPs/cycle / 1000
        double estimated_ghz = 3.0;  // 保守估算频率
        cap.compute_power_tflops =
            static_cast<double>(cap.compute_cores) * estimated_ghz * 16.0 / 1000.0;

        // ---- 6. GPU专属字段设为0/false ----
        cap.max_threads_per_block = 0;
        cap.supports_fp16         = false;
        cap.supports_tensor_core  = false;

        // ---- 7. 设置设备名称 ----
#ifdef _WIN32
        cap.device_name = "CPU (" + std::to_string(cap.compute_cores)
                        + " 核心, " + std::to_string(cap.total_memory_mb)
                        + " MB RAM)";
#else
        cap.device_name = "CPU (" + std::to_string(cap.compute_cores)
                        + " cores, " + std::to_string(cap.total_memory_mb)
                        + " MB RAM)";
#endif

        return cap;
    }

#if SPACEGO_CUDA_ENABLED
    // ========================================================================
    // detect_gpu - 自动检测指定GPU设备硬件能力（仅CUDA环境可用）
    // ========================================================================
    /**
     * @brief 自动检测指定ID的NVIDIA GPU硬件能力
     *
     * 通过CUDA Runtime API查询GPU设备的完整硬件信息，包括：
     *   1. SM数量和核心数
     *   2. 显存大小和带宽
     *   3. 理论FP32算力（基于SM数、频率和CUDA核心数）
     *   4. FP16和Tensor Core支持情况
     *   5. 设备名称和线程块限制
     *
     * @param device_id CUDA设备索引（默认0，即第一块GPU）
     * @return DeviceCapability 包含GPU能力信息的结构体
     */
    static DeviceCapability detect_gpu(int device_id = 0)
    {
        DeviceCapability cap;
        cap.device_type = ComputeDeviceType::GPU;

        // 获取设备总数，验证device_id有效性
        int device_count = 0;
        cudaError_t count_err = cudaGetDeviceCount(&device_count);
        if (count_err != cudaSuccess || device_id >= device_count) {
            // 设备不可用时返回保守默认值
            cap.compute_cores               = 0;
            cap.total_memory_mb             = 0;
            cap.available_memory_mb         = 0;
            cap.memory_bandwidth_gb_per_sec = 0;
            cap.compute_power_tflops        = 0.0;
            cap.max_threads_per_block       = 0;
            cap.supports_fp16               = false;
            cap.supports_tensor_core        = false;
            cap.device_name = "GPU 不可用 (错误: "
                            + std::string(cudaGetErrorString(count_err)) + ")";
            return cap;
        }

        // 查询指定设备属性
        cudaDeviceProp prop;
        cudaError_t prop_err = cudaGetDeviceProperties(&prop, device_id);
        if (prop_err != cudaSuccess) {
            cap.device_name = "GPU 查询失败 (错误: "
                            + std::string(cudaGetErrorString(prop_err)) + ")";
            return cap;
        }

        // ---- 1. 计算核心数 ----
        // GPU的SM（流多处理器）数量
        cap.compute_cores = prop.multiProcessorCount;

        // ---- 2. 显存大小 ----
        cap.total_memory_mb = prop.totalGlobalMem / (1024 * 1024);

        // 查询当前可用显存
        size_t free_mem = 0;
        size_t total_mem = 0;
        cudaError_t mem_err = cudaMemGetInfo(&free_mem, &total_mem);
        if (mem_err == cudaSuccess) {
            cap.available_memory_mb = free_mem / (1024 * 1024);
        } else {
            // 查询失败时使用总显存的80%作为保守估算
            cap.available_memory_mb =
                static_cast<size_t>(cap.total_memory_mb * 0.8);
        }

        // ---- 3. 内存带宽 ----
        // 带宽(GB/s) = 显存时钟频率(kHz) * 总线宽度(bits) * 2 / 8 / 1e6
        // (*2 是因为DDR在时钟上升沿和下降沿都传输数据)
        cap.memory_bandwidth_gb_per_sec = static_cast<size_t>(
            static_cast<double>(prop.memoryClockRate)   // kHz
            * static_cast<double>(prop.memoryBusWidth)  // bits
            * 2.0                                       // DDR double data rate
            / 8.0                                       // bits -> bytes
            / 1e6                                       // per second -> GB/s
        );

        // ---- 4. 理论FP32算力 ----
        // TFLOPS = SM数 * 每SM的CUDA核心数 * 核心频率(GHz) * 2(FMA) / 1000
        // 每SM的CUDA核心数随架构变化：
        //   Volta/Turing: 64 FP32 cores/SM
        //   Ampere: 128 FP32 cores/SM (其中64个可同时执行INT32)
        //   Hopper: 128 FP32 cores/SM
        // 使用 prop.major 判断架构版本
        int cores_per_sm = 64;  // 默认: Volta/Turing 64核/SM
        if (prop.major >= 8) {
            // Ampere (8.x) 和 Hopper (9.x) 有128 FP32核/SM
            cores_per_sm = 128;
        }
        double gpu_clock_ghz = static_cast<double>(prop.clockRate)
                             / 1e6;  // kHz -> GHz
        cap.compute_power_tflops =
            static_cast<double>(prop.multiProcessorCount)
            * static_cast<double>(cores_per_sm)
            * gpu_clock_ghz
            * 2.0       // FMA: 一次乘加 = 2 FLOP
            / 1000.0;   // GFLOP -> TFLOPS

        // ---- 5. 每块最大线程数 ----
        cap.max_threads_per_block = prop.maxThreadsPerBlock;

        // ---- 6. FP16 支持检测 ----
        // 计算能力 >= 5.3 的GPU原生支持FP16
        int compute_capability = prop.major * 10 + prop.minor;
        cap.supports_fp16 = (compute_capability >= 53);

        // ---- 7. Tensor Core 支持检测 ----
        // 计算能力 >= 7.0 (Volta) 开始支持Tensor Core
        cap.supports_tensor_core = (prop.major >= 7);

        // ---- 8. 设备名称 ----
        cap.device_name = std::string(prop.name)
                        + " (SM: " + std::to_string(cap.compute_cores)
                        + ", 显存: " + std::to_string(cap.total_memory_mb)
                        + " MB)";

        return cap;
    }
#endif  // SPACEGO_CUDA_ENABLED
};

// ============================================================================
// 3. MemoryAccessPattern - 内存访问模式枚举
// ============================================================================

/**
 * @brief 描述任务对内存的访问模式
 *
 * 不同的访存模式对CPU缓存和GPU合并访问的效率有显著影响：
 *   - 顺序访问：对CPU预取器和GPU合并访问最友好
 *   - 跨步访问：步长较小时仍可以利用缓存行
 *   - 随机访问：对缓存极不友好，需要高内存带宽
 *   - 混合模式：根据实际模式选择最优设备
 */
enum class MemoryAccessPattern : uint8_t {
    SEQUENTIAL = 0,  // 顺序访问 - 连续地址读写，缓存命中率高
    STRIDED    = 1,  // 跨步访问 - 固定间隔地址读写，如矩阵列访问
    RANDOM     = 2,  // 随机访问 - 不可预测地址读写，如哈希表查找
    MIXED      = 3   // 混合模式 - 多种访存模式组合
};

// ============================================================================
// 4. TaskFeature - 任务特征描述结构体（四维特征向量）
// ============================================================================

/**
 * @brief 描述计算任务的四维特征，用于设备亲和度分析
 *
 * 四个维度的特征共同决定任务更适合在CPU还是GPU上执行：
 *
 *   matrix_density (矩阵密度):
 *     0.0 = 极度稀疏（如哈希查找），CPU更优
 *     1.0 = 完全密集（如矩阵乘法），GPU更优
 *
 *   data_scale (数据规模):
 *     小规模任务（<1000元素）：CPU开销更小
 *     大规模任务（>100000元素）：GPU并行优势明显
 *
 *   mem_pattern (访存模式):
 *     顺序/规则访问 -> GPU合并访问高效
 *     随机访问      -> CPU缓存层次可能更优
 *
 *   parallelism_degree (内在并行度):
 *     0.0 = 完全串行（如递归下降解析器），CPU更优
 *     1.0 = 完全并行（如逐元素向量运算），GPU更优
 */
struct TaskFeature {
    // ---- 四维核心特征 ----
    double              matrix_density;     // 矩阵/张量密度 (0.0=稀疏, 1.0=密集)
                                            // 高密度任务适合GPU的SIMT架构
    size_t              data_scale;         // 数据规模（元素数量）
                                            // 超过GPU阈值的数据量才能抵消传输开销
    MemoryAccessPattern mem_pattern;        // 访存模式
                                            // 影响缓存利用率和合并访问效率
    double              parallelism_degree; // 内在并行度 (0.0=串行, 1.0=完全并行)
                                            // 高并行度任务在GPU上可充分利用数千核心

    // ========================================================================
    // 默认构造函数 - 初始化为中性默认值
    // ========================================================================
    TaskFeature()
        : matrix_density(0.5)
        , data_scale(1024)
        , mem_pattern(MemoryAccessPattern::SEQUENTIAL)
        , parallelism_degree(0.5)
    {
    }
};

// ============================================================================
// 5. DeviceAffinity - 设备亲和度评估结果结构体
// ============================================================================

/**
 * @brief 描述任务与各计算设备的亲和度评估结果
 *
 * 异构调度器根据TaskFeature计算任务对CPU和GPU的亲和度分数，
 * 并给出推荐设备和置信度。
 *
 * 亲和度分数范围 [0.0, 1.0]：
 *   - 1.0 表示任务非常适合在该设备上执行
 *   - 0.0 表示任务完全不适合在该设备上执行
 *
 * 决策逻辑（由调度器实现）：
 *   1. 比较 cpu_affinity_score 和 gpu_affinity_score
 *   2. 结合当前设备负载状态
 *   3. 考虑数据传输开销
 *   4. 输出推荐设备和置信度
 */
struct DeviceAffinity {
    double             cpu_affinity_score;    // CPU亲和度分数 (0.0~1.0)
                                              // 越高表示任务越适合CPU执行
    double             gpu_affinity_score;    // GPU亲和度分数 (0.0~1.0)
                                              // 越高表示任务越适合GPU执行
    ComputeDeviceType  recommended_device;     // 推荐执行设备
    double             confidence;            // 推荐置信度 (0.0~1.0)
                                              // 1.0 = 完全确定，0.0 = 完全不确定

    // ========================================================================
    // 默认构造函数 - 初始化为未决策状态
    // ========================================================================
    DeviceAffinity()
        : cpu_affinity_score(0.0)
        , gpu_affinity_score(0.0)
        , recommended_device(ComputeDeviceType::AUTO)
        , confidence(0.0)
    {
    }
};

// ============================================================================
// 6. HeterogeneousComputeConfig - 异构计算全局配置结构体
// ============================================================================

/**
 * @brief 异构计算引擎的全局配置参数
 *
 * 控制异构计算系统的核心行为，包括：
 *   - 任务调度阈值（GPU过载/CPU空闲判断）
 *   - 负载均衡策略参数（EMA平滑因子）
 *   - 队列和批次大小限制
 *   - 自适应策略开关
 *
 * 所有参数均有经过调优的默认值，可根据实际硬件环境调整。
 */
struct HeterogeneousComputeConfig {
    // ---- 功能开关 ----
    bool   enable_heterogeneous;          // 是否启用异构计算
                                          // false时退化为纯CPU或纯GPU模式

    // ---- 负载阈值控制 ----
    double gpu_overload_threshold;        // GPU过载阈值 (默认0.85)
                                          // GPU利用率超过此值，新任务转向CPU
    double cpu_overload_threshold;        // CPU过载阈值 (默认0.90)
                                          // CPU利用率超过此值，停止接收额外任务
    double cpu_idle_threshold;            // CPU空闲阈值 (默认0.40)
                                          // CPU利用率低于此值，可从GPU分担任务

    // ---- 负载均衡参数 ----
    double load_balance_smooth_factor;    // 负载均衡平滑因子 (EMA alpha, 默认0.3)
                                          // 使用指数移动平均平滑负载读数，避免抖动
                                          // 值越大对新数据越敏感，值越小越稳定

    // ---- 批次与队列限制 ----
    size_t min_batch_for_gpu;             // GPU最小批次大小 (默认16)
                                          // 小于此值的任务直接在CPU执行，
                                          // 避免GPU启动开销大于计算收益
    size_t cpu_task_queue_max_size;       // CPU任务队列最大大小
                                          // 超过此值的新任务将被拒绝或阻塞
    size_t gpu_task_queue_max_size;       // GPU任务队列最大大小
                                          // 超过此值的新任务将被拒绝或阻塞

    // ---- 监控与分片控制 ----
    size_t monitoring_interval_ms;        // 设备负载监控间隔 (毫秒)
                                          // 调度器以此频率检查CPU/GPU利用率
    size_t max_subtask_count;             // 最大子任务数量
                                          // 限制任务分片的粒度，避免过度拆分

    // ---- 自适应策略 ----
    bool   enable_adaptive_threshold;      // 是否启用自适应阈值
                                          // true时系统根据历史数据自动调整阈值
                                          // false时使用固定阈值

    // ========================================================================
    // 默认构造函数 - 使用经过调优的默认参数
    // ========================================================================
    HeterogeneousComputeConfig()
        : enable_heterogeneous(true)
        , gpu_overload_threshold(0.85)
        , cpu_overload_threshold(0.90)
        , cpu_idle_threshold(0.40)
        , load_balance_smooth_factor(0.3)
        , min_batch_for_gpu(16)
        , cpu_task_queue_max_size(1024)
        , gpu_task_queue_max_size(512)
        , monitoring_interval_ms(100)
        , max_subtask_count(64)
        , enable_adaptive_threshold(true)
    {
    }
};

// ============================================================================
// 7. SubTaskStatus - 子任务状态枚举
// ============================================================================

/**
 * @brief 标识子任务在生命周期中的当前状态
 *
 * 状态转换图:
 *   PENDING -> RUNNING -> COMPLETED
 *                      \-> FAILED
 */
enum class SubTaskStatus : uint8_t {
    PENDING   = 0,  // 待处理 - 已被创建但尚未开始执行
    RUNNING   = 1,  // 执行中 - 正在目标设备上运行
    COMPLETED = 2,  // 已完成 - 成功执行完毕
    FAILED    = 3   // 已失败 - 执行过程中发生错误
};

// ============================================================================
// 8. SubTaskPriority - 子任务优先级枚举
// ============================================================================

/**
 * @brief 定义子任务的调度优先级
 *
 * 调度器在任务队列中按优先级排序，高优先级任务优先获得计算资源。
 * CRITICAL任务会中断正在执行的低优先级任务（如果支持抢占）。
 */
enum class SubTaskPriority : uint8_t {
    LOW      = 0,  // 低优先级 - 后台任务、日志处理、非关键统计
    NORMAL   = 1,  // 普通优先级 - 默认级别，大多数计算任务
    HIGH     = 2,  // 高优先级 - 影响搜索质量的评估任务
    CRITICAL = 3   // 关键优先级 - 直接影响走子决策的核心任务
};

// ============================================================================
// 9. SubTask - 子任务结构体
// ============================================================================

/**
 * @brief 描述一个可调度的子任务单元
 *
 * 子任务是异构计算调度的基本单位。一个大任务被拆分为多个子任务，
 * 每个子任务可以独立调度到不同设备上执行。
 *
 * 关键设计：
 *   - 通过 data_offset 和 data_length 定位数据范围
 *   - 通过 dependencies 表达任务间依赖关系（DAG）
 *   - 记录三个时间戳用于性能分析和调度优化
 *   - target_device 可由调度器在提交时重写
 */
struct SubTask {
    // ---- 基本标识 ----
    size_t            subtask_id;          // 子任务唯一ID（在父任务范围内唯一）
    ComputeDeviceType target_device;       // 目标执行设备
                                           // 由调度器根据亲和度分析设置

    // ---- 调度属性 ----
    SubTaskPriority   priority;            // 调度优先级
    SubTaskStatus     status;              // 当前执行状态

    // ---- 数据定位 ----
    size_t            data_offset;         // 在父任务数据缓冲区中的起始偏移
    size_t            data_length;         // 该子任务处理的数据长度（元素数）

    // ---- 依赖关系 ----
    std::vector<size_t> dependencies;      // 依赖的前置子任务ID列表
                                           // 调度器确保所有依赖完成后才调度本任务
                                           // 空列表表示无依赖，可立即调度

    // ---- 时间戳（用于性能分析） ----
    std::chrono::system_clock::time_point submit_time;    // 提交到调度器的时间
    std::chrono::system_clock::time_point start_time;     // 实际开始执行的时间
    std::chrono::system_clock::time_point complete_time;  // 执行完成的时间

    // ========================================================================
    // 默认构造函数 - 初始化为未提交状态
    // ========================================================================
    SubTask()
        : subtask_id(0)
        , target_device(ComputeDeviceType::AUTO)
        , priority(SubTaskPriority::NORMAL)
        , status(SubTaskStatus::PENDING)
        , data_offset(0)
        , data_length(0)
        , dependencies()
        , submit_time(std::chrono::system_clock::now())
        , start_time()
        , complete_time()
    {
    }
};

}  // namespace heterogeneous
}  // namespace go
}  // namespace spacego