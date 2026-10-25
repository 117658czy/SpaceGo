#pragma once

#include <memory>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <functional>
#include <condition_variable>
#include <string>
#include <chrono>
#include "go/heterogeneous/heterogeneous_compute_engine.h"
#include "go/search/spatial_topology_calculator.h"

// 前置声明
namespace spacego {
namespace go {
namespace search {

class SpatialTopologyCalculator;
struct SpatialTopologyFeatures;
struct CubeState;

struct SpatialTopologyResult {
    std::vector<SpatialTopologyFeatures> features;
    bool has_error = false;
    std::string error_message;

    SpatialTopologyResult() = default;
};

} // namespace search
} // namespace go
} // namespace spacego

namespace spacego {
namespace go {
namespace search {

// ============================================================================
// 拓扑计算配置
// 控制空间拓扑加速器的计算行为，包括批次处理阈值、精度策略等参数
// ============================================================================
struct TopologyComputeConfig {
    size_t batch_threshold;         // 批次阈值：小于此值使用CPU单线程路径
    enum class PrecisionStrategy {  // 精度策略枚举
        AUTO,                       // 自动选择：根据批次大小动态决定
        FP32,                       // 单精度：标准计算精度，适合大多数场景
        FP16,                       // 半精度：使用GPU Tensor Core加速，适合大型批次
        MIXED                       // 混合精度：关键计算使用FP32，其余使用FP16
    };
    PrecisionStrategy precision;    // 当前精度策略
    int pipeline_depth;             // 流水线深度：GPU计算流水线的异步阶段数
    bool enable_verification;       // 校验开关：启用时对GPU结果进行CPU交叉验证

    TopologyComputeConfig()
        : batch_threshold(32)
        , precision(PrecisionStrategy::AUTO)
        , pipeline_depth(2)
        , enable_verification(false) {}
};

// ============================================================================
// 拓扑计算统计信息
// 记录加速器的运行时性能数据，用于性能分析和优化决策
// ============================================================================
struct TopologyComputeStats {
    size_t total_computations;          // 总计算次数（自上次重置以来）
    size_t cpu_path_count;              // CPU路径计算次数
    size_t gpu_path_count;              // GPU路径计算次数
    double total_time_ms;               // 总耗时（毫秒）
    double avg_cpu_time_us;             // CPU路径平均单次耗时（微秒）
    double avg_gpu_time_us;             // GPU路径平均单次耗时（微秒）
    double avg_batch_time_ms;           // 批量计算平均耗时（毫秒）
    double throughput_computations_per_sec; // 每秒吞吐量（计算次数/秒）
    size_t batch_count;                 // 批量计算次数
    size_t max_batch_size;              // 最大批次大小
    size_t min_batch_size;              // 最小批次大小（非单次）
    std::chrono::steady_clock::time_point last_reset_time; // 上次重置时间

    TopologyComputeStats()
        : total_computations(0)
        , cpu_path_count(0)
        , gpu_path_count(0)
        , total_time_ms(0.0)
        , avg_cpu_time_us(0.0)
        , avg_gpu_time_us(0.0)
        , avg_batch_time_ms(0.0)
        , throughput_computations_per_sec(0.0)
        , batch_count(0)
        , max_batch_size(0)
        , min_batch_size(0)
        , last_reset_time(std::chrono::steady_clock::now()) {}
};

// ============================================================================
// 批量拓扑计算结果
// 封装批量计算的结果和状态信息
// ============================================================================
struct BatchTopologyResult {
    std::vector<SpatialTopologyFeatures> features;  // 拓扑特征结果列表
    bool has_error;                                  // 错误标志
    std::string error_message;                       // 错误描述信息

    BatchTopologyResult()
        : has_error(false) {}
};

// ============================================================================
// 空间拓扑加速器
// 
// 主要职责：
// 1. 根据批次大小自动选择CPU多线程路径或GPU加速路径
// 2. 管理线程池以支持CPU端并行批量计算
// 3. 提供CUDA GPU加速的批量拓扑特征计算（条件编译）
// 4. 收集和报告运行时性能统计信息
//
// 设计原则：
// - 小批次（< batch_threshold）使用CPU多线程，避免GPU启动开销
// - 大批次使用GPU加速，充分利用并行计算能力
// - 支持运行时可配置的精度策略和流水线深度
// - 所有方法线程安全，支持并发调用
// ============================================================================
class SpatialTopologyAccelerator {
public:
    /**
     * @brief 构造函数
     * @param board_size 棋盘大小，默认为19路
     */
    explicit SpatialTopologyAccelerator(int board_size = 19);

    /**
     * @brief 析构函数
     * @details 自动调用shutdown()释放资源
     */
    ~SpatialTopologyAccelerator();

    // ========================================================================
    // 生命周期管理
    // ========================================================================

    /**
     * @brief 初始化加速器
     * @details 检测CUDA可用性、创建线程池（线程数 = hardware_concurrency - 1，至少1个）、
     *          预分配GPU内存（如果CUDA可用）
     */
    void initialize();

    /**
     * @brief 关闭加速器
     * @details 释放GPU内存、停止线程池、等待所有工作线程退出
     */
    void shutdown();

    // ========================================================================
    // 核心计算接口
    // ========================================================================

    /**
     * @brief 单棋盘CPU计算
     * @param board_state 棋盘状态字符串（行优先编码，'B'=黑子，'W'=白子，'.'=空）
     * @return 该棋盘的空间拓扑特征
     * @details 委托给内部的SpatialTopologyCalculator进行计算，
     *          适用于单个棋盘、不需要GPU加速的场景
     */
    SpatialTopologyFeatures calculate_single(const std::string& board_state);

    /**
     * @brief 批量计算空间拓扑特征
     * @param board_states 棋盘状态字符串列表
     * @return 对应的拓扑特征列表，与输入顺序一致
     * @details 自动路由决策：
     *          - 如果 board_states.size() < config_.batch_threshold 或 CUDA不可用，
     *            使用CPU多线程路径 calculate_batch_cpu()
     *          - 否则使用GPU加速路径 calculate_batch_gpu()
     *          - 每次调用后更新内部统计信息
     */
    std::vector<SpatialTopologyFeatures> calculate_batch(
        const std::vector<std::string>& board_states);

    // ========================================================================
    // 配置与状态查询
    // ========================================================================

    /**
     * @brief 设置计算配置
     * @param config 新的拓扑计算配置
     */
    void set_config(const TopologyComputeConfig& config);

    /**
     * @brief 获取当前计算配置
     * @return 当前配置的常量引用
     */
    const TopologyComputeConfig& get_config() const { return config_; }

    /**
     * @brief 获取运行时统计信息
     * @return 统计数据的常量引用（线程安全）
     * @details 返回自上次重置或加速器创建以来的累计统计数据
     */
    TopologyComputeStats get_stats() const;

    /**
     * @brief 查询CUDA是否可用
     * @return true=CUDA可用，false=不可用
     */
    bool is_cuda_available() const { return cuda_available_; }

    /**
     * @brief 查询是否已初始化
     * @return true=已初始化并可以执行计算，false=尚未初始化或已关闭
     */
    bool is_initialized() const { return initialized_; }

    // ========================================================================
    // 异构计算集成接口 - 支持拓扑计算在CPU/GPU间分片执行
    // ========================================================================

    /**
     * @brief 启用或禁用异构计算模式
     * @details 启用后，拓扑计算任务可根据负载在CPU和GPU间分片执行。
     *          大规模拓扑分析（>100个节点）时自动将部分子任务分配到CPU，
     *          小规模任务保持在GPU上以获得最佳延迟。
     * @param enabled true=启用异构分片, false=使用原有路径
     */
    void set_heterogeneous_enabled(bool enabled);

    /**
     * @brief 查询异构计算模式是否已启用
     * @return true=异构分片模式已启用
     */
    bool is_heterogeneous_enabled() const { return enable_heterogeneous_; }

    /**
     * @brief 将拓扑计算任务在CPU和GPU间进行分片
     * @details 分析待处理拓扑数据规模，根据设备负载动态划分CPU/GPU处理比例。
     *          如果异构引擎不可用，自动降级到原有的纯GPU/CPU路径。
     * @param topology_data 完整的拓扑数据结构
     * @param cpu_ratio 输出参数，CPU应处理的比例(0.0~1.0)
     * @return true=分片成功, false=回退到原有路径
     */
    bool partition_topology_for_heterogeneous(
        const std::shared_ptr<SpatialTopologyResult>& topology_data,
        float& cpu_ratio) const;

private:
    // ========================================================================
    // 基本成员
    // ========================================================================

    int board_size_;                           // 棋盘大小
    TopologyComputeConfig config_;             // 计算配置
    mutable TopologyComputeStats stats_;       // 运行时统计（mutable允许const方法更新）
    mutable std::mutex stats_mutex_;           // 统计数据互斥锁
    std::shared_ptr<SpatialTopologyCalculator> cpu_calculator_; // CPU计算器实例
    bool cuda_available_;                      // CUDA是否可用
    bool initialized_;                         // 是否已初始化
    bool enable_heterogeneous_;                // 是否启用异构分片

    // ========================================================================
    // 线程池成员
    // ========================================================================

    std::vector<std::thread> thread_pool_;     // 工作线程列表
    std::queue<std::function<void()>> task_queue_; // 任务队列
    std::mutex queue_mutex_;                   // 任务队列互斥锁
    std::condition_variable queue_cv_;          // 任务队列条件变量
    std::atomic<bool> pool_running_;            // 线程池运行标志

    // ========================================================================
    // 内部方法
    // ========================================================================

    /**
     * @brief 线程池工作循环
     * @details 每个工作线程的主循环：
     *          1. 等待队列中有任务或池停止信号
     *          2. 取出任务并执行
     *          3. 循环直到 pool_running_ 为 false 且队列为空
     */
    void worker_loop();

    /**
     * @brief CPU多线程批量计算路径
     * @param board_states 棋盘状态列表
     * @return 拓扑特征列表
     * @details 将输入分块后提交到线程池并行计算，使用future/promise收集结果
     */
    std::vector<SpatialTopologyFeatures> calculate_batch_cpu(
        const std::vector<std::string>& board_states);

    /**
     * @brief GPU加速批量计算路径（条件编译）
     * @param board_states 棋盘状态列表
     * @return 拓扑特征列表
     * @details 仅在定义了USE_CUDA宏时提供完整实现。
     *          当前为占位实现，将任务转发到CPU路径并输出日志提示。
     *          后续由CUDA内核任务完善此路径。
     */
    std::vector<SpatialTopologyFeatures> calculate_batch_gpu(
        const std::vector<std::string>& board_states);

    // ========================================================================
    // CUDA资源成员（条件编译）
    // 仅在编译时定义了 USE_CUDA 宏时启用
    // ========================================================================
#ifdef USE_CUDA
    float* d_board_data_;          // GPU设备端棋盘数据指针
    float* d_topology_results_;    // GPU设备端拓扑结果指针
    void* d_cube_states_;          // GPU设备端立方体状态数据
    void* d_temp_buffer_;          // GPU设备端临时缓冲区
    cudaStream_t compute_stream_;  // CUDA计算流，用于异步执行
    size_t gpu_allocated_bytes_;   // GPU已分配内存字节数

    struct AsyncPipeline;                              // 异步流水线内部类（前向声明）
    std::unique_ptr<AsyncPipeline> pipeline_;          // 异步流水线实例，用于CPU/GPU重叠执行

    /**
     * @brief 在GPU上分配工作内存
     * @param max_batch_size 预期的最大批次大小
     * @return true=分配成功，false=分配失败
     */
    bool allocate_gpu_memory(size_t max_batch_size);

    /**
     * @brief 释放GPU设备内存
     */
    void free_gpu_memory();

    /**
     * @brief 将棋盘状态数据传输到GPU
     * @param board_states 棋盘状态列表
     * @return true=传输成功，false=传输失败
     */
    bool upload_board_states_to_gpu(const std::vector<std::string>& board_states);

    /**
     * @brief 从GPU下载计算结果
     * @param batch_size 批次大小
     * @param results 输出的拓扑特征列表（将被填充）
     * @return true=下载成功，false=下载失败
     */
    bool download_results_from_gpu(size_t batch_size, std::vector<SpatialTopologyFeatures>& results);

    /**
     * @brief 启动GPU拓扑计算内核
     * @param batch_size 批次大小
     * @return true=启动成功，false=启动失败
     */
    bool launch_topology_kernel(size_t batch_size);
#endif // USE_CUDA
};

} // namespace search
} // namespace go
} // namespace spacego
