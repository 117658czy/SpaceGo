#pragma once

// ============================================================================
// 异构计算引擎 - 核心调度引擎 (HeterogeneousComputeEngine)
// 命名空间: spacego::go::heterogeneous
// 功能: 作为整个异构计算系统的核心调度引擎，统一管理CPU和GPU计算资源，
//       负责任务接收、分析、分解、调度、执行和结果合并的全流程。
//
// 架构概述：
//   HeterogeneousComputeEngine 是异构计算系统的中央调度器，它聚合了以下
//   五个子模块协同工作：
//     1. WorkloadAnalyzer    - 任务特征分析与设备亲和度计算
//     2. TaskDecomposer      - 大任务分解为CPU/GPU子任务
//     3. DeviceLoadBalancer  - 设备负载实时监控与动态分配比例调整
//     4. ResultMerger        - CPU/GPU子任务结果的合并与精度对齐
//     5. DeviceCapability    - CPU/GPU硬件能力自动检测
//
// 调度流程（以一次棋盘评估任务为例）：
//   1. 用户调用 SubmitBoardEvaluation(board_data, 10000)
//   2. 引擎创建 ComputeTask (task_id=42, category=BOARD_EVALUATION,
//      total_elements=10000)
//   3. WorkloadAnalyzer 分析任务特征 -> TaskFeature (matrix_density=0.3,
//      data_scale=10000, mem_pattern=RANDOM, parallelism=0.85)
//   4. DeviceLoadBalancer 获取当前负载 -> AllocationDecision
//      (cpu_ratio=0.30, gpu_ratio=0.70)
//   5. TaskDecomposer 估算最优分解 -> {cpu_elements=3000, gpu_elements=7000}
//   6. TaskDecomposer 按批次分解 -> 3个CPU子任务 + 7个GPU子任务
//   7. CPU子任务入队 cpu_task_queue_，GPU子任务入队 gpu_task_queue_
//   8. CPU Worker线程取子任务执行（GoBoardCPU评估）
//   9. GPU Worker线程取子任务执行（GoBoardCUDA批量评估）
//  10. Scheduler线程检测所有子任务完成，调用 ResultMerger 合并
//  11. 通过 promise/future 通知调用方结果就绪
//
// 线程模型：
//   - 1 个调度器线程 (scheduler_thread_): 周期性监控负载、收集合并结果
//   - N 个CPU工作线程 (cpu_worker_threads_): N = hardware_concurrency()
//   - M 个GPU工作线程 (gpu_worker_threads_): M = 2 (默认2个CUDA流)
//   - 调用方线程: 提交任务后通过 future 异步等待结果
//
// 线程安全保证：
//   - 任务队列: 独立的 mutex + condition_variable (CPU/GPU分开)
//   - 注册表:   shared_mutex (多读单写)
//   - 统计信息: shared_mutex 保护累积值
//   - 原子变量: initialized_, running_, stop_requested_, next_task_id_
// ============================================================================

#include "heterogeneous_types.h"
#include "workload_analyzer.h"
#include "task_decomposer.h"
#include "device_load_balancer.h"
#include "result_merger.h"
#include "config/platform_config.h"

#include <memory>
#include <vector>
#include <queue>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <functional>
#include <future>
#include <map>
#include <chrono>
#include <cstddef>

namespace spacego {
namespace go {
namespace heterogeneous {

// ============================================================================
// 前向声明 - 避免循环依赖
// ============================================================================

// （本头文件无循环依赖，所有依赖均为已有模块）

// ============================================================================
// TaskCallback - 任务完成回调类型
// ============================================================================

/**
 * @brief 单个子任务完成时的回调函数类型
 *
 * 当CPU或GPU工作线程完成一个子任务后，调用此回调通知上层。
 * 回调接收一个 SubTaskResult 引用，包含子任务的执行结果。
 *
 * 使用场景：
 *   - 进度跟踪：上层可通过回调了解已完成子任务数量
 *   - 部分结果消费：在全部子任务完成前即可开始处理中间结果
 *   - 错误监控：及时发现失败的子任务
 *
 * 线程上下文：回调在工作线程中执行，调用方需确保线程安全。
 */
using TaskCallback = std::function<void(const SubTaskResult&)>;

// ============================================================================
// ComputeTask - 统一计算任务描述结构体
// ============================================================================

/**
 * @brief 描述一个提交到异构计算引擎的统一计算任务
 *
 * ComputeTask 是所有任务提交接口的通用任务描述符。
 * 不同类别的任务（棋盘评估、推理、拓扑、搜索）都使用此结构体，
 * 通过 category 字段区分任务类型，通过 input_data/output_data 传递数据。
 *
 * 数据所有权约定：
 *   - input_data 和 output_data 指针由调用方管理生命周期
 *   - 引擎不负责分配或释放这些内存
 *   - output_data 必须在任务提交前预分配足够大小
 *   - 调用方通过 future 得知任务完成后才可安全读取 output_data
 */
struct ComputeTask {
    size_t task_id;                                    // 任务唯一ID（由引擎自动分配）
                                                       // 调用方通过此ID追踪任务状态
    TaskCategory category;                             // 任务类别（棋盘评估/推理/拓扑/搜索/通用）
                                                       // 决定分析策略和执行路径
    size_t total_elements;                             // 总元素数（任务处理的数据单元总数）
                                                       // 棋盘评估: 棋盘数
                                                       // 推理: 批次样本数
                                                       // 拓扑: 区域数
                                                       // 搜索: 仿真数
    size_t element_size_bytes;                         // 每个元素占用的字节数
                                                       // 用于计算数据传输开销和内存占用估算
    void* input_data;                                  // 输入数据指针（调用方管理生命周期）
                                                       // 多态数据：棋盘/张量/区域描述/仿真参数
    void* output_data;                                 // 输出数据指针（预分配，调用方管理）
                                                       // 引擎将计算结果写入此缓冲区
    TaskCallback callback;                             // 单个子任务完成时的回调（可选）
                                                       // 设为 nullptr 表示不需要子任务级别的进度通知
    std::chrono::system_clock::time_point submit_time; // 任务提交时间戳
                                                       // 用于计算任务总延迟（从提交到完成）
    int priority;                                      // 任务优先级 (0=最低, 10=最高)
                                                       // 影响子任务在队列中的排序

    // ========================================================================
    // 默认构造函数 - 初始化为安全默认值
    // ========================================================================
    ComputeTask()
        : task_id(0)
        , category(TaskCategory::GENERIC_COMPUTE)
        , total_elements(0)
        , element_size_bytes(0)
        , input_data(nullptr)
        , output_data(nullptr)
        , callback(nullptr)
        , submit_time(std::chrono::system_clock::now())
        , priority(5)  // 默认中等优先级
    {
    }
};

// ============================================================================
// EnginePerformanceStats - 引擎性能统计结构体
// ============================================================================

/**
 * @brief 记录异构计算引擎的运行时性能统计数据
 *
 * 提供完整的任务处理吞吐量、设备利用率和延迟分布信息。
 * 用于性能调优、容量规划和问题诊断。
 *
 * 所有数值通过 get_performance_stats() 获取当前快照。
 * 调用 reset_performance_stats() 清零所有计数器。
 */
struct EnginePerformanceStats {
    size_t total_tasks_submitted;       // 总提交任务数（自上次重置以来）
                                        // 包含已完成、执行中和队列中的任务
    size_t total_tasks_completed;       // 总完成任务数
                                        // 所有子任务成功合并后才计入
    size_t cpu_tasks_completed;         // CPU设备完成的子任务总数
                                        // 每个CPU子任务完成时递增
    size_t gpu_tasks_completed;         // GPU设备完成的子任务总数
                                        // 每个GPU子任务完成时递增
    double total_cpu_time_ms;           // CPU总执行时间（毫秒）
                                        // 所有CPU子任务执行时间的累加和
    double total_gpu_time_ms;           // GPU总执行时间（毫秒）
                                        // 所有GPU子任务执行时间的累加和
    double avg_cpu_time_ms;             // CPU平均执行时间（毫秒）
                                        // total_cpu_time_ms / cpu_tasks_completed
    double avg_gpu_time_ms;             // GPU平均执行时间（毫秒）
                                        // total_gpu_time_ms / gpu_tasks_completed
    double cpu_utilization_avg;         // CPU平均利用率 (0.0 ~ 1.0)
                                        // 由 DeviceLoadBalancer 的EMA值提供
    double gpu_utilization_avg;         // GPU平均利用率 (0.0 ~ 1.0)
                                        // 由 DeviceLoadBalancer 的EMA值提供
    size_t task_decompositions;         // 任务分解总次数
                                        // 每次 SubmitTask 调用一次分解
    double throughput_tasks_per_sec;    // 吞吐量（任务数/秒）
                                        // total_tasks_completed / 运行总时间

    // ========================================================================
    // 默认构造函数 - 初始化为零值
    // ========================================================================
    EnginePerformanceStats()
        : total_tasks_submitted(0)
        , total_tasks_completed(0)
        , cpu_tasks_completed(0)
        , gpu_tasks_completed(0)
        , total_cpu_time_ms(0.0)
        , total_gpu_time_ms(0.0)
        , avg_cpu_time_ms(0.0)
        , avg_gpu_time_ms(0.0)
        , cpu_utilization_avg(0.0)
        , gpu_utilization_avg(0.0)
        , task_decompositions(0)
        , throughput_tasks_per_sec(0.0)
    {
    }
};

// ============================================================================
// HeterogeneousComputeEngine - 核心调度引擎类
// ============================================================================

/**
 * @brief 围棋AI异构计算系统的核心调度引擎（单例模式）
 *
 * HeterogeneousComputeEngine 是整个异构计算系统的总控制器，
 * 对外提供统一的任务提交接口，对内协调五个子模块完成以下全流程：
 *   任务接收 -> 特征分析 -> 设备选择 -> 任务分解 -> 队列调度 ->
 *   并行执行 -> 结果收集 -> 精度合并 -> 结果返回
 *
 * 设计特点：
 *   - 线程池架构：CPU/GPU各自独立的工作线程池，最大化并行度
 *   - 异步非阻塞：所有任务提交立即返回 future，调用方可异步等待结果
 *   - 自适应调度：调度器线程周期性监控设备负载，动态调整分配策略
 *   - 优先级队列：高优先级任务优先获得计算资源
 *   - 容错处理：子任务失败不影响其他子任务，部分失败的任务仍可返回部分结果
 *
 * 生命周期：
 *   1. HeterogeneousComputeEngine::instance() 获取单例
 *   2. initialize(config) 启动引擎（检测设备、创建线程池）
 *   3. SubmitXxx(...) 提交任务（可在多线程中并发提交）
 *   4. shutdown() 优雅关闭（等待所有任务完成、join所有线程）
 *
 * 使用示例：
 * @code
 *   auto& engine = HeterogeneousComputeEngine::instance();
 *
 *   // 配置并初始化引擎
 *   HeterogeneousComputeConfig config;
 *   config.min_batch_for_gpu = 32;
 *   engine.initialize(config);
 *
 *   // 提交棋盘评估任务（异步）
 *   auto future = engine.SubmitBoardEvaluation(board_data, 1000);
 *
 *   // 在等待期间可提交更多任务
 *   auto future2 = engine.SubmitNeuralInference(tensor_data, 64, 362);
 *
 *   // 获取结果（阻塞直到完成）
 *   MergedBoardResult result = future.get();
 *
 *   // 获取性能统计
 *   auto stats = engine.get_performance_stats();
 *   LOG_INFO("吞吐量: %.2f 任务/秒", stats.throughput_tasks_per_sec);
 *
 *   // 关闭引擎
 *   engine.shutdown();
 * @endcode
 */
class HeterogeneousComputeEngine {
public:
    // ========================================================================
    // 单例接口
    // ========================================================================

    /**
     * @brief 获取全局唯一的引擎单例
     *
     * 使用 C++11 魔法静态变量（Magic Statics）保证线程安全。
     * 首次调用时构造实例，后续调用返回同一实例的引用。
     *
     * @return HeterogeneousComputeEngine& 引擎单例引用
     */
    static HeterogeneousComputeEngine& instance();

    // ========================================================================
    // 生命周期管理
    // ========================================================================

    /**
     * @brief 初始化引擎（必须在首次提交任务前调用）
     *
     * 执行以下初始化步骤：
     *   1. 检测CPU设备能力（核心数、内存、算力估算）
     *   2. 检测GPU设备能力（如果CUDA可用：SM数、显存、Tensor Core支持）
     *   3. 使用检测到的设备能力初始化 WorkloadAnalyzer
     *   4. 创建 CPU 工作线程池（数量 = hardware_concurrency()）
     *   5. 如果 CUDA 可用，创建 GPU 工作线程池（默认 2 个）
     *   6. 启动调度器线程
     *   7. 设置 initialized_ = true, running_ = true
     *
     * 如果已经初始化，再次调用将直接返回 true（幂等操作）。
     *
     * @param config 异构计算配置参数（使用默认构造获取推荐默认值）
     * @return bool  初始化成功返回 true，失败返回 false
     */
    bool initialize(const HeterogeneousComputeConfig& config = HeterogeneousComputeConfig());

    /**
     * @brief 关闭引擎，等待所有正在执行的任务完成
     *
     * 执行以下关闭步骤：
     *   1. 设置 stop_requested_ = true（停止接收新任务）
     *   2. 通知所有条件变量唤醒等待中的工作线程
     *   3. 等待调度器线程退出（join）
     *   4. 等待所有 CPU 工作线程退出（join）
     *   5. 等待所有 GPU 工作线程退出（join）
     *   6. 处理队列中剩余的任务（尽力完成，设置 running_ = false）
     *
     * 关闭后可以重新调用 initialize() 重启引擎。
     */
    void shutdown();

    /**
     * @brief 检查引擎是否已完成初始化
     * @return bool true 表示已初始化
     */
    bool is_initialized() const { return initialized_.load(std::memory_order_acquire); }

    /**
     * @brief 检查引擎是否正在运行（未关闭）
     * @return bool true 表示正在运行，可以提交任务
     */
    bool is_running() const { return running_.load(std::memory_order_acquire); }

    // ---- 禁止拷贝和移动（单例模式） ----
    HeterogeneousComputeEngine(const HeterogeneousComputeEngine&) = delete;
    HeterogeneousComputeEngine& operator=(const HeterogeneousComputeEngine&) = delete;

    // ========================================================================
    // 任务提交接口 - 支持多种任务类型的统一入口
    // ========================================================================

    /**
     * @brief 统一任务提交接口
     *
     * 通用任务提交方法，适用于所有任务类别。引擎内部会自动：
     *   1. 分配唯一任务ID
     *   2. 调用 WorkloadAnalyzer 分析任务特征
     *   3. 调用 DeviceLoadBalancer 获取当前最优分配比例
     *   4. 调用 TaskDecomposer 估算最优分解并执行分解
     *   5. 将 CPU 子任务和 GPU 子任务分别入队
     *   6. 返回 future 供调用方异步等待
     *
     * 返回的 future 在任务全部子任务完成并合并后变为就绪状态。
     * SubTaskResult 的 raw_data 中包含合并后的完整结果。
     *
     * @param task 计算任务描述（包含类别、数据指针、元素数等信息）
     * @return std::future<SubTaskResult> 可用于异步等待结果的 future
     */
    std::future<SubTaskResult> SubmitTask(const ComputeTask& task);

    /**
     * @brief 批量提交棋盘评估任务
     *
     * 将一批围棋棋盘同时提交评估，引擎自动决定CPU和GPU的分配。
     * 适用于 MCTS 搜索中需要对大量候选局面进行快速评估的场景。
     *
     * 内部流程与 SubmitTask 相同，但结果经过棋盘评估专用的合并处理，
     * 返回 MergedBoardResult（包含合法着法列表、气数等结构化数据）。
     *
     * @param board_data  指向棋盘数据缓冲区的指针（调用方管理生命周期）
     * @param board_count 待评估的棋盘总数
     * @return std::future<MergedBoardResult> 异步等待棋盘评估合并结果
     */
    std::future<MergedBoardResult> SubmitBoardEvaluation(
        void* board_data, size_t board_count);

    /**
     * @brief 批量提交神经网络推理任务
     *
     * 将批量的神经网络推理请求提交给引擎处理。
     *
     * 特殊策略：
     *   - 小批次（batch_size < min_batch_for_gpu，默认16）：强制走 CPU
     *   - 大批次：按负载均衡器建议的比例分配 CPU/GPU
     *   - 支持策略网络和价值网络的批量推理
     *
     * @param input_tensor 输入张量数据指针（调用方管理生命周期）
     * @param batch_size   推理批次大小（同时处理的样本数）
     * @param policy_dim   策略输出维度（围棋通常为 362 = 361 着法点 + pass）
     * @return std::future<MergedInferenceResult> 异步等待推理合并结果
     */
    std::future<MergedInferenceResult> SubmitNeuralInference(
        void* input_tensor, size_t batch_size, int policy_dim);

    /**
     * @brief 批量提交拓扑计算任务
     *
     * 将围棋空间拓扑分析任务提交给引擎处理。
     *
     * 拓扑计算默认偏向 CPU（因为分支多、依赖复杂），但如果 GPU 空闲
     * 且 CPU 繁忙，负载均衡器会自动将部分区域分配给 GPU。
     *
     * @param region_data  区域数据指针（调用方管理生命周期）
     * @param region_count 待计算的区域总数
     * @return std::future<MergedTopologyResult> 异步等待拓扑计算合并结果
     */
    std::future<MergedTopologyResult> SubmitTopologyCalculation(
        void* region_data, size_t region_count);

    /**
     * @brief 批量提交搜索仿真任务
     *
     * 将 MCTS/HMGS-DP 搜索中的蒙特卡洛仿真任务提交给引擎处理。
     * 每个仿真独立执行，天然适合并行化。
     *
     * @param simulation_data  仿真数据指针（调用方管理生命周期）
     * @param simulation_count 仿真总数
     * @return std::future<SubTaskResult> 异步等待仿真结果
     */
    std::future<SubTaskResult> SubmitSearchSimulation(
        void* simulation_data, size_t simulation_count);

    // ========================================================================
    // 设备管理接口
    // ========================================================================

    /**
     * @brief 获取当前检测到的 CPU 设备能力
     *
     * 返回初始化时通过 DeviceCapability::detect_cpu() 检测到的能力快照。
     * 包含核心数、内存、估算算力等信息。
     *
     * @return DeviceCapability CPU 设备能力描述
     */
    DeviceCapability get_cpu_capability() const;

    /**
     * @brief 获取当前检测到的 GPU 设备能力
     *
     * 返回初始化时通过 DeviceCapability::detect_gpu() 检测到的能力快照。
     * 如果 CUDA 不可用，返回零值的能力结构体。
     *
     * @return DeviceCapability GPU 设备能力描述
     */
    DeviceCapability get_gpu_capability() const;

    /**
     * @brief 重新检测并更新设备能力信息
     *
     * 在以下场景中调用：
     *   - 运行时 GPU 热插拔后
     *   - 系统内存变化后（如释放了大量内存）
     *   - 需要刷新可用内存读数
     *
     * 注意：此方法会更新内部缓存的设备能力，并同步更新
     * WorkloadAnalyzer 中的设备能力缓存。
     */
    void update_device_capabilities();

    // ========================================================================
    // 性能统计接口
    // ========================================================================

    /**
     * @brief 获取当前的引擎性能统计快照
     *
     * 返回包含任务吞吐量、设备利用率、平均延迟等指标的完整统计。
     * 此方法会即时计算 avg_* 和 throughput 等派生指标。
     *
     * 线程安全：使用 shared_lock 进行读操作。
     *
     * @return EnginePerformanceStats 当前性能统计快照
     */
    EnginePerformanceStats get_performance_stats() const;

    /**
     * @brief 重置所有性能统计数据
     *
     * 将所有计数器归零，时间累加器清零。
     * 通常在开始新一轮性能基准测试前调用。
     *
     * 注意：reset 不影响当前队列中的任务和正在执行的任务。
     */
    void reset_performance_stats();

    // ========================================================================
    // 配置管理接口
    // ========================================================================

    /**
     * @brief 运行时更新引擎配置
     *
     * 可在不重启引擎的情况下调整调度参数。
     * 更新后下一次调度循环将使用新的参数。
     *
     * @param config 新的配置参数
     */
    void update_config(const HeterogeneousComputeConfig& config);

    /**
     * @brief 获取当前使用的配置（只读）
     *
     * @return const HeterogeneousComputeConfig& 当前配置的常量引用
     */
    const HeterogeneousComputeConfig& get_config() const;

private:
    // ========================================================================
    // 构造函数与析构函数（私有，单例模式）
    // ========================================================================

    /**
     * @brief 私有构造函数
     *
     * 初始化所有成员变量为安全默认值。
     * 实际的初始化工作在 initialize() 中完成。
     */
    HeterogeneousComputeEngine();

    /**
     * @brief 私有析构函数
     *
     * 确保在销毁前调用 shutdown()。
     * 如果有线程仍在运行，强制终止并记录警告。
     */
    ~HeterogeneousComputeEngine();

    // ========================================================================
    // 子模块实例
    // ========================================================================

    WorkloadAnalyzer workload_analyzer_;    // 工作负载分析器：任务特征提取、设备亲和度计算
    TaskDecomposer task_decomposer_;        // 任务分解器：大任务拆分为CPU/GPU子任务
    DeviceLoadBalancer load_balancer_;      // 设备负载均衡器：实时监控负载、动态调整分配比例
    ResultMerger result_merger_;            // 结果合并器：CPU/GPU子任务结果合并与精度对齐

    // ========================================================================
    // 配置与设备能力
    // ========================================================================

    HeterogeneousComputeConfig config_;     // 引擎配置参数（含所有阈值和开关）
    DeviceCapability cpu_capability_;       // CPU设备能力快照（初始化时检测，可刷新）
    DeviceCapability gpu_capability_;       // GPU设备能力快照（初始化时检测，可刷新）

    // ========================================================================
    // 线程管理 - 控制标志
    // ========================================================================

    std::atomic<bool> initialized_;         // 引擎是否已完成初始化
                                            // 使用 memory_order_acquire/release 保证可见性
    std::atomic<bool> running_;             // 引擎是否正在运行
                                            // false 表示已关闭或尚未初始化
    std::atomic<bool> stop_requested_;      // 是否请求停止
                                            // 工作线程检测到此标志为 true 时退出循环

    // ========================================================================
    // 线程管理 - 工作线程
    // ========================================================================

    std::vector<std::thread> cpu_worker_threads_;  // CPU工作线程池
                                                   // 每个线程独立从 cpu_task_queue_ 取任务执行
    std::vector<std::thread> gpu_worker_threads_;  // GPU工作线程池
                                                   // 每个线程对应一个CUDA流（如果CUDA可用）
    std::thread scheduler_thread_;                 // 调度器线程
                                                   // 周期性监控负载、收集合并已完成任务结果

    // ========================================================================
    // 任务队列 - CPU/GPU 各自独立
    // ========================================================================

    std::queue<SubTask> cpu_task_queue_;           // CPU子任务等待队列
                                                   // FIFO顺序，支持优先级排序（通过插入位置实现）
    std::queue<SubTask> gpu_task_queue_;           // GPU子任务等待队列
                                                   // FIFO顺序，支持优先级排序

    std::mutex cpu_queue_mutex_;                   // CPU队列互斥锁
    std::mutex gpu_queue_mutex_;                   // GPU队列互斥锁

    std::condition_variable cpu_queue_cv_;         // CPU队列条件变量
                                                   // 工作线程和调度器通过此CV等待/通知
    std::condition_variable gpu_queue_cv_;         // GPU队列条件变量

    // ========================================================================
    // 任务注册表 - 追踪任务全生命周期
    // ========================================================================

    /**
     * @brief 任务ID到 ComputeTask 的映射
     *
     * 存储所有已提交但尚未完成的任务描述。
     * 工作线程执行子任务时需要查阅此表获取任务元数据。
     * 任务完成（所有子任务合并后）从此表中移除。
     */
    std::map<size_t, ComputeTask> task_registry_;

    /**
     * @brief 任务ID到 DecompositionResult 的映射
     *
     * 存储每个任务的分解结果，包含所有子任务列表和CPU/GPU分配比例。
     * 调度器通过此表追踪哪些子任务属于同一个父任务。
     * 用于 collect_and_merge_results 判断任务是否全部子任务完成。
     */
    std::map<size_t, DecompositionResult> decomposition_registry_;

    /**
     * @brief 任务ID到已完成子任务结果列表的映射
     *
     * 每个任务对应一个 SubTaskResult 列表，存储该任务已完成子任务的结果。
     * 当列表大小等于该任务的子任务总数时，触发合并流程。
     */
    std::map<size_t, std::vector<SubTaskResult>> result_registry_;

    /**
     * @brief 任务ID到 promise<SubTaskResult> 的映射（通用任务）
     *
     * SubmitTask 调用时创建，任务完成时通过 promise 通知调用方。
     * 存储 shared_ptr 因为 promise 是 move-only 类型。
     */
    std::map<size_t, std::shared_ptr<std::promise<SubTaskResult>>> promise_registry_;

    /**
     * @brief 任务ID到 promise<MergedBoardResult> 的映射（棋盘评估任务）
     */
    std::map<size_t, std::shared_ptr<std::promise<MergedBoardResult>>> board_promise_registry_;

    /**
     * @brief 任务ID到 promise<MergedInferenceResult> 的映射（推理任务）
     */
    std::map<size_t, std::shared_ptr<std::promise<MergedInferenceResult>>> inference_promise_registry_;

    /**
     * @brief 任务ID到 promise<MergedTopologyResult> 的映射（拓扑计算任务）
     */
    std::map<size_t, std::shared_ptr<std::promise<MergedTopologyResult>>> topology_promise_registry_;

    /**
     * @brief 注册表读写锁
     *
     * 使用 shared_mutex 支持多读单写：
     *   - 读操作（查询任务状态）：shared_lock
     *   - 写操作（注册新任务、更新结果、清理完成任务）：unique_lock
     */
    mutable std::shared_mutex registry_mutex_;

    // ========================================================================
    // 性能统计
    // ========================================================================

    mutable EnginePerformanceStats perf_stats_;    // 性能统计累加器
                                                   // mutable 允许在 const getter 中更新派生指标

    mutable std::shared_mutex stats_mutex_;        // 统计信息读写锁

    std::chrono::system_clock::time_point engine_start_time_;  // 引擎启动时间
                                                                // 用于计算吞吐量

    // ========================================================================
    // 任务ID生成器
    // ========================================================================

    std::atomic<size_t> next_task_id_;             // 下一个任务ID
                                                   // 原子递增保证线程安全

    // ========================================================================
    // 内部方法 - 调度器
    // ========================================================================

    /**
     * @brief 调度器线程的主循环
     *
     * 周期性执行以下操作：
     *   1. 更新 CPU/GPU 队列深度信息到 DeviceLoadBalancer
     *   2. 调用 DeviceLoadBalancer::calculate_allocation_ratio() 获取最新分配比例
     *   3. 检测设备过载情况，触发分流/回流决策
     *   4. 扫描 result_registry_，收集已全部完成的子任务
     *   5. 对已完成的任务调用 collect_and_merge_results()
     *   6. 清理已完成任务从各注册表中移除
     *   7. 休眠 config_.monitoring_interval_ms 毫秒
     *
     * 退出条件：stop_requested_ == true
     */
    void scheduler_loop();

    // ========================================================================
    // 内部方法 - 工作线程
    // ========================================================================

    /**
     * @brief CPU 工作线程的主循环
     *
     * 循环执行：
     *   1. 从 cpu_task_queue_ 取出一个子任务（阻塞等待直到有任务或停止请求）
     *   2. 将子任务状态设为 RUNNING，记录开始时间
     *   3. 调用 execute_cpu_subtask() 执行实际计算
     *   4. 将结果存入 result_registry_[task_id] 中
     *   5. 如果任务有 callback，调用回调通知
     *   6. 更新性能统计
     *
     * 退出条件：stop_requested_ == true 且队列为空
     *
     * @param worker_id CPU工作线程编号（从0开始，用于日志标识）
     */
    void cpu_worker_loop(int worker_id);

    /**
     * @brief GPU 工作线程的主循环
     *
     * 循环执行：
     *   1. 从 gpu_task_queue_ 取出一个子任务（阻塞等待直到有任务或停止请求）
     *   2. 将子任务状态设为 RUNNING，记录开始时间
     *   3. 调用 execute_gpu_subtask() 执行CUDA加速计算
     *   4. 将结果存入 result_registry_[task_id] 中
     *   5. 如果任务有 callback，调用回调通知
     *   6. 更新性能统计
     *
     * GPU工作线程使用独立的CUDA流实现异步执行，
     * 多个GPU子任务可以在不同的流中并发执行。
     *
     * 退出条件：stop_requested_ == true 且队列为空
     *
     * @param worker_id GPU工作线程编号（从0开始，用于CUDA流选择）
     */
    void gpu_worker_loop(int worker_id);

    // ========================================================================
    // 内部方法 - 子任务执行
    // ========================================================================

    /**
     * @brief 在CPU上执行一个子任务
     *
     * 根据父任务的 category 选择对应的CPU实现：
     *   - BOARD_EVALUATION:  调用围棋棋盘CPU评估逻辑
     *   - NEURAL_INFERENCE:   调用神经网络CPU推理（LibTorch CPU后端或自行实现）
     *   - TOPOLOGY_CALCULATION: 调用空间拓扑CPU计算
     *   - SEARCH_SIMULATION:  调用MCTS仿真CPU执行
     *   - GENERIC_COMPUTE:    通用CPU计算路径
     *
     * 执行后将结果序列化到 SubTaskResult.raw_data 中。
     *
     * @param subtask 待执行的子任务描述
     * @param task    父任务描述（包含数据指针和类别信息）
     * @return SubTaskResult 包含序列化结果的子任务结果
     */
    SubTaskResult execute_cpu_subtask(const SubTask& subtask, const ComputeTask& task);

    /**
     * @brief 在GPU上执行一个子任务
     *
     * 根据父任务的 category 选择对应的CUDA加速实现：
     *   - BOARD_EVALUATION:  GoBoardCUDA 批量棋盘评估
     *   - NEURAL_INFERENCE:   NeuralNetworkCUDA 推理（LibTorch CUDA后端）
     *   - TOPOLOGY_CALCULATION: SpatialComputationAccelerator CUDA路径
     *   - SEARCH_SIMULATION:  批量MCTS仿真CUDA加速
     *   - GENERIC_COMPUTE:    通用GPU计算路径
     *
     * 使用CUDA流实现异步执行，多流并发提高GPU利用率。
     *
     * 如果CUDA不可用，此方法会标记结果为失败（success=false），
     * 并设置 error_message 说明GPU不可用。
     *
     * @param subtask 待执行的子任务描述
     * @param task    父任务描述
     * @return SubTaskResult 包含序列化结果的子任务结果
     */
    SubTaskResult execute_gpu_subtask(const SubTask& subtask, const ComputeTask& task);

    // ========================================================================
    // 内部方法 - 结果收集与合并
    // ========================================================================

    /**
     * @brief 收集某个任务的所有子任务结果并执行合并
     *
     * 处理流程：
     *   1. 从 decomposition_registry_ 获取该任务的分解结果
     *   2. 检查 result_registry_[task_id] 中是否所有子任务已完成
     *   3. 如果全部完成：
     *      a. 从 result_registry_[task_id] 提取所有 SubTaskResult
     *      b. 按 source_device 分为 CPU 结果列表和 GPU 结果列表
     *      c. 根据任务 category 选择对应的 ResultMerger 方法：
     *         - BOARD_EVALUATION -> merge_board_results
     *         - NEURAL_INFERENCE -> merge_inference_results
     *         - TOPOLOGY_CALCULATION -> merge_topology_results
     *         - SEARCH_SIMULATION / GENERIC_COMPUTE -> 直接合并 raw_data
     *      d. 通过 promise 通知调用方（设置 promise 值）
     *      e. 从各注册表中清理该任务
     *   4. 如果尚未全部完成，直接返回（等待下次调度循环检查）
     *
     * 线程安全：调用方需持有 registry_mutex_ 的 unique_lock。
     *
     * @param task_id 待合并的任务ID
     */
    void collect_and_merge_results(size_t task_id);

    // ========================================================================
    // 内部方法 - 性能统计更新
    // ========================================================================

    /**
     * @brief 更新性能统计信息
     *
     * 在某设备上完成一个子任务后调用此方法更新：
     *   - 设备完成计数（cpu_tasks_completed 或 gpu_tasks_completed）
     *   - 设备累计执行时间（total_cpu_time_ms 或 total_gpu_time_ms）
     *
     * 线程安全：使用 unique_lock 保护写操作。
     *
     * @param device   执行子任务的设备类型（CPU 或 GPU）
     * @param time_ms  子任务执行耗时（毫秒）
     */
    void update_performance_stats(ComputeDeviceType device, double time_ms);

    // ========================================================================
    // 内部方法 - 任务提交流水线
    // ========================================================================

    /**
     * @brief 任务提交流水线（内部方法）
     *
     * 对已注册的 ComputeTask 执行完整的分析和分解流程：
     *   1. 根据 category 映射到 WorkloadAnalyzer 的分析参数
     *   2. 调用 workload_analyzer_.analyze_task() 获取 TaskFeature
     *   3. 调用 workload_analyzer_.calculate_device_affinity() 获取 DeviceAffinity
     *   4. 调用 load_balancer_.calculate_allocation_ratio() 获取 AllocationDecision
     *   5. 调用 task_decomposer_.estimate_optimal_split() 获取最优分配
     *   6. 调用 task_decomposer_.decompose_by_batch() 执行分解
     *   7. 存储 DecompositionResult 到 decomposition_registry_
     *   8. 将 CPU 子任务入队 cpu_task_queue_
     *   9. 将 GPU 子任务入队 gpu_task_queue_
     *  10. 通知工作线程（cv notify）
     *
     * @param task_id 已在 task_registry_ 中注册的任务ID
     */
    void process_task_pipeline(size_t task_id);

    /**
     * @brief 将 ComputeTask 的 category 映射为 WorkloadAnalyzer 所需的参数
     *
     * 不同类别的任务有不同的分析参数：
     *   - BOARD_EVALUATION:  element_count=361, is_tensor_op=false
     *   - NEURAL_INFERENCE:   is_tensor_op=true, tensor_dims依赖上下文
     *   - TOPOLOGY_CALCULATION: element_count=total_elements, is_tensor_op=false
     *   - SEARCH_SIMULATION:  element_count=total_elements, is_tensor_op=false
     *   - GENERIC_COMPUTE:    element_count=total_elements, is_tensor_op=false
     *
     * @param task   计算任务描述
     * @param batch_size      [输出] 批次大小
     * @param element_count   [输出] 每实例元素数
     * @param is_tensor_op    [输出] 是否为张量运算
     * @param tensor_dims     [输出] 张量维度数
     */
    void map_task_to_analyzer_params(const ComputeTask& task,
                                     size_t& batch_size,
                                     size_t& element_count,
                                     bool& is_tensor_op,
                                     int& tensor_dims);
};

}  // namespace heterogeneous
}  // namespace go
}  // namespace spacego