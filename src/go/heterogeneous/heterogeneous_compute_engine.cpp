// ============================================================================
// 异构计算引擎 - 核心调度引擎 实现文件
// 命名空间: spacego::go::heterogeneous
// 功能: HeterogeneousComputeEngine 类的完整实现
//
// 包含以下核心模块的实现：
//   1. 单例模式生命周期管理 (instance/constructor/destructor)
//   2. 引擎初始化与关闭 (initialize/shutdown)
//   3. 任务提交流水线 (process_task_pipeline + SubmitXxx 系列)
//   4. 调度器主循环 (scheduler_loop)
//   5. CPU/GPU 工作线程 (cpu_worker_loop/gpu_worker_loop)
//   6. 子任务执行器 (execute_cpu_subtask/execute_gpu_subtask)
//   7. 结果收集与合并 (collect_and_merge_results)
//   8. 性能统计 (update_performance_stats/get_performance_stats)
//   9. 设备能力检测与配置管理
// ============================================================================

#include "go/heterogeneous/heterogeneous_compute_engine.h"
#include "framework/logger.h"

#include <algorithm>      // std::max, std::min
#include <cmath>          // std::isnan, std::isinf
#include <cstring>        // std::memcpy
#include <sstream>        // std::ostringstream
#include <stdexcept>      // std::runtime_error

// ---- CUDA 环境专用头文件 ----
#if SPACEGO_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace spacego {
namespace go {
namespace heterogeneous {

// ============================================================================
// 匿名命名空间 - 内部辅助常量
// ============================================================================

namespace {

// ---- 线程配置常量 ----
constexpr int DEFAULT_GPU_WORKER_COUNT = 2;    // 默认GPU工作线程数（对应2个CUDA流）
                                               // 使用2个流可以在一个流计算时另一个流传输数据

constexpr int SHUTDOWN_POLL_MS = 50;           // shutdown 时轮询工作线程状态的间隔（毫秒）

// ---- 分析参数常量 ----
constexpr size_t GO_BOARD_POINTS = 361;        // 围棋标准19x19棋盘交叉点数
                                               // 用于棋盘评估时的 element_count 参数
constexpr int DEFAULT_TENSOR_DIMS = 2;         // 默认张量维度数（推理时矩阵乘法）

// ---- 子任务ID编码常量 ----
constexpr size_t SUBTASK_ID_MULTIPLIER = 1000000;  // 全局子任务ID编码乘数
                                                    // 编码方案: 全局ID = 父任务ID * MULTIPLIER + 局部ID
                                                    // 逆向解析: 父任务ID = 全局ID / MULTIPLIER

// ---- 序列化缓冲区默认大小 ----
constexpr size_t SERIALIZATION_BUFFER_INITIAL_SIZE = 4096;  // 序列化缓冲区初始大小（字节）

/**
 * @brief 安全地将子任务结果序列化数据追加到目标向量
 *
 * 用于合并阶段将各子任务的 raw_data 连接起来。
 *
 * @param dest 目标字节向量
 * @param src  源字节向量
 */
void append_raw_data(std::vector<uint8_t>& dest, const std::vector<uint8_t>& src) {
    dest.insert(dest.end(), src.begin(), src.end());
}

}  // 匿名命名空间

// ============================================================================
// 构造函数与析构函数
// ============================================================================

/**
 * @brief 私有构造函数
 *
 * 将所有成员初始化为安全默认值。
 * 实际的设备检测和线程创建在 initialize() 中完成。
 *
 * 初始化顺序说明：
 *   - 子模块（analyzer, decomposer, balancer, merger）使用各自的默认构造
 *   - 设备能力缓存先初始化为运行时检测值（CPU通过 detect_cpu，GPU为空设备）
 *   - 原子变量使用最严格的内存序初始化为 false/0
 *   - 引擎启动时间记录为当前时间（在 initialize 中更新）
 */
HeterogeneousComputeEngine::HeterogeneousComputeEngine()
    : workload_analyzer_()                          // 使用默认构造（自动检测CPU能力）
    , task_decomposer_()                            // 使用默认构造（max_subtask_count=64）
    , load_balancer_()                              // 使用默认构造（默认配置）
    , result_merger_()                              // 使用默认构造
    , config_()                                     // 使用默认配置参数
    , cpu_capability_(DeviceCapability::detect_cpu())  // 自动检测CPU硬件能力
    , gpu_capability_()                             // GPU能力待 initialize 中检测
    , initialized_(false)
    , running_(false)
    , stop_requested_(false)
    , cpu_worker_threads_()
    , gpu_worker_threads_()
    , scheduler_thread_()
    , cpu_task_queue_()
    , gpu_task_queue_()
    , cpu_queue_mutex_()
    , gpu_queue_mutex_()
    , cpu_queue_cv_()
    , gpu_queue_cv_()
    , task_registry_()
    , decomposition_registry_()
    , result_registry_()
    , promise_registry_()
    , board_promise_registry_()
    , inference_promise_registry_()
    , topology_promise_registry_()
    , registry_mutex_()
    , perf_stats_()
    , stats_mutex_()
    , engine_start_time_(std::chrono::system_clock::now())
    , next_task_id_(1)  // 从1开始，0保留表示无效ID
{
    LOG_INFO("HeterogeneousComputeEngine 构造完成（尚未初始化）");
}

/**
 * @brief 私有析构函数
 *
 * 如果引擎仍在运行，先调用 shutdown() 确保所有线程安全退出。
 * 如果 shutdown() 抛出异常，记录错误但不重新抛出（析构函数不应抛出异常）。
 */
HeterogeneousComputeEngine::~HeterogeneousComputeEngine()
{
    try {
        if (running_.load(std::memory_order_acquire)) {
            LOG_INFO("HeterogeneousComputeEngine 析构中，正在调用 shutdown()...");
            shutdown();
        }
    } catch (const std::exception& e) {
        // 析构函数中不应抛出异常，记录错误后静默处理
        LOG_ERROR("HeterogeneousComputeEngine 析构期间异常: %s", e.what());
    } catch (...) {
        LOG_ERROR("HeterogeneousComputeEngine 析构期间发生未知异常");
    }
    LOG_INFO("HeterogeneousComputeEngine 已销毁");
}

// ============================================================================
// 单例模式 - instance()
// ============================================================================

/**
 * @brief 获取全局唯一的引擎单例
 *
 * 使用 C++11 保证的线程安全静态局部变量初始化（Magic Statics）。
 * 首次调用时构造实例，后续调用直接返回同一实例引用。
 *
 * 注意：获取单例后仍需调用 initialize() 来启动引擎。
 *
 * @return HeterogeneousComputeEngine& 全局唯一实例的引用
 */
HeterogeneousComputeEngine& HeterogeneousComputeEngine::instance()
{
    // C++11 标准保证静态局部变量初始化的线程安全性
    // 编译器会自动插入双重检查锁定（double-checked locking）或等效机制
    static HeterogeneousComputeEngine engine;
    return engine;
}

// ============================================================================
// 引擎初始化 - initialize()
// ============================================================================

/**
 * @brief 初始化异构计算引擎
 *
 * 执行以下初始化步骤：
 *   1. 幂等性检查：如果已初始化且正在运行，直接返回 true
 *   2. 保存配置参数
 *   3. 检测 CPU 设备能力（如果尚未检测）
 *   4. 检测 GPU 设备能力（如果 CUDA 环境可用）
 *   5. 更新 WorkloadAnalyzer 的设备能力缓存
 *   6. 更新 TaskDecomposer 的最大子任务数
 *   7. 更新 DeviceLoadBalancer 的配置
 *   8. 创建 CPU 工作线程池
 *   9. 创建 GPU 工作线程池（如果 GPU 可用）
 *  10. 启动调度器线程
 *  11. 设置状态标志
 *  12. 记录引擎启动时间
 *
 * @param config 异构计算配置参数
 * @return bool 初始化成功返回 true
 */
bool HeterogeneousComputeEngine::initialize(const HeterogeneousComputeConfig& config)
{
    // ---- 步骤 1: 幂等性检查 ----
    if (initialized_.load(std::memory_order_acquire) &&
        running_.load(std::memory_order_acquire)) {
        LOG_INFO("HeterogeneousComputeEngine 已初始化并运行中，跳过重复初始化");
        return true;
    }

    LOG_INFO("===== HeterogeneousComputeEngine 初始化开始 =====");

    // ---- 步骤 2: 保存配置 ----
    config_ = config;
    LOG_INFO("配置参数: 异构计算=%s, GPU过载阈值=%.2f, CPU空闲阈值=%.2f, GPU最小批次=%zu",
             config_.enable_heterogeneous ? "启用" : "禁用",
             config_.gpu_overload_threshold,
             config_.cpu_idle_threshold,
             config_.min_batch_for_gpu);

    // ---- 步骤 3: 检测 CPU 设备能力 ----
    cpu_capability_ = DeviceCapability::detect_cpu();
    LOG_INFO("CPU检测完成: %s", cpu_capability_.device_name.c_str());
    LOG_INFO("  - 核心数: %d, 内存: %zu MB, 可用: %zu MB",
             cpu_capability_.compute_cores,
             cpu_capability_.total_memory_mb,
             cpu_capability_.available_memory_mb);
    LOG_INFO("  - 估算算力: %.4f TFLOPS, 带宽: %zu GB/s",
             cpu_capability_.compute_power_tflops,
             cpu_capability_.memory_bandwidth_gb_per_sec);

    // ---- 步骤 4: 检测 GPU 设备能力 ----
#if SPACEGO_CUDA_ENABLED
    if (config_.enable_heterogeneous) {
        gpu_capability_ = DeviceCapability::detect_gpu(0);
        LOG_INFO("GPU检测完成: %s", gpu_capability_.device_name.c_str());
        LOG_INFO("  - SM数: %d, 显存: %zu MB, 可用: %zu MB",
                 gpu_capability_.compute_cores,
                 gpu_capability_.total_memory_mb,
                 gpu_capability_.available_memory_mb);
        LOG_INFO("  - 估算算力: %.4f TFLOPS, 带宽: %zu GB/s",
                 gpu_capability_.compute_power_tflops,
                 gpu_capability_.memory_bandwidth_gb_per_sec);
        LOG_INFO("  - FP16: %s, Tensor Core: %s, 最大线程/块: %d",
                 gpu_capability_.supports_fp16 ? "支持" : "不支持",
                 gpu_capability_.supports_tensor_core ? "支持" : "不支持",
                 gpu_capability_.max_threads_per_block);
    } else {
        // 异构计算被禁用，GPU能力保持默认空设备
        LOG_INFO("异构计算已禁用，跳过GPU检测");
    }
#else
    // CUDA 不可用时的处理
    LOG_INFO("CUDA环境不可用，GPU能力使用默认空设备");
    gpu_capability_ = DeviceCapability();  // 空设备（compute_cores=0）
#endif

    // ---- 步骤 5: 更新子模块 ----
    workload_analyzer_.update_device_capabilities(cpu_capability_, gpu_capability_);

    task_decomposer_.set_max_subtask_count(config_.max_subtask_count);
    LOG_INFO("TaskDecomposer 最大子任务数: %zu", config_.max_subtask_count);

    load_balancer_.update_config(config_);

    // ---- 步骤 6: 重置状态标志 ----
    stop_requested_.store(false, std::memory_order_release);

    // ---- 步骤 7: 创建 CPU 工作线程池 ----
    unsigned int cpu_thread_count = std::thread::hardware_concurrency();
    if (cpu_thread_count == 0) {
        // 检测失败时的保守回退值
        cpu_thread_count = 4;
    }
    LOG_INFO("创建 %u 个CPU工作线程", cpu_thread_count);

    cpu_worker_threads_.clear();
    cpu_worker_threads_.reserve(cpu_thread_count);
    for (unsigned int i = 0; i < cpu_thread_count; ++i) {
        cpu_worker_threads_.emplace_back(
            &HeterogeneousComputeEngine::cpu_worker_loop,
            this,
            static_cast<int>(i)
        );
    }

    // ---- 步骤 8: 创建 GPU 工作线程池 ----
    int gpu_thread_count = 0;
#if SPACEGO_CUDA_ENABLED
    if (config_.enable_heterogeneous &&
        gpu_capability_.compute_cores > 0) {
        // GPU可用时创建DEFAULT_GPU_WORKER_COUNT个工作线程
        // 每个线程对应一个独立的CUDA流，实现多流并发
        gpu_thread_count = DEFAULT_GPU_WORKER_COUNT;
        LOG_INFO("创建 %d 个GPU工作线程（对应 %d 个CUDA流）",
                 gpu_thread_count, gpu_thread_count);
    } else {
        LOG_INFO("GPU不可用或异构计算被禁用，不创建GPU工作线程");
    }
#else
    LOG_INFO("CUDA未启用，不创建GPU工作线程");
#endif

    gpu_worker_threads_.clear();
    gpu_worker_threads_.reserve(gpu_thread_count);
    for (int i = 0; i < gpu_thread_count; ++i) {
        gpu_worker_threads_.emplace_back(
            &HeterogeneousComputeEngine::gpu_worker_loop,
            this,
            i
        );
    }

    // ---- 步骤 9: 启动调度器线程 ----
    LOG_INFO("启动调度器线程（监控间隔: %zu ms）", config_.monitoring_interval_ms);
    scheduler_thread_ = std::thread(
        &HeterogeneousComputeEngine::scheduler_loop,
        this
    );

    // ---- 步骤 10: 设置状态标志 ----
    running_.store(true, std::memory_order_release);
    initialized_.store(true, std::memory_order_release);
    engine_start_time_ = std::chrono::system_clock::now();

    LOG_INFO("===== HeterogeneousComputeEngine 初始化完成 =====");
    LOG_INFO("  CPU工作线程: %u 个, GPU工作线程: %d 个, 调度器线程: 1 个",
             cpu_thread_count, gpu_thread_count);

    return true;
}

// ============================================================================
// 引擎关闭 - shutdown()
// ============================================================================

/**
 * @brief 优雅关闭异构计算引擎
 *
 * 关闭流程：
 *   1. 幂等性检查：如果未运行，直接返回
 *   2. 设置 stop_requested_ = true，停止接收新任务
 *   3. 通知所有条件变量唤醒等待中的工作线程
 *   4. 加入调度器线程（join）
 *   5. 加入所有 CPU 工作线程（join）
 *   6. 加入所有 GPU 工作线程（join）
 *   7. 处理队列中剩余的任务（尽力完成）
 *   8. 清理注册表
 *   9. 设置 running_ = false
 */
void HeterogeneousComputeEngine::shutdown()
{
    // ---- 步骤 1: 幂等性检查 ----
    if (!running_.load(std::memory_order_acquire)) {
        LOG_INFO("HeterogeneousComputeEngine 未运行，跳过 shutdown");
        return;
    }

    LOG_INFO("===== HeterogeneousComputeEngine 关闭开始 =====");

    // ---- 步骤 2: 设置停止标志 ----
    stop_requested_.store(true, std::memory_order_release);

    // ---- 步骤 3: 唤醒所有等待中的工作线程 ----
    // 使用 notify_all 确保所有阻塞在条件变量上的线程被唤醒
    {
        std::lock_guard<std::mutex> cpu_lock(cpu_queue_mutex_);
        std::lock_guard<std::mutex> gpu_lock(gpu_queue_mutex_);
    }
    cpu_queue_cv_.notify_all();
    gpu_queue_cv_.notify_all();

    LOG_INFO("已发送停止信号，等待所有线程退出...");

    // ---- 步骤 4: 加入调度器线程 ----
    if (scheduler_thread_.joinable()) {
        scheduler_thread_.join();
        LOG_INFO("调度器线程已退出");
    }

    // ---- 步骤 5: 加入所有 CPU 工作线程 ----
    for (size_t i = 0; i < cpu_worker_threads_.size(); ++i) {
        if (cpu_worker_threads_[i].joinable()) {
            cpu_worker_threads_[i].join();
            LOG_INFO("CPU工作线程 #%zu 已退出", i);
        }
    }
    cpu_worker_threads_.clear();

    // ---- 步骤 6: 加入所有 GPU 工作线程 ----
    for (size_t i = 0; i < gpu_worker_threads_.size(); ++i) {
        if (gpu_worker_threads_[i].joinable()) {
            gpu_worker_threads_[i].join();
            LOG_INFO("GPU工作线程 #%zu 已退出", i);
        }
    }
    gpu_worker_threads_.clear();

    // ---- 步骤 7: 处理剩余队列中的任务 ----
    // 尽力完成队列中剩余的子任务（在处理线程已退出的情况下同步执行）
    size_t cpu_remaining = 0;
    size_t gpu_remaining = 0;
    {
        std::lock_guard<std::mutex> cpu_lock(cpu_queue_mutex_);
        std::lock_guard<std::mutex> gpu_lock(gpu_queue_mutex_);
        cpu_remaining = cpu_task_queue_.size();
        gpu_remaining = gpu_task_queue_.size();
        // 清空队列（剩余任务无法完成，设置对应 promise 为异常）
        while (!cpu_task_queue_.empty()) cpu_task_queue_.pop();
        while (!gpu_task_queue_.empty()) gpu_task_queue_.pop();
    }

    if (cpu_remaining > 0 || gpu_remaining > 0) {
        LOG_WARNING("关闭时丢弃了 %zu 个CPU子任务和 %zu 个GPU子任务（队列中等待的任务）",
                    cpu_remaining, gpu_remaining);
    }

    // ---- 步骤 8: 清理注册表中未完成的任务 ----
    {
        std::unique_lock<std::shared_mutex> lock(registry_mutex_);

        // 通知所有未完成的 promise 为异常状态
        for (auto& pair : promise_registry_) {
            try {
                pair.second->set_exception(
                    std::make_exception_ptr(
                        std::runtime_error("引擎关闭，任务 #" +
                            std::to_string(pair.first) + " 未完成")
                    )
                );
            } catch (...) {
                // promise 可能已经被设置过，忽略异常
            }
        }

        // 同样通知类型化的 promise
        for (auto& pair : board_promise_registry_) {
            try {
                pair.second->set_exception(
                    std::make_exception_ptr(
                        std::runtime_error("引擎关闭，棋盘评估任务 #" +
                            std::to_string(pair.first) + " 未完成")
                    )
                );
            } catch (...) {}
        }
        for (auto& pair : inference_promise_registry_) {
            try {
                pair.second->set_exception(
                    std::make_exception_ptr(
                        std::runtime_error("引擎关闭，推理任务 #" +
                            std::to_string(pair.first) + " 未完成")
                    )
                );
            } catch (...) {}
        }
        for (auto& pair : topology_promise_registry_) {
            try {
                pair.second->set_exception(
                    std::make_exception_ptr(
                        std::runtime_error("引擎关闭，拓扑计算任务 #" +
                            std::to_string(pair.first) + " 未完成")
                    )
                );
            } catch (...) {}
        }

        // 清空所有注册表
        task_registry_.clear();
        decomposition_registry_.clear();
        result_registry_.clear();
        promise_registry_.clear();
        board_promise_registry_.clear();
        inference_promise_registry_.clear();
        topology_promise_registry_.clear();
    }

    // ---- 步骤 9: 设置运行状态为 false ----
    running_.store(false, std::memory_order_release);

    LOG_INFO("===== HeterogeneousComputeEngine 关闭完成 =====");
}

// ============================================================================
// 设备能力接口
// ============================================================================

DeviceCapability HeterogeneousComputeEngine::get_cpu_capability() const
{
    // 直接返回缓存的能力快照（值语义，线程安全）
    return cpu_capability_;
}

DeviceCapability HeterogeneousComputeEngine::get_gpu_capability() const
{
    // 直接返回缓存的能力快照（值语义，线程安全）
    return gpu_capability_;
}

void HeterogeneousComputeEngine::update_device_capabilities()
{
    LOG_INFO("正在刷新设备能力检测...");

    // 重新检测 CPU 和 GPU 能力
    cpu_capability_ = DeviceCapability::detect_cpu();

#if SPACEGO_CUDA_ENABLED
    if (config_.enable_heterogeneous) {
        gpu_capability_ = DeviceCapability::detect_gpu(0);
    }
#endif

    // 同步更新 WorkloadAnalyzer 中的设备能力缓存
    workload_analyzer_.update_device_capabilities(cpu_capability_, gpu_capability_);

    LOG_INFO("设备能力已刷新: CPU(%s), GPU(%s)",
             cpu_capability_.device_name.c_str(),
             gpu_capability_.device_name.c_str());
}

// ============================================================================
// 配置管理接口
// ============================================================================

void HeterogeneousComputeEngine::update_config(const HeterogeneousComputeConfig& config)
{
    config_ = config;

    // 同步更新子模块的配置
    task_decomposer_.set_max_subtask_count(config_.max_subtask_count);
    load_balancer_.update_config(config_);

    LOG_INFO("引擎配置已更新: GPU最小批次=%zu, 监控间隔=%zu ms",
             config_.min_batch_for_gpu,
             config_.monitoring_interval_ms);
}

const HeterogeneousComputeConfig& HeterogeneousComputeEngine::get_config() const
{
    return config_;
}

// ============================================================================
// 任务提交接口 - SubmitTask (统一入口)
// ============================================================================

/**
 * @brief 通用任务提交接口
 *
 * 处理流程：
 *   1. 检查引擎运行状态
 *   2. 分配唯一任务ID
 *   3. 创建 promise/future 对
 *   4. 注册任务到 task_registry_ 和 promise_registry_
 *   5. 调用 process_task_pipeline 执行分析和分解
 *   6. 返回 future
 *
 * @param task 计算任务描述
 * @return std::future<SubTaskResult> 异步结果
 */
std::future<SubTaskResult> HeterogeneousComputeEngine::SubmitTask(const ComputeTask& task)
{
    // ---- 前置检查：引擎必须已初始化且正在运行 ----
    if (!initialized_.load(std::memory_order_acquire) ||
        !running_.load(std::memory_order_acquire)) {
        // 引擎未就绪时使用 promise 立即返回错误
        std::promise<SubTaskResult> error_promise;
        auto error_future = error_promise.get_future();
        SubTaskResult error_result;
        error_result.success = false;
        error_result.error_message = "引擎未初始化或已关闭，无法提交任务";
        error_promise.set_value(error_result);
        return error_future;
    }

    // ---- 步骤 1: 分配任务ID ----
    ComputeTask registered_task = task;
    registered_task.task_id = next_task_id_.fetch_add(1, std::memory_order_relaxed);
    registered_task.submit_time = std::chrono::system_clock::now();

    // ---- 步骤 2: 创建 promise/future 对 ----
    auto promise_ptr = std::make_shared<std::promise<SubTaskResult>>();
    auto future = promise_ptr->get_future();

    // ---- 步骤 3: 注册任务 ----
    {
        std::unique_lock<std::shared_mutex> lock(registry_mutex_);
        task_registry_[registered_task.task_id] = registered_task;
        promise_registry_[registered_task.task_id] = promise_ptr;
        // 初始化空的结果列表
        result_registry_[registered_task.task_id] = std::vector<SubTaskResult>();
    }

    // ---- 步骤 4: 更新提交计数 ----
    {
        std::unique_lock<std::shared_mutex> lock(stats_mutex_);
        perf_stats_.total_tasks_submitted++;
    }

    LOG_INFO("任务 #%zu 已提交: 类别=%d, 元素数=%zu, 字节/元素=%zu, 优先级=%d",
             registered_task.task_id,
             static_cast<int>(registered_task.category),
             registered_task.total_elements,
             registered_task.element_size_bytes,
             registered_task.priority);

    // ---- 步骤 5: 执行分析和分解流水线 ----
    process_task_pipeline(registered_task.task_id);

    // ---- 步骤 6: 返回 future ----
    return future;
}

// ============================================================================
// 任务提交接口 - SubmitBoardEvaluation (棋盘评估)
// ============================================================================

std::future<MergedBoardResult> HeterogeneousComputeEngine::SubmitBoardEvaluation(
    void* board_data, size_t board_count)
{
    // ---- 前置检查 ----
    if (!initialized_.load(std::memory_order_acquire) ||
        !running_.load(std::memory_order_acquire)) {
        std::promise<MergedBoardResult> error_promise;
        auto error_future = error_promise.get_future();
        MergedBoardResult error_result;
        error_result.merge_success = false;
        error_promise.set_value(error_result);
        return error_future;
    }

    // ---- 创建 ComputeTask ----
    ComputeTask task;
    task.task_id = next_task_id_.fetch_add(1, std::memory_order_relaxed);
    task.category = TaskCategory::BOARD_EVALUATION;
    task.total_elements = board_count;
    task.element_size_bytes = sizeof(uint64_t) * 6;  // 围棋位棋盘约48字节
    task.input_data = board_data;
    task.output_data = nullptr;  // 棋盘评估结果通过 promise 返回，不走 output_data
    task.callback = nullptr;
    task.submit_time = std::chrono::system_clock::now();
    task.priority = 8;  // 棋盘评估默认较高优先级（影响搜索质量）

    // ---- 创建类型化的 promise/future ----
    auto promise_ptr = std::make_shared<std::promise<MergedBoardResult>>();
    auto future = promise_ptr->get_future();

    // ---- 注册任务 ----
    {
        std::unique_lock<std::shared_mutex> lock(registry_mutex_);
        task_registry_[task.task_id] = task;
        board_promise_registry_[task.task_id] = promise_ptr;
        result_registry_[task.task_id] = std::vector<SubTaskResult>();
    }

    // ---- 更新统计 ----
    {
        std::unique_lock<std::shared_mutex> lock(stats_mutex_);
        perf_stats_.total_tasks_submitted++;
    }

    LOG_INFO("棋盘评估任务 #%zu 已提交: %zu 个棋盘", task.task_id, board_count);

    // ---- 执行流水线 ----
    process_task_pipeline(task.task_id);

    return future;
}

// ============================================================================
// 任务提交接口 - SubmitNeuralInference (神经网络推理)
// ============================================================================

std::future<MergedInferenceResult> HeterogeneousComputeEngine::SubmitNeuralInference(
    void* input_tensor, size_t batch_size, int policy_dim)
{
    // ---- 前置检查 ----
    if (!initialized_.load(std::memory_order_acquire) ||
        !running_.load(std::memory_order_acquire)) {
        std::promise<MergedInferenceResult> error_promise;
        auto error_future = error_promise.get_future();
        MergedInferenceResult error_result;
        error_result.merge_success = false;
        error_promise.set_value(error_result);
        return error_future;
    }

    // ---- 创建 ComputeTask ----
    ComputeTask task;
    task.task_id = next_task_id_.fetch_add(1, std::memory_order_relaxed);
    task.category = TaskCategory::NEURAL_INFERENCE;
    task.total_elements = batch_size;
    task.element_size_bytes = static_cast<size_t>(policy_dim) * sizeof(float);
    task.input_data = input_tensor;
    task.output_data = nullptr;
    task.callback = nullptr;
    task.submit_time = std::chrono::system_clock::now();
    task.priority = 9;  // 神经网络推理默认高优先级

    // ---- 特殊策略：小批次强制走 CPU ----
    // 批次小于 min_batch_for_gpu 时，GPU启动开销大于计算收益
    // 此逻辑在 process_task_pipeline 中通过调整分配比例实现
    if (batch_size < config_.min_batch_for_gpu) {
        LOG_INFO("推理任务 #%zu: 批次大小 %zu < GPU最小批次 %zu，将强制使用 CPU",
                 task.task_id, batch_size, config_.min_batch_for_gpu);
    }

    // ---- 创建类型化的 promise/future ----
    auto promise_ptr = std::make_shared<std::promise<MergedInferenceResult>>();
    auto future = promise_ptr->get_future();

    // ---- 注册任务 ----
    {
        std::unique_lock<std::shared_mutex> lock(registry_mutex_);
        task_registry_[task.task_id] = task;
        inference_promise_registry_[task.task_id] = promise_ptr;
        result_registry_[task.task_id] = std::vector<SubTaskResult>();
    }

    // ---- 更新统计 ----
    {
        std::unique_lock<std::shared_mutex> lock(stats_mutex_);
        perf_stats_.total_tasks_submitted++;
    }

    LOG_INFO("推理任务 #%zu 已提交: 批次=%zu, 策略维度=%d",
             task.task_id, batch_size, policy_dim);

    // ---- 执行流水线 ----
    process_task_pipeline(task.task_id);

    return future;
}

// ============================================================================
// 任务提交接口 - SubmitTopologyCalculation (拓扑计算)
// ============================================================================

std::future<MergedTopologyResult> HeterogeneousComputeEngine::SubmitTopologyCalculation(
    void* region_data, size_t region_count)
{
    // ---- 前置检查 ----
    if (!initialized_.load(std::memory_order_acquire) ||
        !running_.load(std::memory_order_acquire)) {
        std::promise<MergedTopologyResult> error_promise;
        auto error_future = error_promise.get_future();
        MergedTopologyResult error_result;
        error_result.merge_success = false;
        error_promise.set_value(error_result);
        return error_future;
    }

    // ---- 创建 ComputeTask ----
    // 拓扑计算默认偏向 CPU（分支多、依赖复杂）
    ComputeTask task;
    task.task_id = next_task_id_.fetch_add(1, std::memory_order_relaxed);
    task.category = TaskCategory::TOPOLOGY_CALCULATION;
    task.total_elements = region_count;
    task.element_size_bytes = sizeof(double) * 3;  // 每个区域: connectivity + centrality + strategic
    task.input_data = region_data;
    task.output_data = nullptr;
    task.callback = nullptr;
    task.submit_time = std::chrono::system_clock::now();
    task.priority = 6;  // 拓扑计算默认中等偏上优先级

    // ---- 创建类型化的 promise/future ----
    auto promise_ptr = std::make_shared<std::promise<MergedTopologyResult>>();
    auto future = promise_ptr->get_future();

    // ---- 注册任务 ----
    {
        std::unique_lock<std::shared_mutex> lock(registry_mutex_);
        task_registry_[task.task_id] = task;
        topology_promise_registry_[task.task_id] = promise_ptr;
        result_registry_[task.task_id] = std::vector<SubTaskResult>();
    }

    // ---- 更新统计 ----
    {
        std::unique_lock<std::shared_mutex> lock(stats_mutex_);
        perf_stats_.total_tasks_submitted++;
    }

    LOG_INFO("拓扑计算任务 #%zu 已提交: %zu 个区域", task.task_id, region_count);

    // ---- 执行流水线 ----
    process_task_pipeline(task.task_id);

    return future;
}

// ============================================================================
// 任务提交接口 - SubmitSearchSimulation (搜索仿真)
// ============================================================================

std::future<SubTaskResult> HeterogeneousComputeEngine::SubmitSearchSimulation(
    void* simulation_data, size_t simulation_count)
{
    // ---- 前置检查 ----
    if (!initialized_.load(std::memory_order_acquire) ||
        !running_.load(std::memory_order_acquire)) {
        std::promise<SubTaskResult> error_promise;
        auto error_future = error_promise.get_future();
        SubTaskResult error_result;
        error_result.success = false;
        error_result.error_message = "引擎未初始化或已关闭，无法提交搜索仿真任务";
        error_promise.set_value(error_result);
        return error_future;
    }

    // ---- 创建 ComputeTask ----
    ComputeTask task;
    task.task_id = next_task_id_.fetch_add(1, std::memory_order_relaxed);
    task.category = TaskCategory::SEARCH_SIMULATION;
    task.total_elements = simulation_count;
    task.element_size_bytes = sizeof(void*);  // 每个仿真大约一个指针大小
    task.input_data = simulation_data;
    task.output_data = nullptr;
    task.callback = nullptr;
    task.submit_time = std::chrono::system_clock::now();
    task.priority = 7;  // 搜索仿真默认中高优先级

    // ---- 创建 promise/future ----
    auto promise_ptr = std::make_shared<std::promise<SubTaskResult>>();
    auto future = promise_ptr->get_future();

    // ---- 注册任务 ----
    {
        std::unique_lock<std::shared_mutex> lock(registry_mutex_);
        task_registry_[task.task_id] = task;
        promise_registry_[task.task_id] = promise_ptr;
        result_registry_[task.task_id] = std::vector<SubTaskResult>();
    }

    // ---- 更新统计 ----
    {
        std::unique_lock<std::shared_mutex> lock(stats_mutex_);
        perf_stats_.total_tasks_submitted++;
    }

    LOG_INFO("搜索仿真任务 #%zu 已提交: %zu 个仿真", task.task_id, simulation_count);

    // ---- 执行流水线 ----
    process_task_pipeline(task.task_id);

    return future;
}

// ============================================================================
// 内部方法 - process_task_pipeline (任务提交流水线)
// ============================================================================

/**
 * @brief 对已注册的任务执行分析和分解流水线
 *
 * 此方法是 Engine 的核心调度逻辑入口。它协调 WorkloadAnalyzer、
 * DeviceLoadBalancer 和 TaskDecomposer 三个子模块，完成从任务描述
 * 到可执行子任务列表的完整转换。
 *
 * @param task_id 已在 task_registry_ 中注册的任务ID
 */
void HeterogeneousComputeEngine::process_task_pipeline(size_t task_id)
{
    ComputeTask task;
    {
        std::shared_lock<std::shared_mutex> lock(registry_mutex_);
        auto it = task_registry_.find(task_id);
        if (it == task_registry_.end()) {
            LOG_ERROR("process_task_pipeline: 任务 #%zu 不在注册表中！", task_id);
            return;
        }
        task = it->second;
    }

    // ---- 边界处理：零元素任务立即完成 ----
    if (task.total_elements == 0) {
        LOG_INFO("任务 #%zu: 总元素数为0，立即标记完成", task_id);

        // 创建空的分解结果
        DecompositionResult empty_decomp;
        empty_decomp.total_subtask_count = 0;
        empty_decomp.cpu_ratio = 0.0;
        empty_decomp.gpu_ratio = 0.0;

        {
            std::unique_lock<std::shared_mutex> lock(registry_mutex_);
            decomposition_registry_[task_id] = empty_decomp;

            // 直接触发合并（空结果集）
            collect_and_merge_results(task_id);
        }
        return;
    }

    // ---- 步骤 1: 映射任务参数到分析器参数 ----
    size_t batch_size = 0;
    size_t element_count = 0;
    bool is_tensor_op = false;
    int tensor_dims = 0;
    map_task_to_analyzer_params(task, batch_size, element_count, is_tensor_op, tensor_dims);

    LOG_INFO("任务 #%zu 流水线: batch_size=%zu, element_count=%zu, is_tensor=%s, tensor_dims=%d",
             task_id, batch_size, element_count,
             is_tensor_op ? "是" : "否", tensor_dims);

    // ---- 步骤 2: 工作负载分析 ----
    TaskFeature feature = workload_analyzer_.analyze_task(
        task.category, batch_size, element_count, is_tensor_op, tensor_dims);

    LOG_INFO("任务 #%zu 特征分析: 矩阵密度=%.2f, 数据规模=%zu, 访存模式=%d, 并行度=%.2f",
             task_id, feature.matrix_density,
             feature.data_scale,
             static_cast<int>(feature.mem_pattern),
             feature.parallelism_degree);

    // ---- 步骤 3: 设备亲和度计算 ----
    DeviceAffinity affinity = workload_analyzer_.calculate_device_affinity(
        feature, cpu_capability_, gpu_capability_);

    LOG_INFO("任务 #%zu 亲和度: CPU=%.3f, GPU=%.3f, 推荐设备=%d, 置信度=%.2f",
             task_id, affinity.cpu_affinity_score,
             affinity.gpu_affinity_score,
             static_cast<int>(affinity.recommended_device),
             affinity.confidence);

    // ---- 步骤 4: 负载均衡 - 获取当前动态分配比例 ----
    AllocationDecision allocation = load_balancer_.calculate_allocation_ratio();

    LOG_INFO("任务 #%zu 分配决策: CPU比例=%.2f, GPU比例=%.2f, 原因=%s",
             task_id, allocation.cpu_ratio,
             allocation.gpu_ratio, allocation.reason.c_str());

    // ---- 步骤 5: 估算最优分解 ----
    auto [cpu_elements, gpu_elements] = task_decomposer_.estimate_optimal_split(
        task.total_elements, cpu_capability_, gpu_capability_, affinity);

    // ---- 特殊处理：小批次推理强制走CPU ----
    if (task.category == TaskCategory::NEURAL_INFERENCE &&
        task.total_elements < config_.min_batch_for_gpu) {
        // 小批次：GPU启动开销 > 计算收益，全部交给CPU
        cpu_elements = task.total_elements;
        gpu_elements = 0;
        LOG_INFO("任务 #%zu: 小批次推理(%zu < %zu)，强制全部使用CPU",
                 task_id, task.total_elements, config_.min_batch_for_gpu);
    }

    // ---- 特殊处理：拓扑计算默认偏向CPU ----
    if (task.category == TaskCategory::TOPOLOGY_CALCULATION) {
        // 拓扑计算有复杂的数据依赖和分支预测，CPU的缓存层次更合适
        // 默认将CPU分配比例至少提高到50%
        size_t cpu_min = task.total_elements / 2;
        if (cpu_elements < cpu_min && task.total_elements > 0) {
            cpu_elements = cpu_min;
            gpu_elements = task.total_elements - cpu_elements;
            LOG_INFO("任务 #%zu: 拓扑计算默认偏向CPU，调整CPU分配为 %zu",
                     task_id, cpu_elements);
        }
    }

    // ---- 如果GPU不可用，全部交给CPU ----
    if (gpu_capability_.compute_cores == 0) {
        cpu_elements = task.total_elements;
        gpu_elements = 0;
    }

    LOG_INFO("任务 #%zu 分解: 总元素=%zu, CPU元素=%zu, GPU元素=%zu",
             task_id, task.total_elements, cpu_elements, gpu_elements);

    // ---- 步骤 6: 执行分解 ----
    DecompositionResult decomposition;
    switch (task.category) {
        case TaskCategory::BOARD_EVALUATION:
            decomposition = task_decomposer_.decompose_board_evaluation(
                task.total_elements, cpu_elements, gpu_elements);
            break;
        case TaskCategory::NEURAL_INFERENCE:
            decomposition = task_decomposer_.decompose_neural_inference(
                task.total_elements, cpu_elements, gpu_elements);
            break;
        case TaskCategory::TOPOLOGY_CALCULATION:
            decomposition = task_decomposer_.decompose_topology_calculation(
                task.total_elements, cpu_elements, gpu_elements);
            break;
        case TaskCategory::SEARCH_SIMULATION:
            decomposition = task_decomposer_.decompose_simulation(
                task.total_elements, cpu_elements, gpu_elements);
            break;
        case TaskCategory::GENERIC_COMPUTE:
        default:
            decomposition = task_decomposer_.decompose_by_batch(
                task.total_elements, cpu_elements, gpu_elements);
            break;
    }

    // ---- 步骤 7: 存储分解结果 ----
    {
        std::unique_lock<std::shared_mutex> lock(registry_mutex_);
        decomposition_registry_[task_id] = decomposition;
    }

    // ---- 更新分解计数 ----
    {
        std::unique_lock<std::shared_mutex> lock(stats_mutex_);
        perf_stats_.task_decompositions++;
    }

    LOG_INFO("任务 #%zu 分解完成: CPU子任务=%zu个, GPU子任务=%zu个, 总计=%zu个",
             task_id, decomposition.cpu_subtasks.size(),
             decomposition.gpu_subtasks.size(),
             decomposition.total_subtask_count);

    // ---- 步骤 8: 赋予子任务全局唯一ID并CPU子任务入队 ----
    // 子任务ID编码方案：全局子任务ID = 父任务ID * SUBTASK_ID_MULTIPLIER + 局部子任务ID
    // 这样确保在跨任务场景下子任务ID全局唯一，工作线程可以逆向解析出父任务ID

    {
        std::lock_guard<std::mutex> lock(cpu_queue_mutex_);
        for (auto& subtask : decomposition.cpu_subtasks) {
            // 将局部子任务ID编码为全局唯一ID
            subtask.subtask_id = task_id * SUBTASK_ID_MULTIPLIER + subtask.subtask_id;
            // 设置子任务的提交时间
            subtask.submit_time = std::chrono::system_clock::now();
            cpu_task_queue_.push(subtask);
        }
    }
    // 通知 CPU 工作线程有新任务
    if (!decomposition.cpu_subtasks.empty()) {
        cpu_queue_cv_.notify_all();
    }

    // ---- 步骤 9: 赋予子任务全局唯一ID并GPU子任务入队 ----
    if (!decomposition.gpu_subtasks.empty()) {
        {
            std::lock_guard<std::mutex> lock(gpu_queue_mutex_);
            for (auto& subtask : decomposition.gpu_subtasks) {
                // 将局部子任务ID编码为全局唯一ID
                subtask.subtask_id = task_id * SUBTASK_ID_MULTIPLIER + subtask.subtask_id;
                subtask.submit_time = std::chrono::system_clock::now();
                gpu_task_queue_.push(subtask);
            }
        }
        // 通知 GPU 工作线程有新任务
        gpu_queue_cv_.notify_all();
    }

    // ---- 步骤 10: 更新队列深度到负载均衡器 ----
    {
        std::lock_guard<std::mutex> cpu_lock(cpu_queue_mutex_);
        std::lock_guard<std::mutex> gpu_lock(gpu_queue_mutex_);
        load_balancer_.update_cpu_queue_depth(cpu_task_queue_.size());
        load_balancer_.update_gpu_queue_depth(gpu_task_queue_.size());
    }
}

// ============================================================================
// 内部方法 - map_task_to_analyzer_params
// ============================================================================

/**
 * @brief 将 ComputeTask 的 category 映射为 WorkloadAnalyzer 所需的参数
 *
 * 不同任务类别的参数映射逻辑：
 *   - BOARD_EVALUATION: batch_size=棋盘数, element_count=361(围棋交叉点数)
 *   - NEURAL_INFERENCE: batch_size=批次大小, element_count=批次大小,
 *                       is_tensor_op=true, tensor_dims=2(默认)
 *   - TOPOLOGY_CALCULATION: batch_size=区域数, element_count=区域数
 *   - SEARCH_SIMULATION: batch_size=仿真数, element_count=仿真数
 *   - GENERIC_COMPUTE: batch_size=元素数, element_count=元素数
 */
void HeterogeneousComputeEngine::map_task_to_analyzer_params(
    const ComputeTask& task,
    size_t& batch_size,
    size_t& element_count,
    bool& is_tensor_op,
    int& tensor_dims)
{
    switch (task.category) {
        case TaskCategory::BOARD_EVALUATION:
            // 围棋棋盘评估：每个棋盘有361个交叉点需要评估
            batch_size = task.total_elements;
            element_count = GO_BOARD_POINTS;  // 361
            is_tensor_op = false;
            tensor_dims = 0;
            break;

        case TaskCategory::NEURAL_INFERENCE:
            // 神经网络推理：密集矩阵运算，使用张量分析
            batch_size = task.total_elements;
            element_count = task.total_elements;  // 每个样本一个输出
            is_tensor_op = true;
            tensor_dims = DEFAULT_TENSOR_DIMS;  // 2维矩阵乘法
            break;

        case TaskCategory::TOPOLOGY_CALCULATION:
            // 拓扑计算：按区域为单位进行分析
            batch_size = task.total_elements;
            element_count = task.total_elements;
            is_tensor_op = false;
            tensor_dims = 0;
            break;

        case TaskCategory::SEARCH_SIMULATION:
            // 搜索仿真：每个仿真独立执行
            batch_size = task.total_elements;
            element_count = task.total_elements;
            is_tensor_op = false;
            tensor_dims = 0;
            break;

        case TaskCategory::GENERIC_COMPUTE:
        default:
            // 通用计算：保守的参数映射
            batch_size = task.total_elements;
            element_count = task.total_elements;
            is_tensor_op = false;
            tensor_dims = 0;
            break;
    }
}

// ============================================================================
// 调度器主循环 - scheduler_loop()
// ============================================================================

/**
 * @brief 调度器线程的主循环
 *
 * 周期性执行以下操作：
 *   1. 更新队列深度信息到 DeviceLoadBalancer
 *   2. 调用负载均衡器获取当前最优分配比例
 *   3. 检测设备过载并触发分流/回流
 *   4. 扫描 result_registry_，收集已完成的任务并触发合并
 *   5. 清理已完成的任务
 *   6. 按配置的间隔休眠
 *
 * 退出条件：stop_requested_ == true
 */
void HeterogeneousComputeEngine::scheduler_loop()
{
    LOG_INFO("调度器线程已启动（间隔=%zu ms）", config_.monitoring_interval_ms);

    while (!stop_requested_.load(std::memory_order_acquire)) {

        // ---- 操作 1: 更新队列深度 ----
        {
            std::lock_guard<std::mutex> cpu_lock(cpu_queue_mutex_);
            std::lock_guard<std::mutex> gpu_lock(gpu_queue_mutex_);
            load_balancer_.update_cpu_queue_depth(cpu_task_queue_.size());
            load_balancer_.update_gpu_queue_depth(gpu_task_queue_.size());
        }

        // ---- 操作 2: 获取最新分配比例 ----
        AllocationDecision allocation = load_balancer_.calculate_allocation_ratio();

        // ---- 操作 3: 检测过载情况 ----
        if (allocation.should_offload_to_cpu) {
            LOG_INFO("调度器: GPU过载，建议将任务分流到CPU (GPU分配: %.2f -> CPU分配: %.2f)",
                     allocation.gpu_ratio, allocation.cpu_ratio);
        }
        if (allocation.should_route_to_gpu) {
            LOG_INFO("调度器: CPU过载，建议将任务回流到GPU (CPU分配: %.2f -> GPU分配: %.2f)",
                     allocation.cpu_ratio, allocation.gpu_ratio);
        }

        // ---- 操作 4: 扫描已完成任务并触发合并 ----
        {
            std::unique_lock<std::shared_mutex> lock(registry_mutex_);

            // 收集所有需要检查的任务ID（遍历副本以避免在迭代中修改容器）
            std::vector<size_t> task_ids_to_check;
            task_ids_to_check.reserve(task_registry_.size());
            for (const auto& pair : task_registry_) {
                task_ids_to_check.push_back(pair.first);
            }

            // 对每个任务检查是否所有子任务已完成
            for (size_t tid : task_ids_to_check) {
                // 获取该任务的分解结果
                auto decomp_it = decomposition_registry_.find(tid);
                if (decomp_it == decomposition_registry_.end()) {
                    continue;  // 分解结果尚未就绪
                }

                // 获取该任务已完成的结果列表
                auto result_it = result_registry_.find(tid);
                if (result_it == result_registry_.end()) {
                    continue;  // 结果列表尚未初始化
                }

                size_t total_subtasks = decomp_it->second.total_subtask_count;
                size_t completed_count = result_it->second.size();

                // 判断是否所有子任务已完成
                if (completed_count >= total_subtasks && total_subtasks > 0) {
                    // 所有子任务已完成，触发合并
                    collect_and_merge_results(tid);
                }
            }
        }

        // ---- 操作 5: 休眠指定间隔 ----
        // 使用短休眠循环来响应 stop_requested_ 标志
        auto sleep_start = std::chrono::system_clock::now();
        while (!stop_requested_.load(std::memory_order_acquire)) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now() - sleep_start).count();
            if (static_cast<size_t>(elapsed) >= config_.monitoring_interval_ms) {
                break;
            }
            // 短休眠（10ms）以响应停止请求
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    LOG_INFO("调度器线程已退出");
}

// ============================================================================
// CPU 工作线程 - cpu_worker_loop()
// ============================================================================

/**
 * @brief CPU 工作线程的主循环
 *
 * 从 cpu_task_queue_ 中取出子任务并执行。
 * 使用条件变量进行阻塞等待，在 stop_requested_ 时安全退出。
 *
 * @param worker_id CPU工作线程编号
 */
void HeterogeneousComputeEngine::cpu_worker_loop(int worker_id)
{
    LOG_INFO("CPU工作线程 #%d 已启动", worker_id);

    while (true) {
        SubTask subtask;
        bool has_task = false;

        // ---- 从队列取任务（带超时的条件变量等待） ----
        {
            std::unique_lock<std::mutex> lock(cpu_queue_mutex_);

            // 等待条件：队列非空 OR 停止请求
            cpu_queue_cv_.wait(lock, [this]() {
                return !cpu_task_queue_.empty() ||
                       stop_requested_.load(std::memory_order_acquire);
            });

            // 检查退出条件：已请求停止且队列为空
            if (stop_requested_.load(std::memory_order_acquire) &&
                cpu_task_queue_.empty()) {
                break;  // 退出循环
            }

            // 取出队首任务
            if (!cpu_task_queue_.empty()) {
                subtask = cpu_task_queue_.front();
                cpu_task_queue_.pop();
                has_task = true;
            }
        }

        // 如果没有取到任务（空唤醒），继续等待
        if (!has_task) {
            continue;
        }

        // ---- 执行子任务 ----
        // 更新子任务状态为 RUNNING 并记录开始时间
        subtask.status = SubTaskStatus::RUNNING;
        subtask.start_time = std::chrono::system_clock::now();

        // 从全局唯一子任务ID逆向解析父任务ID
        // 编码方案: 全局子任务ID = 父任务ID * SUBTASK_ID_MULTIPLIER + 局部子任务ID
        // 因此: 父任务ID = 全局子任务ID / SUBTASK_ID_MULTIPLIER
        size_t parent_task_id = subtask.subtask_id / SUBTASK_ID_MULTIPLIER;

        // 从注册表获取父任务信息
        ComputeTask parent_task;
        bool found_parent = false;
        {
            std::shared_lock<std::shared_mutex> lock(registry_mutex_);
            auto task_it = task_registry_.find(parent_task_id);
            if (task_it != task_registry_.end()) {
                parent_task = task_it->second;
                found_parent = true;
            }
        }

        if (!found_parent) {
            // 父任务可能已被清理（引擎关闭等场景）
            LOG_WARNING("CPU工作线程 #%d: 无法找到父任务 #%zu (子任务 #%zu)，跳过执行",
                        worker_id, parent_task_id, subtask.subtask_id);
            continue;
        }

        // 执行CPU子任务计算
        SubTaskResult result = execute_cpu_subtask(subtask, parent_task);

        // 设置完成时间和来源设备
        result.subtask_id = subtask.subtask_id;
        result.source_device = ComputeDeviceType::CPU;
        result.data_offset = subtask.data_offset;
        result.data_length = subtask.data_length;
        subtask.complete_time = std::chrono::system_clock::now();
        subtask.status = result.success ? SubTaskStatus::COMPLETED : SubTaskStatus::FAILED;

        // ---- 计算执行时间 ----
        auto exec_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            subtask.complete_time - subtask.start_time).count();
        double exec_time_ms = static_cast<double>(exec_time_us) / 1000.0;

        // ---- 存储结果到注册表 ----
        {
            std::unique_lock<std::shared_mutex> lock(registry_mutex_);
            auto result_it = result_registry_.find(parent_task_id);
            if (result_it != result_registry_.end()) {
                result_it->second.push_back(result);
            }

            // 调用回调通知（如果设置了）
            auto task_it = task_registry_.find(parent_task_id);
            if (task_it != task_registry_.end() && task_it->second.callback) {
                // 注意：回调在工作线程中执行，调用方需确保线程安全
                task_it->second.callback(result);
            }
        }

        // ---- 更新性能统计 ----
        update_performance_stats(ComputeDeviceType::CPU, exec_time_ms);

        LOG_INFO("CPU工作线程 #%d: 完成子任务 #%zu (父任务 #%zu), 耗时=%.2f ms, 成功=%s",
                 worker_id, subtask.subtask_id, parent_task_id,
                 exec_time_ms, result.success ? "是" : "否");
    }

    LOG_INFO("CPU工作线程 #%d 已退出", worker_id);
}

// ============================================================================
// GPU 工作线程 - gpu_worker_loop()
// ============================================================================

/**
 * @brief GPU 工作线程的主循环
 *
 * 从 gpu_task_queue_ 中取出子任务并使用CUDA加速执行。
 * 每个GPU工作线程对应一个独立的CUDA流，实现多流并发。
 *
 * @param worker_id GPU工作线程编号（对应CUDA流索引）
 */
void HeterogeneousComputeEngine::gpu_worker_loop(int worker_id)
{
    LOG_INFO("GPU工作线程 #%d 已启动", worker_id);

    // ---- 设置此线程使用的 CUDA 设备 ----
#if SPACEGO_CUDA_ENABLED
    cudaError_t set_dev_err = cudaSetDevice(0);  // 使用第一块GPU
    if (set_dev_err != cudaSuccess) {
        LOG_ERROR("GPU工作线程 #%d: cudaSetDevice(0) 失败: %s",
                  worker_id, cudaGetErrorString(set_dev_err));
    }
#endif

    while (true) {
        SubTask subtask;
        bool has_task = false;

        // ---- 从队列取任务 ----
        {
            std::unique_lock<std::mutex> lock(gpu_queue_mutex_);

            gpu_queue_cv_.wait(lock, [this]() {
                return !gpu_task_queue_.empty() ||
                       stop_requested_.load(std::memory_order_acquire);
            });

            // 退出条件
            if (stop_requested_.load(std::memory_order_acquire) &&
                gpu_task_queue_.empty()) {
                break;
            }

            if (!gpu_task_queue_.empty()) {
                subtask = gpu_task_queue_.front();
                gpu_task_queue_.pop();
                has_task = true;
            }
        }

        if (!has_task) {
            continue;
        }

        // ---- 执行子任务 ----
        subtask.status = SubTaskStatus::RUNNING;
        subtask.start_time = std::chrono::system_clock::now();

        // 从全局唯一子任务ID逆向解析父任务ID
        size_t parent_task_id = subtask.subtask_id / SUBTASK_ID_MULTIPLIER;

        // 查找父任务
        ComputeTask parent_task;
        bool found_parent = false;
        {
            std::shared_lock<std::shared_mutex> lock(registry_mutex_);
            auto task_it = task_registry_.find(parent_task_id);
            if (task_it != task_registry_.end()) {
                parent_task = task_it->second;
                found_parent = true;
            }
        }

        if (!found_parent) {
            LOG_WARNING("GPU工作线程 #%d: 无法找到父任务 #%zu (子任务 #%zu)，跳过执行",
                        worker_id, parent_task_id, subtask.subtask_id);
            continue;
        }

        // 执行GPU子任务计算
        SubTaskResult result = execute_gpu_subtask(subtask, parent_task);

        // 设置结果元数据
        result.subtask_id = subtask.subtask_id;
        result.source_device = ComputeDeviceType::GPU;
        result.data_offset = subtask.data_offset;
        result.data_length = subtask.data_length;
        subtask.complete_time = std::chrono::system_clock::now();
        subtask.status = result.success ? SubTaskStatus::COMPLETED : SubTaskStatus::FAILED;

        // ---- 计算执行时间 ----
        auto exec_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            subtask.complete_time - subtask.start_time).count();
        double exec_time_ms = static_cast<double>(exec_time_us) / 1000.0;

        // ---- 存储结果 ----
        {
            std::unique_lock<std::shared_mutex> lock(registry_mutex_);
            auto result_it = result_registry_.find(parent_task_id);
            if (result_it != result_registry_.end()) {
                result_it->second.push_back(result);
            }

            // 调用回调
            auto task_it = task_registry_.find(parent_task_id);
            if (task_it != task_registry_.end() && task_it->second.callback) {
                task_it->second.callback(result);
            }
        }

        // ---- 更新性能统计 ----
        update_performance_stats(ComputeDeviceType::GPU, exec_time_ms);

        LOG_INFO("GPU工作线程 #%d: 完成子任务 #%zu (父任务 #%zu), 耗时=%.2f ms, 成功=%s",
                 worker_id, subtask.subtask_id, parent_task_id,
                 exec_time_ms, result.success ? "是" : "否");
    }

    LOG_INFO("GPU工作线程 #%d 已退出", worker_id);
}

// ============================================================================
// 子任务执行器 - execute_cpu_subtask()
// ============================================================================

/**
 * @brief 在CPU上执行一个子任务
 *
 * 根据父任务类别选择对应的CPU实现路径。
 * 当前实现为占位符（placeholder），标记了各实现应该被调用的位置。
 * 后续在集成阶段将替换为对实际CPU实现的调用。
 *
 * @param subtask 子任务描述
 * @param task    父任务描述
 * @return SubTaskResult 执行结果
 */
SubTaskResult HeterogeneousComputeEngine::execute_cpu_subtask(
    const SubTask& subtask, const ComputeTask& task)
{
    SubTaskResult result;
    result.source_device = ComputeDeviceType::CPU;
    result.data_offset = subtask.data_offset;
    result.data_length = subtask.data_length;

    switch (task.category) {
        case TaskCategory::BOARD_EVALUATION:
            // ================================================================
            // TODO: 集成 GoBoardCPU 评估逻辑
            // 此处应调用围棋棋盘CPU评估实现：
            //   - 计算合法着法列表
            //   - 计算气数统计
            //   - 将结果序列化到 raw_data
            //
            // 当前使用占位符数据，返回成功但空的结果集
            // ================================================================
            {
                // 占位符：生成空的棋盘评估结果
                // 格式：[boards_count:4B][per_board: board_count个每个board的评估数据]
                size_t boards_in_subtask = subtask.data_length;
                result.raw_data.resize(SERIALIZATION_BUFFER_INITIAL_SIZE);
                result.success = true;
                result.error_message.clear();
            }
            break;

        case TaskCategory::NEURAL_INFERENCE:
            // ================================================================
            // TODO: 集成 NeuralNetworkCPU 推理
            // 此处应调用神经网络CPU推理实现：
            //   - 加载模型权重（LibTorch CPU后端）
            //   - 执行前向传播
            //   - 输出 policy 和 value
            //   - 将结果序列化到 raw_data
            //
            // 序列化格式：
            //   [samples_count:4B][per_sample: samples_count个]
            //   每个sample: [policy: policy_dim*4B][value: 4B]
            // ================================================================
            {
                size_t samples_in_subtask = subtask.data_length;
                result.raw_data.resize(SERIALIZATION_BUFFER_INITIAL_SIZE);
                result.success = true;
                result.error_message.clear();
            }
            break;

        case TaskCategory::TOPOLOGY_CALCULATION:
            // ================================================================
            // TODO: 集成 SpatialTopologyCalculator CPU路径
            // 此处应调用空间拓扑CPU计算实现：
            //   - 计算连通性分数
            //   - 计算中心性分数
            //   - 计算战略价值分数
            //   - 将结果序列化到 raw_data
            //
            // 序列化格式：
            //   [regions_count:4B][per_region: regions_count个]
            //   每个region: [connectivity:8B][centrality:8B][strategic_value:8B]
            // ================================================================
            {
                size_t regions_in_subtask = subtask.data_length;
                result.raw_data.resize(SERIALIZATION_BUFFER_INITIAL_SIZE);
                result.success = true;
                result.error_message.clear();
            }
            break;

        case TaskCategory::SEARCH_SIMULATION:
            // ================================================================
            // TODO: 集成 MCTS仿真CPU执行
            // 此处应调用搜索仿真CPU实现：
            //   - 执行蒙特卡洛树搜索模拟
            //   - 记录仿真结果（胜率、访问次数等）
            //   - 将结果序列化到 raw_data
            // ================================================================
            {
                size_t sims_in_subtask = subtask.data_length;
                result.raw_data.resize(SERIALIZATION_BUFFER_INITIAL_SIZE);
                result.success = true;
                result.error_message.clear();
            }
            break;

        case TaskCategory::GENERIC_COMPUTE:
        default:
            // ================================================================
            // 通用CPU计算路径 - 直接复制输入数据到输出缓冲区
            // 适用于调用方已自行实现了计算逻辑的场景
            // ================================================================
            {
                if (task.input_data != nullptr && task.output_data != nullptr) {
                    // 安全的数据复制
                    size_t copy_bytes = subtask.data_length * task.element_size_bytes;
                    // 复制输入到输出（占位符实现）
                    std::memcpy(task.output_data, task.input_data,
                                (std::min)(copy_bytes, static_cast<size_t>(1024 * 1024)));
                }
                result.raw_data.resize(SERIALIZATION_BUFFER_INITIAL_SIZE);
                result.success = true;
                result.error_message.clear();
            }
            break;
    }

    return result;
}

// ============================================================================
// 子任务执行器 - execute_gpu_subtask()
// ============================================================================

/**
 * @brief 在GPU上执行一个子任务（CUDA加速）
 *
 * 根据父任务类别选择对应的CUDA加速实现路径。
 * 如果CUDA不可用，标记结果为失败。
 *
 * @param subtask 子任务描述
 * @param task    父任务描述
 * @return SubTaskResult 执行结果
 */
SubTaskResult HeterogeneousComputeEngine::execute_gpu_subtask(
    const SubTask& subtask, const ComputeTask& task)
{
    SubTaskResult result;
    result.source_device = ComputeDeviceType::GPU;
    result.data_offset = subtask.data_offset;
    result.data_length = subtask.data_length;

#if SPACEGO_CUDA_ENABLED
    // ---- CUDA 环境可用时的GPU执行路径 ----

    switch (task.category) {
        case TaskCategory::BOARD_EVALUATION:
            // ================================================================
            // TODO: 集成 GoBoardCUDA 批量棋盘评估
            // 此处应调用CUDA加速的围棋棋盘批量评估：
            //   - 使用CUDA内核并行计算多个棋盘的合法着法和气数
            //   - 支持 Tensor Core 加速（如果可用）
            //   - 支持 FP16 混合精度（如果设备支持）
            //   - 将结果从 GPU 显存拷贝回主机内存
            //   - 序列化到 raw_data
            //
            // 性能优化要点：
            //   - 使用共享内存缓存棋盘状态
            //   - 使用 warp 级别归约计算气数
            //   - 合并访存模式访问棋盘数据
            // ================================================================
            {
                size_t boards_in_subtask = subtask.data_length;
                result.raw_data.resize(SERIALIZATION_BUFFER_INITIAL_SIZE);
                result.success = true;
                result.error_message.clear();
            }
            break;

        case TaskCategory::NEURAL_INFERENCE:
            // ================================================================
            // TODO: 集成 NeuralNetworkCUDA 推理
            // 此处应调用CUDA加速的神经网络推理：
            //   - 使用 LibTorch CUDA 后端进行前向传播
            //   - 利用 Tensor Core 加速矩阵乘法
            //   - 使用 FP16 混合精度减少显存占用和计算时间
            //   - 支持批量推理（一次处理多个输入）
            //   - 将 policy 和 value 输出从GPU拷贝回CPU
            //   - 序列化到 raw_data
            //
            // 序列化格式同CPU版本
            // ================================================================
            {
                size_t samples_in_subtask = subtask.data_length;
                result.raw_data.resize(SERIALIZATION_BUFFER_INITIAL_SIZE);
                result.success = true;
                result.error_message.clear();
            }
            break;

        case TaskCategory::TOPOLOGY_CALCULATION:
            // ================================================================
            // TODO: 集成 SpatialComputationAccelerator CUDA路径
            // 此处应调用CUDA加速的空间拓扑计算：
            //   - 使用CUDA并行计算多个区域的连通性和中心性
            //   - 利用图算法CUDA实现（如并行BFS/DFS）
            //   - 批量处理区域组以减少内核启动次数
            //   - 序列化到 raw_data
            // ================================================================
            {
                size_t regions_in_subtask = subtask.data_length;
                result.raw_data.resize(SERIALIZATION_BUFFER_INITIAL_SIZE);
                result.success = true;
                result.error_message.clear();
            }
            break;

        case TaskCategory::SEARCH_SIMULATION:
            // ================================================================
            // TODO: 集成 CUDA加速的批量MCTS仿真
            // 此处应调用CUDA加速的搜索仿真：
            //   - 并行执行多个独立的MCTS仿真
            //   - 每个仿真在独立的CUDA线程块中运行
            //   - 批量随机数生成（cuRAND）
            //   - 使用原子操作更新共享统计信息
            //   - 序列化结果到 raw_data
            // ================================================================
            {
                size_t sims_in_subtask = subtask.data_length;
                result.raw_data.resize(SERIALIZATION_BUFFER_INITIAL_SIZE);
                result.success = true;
                result.error_message.clear();
            }
            break;

        case TaskCategory::GENERIC_COMPUTE:
        default:
            // 通用GPU计算路径
            {
                if (task.input_data != nullptr && task.output_data != nullptr) {
                    size_t copy_bytes = subtask.data_length * task.element_size_bytes;
                    // 异步拷贝输入到GPU，处理后拷贝回输出
                    // 占位符实现：简单的 cudaMemcpy
                    // TODO: 替换为实际GPU计算内核
                    cudaMemcpy(task.output_data, task.input_data,
                               (std::min)(copy_bytes, static_cast<size_t>(1024 * 1024)),
                               cudaMemcpyDefault);
                }
                result.raw_data.resize(SERIALIZATION_BUFFER_INITIAL_SIZE);
                result.success = true;
                result.error_message.clear();
            }
            break;
    }

#else
    // ---- CUDA 环境不可用时的回退处理 ----
    // 标记结果为失败，设置错误信息
    result.success = false;
    result.error_message = "GPU不可用：编译时未启用CUDA支持 (SPACEGO_CUDA_ENABLED=0)";
    result.raw_data.clear();

    LOG_WARNING("GPU子任务 #%zu 因CUDA不可用而失败", subtask.subtask_id);
#endif  // SPACEGO_CUDA_ENABLED

    return result;
}

// ============================================================================
// 结果收集与合并 - collect_and_merge_results()
// ============================================================================

/**
 * @brief 收集某个任务的所有子任务结果并执行合并
 *
 * 当所有子任务完成后，调度器调用此方法进行合并。
 * 合并后的结果通过相应的 promise 通知调用方。
 *
 * 调用方必须持有 registry_mutex_ 的 unique_lock。
 *
 * @param task_id 已完成所有子任务的任务ID
 */
void HeterogeneousComputeEngine::collect_and_merge_results(size_t task_id)
{
    // ---- 获取任务信息 ----
    auto task_it = task_registry_.find(task_id);
    auto decomp_it = decomposition_registry_.find(task_id);
    auto result_it = result_registry_.find(task_id);

    if (task_it == task_registry_.end() ||
        decomp_it == decomposition_registry_.end() ||
        result_it == result_registry_.end()) {
        LOG_ERROR("collect_and_merge_results: 任务 #%zu 信息不完整！", task_id);
        return;
    }

    const ComputeTask& task = task_it->second;
    const DecompositionResult& decomp = decomp_it->second;
    const std::vector<SubTaskResult>& all_results = result_it->second;

    LOG_INFO("任务 #%zu: 开始合并 %zu 个子任务结果 (类别=%d)",
             task_id, all_results.size(), static_cast<int>(task.category));

    // ---- 按来源设备分离结果 ----
    std::vector<SubTaskResult> cpu_results;
    std::vector<SubTaskResult> gpu_results;
    cpu_results.reserve(all_results.size());
    gpu_results.reserve(all_results.size());

    for (const auto& r : all_results) {
        if (r.source_device == ComputeDeviceType::CPU) {
            cpu_results.push_back(r);
        } else if (r.source_device == ComputeDeviceType::GPU) {
            gpu_results.push_back(r);
        }
    }

    LOG_INFO("任务 #%zu: CPU结果=%zu个, GPU结果=%zu个",
             task_id, cpu_results.size(), gpu_results.size());

    // ---- 根据任务类别选择合并策略 ----
    bool merge_success = false;

    switch (task.category) {
        case TaskCategory::BOARD_EVALUATION: {
            // 使用 ResultMerger 合并棋盘评估结果
            MergedBoardResult merged = result_merger_.merge_board_results(
                cpu_results, gpu_results, task.total_elements);

            merge_success = merged.merge_success;

            // 通知类型化的 promise
            auto board_promise_it = board_promise_registry_.find(task_id);
            if (board_promise_it != board_promise_registry_.end()) {
                board_promise_it->second->set_value(merged);
                board_promise_registry_.erase(board_promise_it);
            }

            // 同时通知通用 promise（如果存在）
            auto promise_it = promise_registry_.find(task_id);
            if (promise_it != promise_registry_.end()) {
                // 将 MergedBoardResult 序列化为 SubTaskResult.raw_data
                SubTaskResult generic_result;
                generic_result.success = merged.merge_success;
                generic_result.source_device = ComputeDeviceType::AUTO;
                generic_result.data_offset = 0;
                generic_result.data_length = task.total_elements;
                generic_result.raw_data.clear();
                promise_it->second->set_value(generic_result);
                promise_registry_.erase(promise_it);
            }

            LOG_INFO("任务 #%zu: 棋盘评估合并%s, %zu个棋盘",
                     task_id, merge_success ? "成功" : "失败", merged.total_boards);
            break;
        }

        case TaskCategory::NEURAL_INFERENCE: {
            // 使用 ResultMerger 合并推理结果
            // 从任务描述中推断 policy_dim（需要额外存储或从上下文获取）
            // 默认使用 362（361个着法点 + pass）
            int policy_dim = 362;

            MergedInferenceResult merged = result_merger_.merge_inference_results(
                cpu_results, gpu_results, task.total_elements, policy_dim);

            merge_success = merged.merge_success;

            // 通知类型化的 promise
            auto inference_promise_it = inference_promise_registry_.find(task_id);
            if (inference_promise_it != inference_promise_registry_.end()) {
                inference_promise_it->second->set_value(merged);
                inference_promise_registry_.erase(inference_promise_it);
            }

            // 通知通用 promise
            auto promise_it = promise_registry_.find(task_id);
            if (promise_it != promise_registry_.end()) {
                SubTaskResult generic_result;
                generic_result.success = merged.merge_success;
                generic_result.source_device = ComputeDeviceType::AUTO;
                generic_result.data_offset = 0;
                generic_result.data_length = task.total_elements;
                promise_it->second->set_value(generic_result);
                promise_registry_.erase(promise_it);
            }

            LOG_INFO("任务 #%zu: 推理合并%s, %zu个样本",
                     task_id, merge_success ? "成功" : "失败", merged.total_batch_size);
            break;
        }

        case TaskCategory::TOPOLOGY_CALCULATION: {
            // 使用 ResultMerger 合并拓扑计算结果
            MergedTopologyResult merged = result_merger_.merge_topology_results(
                cpu_results, gpu_results, task.total_elements);

            merge_success = merged.merge_success;

            // 通知类型化的 promise
            auto topology_promise_it = topology_promise_registry_.find(task_id);
            if (topology_promise_it != topology_promise_registry_.end()) {
                topology_promise_it->second->set_value(merged);
                topology_promise_registry_.erase(topology_promise_it);
            }

            // 通知通用 promise
            auto promise_it = promise_registry_.find(task_id);
            if (promise_it != promise_registry_.end()) {
                SubTaskResult generic_result;
                generic_result.success = merged.merge_success;
                generic_result.source_device = ComputeDeviceType::AUTO;
                generic_result.data_offset = 0;
                generic_result.data_length = task.total_elements;
                promise_it->second->set_value(generic_result);
                promise_registry_.erase(promise_it);
            }

            LOG_INFO("任务 #%zu: 拓扑合并%s, %zu个区域",
                     task_id, merge_success ? "成功" : "失败", merged.total_regions);
            break;
        }

        case TaskCategory::SEARCH_SIMULATION:
        case TaskCategory::GENERIC_COMPUTE:
        default: {
            // 搜索仿真和通用计算：直接合并 raw_data
            SubTaskResult merged_result;
            merged_result.success = true;
            merged_result.source_device = ComputeDeviceType::AUTO;
            merged_result.data_offset = 0;
            merged_result.data_length = task.total_elements;

            // 将所有子任务的 raw_data 拼接起来
            merged_result.raw_data.clear();
            for (const auto& r : all_results) {
                if (r.success) {
                    append_raw_data(merged_result.raw_data, r.raw_data);
                } else {
                    merged_result.success = false;
                    if (!merged_result.error_message.empty()) {
                        merged_result.error_message += "; ";
                    }
                    merged_result.error_message += r.error_message;
                }
            }

            merge_success = merged_result.success;

            // 通知通用 promise
            auto promise_it = promise_registry_.find(task_id);
            if (promise_it != promise_registry_.end()) {
                promise_it->second->set_value(merged_result);
                promise_registry_.erase(promise_it);
            }

            LOG_INFO("任务 #%zu: 通用合并%s, %zu个元素",
                     task_id, merge_success ? "成功" : "失败", task.total_elements);
            break;
        }
    }

    // ---- 从注册表中清理已完成的任务 ----
    task_registry_.erase(task_id);
    decomposition_registry_.erase(task_id);
    result_registry_.erase(task_id);

    // 清理可能残留的 promise（都已在上方处理）

    // ---- 更新完成统计 ----
    {
        std::unique_lock<std::shared_mutex> lock(stats_mutex_);
        perf_stats_.total_tasks_completed++;
    }
}

// ============================================================================
// 性能统计 - update_performance_stats()
// ============================================================================

/**
 * @brief 更新性能统计
 *
 * 在完成一个子任务后，更新对应设备的完成计数和累计执行时间。
 *
 * @param device  执行设备类型
 * @param time_ms 子任务执行耗时（毫秒）
 */
void HeterogeneousComputeEngine::update_performance_stats(
    ComputeDeviceType device, double time_ms)
{
    // 防御性检查：过滤无效的时间值
    if (std::isnan(time_ms) || std::isinf(time_ms) || time_ms < 0.0) {
        return;
    }

    std::unique_lock<std::shared_mutex> lock(stats_mutex_);

    if (device == ComputeDeviceType::CPU) {
        perf_stats_.cpu_tasks_completed++;
        perf_stats_.total_cpu_time_ms += time_ms;
    } else if (device == ComputeDeviceType::GPU) {
        perf_stats_.gpu_tasks_completed++;
        perf_stats_.total_gpu_time_ms += time_ms;
    }
    // AUTO 类型不计入具体设备统计
}

// ============================================================================
// 性能统计 - get_performance_stats()
// ============================================================================

/**
 * @brief 获取性能统计快照
 *
 * 计算派生指标（平均值、吞吐量等）并返回完整的统计快照。
 * 此方法是 const 的，使用 shared_lock 保护读操作。
 *
 * @return EnginePerformanceStats 当前统计快照
 */
EnginePerformanceStats HeterogeneousComputeEngine::get_performance_stats() const
{
    std::shared_lock<std::shared_mutex> lock(stats_mutex_);

    EnginePerformanceStats stats = perf_stats_;  // 复制基础数据

    // ---- 计算派生指标 ----

    // CPU平均执行时间
    if (stats.cpu_tasks_completed > 0) {
        stats.avg_cpu_time_ms = stats.total_cpu_time_ms /
            static_cast<double>(stats.cpu_tasks_completed);
    }

    // GPU平均执行时间
    if (stats.gpu_tasks_completed > 0) {
        stats.avg_gpu_time_ms = stats.total_gpu_time_ms /
            static_cast<double>(stats.gpu_tasks_completed);
    }

    // 吞吐量（任务数/秒）
    auto now = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - engine_start_time_).count();
    if (elapsed > 0 && stats.total_tasks_completed > 0) {
        stats.throughput_tasks_per_sec =
            static_cast<double>(stats.total_tasks_completed) /
            (static_cast<double>(elapsed) / 1000.0);
    }

    return stats;
}

// ============================================================================
// 性能统计 - reset_performance_stats()
// ============================================================================

void HeterogeneousComputeEngine::reset_performance_stats()
{
    std::unique_lock<std::shared_mutex> lock(stats_mutex_);
    perf_stats_ = EnginePerformanceStats();  // 重置为默认零值
    engine_start_time_ = std::chrono::system_clock::now();  // 重置启动时间

    LOG_INFO("性能统计已重置");
}

}  // namespace heterogeneous
}  // namespace go
}  // namespace spacego