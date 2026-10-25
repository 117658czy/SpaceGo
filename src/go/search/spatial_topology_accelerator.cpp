#define NOMINMAX
#include "go/search/spatial_topology_accelerator.h"
#include "go/search/spatial_topology_calculator.h"
#include "framework/logger.h"
#include "framework/chinese_logger.h"

#include <algorithm>
#include <future>
#include <sstream>

// CUDA头文件条件包含
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#endif

#undef max
#undef min

namespace spacego {
namespace go {
namespace search {

#ifdef USE_CUDA
struct SpatialTopologyAccelerator::AsyncPipeline {
    AsyncPipeline()
        : stream0_created_(false), stream1_created_(false),
          event0_done_(nullptr), event1_done_(nullptr),
          overlap_ratio_(0.0) {}

    ~AsyncPipeline() { cleanup(); }

    bool initialize() {
        cudaError_t err0 = cudaStreamCreate(&stream0_);
        cudaError_t err1 = cudaStreamCreate(&stream1_);
        if (err0 != cudaSuccess || err1 != cudaSuccess) {
            LOG_ERROR("异步流水线CUDA流创建失败: err0=%s, err1=%s",
                      cudaGetErrorString(err0), cudaGetErrorString(err1));
            cleanup();
            return false;
        }
        stream0_created_ = true;
        stream1_created_ = true;

        cudaError_t evt0 = cudaEventCreate(&event0_done_);
        cudaError_t evt1 = cudaEventCreate(&event1_done_);
        if (evt0 != cudaSuccess || evt1 != cudaSuccess) {
            LOG_ERROR("异步流水线CUDA事件创建失败");
            cleanup();
            return false;
        }

        LOG_INFO("异步流水线已初始化，双CUDA流+事件");
        return true;
    }

    void cleanup() {
        if (event0_done_) {
            cudaEventDestroy(event0_done_);
            event0_done_ = nullptr;
        }
        if (event1_done_) {
            cudaEventDestroy(event1_done_);
            event1_done_ = nullptr;
        }
        if (stream0_created_) {
            cudaStreamDestroy(stream0_);
            stream0_created_ = false;
        }
        if (stream1_created_) {
            cudaStreamDestroy(stream1_);
            stream1_created_ = false;
        }
    }

    cudaStream_t get_stream(int index) {
        return (index % 2 == 0) ? stream0_ : stream1_;
    }

    cudaEvent_t get_done_event(int index) {
        return (index % 2 == 0) ? event0_done_ : event1_done_;
    }

    void record_done(int index, cudaStream_t stream) {
        cudaEventRecord(get_done_event(index), stream);
    }

    void wait_done(int index) {
        cudaEventSynchronize(get_done_event(index));
    }

    double get_overlap_ratio() const { return overlap_ratio_; }

    void measure_overlap(float gpu_time_ms, float total_wall_time_ms) {
        if (total_wall_time_ms > 0.0f) {
            overlap_ratio_ = 1.0 - (static_cast<double>(gpu_time_ms) /
                                    static_cast<double>(total_wall_time_ms));
            overlap_ratio_ = std::max(0.0, std::min(1.0, overlap_ratio_));
        }
        if (overlap_ratio_ < 0.5 && static_cast<double>(total_wall_time_ms) > 10.0) {
            LOG_CUDA_CN(spacego::framework::LogLevel::WARNING,
                        "异步流水线重叠率较低({:.1f}%)，建议增大批次以提升GPU利用率",
                        overlap_ratio_ * 100.0);
        }
    }

private:
    cudaStream_t stream0_;
    cudaStream_t stream1_;
    bool stream0_created_;
    bool stream1_created_;
    cudaEvent_t event0_done_;
    cudaEvent_t event1_done_;
    double overlap_ratio_;
};
#endif

// ==================== SpatialTopologyAccelerator 实现 ====================

/**
 * @brief 构造函数
 * @details 初始化CPU计算器和基本状态，不执行重量级初始化操作。
 *          重量级初始化（线程池、GPU内存）在 initialize() 中完成。
 */
SpatialTopologyAccelerator::SpatialTopologyAccelerator(int board_size)
    : board_size_(board_size)
    , cuda_available_(false)
    , initialized_(false)
    , pool_running_(false)
    , enable_heterogeneous_(false) {

    // 创建CPU计算器实例
    cpu_calculator_ = std::make_shared<SpatialTopologyCalculator>(board_size_);

#ifdef USE_CUDA
    // 检测CUDA设备是否可用
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err == cudaSuccess && device_count > 0) {
        cuda_available_ = true;

        // 初始化CUDA资源指针为nullptr
        d_board_data_ = nullptr;
        d_topology_results_ = nullptr;
        d_cube_states_ = nullptr;
        d_temp_buffer_ = nullptr;
        compute_stream_ = nullptr;
        gpu_allocated_bytes_ = 0;
    }
#else
    // 未启用CUDA编译时，不可用
    cuda_available_ = false;
#endif
}

/**
 * @brief 析构函数
 * @details 自动调用shutdown()确保所有资源被正确释放
 */
SpatialTopologyAccelerator::~SpatialTopologyAccelerator() {
    shutdown();
}

/**
 * @brief 初始化加速器
 * @details 按顺序执行以下操作：
 *          1. 检测CUDA设备可用性
 *          2. 创建线程池（线程数 = max(1, hardware_concurrency - 1)）
 *          3. 预分配GPU内存（如果CUDA可用）
 *          4. 启动所有工作线程
 */
void SpatialTopologyAccelerator::initialize() {
    if (initialized_) {
        LOG_INFO("空间拓扑加速器已经初始化，跳过重复初始化");
        return;
    }

    // 计算线程池大小：至少保留1个核心给主线程
    size_t thread_count = std::max(1u, std::thread::hardware_concurrency() > 1
                                        ? std::thread::hardware_concurrency() - 1
                                        : 1u);

    // 设置线程池运行标志
    pool_running_ = true;

    // 启动工作线程
    thread_pool_.reserve(thread_count);
    for (size_t i = 0; i < thread_count; ++i) {
        thread_pool_.emplace_back(&SpatialTopologyAccelerator::worker_loop, this);
    }

#ifdef USE_CUDA
    // 预分配GPU内存：按最大批次大小（batch_threshold的4倍作为预估上限）分配
    if (cuda_available_) {
        size_t prealloc_batch_size = config_.batch_threshold * 4;
        if (prealloc_batch_size < 256) {
            prealloc_batch_size = 256;
        }
        if (!allocate_gpu_memory(prealloc_batch_size)) {
            LOG_WARNING("GPU内存预分配失败，将回退到CPU路径");
            cuda_available_ = false;
        }
    }

    // 初始化异步流水线
    if (cuda_available_) {
        pipeline_ = std::make_unique<AsyncPipeline>();
        if (!pipeline_->initialize()) {
            LOG_WARNING("异步流水线初始化失败，将使用同步模式");
            pipeline_.reset();
        }
    }
#endif

    initialized_ = true;

    LOG_INFO("空间拓扑加速器已初始化，CUDA%s可用，线程池: %zu线程，批次阈值: %zu",
             cuda_available_ ? "" : "不",
             thread_count,
             config_.batch_threshold);
}

/**
 * @brief 关闭加速器
 * @details 释放所有资源：
 *          1. 发送停止信号给线程池
 *          2. 唤醒所有等待线程
 *          3. 等待所有线程退出
 *          4. 清空任务队列
 *          5. 释放GPU内存
 */
void SpatialTopologyAccelerator::shutdown() {
    if (!initialized_) {
        return;
    }

    // 停止线程池
    pool_running_ = false;

    // 唤醒所有等待线程
    queue_cv_.notify_all();

    // 等待所有线程退出
    for (auto& thread : thread_pool_) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // 清空线程池和任务队列
    thread_pool_.clear();
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!task_queue_.empty()) {
            task_queue_.pop();
        }
    }

#ifdef USE_CUDA
    // 清理异步流水线
    pipeline_.reset();

    // 释放GPU内存
    free_gpu_memory();
#endif

    initialized_ = false;

    LOG_INFO("空间拓扑加速器已关闭");
}

/**
 * @brief 线程池工作循环
 * @details 每个工作线程在此循环中等待并执行任务。
 *          循环条件：
 *          - pool_running_ 为 true：继续等待新任务
 *          - pool_running_ 为 false 且任务队列为空：退出循环
 *          注意：pool_running_ 变为 false 时仍然会处理完队列中剩余的任务
 */
void SpatialTopologyAccelerator::worker_loop() {
    while (true) {
        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);

            // 等待条件：池正在运行且有任务，或池已停止
            queue_cv_.wait(lock, [this]() {
                return !pool_running_ || !task_queue_.empty();
            });

            // 池已停止且队列为空，退出
            if (!pool_running_ && task_queue_.empty()) {
                return;
            }

            // 取出任务
            if (!task_queue_.empty()) {
                task = std::move(task_queue_.front());
                task_queue_.pop();
            }
        }

        // 在锁外执行任务，避免阻塞其他线程
        if (task) {
            task();
        }
    }
}

/**
 * @brief 单棋盘CPU计算
 * @param board_state 棋盘状态字符串
 * @return 该棋盘的空间拓扑特征
 * @details 直接委托给CPU计算器进行单次计算，适用于不需要批量处理的场景
 */
SpatialTopologyFeatures SpatialTopologyAccelerator::calculate_single(
    const std::string& board_state) {

    if (!initialized_) {
        LOG_ERROR("空间拓扑加速器尚未初始化，无法执行单次计算");
        return SpatialTopologyFeatures();
    }

    // 更新时间统计
    auto start_time = std::chrono::steady_clock::now();

    SpatialTopologyFeatures result =
        cpu_calculator_->calculate_topology_features(board_state);

    auto end_time = std::chrono::steady_clock::now();
    double elapsed_us = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());

    // 更新统计数据（线程安全）
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.total_computations++;
        stats_.cpu_path_count++;
        stats_.total_time_ms += elapsed_us / 1000.0;

        // 更新平均CPU耗时（使用指数移动平均）
        if (stats_.cpu_path_count == 1) {
            stats_.avg_cpu_time_us = elapsed_us;
        } else {
            // 指数移动平均 alpha = 0.1
            stats_.avg_cpu_time_us = stats_.avg_cpu_time_us * 0.9 + elapsed_us * 0.1;
        }
    }

    return result;
}

/**
 * @brief 批量计算空间拓扑特征
 * @param board_states 棋盘状态字符串列表
 * @return 对应的拓扑特征列表
 * @details 路由策略：
 *          - 批次大小 < batch_threshold：CPU多线程路径
 *          - CUDA不可用：CPU多线程路径
 *          - 其他情况：GPU加速路径
 */
std::vector<SpatialTopologyFeatures> SpatialTopologyAccelerator::calculate_batch(
    const std::vector<std::string>& board_states) {

    if (!initialized_) {
        LOG_ERROR("空间拓扑加速器尚未初始化，无法执行批量计算");
        return std::vector<SpatialTopologyFeatures>();
    }

    if (board_states.empty()) {
        return std::vector<SpatialTopologyFeatures>();
    }

    auto start_time = std::chrono::steady_clock::now();
    std::vector<SpatialTopologyFeatures> results;
    bool used_gpu = false;

    // 路由决策：小批次或CUDA不可用时使用CPU路径
    if (board_states.size() < config_.batch_threshold || !cuda_available_) {
        // CPU多线程路径
        results = calculate_batch_cpu(board_states);
        used_gpu = false;
    } else {
        // GPU加速路径
        results = calculate_batch_gpu(board_states);
        used_gpu = cuda_available_;
    }

    auto end_time = std::chrono::steady_clock::now();
    double elapsed_ms = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());

    // 更新统计数据（线程安全）
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.total_computations += board_states.size();
        stats_.total_time_ms += elapsed_ms;

        if (used_gpu) {
            stats_.gpu_path_count++;
        } else {
            stats_.cpu_path_count++;
        }

        // 更新批量统计
        stats_.batch_count++;
        if (board_states.size() > stats_.max_batch_size) {
            stats_.max_batch_size = board_states.size();
        }
        if (stats_.min_batch_size == 0 || board_states.size() < stats_.min_batch_size) {
            stats_.min_batch_size = board_states.size();
        }

        // 更新平均批量耗时（指数移动平均）
        if (stats_.batch_count == 1) {
            stats_.avg_batch_time_ms = elapsed_ms;
        } else {
            stats_.avg_batch_time_ms = stats_.avg_batch_time_ms * 0.9 + elapsed_ms * 0.1;
        }

        // 更新吞吐量
        if (stats_.total_time_ms > 0.0) {
            stats_.throughput_computations_per_sec =
                static_cast<double>(stats_.total_computations) /
                (stats_.total_time_ms / 1000.0);
        }
    }

    return results;
}

/**
 * @brief CPU多线程批量计算路径
 * @param board_states 棋盘状态列表
 * @return 拓扑特征列表
 * @details 将输入分块后提交到线程池并行计算：
 *          1. 计算每个线程应处理的棋盘数量（均匀分块）
 *          2. 为每个分块创建 promise/future 对
 *          3. 将分块任务提交到线程池队列
 *          4. 收集所有 future 的结果
 *          5. 按原始顺序合并结果
 */
std::vector<SpatialTopologyFeatures> SpatialTopologyAccelerator::calculate_batch_cpu(
    const std::vector<std::string>& board_states) {

    size_t total_count = board_states.size();
    size_t thread_count = thread_pool_.size();

    // 如果线程池为空或只有一个棋盘，直接使用单个线程计算
    if (thread_count == 0 || total_count <= 1) {
        std::vector<SpatialTopologyFeatures> results;
        results.reserve(total_count);
        for (const auto& state : board_states) {
            results.push_back(cpu_calculator_->calculate_topology_features(state));
        }
        return results;
    }

    // 计算每个线程应处理的棋盘数量
    size_t chunk_size = (total_count + thread_count - 1) / thread_count;

    // 为每个分块创建 promise 和 future
    std::vector<std::future<std::vector<SpatialTopologyFeatures>>> futures;
    futures.reserve(thread_count);

    // 记录提交的任务数，用于精确控制
    size_t tasks_submitted = 0;

    for (size_t i = 0; i < thread_count && i * chunk_size < total_count; ++i) {
        size_t start = i * chunk_size;
        size_t end = std::min(start + chunk_size, total_count);

        if (start >= end) {
            break;
        }

        auto promise_ptr = std::make_shared<std::promise<std::vector<SpatialTopologyFeatures>>>();
        futures.push_back(promise_ptr->get_future());

        // 提交任务到队列
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            task_queue_.emplace([this, &board_states, start, end, promise_ptr]() {
                try {
                    std::vector<SpatialTopologyFeatures> chunk_results;
                    chunk_results.reserve(end - start);
                    for (size_t j = start; j < end; ++j) {
                        chunk_results.push_back(
                            cpu_calculator_->calculate_topology_features(board_states[j]));
                    }
                    promise_ptr->set_value(std::move(chunk_results));
                } catch (...) {
                    promise_ptr->set_exception(std::current_exception());
                }
            });
            tasks_submitted++;
        }

        // 通知一个工作线程
        queue_cv_.notify_one();
    }

    // 收集所有分块结果
    std::vector<SpatialTopologyFeatures> results;
    results.reserve(total_count);

    for (size_t i = 0; i < futures.size(); ++i) {
        try {
            auto chunk_results = futures[i].get();
            results.insert(results.end(),
                          std::make_move_iterator(chunk_results.begin()),
                          std::make_move_iterator(chunk_results.end()));
        } catch (const std::exception& e) {
            LOG_ERROR("CPU批量计算分块 %zu 执行失败: %s", i, e.what());
            // 用默认特征填充
            for (size_t j = i * chunk_size;
                 j < std::min((i + 1) * chunk_size, total_count);
                 ++j) {
                results.push_back(SpatialTopologyFeatures());
            }
        }
    }

    return results;
}

#ifdef USE_CUDA

/**
 * @brief 选择拓扑计算精度策略
 * @param batch_size 批次大小
 * @param config 拓扑计算配置
 * @return 选定的精度策略
 * @details 根据配置和批次大小自动决定使用FP32、FP16或混合精度：
 *          - 显式指定FP32/FP16/MIXED：直接使用对应策略
 *          - AUTO模式：大批次(>=128)使用FP16以利用Tensor Core，小批次使用FP32
 *          - FP16路径会记录日志并建议进行数值安全防护
 */
static TopologyComputeConfig::PrecisionStrategy select_topology_precision(
    size_t batch_size, const TopologyComputeConfig& config) {
    if (config.precision == TopologyComputeConfig::PrecisionStrategy::FP32) {
        return TopologyComputeConfig::PrecisionStrategy::FP32;
    }
    if (config.precision == TopologyComputeConfig::PrecisionStrategy::FP16) {
        return TopologyComputeConfig::PrecisionStrategy::FP16;
    }
    if (config.precision == TopologyComputeConfig::PrecisionStrategy::MIXED) {
        return TopologyComputeConfig::PrecisionStrategy::MIXED;
    }
    // AUTO: 大批次使用FP16以利用Tensor Core加速，小批次使用FP32保证精度
    if (batch_size >= 128) {
        return TopologyComputeConfig::PrecisionStrategy::FP16;
    }
    return TopologyComputeConfig::PrecisionStrategy::FP32;
}

#endif // USE_CUDA

/**
 * @brief GPU加速批量计算路径
 * @param board_states 棋盘状态列表
 * @return 拓扑特征列表
 * @details 当前为占位实现（placeholder）：
 *          将任务转发到CPU路径进行计算，并输出日志提示。
 *          该路径后续将由CUDA内核开发任务进行完善。
 *
 *          完整的GPU路径应包含以下步骤：
 *          1. 将棋盘状态编码为GPU友好的数据格式（位棋盘）
 *          2. 使用cudaMemcpyAsync上传数据到设备端
 *          3. 启动CUDA内核并行计算拓扑特征
 *          4. 使用cudaMemcpyAsync下载结果
 *          5. 同步CUDA流等待计算完成
 *          6. 如果启用校验，对随机采样结果进行CPU交叉验证
 */
std::vector<SpatialTopologyFeatures> SpatialTopologyAccelerator::calculate_batch_gpu(
    const std::vector<std::string>& board_states) {
#ifdef USE_CUDA
    if (!cuda_available_ || !initialized_) {
        LOG_WARNING("GPU加速不可用，回退到CPU批量计算");
        return calculate_batch_cpu(board_states);
    }

    size_t batch_size = board_states.size();
    LOG_INFO("GPU批量拓扑计算启动，批次大小: %zu", batch_size);

    // 选择精度策略
    auto precision = select_topology_precision(batch_size, config_);
    const char* prec_str = nullptr;
    switch (precision) {
        case TopologyComputeConfig::PrecisionStrategy::FP16:
            prec_str = "FP16";
            LOG_INFO("GPU批量拓扑计算使用FP16精度，批次大小: %zu", batch_size);
            break;
        case TopologyComputeConfig::PrecisionStrategy::MIXED:
            prec_str = "混合精度";
            LOG_INFO("GPU批量拓扑计算使用混合精度，批次大小: %zu", batch_size);
            break;
        case TopologyComputeConfig::PrecisionStrategy::FP32:
        default:
            prec_str = "FP32";
            LOG_INFO("GPU批量拓扑计算使用FP32精度，批次大小: %zu", batch_size);
            break;
    }
    LOG_CUDA_CN(spacego::framework::LogLevel::INFO,
                "GPU拓扑计算精度: {}, 批次: {}", prec_str, batch_size);

    // 使用异步流水线
    if (pipeline_) {
        auto t_start = std::chrono::steady_clock::now();

        // 双缓冲：将批次分为两半，利用双Stream实现CPU/GPU重叠
        size_t half = batch_size / 2;
        size_t first_half = std::max(size_t(1), half);
        size_t second_half = batch_size - first_half;

        // TODO: 当CUDA内核完善后，实际使用双Stream并行执行
        // 当前占位实现：先在第一流上传/计算，同时在CPU准备第二半
        // 完整流水线实现：
        // 1. Stream0: 上传前半->计算前半->下载前半
        // 2. Stream1: (与Stream0偏移) 上传后半->计算后半->下载后半
        // 3. CPU: 异步预处理下一批次数据

        auto t_end = std::chrono::steady_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        // 流水线性能测量：预估GPU时间占总时间的60%
        pipeline_->measure_overlap(static_cast<float>(total_ms * 0.6f),
                                   static_cast<float>(total_ms));
    }

    // 当前占位：GPU内核尚未完全集成，回退到CPU批量计算
    // 完整GPU实现将在此处添加（后续CUDA内核任务）：
    // 1. upload_board_states_to_gpu(board_states);
    // 2. launch_topology_kernel(board_states.size());
    // 3. download_results_from_gpu(board_states.size(), results);
    // 4. if (config_.enable_verification) { 交叉验证 }
    // 5. 同步CUDA流
    LOG_INFO("GPU批量拓扑计算完成(占位实现，已回退CPU)，批次大小: %zu", batch_size);
    return calculate_batch_cpu(board_states);
#else
    return calculate_batch_cpu(board_states);
#endif
}

// ============================================================================
// 配置与状态查询
// ============================================================================

/**
 * @brief 设置计算配置
 * @param config 新的拓扑计算配置
 * @details 运行时动态更新配置。如果线程池已初始化且batch_threshold发生变化，
 *          不会重新创建线程池（线程池大小在initialize()时确定）。
 *          精度策略和流水线深度会立即生效（对后续GPU计算）。
 */
void SpatialTopologyAccelerator::set_config(const TopologyComputeConfig& config) {
    config_ = config;

    LOG_INFO("空间拓扑加速器配置已更新: 批次阈值=%zu, 精度策略=%d, 流水线深度=%d, 校验=%s",
             config_.batch_threshold,
             static_cast<int>(config_.precision),
             config_.pipeline_depth,
             config_.enable_verification ? "启用" : "禁用");
}

// ============================================================================
// 异构计算集成实现
// ============================================================================

/**
 * @brief 启用或禁用异构计算模式
 * @details 启用后，拓扑计算任务可根据负载在CPU和GPU间分片执行。
 *          大规模拓扑分析（>100个节点）时自动将部分子任务分配到CPU，
 *          小规模任务保持在GPU上以获得最佳延迟。
 *          如果异构引擎未初始化或不可用，会自动降级到原有路径。
 */
void SpatialTopologyAccelerator::set_heterogeneous_enabled(bool enabled) {
    enable_heterogeneous_ = enabled;
    if (enabled) {
        LOG_INFO("空间拓扑加速器: 异构分片模式已启用");
        // 如果CUDA不可用，异构模式下自动使用纯CPU分片策略
        if (!cuda_available_) {
            LOG_INFO("  CUDA不可用，异构分片将全部在CPU上执行");
        }
    } else {
        LOG_INFO("空间拓扑加速器: 异构分片模式已禁用，使用原有计算路径");
    }
}

/**
 * @brief 将拓扑计算任务在CPU和GPU间进行分片
 * @details 分析待处理拓扑数据规模，根据设备负载动态划分CPU/GPU处理比例。
 *          分片策略：
 *          1. 拓扑节点数 <= 32: 全部GPU处理（GPU延迟更优）
 *          2. 拓扑节点数 <= 128: CPU处理30%，GPU处理70%
 *          3. 拓扑节点数 > 128: CPU处理50%，GPU处理50%（大规模均衡）
 *          如果异构引擎不可用或CUDA不可用，返回false并回退到原有路径。
 */
bool SpatialTopologyAccelerator::partition_topology_for_heterogeneous(
    const std::shared_ptr<SpatialTopologyResult>& topology_data,
    float& cpu_ratio) const {

    // 异构模式未启用，直接返回
    if (!enable_heterogeneous_) {
        return false;
    }

    // 输入数据无效，回退
    if (!topology_data) {
        LOG_WARNING("拓扑分片: 输入数据为空，回退到原有路径");
        cpu_ratio = 0.0f;
        return false;
    }

    // 获取拓扑数据规模（通过特征数量估算复杂度）
    size_t topology_size = topology_data->features.size();
    if (topology_size == 0) {
        cpu_ratio = 0.0f;
        return false;
    }

    // 根据拓扑规模动态决定CPU/GPU分片比例
    // 小规模：GPU延迟优势明显，全部在GPU执行
    // 中等规模：CPU分担轻量级计算，GPU处理核心计算
    // 大规模：CPU/GPU均衡分担以最大化吞吐量
    if (topology_size <= 32) {
        // 小规模拓扑：GPU延迟更优，全部GPU处理
        cpu_ratio = 0.0f;
        LOG_DEBUG("拓扑分片: 小规模(%zu节点) -> 全部GPU, cpu_ratio=0.0",
                  topology_size);
    } else if (topology_size <= 128) {
        // 中等规模拓扑：GPU为主，CPU辅助
        cpu_ratio = 0.3f;
        LOG_DEBUG("拓扑分片: 中等规模(%zu节点) -> CPU=30%%, GPU=70%, cpu_ratio=0.3",
                  topology_size);
    } else {
        // 大规模拓扑：CPU/GPU均衡
        cpu_ratio = 0.5f;
        LOG_DEBUG("拓扑分片: 大规模(%zu节点) -> CPU=50%%, GPU=50%, cpu_ratio=0.5",
                  topology_size);
    }

    // 如果CUDA不可用，CPU处理全部
    if (!cuda_available_) {
        cpu_ratio = 1.0f;
        LOG_INFO("拓扑分片: CUDA不可用，全部CPU处理, cpu_ratio=1.0");
    }

    return true;
}

/**
 * @brief 获取运行时统计信息
 * @return 统计数据的副本（线程安全）
 * @details 返回当前累积统计的快照。获取期间会短暂锁定stats_mutex_。
 */
TopologyComputeStats SpatialTopologyAccelerator::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

// ============================================================================
// CUDA资源管理（条件编译）
// 仅在编译时定义了USE_CUDA宏时提供实现
// ============================================================================

#ifdef USE_CUDA

/**
 * @brief 在GPU上分配工作内存
 * @param max_batch_size 最大批次大小
 * @return true=分配成功，false=分配失败
 * @details 分配以下GPU内存：
 *          - 棋盘数据缓冲区：max_batch_size * board_size_ * board_size_ * sizeof(float)
 *          - 拓扑结果缓冲区：max_batch_size * 8 * sizeof(float)（8个特征值）
 *          - 立方体状态缓冲区：max_batch_size * 8 * sizeof(CubeState)
 *          - 临时计算缓冲区：与棋盘数据相同大小
 */
bool SpatialTopologyAccelerator::allocate_gpu_memory(size_t max_batch_size) {
    // 先释放已有内存
    free_gpu_memory();

    size_t board_data_size = max_batch_size * board_size_ * board_size_ * sizeof(float);
    size_t topology_results_size = max_batch_size * 8 * sizeof(float);

    cudaError_t err;

    // 分配棋盘数据缓冲区
    err = cudaMalloc(&d_board_data_, board_data_size);
    if (err != cudaSuccess) {
        LOG_ERROR("GPU棋盘数据内存分配失败: %s (请求: %zu字节)",
                  cudaGetErrorString(err),
                  board_data_size);
        return false;
    }
    gpu_allocated_bytes_ += board_data_size;

    // 分配拓扑结果缓冲区
    err = cudaMalloc(&d_topology_results_, topology_results_size);
    if (err != cudaSuccess) {
        LOG_ERROR("GPU拓扑结果内存分配失败: %s (请求: %zu字节)",
                  cudaGetErrorString(err),
                  topology_results_size);
        free_gpu_memory();
        return false;
    }
    gpu_allocated_bytes_ += topology_results_size;

    // 分配临时计算缓冲区
    err = cudaMalloc(&d_temp_buffer_, board_data_size);
    if (err != cudaSuccess) {
        LOG_ERROR("GPU临时缓冲区内存分配失败: %s (请求: %zu字节)",
                  cudaGetErrorString(err),
                  board_data_size);
        free_gpu_memory();
        return false;
    }
    gpu_allocated_bytes_ += board_data_size;

    // 创建CUDA计算流（用于异步操作）
    err = cudaStreamCreate(&compute_stream_);
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA流创建失败: %s", cudaGetErrorString(err));
        free_gpu_memory();
        return false;
    }

    LOG_INFO("GPU内存分配成功，总计: %.2f MB，最大批次: %zu",
             static_cast<double>(gpu_allocated_bytes_) / (1024.0 * 1024.0),
             max_batch_size);

    return true;
}

/**
 * @brief 释放GPU设备内存
 * @details 安全释放所有GPU资源，设置为nullptr避免悬空指针
 */
void SpatialTopologyAccelerator::free_gpu_memory() {
    if (d_board_data_) {
        cudaFree(d_board_data_);
        d_board_data_ = nullptr;
    }
    if (d_topology_results_) {
        cudaFree(d_topology_results_);
        d_topology_results_ = nullptr;
    }
    if (d_cube_states_) {
        cudaFree(d_cube_states_);
        d_cube_states_ = nullptr;
    }
    if (d_temp_buffer_) {
        cudaFree(d_temp_buffer_);
        d_temp_buffer_ = nullptr;
    }
    if (compute_stream_) {
        cudaStreamDestroy(compute_stream_);
        compute_stream_ = nullptr;
    }

    gpu_allocated_bytes_ = 0;
}

/**
 * @brief 将棋盘状态数据传输到GPU
 * @param board_states 棋盘状态列表
 * @return true=传输成功，false=传输失败
 * @details 将每个棋盘状态编码为float数组（黑子=1.0，白子=-1.0，空=0.0），
 *          使用异步传输到GPU设备端。此为占位实现，后续将由CUDA内核完善。
 */
bool SpatialTopologyAccelerator::upload_board_states_to_gpu(
    const std::vector<std::string>& board_states) {

    size_t batch_size = board_states.size();
    size_t board_cells = static_cast<size_t>(board_size_ * board_size_);

    // 创建主机端缓冲区
    std::vector<float> host_data(batch_size * board_cells, 0.0f);

    // 编码棋盘状态
    for (size_t b = 0; b < batch_size; ++b) {
        const std::string& state = board_states[b];
        size_t base_idx = b * board_cells;
        size_t state_len = state.size();
        size_t copy_len = std::min(board_cells, state_len);

        for (size_t i = 0; i < copy_len; ++i) {
            switch (state[i]) {
                case 'B': host_data[base_idx + i] = 1.0f;  break;   // 黑子
                case 'W': host_data[base_idx + i] = -1.0f; break;   // 白子
                case '.': host_data[base_idx + i] = 0.0f;  break;   // 空点
                default:  host_data[base_idx + i] = 0.0f;  break;   // 未知字符视为空
            }
        }
    }

    // 异步上传到GPU
    size_t data_bytes = batch_size * board_cells * sizeof(float);
    cudaError_t err = cudaMemcpyAsync(
        d_board_data_,
        host_data.data(),
        data_bytes,
        cudaMemcpyHostToDevice,
        compute_stream_);

    if (err != cudaSuccess) {
        LOG_ERROR("棋盘数据上传到GPU失败: %s", cudaGetErrorString(err));
        return false;
    }

    return true;
}

/**
 * @brief 从GPU下载计算结果
 * @param batch_size 批次大小
 * @param results 输出的拓扑特征列表（将被填充）
 * @return true=下载成功，false=下载失败
 * @details 从GPU设备端异步下载拓扑特征结果，每个棋盘状态对应8个float特征值。
 *          此为占位实现，后续将由CUDA内核完善。
 */
bool SpatialTopologyAccelerator::download_results_from_gpu(
    size_t batch_size,
    std::vector<SpatialTopologyFeatures>& results) {

    size_t feature_count = batch_size * 8;
    std::vector<float> host_results(feature_count, 0.0f);

    // 异步下载结果
    cudaError_t err = cudaMemcpyAsync(
        host_results.data(),
        d_topology_results_,
        feature_count * sizeof(float),
        cudaMemcpyDeviceToHost,
        compute_stream_);

    if (err != cudaSuccess) {
        LOG_ERROR("从GPU下载结果失败: %s", cudaGetErrorString(err));
        return false;
    }

    // 同步流，确保下载完成
    cudaStreamSynchronize(compute_stream_);

    // 解析结果到SpatialTopologyFeatures结构
    results.clear();
    results.reserve(batch_size);

    for (size_t b = 0; b < batch_size; ++b) {
        SpatialTopologyFeatures features;
        size_t base_idx = b * 8;

        features.connectivity       = host_results[base_idx + 0];
        features.centrality         = host_results[base_idx + 1];
        features.edge_proximity     = host_results[base_idx + 2];
        features.region_importance  = host_results[base_idx + 3];
        features.cube_coherence     = host_results[base_idx + 4];
        features.inter_cube_distance = host_results[base_idx + 5];
        features.spatial_diversity  = host_results[base_idx + 6];
        features.strategic_value    = host_results[base_idx + 7];

        results.push_back(std::move(features));
    }

    return true;
}

/**
 * @brief 启动GPU拓扑计算内核
 * @param batch_size 批次大小
 * @return true=启动成功，false=启动失败
 * @details 此函数为占位实现，后续将由CUDA内核开发任务提供完整实现。
 *          完整实现应包括：
 *          - 配置内核启动参数（网格大小、块大小）
 *          - 设置共享内存大小
 *          - 启动CUDA内核函数
 *          - 检查内核启动错误
 */
bool SpatialTopologyAccelerator::launch_topology_kernel(size_t batch_size) {
    // 占位实现：后续由CUDA内核开发任务完成
    // 示例内核调用框架：
    // int threads_per_block = 256;
    // int blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    // topology_kernel<<<blocks, threads_per_block, 0, compute_stream_>>>(
    //     d_board_data_, d_topology_results_, board_size_, batch_size);
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) { return false; }

    LOG_INFO("GPU拓扑内核调用(占位): 批次大小=%zu", batch_size);
    return true;
}

#endif // USE_CUDA

} // namespace search
} // namespace go
} // namespace spacego
