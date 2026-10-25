// ============================================================================
// 异构计算引擎 - 设备负载均衡器 实现
// 命名空间: spacego::go::heterogeneous
// 功能: DeviceLoadBalancer类的完整实现
//
// 包含以下核心模块的实现：
//   1. CPU利用率计算 - Windows GetSystemTimes / Linux /proc/stat
//   2. GPU利用率计算 - NVML API（CUDA环境）
//   3. EMA（指数移动平均）平滑算法
//   4. 动态任务分配比例调整
//   5. 设备过载检测与任务分流/回流
//   6. 负载历史记录管理
// ============================================================================

#include "go/heterogeneous/device_load_balancer.h"
#include "framework/logger.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <string>

// ---- Windows 平台专用头文件 ----
#ifdef _WIN32
// 注意：device_load_balancer.h 已经包含了 windows.h，
// 但这里需要一些额外的Windows API声明
// GlobalMemoryStatusEx 已经在 windows.h 中声明
#endif

// ---- Linux 平台专用头文件 ----
#ifndef _WIN32
#include <unistd.h>
#include <sys/sysinfo.h>
#include <fstream>
#endif

// ---- CUDA 环境专用头文件 ----
#if SPACEGO_CUDA_ENABLED
#include <cuda_runtime.h>
// NVML（NVIDIA Management Library）用于GPU利用率监控
// NVML随NVIDIA驱动安装，头文件通常位于CUDA Toolkit的include目录
#include <nvml.h>
#endif

namespace spacego {
namespace go {
namespace heterogeneous {

// ============================================================================
// 辅助函数：Windows FILETIME 转 uint64_t（100纳秒单位）
// ============================================================================

#ifdef _WIN32
/**
 * @brief 将Windows FILETIME结构体转换为uint64_t
 *
 * FILETIME是一个64位值，表示自1601年1月1日以来的100纳秒间隔数。
 * 使用ULARGE_INTEGER进行安全的位转换。
 *
 * @param ft Windows FILETIME结构体
 * @return 转换后的uint64_t值（单位：100纳秒）
 */
static uint64_t filetime_to_uint64(const FILETIME& ft) {
    ULARGE_INTEGER uli;
    uli.LowPart  = ft.dwLowDateTime;
    uli.HighPart = ft.dwHighDateTime;
    return uli.QuadPart;
}
#endif

// ============================================================================
// 构造与析构
// ============================================================================

/**
 * @brief 默认构造函数
 *
 * 使用默认的HeterogeneousComputeConfig配置初始化负载均衡器。
 * 所有EMA值初始化为0（假设系统初始为空闲状态）。
 * CPU/GPU分配比例初始化为均衡的0.5/0.5。
 */
DeviceLoadBalancer::DeviceLoadBalancer()
    : config_(HeterogeneousComputeConfig())
    , cpu_utilization_ema_(0.0)
    , cpu_allocation_ema_(0.5)
    , gpu_utilization_ema_(0.0)
    , gpu_allocation_ema_(0.5)
    , current_cpu_ratio_(0.5)
    , current_gpu_ratio_(0.5)
    , cpu_queue_depth_(0)
    , gpu_queue_depth_(0)
    , consecutive_gpu_overload_count_(0)
    , consecutive_cpu_overload_count_(0)
    , cpu_load_history_()
    , gpu_load_history_()
    , last_monitor_time_(std::chrono::system_clock::now())
#ifdef _WIN32
    , prev_idle_time_(0)
    , prev_kernel_time_(0)
    , prev_user_time_(0)
    , cpu_sample_valid_(false)
#else
    , prev_total_time_(0)
    , prev_idle_time_(0)
    , cpu_sample_valid_(false)
#endif
#if SPACEGO_CUDA_ENABLED
    , nvml_initialized_(false)
    , nvml_device_handle_(nullptr)
#endif
{
    // 预留历史记录空间，避免运行时频繁分配
    cpu_load_history_.reserve(MAX_LOAD_HISTORY);
    gpu_load_history_.reserve(MAX_LOAD_HISTORY);

    LOG_INFO("设备负载均衡器初始化完成（默认配置）");
}

/**
 * @brief 带配置的构造函数
 *
 * 使用指定的HeterogeneousComputeConfig配置初始化。
 * 允许调用方自定义所有负载均衡参数。
 *
 * @param config 异构计算配置参数
 */
DeviceLoadBalancer::DeviceLoadBalancer(const HeterogeneousComputeConfig& config)
    : config_(config)
    , cpu_utilization_ema_(0.0)
    , cpu_allocation_ema_(0.5)
    , gpu_utilization_ema_(0.0)
    , gpu_allocation_ema_(0.5)
    , current_cpu_ratio_(0.5)
    , current_gpu_ratio_(0.5)
    , cpu_queue_depth_(0)
    , gpu_queue_depth_(0)
    , consecutive_gpu_overload_count_(0)
    , consecutive_cpu_overload_count_(0)
    , cpu_load_history_()
    , gpu_load_history_()
    , last_monitor_time_(std::chrono::system_clock::now())
#ifdef _WIN32
    , prev_idle_time_(0)
    , prev_kernel_time_(0)
    , prev_user_time_(0)
    , cpu_sample_valid_(false)
#else
    , prev_total_time_(0)
    , prev_idle_time_(0)
    , cpu_sample_valid_(false)
#endif
#if SPACEGO_CUDA_ENABLED
    , nvml_initialized_(false)
    , nvml_device_handle_(nullptr)
#endif
{
    cpu_load_history_.reserve(MAX_LOAD_HISTORY);
    gpu_load_history_.reserve(MAX_LOAD_HISTORY);

    LOG_INFO("设备负载均衡器初始化完成（自定义配置），GPU过载阈值="
             + std::to_string(config_.gpu_overload_threshold)
             + "，CPU过载阈值=" + std::to_string(config_.cpu_overload_threshold));
}

/**
 * @brief 析构函数
 *
 * 清理NVML资源（如果已初始化）。
 * 注意：nvmlShutdown是进程级别的操作，如果有多个DeviceLoadBalancer实例，
 * 最后一个析构的实例会关闭NVML。实际使用中通常只有一个实例。
 */
DeviceLoadBalancer::~DeviceLoadBalancer() {
#if SPACEGO_CUDA_ENABLED
    if (nvml_initialized_) {
        // NVML关闭是进程级操作，记录日志但不做严格的错误检查
        nvmlReturn_t result = nvmlShutdown();
        if (result == NVML_SUCCESS) {
            LOG_INFO("NVML已成功关闭");
        } else {
            LOG_WARNING("NVML关闭时出现警告（可能已被其他组件关闭）");
        }
        nvml_initialized_ = false;
        nvml_device_handle_ = nullptr;
    }
#endif
    LOG_INFO("设备负载均衡器已销毁");
}

// ============================================================================
// 设备负载快照获取
// ============================================================================

/**
 * @brief 获取CPU当前负载快照
 *
 * 执行流程：
 *   1. 调用 calculate_cpu_utilization() 获取CPU利用率
 *   2. 调用 get_available_system_memory_mb() 获取可用内存
 *   3. 读取当前CPU任务队列深度（原子读取）
 *   4. 温度字段置为0.0（CPU温度不通过此接口获取）
 *   5. 记录时间戳
 *
 * @return DeviceLoadSnapshot CPU当前负载快照
 */
DeviceLoadSnapshot DeviceLoadBalancer::get_cpu_load() {
    DeviceLoadSnapshot snapshot;
    snapshot.device_type = ComputeDeviceType::CPU;

    // 1. 获取CPU利用率（首次调用返回0.0，因为需要历史差值）
    snapshot.utilization = calculate_cpu_utilization();

    // 2. 获取系统可用内存
    snapshot.available_memory_mb = get_available_system_memory_mb();

    // 3. 读取任务队列深度（原子操作，线程安全）
    snapshot.queue_depth = cpu_queue_depth_.load(std::memory_order_acquire);

    // 4. CPU温度不通过此接口获取，设为0.0
    snapshot.temperature_celsius = 0.0;

    // 5. 设置采样时间戳
    snapshot.timestamp = std::chrono::system_clock::now();

    // 记录到历史列表（线程安全）
    {
        std::lock_guard<std::mutex> lock(history_mutex_);
        record_load_snapshot(snapshot, cpu_load_history_);
    }

    // 更新最后监控时间
    last_monitor_time_ = snapshot.timestamp;

    return snapshot;
}

#if SPACEGO_CUDA_ENABLED
/**
 * @brief 获取GPU当前负载快照（CUDA环境）
 *
 * 执行流程：
 *   1. 确保NVML已初始化（延迟初始化）
 *   2. 调用 calculate_gpu_utilization() 获取GPU利用率
 *   3. 通过 cudaMemGetInfo() 获取可用显存
 *   4. 通过NVML获取GPU温度
 *   5. 读取当前GPU任务队列深度（原子读取）
 *
 * @param device_id CUDA设备索引（默认0）
 * @return DeviceLoadSnapshot GPU当前负载快照
 */
DeviceLoadSnapshot DeviceLoadBalancer::get_gpu_load(int device_id) {
    DeviceLoadSnapshot snapshot;
    snapshot.device_type = ComputeDeviceType::GPU;

    // 1. 计算GPU利用率（通过NVML或启发式估算）
    snapshot.utilization = calculate_gpu_utilization(device_id);

    // 2. 获取GPU可用显存（通过CUDA Runtime API）
    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaError_t mem_err = cudaMemGetInfo(&free_mem, &total_mem);
    if (mem_err == cudaSuccess) {
        snapshot.available_memory_mb = free_mem / (1024 * 1024);
    } else {
        // CUDA API调用失败，使用0表示不可用
        snapshot.available_memory_mb = 0;
        LOG_WARNING("无法获取GPU显存信息（CUDA错误：" 
                    + std::string(cudaGetErrorString(mem_err)) + "）");
    }

    // 3. 读取任务队列深度（原子操作，线程安全）
    snapshot.queue_depth = gpu_queue_depth_.load(std::memory_order_acquire);

    // 4. 获取GPU温度（通过NVML）
    snapshot.temperature_celsius = 0.0;
    if (nvml_initialized_ && nvml_device_handle_ != nullptr) {
        unsigned int temperature = 0;
        nvmlReturn_t temp_result = nvmlDeviceGetTemperature(
            static_cast<nvmlDevice_t>(nvml_device_handle_),
            NVML_TEMPERATURE_GPU,
            &temperature
        );
        if (temp_result == NVML_SUCCESS) {
            snapshot.temperature_celsius = static_cast<double>(temperature);
        }
        // NVML温度查询失败时不记录错误，温度保持0.0
    }

    // 5. 设置采样时间戳
    snapshot.timestamp = std::chrono::system_clock::now();

    // 记录到历史列表（线程安全）
    {
        std::lock_guard<std::mutex> lock(history_mutex_);
        record_load_snapshot(snapshot, gpu_load_history_);
    }

    // 更新最后监控时间
    last_monitor_time_ = snapshot.timestamp;

    return snapshot;
}
#else
/**
 * @brief 获取GPU当前负载快照（非CUDA环境降级实现）
 *
 * 在非CUDA环境（无GPU）下返回零值快照。
 * 所有字段初始化为0/空值，设备类型标记为GPU但利用率等均为0。
 *
 * @param device_id CUDA设备索引（忽略，因为无GPU可用）
 * @return DeviceLoadSnapshot 空GPU负载快照
 */
DeviceLoadSnapshot DeviceLoadBalancer::get_gpu_load(int device_id) {
    // 非CUDA环境下GPU不可用，返回零值快照
    // 抑制未使用参数警告
    (void)device_id;

    DeviceLoadSnapshot snapshot;
    snapshot.device_type        = ComputeDeviceType::GPU;
    snapshot.utilization        = 0.0;
    snapshot.available_memory_mb = 0;
    snapshot.queue_depth        = gpu_queue_depth_.load(std::memory_order_acquire);
    snapshot.temperature_celsius = 0.0;
    snapshot.timestamp          = std::chrono::system_clock::now();
    return snapshot;
}
#endif  // SPACEGO_CUDA_ENABLED

// ============================================================================
// 负载均衡核心决策
// ============================================================================

/**
 * @brief 计算动态CPU/GPU任务分配比例
 *
 * 这是负载均衡器的核心算法，实现自适应的任务分配策略。
 *
 * 算法详细流程：
 *   步骤1 - 采样: 分别获取CPU和GPU的当前负载快照。
 *   步骤2 - EMA平滑: 使用指数移动平均更新CPU和GPU的利用率EMA值，
 *           消除瞬时波动，提供稳定的利用率趋势。
 *   步骤3 - 负载比较与比例调整:
 *           a) 如果GPU EMA利用率 > GPU过载阈值(0.85)
 *              且CPU EMA利用率 < CPU空闲阈值(0.40):
 *              认为GPU过载而CPU空闲，增加CPU分配比例(补偿步长+0.05)，
 *              同时减少GPU分配比例。
 *           b) 如果CPU EMA利用率 > CPU过载阈值(0.90):
 *              认为CPU过载，减少CPU分配比例(补偿步长-0.05)，
 *              同时增加GPU分配比例。
 *           c) 如果都不是（负载均衡状态）：保持当前比例不变。
 *   步骤4 - 比例约束: 使用 clamp_ratio() 将比例限制在 [0.1, 0.9]，
 *           永远不给单一设备100%的任务分配。保证 gpu_ratio = 1.0 - cpu_ratio。
 *   步骤5 - EMA平滑过渡: 使用EMA平滑分配比例的变化，
 *           避免比例突变导致调度震荡。
 *   步骤6 - 生成决策: 创建 AllocationDecision 包含比例和描述信息。
 *
 * @return AllocationDecision 包含新的CPU/GPU分配比例和决策描述的决策结果
 */
AllocationDecision DeviceLoadBalancer::calculate_allocation_ratio() {
    // ---- 步骤1: 获取当前负载采样 ----
    DeviceLoadSnapshot cpu_snapshot = get_cpu_load();
    DeviceLoadSnapshot gpu_snapshot = get_gpu_load();

    double current_cpu_util = cpu_snapshot.utilization;
    double current_gpu_util = gpu_snapshot.utilization;

    // ---- 步骤2: EMA平滑利用率 ----
    double alpha = config_.load_balance_smooth_factor;

    // 更新CPU利用率EMA
    cpu_utilization_ema_ = update_ema(cpu_utilization_ema_, current_cpu_util, alpha);

    // 更新GPU利用率EMA
    gpu_utilization_ema_ = update_ema(gpu_utilization_ema_, current_gpu_util, alpha);

    // ---- 步骤3: 负载比较与比例调整 ----

    // 获取配置中的阈值
    double gpu_overload_threshold = config_.gpu_overload_threshold;   // 默认0.85
    double cpu_overload_threshold = config_.cpu_overload_threshold;   // 默认0.90
    double cpu_idle_threshold     = config_.cpu_idle_threshold;       // 默认0.40

    // 情况A: GPU过载 + CPU空闲 => 将任务从GPU转移到CPU
    // 条件: GPU EMA利用率超过过载阈值 且 CPU EMA利用率低于空闲阈值
    if (gpu_utilization_ema_ > gpu_overload_threshold
        && cpu_utilization_ema_ < cpu_idle_threshold) {

        // 增加CPU分配比例，减少GPU分配比例
        current_cpu_ratio_ += COMPENSATION_DELTA;
        current_gpu_ratio_ -= COMPENSATION_DELTA;

        LOG_INFO("负载均衡: GPU过载(EMA=" + std::to_string(gpu_utilization_ema_)
                 + ")，CPU空闲(EMA=" + std::to_string(cpu_utilization_ema_)
                 + ")，增加CPU分配比例至" + std::to_string(current_cpu_ratio_));
    }
    // 情况B: CPU过载 => 将任务从CPU转移到GPU（只要GPU不是也过载）
    // 条件: CPU EMA利用率超过过载阈值
    else if (cpu_utilization_ema_ > cpu_overload_threshold) {

        // 减少CPU分配比例，增加GPU分配比例
        current_cpu_ratio_ -= COMPENSATION_DELTA;
        current_gpu_ratio_ += COMPENSATION_DELTA;

        LOG_INFO("负载均衡: CPU过载(EMA=" + std::to_string(cpu_utilization_ema_)
                 + ")，增加GPU分配比例至" + std::to_string(current_gpu_ratio_));
    }
    // 情况C: 负载均衡状态 - 保持当前比例不变
    // 不做任何调整，避免不必要的震荡

    // ---- 步骤4: 比例约束 ----
    // 确保比例在安全范围 [0.1, 0.9] 内
    current_cpu_ratio_ = clamp_ratio(current_cpu_ratio_);
    current_gpu_ratio_ = 1.0 - current_cpu_ratio_;
    // 也对GPU比例做约束（虽然理论上由cpu_ratio推导，但浮点精度可能导致微小偏差）
    current_gpu_ratio_ = clamp_ratio(current_gpu_ratio_);
    // 归一化确保两者之和为1.0
    double total_ratio = current_cpu_ratio_ + current_gpu_ratio_;
    if (total_ratio > 0.0) {
        current_cpu_ratio_ /= total_ratio;
        current_gpu_ratio_ /= total_ratio;
    }

    // ---- 步骤5: EMA平滑过渡分配比例 ----
    // 使用EMA平滑分配比例以避免突变
    cpu_allocation_ema_ = update_ema(cpu_allocation_ema_, current_cpu_ratio_, alpha);
    gpu_allocation_ema_ = update_ema(gpu_allocation_ema_, current_gpu_ratio_, alpha);

    // ---- 步骤6: 生成分配决策 ----
    AllocationDecision decision;

    // 使用EMA平滑后的比例作为最终决策
    decision.cpu_ratio = cpu_allocation_ema_;
    decision.gpu_ratio = gpu_allocation_ema_;

    // 默认不触发分流/回流（由check_gpu_overload/check_cpu_overload单独判断）
    decision.should_offload_to_cpu = false;
    decision.should_route_to_gpu   = false;

    // 生成决策原因描述（用于日志和调试）
    std::ostringstream reason_stream;
    reason_stream << "CPU利用率(EMA)=" << std::fixed << std::setprecision(2)
                  << cpu_utilization_ema_
                  << "，GPU利用率(EMA)=" << gpu_utilization_ema_
                  << "，分配比例: CPU=" << decision.cpu_ratio
                  << "，GPU=" << decision.gpu_ratio
                  << "，平滑因子alpha=" << alpha;
    decision.reason = reason_stream.str();

    return decision;
}

/**
 * @brief GPU过载检测与CPU分流逻辑
 *
 * 检测GPU是否持续处于过载状态。如果连续3次采样GPU利用率
 * 都超过过载阈值（默认0.85），则认为GPU确实过载而非瞬时波动，
 * 触发 should_offload_to_cpu=true，建议调度引擎将部分GPU任务
 * 卸载到CPU。
 *
 * 连续过载检测的设计理由：
 *   - 避免因瞬时GPU利用率尖峰而错误触发分流
 *   - 3次连续检测在响应速度和稳定性之间取得平衡
 *   - 每次检测间隔由 monitoring_interval_ms 控制（默认100ms）
 *
 * 分流策略：
 *   - 即使触发分流，也需要CPU有空闲容量才真正卸载
 *   - 如果CPU也过载，则不会触发分流（避免雪上加霜）
 *
 * @return AllocationDecision 包含分流判断的决策结果
 */
AllocationDecision DeviceLoadBalancer::check_gpu_overload() {
    AllocationDecision decision;

    // 获取GPU当前负载
    DeviceLoadSnapshot gpu_snapshot = get_gpu_load();
    double current_gpu_util = gpu_snapshot.utilization;

    // 用EMA更新GPU利用率
    double alpha = config_.load_balance_smooth_factor;
    gpu_utilization_ema_ = update_ema(gpu_utilization_ema_, current_gpu_util, alpha);

    // 判断是否超过GPU过载阈值
    if (gpu_utilization_ema_ > config_.gpu_overload_threshold) {
        // GPU过载，增加连续过载计数
        consecutive_gpu_overload_count_++;

        // 检查是否连续3次过载
        if (consecutive_gpu_overload_count_ >= 3) {
            // GPU持续过载，检查CPU是否有空闲容量
            DeviceLoadSnapshot cpu_snapshot = get_cpu_load();
            double current_cpu_util = cpu_snapshot.utilization;
            cpu_utilization_ema_ = update_ema(cpu_utilization_ema_, current_cpu_util, alpha);

            if (cpu_utilization_ema_ < config_.cpu_overload_threshold) {
                // CPU未过载，可以接收从GPU卸载的任务
                decision.should_offload_to_cpu = true;
                decision.should_route_to_gpu   = false;
                decision.cpu_ratio = cpu_allocation_ema_;
                decision.gpu_ratio = gpu_allocation_ema_;

                std::ostringstream reason_stream;
                reason_stream << "GPU持续过载（连续" 
                              << consecutive_gpu_overload_count_
                              << "次采样，EMA=" << std::fixed << std::setprecision(2)
                              << gpu_utilization_ema_ << "），触发CPU分流";
                decision.reason = reason_stream.str();

                LOG_INFO(decision.reason);

                // 触发分流后重置计数器，避免重复触发
                // 注意：实际是否执行分流由调度引擎决定
                // 这里只发出信号，计数器保持在触发状态直到GPU利用率下降
            } else {
                // CPU也已过载，无法分流
                decision.should_offload_to_cpu = false;
                decision.should_route_to_gpu   = false;
                decision.cpu_ratio = cpu_allocation_ema_;
                decision.gpu_ratio = gpu_allocation_ema_;

                std::ostringstream reason_stream;
                reason_stream << "GPU持续过载但CPU也已过载（EMA="
                              << std::fixed << std::setprecision(2)
                              << cpu_utilization_ema_ << "），无法分流";
                decision.reason = reason_stream.str();

                LOG_WARNING(decision.reason);
            }
        } else {
            // 过载次数不足3次，继续观察
            decision.should_offload_to_cpu = false;
            decision.should_route_to_gpu   = false;
            decision.cpu_ratio = cpu_allocation_ema_;
            decision.gpu_ratio = gpu_allocation_ema_;

            std::ostringstream reason_stream;
            reason_stream << "GPU利用率偏高（EMA=" << std::fixed << std::setprecision(2)
                          << gpu_utilization_ema_ << "），连续过载计数="
                          << consecutive_gpu_overload_count_ << "/3，继续观察";
            decision.reason = reason_stream.str();
        }
    } else {
        // GPU未过载，重置连续过载计数器
        consecutive_gpu_overload_count_ = 0;

        decision.should_offload_to_cpu = false;
        decision.should_route_to_gpu   = false;
        decision.cpu_ratio = cpu_allocation_ema_;
        decision.gpu_ratio = gpu_allocation_ema_;

        decision.reason = "GPU负载正常（EMA=" + std::to_string(gpu_utilization_ema_) + "）";
    }

    return decision;
}

/**
 * @brief CPU过载保护与GPU回路由逻辑
 *
 * 检测CPU是否持续处于过载状态。如果连续3次采样CPU利用率
 * 都超过过载阈值（默认0.90），则认为CPU确实过载，
 * 触发 should_route_to_gpu=true，建议调度引擎将部分任务
 * 从CPU转移到GPU。
 *
 * 回路由策略的保护性质：
 *   - 当CPU不堪重负时，即使某些任务更适合CPU执行，
 *     也应考虑GPU作为替代方案
 *   - 回路由的前提是GPU未过载（否则无法接收更多任务）
 *
 * @return AllocationDecision 包含回路由判断的决策结果
 */
AllocationDecision DeviceLoadBalancer::check_cpu_overload() {
    AllocationDecision decision;

    // 获取CPU当前负载
    DeviceLoadSnapshot cpu_snapshot = get_cpu_load();
    double current_cpu_util = cpu_snapshot.utilization;

    // 用EMA更新CPU利用率
    double alpha = config_.load_balance_smooth_factor;
    cpu_utilization_ema_ = update_ema(cpu_utilization_ema_, current_cpu_util, alpha);

    // 判断是否超过CPU过载阈值（默认0.90 - 比GPU阈值更严格，
    // 因为CPU是主要的后备执行设备，不应轻易判定为过载）
    if (cpu_utilization_ema_ > config_.cpu_overload_threshold) {
        // CPU过载，增加连续过载计数
        consecutive_cpu_overload_count_++;

        // 检查是否连续3次过载
        if (consecutive_cpu_overload_count_ >= 3) {
            // CPU持续过载，检查GPU是否有空闲容量
            DeviceLoadSnapshot gpu_snapshot = get_gpu_load();
            double current_gpu_util = gpu_snapshot.utilization;
            gpu_utilization_ema_ = update_ema(gpu_utilization_ema_, current_gpu_util, alpha);

            if (gpu_utilization_ema_ < config_.gpu_overload_threshold) {
                // GPU未过载，可以将任务从CPU路由到GPU
                decision.should_route_to_gpu   = true;
                decision.should_offload_to_cpu = false;
                decision.cpu_ratio = cpu_allocation_ema_;
                decision.gpu_ratio = gpu_allocation_ema_;

                std::ostringstream reason_stream;
                reason_stream << "CPU持续过载（连续"
                              << consecutive_cpu_overload_count_
                              << "次采样，EMA=" << std::fixed << std::setprecision(2)
                              << cpu_utilization_ema_ << "），触发GPU回路由";
                decision.reason = reason_stream.str();

                LOG_INFO(decision.reason);
            } else {
                // GPU也已过载，无法回路由
                decision.should_route_to_gpu   = false;
                decision.should_offload_to_cpu = false;
                decision.cpu_ratio = cpu_allocation_ema_;
                decision.gpu_ratio = gpu_allocation_ema_;

                std::ostringstream reason_stream;
                reason_stream << "CPU持续过载但GPU也已过载（EMA="
                              << std::fixed << std::setprecision(2)
                              << gpu_utilization_ema_ << "），无法回路由到GPU";
                decision.reason = reason_stream.str();

                LOG_WARNING(decision.reason);
            }
        } else {
            // 过载次数不足3次，继续观察
            decision.should_route_to_gpu   = false;
            decision.should_offload_to_cpu = false;
            decision.cpu_ratio = cpu_allocation_ema_;
            decision.gpu_ratio = gpu_allocation_ema_;

            std::ostringstream reason_stream;
            reason_stream << "CPU利用率偏高（EMA=" << std::fixed << std::setprecision(2)
                          << cpu_utilization_ema_ << "），连续过载计数="
                          << consecutive_cpu_overload_count_ << "/3，继续观察";
            decision.reason = reason_stream.str();
        }
    } else {
        // CPU未过载，重置连续过载计数器
        consecutive_cpu_overload_count_ = 0;

        decision.should_route_to_gpu   = false;
        decision.should_offload_to_cpu = false;
        decision.cpu_ratio = cpu_allocation_ema_;
        decision.gpu_ratio = gpu_allocation_ema_;

        decision.reason = "CPU负载正常（EMA=" + std::to_string(cpu_utilization_ema_) + "）";
    }

    return decision;
}

// ============================================================================
// 配置管理
// ============================================================================

/**
 * @brief 更新异构计算配置（线程安全）
 *
 * 在运行时动态调整负载均衡参数。
 * 使用互斥锁保护配置更新操作。
 * 更新后会重置EMA值和分配比例（因为新配置可能改变阈值语义）。
 *
 * @param config 新的配置参数
 */
void DeviceLoadBalancer::update_config(const HeterogeneousComputeConfig& config) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    config_ = config;

    // 配置变更时重置EMA值（避免旧配置的累积值影响新配置的决策）
    // 这是一个保守的策略：配置变更意味着运行环境可能发生了变化
    cpu_utilization_ema_ = 0.0;
    gpu_utilization_ema_ = 0.0;
    cpu_allocation_ema_  = 0.5;
    gpu_allocation_ema_  = 0.5;
    current_cpu_ratio_   = 0.5;
    current_gpu_ratio_   = 0.5;

    // 重置过载计数器
    consecutive_gpu_overload_count_ = 0;
    consecutive_cpu_overload_count_ = 0;

    LOG_INFO("设备负载均衡器配置已更新：GPU过载阈值="
             + std::to_string(config_.gpu_overload_threshold)
             + "，CPU过载阈值=" + std::to_string(config_.cpu_overload_threshold)
             + "，EMA平滑因子=" + std::to_string(config_.load_balance_smooth_factor));
}

// ============================================================================
// 历史负载数据访问
// ============================================================================

/**
 * @brief 获取CPU负载历史快照列表
 *
 * 返回最近最多 MAX_LOAD_HISTORY 个CPU负载快照的副本。
 * 线程安全：使用互斥锁保护历史记录的读取。
 *
 * @return 历史快照列表副本（按时间顺序，旧->新）
 */
std::vector<DeviceLoadSnapshot> DeviceLoadBalancer::get_cpu_load_history() const {
    std::lock_guard<std::mutex> lock(history_mutex_);
    return cpu_load_history_;  // 返回副本，释放锁后调用方可安全使用
}

/**
 * @brief 获取GPU负载历史快照列表
 *
 * 返回最近最多 MAX_LOAD_HISTORY 个GPU负载快照的副本。
 * 线程安全：使用互斥锁保护历史记录的读取。
 *
 * @return 历史快照列表副本（按时间顺序，旧->新）
 */
std::vector<DeviceLoadSnapshot> DeviceLoadBalancer::get_gpu_load_history() const {
    std::lock_guard<std::mutex> lock(history_mutex_);
    return gpu_load_history_;  // 返回副本，释放锁后调用方可安全使用
}

// ============================================================================
// CPU利用率计算
// ============================================================================

#ifdef _WIN32
/**
 * @brief Windows平台CPU利用率计算
 *
 * 使用 GetSystemTimes() API 获取系统级别的CPU时间统计：
 *   - lpIdleTime:   系统空闲时间
 *   - lpKernelTime: 内核态执行时间（含空闲）
 *   - lpUserTime:   用户态执行时间
 *
 * 计算方法：
 *   1. 获取当前时间快照 (idle_now, kernel_now, user_now)
 *   2. 如果有上次采样数据，计算差值：
 *      kernel_delta = kernel_now - kernel_prev
 *      user_delta   = user_now   - user_prev
 *      idle_delta   = idle_now   - idle_prev
 *      total_delta  = kernel_delta + user_delta
 *      utilization  = 1.0 - idle_delta / total_delta
 *   3. 首次调用时保存快照，返回0.0（无历史对比数据）
 *   4. 更新 prev_* 值为当前采样值
 *
 * 注意：
 *   - 此方法计算的是自上次采样以来整个系统的平均CPU利用率
 *   - 不是瞬时利用率，而是时间段内的平均值
 *   - 监控间隔越短，读数越接近瞬时值
 *
 * @return CPU利用率 (0.0 ~ 1.0)，首次调用返回0.0
 */
double DeviceLoadBalancer::calculate_cpu_utilization() {
    // 获取当前系统CPU时间统计
    FILETIME idle_time_ft, kernel_time_ft, user_time_ft;

    BOOL success = GetSystemTimes(&idle_time_ft, &kernel_time_ft, &user_time_ft);
    if (!success) {
        // GetSystemTimes调用失败（极少发生），返回上次EMA值
        LOG_WARNING("GetSystemTimes()调用失败，返回上次EMA值");
        return cpu_utilization_ema_;
    }

    // 将FILETIME转换为uint64_t（单位：100纳秒）
    uint64_t idle_time_now   = filetime_to_uint64(idle_time_ft);
    uint64_t kernel_time_now = filetime_to_uint64(kernel_time_ft);
    uint64_t user_time_now   = filetime_to_uint64(user_time_ft);

    double utilization = 0.0;

    if (cpu_sample_valid_) {
        // ---- 有历史数据，计算差值得到利用率 ----

        // 计算各时间分量的差值
        uint64_t kernel_delta = kernel_time_now - prev_kernel_time_;
        uint64_t user_delta   = user_time_now   - prev_user_time_;
        uint64_t idle_delta   = idle_time_now   - prev_idle_time_;

        // 总CPU时间 = 内核态时间 + 用户态时间
        // 注意：内核态时间已经包含了空闲时间
        uint64_t total_delta = kernel_delta + user_delta;

        if (total_delta > 0) {
            // 利用率 = 1 - (空闲时间 / 总时间)
            // idle_delta可能大于total_delta（因为kernel_time含idle），需要做保护
            double idle_ratio = static_cast<double>(idle_delta)
                              / static_cast<double>(total_delta);

            // idle_ratio超过1.0属于正常情况（idle是kernel的子集），截断到1.0
            if (idle_ratio > 1.0) {
                idle_ratio = 1.0;
            }

            utilization = 1.0 - idle_ratio;

            // 确保利用率在有效范围内
            if (utilization < 0.0) utilization = 0.0;
            if (utilization > 1.0) utilization = 1.0;
        } else {
            // total_delta为0说明两次采样间隔极短（<1个时钟滴答），
            // 返回上次EMA值作为合理近似
            utilization = cpu_utilization_ema_;
        }
    } else {
        // ---- 首次调用，无历史数据 ----
        // 保存当前值作为下次对比的基准，返回0.0
        cpu_sample_valid_ = true;
        utilization = 0.0;
    }

    // 更新上次采样值，供下次调用使用
    prev_idle_time_   = idle_time_now;
    prev_kernel_time_ = kernel_time_now;
    prev_user_time_   = user_time_now;

    return utilization;
}
#else
/**
 * @brief Linux平台CPU利用率计算
 *
 * 读取 /proc/stat 文件的第一行（总体CPU时间统计），
 * 格式: "cpu  user nice system idle iowait irq softirq steal guest guest_nice"
 *
 * 计算方法：
 *   total = user + nice + system + idle + iowait + irq + softirq + steal
 *   idle  = idle + iowait
 *   utilization = 1.0 - (idle_delta / total_delta)
 *
 * 注意：
 *   - Linux的iowait时间被视为空闲（CPU在等待I/O，不算有效工作）
 *   - steal时间（虚拟化环境中被hypervisor偷走的时间）计入总量但不计入空闲
 *   - 首次调用保存快照，返回0.0
 *
 * @return CPU利用率 (0.0 ~ 1.0)，首次调用返回0.0
 */
double DeviceLoadBalancer::calculate_cpu_utilization() {
    // 打开/proc/stat文件读取CPU统计信息
    std::ifstream stat_file("/proc/stat");
    if (!stat_file.is_open()) {
        // 文件打开失败（极少见），返回上次EMA值
        LOG_WARNING("/proc/stat文件打开失败，返回上次EMA值");
        return cpu_utilization_ema_;
    }

    std::string line;
    if (!std::getline(stat_file, line)) {
        // 读取失败
        LOG_WARNING("/proc/stat文件读取失败，返回上次EMA值");
        return cpu_utilization_ema_;
    }

    // 解析CPU时间统计行
    // 格式: "cpu  user nice system idle iowait irq softirq steal guest guest_nice"
    std::istringstream iss(line);
    std::string cpu_label;
    uint64_t user_time   = 0;
    uint64_t nice_time   = 0;
    uint64_t system_time = 0;
    uint64_t idle_time   = 0;
    uint64_t iowait_time = 0;
    uint64_t irq_time    = 0;
    uint64_t softirq_time = 0;
    uint64_t steal_time  = 0;

    iss >> cpu_label
        >> user_time >> nice_time >> system_time
        >> idle_time >> iowait_time >> irq_time
        >> softirq_time >> steal_time;

    // 验证标签是否为"cpu"
    if (cpu_label != "cpu") {
        LOG_WARNING("/proc/stat首行格式异常，返回上次EMA值");
        return cpu_utilization_ema_;
    }

    // 计算总CPU时间和空闲时间
    // 总时间包含所有字段
    uint64_t total_time_now = user_time + nice_time + system_time
                            + idle_time + iowait_time + irq_time
                            + softirq_time + steal_time;
    // 空闲时间 = 纯空闲 + I/O等待（I/O等待期间CPU实际空闲）
    uint64_t idle_time_now = idle_time + iowait_time;

    double utilization = 0.0;

    if (cpu_sample_valid_) {
        // ---- 有历史数据，计算差值得到利用率 ----
        uint64_t total_delta = total_time_now - prev_total_time_;
        uint64_t idle_delta  = idle_time_now  - prev_idle_time_;

        if (total_delta > 0) {
            utilization = 1.0 - static_cast<double>(idle_delta)
                              / static_cast<double>(total_delta);

            // 确保利用率在有效范围内
            if (utilization < 0.0) utilization = 0.0;
            if (utilization > 1.0) utilization = 1.0;
        } else {
            // total_delta为0，返回上次EMA值
            utilization = cpu_utilization_ema_;
        }
    } else {
        // ---- 首次调用，无历史数据 ----
        cpu_sample_valid_ = true;
        utilization = 0.0;
    }

    // 更新上次采样值
    prev_total_time_ = total_time_now;
    prev_idle_time_  = idle_time_now;

    return utilization;
}
#endif  // _WIN32

// ============================================================================
// GPU利用率计算（CUDA环境，通过NVML API）
// ============================================================================

#if SPACEGO_CUDA_ENABLED
/**
 * @brief 通过NVML API计算GPU当前利用率
 *
 * NVML（NVIDIA Management Library）提供了精确的GPU利用率查询。
 * 此方法在首次调用时自动初始化NVML（延迟初始化）。
 *
 * 实现策略：
 *   1. 如果NVML已初始化（nvml_initialized_=true），直接查询GPU利用率
 *   2. 如果未初始化，调用 initialize_nvml() 尝试初始化
 *   3. 初始化成功后使用 nvmlDeviceGetUtilizationRates() 获取GPU SM使用率
 *   4. 返回的利用率值范围 [0, 100]，转换为 [0.0, 1.0]
 *   5. 如果NVML不可用（初始化失败），回退到启发式估算：
 *      - 基于GPU任务队列深度与最大队列容量的比值估算
 *      - 这是保守估算，实际GPU利用率可能更高
 *
 * @param device_id CUDA设备索引（默认0）
 * @return GPU利用率 (0.0 ~ 1.0)，不可用时返回0.0
 */
double DeviceLoadBalancer::calculate_gpu_utilization(int device_id) {
    // ---- 策略1: 尝试通过NVML获取精确利用率 ----

    // 确保NVML已初始化（首次调用时延迟初始化）
    if (!nvml_initialized_) {
        if (!initialize_nvml(device_id)) {
            // NVML初始化失败，回退到启发式估算
            // 注意：已经记录过警告日志，此处不重复记录
        }
    }

    // 如果NVML可用，查询GPU SM使用率
    if (nvml_initialized_ && nvml_device_handle_ != nullptr) {
        nvmlUtilization_t utilization_rates;
        nvmlReturn_t result = nvmlDeviceGetUtilizationRates(
            static_cast<nvmlDevice_t>(nvml_device_handle_),
            &utilization_rates
        );

        if (result == NVML_SUCCESS) {
            // NVML返回的利用率范围 [0, 100]，转换为 [0.0, 1.0]
            // utilization_rates.gpu 表示GPU核心（SM）的使用率
            // utilization_rates.memory 表示显存控制器的使用率
            // 我们主要关注GPU核心使用率（与计算任务直接相关）
            return static_cast<double>(utilization_rates.gpu) / 100.0;
        }
        // NVML查询失败（GPU可能被重置或驱动状态异常），回退到启发式估算
    }

    // ---- 策略2: 启发式估算（NVML不可用时的降级方案） ----

    // 基于GPU任务队列深度的启发式估算：
    //   utilization = queue_depth / max_queue_size
    // 这是非常粗略的估算，但总比返回0.0要好
    // 至少能反映GPU当前是否有任务在排队等待
    size_t queue_depth = gpu_queue_depth_.load(std::memory_order_acquire);
    size_t max_queue   = config_.gpu_task_queue_max_size;

    if (max_queue > 0) {
        double heuristic_util = static_cast<double>(queue_depth)
                              / static_cast<double>(max_queue);
        // 限制在 [0.0, 1.0] 范围内
        if (heuristic_util > 1.0) heuristic_util = 1.0;
        if (heuristic_util < 0.0) heuristic_util = 0.0;

        // 启发式估算通常低于实际利用率（因为GPU可能同时执行多个内核），
        // 应用1.2倍放大因子使其更接近实际情况
        // 放大因子的选择基于经验观察：队列深度为50%时GPU实际利用率通常在60%左右
        constexpr double HEURISTIC_AMPLIFICATION = 1.2;
        double amplified_util = heuristic_util * HEURISTIC_AMPLIFICATION;
        if (amplified_util > 1.0) amplified_util = 1.0;

        return amplified_util;
    }

    // 完全无法估算，返回0.0
    return 0.0;
}

/**
 * @brief 尝试初始化NVML库
 *
 * 延迟初始化策略：不在构造函数中初始化NVML，
 * 而是在首次需要GPU利用率数据时才初始化。
 * 这样即使NVML不可用，也不会影响DeviceLoadBalancer的构造。
 *
 * 初始化流程：
 *   1. 调用 nvmlInit_v2() 初始化NVML运行时
 *   2. 调用 nvmlDeviceGetHandleByIndex() 获取指定GPU的设备句柄
 *   3. 验证设备句柄有效性
 *   4. 设置 nvml_initialized_ = true 和 nvml_device_handle_
 *
 * 错误处理：
 *   - NVML初始化失败：记录警告，设置nvml_initialized_=false
 *   - 设备句柄获取失败：记录警告，设置nvml_initialized_=false
 *   - 所有失败情况都会回退到启发式估算
 *
 * @param device_id CUDA设备索引
 * @return true 初始化成功
 */
bool DeviceLoadBalancer::initialize_nvml(int device_id) {
    // ---- 步骤1: 初始化NVML运行时 ----
    nvmlReturn_t init_result = nvmlInit_v2();
    if (init_result != NVML_SUCCESS) {
        // NVML初始化失败的可能原因：
        //   - NVIDIA驱动未安装或版本过旧
        //   - NVML库文件缺失或损坏
        //   - 权限不足（Linux上可能需要root权限）
        LOG_WARNING("NVML初始化失败（错误代码：" 
                    + std::to_string(static_cast<int>(init_result))
                    + "），将使用启发式GPU利用率估算");
        nvml_initialized_   = false;
        nvml_device_handle_ = nullptr;
        return false;
    }

    // ---- 步骤2: 获取GPU设备句柄 ----
    nvmlDevice_t device_handle = nullptr;
    nvmlReturn_t handle_result = nvmlDeviceGetHandleByIndex(
        static_cast<unsigned int>(device_id),
        &device_handle
    );

    if (handle_result != NVML_SUCCESS || device_handle == nullptr) {
        LOG_WARNING("无法获取GPU设备句柄（设备ID=" + std::to_string(device_id)
                    + "，错误代码=" + std::to_string(static_cast<int>(handle_result))
                    + "），将使用启发式GPU利用率估算");
        // 设备句柄获取失败，关闭已初始化的NVML
        nvmlShutdown();
        nvml_initialized_   = false;
        nvml_device_handle_ = nullptr;
        return false;
    }

    // ---- 步骤3: NVML初始化成功 ----
    nvml_initialized_   = true;
    nvml_device_handle_ = static_cast<void*>(device_handle);

    LOG_INFO("NVML初始化成功，GPU设备ID=" + std::to_string(device_id)
             + "，已准备好GPU利用率监控");
    return true;
}
#endif  // SPACEGO_CUDA_ENABLED

// ============================================================================
// 指数移动平均（EMA）更新
// ============================================================================

/**
 * @brief 执行指数移动平均（EMA）更新
 *
 * EMA（Exponential Moving Average）是一种时间序列平滑技术，
 * 赋予近期数据更高的权重，同时保留历史趋势信息。
 *
 * 公式:
 *   new_ema = alpha * new_value + (1 - alpha) * old_ema
 *
 * 参数说明:
 *   - alpha (0.0 ~ 1.0): 平滑因子
 *     - 接近1.0: 对新数据非常敏感，EMA快速跟踪实际值变化
 *               适合波动较大的场景
 *     - 接近0.0: 对新数据不敏感，EMA变化缓慢稳定
 *               适合需要消除噪声的场景
 *     - 推荐默认值: 0.3（在响应速度和稳定性间取得平衡）
 *
 * 本项目中的应用:
 *   - CPU/GPU利用率EMA: 消除瞬时负载波动，提供稳定的利用率趋势
 *   - 分配比例EMA: 避免比例突变导致调度震荡
 *
 * @param current_ema 当前的EMA值
 * @param new_value   新采样的原始值
 * @param alpha       EMA平滑因子 (0.0 ~ 1.0)
 * @return 更新后的EMA值
 */
double DeviceLoadBalancer::update_ema(double current_ema, double new_value, double alpha) {
    // 对alpha进行范围约束，确保数值稳定
    if (alpha < 0.0) alpha = 0.0;
    if (alpha > 1.0) alpha = 1.0;

    // EMA核心公式
    return alpha * new_value + (1.0 - alpha) * current_ema;
}

// ============================================================================
// 监控间隔判断
// ============================================================================

/**
 * @brief 判断是否已达到监控采样间隔
 *
 * 通过比较当前时间与上次监控时间的差值，
 * 判断是否超过了 config_.monitoring_interval_ms 设定的间隔。
 *
 * 设计理由：
 *   - 避免过于频繁的系统API调用（GetSystemTimes/cudaMemGetInfo有开销）
 *   - 默认间隔100ms提供了足够的响应速度
 *   - 即使间隔未到，调用方仍可手动触发采样
 *
 * @return true 表示已经超过间隔，应该进行新的采样
 *         false 表示尚未到达间隔，可以跳过本次采样
 */
bool DeviceLoadBalancer::should_monitor() const {
    auto now = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_monitor_time_
    );

    return static_cast<size_t>(elapsed.count()) >= config_.monitoring_interval_ms;
}

// ============================================================================
// 负载快照记录
// ============================================================================

/**
 * @brief 记录负载快照到历史列表
 *
 * 将新快照追加到历史列表末尾。当历史记录数量超过
 * MAX_LOAD_HISTORY（64条）时，移除最旧的记录。
 *
 * 历史记录的用途:
 *   - 诊断负载变化趋势（是否持续上升/下降）
 *   - 性能分析（查看特定时间段的设备负载）
 *   - 回归分析（评估不同配置参数的调度效果）
 *
 * 注意：此方法不直接加锁，调用方需要在外部持有 history_mutex_。
 *
 * @param snapshot 新的负载快照
 * @param history  目标历史列表（cpu_load_history_ 或 gpu_load_history_）
 */
void DeviceLoadBalancer::record_load_snapshot(
    const DeviceLoadSnapshot& snapshot,
    std::vector<DeviceLoadSnapshot>& history) {

    // 追加新快照到列表末尾
    history.push_back(snapshot);

    // 如果超过最大容量，移除最旧的记录（FIFO策略）
    // 这样历史记录始终保留最近 MAX_LOAD_HISTORY 个采样点
    while (history.size() > MAX_LOAD_HISTORY) {
        // 移除最旧记录（vector头部删除，时间复杂度O(n)
        // 由于MAX_LOAD_HISTORY仅为64，此操作开销可忽略）
        history.erase(history.begin());
    }
}

// ============================================================================
// 分配比例约束
// ============================================================================

/**
 * @brief 将分配比例约束在安全范围 [0.1, 0.9] 内
 *
 * 永远不给单一设备100%的任务分配。理由：
 *   - 保留弹性: 避免某设备故障时没有后备方案
 *   - 负载震荡: 100%分配会导致设备间负载剧烈切换
 *   - 数据预热: 即使只有10%的任务在另一设备，也能保持数据通道活跃
 *
 * 边界值说明：
 *   - 下限 0.10: 至少保留10%的任务给该设备，确保不会被完全闲置
 *   - 上限 0.90: 最多分配90%的任务给该设备，保留10%给另一设备
 *
 * @param ratio 原始分配比例
 * @return 约束后的分配比例
 */
double DeviceLoadBalancer::clamp_ratio(double ratio) {
    constexpr double MIN_RATIO = 0.10;  // 最小分配比例
    constexpr double MAX_RATIO = 0.90;  // 最大分配比例

    if (ratio < MIN_RATIO) return MIN_RATIO;
    if (ratio > MAX_RATIO) return MAX_RATIO;
    return ratio;
}

// ============================================================================
// 系统可用内存获取
// ============================================================================

#ifdef _WIN32
/**
 * @brief Windows平台获取当前系统可用内存（MB）
 *
 * 使用 GlobalMemoryStatusEx() API 获取系统内存状态。
 * 返回当前可用的物理内存大小（单位：MB）。
 *
 * 注意：
 *   - ullAvailPhys 包括空闲内存和可以被快速回收的缓存内存
 *   - 返回值为0表示API调用失败
 *
 * @return 可用内存大小（MB）
 */
size_t DeviceLoadBalancer::get_available_system_memory_mb() {
    MEMORYSTATUSEX mem_info;
    mem_info.dwLength = sizeof(MEMORYSTATUSEX);

    if (GlobalMemoryStatusEx(&mem_info)) {
        return static_cast<size_t>(mem_info.ullAvailPhys / (1024 * 1024));
    } else {
        // API调用失败，返回0表示无法获取
        LOG_WARNING("GlobalMemoryStatusEx()调用失败，无法获取系统可用内存");
        return 0;
    }
}
#else
/**
 * @brief Linux平台获取当前系统可用内存（MB）
 *
 * 使用 sysconf() 获取系统内存信息：
 *   - _SC_AVPHYS_PAGES: 可用物理页面数
 *   - _SC_PAGESIZE:     每个页面的大小（字节）
 *
 * 可用内存 = 可用页面数 * 页面大小
 *
 * 注意：
 *   - 返回值包括可以被快速回收的缓存页面
 *   - sysconf返回-1表示获取失败
 *
 * @return 可用内存大小（MB）
 */
size_t DeviceLoadBalancer::get_available_system_memory_mb() {
    long avail_pages = sysconf(_SC_AVPHYS_PAGES);
    long page_size   = sysconf(_SC_PAGESIZE);

    if (avail_pages > 0 && page_size > 0) {
        // 计算可用内存（字节 -> MB）
        uint64_t avail_bytes = static_cast<uint64_t>(avail_pages)
                             * static_cast<uint64_t>(page_size);
        return static_cast<size_t>(avail_bytes / (1024 * 1024));
    } else {
        // sysconf调用失败，返回0表示无法获取
        LOG_WARNING("sysconf()调用失败，无法获取系统可用内存");
        return 0;
    }
}
#endif  // _WIN32

}  // namespace heterogeneous
}  // namespace go
}  // namespace spacego