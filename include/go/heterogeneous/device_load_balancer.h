#pragma once

// ============================================================================
// 异构计算引擎 - 设备负载均衡器
// 命名空间: spacego::go::heterogeneous
// 功能: 实时监控CPU和GPU的负载状态，动态调整任务分配比例，
//       实现自适应负载均衡。当某设备过载时自动将任务分流到其他设备。
//
// 核心机制：
//   1. CPU利用率采样 - 通过Windows GetSystemTimes()或Linux /proc/stat
//   2. GPU利用率采样 - 通过NVML API（CUDA环境）或启发式估算
//   3. EMA平滑 - 使用指数移动平均消除瞬时波动
//   4. 动态比例调整 - 根据设备负载实时调整CPU/GPU任务分配比例
//   5. 过载保护 - 连续过载检测触发任务分流/回流
// ============================================================================

#include "heterogeneous_types.h"
#include "config/platform_config.h"

#include <memory>
#include <vector>
#include <atomic>
#include <mutex>
#include <chrono>
#include <cstddef>

// Windows API 类型前向声明（避免在头文件中暴露完整Windows.h）
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#define SPACEGO_DEV_LOAD_BALANCER_DEFINED_LEAN
#endif
#include <windows.h>
#ifdef SPACEGO_DEV_LOAD_BALANCER_DEFINED_LEAN
#undef WIN32_LEAN_AND_MEAN
#undef SPACEGO_DEV_LOAD_BALANCER_DEFINED_LEAN
#endif
#endif

namespace spacego {
namespace go {
namespace heterogeneous {

// ============================================================================
// DeviceLoadSnapshot - 设备负载快照
// ============================================================================

/**
 * @brief 记录某一时刻计算设备的负载状态快照
 *
 * 包含设备类型、利用率、可用内存、任务队列深度等信息。
 * GPU设备额外包含温度信息（通过NVML获取）。
 * 每次调用 get_cpu_load()/get_gpu_load() 时生成一个快照，
 * 历史快照保留最近 MAX_LOAD_HISTORY 个用于诊断分析。
 */
struct DeviceLoadSnapshot {
    ComputeDeviceType device_type;       // 设备类型（CPU或GPU）
    double            utilization;       // 利用率 (0.0 ~ 1.0)
                                         //   CPU: 通过系统API计算的整体使用率
                                         //   GPU: 通过NVML获取的SM使用率
    size_t            available_memory_mb;  // 当前可用内存 (MB)
                                            //   CPU: 系统可用RAM
                                            //   GPU: 设备可用显存
    size_t            queue_depth;         // 当前任务队列深度
                                           //   表示等待在该设备上执行的任务数量
    double            temperature_celsius; // 设备温度（摄氏度）
                                           //   CPU: 设为0.0（通常不通过此接口获取）
                                           //   GPU: 通过NVML获取，CUDA不可用时设为0.0
    std::chrono::system_clock::time_point timestamp;  // 采样时间戳

    // ========================================================================
    // 默认构造函数 - 初始化为零值/空状态
    // ========================================================================
    DeviceLoadSnapshot()
        : device_type(ComputeDeviceType::CPU)
        , utilization(0.0)
        , available_memory_mb(0)
        , queue_depth(0)
        , temperature_celsius(0.0)
        , timestamp(std::chrono::system_clock::now())
    {
    }
};

// ============================================================================
// AllocationDecision - 负载均衡分配决策
// ============================================================================

/**
 * @brief 描述负载均衡器的一次分配决策结果
 *
 * 决策内容包括：
 *   - CPU和GPU各自的任务分配比例（两者之和为1.0）
 *   - 是否需要将任务从GPU卸载到CPU
 *   - 是否需要将额外任务路由到GPU
 *   - 决策原因（用于日志记录和调试）
 */
struct AllocationDecision {
    double      cpu_ratio;            // CPU任务分配比例 (0.0 ~ 1.0)
                                      //   表示新任务应分配给CPU的比例
    double      gpu_ratio;            // GPU任务分配比例 (0.0 ~ 1.0)
                                      //   cpu_ratio + gpu_ratio 恒为 1.0
    bool        should_offload_to_cpu; // 是否应将GPU任务卸载到CPU
                                       //   当GPU持续过载时为true
    bool        should_route_to_gpu;   // 是否应将CPU任务路由到GPU
                                       //   当CPU持续过载时为true
    std::string reason;                // 决策原因描述
                                       //   用于日志记录，便于诊断调度行为

    // ========================================================================
    // 默认构造函数 - 初始化为均衡分配状态
    // ========================================================================
    AllocationDecision()
        : cpu_ratio(0.5)
        , gpu_ratio(0.5)
        , should_offload_to_cpu(false)
        , should_route_to_gpu(false)
        , reason("默认均衡分配（尚未进行负载检测）")
    {
    }
};

// ============================================================================
// DeviceLoadBalancer - 设备负载均衡器
// ============================================================================

/**
 * @brief 异构计算系统的设备负载均衡器
 *
 * 核心职责：
 *   1. 周期性监控CPU和GPU的负载状态（利用率、内存、队列深度）
 *   2. 使用EMA（指数移动平均）平滑负载数据，避免短时波动导致错误决策
 *   3. 根据负载状态动态计算CPU/GPU任务分配比例
 *   4. 检测设备过载并触发任务分流/回流机制
 *   5. 保存负载历史记录供诊断分析使用
 *
 * 线程安全说明：
 *   - 队列深度更新使用 std::atomic，无锁操作
 *   - 历史记录访问使用 std::mutex 保护
 *   - 配置更新使用 std::mutex 保护
 *   - 利用率采样（涉及系统API调用）不是严格线程安全的，
 *     调用方应确保不在多线程中并发调用采样方法
 *
 * 使用示例：
 *   @code
 *   HeterogeneousComputeConfig config;
 *   DeviceLoadBalancer balancer(config);
 *
 *   // 更新队列深度（由调度引擎调用）
 *   balancer.update_cpu_queue_depth(128);
 *   balancer.update_gpu_queue_depth(64);
 *
 *   // 获取分配决策
 *   AllocationDecision decision = balancer.calculate_allocation_ratio();
 *   LOG_INFO("CPU分配比例: {:.2f}, GPU分配比例: {:.2f}, 原因: {}",
 *            decision.cpu_ratio, decision.gpu_ratio, decision.reason);
 *   @endcode
 */
class DeviceLoadBalancer {
public:
    // ========================================================================
    // 构造/析构
    // ========================================================================

    /**
     * @brief 默认构造函数 - 使用默认的异构计算配置
     */
    DeviceLoadBalancer();

    /**
     * @brief 使用指定配置构造负载均衡器
     * @param config 异构计算配置参数
     */
    explicit DeviceLoadBalancer(const HeterogeneousComputeConfig& config);

    /**
     * @brief 析构函数
     */
    ~DeviceLoadBalancer();

    // ---- 禁止拷贝（包含互斥锁等不可拷贝成员） ----
    DeviceLoadBalancer(const DeviceLoadBalancer&) = delete;
    DeviceLoadBalancer& operator=(const DeviceLoadBalancer&) = delete;

    // ========================================================================
    // 设备负载快照获取
    // ========================================================================

    /**
     * @brief 获取CPU当前负载快照
     *
     * 通过系统API（Windows: GetSystemTimes, Linux: /proc/stat）计算CPU利用率。
     * 同时通过 GlobalMemoryStatusEx (Windows) 或 sysconf (Linux) 获取可用内存。
     * 读取当前CPU任务队列深度。
     *
     * @return DeviceLoadSnapshot CPU当前负载快照
     */
    DeviceLoadSnapshot get_cpu_load();

    /**
     * @brief 获取GPU当前负载快照
     *
     * CUDA环境下通过NVML API获取GPU使用率和显存信息。
     * 非CUDA环境下降级返回零值快照。
     * 读取当前GPU任务队列深度。
     *
     * @param device_id CUDA设备索引（默认0，即第一块GPU）
     * @return DeviceLoadSnapshot GPU当前负载快照
     */
#if SPACEGO_CUDA_ENABLED
    DeviceLoadSnapshot get_gpu_load(int device_id = 0);
#else
    DeviceLoadSnapshot get_gpu_load(int device_id = 0);
#endif

    // ========================================================================
    // 队列深度管理（由调度引擎调用，线程安全）
    // ========================================================================

    /**
     * @brief 获取当前CPU任务队列深度
     * @return CPU队列中等待执行的任务数量
     */
    size_t get_cpu_queue_depth() const { return cpu_queue_depth_.load(std::memory_order_acquire); }

    /**
     * @brief 获取当前GPU任务队列深度
     * @return GPU队列中等待执行的任务数量
     */
    size_t get_gpu_queue_depth() const { return gpu_queue_depth_.load(std::memory_order_acquire); }

    /**
     * @brief 更新CPU任务队列深度（由调度引擎调用）
     * @param depth 新的队列深度值
     */
    void update_cpu_queue_depth(size_t depth) { cpu_queue_depth_.store(depth, std::memory_order_release); }

    /**
     * @brief 更新GPU任务队列深度（由调度引擎调用）
     * @param depth 新的队列深度值
     */
    void update_gpu_queue_depth(size_t depth) { gpu_queue_depth_.store(depth, std::memory_order_release); }

    // ========================================================================
    // 负载均衡核心决策
    // ========================================================================

    /**
     * @brief 计算动态CPU/GPU任务分配比例
     *
     * 核心算法流程：
     *   1. 获取CPU和GPU当前负载快照
     *   2. 使用EMA更新累积利用率
     *   3. 比较GPU和CPU的EMA利用率：
     *      - GPU_util > overload_threshold(0.85) 且 CPU_util < idle_threshold(0.40):
     *        增加CPU分配比例，减少GPU分配比例
     *      - CPU_util > overload_threshold(0.90):
     *        减少CPU分配比例，增加GPU分配比例
     *   4. 将分配比例约束在 [0.1, 0.9] 范围内（永远不给单一设备100%）
     *   5. 使用EMA平滑过渡比例变化
     *
     * @return AllocationDecision 包含新的CPU/GPU分配比例的决策
     */
    AllocationDecision calculate_allocation_ratio();

    /**
     * @brief GPU过载检测与CPU分流逻辑
     *
     * 检测GPU是否持续过载（连续3次采样利用率 > 85%阈值），
     * 如果是，则触发 should_offload_to_cpu=true，
     * 将部分GPU任务卸载到CPU执行。
     *
     * 分流不会立即改变分配比例，而是发出一个信号，
     * 由调度引擎根据此信号决定是否将当前排队的GPU任务重定向到CPU。
     *
     * @return AllocationDecision 决策结果，包含分流建议
     */
    AllocationDecision check_gpu_overload();

    /**
     * @brief CPU过载保护与GPU回路由逻辑
     *
     * 检测CPU是否持续过载（连续3次采样利用率 > 90%阈值），
     * 如果是，则触发 should_route_to_gpu=true，
     * 建议调度引擎将部分任务从CPU转移到GPU。
     *
     * 回路由策略是保护性的：当CPU不堪重负时，
     * 即使任务更适合CPU执行，也应考虑GPU作为替代。
     *
     * @return AllocationDecision 决策结果，包含回路由建议
     */
    AllocationDecision check_cpu_overload();

    // ========================================================================
    // 配置管理
    // ========================================================================

    /**
     * @brief 更新异构计算配置（线程安全）
     *
     * 可在运行时动态调整负载均衡参数。
     * 更新后下一次采样将使用新的阈值和参数。
     *
     * @param config 新的配置参数
     */
    void update_config(const HeterogeneousComputeConfig& config);

    /**
     * @brief 获取当前使用的配置（只读）
     * @return 当前配置的常量引用
     */
    const HeterogeneousComputeConfig& get_config() const { return config_; }

    // ========================================================================
    // 历史负载数据（用于诊断和性能分析）
    // ========================================================================

    /**
     * @brief 获取CPU负载历史快照列表
     *
     * 返回最近最多 MAX_LOAD_HISTORY 个CPU负载快照。
     * 用于性能诊断和可视化分析。
     *
     * @return 历史快照的副本列表（按时间顺序排列）
     */
    std::vector<DeviceLoadSnapshot> get_cpu_load_history() const;

    /**
     * @brief 获取GPU负载历史快照列表
     *
     * 返回最近最多 MAX_LOAD_HISTORY 个GPU负载快照。
     * 用于性能诊断和可视化分析。
     *
     * @return 历史快照的副本列表（按时间顺序排列）
     */
    std::vector<DeviceLoadSnapshot> get_gpu_load_history() const;

private:
    // ========================================================================
    // 配置与状态
    // ========================================================================

    HeterogeneousComputeConfig config_;          // 异构计算配置参数
    mutable std::mutex         config_mutex_;    // 配置访问互斥锁

    // ---- CPU利用率EMA（指数移动平均） ----
    double cpu_utilization_ema_;    // CPU利用率的EMA平滑值 (0.0 ~ 1.0)
                                    //   初始值设为0.0（假设初始空闲）
    double cpu_allocation_ema_;     // CPU分配比例的EMA平滑值 (0.1 ~ 0.9)
                                    //   避免比例突变导致调度震荡

    // ---- GPU利用率EMA ----
    double gpu_utilization_ema_;    // GPU利用率的EMA平滑值 (0.0 ~ 1.0)
                                    //   初始值设为0.0（假设初始空闲）
    double gpu_allocation_ema_;     // GPU分配比例的EMA平滑值 (0.1 ~ 0.9)

    // ---- 当前分配比例（未经EMA平滑的即时值） ----
    double current_cpu_ratio_;      // 当前CPU任务分配比例
    double current_gpu_ratio_;      // 当前GPU任务分配比例 (恒为 1.0 - cpu_ratio)

    // ---- 任务队列深度（原子操作，线程安全） ----
    std::atomic<size_t> cpu_queue_depth_;  // CPU任务队列当前深度
    std::atomic<size_t> gpu_queue_depth_;  // GPU任务队列当前深度

    // ---- 连续过载计数器 ----
    int consecutive_gpu_overload_count_;   // GPU连续过载采样次数
                                           //   用于判断是否触发CPU分流
    int consecutive_cpu_overload_count_;   // CPU连续过载采样次数
                                           //   用于判断是否触发GPU回流

    // ---- 历史负载记录 ----
    static constexpr size_t MAX_LOAD_HISTORY = 64;  // 最大历史记录数
    std::vector<DeviceLoadSnapshot> cpu_load_history_;  // CPU负载历史
    std::vector<DeviceLoadSnapshot> gpu_load_history_;  // GPU负载历史
    mutable std::mutex               history_mutex_;    // 历史记录互斥锁

    // ---- 最后监控时间 ----
    std::chrono::system_clock::time_point last_monitor_time_;  // 上次监控时间
                                                                 // 用于判断是否达到监控间隔

    // ---- CPU利用率采样状态（用于Windows GetSystemTimes差值计算） ----
#ifdef _WIN32
    // Windows下使用GetSystemTimes()需要前后两次采样计算差值
    // 存储上次采样的原始FILETIME值，转换为uint64_t便于计算差值
    uint64_t prev_idle_time_;    // 上次采样的系统空闲时间（100ns单位）
    uint64_t prev_kernel_time_;  // 上次采样的内核态时间（100ns单位）
    uint64_t prev_user_time_;    // 上次采样的用户态时间（100ns单位）
    bool     cpu_sample_valid_;  // 是否有有效的上次采样数据
#else
    // Linux下使用/proc/stat读取CPU时间统计
    // 存储上次采样的各CPU时间值
    uint64_t prev_total_time_;   // 上次采样的总CPU时间
    uint64_t prev_idle_time_;    // 上次采样的空闲CPU时间
    bool     cpu_sample_valid_;  // 是否有有效的上次采样数据
#endif

    // ---- GPU利用率采样状态（NVML相关） ----
#if SPACEGO_CUDA_ENABLED
    // NVML初始化状态（延迟初始化，首次调用时初始化）
    bool     nvml_initialized_;   // NVML是否已成功初始化
    void*    nvml_device_handle_; // NVML设备句柄（void*避免暴露NVML类型）
                                  //   对应 nvmlDevice_t
#endif

    // ========================================================================
    // 内部方法
    // ========================================================================

    // ---- 设备利用率计算 ----

    /**
     * @brief 通过系统API计算CPU当前利用率
     *
     * Windows实现：使用 GetSystemTimes() 获取空闲/内核/用户时间，
     *   与上次采样的差值计算利用率：util = 1 - idle_delta / total_delta。
     *   首次调用返回0.0（无历史数据可用于差值计算）。
     *
     * Linux实现：读取 /proc/stat 文件，解析CPU时间统计行。
     *
     * @return CPU利用率 (0.0 ~ 1.0)，首次调用返回0.0
     */
    double calculate_cpu_utilization();

#if SPACEGO_CUDA_ENABLED
    /**
     * @brief 通过NVML API计算GPU当前利用率
     *
     * 使用 nvmlDeviceGetUtilizationRates() 获取GPU的SM（流多处理器）
     * 使用率。如果NVML初始化失败，回退到基于队列深度的启发式估算。
     *
     * @param device_id CUDA设备索引（默认0）
     * @return GPU利用率 (0.0 ~ 1.0)，不可用时返回0.0
     */
    double calculate_gpu_utilization(int device_id = 0);
#endif

    // ========================================================================
    // 内部辅助方法
    // ========================================================================

    /**
     * @brief 执行指数移动平均（EMA）更新
     *
     * 公式: new_ema = alpha * new_value + (1 - alpha) * old_ema
     *
     * alpha值越大，对新数据的响应越快（越敏感）；
     * alpha值越小，平滑效果越好（越稳定）。
     * 默认使用 config_.load_balance_smooth_factor (0.3)。
     *
     * @param current_ema 当前的EMA值
     * @param new_value   新采样的原始值
     * @param alpha       EMA平滑因子 (0.0 ~ 1.0)
     * @return 更新后的EMA值
     */
    double update_ema(double current_ema, double new_value, double alpha);

    /**
     * @brief 判断是否已达到监控采样间隔
     *
     * 比较当前时间与上次监控时间，判断是否超过了
     * config_.monitoring_interval_ms 设定的间隔。
     * 如果尚未到达间隔，返回false避免过于频繁的系统调用。
     *
     * @return true 表示应该进行新的采样
     */
    bool should_monitor() const;

    /**
     * @brief 记录负载快照到历史列表
     *
     * 将新的快照添加到历史列表末尾。
     * 如果历史记录数超过 MAX_LOAD_HISTORY，移除最旧的记录（循环队列）。
     *
     * @param snapshot 新的负载快照
     * @param history  目标历史列表（cpu_load_history_ 或 gpu_load_history_）
     */
    void record_load_snapshot(const DeviceLoadSnapshot& snapshot,
                              std::vector<DeviceLoadSnapshot>& history);

    /**
     * @brief 将分配比例约束在安全范围内
     *
     * 确保分配比例在 [0.1, 0.9] 范围内。
     * 永远不给单一设备100%的任务分配，保留一定弹性。
     *
     * @param ratio 原始分配比例
     * @return 约束后的分配比例
     */
    static double clamp_ratio(double ratio);

    /**
     * @brief 获取当前系统可用内存（MB）
     *
     * Windows: 使用 GlobalMemoryStatusEx
     * Linux: 使用 sysconf(_SC_AVPHYS_PAGES) * sysconf(_SC_PAGESIZE)
     *
     * @return 可用内存大小（MB）
     */
    size_t get_available_system_memory_mb();

#if SPACEGO_CUDA_ENABLED
    /**
     * @brief 尝试初始化NVML库
     *
     * 在首次调用GPU相关功能时延迟初始化。
     * 初始化成功后设置 nvml_initialized_ = true 并获取设备句柄。
     * 失败时记录警告并设置 nvml_initialized_ = false。
     *
     * @param device_id CUDA设备索引
     * @return true 初始化成功
     */
    bool initialize_nvml(int device_id);
#endif

    // ---- 补偿步长常量 ----
    static constexpr double COMPENSATION_DELTA = 0.05;  // 每次调整分配比例的步长
                                                        // 较大的步长响应更快但可能震荡
                                                        // 较小的步长更稳定但响应慢
};

}  // namespace heterogeneous
}  // namespace go
}  // namespace spacego