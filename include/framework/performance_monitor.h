#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>
#include <functional>
#include <unordered_map>
#include <queue>

#ifdef _WIN32
#include <windows.h>
#include <pdh.h>
#include <pdhmsg.h>
#endif

// NVML头文件（NVIDIA GPU监控）
#ifdef USE_NVML
#include <nvml.h>
#endif

#include "framework/logger.h"

#ifdef USE_CUDA
#include "go/search/hmgs_dp_cuda.h"
#endif

namespace spacego {
namespace framework {

/**
 * @brief 增强型性能监控器
 * 
 * 此类扩展了HMGSDP_CUDA_Monitor，提供全面的系统性能监控功能。
 * 监控范围包括：CPU使用率、内存占用、GPU显存、温度、功耗等。
 * 支持跨平台（Windows/Linux），并与Logger系统深度集成。
 * 
 * 主要功能：
 * 1. 系统级性能监控：CPU、内存、磁盘IO、网络
 * 2. GPU监控：显存使用、温度、功耗、利用率
 * 3. 进程级监控：当前进程的资源使用情况
 * 4. 性能警报系统：基于阈值触发警报
 * 5. 历史数据记录和趋势分析
 * 6. 性能报告生成（文本、JSON、HTML格式）
 * 7. 与Logger系统集成，自动记录性能数据
 * 
 * 使用示例：
 * @code
 * // 创建监控器
 * EnhancedPerformanceMonitor monitor;
 * monitor.set_monitoring_interval(1000);  // 1秒采样间隔
 * monitor.enable_gpu_monitoring(true);    // 启用GPU监控
 * 
 * // 开始监控
 * monitor.start_monitoring();
 * 
 * // 运行程序...
 * 
 * // 获取当前性能状态
 * auto status = monitor.get_current_status();
 * Logger::instance().info("CPU使用率: {:.1f}%", status.cpu_usage_percent);
 * 
 * // 停止监控并生成报告
 * monitor.stop_monitoring();
 * monitor.generate_report("performance_report.html");
 * @endcode
 */
class EnhancedPerformanceMonitor {
public:
    /// 监控配置
    struct MonitorConfig {
        int sampling_interval_ms = 1000;           ///< 采样间隔（毫秒）
        bool log_to_file = true;                   ///< 是否记录到文件
        std::string log_file_path = "performance.log";  ///< 日志文件路径
        bool enable_alerts = true;                 ///< 是否启用警报系统
        bool enable_cpu_monitoring = true;         ///< 是否启用CPU监控
        bool enable_memory_monitoring = true;      ///< 是否启用内存监控
        bool enable_gpu_monitoring = true;         ///< 是否启用GPU监控
        bool enable_disk_monitoring = false;       ///< 是否启用磁盘监控
        bool enable_network_monitoring = false;    ///< 是否启用网络监控
        bool monitor_process_only = false;         ///< 是否仅监控当前进程
        
        // 警报阈值
        double cpu_alert_threshold = 90.0;         ///< CPU使用率警报阈值（%）
        double memory_alert_threshold = 85.0;      ///< 内存使用率警报阈值（%）
        double gpu_memory_alert_threshold = 85.0;  ///< GPU显存使用率警报阈值（%）
        double gpu_temperature_alert = 85.0;       ///< GPU温度警报阈值（摄氏度）
        
        // 日志配置
        LogLevel log_level = LogLevel::INFO;       ///< 日志级别
        std::string log_category = "PerformanceMonitor";  ///< 日志分类
        bool log_summary_periodically = true;      ///< 是否定期记录性能摘要
        int summary_log_interval_sec = 60;         ///< 性能摘要记录间隔（秒）
    };

    /// 性能数据（单次采样）
    struct PerformanceData {
        std::chrono::system_clock::time_point timestamp;  ///< 时间戳
        
        // 系统级指标
        double cpu_usage_percent = 0.0;            ///< 系统CPU使用率（%）
        double memory_usage_percent = 0.0;         ///< 系统内存使用率（%）
        uint64_t total_memory_bytes = 0;           ///< 总内存（字节）
        uint64_t used_memory_bytes = 0;            ///< 已使用内存（字节）
        uint64_t available_memory_bytes = 0;       ///< 可用内存（字节）
        
        // 进程级指标
        double process_cpu_usage_percent = 0.0;    ///< 当前进程CPU使用率（%）
        uint64_t process_memory_bytes = 0;         ///< 当前进程内存使用（字节）
        uint64_t process_private_bytes = 0;        ///< 当前进程私有内存（字节）
        uint64_t process_working_set_bytes = 0;    ///< 当前进程工作集（字节）
        uint64_t process_handle_count = 0;         ///< 当前进程句柄数
        uint64_t process_thread_count = 0;         ///< 当前进程线程数
        
        // GPU指标（如果可用）
        struct GPUData {
            uint32_t gpu_index = 0;                ///< GPU索引
            std::string gpu_name;                  ///< GPU名称
            double utilization_percent = 0.0;      ///< GPU利用率（%）
            uint64_t total_memory_bytes = 0;       ///< GPU总显存（字节）
            uint64_t used_memory_bytes = 0;        ///< GPU已用显存（字节）
            double memory_usage_percent = 0.0;     ///< GPU显存使用率（%）
            double temperature_celsius = 0.0;      ///< GPU温度（摄氏度）
            double power_usage_watts = 0.0;        ///< GPU功耗（瓦特）
            uint64_t fan_speed_percent = 0;        ///< 风扇转速（%）
            uint64_t clock_speed_mhz = 0;          ///< 时钟频率（MHz）
        };
        std::vector<GPUData> gpu_data;             ///< 多个GPU的数据
        
        // 磁盘指标（如果启用）
        struct DiskData {
            std::string disk_name;                 ///< 磁盘名称
            double read_speed_mbps = 0.0;          ///< 读取速度（MB/s）
            double write_speed_mbps = 0.0;         ///< 写入速度（MB/s）
            uint64_t read_bytes_total = 0;         ///< 总读取字节数
            uint64_t write_bytes_total = 0;        ///< 总写入字节数
            double utilization_percent = 0.0;      ///< 磁盘利用率（%）
        };
        std::vector<DiskData> disk_data;           ///< 多个磁盘的数据
        
        // 网络指标（如果启用）
        struct NetworkData {
            std::string interface_name;            ///< 网络接口名称
            double receive_speed_mbps = 0.0;       ///< 接收速度（MB/s）
            double transmit_speed_mbps = 0.0;      ///< 发送速度（MB/s）
            uint64_t receive_bytes_total = 0;      ///< 总接收字节数
            uint64_t transmit_bytes_total = 0;     ///< 总发送字节数
        };
        std::vector<NetworkData> network_data;     ///< 多个网络接口的数据
        
        // 自定义应用程序指标
        std::unordered_map<std::string, double> custom_metrics;
        
        // 转换为字符串
        std::string to_string() const;
        std::string to_json() const;
        std::string to_csv() const;
        
        // 获取摘要信息
        std::string get_summary() const;
    };

    /// 性能警报
    struct PerformanceAlert {
        enum class AlertType {
            CPU_HIGH_USAGE,                ///< CPU使用率过高
            MEMORY_HIGH_USAGE,             ///< 内存使用率过高
            GPU_MEMORY_HIGH_USAGE,         ///< GPU显存使用率过高
            GPU_TEMPERATURE_HIGH,          ///< GPU温度过高
            GPU_POWER_HIGH,                ///< GPU功耗过高
            DISK_HIGH_USAGE,               ///< 磁盘使用率过高
            NETWORK_HIGH_USAGE,            ///< 网络使用率过高
            PROCESS_MEMORY_HIGH,           ///< 进程内存使用过高
            CUSTOM_METRIC_THRESHOLD        ///< 自定义指标阈值
        };
        
        AlertType type;                     ///< 警报类型
        std::string message;                ///< 警报消息
        double severity = 0.0;              ///< 严重程度（0-1）
        std::chrono::system_clock::time_point timestamp;  ///< 时间戳
        PerformanceData trigger_data;       ///< 触发警报时的性能数据
        std::vector<std::string> suggestions;  ///< 优化建议
        
        std::string to_string() const;
    };

    /// 性能报告
    struct PerformanceReport {
        std::chrono::system_clock::time_point start_time;  ///< 监控开始时间
        std::chrono::system_clock::time_point end_time;    ///< 监控结束时间
        std::vector<PerformanceData> history_data;         ///< 历史数据
        std::vector<PerformanceAlert> alerts_triggered;    ///< 触发的警报
        PerformanceData average_performance;               ///< 平均性能数据
        PerformanceData peak_performance;                  ///< 峰值性能数据
        PerformanceData minimum_performance;               ///< 最低性能数据
        
        struct StatisticalSummary {
            double cpu_usage_avg = 0.0;
            double cpu_usage_max = 0.0;
            double memory_usage_avg = 0.0;
            double memory_usage_max = 0.0;
            double gpu_memory_usage_avg = 0.0;
            double gpu_memory_usage_max = 0.0;
            // 更多统计指标...
        };
        StatisticalSummary stats_summary;
        
        std::string to_string() const;
        std::string to_json() const;
        std::string to_html() const;
    };

    EnhancedPerformanceMonitor();
    explicit EnhancedPerformanceMonitor(const MonitorConfig& config);
    ~EnhancedPerformanceMonitor();

    // 禁止拷贝和赋值
    EnhancedPerformanceMonitor(const EnhancedPerformanceMonitor&) = delete;
    EnhancedPerformanceMonitor& operator=(const EnhancedPerformanceMonitor&) = delete;

    /// 开始性能监控
    bool start_monitoring();

    /// 停止性能监控
    void stop_monitoring();

    /// 暂停监控
    void pause_monitoring();

    /// 恢复监控
    void resume_monitoring();

    /// 获取当前是否正在监控
    bool is_monitoring() const;

    /// 获取当前性能状态
    PerformanceData get_current_status() const;

    /// 获取历史性能数据
    std::vector<PerformanceData> get_history_data(int max_samples = 0) const;

    /// 获取触发的警报列表
    std::vector<PerformanceAlert> get_active_alerts() const;

    /// 确认警报（标记为已处理）
    void acknowledge_alert(size_t alert_index);

    /// 添加自定义性能指标
    void add_custom_metric(const std::string& name, double value);

    /// 设置自定义指标阈值
    void set_custom_metric_threshold(const std::string& name, double threshold, 
                                     PerformanceAlert::AlertType alert_type = PerformanceAlert::AlertType::CUSTOM_METRIC_THRESHOLD);

    /// 生成性能报告
    PerformanceReport generate_report() const;

    /// 导出报告到文件
    bool export_report_to_file(const std::string& file_path, const std::string& format = "html");

    /// 设置监控配置
    void set_config(const MonitorConfig& config);

    /// 获取监控配置
    MonitorConfig get_config() const;

    /// 重置监控数据
    void reset_monitoring_data();

    /// 静态方法：获取系统信息
    static std::string get_system_info();

    /// 静态方法：获取GPU信息
    static std::vector<PerformanceData::GPUData> get_gpu_info();

private:
    /// 配置
    MonitorConfig config_;
    mutable std::mutex config_mutex_;

    /// 监控状态
    std::atomic<bool> monitoring_active_;
    std::atomic<bool> monitoring_paused_;
    std::thread monitoring_thread_;
    std::chrono::system_clock::time_point monitoring_start_time_;

    /// 数据存储
    std::vector<PerformanceData> history_data_;
    mutable std::mutex data_mutex_;
    const size_t max_history_size_ = 10000;  ///< 最大历史记录数

    /// 警报系统
    std::vector<PerformanceAlert> active_alerts_;
    mutable std::mutex alerts_mutex_;

    /// 自定义指标
    std::unordered_map<std::string, double> custom_metrics_;
    std::unordered_map<std::string, std::pair<double, PerformanceAlert::AlertType>> custom_thresholds_;
    mutable std::mutex metrics_mutex_;

    /// 平台特定实现
#ifdef _WIN32
    // Windows性能计数器
    PDH_HQUERY cpu_query_ = NULL;
    PDH_HCOUNTER cpu_counter_ = NULL;
    PDH_HQUERY memory_query_ = NULL;
    PDH_HCOUNTER memory_counter_ = NULL;
    
    // 进程信息
    HANDLE process_handle_ = INVALID_HANDLE_VALUE;
    FILETIME last_cpu_time_ = {0, 0};
    FILETIME last_sys_time_ = {0, 0};
    FILETIME last_user_time_ = {0, 0};
#endif

#ifdef USE_NVML
    // NVIDIA NVML
    bool nvml_initialized_ = false;
    std::vector<nvmlDevice_t> nvml_devices_;
#endif

    /// 私有方法
    void monitoring_loop();
    PerformanceData collect_performance_data();
    void analyze_performance_data(const PerformanceData& data);
    void check_for_alerts(const PerformanceData& data);
    void log_performance_data(const PerformanceData& data);
    void log_performance_summary();
    
    /// 平台特定的数据收集方法
    PerformanceData collect_windows_performance_data();
    PerformanceData collect_linux_performance_data();  // 占位符，未来实现
    
    /// GPU监控
    void initialize_gpu_monitoring();
    void cleanup_gpu_monitoring();
    std::vector<PerformanceData::GPUData> collect_gpu_data();
    
    /// 进程监控
    void initialize_process_monitoring();
    void cleanup_process_monitoring();
    PerformanceData::GPUData collect_gpu_data_nvml(int gpu_index);
    
    /// 工具方法
    static double calculate_cpu_usage(const FILETIME& prev_idle, const FILETIME& prev_kernel, 
                                     const FILETIME& prev_user, const FILETIME& curr_idle, 
                                     const FILETIME& curr_kernel, const FILETIME& curr_user);
    static uint64_t get_process_memory_usage();
    static uint64_t get_system_memory_usage();
    
    /// 警报生成
    void generate_cpu_alert(const PerformanceData& data);
    void generate_memory_alert(const PerformanceData& data);
    void generate_gpu_alerts(const PerformanceData& data);
    void generate_custom_metric_alerts(const PerformanceData& data);
    
    /// 报告生成辅助方法
    PerformanceData calculate_average_performance() const;
    PerformanceData calculate_peak_performance() const;
    PerformanceData calculate_minimum_performance() const;
};

using PerformanceMonitor = EnhancedPerformanceMonitor;

} // namespace framework
} // namespace spacego