#include "framework/performance_monitor.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <numeric>
#include <cmath>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <tlhelp32.h>
#pragma comment(lib, "pdh.lib")
#endif

namespace spacego {
namespace framework {

// ============================================================================
// EnhancedPerformanceMonitor::PerformanceData 方法实现
// ============================================================================

std::string EnhancedPerformanceMonitor::PerformanceData::to_string() const {
    std::ostringstream oss;
    oss << "时间戳: " << std::chrono::system_clock::to_time_t(timestamp) << "\n";
    oss << "系统CPU使用率: " << std::fixed << std::setprecision(1) << cpu_usage_percent << "%\n";
    oss << "系统内存使用率: " << memory_usage_percent << "%\n";
    oss << "总内存: " << total_memory_bytes / (1024 * 1024) << " MB\n";
    oss << "已使用内存: " << used_memory_bytes / (1024 * 1024) << " MB\n";
    oss << "可用内存: " << available_memory_bytes / (1024 * 1024) << " MB\n";
    
    if (!gpu_data.empty()) {
        oss << "GPU数量: " << gpu_data.size() << "\n";
        for (size_t i = 0; i < gpu_data.size(); ++i) {
            const auto& gpu = gpu_data[i];
            oss << "  GPU[" << i << "]: " << gpu.gpu_name << "\n";
            oss << "    利用率: " << gpu.utilization_percent << "%\n";
            oss << "    显存使用: " << gpu.used_memory_bytes / (1024 * 1024) << " MB / " 
                << gpu.total_memory_bytes / (1024 * 1024) << " MB (" 
                << gpu.memory_usage_percent << "%)\n";
            oss << "    温度: " << gpu.temperature_celsius << "°C\n";
        }
    }
    
    return oss.str();
}

std::string EnhancedPerformanceMonitor::PerformanceData::to_json() const {
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"timestamp\": " << std::chrono::system_clock::to_time_t(timestamp) << ",\n";
    oss << "  \"cpu_usage_percent\": " << cpu_usage_percent << ",\n";
    oss << "  \"memory_usage_percent\": " << memory_usage_percent << ",\n";
    oss << "  \"total_memory_bytes\": " << total_memory_bytes << ",\n";
    oss << "  \"used_memory_bytes\": " << used_memory_bytes << ",\n";
    oss << "  \"available_memory_bytes\": " << available_memory_bytes << ",\n";
    
    if (!gpu_data.empty()) {
        oss << "  \"gpus\": [\n";
        for (size_t i = 0; i < gpu_data.size(); ++i) {
            const auto& gpu = gpu_data[i];
            oss << "    {\n";
            oss << "      \"index\": " << gpu.gpu_index << ",\n";
            oss << "      \"name\": \"" << gpu.gpu_name << "\",\n";
            oss << "      \"utilization_percent\": " << gpu.utilization_percent << ",\n";
            oss << "      \"total_memory_bytes\": " << gpu.total_memory_bytes << ",\n";
            oss << "      \"used_memory_bytes\": " << gpu.used_memory_bytes << ",\n";
            oss << "      \"memory_usage_percent\": " << gpu.memory_usage_percent << ",\n";
            oss << "      \"temperature_celsius\": " << gpu.temperature_celsius << "\n";
            oss << "    }" << (i < gpu_data.size() - 1 ? "," : "") << "\n";
        }
        oss << "  ],\n";
    }
    
    oss << "  \"process_memory_bytes\": " << process_memory_bytes << "\n";
    oss << "}";
    
    return oss.str();
}

std::string EnhancedPerformanceMonitor::PerformanceData::to_csv() const {
    std::ostringstream oss;
    oss << std::chrono::system_clock::to_time_t(timestamp) << ",";
    oss << cpu_usage_percent << ",";
    oss << memory_usage_percent << ",";
    oss << total_memory_bytes << ",";
    oss << used_memory_bytes << ",";
    oss << available_memory_bytes << ",";
    oss << process_memory_bytes;
    
    if (!gpu_data.empty()) {
        oss << "," << gpu_data[0].utilization_percent;
        oss << "," << gpu_data[0].memory_usage_percent;
        oss << "," << gpu_data[0].temperature_celsius;
    } else {
        oss << ",0,0,0";
    }
    
    return oss.str();
}

std::string EnhancedPerformanceMonitor::PerformanceData::get_summary() const {
    std::ostringstream oss;
    oss << "CPU: " << std::fixed << std::setprecision(1) << cpu_usage_percent << "% | ";
    oss << "内存: " << memory_usage_percent << "% | ";
    oss << "进程内存: " << process_memory_bytes / (1024 * 1024) << " MB";
    
    if (!gpu_data.empty()) {
        const auto& gpu = gpu_data[0];
        oss << " | GPU: " << gpu.utilization_percent << "%";
        oss << " (" << gpu.memory_usage_percent << "%)";
        oss << " " << gpu.temperature_celsius << "°C";
    }
    
    return oss.str();
}

// ============================================================================
// PerformanceAlert 方法实现
// ============================================================================

std::string EnhancedPerformanceMonitor::PerformanceAlert::to_string() const {
    std::ostringstream oss;
    
    std::string type_str;
    switch (type) {
        case PerformanceAlert::AlertType::CPU_HIGH_USAGE: type_str = "CPU 使用率过高"; break;
        case PerformanceAlert::AlertType::MEMORY_HIGH_USAGE: type_str = "内存使用率过高"; break;
        case PerformanceAlert::AlertType::GPU_MEMORY_HIGH_USAGE: type_str = "GPU 显存使用率过高"; break;
        case PerformanceAlert::AlertType::GPU_TEMPERATURE_HIGH: type_str = "GPU 温度过高"; break;
        case PerformanceAlert::AlertType::GPU_POWER_HIGH: type_str = "GPU 功耗过高"; break;
        case PerformanceAlert::AlertType::DISK_HIGH_USAGE: type_str = "磁盘使用率过高"; break;
        case PerformanceAlert::AlertType::NETWORK_HIGH_USAGE: type_str = "网络使用率过高"; break;
        case PerformanceAlert::AlertType::PROCESS_MEMORY_HIGH: type_str = "进程内存使用过高"; break;
        case PerformanceAlert::AlertType::CUSTOM_METRIC_THRESHOLD: type_str = "自定义指标阈值"; break;
    }
    
    std::time_t time = std::chrono::system_clock::to_time_t(timestamp);
    std::tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &time);
#else
    localtime_r(&time, &tm_buf);
#endif
    char time_str[64];
    std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", &tm_buf);
    
    oss << "[" << time_str << "] ";
    oss << "[" << type_str << "] ";
    oss << "严重程度: " << std::fixed << std::setprecision(2) << severity << " ";
    oss << "消息: " << message;
    
    if (!suggestions.empty()) {
        oss << "\n优化建议:\n";
        for (size_t i = 0; i < suggestions.size(); ++i) {
            oss << "  " << (i + 1) << ". " << suggestions[i] << "\n";
        }
    }
    
    return oss.str();
}

// ============================================================================
// PerformanceReport 方法实现
// ============================================================================

std::string EnhancedPerformanceMonitor::PerformanceReport::to_string() const {
    std::ostringstream oss;
    
    std::time_t start_time_t = std::chrono::system_clock::to_time_t(start_time);
    std::time_t end_time_t = std::chrono::system_clock::to_time_t(end_time);
    std::tm tm_buf;
    
    char start_str[64], end_str[64];
#ifdef _WIN32
    localtime_s(&tm_buf, &start_time_t);
    std::strftime(start_str, sizeof(start_str), "%Y-%m-%d %H:%M:%S", &tm_buf);
    localtime_s(&tm_buf, &end_time_t);
    std::strftime(end_str, sizeof(end_str), "%Y-%m-%d %H:%M:%S", &tm_buf);
#else
    localtime_r(&start_time_t, &tm_buf);
    std::strftime(start_str, sizeof(start_str), "%Y-%m-%d %H:%M:%S", &tm_buf);
    localtime_r(&end_time_t, &tm_buf);
    std::strftime(end_str, sizeof(end_str), "%Y-%m-%d %H:%M:%S", &tm_buf);
#endif
    
    double duration_sec = std::chrono::duration<double>(end_time - start_time).count();
    
    oss << "性能监控报告\n";
    oss << "============\n";
    oss << "监控时段: " << start_str << " 至 " << end_str << "\n";
    oss << "持续时间: " << duration_sec << " 秒\n";
    oss << "数据采样数: " << history_data.size() << "\n";
    oss << "触发警报数: " << alerts_triggered.size() << "\n\n";
    
    oss << "平均性能指标:\n";
    oss << average_performance.get_summary() << "\n\n";
    
    oss << "峰值性能指标:\n";
    oss << peak_performance.get_summary() << "\n\n";
    
    oss << "最低性能指标:\n";
    oss << minimum_performance.get_summary() << "\n\n";
    
    if (!alerts_triggered.empty()) {
        oss << "警报列表:\n";
        for (size_t i = 0; i < alerts_triggered.size(); ++i) {
            oss << "警报 #" << (i + 1) << ":\n";
            oss << alerts_triggered[i].to_string() << "\n\n";
        }
    }
    
    return oss.str();
}

std::string EnhancedPerformanceMonitor::PerformanceReport::to_json() const {
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"report\": {\n";
    
    std::time_t start_time_t = std::chrono::system_clock::to_time_t(start_time);
    std::time_t end_time_t = std::chrono::system_clock::to_time_t(end_time);
    
    oss << "    \"start_time\": " << start_time_t << ",\n";
    oss << "    \"end_time\": " << end_time_t << ",\n";
    oss << "    \"data_points\": " << history_data.size() << ",\n";
    oss << "    \"alerts_count\": " << alerts_triggered.size() << ",\n";
    
    oss << "    \"average_performance\": " << average_performance.to_json() << ",\n";
    oss << "    \"peak_performance\": " << peak_performance.to_json() << ",\n";
    oss << "    \"minimum_performance\": " << minimum_performance.to_json() << "\n";
    
    oss << "  }\n";
    oss << "}";
    
    return oss.str();
}

std::string EnhancedPerformanceMonitor::PerformanceReport::to_html() const {
    std::ostringstream oss;
    
    std::time_t start_time_t = std::chrono::system_clock::to_time_t(start_time);
    std::time_t end_time_t = std::chrono::system_clock::to_time_t(end_time);
    std::tm tm_buf;
    
    char start_str[64], end_str[64];
#ifdef _WIN32
    localtime_s(&tm_buf, &start_time_t);
    std::strftime(start_str, sizeof(start_str), "%Y-%m-%d %H:%M:%S", &tm_buf);
    localtime_s(&tm_buf, &end_time_t);
    std::strftime(end_str, sizeof(end_str), "%Y-%m-%d %H:%M:%S", &tm_buf);
#else
    localtime_r(&start_time_t, &tm_buf);
    std::strftime(start_str, sizeof(start_str), "%Y-%m-%d %H:%M:%S", &tm_buf);
    localtime_r(&end_time_t, &tm_buf);
    std::strftime(end_str, sizeof(end_str), "%Y-%m-%d %H:%M:%S", &tm_buf);
#endif
    
    double duration_sec = std::chrono::duration<double>(end_time - start_time).count();
    
    oss << "<!DOCTYPE html>\n";
    oss << "<html lang=\"zh-CN\">\n";
    oss << "<head>\n";
    oss << "    <meta charset=\"UTF-8\">\n";
    oss << "    <title>性能监控报告</title>\n";
    oss << "    <style>\n";
    oss << "        body { font-family: Arial, sans-serif; margin: 40px; }\n";
    oss << "        h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }\n";
    oss << "        h2 { color: #555; margin-top: 30px; }\n";
    oss << "        .summary { background: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 20px; }\n";
    oss << "        .metric { display: inline-block; margin-right: 30px; margin-bottom: 10px; }\n";
    oss << "        .metric-label { font-weight: bold; color: #666; }\n";
    oss << "        .metric-value { font-size: 18px; color: #2196F3; }\n";
    oss << "        .alert { background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin-bottom: 10px; }\n";
    oss << "        .alert-high { background: #f8d7da; border-left-color: #dc3545; }\n";
    oss << "        .suggestion { margin-left: 20px; color: #666; font-size: 14px; }\n";
    oss << "        table { border-collapse: collapse; width: 100%; margin-top: 20px; }\n";
    oss << "        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }\n";
    oss << "        th { background-color: #4CAF50; color: white; }\n";
    oss << "        tr:nth-child(even) { background-color: #f2f2f2; }\n";
    oss << "    </style>\n";
    oss << "</head>\n";
    oss << "<body>\n";
    oss << "    <h1>性能监控报告</h1>\n";
    
    oss << "    <div class=\"summary\">\n";
    oss << "        <div class=\"metric\">\n";
    oss << "            <div class=\"metric-label\">监控时段</div>\n";
    oss << "            <div class=\"metric-value\">" << start_str << " 至 " << end_str << "</div>\n";
    oss << "        </div>\n";
    oss << "        <div class=\"metric\">\n";
    oss << "            <div class=\"metric-label\">持续时间</div>\n";
    oss << "            <div class=\"metric-value\">" << std::fixed << std::setprecision(1) << duration_sec << " 秒</div>\n";
    oss << "        </div>\n";
    oss << "        <div class=\"metric\">\n";
    oss << "            <div class=\"metric-label\">数据采样数</div>\n";
    oss << "            <div class=\"metric-value\">" << history_data.size() << "</div>\n";
    oss << "        </div>\n";
    oss << "        <div class=\"metric\">\n";
    oss << "            <div class=\"metric-label\">触发警报数</div>\n";
    oss << "            <div class=\"metric-value\">" << alerts_triggered.size() << "</div>\n";
    oss << "        </div>\n";
    oss << "    </div>\n";
    
    oss << "    <h2>性能摘要</h2>\n";
    oss << "    <div class=\"summary\">\n";
    oss << "        <h3>平均性能</h3>\n";
    oss << "        <p>" << average_performance.get_summary() << "</p>\n";
    
    oss << "        <h3>峰值性能</h3>\n";
    oss << "        <p>" << peak_performance.get_summary() << "</p>\n";
    
    oss << "        <h3>最低性能</h3>\n";
    oss << "        <p>" << minimum_performance.get_summary() << "</p>\n";
    oss << "    </div>\n";
    
    if (!alerts_triggered.empty()) {
        oss << "    <h2>性能警报</h2>\n";
        for (size_t i = 0; i < alerts_triggered.size(); ++i) {
            const auto& alert = alerts_triggered[i];
            std::string alert_class = alert.severity > 0.7 ? "alert alert-high" : "alert";
            
            std::time_t alert_time = std::chrono::system_clock::to_time_t(alert.timestamp);
            char alert_time_str[64];
#ifdef _WIN32
            localtime_s(&tm_buf, &alert_time);
            std::strftime(alert_time_str, sizeof(alert_time_str), "%Y-%m-%d %H:%M:%S", &tm_buf);
#else
            localtime_r(&alert_time, &tm_buf);
            std::strftime(alert_time_str, sizeof(alert_time_str), "%Y-%m-%d %H:%M:%S", &tm_buf);
#endif
            
            oss << "    <div class=\"" << alert_class << "\">\n";
            oss << "        <strong>" << alert_time_str << " - ";
            
            switch (alert.type) {
                case PerformanceAlert::AlertType::CPU_HIGH_USAGE: 
                    oss << "CPU使用率过高"; break;
                case PerformanceAlert::AlertType::MEMORY_HIGH_USAGE: 
                    oss << "内存使用率过高"; break;
                case PerformanceAlert::AlertType::GPU_MEMORY_HIGH_USAGE: 
                    oss << "GPU显存使用率过高"; break;
                case PerformanceAlert::AlertType::GPU_TEMPERATURE_HIGH: 
                    oss << "GPU温度过高"; break;
                default: 
                    oss << "性能警报"; break;
            }
            
            oss << "</strong><br>\n";
            oss << "        " << alert.message << "<br>\n";
            oss << "        严重程度: " << std::fixed << std::setprecision(2) << alert.severity << "\n";
            
            if (!alert.suggestions.empty()) {
                oss << "        <div class=\"suggestion\">\n";
                oss << "            <strong>优化建议:</strong>\n";
                oss << "            <ul>\n";
                for (const auto& suggestion : alert.suggestions) {
                    oss << "                <li>" << suggestion << "</li>\n";
                }
                oss << "            </ul>\n";
                oss << "        </div>\n";
            }
            
            oss << "    </div>\n";
        }
    }
    
    if (!history_data.empty() && history_data.size() <= 100) {
        oss << "    <h2>性能数据记录（前" << std::min(100, (int)history_data.size()) << "条）</h2>\n";
        oss << "    <table>\n";
        oss << "        <tr>\n";
        oss << "            <th>时间戳</th>\n";
        oss << "            <th>CPU使用率 (%)</th>\n";
        oss << "            <th>内存使用率 (%)</th>\n";
        oss << "            <th>进程内存 (MB)</th>\n";
        if (!history_data[0].gpu_data.empty()) {
            oss << "            <th>GPU使用率 (%)</th>\n";
            oss << "            <th>GPU显存使用率 (%)</th>\n";
            oss << "            <th>GPU温度 (°C)</th>\n";
        }
        oss << "        </tr>\n";
        
        int display_count = std::min(100, (int)history_data.size());
        for (int i = 0; i < display_count; ++i) {
            const auto& data = history_data[i];
            std::time_t data_time = std::chrono::system_clock::to_time_t(data.timestamp);
            char data_time_str[64];
#ifdef _WIN32
            localtime_s(&tm_buf, &data_time);
            std::strftime(data_time_str, sizeof(data_time_str), "%H:%M:%S", &tm_buf);
#else
            localtime_r(&data_time, &tm_buf);
            std::strftime(data_time_str, sizeof(data_time_str), "%H:%M:%S", &tm_buf);
#endif
            
            oss << "        <tr>\n";
            oss << "            <td>" << data_time_str << "</td>\n";
            oss << "            <td>" << std::fixed << std::setprecision(1) << data.cpu_usage_percent << "</td>\n";
            oss << "            <td>" << data.memory_usage_percent << "</td>\n";
            oss << "            <td>" << data.process_memory_bytes / (1024 * 1024) << "</td>\n";
            
            if (!data.gpu_data.empty()) {
                const auto& gpu = data.gpu_data[0];
                oss << "            <td>" << gpu.utilization_percent << "</td>\n";
                oss << "            <td>" << gpu.memory_usage_percent << "</td>\n";
                oss << "            <td>" << gpu.temperature_celsius << "</td>\n";
            }
            
            oss << "        </tr>\n";
        }
        
        oss << "    </table>\n";
    }
    
    oss << "    <footer style=\"margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #777; font-size: 12px;\">\n";
    oss << "        <p>报告生成时间: " << end_str << " | SpaceGo 维空弈 性能监控系统</p>\n";
    oss << "    </footer>\n";
    oss << "</body>\n";
    oss << "</html>\n";
    
    return oss.str();
}

// ============================================================================
// EnhancedPerformanceMonitor 构造函数和析构函数
// ============================================================================

EnhancedPerformanceMonitor::EnhancedPerformanceMonitor() 
    : monitoring_active_(false)
    , monitoring_paused_(false)
#ifdef _WIN32
    , cpu_query_(NULL)
    , cpu_counter_(NULL)
    , memory_query_(NULL)
    , memory_counter_(NULL)
    , process_handle_(INVALID_HANDLE_VALUE)
#endif
#ifdef USE_NVML
    , nvml_initialized_(false)
#endif
{
    config_.sampling_interval_ms = 1000;
    config_.log_to_file = true;
    config_.log_file_path = "performance.log";
    config_.enable_alerts = true;
    config_.enable_cpu_monitoring = true;
    config_.enable_memory_monitoring = true;
    config_.enable_gpu_monitoring = true;
    config_.enable_disk_monitoring = false;
    config_.enable_network_monitoring = false;
    config_.monitor_process_only = false;
    
    config_.cpu_alert_threshold = 90.0;
    config_.memory_alert_threshold = 85.0;
    config_.gpu_memory_alert_threshold = 85.0;
    config_.gpu_temperature_alert = 85.0;
    
    config_.log_level = LogLevel::INFO;
    config_.log_category = "PerformanceMonitor";
    config_.log_summary_periodically = true;
    config_.summary_log_interval_sec = 60;
    
#ifdef _WIN32
    FILETIME ftIdle, ftKernel, ftUser;
    GetSystemTimes(&ftIdle, &ftKernel, &ftUser);
    last_cpu_time_ = ftIdle;
    last_sys_time_ = ftKernel;
    last_user_time_ = ftUser;
    
    // 打开当前进程句柄
    process_handle_ = GetCurrentProcess();
#endif
    
    // 初始化GPU监控
    if (config_.enable_gpu_monitoring) {
        initialize_gpu_monitoring();
    }
}

EnhancedPerformanceMonitor::PerformanceData EnhancedPerformanceMonitor::collect_performance_data() {
    PerformanceData data;
    data.timestamp = std::chrono::system_clock::now();
    
    // 根据平台收集数据
#ifdef _WIN32
    data = collect_windows_performance_data();
#else
    // Linux实现占位符
    data = collect_linux_performance_data();
#endif
    
    // 收集GPU数据
    if (config_.enable_gpu_monitoring) {
        data.gpu_data = collect_gpu_data();
    }
    
    // 添加自定义指标
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        data.custom_metrics = custom_metrics_;
    }
    
    return data;
}

void EnhancedPerformanceMonitor::analyze_performance_data(const PerformanceData& data) {
    // 这里可以进行更复杂的分析，比如趋势检测、异常检测等
    // 目前只是记录到历史数据中
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        history_data_.push_back(data);
        
        // 限制历史数据大小
        if (history_data_.size() > max_history_size_) {
            history_data_.erase(history_data_.begin(), 
                               history_data_.begin() + (history_data_.size() - max_history_size_));
        }
    }
    
    // 记录到日志（如果启用）
    if (config_.log_to_file) {
        Logger::instance().log(config_.log_level,
                              "性能数据: " + data.get_summary(), __FILE__, __LINE__);
    }
}

void EnhancedPerformanceMonitor::check_for_alerts(const PerformanceData& data) {
    if (!config_.enable_alerts) {
        return;
    }
    
    // 检查CPU警报
    if (data.cpu_usage_percent > config_.cpu_alert_threshold) {
        generate_cpu_alert(data);
    }
    
    // 检查内存警报
    if (data.memory_usage_percent > config_.memory_alert_threshold) {
        generate_memory_alert(data);
    }
    
    // 检查GPU警报
    if (!data.gpu_data.empty()) {
        generate_gpu_alerts(data);
    }
    
    // 检查自定义指标警报
    generate_custom_metric_alerts(data);
}

void EnhancedPerformanceMonitor::log_performance_data(const PerformanceData& data) {
    // 如果启用了文件日志，将数据写入文件
    if (config_.log_to_file && !config_.log_file_path.empty()) {
        std::ofstream log_file(config_.log_file_path, std::ios::app);
        if (log_file.is_open()) {
            log_file << data.to_csv() << "\n";
            log_file.close();
        }
    }
}

void EnhancedPerformanceMonitor::log_performance_summary() {
    if (history_data_.empty()) {
        return;
    }
    
    PerformanceData avg_data = calculate_average_performance();
    PerformanceData peak_data = calculate_peak_performance();
    
    Logger::instance().info("性能摘要 - 平均: {} | 峰值: {}", 
                           avg_data.get_summary(), peak_data.get_summary());
}

EnhancedPerformanceMonitor::EnhancedPerformanceMonitor(const MonitorConfig& config)
    : config_(config)
    , monitoring_active_(false)
    , monitoring_paused_(false)
#ifdef _WIN32
    , cpu_query_(NULL)
    , cpu_counter_(NULL)
    , memory_query_(NULL)
    , memory_counter_(NULL)
    , process_handle_(INVALID_HANDLE_VALUE)
#endif
#ifdef USE_NVML
    , nvml_initialized_(false)
#endif
{
#ifdef _WIN32
    FILETIME ftIdle, ftKernel, ftUser;
    GetSystemTimes(&ftIdle, &ftKernel, &ftUser);
    last_cpu_time_ = ftIdle;
    last_sys_time_ = ftKernel;
    last_user_time_ = ftUser;
    
    process_handle_ = GetCurrentProcess();
#endif
    
    if (config_.enable_gpu_monitoring) {
        initialize_gpu_monitoring();
    }
}

EnhancedPerformanceMonitor::~EnhancedPerformanceMonitor() {
    stop_monitoring();
    cleanup_gpu_monitoring();
    
#ifdef _WIN32
    if (cpu_query_ != NULL) {
        PdhCloseQuery(cpu_query_);
    }
    if (memory_query_ != NULL) {
        PdhCloseQuery(memory_query_);
    }
    
    // 注意：GetCurrentProcess()返回的是伪句柄，不应该被CloseHandle关闭
    // 如果是真实句柄才需要关闭
    if (process_handle_ != INVALID_HANDLE_VALUE && process_handle_ != GetCurrentProcess()) {
        CloseHandle(process_handle_);
    }
#endif
}

// ============================================================================
// 公共方法实现
// ============================================================================

bool EnhancedPerformanceMonitor::start_monitoring() {
    if (monitoring_active_) {
        Logger::instance().warning("性能监控器已在运行中");
        return false;
    }
    
    // 初始化性能计数器（Windows）
#ifdef _WIN32
    if (config_.enable_cpu_monitoring) {
        PdhOpenQuery(NULL, 0, &cpu_query_);
        PdhAddEnglishCounterW(cpu_query_, L"\\Processor(_Total)\\% Processor Time", 0, &cpu_counter_);
        PdhCollectQueryData(cpu_query_);
    }
    
    if (config_.enable_memory_monitoring) {
        PdhOpenQuery(NULL, 0, &memory_query_);
        PdhAddEnglishCounterW(memory_query_, L"\\Memory\\% Committed Bytes In Use", 0, &memory_counter_);
        PdhCollectQueryData(memory_query_);
    }
#endif
    
    monitoring_active_ = true;
    monitoring_paused_ = false;
    monitoring_start_time_ = std::chrono::system_clock::now();
    
    monitoring_thread_ = std::thread(&EnhancedPerformanceMonitor::monitoring_loop, this);
    
    Logger::instance().info("性能监控器已启动，采样间隔: {} ms", config_.sampling_interval_ms);
    return true;
}

void EnhancedPerformanceMonitor::stop_monitoring() {
    if (!monitoring_active_) {
        return;
    }
    
    monitoring_active_ = false;
    if (monitoring_thread_.joinable()) {
        monitoring_thread_.join();
    }
    
    Logger::instance().info("性能监控器已停止");
}

void EnhancedPerformanceMonitor::pause_monitoring() {
    std::lock_guard<std::mutex> lock(config_mutex_);
    monitoring_paused_ = true;
    Logger::instance().info("性能监控器已暂停");
}

void EnhancedPerformanceMonitor::resume_monitoring() {
    std::lock_guard<std::mutex> lock(config_mutex_);
    monitoring_paused_ = false;
    Logger::instance().info("性能监控器已恢复");
}

bool EnhancedPerformanceMonitor::is_monitoring() const {
    return monitoring_active_;
}

EnhancedPerformanceMonitor::PerformanceData EnhancedPerformanceMonitor::get_current_status() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (history_data_.empty()) {
        return EnhancedPerformanceMonitor::PerformanceData();
    }
    return history_data_.back();
}

std::vector<EnhancedPerformanceMonitor::PerformanceData> EnhancedPerformanceMonitor::get_history_data(int max_samples) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (max_samples <= 0 || max_samples >= static_cast<int>(history_data_.size())) {
        return history_data_;
    }

    std::vector<EnhancedPerformanceMonitor::PerformanceData> result;
    int start_idx = static_cast<int>(history_data_.size()) - max_samples;
    if (start_idx < 0) start_idx = 0;
    
    for (int i = start_idx; i < static_cast<int>(history_data_.size()); ++i) {
        result.push_back(history_data_[i]);
    }
    
    return result;
}

std::vector<EnhancedPerformanceMonitor::PerformanceAlert> EnhancedPerformanceMonitor::get_active_alerts() const {
    std::lock_guard<std::mutex> lock(alerts_mutex_);
    return active_alerts_;
}

void EnhancedPerformanceMonitor::acknowledge_alert(size_t alert_index) {
    std::lock_guard<std::mutex> lock(alerts_mutex_);
    if (alert_index < active_alerts_.size()) {
        active_alerts_.erase(active_alerts_.begin() + alert_index);
        Logger::instance().info("已确认警报 #{}", alert_index);
    }
}

void EnhancedPerformanceMonitor::add_custom_metric(const std::string& name, double value) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    custom_metrics_[name] = value;
}

void EnhancedPerformanceMonitor::set_custom_metric_threshold(
    const std::string& name, double threshold, 
    PerformanceAlert::AlertType alert_type) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    custom_thresholds_[name] = {threshold, alert_type};
}

EnhancedPerformanceMonitor::PerformanceReport EnhancedPerformanceMonitor::generate_report() const {
    EnhancedPerformanceMonitor::PerformanceReport report;
    
    std::lock_guard<std::mutex> lock(data_mutex_);
    std::lock_guard<std::mutex> lock2(alerts_mutex_);
    
    report.start_time = monitoring_start_time_;
    report.end_time = std::chrono::system_clock::now();
    report.history_data = history_data_;
    report.alerts_triggered = active_alerts_;
    
    if (!history_data_.empty()) {
        report.average_performance = calculate_average_performance();
        report.peak_performance = calculate_peak_performance();
        report.minimum_performance = calculate_minimum_performance();
    }
    
    // 计算统计摘要
    if (!history_data_.empty()) {
        double cpu_sum = 0.0, cpu_max = 0.0;
        double mem_sum = 0.0, mem_max = 0.0;
        double gpu_mem_sum = 0.0, gpu_mem_max = 0.0;
        
        for (const auto& data : history_data_) {
            cpu_sum += data.cpu_usage_percent;
            cpu_max = std::max(cpu_max, data.cpu_usage_percent);
            
            mem_sum += data.memory_usage_percent;
            mem_max = std::max(mem_max, data.memory_usage_percent);
            
            if (!data.gpu_data.empty()) {
                gpu_mem_sum += data.gpu_data[0].memory_usage_percent;
                gpu_mem_max = std::max(gpu_mem_max, data.gpu_data[0].memory_usage_percent);
            }
        }
        
        report.stats_summary.cpu_usage_avg = cpu_sum / history_data_.size();
        report.stats_summary.cpu_usage_max = cpu_max;
        report.stats_summary.memory_usage_avg = mem_sum / history_data_.size();
        report.stats_summary.memory_usage_max = mem_max;
        
        if (!history_data_.empty() && !history_data_[0].gpu_data.empty()) {
            report.stats_summary.gpu_memory_usage_avg = gpu_mem_sum / history_data_.size();
            report.stats_summary.gpu_memory_usage_max = gpu_mem_max;
        }
    }
    
    return report;
}

bool EnhancedPerformanceMonitor::export_report_to_file(const std::string& file_path, const std::string& format) {
    try {
        PerformanceReport report = generate_report();
        std::ofstream file(file_path);
        
        if (!file.is_open()) {
            Logger::instance().error("无法打开文件用于导出性能报告: {}", file_path);
            return false;
        }
        
        std::string content;
        if (format == "json") {
            content = report.to_json();
        } else if (format == "html") {
            content = report.to_html();
        } else {
            content = report.to_string();
        }
        
        file << content;
        file.close();
        
        Logger::instance().info("性能报告已导出到: {} (格式: {})", file_path, format);
        return true;
    } catch (const std::exception& e) {
        Logger::instance().error("导出性能报告时发生错误: {}", e.what());
        return false;
    }
}

void EnhancedPerformanceMonitor::set_config(const MonitorConfig& config) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    config_ = config;
}

EnhancedPerformanceMonitor::MonitorConfig EnhancedPerformanceMonitor::get_config() const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return config_;
}

void EnhancedPerformanceMonitor::reset_monitoring_data() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    std::lock_guard<std::mutex> lock2(alerts_mutex_);
    
    history_data_.clear();
    active_alerts_.clear();
    
    Logger::instance().info("性能监控数据已重置");
}

std::string EnhancedPerformanceMonitor::get_system_info() {
    std::ostringstream oss;
    
#ifdef _WIN32
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    
    oss << "操作系统: Windows\n";
    oss << "处理器架构: ";
    switch (sysInfo.wProcessorArchitecture) {
        case PROCESSOR_ARCHITECTURE_AMD64: oss << "x64"; break;
        case PROCESSOR_ARCHITECTURE_ARM: oss << "ARM"; break;
        case PROCESSOR_ARCHITECTURE_ARM64: oss << "ARM64"; break;
        case PROCESSOR_ARCHITECTURE_IA64: oss << "IA64"; break;
        case PROCESSOR_ARCHITECTURE_INTEL: oss << "x86"; break;
        default: oss << "未知"; break;
    }
    oss << "\n";
    oss << "处理器数量: " << sysInfo.dwNumberOfProcessors << "\n";
    oss << "物理内存总量: " << memInfo.ullTotalPhys / (1024 * 1024 * 1024) << " GB\n";
    oss << "可用物理内存: " << memInfo.ullAvailPhys / (1024 * 1024 * 1024) << " GB\n";
    
    // 获取Windows版本
    DWORD dwVersion = GetVersion();
    DWORD dwMajorVersion = (DWORD)(LOBYTE(LOWORD(dwVersion)));
    DWORD dwMinorVersion = (DWORD)(HIBYTE(LOWORD(dwVersion)));
    oss << "Windows版本: " << dwMajorVersion << "." << dwMinorVersion << "\n";
#else
    oss << "操作系统: Linux/Unix (详细信息需要特定实现)\n";
#endif
    
    return oss.str();
}

std::vector<EnhancedPerformanceMonitor::PerformanceData::GPUData> EnhancedPerformanceMonitor::get_gpu_info() {
    std::vector<PerformanceData::GPUData> gpu_info;
    
#ifdef USE_NVML
    nvmlReturn_t result;
    unsigned int device_count;
    
    result = nvmlInit();
    if (result != NVML_SUCCESS) {
        Logger::instance().warning("NVML初始化失败: {}", nvmlErrorString(result));
        return gpu_info;
    }
    
    result = nvmlDeviceGetCount(&device_count);
    if (result != NVML_SUCCESS) {
        Logger::instance().warning("获取GPU数量失败: {}", nvmlErrorString(result));
        nvmlShutdown();
        return gpu_info;
    }
    
    for (unsigned int i = 0; i < device_count; i++) {
        nvmlDevice_t device;
        result = nvmlDeviceGetHandleByIndex(i, &device);
        if (result != NVML_SUCCESS) {
            continue;
        }
        
        PerformanceData::GPUData gpu_data;
        gpu_data.gpu_index = i;
        
        char name[NVML_DEVICE_NAME_BUFFER_SIZE];
        result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
        if (result == NVML_SUCCESS) {
            gpu_data.gpu_name = name;
        }
        
        nvmlMemory_t memory;
        result = nvmlDeviceGetMemoryInfo(device, &memory);
        if (result == NVML_SUCCESS) {
            gpu_data.total_memory_bytes = memory.total;
            gpu_data.used_memory_bytes = memory.used;
            gpu_data.memory_usage_percent = (static_cast<double>(memory.used) / memory.total) * 100.0;
        }
        
        unsigned int temperature;
        result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature);
        if (result == NVML_SUCCESS) {
            gpu_data.temperature_celsius = static_cast<double>(temperature);
        }
        
        unsigned int utilization;
        result = nvmlDeviceGetUtilizationRates(device, &utilization);
        if (result == NVML_SUCCESS) {
            gpu_data.utilization_percent = static_cast<double>(utilization.gpu);
        }
        
        gpu_info.push_back(gpu_data);
    }
    
    nvmlShutdown();
#endif
    
    return gpu_info;
}

// ============================================================================
// 私有方法实现
// ============================================================================

void EnhancedPerformanceMonitor::monitoring_loop() {
    auto last_summary_time = std::chrono::system_clock::now();
    
    while (monitoring_active_) {
        // 检查是否暂停
        {
            std::lock_guard<std::mutex> lock(config_mutex_);
            if (monitoring_paused_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
        }
        
        // 收集性能数据
        PerformanceData data = collect_performance_data();
        
        // 分析数据
        analyze_performance_data(data);
        
        // 检查警报
        check_for_alerts(data);
        
        // 记录数据
        log_performance_data(data);
        
        // 定期记录性能摘要
        auto now = std::chrono::system_clock::now();
        if (config_.log_summary_periodically) {
            auto elapsed_sec = std::chrono::duration<double>(now - last_summary_time).count();
            if (elapsed_sec >= config_.summary_log_interval_sec) {
                log_performance_summary();
                last_summary_time = now;
            }
        }
        
        // 等待下一个采样周期
        std::this_thread::sleep_for(std::chrono::milliseconds(config_.sampling_interval_ms));
    }
}

EnhancedPerformanceMonitor::PerformanceData EnhancedPerformanceMonitor::collect_windows_performance_data() {
    PerformanceData data;
    data.timestamp = std::chrono::system_clock::now();
    
    // 获取系统内存信息
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    
    data.total_memory_bytes = memInfo.ullTotalPhys;
    data.used_memory_bytes = memInfo.ullTotalPhys - memInfo.ullAvailPhys;
    data.available_memory_bytes = memInfo.ullAvailPhys;
    data.memory_usage_percent = memInfo.dwMemoryLoad;
    
    // 获取CPU使用率（使用PDH）
    if (config_.enable_cpu_monitoring && cpu_query_ != NULL) {
        PDH_FMT_COUNTERVALUE counterValue;
        PdhCollectQueryData(cpu_query_);
        PdhGetFormattedCounterValue(cpu_counter_, PDH_FMT_DOUBLE, NULL, &counterValue);
        data.cpu_usage_percent = counterValue.doubleValue;
    }
    
    // 获取进程内存信息
    if (process_handle_ != INVALID_HANDLE_VALUE) {
        PROCESS_MEMORY_COUNTERS pmc;
        if (GetProcessMemoryInfo(process_handle_, &pmc, sizeof(pmc))) {
            data.process_memory_bytes = pmc.WorkingSetSize;
            data.process_private_bytes = pmc.WorkingSetSize;
            data.process_working_set_bytes = pmc.WorkingSetSize;
        }
        
        // 获取进程句柄数
        DWORD handleCount;
        if (GetProcessHandleCount(process_handle_, &handleCount)) {
            data.process_handle_count = handleCount;
        }
    }
    
    // 获取进程线程数（简单实现）
    data.process_thread_count = 0;
    
    return data;
}

EnhancedPerformanceMonitor::PerformanceData EnhancedPerformanceMonitor::collect_linux_performance_data() {
    // Linux实现占位符
    PerformanceData data;
    data.timestamp = std::chrono::system_clock::now();
    
    // 在实际的Linux实现中，这里会解析/proc文件系统
    // 例如：/proc/meminfo, /proc/stat, /proc/self/status等
    
    return data;
}

void EnhancedPerformanceMonitor::initialize_gpu_monitoring() {
#ifdef USE_NVML
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        Logger::instance().warning("NVML初始化失败: {}", nvmlErrorString(result));
        nvml_initialized_ = false;
        return;
    }
    
    nvml_initialized_ = true;
    
    // 获取GPU设备列表
    unsigned int device_count;
    result = nvmlDeviceGetCount(&device_count);
    if (result != NVML_SUCCESS) {
        Logger::instance().warning("获取GPU数量失败: {}", nvmlErrorString(result));
        nvmlShutdown();
        nvml_initialized_ = false;
        return;
    }
    
    nvml_devices_.resize(device_count);
    for (unsigned int i = 0; i < device_count; i++) {
        result = nvmlDeviceGetHandleByIndex(i, &nvml_devices_[i]);
        if (result != NVML_SUCCESS) {
            Logger::instance().warning("获取GPU {} 句柄失败: {}", i, nvmlErrorString(result));
        }
    }
    
    Logger::instance().info("GPU监控已初始化，检测到 {} 个GPU", device_count);
#endif
}

void EnhancedPerformanceMonitor::cleanup_gpu_monitoring() {
#ifdef USE_NVML
    if (nvml_initialized_) {
        nvmlShutdown();
        nvml_initialized_ = false;
        nvml_devices_.clear();
    }
#endif
}

std::vector<EnhancedPerformanceMonitor::PerformanceData::GPUData> EnhancedPerformanceMonitor::collect_gpu_data() {
    std::vector<PerformanceData::GPUData> gpu_data_list;
    
#ifdef USE_NVML
    if (!nvml_initialized_) {
        return gpu_data_list;
    }
    
    for (size_t i = 0; i < nvml_devices_.size(); i++) {
        GPUData gpu_data = collect_gpu_data_nvml(static_cast<int>(i));
        gpu_data_list.push_back(gpu_data);
    }
#endif
    
    return gpu_data_list;
}

void EnhancedPerformanceMonitor::initialize_process_monitoring() {
    // Windows实现：已经在构造函数中打开了进程句柄
    // Linux实现需要打开/proc/self/目录
}

void EnhancedPerformanceMonitor::cleanup_process_monitoring() {
    // Windows实现：在析构函数中关闭进程句柄
    // Linux实现：无需特殊清理
}

EnhancedPerformanceMonitor::PerformanceData::GPUData EnhancedPerformanceMonitor::collect_gpu_data_nvml(int gpu_index) {
    PerformanceData::GPUData gpu_data;
    gpu_data.gpu_index = gpu_index;
    
#ifdef USE_NVML
    if (gpu_index < 0 || static_cast<size_t>(gpu_index) >= nvml_devices_.size()) {
        return gpu_data;
    }
    
    nvmlDevice_t device = nvml_devices_[gpu_index];
    
    // 获取GPU名称
    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    nvmlReturn_t result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
    if (result == NVML_SUCCESS) {
        gpu_data.gpu_name = name;
    }
    
    // 获取显存信息
    nvmlMemory_t memory;
    result = nvmlDeviceGetMemoryInfo(device, &memory);
    if (result == NVML_SUCCESS) {
        gpu_data.total_memory_bytes = memory.total;
        gpu_data.used_memory_bytes = memory.used;
        gpu_data.memory_usage_percent = (static_cast<double>(memory.used) / memory.total) * 100.0;
    }
    
    // 获取温度
    unsigned int temperature;
    result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature);
    if (result == NVML_SUCCESS) {
        gpu_data.temperature_celsius = static_cast<double>(temperature);
    }
    
    // 获取利用率
    nvmlUtilization_t utilization;
    result = nvmlDeviceGetUtilizationRates(device, &utilization);
    if (result == NVML_SUCCESS) {
        gpu_data.utilization_percent = static_cast<double>(utilization.gpu);
    }
    
    // 获取功耗（如果可用）
    unsigned int power;
    result = nvmlDeviceGetPowerUsage(device, &power);
    if (result == NVML_SUCCESS) {
        gpu_data.power_usage_watts = static_cast<double>(power) / 1000.0;  // 转换为瓦特
    }
    
    // 获取风扇转速（如果可用）
    unsigned int fan_speed;
    result = nvmlDeviceGetFanSpeed(device, &fan_speed);
    if (result == NVML_SUCCESS) {
        gpu_data.fan_speed_percent = fan_speed;
    }
    
    // 获取时钟频率（如果可用）
    unsigned int clock_speed;
    result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &clock_speed);
    if (result == NVML_SUCCESS) {
        gpu_data.clock_speed_mhz = clock_speed;
    }
#endif
    
    return gpu_data;
}

double EnhancedPerformanceMonitor::calculate_cpu_usage(const FILETIME& prev_idle, const FILETIME& prev_kernel, 
                                                      const FILETIME& prev_user, const FILETIME& curr_idle, 
                                                      const FILETIME& curr_kernel, const FILETIME& curr_user) {
    ULARGE_INTEGER idle1, kernel1, user1;
    ULARGE_INTEGER idle2, kernel2, user2;
    
    idle1.LowPart = prev_idle.dwLowDateTime;
    idle1.HighPart = prev_idle.dwHighDateTime;
    kernel1.LowPart = prev_kernel.dwLowDateTime;
    kernel1.HighPart = prev_kernel.dwHighDateTime;
    user1.LowPart = prev_user.dwLowDateTime;
    user1.HighPart = prev_user.dwHighDateTime;
    
    idle2.LowPart = curr_idle.dwLowDateTime;
    idle2.HighPart = curr_idle.dwHighDateTime;
    kernel2.LowPart = curr_kernel.dwLowDateTime;
    kernel2.HighPart = curr_kernel.dwHighDateTime;
    user2.LowPart = curr_user.dwLowDateTime;
    user2.HighPart = curr_user.dwHighDateTime;
    
    ULONGLONG idle_diff = idle2.QuadPart - idle1.QuadPart;
    ULONGLONG kernel_diff = kernel2.QuadPart - kernel1.QuadPart;
    ULONGLONG user_diff = user2.QuadPart - user1.QuadPart;
    
    ULONGLONG total_diff = kernel_diff + user_diff;
    if (total_diff == 0) {
        return 0.0;
    }
    
    double cpu_usage = 100.0 * (1.0 - (static_cast<double>(idle_diff) / total_diff));
    return std::max(0.0, std::min(100.0, cpu_usage));
}

uint64_t EnhancedPerformanceMonitor::get_process_memory_usage() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize;
    }
#endif
    return 0;
}

uint64_t EnhancedPerformanceMonitor::get_system_memory_usage() {
#ifdef _WIN32
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&memInfo)) {
        return memInfo.ullTotalPhys - memInfo.ullAvailPhys;
    }
#endif
    return 0;
}

void EnhancedPerformanceMonitor::generate_cpu_alert(const PerformanceData& data) {
    PerformanceAlert alert;
    alert.type = PerformanceAlert::AlertType::CPU_HIGH_USAGE;
    {
        std::ostringstream oss;
        oss << "CPU使用率过高: " << std::fixed << std::setprecision(1) << data.cpu_usage_percent 
            << "% (阈值: " << config_.cpu_alert_threshold << "%)";
        alert.message = oss.str();
    }
    alert.severity = std::min(1.0, data.cpu_usage_percent / 100.0);
    alert.timestamp = std::chrono::system_clock::now();
    alert.trigger_data = data;
    
    alert.suggestions = {
        "检查是否有CPU密集型进程在运行",
        "考虑优化算法以减少CPU使用",
        "增加CPU使用率阈值或调整监控配置"
    };
    
    std::lock_guard<std::mutex> lock(alerts_mutex_);
    active_alerts_.push_back(alert);
    
    Logger::instance().warning("CPU警报触发: {}", alert.message);
}

void EnhancedPerformanceMonitor::generate_memory_alert(const PerformanceData& data) {
    PerformanceAlert alert;
    alert.type = PerformanceAlert::AlertType::MEMORY_HIGH_USAGE;
    {
        std::ostringstream oss;
        oss << "内存使用率过高: " << std::fixed << std::setprecision(1) << data.memory_usage_percent 
            << "% (阈值: " << config_.memory_alert_threshold << "%)";
        alert.message = oss.str();
    }
    alert.severity = std::min(1.0, data.memory_usage_percent / 100.0);
    alert.timestamp = std::chrono::system_clock::now();
    alert.trigger_data = data;
    
    alert.suggestions = {
        "检查内存泄漏",
        "优化数据结构以减少内存使用",
        "增加系统内存或调整交换空间",
        "考虑使用内存池或缓存机制"
    };
    
    std::lock_guard<std::mutex> lock(alerts_mutex_);
    active_alerts_.push_back(alert);
    
    Logger::instance().warning("内存警报触发: {}", alert.message);
}

void EnhancedPerformanceMonitor::generate_gpu_alerts(const PerformanceData& data) {
    if (data.gpu_data.empty()) {
        return;
    }
    
    for (const auto& gpu : data.gpu_data) {
        // GPU显存警报
        if (gpu.memory_usage_percent > config_.gpu_memory_alert_threshold) {
            PerformanceAlert alert;
            alert.type = PerformanceAlert::AlertType::GPU_MEMORY_HIGH_USAGE;
            {
                std::ostringstream oss;
                oss << "GPU " << gpu.gpu_index << " 显存使用率过高: " << std::fixed << std::setprecision(1) 
                    << gpu.memory_usage_percent << "% (阈值: " << config_.gpu_memory_alert_threshold << "%)";
                alert.message = oss.str();
            }
            alert.severity = std::min(1.0, gpu.memory_usage_percent / 100.0);
            alert.timestamp = std::chrono::system_clock::now();
            alert.trigger_data = data;
            
            alert.suggestions = {
                "减少批量大小以降低显存使用",
                "使用混合精度计算以减少显存占用",
                "优化模型以减少显存需求",
                "考虑使用多GPU分布式计算"
            };
            
            std::lock_guard<std::mutex> lock(alerts_mutex_);
            active_alerts_.push_back(alert);
            
            Logger::instance().warning("GPU显存警报触发: {}", alert.message);
        }
        
        // GPU温度警报
        if (gpu.temperature_celsius > config_.gpu_temperature_alert) {
            PerformanceAlert alert;
            alert.type = PerformanceAlert::AlertType::GPU_TEMPERATURE_HIGH;
            {
                std::ostringstream oss;
                oss << "GPU " << gpu.gpu_index << " 温度过高: " << std::fixed << std::setprecision(1) 
                    << gpu.temperature_celsius << "°C (阈值: " << config_.gpu_temperature_alert << "°C)";
                alert.message = oss.str();
            }
            alert.severity = std::min(1.0, gpu.temperature_celsius / 100.0);
            alert.timestamp = std::chrono::system_clock::now();
            alert.trigger_data = data;
            
            alert.suggestions = {
                "检查GPU散热系统",
                "降低GPU负载或频率",
                "改善机箱通风和散热",
                "考虑使用水冷散热系统"
            };
            
            std::lock_guard<std::mutex> lock(alerts_mutex_);
            active_alerts_.push_back(alert);
            
            Logger::instance().warning("GPU温度警报触发: {}", alert.message);
        }
    }
}

void EnhancedPerformanceMonitor::generate_custom_metric_alerts(const PerformanceData& data) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    for (const auto& [name, threshold_pair] : custom_thresholds_) {
        double threshold = threshold_pair.first;
        PerformanceAlert::AlertType alert_type = threshold_pair.second;
        
        auto it = data.custom_metrics.find(name);
        if (it != data.custom_metrics.end() && it->second > threshold) {
            PerformanceAlert alert;
            alert.type = alert_type;
            {
                std::ostringstream oss;
                oss << "自定义指标 '" << name << "' 超过阈值: " << std::fixed << std::setprecision(2) 
                    << it->second << " > " << threshold;
                alert.message = oss.str();
            }
            alert.severity = std::min(1.0, it->second / (threshold * 2.0));
            alert.timestamp = std::chrono::system_clock::now();
            alert.trigger_data = data;
            
            alert.suggestions = {
                std::string("检查指标 '") + name + "' 的计算逻辑",
                "调整阈值或优化相关算法"
            };
            
            std::lock_guard<std::mutex> lock2(alerts_mutex_);
            active_alerts_.push_back(alert);
            
            Logger::instance().warning("自定义指标警报触发: {}", alert.message);
        }
    }
}

EnhancedPerformanceMonitor::PerformanceData EnhancedPerformanceMonitor::calculate_average_performance() const {
    if (history_data_.empty()) {
        return PerformanceData();
    }
    
    PerformanceData avg_data;
    size_t count = history_data_.size();
    
    // 初始化累加器
    double cpu_sum = 0.0;
    double mem_sum = 0.0;
    uint64_t total_memory_sum = 0;
    uint64_t used_memory_sum = 0;
    uint64_t process_memory_sum = 0;
    
    for (const auto& data : history_data_) {
        cpu_sum += data.cpu_usage_percent;
        mem_sum += data.memory_usage_percent;
        total_memory_sum += data.total_memory_bytes;
        used_memory_sum += data.used_memory_bytes;
        process_memory_sum += data.process_memory_bytes;
    }
    
    avg_data.cpu_usage_percent = cpu_sum / count;
    avg_data.memory_usage_percent = mem_sum / count;
    avg_data.total_memory_bytes = total_memory_sum / count;
    avg_data.used_memory_bytes = used_memory_sum / count;
    avg_data.process_memory_bytes = process_memory_sum / count;
    
    // 计算GPU平均值（如果有）
    if (!history_data_.empty() && !history_data_[0].gpu_data.empty()) {
        size_t gpu_count = history_data_[0].gpu_data.size();
        std::vector<PerformanceData::GPUData> avg_gpu_data(gpu_count);
        
        for (size_t gpu_idx = 0; gpu_idx < gpu_count; gpu_idx++) {
            double util_sum = 0.0, mem_usage_sum = 0.0, temp_sum = 0.0;
            
            for (const auto& data : history_data_) {
                if (gpu_idx < data.gpu_data.size()) {
                    const auto& gpu = data.gpu_data[gpu_idx];
                    util_sum += gpu.utilization_percent;
                    mem_usage_sum += gpu.memory_usage_percent;
                    temp_sum += gpu.temperature_celsius;
                }
            }
            
            avg_gpu_data[gpu_idx].gpu_index = static_cast<uint32_t>(gpu_idx);
            avg_gpu_data[gpu_idx].utilization_percent = util_sum / count;
            avg_gpu_data[gpu_idx].memory_usage_percent = mem_usage_sum / count;
            avg_gpu_data[gpu_idx].temperature_celsius = temp_sum / count;
        }
        
        avg_data.gpu_data = avg_gpu_data;
    }
    
    return avg_data;
}

EnhancedPerformanceMonitor::PerformanceData EnhancedPerformanceMonitor::calculate_peak_performance() const {
    if (history_data_.empty()) {
        return PerformanceData();
    }
    
    PerformanceData peak_data = history_data_[0];
    
    for (const auto& data : history_data_) {
        if (data.cpu_usage_percent > peak_data.cpu_usage_percent) {
            peak_data.cpu_usage_percent = data.cpu_usage_percent;
        }
        if (data.memory_usage_percent > peak_data.memory_usage_percent) {
            peak_data.memory_usage_percent = data.memory_usage_percent;
        }
        if (data.process_memory_bytes > peak_data.process_memory_bytes) {
            peak_data.process_memory_bytes = data.process_memory_bytes;
        }
        
        // GPU峰值
        if (!data.gpu_data.empty() && !peak_data.gpu_data.empty() && 
            data.gpu_data.size() == peak_data.gpu_data.size()) {
            for (size_t i = 0; i < data.gpu_data.size(); i++) {
                if (data.gpu_data[i].utilization_percent > peak_data.gpu_data[i].utilization_percent) {
                    peak_data.gpu_data[i].utilization_percent = data.gpu_data[i].utilization_percent;
                }
                if (data.gpu_data[i].memory_usage_percent > peak_data.gpu_data[i].memory_usage_percent) {
                    peak_data.gpu_data[i].memory_usage_percent = data.gpu_data[i].memory_usage_percent;
                }
                if (data.gpu_data[i].temperature_celsius > peak_data.gpu_data[i].temperature_celsius) {
                    peak_data.gpu_data[i].temperature_celsius = data.gpu_data[i].temperature_celsius;
                }
            }
        }
    }
    
    return peak_data;
}

EnhancedPerformanceMonitor::PerformanceData EnhancedPerformanceMonitor::calculate_minimum_performance() const {
    if (history_data_.empty()) {
        return PerformanceData();
    }
    
    PerformanceData min_data = history_data_[0];
    
    for (const auto& data : history_data_) {
        if (data.cpu_usage_percent < min_data.cpu_usage_percent) {
            min_data.cpu_usage_percent = data.cpu_usage_percent;
        }
        if (data.memory_usage_percent < min_data.memory_usage_percent) {
            min_data.memory_usage_percent = data.memory_usage_percent;
        }
        if (data.process_memory_bytes < min_data.process_memory_bytes) {
            min_data.process_memory_bytes = data.process_memory_bytes;
        }
        
        // GPU最小值
        if (!data.gpu_data.empty() && !min_data.gpu_data.empty() && 
            data.gpu_data.size() == min_data.gpu_data.size()) {
            for (size_t i = 0; i < data.gpu_data.size(); i++) {
                if (data.gpu_data[i].utilization_percent < min_data.gpu_data[i].utilization_percent) {
                    min_data.gpu_data[i].utilization_percent = data.gpu_data[i].utilization_percent;
                }
                if (data.gpu_data[i].memory_usage_percent < min_data.gpu_data[i].memory_usage_percent) {
                    min_data.gpu_data[i].memory_usage_percent = data.gpu_data[i].memory_usage_percent;
                }
                if (data.gpu_data[i].temperature_celsius < min_data.gpu_data[i].temperature_celsius) {
                    min_data.gpu_data[i].temperature_celsius = data.gpu_data[i].temperature_celsius;
                }
            }
        }
    }
    
    return min_data;
}

} // namespace framework
} // namespace spacego