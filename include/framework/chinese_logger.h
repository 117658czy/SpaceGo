#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <string>
#include <sstream>
#include <iomanip>
#include <locale>
#include <codecvt>
#include <vector>
#include <map>

#include "framework/logger.h"

namespace spacego {
namespace framework {

/**
 * @brief 中文日志支持工具
 * 
 * 提供中文日志消息的格式化、本地化和辅助功能。
 * 与现有Logger系统集成，确保所有日志消息使用中文，便于快速定位问题。
 * 
 * 主要功能：
 * 1. 中文日志消息格式化
 * 2. 本地化支持（可切换中英文）
 * 3. 错误代码到中文描述的映射
 * 4. 性能数据中文格式化
 * 5. 异常信息中文转换
 * 6. 提供便捷的宏定义简化中文日志记录
 * 
 * 使用示例：
 * @code
 * // 使用中文日志宏
 * LOG_CN_INFO("神经网络推理完成，用时 {}ms", 125.4);
 * 
 * // 使用带分类的中文日志
 * LOG_CN_CATEGORY("围棋AI", "HMGS-DP搜索完成{}次模拟", 1000);
 * 
 * // 错误处理
 * try {
 *     risky_operation();
 * } catch (const std::exception& e) {
 *     LOG_CN_ERROR("操作失败: {}", ChineseLogger::translate_exception(e));
 * }
 * @endcode
 */
class ChineseLogger {
public:
    /// 日志分类（对应不同模块）
    enum class LogCategory {
        SYSTEM,         ///< 系统
        NETWORK,        ///< 网络
        DATABASE,       ///< 数据库
        GUI,            ///< 图形界面
        AI_ENGINE,      ///< AI引擎
        SEARCH_ALG,     ///< 搜索算法
        NEURAL_NET,     ///< 神经网络
        CUDA,           ///< CUDA加速
        PERFORMANCE,    ///< 性能监控
        SECURITY,       ///< 安全
        UTILITY,        ///< 工具类
        TESTING,        ///< 测试
        CUSTOM          ///< 自定义分类
    };

    /// 语言设置
    enum class Language {
        CHINESE,        ///< 中文（默认）
        ENGLISH         ///< 英文
    };

    /// 初始化中文日志系统
    static void initialize(Language lang = Language::CHINESE);

    /// 设置当前语言
    static void set_language(Language lang);

    /// 获取当前语言
    static Language get_language();

    /// 设置分类名称映射（可自定义分类显示名称）
    static void set_category_name(LogCategory category, const std::string& chinese_name, const std::string& english_name = "");

    /// 获取分类名称（根据当前语言）
    static std::string get_category_name(LogCategory category);

    /// 格式化带参数的中文日志消息（类型安全模板版本）
    template<typename... Args>
    static std::string format_message(const std::string& format, Args... args);

    /// 翻译异常信息为中文
    static std::string translate_exception(const std::exception& e);

    /// 翻译错误代码为中文描述
    static std::string translate_error_code(int error_code, const std::string& context = "");

    /// 格式化性能数据为中文
    static std::string format_performance_data(const std::string& component, double time_ms, 
                                               const std::string& unit = "ms");

    /// 格式化内存大小为中文
    static std::string format_memory_size(uint64_t bytes);

    /// 格式化百分比为中文
    static std::string format_percentage(double percent, int precision = 1);

    /// 获取当前时间的中文格式化字符串
    static std::string current_time_string();

    /// 记录中文日志（直接调用Logger）
    static void log(LogLevel level, LogCategory category, const std::string& message, 
                    const std::string& file = "", int line = 0, const std::string& function = "");

    /// 记录中文日志（带格式化参数）- 内部使用
    template<typename... Args>
    static void log_with_location(LogLevel level, LogCategory category, 
                    const std::string& file, int line, const std::string& function,
                    const std::string& format, Args... args);

    // 便捷日志方法（按级别）
    static void trace(LogCategory category, const std::string& message, 
                      const std::string& file = "", int line = 0, const std::string& function = "");
    static void debug(LogCategory category, const std::string& message, 
                      const std::string& file = "", int line = 0, const std::string& function = "");
    static void info(LogCategory category, const std::string& message, 
                     const std::string& file = "", int line = 0, const std::string& function = "");
    static void warning(LogCategory category, const std::string& message, 
                        const std::string& file = "", int line = 0, const std::string& function = "");
    static void error(LogCategory category, const std::string& message, 
                      const std::string& file = "", int line = 0, const std::string& function = "");
    static void fatal(LogCategory category, const std::string& message, 
                      const std::string& file = "", int line = 0, const std::string& function = "");

    // 带格式化的便捷日志方法 - 内部使用
    template<typename... Args>
    static void trace_with_location(LogCategory category, 
                      const std::string& file, int line, const std::string& function,
                      const std::string& format, Args... args);
    template<typename... Args>
    static void debug_with_location(LogCategory category, 
                      const std::string& file, int line, const std::string& function,
                      const std::string& format, Args... args);
    template<typename... Args>
    static void info_with_location(LogCategory category, 
                     const std::string& file, int line, const std::string& function,
                     const std::string& format, Args... args);
    template<typename... Args>
    static void warning_with_location(LogCategory category, 
                        const std::string& file, int line, const std::string& function,
                        const std::string& format, Args... args);
    template<typename... Args>
    static void error_with_location(LogCategory category, 
                      const std::string& file, int line, const std::string& function,
                      const std::string& format, Args... args);
    template<typename... Args>
    static void fatal_with_location(LogCategory category, 
                      const std::string& file, int line, const std::string& function,
                      const std::string& format, Args... args);

    /// 错误代码到中文描述的映射
    struct ErrorCodeMapping {
        int code;
        std::string chinese_description;
        std::string english_description;
        std::vector<std::string> suggested_solutions;
    };

    /// 添加自定义错误代码映射
    static void add_error_code_mapping(const ErrorCodeMapping& mapping);

    /// 异常类型到中文描述的映射
    struct ExceptionMapping {
        std::string exception_type;
        std::string chinese_description;
        std::string english_description;
        std::vector<std::string> common_causes;
    };

    /// 添加自定义异常映射
    static void add_exception_mapping(const ExceptionMapping& mapping);

private:
    /// 语言设置
    static Language current_language_;
    static std::mutex language_mutex_;

    /// 分类名称映射
    static std::map<LogCategory, std::pair<std::string, std::string>> category_names_;
    static std::mutex category_mutex_;

    /// 错误代码映射
    static std::map<int, ErrorCodeMapping> error_code_mappings_;
    static std::mutex error_code_mutex_;

    /// 异常映射
    static std::map<std::string, ExceptionMapping> exception_mappings_;
    static std::mutex exception_mutex_;

    /// 初始化默认映射
    static void initialize_default_mappings();

    /// 工具方法
    static std::string get_level_string_cn(LogLevel level);
    static std::string get_level_string_en(LogLevel level);
    static std::string ensure_utf8(const std::string& str);
    static std::wstring to_wide_string(const std::string& str);
    static std::string from_wide_string(const std::wstring& wstr);
};

// 模板方法实现
template<typename... Args>
std::string ChineseLogger::format_message(const std::string& format, Args... args) {
    char buffer[4096];
    snprintf(buffer, sizeof(buffer), format.c_str(), args...);
    return std::string(buffer);
}

template<typename... Args>
void ChineseLogger::log_with_location(LogLevel level, LogCategory category, 
                        const std::string& file, int line, const std::string& function,
                        const std::string& format, Args... args) {
    std::string message = format_message(format, args...);
    log(level, category, message, file, line, function);
}

template<typename... Args>
void ChineseLogger::trace_with_location(LogCategory category, 
                          const std::string& file, int line, const std::string& function,
                          const std::string& format, Args... args) {
    log_with_location(LogLevel::TRACE, category, file, line, function, format, args...);
}

template<typename... Args>
void ChineseLogger::debug_with_location(LogCategory category, 
                          const std::string& file, int line, const std::string& function,
                          const std::string& format, Args... args) {
    log_with_location(LogLevel::DEBUG, category, file, line, function, format, args...);
}

template<typename... Args>
void ChineseLogger::info_with_location(LogCategory category, 
                         const std::string& file, int line, const std::string& function,
                         const std::string& format, Args... args) {
    log_with_location(LogLevel::INFO, category, file, line, function, format, args...);
}

template<typename... Args>
void ChineseLogger::warning_with_location(LogCategory category, 
                            const std::string& file, int line, const std::string& function,
                            const std::string& format, Args... args) {
    log_with_location(LogLevel::WARNING, category, file, line, function, format, args...);
}

template<typename... Args>
void ChineseLogger::error_with_location(LogCategory category, 
                          const std::string& file, int line, const std::string& function,
                          const std::string& format, Args... args) {
    log_with_location(LogLevel::LVL_ERROR, category, file, line, function, format, args...);
}

template<typename... Args>
void ChineseLogger::fatal_with_location(LogCategory category, 
                          const std::string& file, int line, const std::string& function,
                          const std::string& format, Args... args) {
    log_with_location(LogLevel::FATAL, category, file, line, function, format, args...);
}

} // namespace framework
} // namespace spacego

// ============================================================================
// 中文日志宏定义
// ============================================================================

/**
 * @brief 中文日志宏定义
 * 
 * 提供一组便捷的宏来记录中文日志，自动处理文件名、行号和函数名。
 * 所有宏都支持格式化参数（类似printf风格）。
 */

/// 基础中文日志宏
#define LOG_CN(level, category, ...) \
    spacego::framework::ChineseLogger::log_with_location(level, category, __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)

/// 中文TRACE级别日志
#define LOG_CN_TRACE(category, ...) \
    spacego::framework::ChineseLogger::trace_with_location(category, __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)

/// 中文DEBUG级别日志
#define LOG_CN_DEBUG(category, ...) \
    spacego::framework::ChineseLogger::debug_with_location(category, __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)

/// 中文INFO级别日志
#define LOG_CN_INFO(category, ...) \
    spacego::framework::ChineseLogger::info_with_location(category, __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)

/// 中文WARNING级别日志
#define LOG_CN_WARNING(category, ...) \
    spacego::framework::ChineseLogger::warning_with_location(category, __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)

/// 中文ERROR级别日志
#define LOG_CN_ERROR(category, ...) \
    spacego::framework::ChineseLogger::error_with_location(category, __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)

/// 中文FATAL级别日志
#define LOG_CN_FATAL(category, ...) \
    spacego::framework::ChineseLogger::fatal_with_location(category, __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)

// ============================================================================
// 简化版中文日志宏（不带分类）
// ============================================================================

/// 简化版中文TRACE日志（使用默认分类）
#define LOG_CN_TRACE_SIMPLE(...) \
    LOG_CN_TRACE(spacego::framework::ChineseLogger::LogCategory::SYSTEM, __VA_ARGS__)

/// 简化版中文DEBUG日志
#define LOG_CN_DEBUG_SIMPLE(...) \
    LOG_CN_DEBUG(spacego::framework::ChineseLogger::LogCategory::SYSTEM, __VA_ARGS__)

/// 简化版中文INFO日志
#define LOG_CN_INFO_SIMPLE(...) \
    LOG_CN_INFO(spacego::framework::ChineseLogger::LogCategory::SYSTEM, __VA_ARGS__)

/// 简化版中文WARNING日志
#define LOG_CN_WARNING_SIMPLE(...) \
    LOG_CN_WARNING(spacego::framework::ChineseLogger::LogCategory::SYSTEM, __VA_ARGS__)

/// 简化版中文ERROR日志
#define LOG_CN_ERROR_SIMPLE(...) \
    LOG_CN_ERROR(spacego::framework::ChineseLogger::LogCategory::SYSTEM, __VA_ARGS__)

/// 简化版中文FATAL日志
#define LOG_CN_FATAL_SIMPLE(...) \
    LOG_CN_FATAL(spacego::framework::ChineseLogger::LogCategory::SYSTEM, __VA_ARGS__)

// ============================================================================
// 性能日志宏
// ============================================================================

/// 性能数据日志（中文）
#define LOG_PERFORMANCE_CN(component, time_ms, unit) \
    LOG_CN_INFO(spacego::framework::ChineseLogger::LogCategory::PERFORMANCE, \
                "{} 耗时 {:.2f}{}", component, time_ms, unit)

/// 内存使用日志（中文）
#define LOG_MEMORY_CN(component, bytes) \
    LOG_CN_INFO(spacego::framework::ChineseLogger::LogCategory::PERFORMANCE, \
                "{} 内存使用: {}", component, spacego::framework::ChineseLogger::format_memory_size(bytes))

/// GPU性能日志（中文）
#define LOG_GPU_CN(gpu_index, utilization, memory_used, memory_total) \
    LOG_CN_INFO(spacego::framework::ChineseLogger::LogCategory::PERFORMANCE, \
                "GPU{} 利用率: {:.1f}%, 显存: {}/{}", \
                gpu_index, utilization, \
                spacego::framework::ChineseLogger::format_memory_size(memory_used), \
                spacego::framework::ChineseLogger::format_memory_size(memory_total))

// ============================================================================
// 错误处理宏
// ============================================================================

/// 记录异常（中文）
#define LOG_EXCEPTION_CN(category, e) \
    LOG_CN_ERROR(category, "异常: {}", spacego::framework::ChineseLogger::translate_exception(e))

/// 记录错误代码（中文）
#define LOG_ERROR_CODE_CN(category, error_code, context) \
    LOG_CN_ERROR(category, "错误代码 {} ({}): {}", error_code, context, \
                spacego::framework::ChineseLogger::translate_error_code(error_code, context))

/// 断言失败日志（中文）
#define LOG_ASSERT_FAIL_CN(condition, category, ...) \
    if (!(condition)) { \
        LOG_CN_FATAL(category, "断言失败: " #condition ". " __VA_ARGS__); \
    }

// ============================================================================
// 组件专用日志宏
// ============================================================================

/// 围棋AI组件日志
#define LOG_GO_AI_CN(level, ...) \
    LOG_CN(level, spacego::framework::ChineseLogger::LogCategory::AI_ENGINE, __VA_ARGS__)

/// 搜索算法日志
#define LOG_SEARCH_ALG_CN(level, ...) \
    LOG_CN(level, spacego::framework::ChineseLogger::LogCategory::SEARCH_ALG, __VA_ARGS__)

/// 神经网络日志
#define LOG_NEURAL_NET_CN(level, ...) \
    LOG_CN(level, spacego::framework::ChineseLogger::LogCategory::NEURAL_NET, __VA_ARGS__)

/// CUDA日志
#define LOG_CUDA_CN(level, ...) \
    LOG_CN(level, spacego::framework::ChineseLogger::LogCategory::CUDA, __VA_ARGS__)

/// 性能监控日志
#define LOG_PERF_MONITOR_CN(level, ...) \
    LOG_CN(level, spacego::framework::ChineseLogger::LogCategory::PERFORMANCE, __VA_ARGS__)

inline void spacego::framework::ChineseLogger::log(
    spacego::framework::LogLevel level,
    spacego::framework::ChineseLogger::LogCategory category,
    const std::string& message,
    const std::string& file, int line, const std::string& function) {
    spacego::framework::Logger::instance().log(level, message, file, line, function);
}