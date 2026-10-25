#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <string>
#include <memory>
#include <functional>
#include <vector>
#include <mutex>
#include <atomic>
#include <queue>
#include <condition_variable>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <thread>
#include <fstream>
#include <limits>
#include <cstdio>

namespace spacego {
namespace framework {

enum class LogLevel {
    TRACE,
    DEBUG,
    INFO,
    WARNING,
    LVL_ERROR,
    FATAL,
    OFF
};

enum class Language {
    ENGLISH,
    CHINESE
};

struct LogEntry {
    std::chrono::system_clock::time_point timestamp;
    LogLevel level;
    std::string message;
    std::string source_file;
    int source_line;
    std::string source_function;
    std::thread::id thread_id;
    
    std::string to_string(const std::string& format = "[{timestamp}] [{level}] [{thread}] {message}") const;
    
    static std::string default_format();
};

class LogSink {
public:
    virtual ~LogSink() = default;
    
    virtual void write(const LogEntry& entry) = 0;
    
    virtual void flush() = 0;
    
    virtual bool should_log(LogLevel level) const = 0;
    
    virtual void set_language(Language language) = 0;
};

class ConsoleSink : public LogSink {
public:
    ConsoleSink(bool color_output = true, bool output_to_stderr = false);
    ~ConsoleSink() override = default;
    
    void write(const LogEntry& entry) override;
    
    void flush() override;
    
    bool should_log(LogLevel level) const override;
    
    void set_min_level(LogLevel level);
    
    void set_color_output(bool enable);
    
    void set_output_to_stderr(bool enable);
    
    void set_language(Language language) override;

private:
    Language language_ = Language::ENGLISH;
    LogLevel min_level_ = LogLevel::TRACE;
    bool color_output_ = true;
    bool output_to_stderr_ = false;
    std::mutex console_mutex_;
    
    std::string get_level_color(LogLevel level);
    
    std::string get_level_string(LogLevel level);
};

class FileSink : public LogSink {
public:
    FileSink(const std::string& file_path, bool append = true);
    ~FileSink() override;
    
    void write(const LogEntry& entry) override;
    
    void flush() override;
    
    bool should_log(LogLevel level) const override;
    
    void set_min_level(LogLevel level);
    
    void set_file_path(const std::string& file_path);
    
    void set_append(bool append);
    
    void set_language(Language language) override {}

private:
    std::string file_path_;
    bool append_;
    LogLevel min_level_ = LogLevel::TRACE;
    std::ofstream file_stream_;
    std::mutex file_mutex_;
    
    std::string get_level_string(LogLevel level);
};

class RotatingFileSink : public LogSink {
public:
    RotatingFileSink(const std::string& base_path, size_t max_file_size, int max_backup_files);
    ~RotatingFileSink() override;
    
    void write(const LogEntry& entry) override;
    
    void flush() override;
    
    bool should_log(LogLevel level) const override;
    
    void set_min_level(LogLevel level);
    
    void set_base_path(const std::string& base_path);
    
    void set_max_file_size(size_t size);
    
    void set_max_backup_files(int count);
    
    void set_language(Language language) override {}

private:
    std::string base_path_;
    size_t max_file_size_;
    int max_backup_files_;
    LogLevel min_level_ = LogLevel::TRACE;
    std::ofstream current_file_;
    size_t current_file_size_;
    std::mutex file_mutex_;
    
    void rotate_files();
    
    std::string get_level_string(LogLevel level);
};

class Logger {
public:
    static Logger& instance();
    
    void initialize(const std::string& app_name = "SpaceGo");
    
    void shutdown();
    
    void add_sink(std::unique_ptr<LogSink> sink);
    
    void remove_sink(LogSink* sink);
    
    void clear_sinks();
    
    void set_min_level(LogLevel level);
    
    void set_pattern(const std::string& pattern);
    
    void set_app_name(const std::string& app_name);
    
    void set_language(Language language);
    
    void set_thread_name(const std::string& name);
    
    void set_flush_level(LogLevel level);
    
    void set_flush_interval(int milliseconds);
    
    void flush();
    
    void log(LogLevel level, const std::string& message,
             const std::string& file = "", int line = 0,
             const std::string& function = "");
    
    void trace(const std::string& message, const std::string& file = "", int line = 0, const std::string& function = "");
    void debug(const std::string& message, const std::string& file = "", int line = 0, const std::string& function = "");
    void info(const std::string& message, const std::string& file = "", int line = 0, const std::string& function = "");
    void warning(const std::string& message, const std::string& file = "", int line = 0, const std::string& function = "");
    void error(const std::string& message, const std::string& file = "", int line = 0, const std::string& function = "");
    void fatal(const std::string& message, const std::string& file = "", int line = 0, const std::string& function = "");
    
    template<typename... Args>
    void trace(const std::string& format, Args... args);
    
    template<typename... Args>
    void debug(const std::string& format, Args... args);
    
    template<typename... Args>
    void info(const std::string& format, Args... args);
    
    template<typename... Args>
    void warning(const std::string& format, Args... args);
    
    template<typename... Args>
    void error(const std::string& format, Args... args);
    
    template<typename... Args>
    void fatal(const std::string& format, Args... args);

private:
    Logger();
    ~Logger();
    
    void write_log(LogLevel level, const std::string& message,
                   const std::string& file, int line, const std::string& function);
    
    std::string format_message(const std::string& format, LogLevel level, const std::string& message);
    
    void async_write_loop();
    
    void update_thread_name();
    
    std::string app_name_;
    std::string thread_name_;
    std::string pattern_;
    Language language_ = Language::ENGLISH;
    LogLevel min_level_ = LogLevel::TRACE;
    LogLevel flush_level_ = LogLevel::LVL_ERROR;
    int flush_interval_ms_ = 3000;
    
    std::vector<std::unique_ptr<LogSink>> sinks_;
    std::mutex sinks_mutex_;
    
    std::queue<std::pair<LogEntry, std::string>> async_queue_;
    std::mutex async_mutex_;
    std::condition_variable async_cond_;
    std::unique_ptr<std::thread> async_thread_;
    std::atomic<bool> running_;
    std::atomic<bool> flush_requested_;
    
    std::chrono::system_clock::time_point last_flush_time_;
};

template<typename... Args>
void Logger::trace(const std::string& format, Args... args) {
    char buffer[4096];
    snprintf(buffer, sizeof(buffer), format.c_str(), args...);
    trace(buffer);
}

template<typename... Args>
void Logger::debug(const std::string& format, Args... args) {
    char buffer[4096];
    snprintf(buffer, sizeof(buffer), format.c_str(), args...);
    debug(buffer);
}

template<typename... Args>
void Logger::info(const std::string& format, Args... args) {
    char buffer[4096];
    snprintf(buffer, sizeof(buffer), format.c_str(), args...);
    info(buffer);
}

template<typename... Args>
void Logger::warning(const std::string& format, Args... args) {
    char buffer[4096];
    snprintf(buffer, sizeof(buffer), format.c_str(), args...);
    warning(buffer);
}

template<typename... Args>
void Logger::error(const std::string& format, Args... args) {
    char buffer[4096];
    snprintf(buffer, sizeof(buffer), format.c_str(), args...);
    error(buffer);
}

template<typename... Args>
void Logger::fatal(const std::string& format, Args... args) {
    char buffer[4096];
    snprintf(buffer, sizeof(buffer), format.c_str(), args...);
    fatal(buffer);
}

// ============================================================================
// 便捷日志宏定义
// 使用方法: LOG_INFO("消息"); 或 LOG_INFO("格式 %s %d", str, num);
// 注意: 使用 printf 风格格式 (%s, %d, %zu, %.2f 等)，不支持 {} 格式
// ============================================================================

/// 记录 TRACE 级别日志
#define LOG_TRACE(...) spacego::framework::Logger::instance().trace(__VA_ARGS__)

/// 记录 DEBUG 级别日志
#define LOG_DEBUG(...) spacego::framework::Logger::instance().debug(__VA_ARGS__)

/// 记录 INFO 级别日志
#define LOG_INFO(...) spacego::framework::Logger::instance().info(__VA_ARGS__)

/// 记录 WARNING 级别日志
#define LOG_WARNING(...) spacego::framework::Logger::instance().warning(__VA_ARGS__)

/// 记录 ERROR 级别日志
#define LOG_ERROR(...) spacego::framework::Logger::instance().error(__VA_ARGS__)

/// 记录 FATAL 级别日志
#define LOG_FATAL(...) spacego::framework::Logger::instance().fatal(__VA_ARGS__)

} // namespace framework
} // namespace spacego
