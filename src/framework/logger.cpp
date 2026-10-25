#include "framework/logger.h"

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

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace spacego {
namespace framework {

namespace {
    Language current_language = Language::ENGLISH;
}

void set_log_language(Language language) {
    current_language = language;
}

Language get_log_language() {
    return current_language;
}

std::string LogEntry::to_string(const std::string& format) const {
    std::string result = format;
    
    std::time_t time = std::chrono::system_clock::to_time_t(timestamp);
    std::tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &time);
#else
    localtime_r(&time, &tm_buf);
#endif
    char time_str[64];
    std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", &tm_buf);
    
    auto pos = result.find("{timestamp}");
    while (pos != std::string::npos) {
        result.replace(pos, 11, time_str);
        pos = result.find("{timestamp}", pos + 11);
    }
    
    std::string level_str;
    switch (level) {
        case LogLevel::TRACE: level_str = "TRACE"; break;
        case LogLevel::DEBUG: level_str = "DEBUG"; break;
        case LogLevel::INFO: level_str = "INFO"; break;
        case LogLevel::WARNING: level_str = "WARNING"; break;
        case LogLevel::LVL_ERROR: level_str = "ERROR"; break;
        case LogLevel::FATAL: level_str = "FATAL"; break;
        case LogLevel::OFF: level_str = "OFF"; break;
        default: level_str = "UNKNOWN"; break;
    }
    
    pos = result.find("{level}");
    while (pos != std::string::npos) {
        result.replace(pos, 7, level_str);
        pos = result.find("{level}", pos + level_str.length());
    }
    
    std::stringstream thread_ss;
    thread_ss << std::this_thread::get_id();
    
    pos = result.find("{thread}");
    while (pos != std::string::npos) {
        result.replace(pos, 8, thread_ss.str());
        pos = result.find("{thread}", pos + thread_ss.str().length());
    }
    
    pos = result.find("{message}");
    while (pos != std::string::npos) {
        result.replace(pos, 9, message);
        pos = result.find("{message}", pos + message.length());
    }
    
    pos = result.find("{file}");
    while (pos != std::string::npos) {
        result.replace(pos, 6, source_file);
        pos = result.find("{file}", pos + source_file.length());
    }
    
    auto pos_line = result.find("{line}");
    while (pos_line != std::string::npos) {
        result.replace(pos_line, 6, std::to_string(source_line));
        pos_line = result.find("{line}", pos_line + 6);
    }
    
    auto pos_func = result.find("{function}");
    while (pos_func != std::string::npos) {
        result.replace(pos_func, 10, source_function);
        pos_func = result.find("{function}", pos_func + source_function.length());
    }
    
    return result;
}

std::string LogEntry::default_format() {
    return "[{timestamp}] [{level}] [{thread}] {message}";
}

ConsoleSink::ConsoleSink(bool color_output, bool output_to_stderr) 
    : color_output_(color_output), output_to_stderr_(output_to_stderr) {
}

void ConsoleSink::write(const LogEntry& entry) {
    if (!should_log(entry.level)) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(console_mutex_);
    
    std::string formatted = entry.to_string("[{timestamp}] [{level}] [{thread}] {message}");
    
    if (color_output_) {
        if (output_to_stderr_) {
            std::cerr << get_level_color(entry.level) << formatted << "\033[0m" << std::endl;
        } else {
            std::cout << get_level_color(entry.level) << formatted << "\033[0m" << std::endl;
        }
    } else {
        if (output_to_stderr_) {
            std::cerr << formatted << std::endl;
        } else {
            std::cout << formatted << std::endl;
        }
    }
}

void ConsoleSink::flush() {
    std::lock_guard<std::mutex> lock(console_mutex_);
    if (output_to_stderr_) {
        std::cerr.flush();
    } else {
        std::cout.flush();
    }
}

bool ConsoleSink::should_log(LogLevel level) const {
    return level >= min_level_;
}

void ConsoleSink::set_min_level(LogLevel level) {
    min_level_ = level;
}

void ConsoleSink::set_color_output(bool enable) {
    color_output_ = enable;
}

void ConsoleSink::set_output_to_stderr(bool enable) {
    output_to_stderr_ = enable;
}

void ConsoleSink::set_language(Language language) {
    language_ = language;
}

std::string ConsoleSink::get_level_color(LogLevel level) {
    switch (level) {
        case LogLevel::TRACE: return "\033[37m";
        case LogLevel::DEBUG: return "\033[36m";
        case LogLevel::INFO: return "\033[32m";
        case LogLevel::WARNING: return "\033[33m";
        case LogLevel::LVL_ERROR: return "\033[31m";
        case LogLevel::FATAL: return "\033[35m";
        default: return "\033[0m";
    }
}

std::string ConsoleSink::get_level_string(LogLevel level) {
    Language lang = get_log_language();
    if (lang == Language::CHINESE) {
        switch (level) {
            case LogLevel::TRACE: return "跟踪";
            case LogLevel::DEBUG: return "调试";
            case LogLevel::INFO: return "信息";
            case LogLevel::WARNING: return "警告";
            case LogLevel::LVL_ERROR: return "错误";
            case LogLevel::FATAL: return "致命";
            case LogLevel::OFF: return "关闭";
            default: return "未知";
        }
    } else {
        switch (level) {
            case LogLevel::TRACE: return "TRACE";
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARNING: return "WARNING";
            case LogLevel::LVL_ERROR: return "ERROR";
            case LogLevel::FATAL: return "FATAL";
            case LogLevel::OFF: return "OFF";
            default: return "UNKNOWN";
        }
    }
}

FileSink::FileSink(const std::string& file_path, bool append) 
    : file_path_(file_path), append_(append) {
    if (!file_path_.empty()) {
        file_stream_.open(file_path_, append_ ? std::ios::app : std::ios::out);
        if (!file_stream_.is_open()) {
            file_stream_.clear();
            file_stream_.open(file_path_, std::ios::out);
        }
    }
}

FileSink::~FileSink() {
    if (file_stream_.is_open()) {
        file_stream_.close();
    }
}

void FileSink::write(const LogEntry& entry) {
    if (!should_log(entry.level)) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(file_mutex_);
    
    if (file_stream_.is_open()) {
        file_stream_ << entry.to_string("[{timestamp}] [{level}] [{thread}] {message}") << std::endl;
    }
}

void FileSink::flush() {
    std::lock_guard<std::mutex> lock(file_mutex_);
    if (file_stream_.is_open()) {
        file_stream_.flush();
    }
}

bool FileSink::should_log(LogLevel level) const {
    return level >= min_level_;
}

void FileSink::set_min_level(LogLevel level) {
    min_level_ = level;
}

void FileSink::set_file_path(const std::string& file_path) {
    std::lock_guard<std::mutex> lock(file_mutex_);
    
    if (file_stream_.is_open()) {
        file_stream_.close();
    }
    
    file_path_ = file_path;
    
    if (!file_path_.empty()) {
        file_stream_.open(file_path_, append_ ? std::ios::app : std::ios::out);
    }
}

void FileSink::set_append(bool append) {
    append_ = append;
}

std::string FileSink::get_level_string(LogLevel level) {
    Language lang = get_log_language();
    if (lang == Language::CHINESE) {
        switch (level) {
            case LogLevel::TRACE: return "跟踪";
            case LogLevel::DEBUG: return "调试";
            case LogLevel::INFO: return "信息";
            case LogLevel::WARNING: return "警告";
            case LogLevel::LVL_ERROR: return "错误";
            case LogLevel::FATAL: return "致命";
            case LogLevel::OFF: return "关闭";
            default: return "未知";
        }
    } else {
        switch (level) {
            case LogLevel::TRACE: return "TRACE";
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARNING: return "WARNING";
            case LogLevel::LVL_ERROR: return "ERROR";
            case LogLevel::FATAL: return "FATAL";
            case LogLevel::OFF: return "OFF";
            default: return "UNKNOWN";
        }
    }
}

RotatingFileSink::RotatingFileSink(const std::string& base_path, size_t max_file_size, int max_backup_files)
    : base_path_(base_path), max_file_size_(max_file_size), max_backup_files_(max_backup_files),
      current_file_size_(0) {
    if (!base_path_.empty()) {
        rotate_files();
        current_file_.open(base_path_, std::ios::app);
        if (!current_file_.is_open()) {
            current_file_.clear();
            current_file_.open(base_path_, std::ios::out);
        }
        current_file_size_ = current_file_.tellp();
    }
}

RotatingFileSink::~RotatingFileSink() {
    if (current_file_.is_open()) {
        current_file_.close();
    }
}

void RotatingFileSink::write(const LogEntry& entry) {
    if (!should_log(entry.level)) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(file_mutex_);
    
    std::string formatted = entry.to_string("[{timestamp}] [{level}] [{thread}] {message}") + "\n";
    size_t entry_size = formatted.length();
    
    if (current_file_size_ + entry_size > max_file_size_) {
        rotate_files();
        current_file_.open(base_path_, std::ios::out);
        current_file_size_ = 0;
    }
    
    if (current_file_.is_open()) {
        current_file_ << formatted;
        current_file_.flush();
        current_file_size_ += entry_size;
    }
}

void RotatingFileSink::flush() {
    std::lock_guard<std::mutex> lock(file_mutex_);
    if (current_file_.is_open()) {
        current_file_.flush();
    }
}

bool RotatingFileSink::should_log(LogLevel level) const {
    return level >= min_level_;
}

void RotatingFileSink::set_min_level(LogLevel level) {
    min_level_ = level;
}

void RotatingFileSink::set_base_path(const std::string& base_path) {
    std::lock_guard<std::mutex> lock(file_mutex_);
    
    if (current_file_.is_open()) {
        current_file_.close();
    }
    
    base_path_ = base_path;
    current_file_size_ = 0;
    
    if (!base_path_.empty()) {
        rotate_files();
        current_file_.open(base_path_, std::ios::out);
    }
}

void RotatingFileSink::set_max_file_size(size_t size) {
    max_file_size_ = size;
}

void RotatingFileSink::set_max_backup_files(int count) {
    max_backup_files_ = count;
}

void RotatingFileSink::rotate_files() {
    if (max_backup_files_ <= 0) {
        return;
    }
    
    std::string max_backup_path = base_path_ + "." + std::to_string(max_backup_files_);
    
    if (std::ifstream(max_backup_path).good()) {
        std::remove(max_backup_path.c_str());
    }
    
    for (int i = max_backup_files_ - 1; i >= 1; --i) {
        std::string old_path = base_path_ + "." + std::to_string(i);
        std::string new_path = base_path_ + "." + std::to_string(i + 1);
        
        if (std::ifstream(old_path).good()) {
            std::rename(old_path.c_str(), new_path.c_str());
        }
    }
    
    std::string current_backup_path = base_path_ + ".1";
    if (std::ifstream(base_path_).good()) {
        std::rename(base_path_.c_str(), current_backup_path.c_str());
    }
}

std::string RotatingFileSink::get_level_string(LogLevel level) {
    Language lang = get_log_language();
    if (lang == Language::CHINESE) {
        switch (level) {
            case LogLevel::TRACE: return "跟踪";
            case LogLevel::DEBUG: return "调试";
            case LogLevel::INFO: return "信息";
            case LogLevel::WARNING: return "警告";
            case LogLevel::LVL_ERROR: return "错误";
            case LogLevel::FATAL: return "致命";
            case LogLevel::OFF: return "关闭";
            default: return "未知";
        }
    } else {
        switch (level) {
            case LogLevel::TRACE: return "TRACE";
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARNING: return "WARNING";
            case LogLevel::LVL_ERROR: return "ERROR";
            case LogLevel::FATAL: return "FATAL";
            case LogLevel::OFF: return "OFF";
            default: return "UNKNOWN";
        }
    }
}

Logger::Logger() 
    : running_(false), flush_requested_(false) {
}

Logger::~Logger() {
    shutdown();
}

Logger& Logger::instance() {
    static Logger instance_;
    return instance_;
}

void Logger::initialize(const std::string& app_name) {
    app_name_ = app_name;
    pattern_ = "[{timestamp}] [{level}] [{thread}] {message}";
    
    running_ = true;
    flush_requested_ = false;
    last_flush_time_ = std::chrono::system_clock::now();
    
    async_thread_ = std::make_unique<std::thread>(&Logger::async_write_loop, this);
}

void Logger::shutdown() {
    running_ = false;
    flush_requested_ = true;
    
    async_cond_.notify_all();
    
    if (async_thread_ && async_thread_->joinable()) {
        async_thread_->join();
    }
    
    {
        std::lock_guard<std::mutex> lock(async_mutex_);
        while (!async_queue_.empty()) {
            async_queue_.pop();
        }
    }
    
    clear_sinks();
}

void Logger::add_sink(std::unique_ptr<LogSink> sink) {
    std::lock_guard<std::mutex> lock(sinks_mutex_);
    sinks_.push_back(std::move(sink));
}

void Logger::remove_sink(LogSink* sink) {
    std::lock_guard<std::mutex> lock(sinks_mutex_);
    auto it = std::find_if(sinks_.begin(), sinks_.end(),
        [sink](const std::unique_ptr<LogSink>& ptr) {
            return ptr.get() == sink;
        });
    
    if (it != sinks_.end()) {
        sinks_.erase(it);
    }
}

void Logger::clear_sinks() {
    std::lock_guard<std::mutex> lock(sinks_mutex_);
    sinks_.clear();
}

void Logger::set_min_level(LogLevel level) {
    min_level_ = level;
}

void Logger::set_pattern(const std::string& pattern) {
    pattern_ = pattern;
}

void Logger::set_app_name(const std::string& app_name) {
    app_name_ = app_name;
}

void Logger::set_thread_name(const std::string& name) {
    thread_name_ = name;
}

void Logger::set_language(Language language) {
    language_ = language;
    set_log_language(language);
    for (auto& sink : sinks_) {
        sink->set_language(language);
    }
}

void Logger::set_flush_level(LogLevel level) {
    flush_level_ = level;
}

void Logger::set_flush_interval(int milliseconds) {
    flush_interval_ms_ = milliseconds;
}

void Logger::flush() {
    std::lock_guard<std::mutex> lock(sinks_mutex_);
    for (auto& sink : sinks_) {
        sink->flush();
    }
}

void Logger::log(LogLevel level, const std::string& message,
                const std::string& file, int line, const std::string& function) {
    if (level < min_level_) {
        return;
    }
    
    LogEntry entry;
    entry.timestamp = std::chrono::system_clock::now();
    entry.level = level;
    entry.message = message;
    entry.source_file = file;
    entry.source_line = line;
    entry.source_function = function;
    entry.thread_id = std::this_thread::get_id();
    
    std::string formatted = format_message(pattern_, level, message);
    
    {
        std::lock_guard<std::mutex> lock(async_mutex_);
        async_queue_.push({entry, formatted});
    }
    
    async_cond_.notify_one();
}

void Logger::trace(const std::string& message, const std::string& file, int line, const std::string& function) {
    log(LogLevel::TRACE, message, file, line, function);
}

void Logger::debug(const std::string& message, const std::string& file, int line, const std::string& function) {
    log(LogLevel::DEBUG, message, file, line, function);
}

void Logger::info(const std::string& message, const std::string& file, int line, const std::string& function) {
    log(LogLevel::INFO, message, file, line, function);
}

void Logger::warning(const std::string& message, const std::string& file, int line, const std::string& function) {
    log(LogLevel::WARNING, message, file, line, function);
}

void Logger::error(const std::string& message, const std::string& file, int line, const std::string& function) {
    log(LogLevel::LVL_ERROR, message, file, line, function);
}

void Logger::fatal(const std::string& message, const std::string& file, int line, const std::string& function) {
    log(LogLevel::FATAL, message, file, line, function);
}

void Logger::write_log(LogLevel level, const std::string& message,
                      const std::string& file, int line, const std::string& function) {
    log(level, message, file, line, function);
}

std::string Logger::format_message(const std::string& format, LogLevel level, const std::string& message) {
    std::string result = format;
    
    auto pos = result.find("{timestamp}");
    if (pos != std::string::npos) {
        auto now = std::chrono::system_clock::now();
        std::time_t time = std::chrono::system_clock::to_time_t(now);
        std::tm tm_buf;
#ifdef _WIN32
        localtime_s(&tm_buf, &time);
#else
        localtime_r(&time, &tm_buf);
#endif
        char buffer[64];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm_buf);
        result.replace(pos, 11, buffer);
    }
    
    std::string level_str;
    switch (level) {
        case LogLevel::TRACE: level_str = "TRACE"; break;
        case LogLevel::DEBUG: level_str = "DEBUG"; break;
        case LogLevel::INFO: level_str = "INFO"; break;
        case LogLevel::WARNING: level_str = "WARNING"; break;
        case LogLevel::LVL_ERROR: level_str = "ERROR"; break;
        case LogLevel::FATAL: level_str = "FATAL"; break;
        case LogLevel::OFF: level_str = "OFF"; break;
        default: level_str = "UNKNOWN"; break;
    }
    
    pos = result.find("{level}");
    while (pos != std::string::npos) {
        result.replace(pos, 7, level_str);
        pos = result.find("{level}", pos + level_str.length());
    }
    
    std::stringstream thread_ss;
    thread_ss << std::this_thread::get_id();
    
    pos = result.find("{thread}");
    while (pos != std::string::npos) {
        result.replace(pos, 8, thread_ss.str());
        pos = result.find("{thread}", pos + thread_ss.str().length());
    }
    
    pos = result.find("{message}");
    while (pos != std::string::npos) {
        result.replace(pos, 9, message);
        pos = result.find("{message}", pos + message.length());
    }
    
    return result;
}

void Logger::async_write_loop() {
    while (running_ || !async_queue_.empty()) {
        std::unique_lock<std::mutex> lock(async_mutex_);
        async_cond_.wait_for(lock, std::chrono::milliseconds(100),
            [this]() {
                return !running_ || !async_queue_.empty() || flush_requested_;
            });
        
        bool needs_flush = flush_requested_;
        flush_requested_ = false;
        
        while (!async_queue_.empty()) {
            auto [entry, formatted] = std::move(async_queue_.front());
            async_queue_.pop();
            
            lock.unlock();
            
            {
                std::lock_guard<std::mutex> sinks_lock(sinks_mutex_);
                for (auto& sink : sinks_) {
                    sink->write(entry);
                }
            }
            
            if (entry.level <= flush_level_ || needs_flush) {
                std::lock_guard<std::mutex> sinks_lock(sinks_mutex_);
                for (auto& sink : sinks_) {
                    sink->flush();
                }
            }
            
            lock.lock();
        }
        
        auto now = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_flush_time_).count();
        
        if (elapsed >= flush_interval_ms_) {
            std::lock_guard<std::mutex> sinks_lock(sinks_mutex_);
            for (auto& sink : sinks_) {
                sink->flush();
            }
            last_flush_time_ = now;
        }
    }
}

void Logger::update_thread_name() {
}

} // namespace framework
} // namespace spacego
