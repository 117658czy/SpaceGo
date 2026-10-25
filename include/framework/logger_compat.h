#pragma once

#include "framework/logger.h"

namespace spacego {
namespace utils {

using Level = framework::LogLevel;

class Logger {
public:
    static Logger& instance() {
        static Logger inst;
        return inst;
    }

    void set_level(Level level) {
        framework::Logger::instance().set_min_level(level);
    }

    Level get_level() const {
        return Level::TRACE;
    }

    void set_log_file(const std::string& file_path) {
        framework::Logger::instance().add_sink(
            std::make_unique<framework::FileSink>(file_path, true));
    }

    void log(Level level, const std::string& message) {
        framework::Logger::instance().log(level, message);
    }

    void debug(const std::string& message) {
        framework::Logger::instance().debug(message);
    }

    void info(const std::string& message) {
        framework::Logger::instance().info(message);
    }

    void warning(const std::string& message) {
        framework::Logger::instance().warning(message);
    }

    void error(const std::string& message) {
        framework::Logger::instance().error(message);
    }

    void fatal(const std::string& message) {
        framework::Logger::instance().fatal(message);
    }

    void set_log_callback(std::function<void(Level, const std::string&)> cb) {
        (void)cb;
    }
};

}

namespace framework { }

}

namespace SpaceGo {
namespace GUI {

enum class GUILogLevel {
    Debug,
    Info,
    Warning,
    Error,
    Critical
};

inline spacego::framework::LogLevel map_gui_level(GUILogLevel level) {
    switch (level) {
        case GUILogLevel::Debug:    return spacego::framework::LogLevel::DEBUG;
        case GUILogLevel::Info:     return spacego::framework::LogLevel::INFO;
        case GUILogLevel::Warning:  return spacego::framework::LogLevel::WARNING;
        case GUILogLevel::Error:    return spacego::framework::LogLevel::LVL_ERROR;
        case GUILogLevel::Critical: return spacego::framework::LogLevel::FATAL;
        default:                    return spacego::framework::LogLevel::INFO;
    }
}

class Logger {
public:
    static Logger& GetInstance() {
        static Logger instance;
        return instance;
    }

    void SetLogLevel(GUILogLevel level) {
        spacego::framework::Logger::instance().set_min_level(map_gui_level(level));
    }

    void SetLogToConsole(bool enable) {
        (void)enable;
    }

    void SetLogToFile(bool enable) {
        (void)enable;
    }

    void Log(GUILogLevel level, const std::string& message, const char* file = nullptr, int line = 0) {
        auto fw_level = map_gui_level(level);
        spacego::framework::Logger::instance().log(fw_level, message,
            file ? file : "", line, "");
    }

    void Debug(const std::string& message, const char* file = nullptr, int line = 0) {
        Log(GUILogLevel::Debug, message, file, line);
    }

    void Info(const std::string& message, const char* file = nullptr, int line = 0) {
        Log(GUILogLevel::Info, message, file, line);
    }

    void Warning(const std::string& message, const char* file = nullptr, int line = 0) {
        Log(GUILogLevel::Warning, message, file, line);
    }

    void Error(const std::string& message, const char* file = nullptr, int line = 0) {
        Log(GUILogLevel::Error, message, file, line);
    }

    void Critical(const std::string& message, const char* file = nullptr, int line = 0) {
        Log(GUILogLevel::Critical, message, file, line);
    }

    void Close() {}
};

}
}
