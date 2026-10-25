#pragma once

#include "framework/logger_compat.h"
#include <sstream>

namespace SpaceGo {
namespace GUI {

using LogLevel = GUILogLevel;

inline Logger& GetLoggerInstance() {
    return Logger::GetInstance();
}

#define GUI_LOG_DEBUG(msg) SpaceGo::GUI::Logger::GetInstance().Debug(msg, __FILE__, __LINE__)
#define GUI_LOG_INFO(msg) SpaceGo::GUI::Logger::GetInstance().Info(msg, __FILE__, __LINE__)
#define GUI_LOG_WARNING(msg) SpaceGo::GUI::Logger::GetInstance().Warning(msg, __FILE__, __LINE__)
#define GUI_LOG_ERROR(msg) SpaceGo::GUI::Logger::GetInstance().Error(msg, __FILE__, __LINE__)
#define GUI_LOG_CRITICAL(msg) SpaceGo::GUI::Logger::GetInstance().Critical(msg, __FILE__, __LINE__)

#define GUI_LOG_EXCEPTION(msg) \
    do { \
        std::ostringstream oss; \
        oss << msg; \
        GUI_LOG_ERROR(oss.str()); \
    } while(0)

#define GUI_LOG_CHECK(condition, msg) \
    do { \
        if (!(condition)) { \
            GUI_LOG_WARNING(msg); \
        } \
    } while(0)

#define GUI_LOG_FATAL(condition, msg) \
    do { \
        if (!(condition)) { \
            GUI_LOG_CRITICAL(msg); \
        } \
    } while(0)

} // namespace GUI
} // namespace SpaceGo
