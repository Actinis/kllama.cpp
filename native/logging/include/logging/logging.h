#ifndef ACTINIS_LOGGING_H
#define ACTINIS_LOGGING_H

#include <string>
#include <sstream>


// Define log levels
#define LOG_LEVEL_DEBUG 0
#define LOG_LEVEL_INFO 1
#define LOG_LEVEL_WARN 2
#define LOG_LEVEL_ERROR 3

// Choose the minimum log level here
// #if defined(IS_RELEASE_BUILD) && IS_RELEASE_BUILD
// #define LOG_LEVEL_MIN LOG_LEVEL_INFO
// #else
#define LOG_LEVEL_MIN LOG_LEVEL_DEBUG
// #endif

#if defined(ANDROID)
#include <android/log.h>
#define DEFAULT_LOG_TAG "ActinisRemoteClient"
#elif defined(__APPLE__) && defined(__MACH__)
#include <os/log.h>
#define DEFAULT_LOG_TAG "ActinisRemoteClient"
#else
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include <cstdio>
#include <iostream>
#include <vector>
#endif

// Platform-specific initialization
#if !defined(ANDROID) && !(defined(__APPLE__) && defined(__MACH__))
namespace {
    class LoggerManager {
    public:
        static std::shared_ptr<spdlog::logger> &get_logger() {
            static LoggerManager instance;
            return instance.logger;
        }

    private:
        LoggerManager() {
            try {
                logger = spdlog::get("console");
                if (!logger) {
                    logger = spdlog::stdout_color_mt("console");
                }
                logger->set_level(spdlog::level::debug);
                logger->flush_on(spdlog::level::debug);
            } catch (const spdlog::spdlog_ex &ex) {
                std::cerr << "Logger initialization failed: " << ex.what() << std::endl;
            }
        }

        std::shared_ptr<spdlog::logger> logger;
    };

    template<typename... Args>
    std::string format_log_message(const char *fmt, Args... args) {
        int size = snprintf(nullptr, 0, fmt, args...) + 1;
        if (size <= 0) { return "Error during formatting."; }
        std::vector<char> buf(size);
        snprintf(buf.data(), size, fmt, args...);
        return std::string(buf.data(), buf.data() + size - 1);
    }
}
#endif

// Main logging macro
#define LOG_PRINT(level, tag, fmt, ...) \
    do { \
        if (level >= LOG_LEVEL_MIN) { \
            _LOG_PRINT_IMPL(level, tag, fmt, ##__VA_ARGS__); \
        } \
    } while(0)

// Platform-specific implementation
#if defined(ANDROID)
#define _LOG_PRINT_IMPL(level, tag, fmt, ...) \
        __android_log_print(android_log_priority(level), tag, fmt, ##__VA_ARGS__)
#elif defined(__APPLE__) && defined(__MACH__)
inline os_log_type_t apple_log_type(const int level) {
    switch (level) {
        case LOG_LEVEL_DEBUG: return OS_LOG_TYPE_INFO; // Otherwise it won't log
        case LOG_LEVEL_INFO: return OS_LOG_TYPE_INFO;
        case LOG_LEVEL_WARN: return OS_LOG_TYPE_DEFAULT;
        case LOG_LEVEL_ERROR: return OS_LOG_TYPE_ERROR;
        default: return OS_LOG_TYPE_DEFAULT;
    }
}

#define _LOG_PRINT_IMPL(level, tag, fmt, ...) \
os_log_with_type(OS_LOG_DEFAULT, apple_log_type(level), "[%s] " fmt, tag, ##__VA_ARGS__)
#else
#define _LOG_PRINT_IMPL(level, tag, fmt, ...) \
        LoggerManager::get_logger()->log(spdlog_level(level), "[{}] {}", tag, format_log_message(fmt, ##__VA_ARGS__))
#endif

// Helper functions
#ifdef ANDROID
inline android_LogPriority android_log_priority(const int level) {
    switch (level) {
        case LOG_LEVEL_DEBUG: return ANDROID_LOG_DEBUG;
        case LOG_LEVEL_INFO: return ANDROID_LOG_INFO;
        case LOG_LEVEL_WARN: return ANDROID_LOG_WARN;
        case LOG_LEVEL_ERROR: return ANDROID_LOG_ERROR;
        default: return ANDROID_LOG_DEFAULT;
    }
}
#elif defined(__APPLE__) && defined(__MACH__)

#else
inline spdlog::level::level_enum spdlog_level(const int level) {
    switch (level) {
        case LOG_LEVEL_DEBUG: return spdlog::level::debug;
        case LOG_LEVEL_INFO: return spdlog::level::info;
        case LOG_LEVEL_WARN: return spdlog::level::warn;
        case LOG_LEVEL_ERROR: return spdlog::level::err;
        default: return spdlog::level::info;
    }
}
#endif

// Convenience macros for different log levels
#define LOG_DEBUG(tag, fmt, ...) LOG_PRINT(LOG_LEVEL_DEBUG, tag, fmt, ##__VA_ARGS__)
#define LOG_INFO(tag, fmt, ...)  LOG_PRINT(LOG_LEVEL_INFO,  tag, fmt, ##__VA_ARGS__)
#define LOG_WARN(tag, fmt, ...)  LOG_PRINT(LOG_LEVEL_WARN,  tag, fmt, ##__VA_ARGS__)
#define LOG_ERROR(tag, fmt, ...) LOG_PRINT(LOG_LEVEL_ERROR, tag, fmt, ##__VA_ARGS__)

// Macros with default tag
#define LOG_DEBUG_DEFAULT(fmt, ...) LOG_DEBUG(DEFAULT_LOG_TAG, fmt, ##__VA_ARGS__)
#define LOG_INFO_DEFAULT(fmt, ...)  LOG_INFO(DEFAULT_LOG_TAG, fmt, ##__VA_ARGS__)
#define LOG_WARN_DEFAULT(fmt, ...)  LOG_WARN(DEFAULT_LOG_TAG, fmt, ##__VA_ARGS__)
#define LOG_ERROR_DEFAULT(fmt, ...) LOG_ERROR(DEFAULT_LOG_TAG, fmt, ##__VA_ARGS__)

#endif
