#include "framework/algorithm_base.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace spacego {
namespace framework {

bool AlgorithmBase::initialize() {
    if (initialized_.load()) {
        return true;
    }
    
    set_status("Initializing...");
    
    if (!on_initialize()) {
        set_status("Initialization failed");
        return false;
    }
    
    initialized_ = true;
    set_status("Initialized");
    notify_callback("initialized", std::any(nullptr));
    
    return true;
}

bool AlgorithmBase::execute() {
    if (!initialized_.load()) {
        if (!initialize()) {
            return false;
        }
    }
    
    if (running_.load()) {
        return true;
    }
    
    set_status("Running");
    running_ = true;
    idle_ = false;
    
    bool result = on_execute();
    
    if (!result) {
        running_ = false;
        idle_ = true;
        set_status("Execution failed");
    }
    
    return result;
}

bool AlgorithmBase::shutdown() {
    if (!initialized_.load()) {
        return true;
    }
    
    stop();
    
    if (!on_shutdown()) {
        set_status("Shutdown failed");
        return false;
    }
    
    initialized_ = false;
    running_ = false;
    idle_ = true;
    
    set_status("Shutdown complete");
    notify_callback("shutdown", std::any(nullptr));
    
    return true;
}

bool AlgorithmBase::is_initialized() const {
    return initialized_.load();
}

bool AlgorithmBase::is_running() const {
    return running_.load();
}

bool AlgorithmBase::is_idle() const {
    return idle_.load();
}

std::string AlgorithmBase::get_name() const {
    return name_;
}

std::string AlgorithmBase::get_version() const {
    return version_;
}

std::string AlgorithmBase::get_status() const {
    return status_;
}

void AlgorithmBase::set_name(const std::string& name) {
    name_ = name;
}

void AlgorithmBase::set_version(const std::string& version) {
    version_ = version;
}

void AlgorithmBase::pause() {
    pause_requested_ = true;
    set_status("Paused");
    notify_callback("paused", std::any(nullptr));
}

void AlgorithmBase::resume() {
    pause_requested_ = false;
    pause_cond_.notify_all();
    set_status("Running");
    notify_callback("resumed", std::any(nullptr));
}

void AlgorithmBase::stop() {
    stop_requested_ = true;
    pause_cond_.notify_all();
    
    if (execution_thread_.joinable()) {
        execution_thread_.join();
    }
    
    running_ = false;
    idle_ = true;
    set_status("Stopped");
    notify_callback("stopped", std::any(nullptr));
}

double AlgorithmBase::get_cpu_usage() const {
    return 0.0;
}

double AlgorithmBase::get_memory_usage() const {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
    }
    return 0.0;
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return static_cast<double>(usage.ru_maxrss) / 1024.0;
    }
    return 0.0;
#endif
}

size_t AlgorithmBase::get_tasks_completed() const {
    return tasks_completed_.load();
}

size_t AlgorithmBase::get_tasks_in_progress() const {
    return tasks_in_progress_.load();
}

void AlgorithmBase::reset_statistics() {
    tasks_completed_ = 0;
    tasks_in_progress_ = 0;
}

void AlgorithmBase::register_callback(std::function<void(const std::string&, const std::any&)> callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    callback_ = std::move(callback);
}

void AlgorithmBase::set_parameter(const std::string& key, const std::any& value) {
    std::lock_guard<std::mutex> lock(parameters_mutex_);
    parameters_[key] = value;
    notify_callback("parameter_changed", std::any(key));
}

std::any AlgorithmBase::get_parameter(const std::string& key) const {
    std::lock_guard<std::mutex> lock(parameters_mutex_);
    auto it = parameters_.find(key);
    if (it != parameters_.end()) {
        return it->second;
    }
    return std::any();
}

void AlgorithmBase::clear_parameters() {
    std::lock_guard<std::mutex> lock(parameters_mutex_);
    parameters_.clear();
}

void AlgorithmBase::set_running(bool running) {
    running_ = running;
}

void AlgorithmBase::set_idle(bool idle) {
    idle_ = idle;
}

void AlgorithmBase::set_initialized(bool initialized) {
    initialized_ = initialized;
}

void AlgorithmBase::set_status(const std::string& status) {
    status_ = status;
}

void AlgorithmBase::increment_tasks_completed() {
    ++tasks_completed_;
}

void AlgorithmBase::decrement_tasks_in_progress() {
    if (tasks_in_progress_ > 0) {
        --tasks_in_progress_;
    }
}

void AlgorithmBase::increment_tasks_in_progress() {
    ++tasks_in_progress_;
}

void AlgorithmBase::notify_callback(const std::string& event, const std::any& data) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    if (callback_) {
        callback_(event, data);
    }
}

AlgorithmRegistry& AlgorithmRegistry::instance() {
    static AlgorithmRegistry instance_;
    return instance_;
}

void AlgorithmRegistry::register_algorithm(const std::string& name, 
                                         std::function<std::unique_ptr<AlgorithmBase>()> creator) {
    std::lock_guard<std::mutex> lock(mutex_);
    algorithms_[name] = std::move(creator);
}

void AlgorithmRegistry::unregister_algorithm(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    algorithms_.erase(name);
}

std::unique_ptr<AlgorithmBase> AlgorithmRegistry::create_algorithm(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = algorithms_.find(name);
    if (it != algorithms_.end() && it->second) {
        return it->second();
    }
    return nullptr;
}

std::vector<std::string> AlgorithmRegistry::get_algorithm_names() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> names;
    names.reserve(algorithms_.size());
    for (const auto& pair : algorithms_) {
        names.push_back(pair.first);
    }
    return names;
}

bool AlgorithmRegistry::has_algorithm(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return algorithms_.find(name) != algorithms_.end();
}

} // namespace framework
} // namespace spacego
