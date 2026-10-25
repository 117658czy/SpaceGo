#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

/**
 * @file algorithm_base.h
 * @brief Algorithm Base Class - 算法基类
 * @details 提供统一的算法生命周期管理、线程管理、锁管理、性能统计和错误处理机制
 *          所有算法类应该继承自此类以获得统一的基础设施支持
 */

#include <string>
#include <memory>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>
#include <vector>
#include <unordered_map>
#include <functional>
#include <stdexcept>
#include <variant>
#include <optional>
#include <any>
#include <condition_variable>

namespace spacego {
namespace framework {

class AlgorithmBase {
public:
    AlgorithmBase() = default;
    explicit AlgorithmBase(const std::string& name) : name_(name) {}
    virtual ~AlgorithmBase() = default;

    virtual bool initialize();
    virtual bool execute();
    virtual bool shutdown();
    
    virtual bool is_initialized() const;
    virtual bool is_running() const;
    virtual bool is_idle() const;
    
    virtual std::string get_name() const;
    virtual std::string get_version() const;
    virtual std::string get_status() const;
    
    virtual void set_name(const std::string& name);
    virtual void set_version(const std::string& version);
    
    virtual void pause();
    virtual void resume();
    virtual void stop();
    
    virtual double get_cpu_usage() const;
    virtual double get_memory_usage() const;
    virtual size_t get_tasks_completed() const;
    virtual size_t get_tasks_in_progress() const;
    
    virtual void reset_statistics();
    
    virtual void register_callback(std::function<void(const std::string&, const std::any&)> callback);
    
    virtual void set_parameter(const std::string& key, const std::any& value);
    virtual std::any get_parameter(const std::string& key) const;
    
    template<typename T>
    void set_parameter(const std::string& key, const T& value);
    
    template<typename T>
    std::optional<T> get_parameter(const std::string& key) const;
    
    template<typename T>
    T get_parameter_or(const std::string& key, const T& default_value) const;
    
    virtual void clear_parameters();

protected:
    virtual bool on_initialize() { return true; }
    virtual bool on_execute() { return true; }
    virtual bool on_shutdown() { return true; }
    virtual void on_pause() {}
    virtual void on_resume() {}
    virtual void on_stop() {}
    
    void set_running(bool running);
    void set_idle(bool idle);
    void set_initialized(bool initialized);
    void set_status(const std::string& status);
    
    void increment_tasks_completed();
    void decrement_tasks_in_progress();
    void increment_tasks_in_progress();
    
    void notify_callback(const std::string& event, const std::any& data);
    
    std::string name_;
    std::string version_ = "1.0.0";
    std::string status_;
    
    std::atomic<bool> initialized_ = false;
    std::atomic<bool> running_ = false;
    std::atomic<bool> idle_ = true;
    
    std::atomic<size_t> tasks_completed_ = 0;
    std::atomic<size_t> tasks_in_progress_ = 0;
    
    std::unordered_map<std::string, std::any> parameters_;
    mutable std::mutex parameters_mutex_;
    
    std::function<void(const std::string&, const std::any&)> callback_;
    mutable std::mutex callback_mutex_;
    
    std::thread execution_thread_;
    std::atomic<bool> stop_requested_ = false;
    std::atomic<bool> pause_requested_ = false;
    
    std::mutex pause_mutex_;
    std::condition_variable pause_cond_;
};

template<typename T>
void AlgorithmBase::set_parameter(const std::string& key, const T& value) {
    std::lock_guard<std::mutex> lock(parameters_mutex_);
    parameters_[key] = std::any(value);
    notify_callback("parameter_changed", std::any(key));
}

template<typename T>
std::optional<T> AlgorithmBase::get_parameter(const std::string& key) const {
    std::lock_guard<std::mutex> lock(parameters_mutex_);
    auto it = parameters_.find(key);
    if (it != parameters_.end() && it->second.has_value()) {
        try {
            return std::any_cast<T>(it->second);
        } catch (const std::bad_any_cast&) {
            return std::nullopt;
        }
    }
    return std::nullopt;
}

template<typename T>
T AlgorithmBase::get_parameter_or(const std::string& key, const T& default_value) const {
    auto value = get_parameter<T>(key);
    return value.value_or(default_value);
}

class AlgorithmRegistry {
public:
    static AlgorithmRegistry& instance();
    
    void register_algorithm(const std::string& name, std::function<std::unique_ptr<AlgorithmBase>()> creator);
    void unregister_algorithm(const std::string& name);
    
    std::unique_ptr<AlgorithmBase> create_algorithm(const std::string& name);
    
    std::vector<std::string> get_algorithm_names() const;
    bool has_algorithm(const std::string& name) const;
    
private:
    AlgorithmRegistry() = default;
    ~AlgorithmRegistry() = default;
    
    std::unordered_map<std::string, std::function<std::unique_ptr<AlgorithmBase>()>> algorithms_;
    mutable std::mutex mutex_;
};

} // namespace framework
} // namespace spacego
