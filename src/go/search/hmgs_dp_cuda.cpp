#include "go/search/hmgs_dp_cuda.h"
#include "framework/logger.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <chrono>

namespace spacego {
namespace go {
namespace search {

#ifdef USE_CUDA
// ============================================================================
// HMGSDP_CUDA 实现 - CUDA加速的HMGS-DP搜索算法
// 继承自HMGSDP_Optimized，增加CUDA批量推理和异构计算支持
// ============================================================================

HMGSDP_CUDA::HMGSDP_CUDA()
    : cuda_enabled_(false)
    , enable_heterogeneous_(false)
    , batch_size_(256)
    , cuda_stream_count_(2)
    , parallel_simulation_count_(4)
    , processing_stopped_(false)
    , next_search_id_(1) {
}

HMGSDP_CUDA::~HMGSDP_CUDA() {
    processing_stopped_ = true;
    queue_cv_.notify_all();
    if (batch_processing_thread_.joinable()) {
        batch_processing_thread_.join();
    }
}

void HMGSDP_CUDA::initialize(const std::string& model_path) {
    HMGSDP_Optimized::initialize(model_path);

    initialize_cuda_components();

    if (!processing_stopped_) {
        batch_processing_thread_ = std::thread(&HMGSDP_CUDA::batch_processing_loop, this);
    }

    LOG_INFO("HMGSDP_CUDA初始化完成 (CUDA=%s, 异构=%s)",
             cuda_enabled_ ? "已启用" : "未启用",
             enable_heterogeneous_ ? "已启用" : "未启用");
}

void HMGSDP_CUDA::initialize_cuda_components() {
    if (!torch::cuda::is_available()) {
        LOG_WARNING("CUDA不可用，HMGSDP_CUDA将使用CPU模式");
        cuda_enabled_ = false;
        return;
    }

    try {
        neural_network_cuda_ = std::make_shared<neural::NeuralNetworkCUDA>();
        if (!neural_network_cuda_) {
            LOG_WARNING("CUDA神经网络创建失败");
            cuda_enabled_ = false;
            return;
        }

        batch_manager_ = std::make_shared<neural::NeuralNetworkBatchManager>(
            neural_network_cuda_);

        cuda_enabled_ = true;
        LOG_INFO("CUDA组件初始化成功 (batch_size=%zu, streams=%d)",
                 batch_size_, cuda_stream_count_);
    } catch (const std::exception& e) {
        LOG_ERROR("CUDA组件初始化异常: %s", e.what());
        cuda_enabled_ = false;
    }
}

std::pair<int, int> HMGSDP_CUDA::search(const std::string& board_state,
                                          int max_iterations,
                                          double exploration_weight) {
    return HMGSDP_Optimized::search(board_state, max_iterations, exploration_weight);
}

std::vector<std::pair<int, int>> HMGSDP_CUDA::batch_search(
    const std::vector<std::string>& board_states,
    int max_iterations_per_board,
    double exploration_weight) {
    std::vector<std::pair<int, int>> results;
    results.reserve(board_states.size());
    for (const auto& state : board_states) {
        results.push_back(search(state, max_iterations_per_board, exploration_weight));
    }
    return results;
}

uint64_t HMGSDP_CUDA::start_async_search(const std::string& board_state,
                                           int max_iterations,
                                           double exploration_weight) {
    uint64_t id = next_search_id_.fetch_add(1);
    std::lock_guard<std::mutex> lock(async_search_mutex_);
    async_search_results_[id] = AsyncSearchResult{
        board_state, {-1, -1}, 0.0, id, false};
    return id;
}

bool HMGSDP_CUDA::get_async_search_result(uint64_t search_id, AsyncSearchResult& result) {
    std::lock_guard<std::mutex> lock(async_search_mutex_);
    auto it = async_search_results_.find(search_id);
    if (it == async_search_results_.end()) return false;
    result = it->second;
    return result.completed;
}

void HMGSDP_CUDA::set_batch_size(size_t batch_size) {
    batch_size_ = batch_size;
    if (batch_manager_) {
        batch_manager_ = std::make_shared<neural::NeuralNetworkBatchManager>(
            neural_network_cuda_);
    }
}

void HMGSDP_CUDA::enable_cuda(bool enable) {
    cuda_enabled_ = enable && torch::cuda::is_available();
}

void HMGSDP_CUDA::set_cuda_stream_count(int stream_count) {
    cuda_stream_count_ = std::max(1, stream_count);
}

void HMGSDP_CUDA::set_parallel_simulation_count(int count) {
    parallel_simulation_count_ = std::max(1, count);
}

HMGSDP_CUDA::PerformanceStats HMGSDP_CUDA::get_performance_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return performance_stats_;
}

void HMGSDP_CUDA::reset_performance_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    performance_stats_ = PerformanceStats();
}

void HMGSDP_CUDA::set_heterogeneous_enabled(bool enabled) {
    enable_heterogeneous_ = enabled;
}

bool HMGSDP_CUDA::split_simulations_for_heterogeneous(
    size_t total_simulations, int avg_depth,
    size_t& cpu_simulations, size_t& gpu_simulations) const {
    if (!enable_heterogeneous_) {
        cpu_simulations = 0;
        gpu_simulations = total_simulations;
        return false;
    }

    if (total_simulations <= 256) {
        cpu_simulations = 0;
        gpu_simulations = total_simulations;
    } else if (total_simulations <= 1024) {
        cpu_simulations = total_simulations / 4;
        gpu_simulations = total_simulations - cpu_simulations;
    } else {
        cpu_simulations = total_simulations * 2 / 5;
        gpu_simulations = total_simulations - cpu_simulations;
    }

    if (avg_depth < 5) {
        size_t swap = std::min(cpu_simulations, gpu_simulations);
        if (avg_depth < 3) {
            cpu_simulations += swap;
            gpu_simulations -= swap;
        }
    }

    return true;
}

double HMGSDP_CUDA::simulate(HMGS_Node* node, const std::string& state) {
    return HMGSDP_Optimized::simulate(node, state);
}

HMGS_Node* HMGSDP_CUDA::expand_node(HMGS_Node* node, const std::string& state) {
    return HMGSDP_Optimized::expand_node(node, state);
}

void HMGSDP_CUDA::backpropagate(HMGS_Node* node, double reward) {
    HMGSDP_Optimized::backpropagate(node, reward);
}

void HMGSDP_CUDA::batch_processing_loop() {
    while (!processing_stopped_) {
        std::vector<BatchSimulationRequest> batch;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait_for(lock, std::chrono::milliseconds(1), [this] {
                return !simulation_queue_.empty() || processing_stopped_;
            });
            if (processing_stopped_ && simulation_queue_.empty()) break;

            while (!simulation_queue_.empty() && batch.size() < batch_size_) {
                batch.push_back(std::move(simulation_queue_.front()));
                simulation_queue_.pop();
            }
        }

        if (!batch.empty()) {
            process_simulation_batch(batch);
        }
    }
}

void HMGSDP_CUDA::process_simulation_batch(
    std::vector<BatchSimulationRequest>& batch) {
    std::vector<std::string> states;
    states.reserve(batch.size());
    for (const auto& req : batch) {
        states.push_back(req.board_state);
    }

    auto results = batch_simulate_cuda(states);
    for (size_t i = 0; i < batch.size() && i < results.size(); ++i) {
        batch[i].result_promise.set_value(results[i]);
    }
}

std::vector<double> HMGSDP_CUDA::batch_simulate_cuda(
    const std::vector<std::string>& board_states) {
    std::vector<double> results(board_states.size(), 0.0);

    if (!cuda_enabled_ || !neural_network_cuda_) {
        for (size_t i = 0; i < board_states.size(); ++i) {
            results[i] = HMGSDP_Optimized::simulate(nullptr, board_states[i]);
        }
        return results;
    }

    try {
        if (neural_network_cuda_) {
            auto nn_results = neural_network_cuda_->batch_inference(board_states);
            for (size_t i = 0; i < nn_results.size(); ++i) {
                results[i] = nn_results[i].value_estimate;
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR("CUDA批量仿真异常: %s", e.what());
        for (size_t i = 0; i < board_states.size(); ++i) {
            results[i] = HMGSDP_Optimized::simulate(nullptr, board_states[i]);
        }
    }

    return results;
}

GoBoardCPU HMGSDP_CUDA::get_cached_board(const std::string& state_str) const {
    std::lock_guard<std::mutex> lock(board_cache_mutex_);
    auto it = board_cache_.find(state_str);
    if (it != board_cache_.end()) {
        return it->second;
    }
    GoBoardCPU board(state_str);
    board_cache_[state_str] = board;
    return board;
}

std::vector<GoBoardCPU> HMGSDP_CUDA::convert_to_boards(
    const std::vector<std::string>& states) const {
    std::vector<GoBoardCPU> boards;
    boards.reserve(states.size());
    for (const auto& s : states) {
        boards.push_back(get_cached_board(s));
    }
    return boards;
}

void HMGSDP_CUDA::update_performance_stats(size_t simulations, bool used_cuda,
                                              double simulation_time_ms, size_t batch_size) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    performance_stats_.total_simulations += simulations;
    if (used_cuda) {
        performance_stats_.cuda_accelerated_simulations += simulations;
    }
    performance_stats_.average_simulation_time_ms =
        (performance_stats_.average_simulation_time_ms * (performance_stats_.total_simulations - simulations) +
         simulation_time_ms * simulations) / performance_stats_.total_simulations;
    if (batch_size > 0) {
        performance_stats_.batch_processing_count++;
    }
}

void HMGSDP_CUDA::validate_cuda_results(const std::vector<double>& cuda_results,
                                           const std::vector<double>& cpu_results) const {
}

// ============================================================================
// HMGSDP_CUDA_Factory 实现
// ============================================================================

std::unique_ptr<HMGSDP_Optimized> HMGSDP_CUDA_Factory::create_hmgs_dp_cuda(
    const std::string& model_path, bool enable_cuda, size_t batch_size, int cuda_stream_count) {
    auto hmgs = std::make_unique<HMGSDP_CUDA>();
    hmgs->enable_cuda(enable_cuda);
    hmgs->set_batch_size(batch_size);
    hmgs->set_cuda_stream_count(cuda_stream_count);
    hmgs->initialize(model_path);
    return hmgs;
}

std::unique_ptr<HMGSDP_Optimized> HMGSDP_CUDA_Factory::create_hmgs_dp_hybrid(
    const std::string& model_path, size_t cpu_threads, size_t gpu_batch_size) {
    auto hmgs = std::make_unique<HMGSDP_CUDA>();
    hmgs->set_batch_size(gpu_batch_size);
    hmgs->initialize(model_path);
    return hmgs;
}

// ============================================================================
// HMGSDP_CUDA_Monitor 实现
// ============================================================================

HMGSDP_CUDA_Monitor::HMGSDP_CUDA_Monitor(HMGSDP_CUDA* hmgs_dp, const MonitorConfig& config)
    : hmgs_dp_(hmgs_dp)
    , config_(config)
    , monitoring_active_(false)
    , baseline_performance_(0.0) {
}

HMGSDP_CUDA_Monitor::~HMGSDP_CUDA_Monitor() {
    stop_monitoring();
}

void HMGSDP_CUDA_Monitor::start_monitoring() {
    if (monitoring_active_) return;
    monitoring_active_ = true;
    monitoring_start_time_ = std::chrono::system_clock::now();
    monitoring_thread_ = std::thread(&HMGSDP_CUDA_Monitor::monitoring_loop, this);
}

void HMGSDP_CUDA_Monitor::stop_monitoring() {
    monitoring_active_ = false;
    if (monitoring_thread_.joinable()) {
        monitoring_thread_.join();
    }
}

std::vector<HMGSDP_CUDA_Monitor::MonitoringData> HMGSDP_CUDA_Monitor::get_monitoring_history() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return monitoring_history_;
}

HMGSDP_CUDA_Monitor::MonitoringData HMGSDP_CUDA_Monitor::get_current_status() const {
    MonitoringData data;
    data.timestamp = std::chrono::system_clock::now();
    if (hmgs_dp_) {
        data.stats = hmgs_dp_->get_performance_stats();
    }
    data.gpu_memory_usage_mb = 0.0;
    data.cpu_usage_percent = 0.0;
    data.active_search_count = 0;
    data.average_batch_utilization = 0.0;
    data.simulations_per_second = 0.0;
    data.effective_speedup = 1.0;
    data.cache_hit_rate = 0.0;
    return data;
}

std::vector<HMGSDP_CUDA_Monitor::PerformanceAlert> HMGSDP_CUDA_Monitor::get_active_alerts() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return active_alerts_;
}

void HMGSDP_CUDA_Monitor::acknowledge_alert(size_t alert_index) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (alert_index < active_alerts_.size()) {
        active_alerts_.erase(active_alerts_.begin() + alert_index);
    }
}

std::string HMGSDP_CUDA_Monitor::generate_performance_report() const {
    return "HMGS-DP CUDA 性能报告: 暂无数据";
}

void HMGSDP_CUDA_Monitor::export_report_to_file(const std::string& file_path) const {
}

void HMGSDP_CUDA_Monitor::monitoring_loop() {
    while (monitoring_active_) {
        auto data = get_current_status();
        analyze_performance_data(data);
        check_for_alerts(data);
        log_monitoring_data(data);

        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            monitoring_history_.push_back(data);
        }

        std::this_thread::sleep_for(
            std::chrono::milliseconds(config_.sampling_interval_ms));
    }
}

void HMGSDP_CUDA_Monitor::analyze_performance_data(const MonitoringData& data) {
}

void HMGSDP_CUDA_Monitor::check_for_alerts(const MonitoringData& data) {
}

void HMGSDP_CUDA_Monitor::log_monitoring_data(const MonitoringData& data) {
}

double HMGSDP_CUDA_Monitor::calculate_performance_score(const MonitoringData& data) const {
    return 0.0;
}

std::vector<std::string> HMGSDP_CUDA_Monitor::generate_optimization_suggestions(
    const MonitoringData& data) const {
    return {};
}

// ============================================================================
// ParallelHMGSDP_CUDA 实现
// ============================================================================

ParallelHMGSDP_CUDA::ParallelHMGSDP_CUDA(size_t num_searchers,
                                             const std::string& model_path,
                                             bool enable_cuda)
    : stop_searchers_(false)
    , active_searcher_count_(0)
    , next_task_id_(1) {
    for (size_t i = 0; i < num_searchers; ++i) {
        searchers_.push_back(std::make_unique<HMGSDP_CUDA>());
        searchers_.back()->enable_cuda(enable_cuda);
        searchers_.back()->initialize(model_path);
    }
}

ParallelHMGSDP_CUDA::~ParallelHMGSDP_CUDA() {
    stop_all_searches();
}

uint64_t ParallelHMGSDP_CUDA::submit_search_task(const std::string& board_state,
                                                    int max_iterations,
                                                    double exploration_weight) {
    uint64_t id = next_task_id_.fetch_add(1);
    SearchTask task;
    task.board_state = board_state;
    task.max_iterations = max_iterations;
    task.exploration_weight = exploration_weight;
    task.task_id = id;

    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        task_queue_.push(std::move(task));
    }
    task_cv_.notify_one();
    return id;
}

std::vector<uint64_t> ParallelHMGSDP_CUDA::submit_batch_search_tasks(
    const std::vector<std::string>& board_states,
    int max_iterations_per_board,
    double exploration_weight) {
    std::vector<uint64_t> ids;
    ids.reserve(board_states.size());
    for (const auto& state : board_states) {
        ids.push_back(submit_search_task(state, max_iterations_per_board, exploration_weight));
    }
    return ids;
}

bool ParallelHMGSDP_CUDA::try_get_result(uint64_t task_id, std::pair<int, int>& result) {
    std::lock_guard<std::mutex> lock(task_mutex_);
    auto it = completed_tasks_.find(task_id);
    if (it == completed_tasks_.end()) return false;
    try {
        result = it->second.result_promise.get_future().get();
        completed_tasks_.erase(it);
        return true;
    } catch (...) {
        return false;
    }
}

std::pair<int, int> ParallelHMGSDP_CUDA::wait_for_result(uint64_t task_id) {
    while (true) {
        {
            std::lock_guard<std::mutex> lock(task_mutex_);
            auto it = completed_tasks_.find(task_id);
            if (it != completed_tasks_.end()) {
                try {
                    auto result = it->second.result_promise.get_future().get();
                    completed_tasks_.erase(it);
                    return result;
                } catch (...) {
                    return {-1, -1};
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

std::vector<std::pair<int, int>> ParallelHMGSDP_CUDA::wait_for_all_results() {
    std::vector<std::pair<int, int>> results;
    std::lock_guard<std::mutex> lock(task_mutex_);
    for (auto& [id, task] : completed_tasks_) {
        try {
            results.push_back(task.result_promise.get_future().get());
        } catch (...) {
            results.push_back({-1, -1});
        }
    }
    completed_tasks_.clear();
    return results;
}

void ParallelHMGSDP_CUDA::set_concurrent_searches(size_t max_concurrent) {
    if (searcher_threads_.size() < max_concurrent) {
        for (size_t i = searcher_threads_.size(); i < max_concurrent && i < searchers_.size(); ++i) {
            active_searcher_count_++;
            searcher_threads_.emplace_back(&ParallelHMGSDP_CUDA::searcher_worker, this, i);
        }
    }
}

void ParallelHMGSDP_CUDA::pause_searches() {
}

void ParallelHMGSDP_CUDA::resume_searches() {
}

void ParallelHMGSDP_CUDA::stop_all_searches() {
    stop_searchers_ = true;
    task_cv_.notify_all();
    for (auto& t : searcher_threads_) {
        if (t.joinable()) t.join();
    }
    searcher_threads_.clear();
}

ParallelHMGSDP_CUDA::ParallelPerformanceStats ParallelHMGSDP_CUDA::get_parallel_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return parallel_stats_;
}

void ParallelHMGSDP_CUDA::searcher_worker(size_t searcher_index) {
    while (!stop_searchers_) {
        SearchTask task;
        {
            std::unique_lock<std::mutex> lock(task_mutex_);
            task_cv_.wait_for(lock, std::chrono::milliseconds(100), [this] {
                return !task_queue_.empty() || stop_searchers_;
            });
            if (stop_searchers_) break;
            if (task_queue_.empty()) continue;
            task = std::move(task_queue_.front());
            task_queue_.pop();
        }

        if (searcher_index < searchers_.size()) {
            process_search_task(task, *searchers_[searcher_index]);
        }
    }
}

void ParallelHMGSDP_CUDA::process_search_task(SearchTask& task, HMGSDP_CUDA& searcher) {
    auto start = std::chrono::high_resolution_clock::now();
    auto result = searcher.search(task.board_state, task.max_iterations, task.exploration_weight);
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    task.result_promise.set_value(result);
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        completed_tasks_[task.task_id] = std::move(task);
    }

    update_parallel_stats(0, time_ms);
}

void ParallelHMGSDP_CUDA::update_parallel_stats(size_t searcher_index, double task_time_ms) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    parallel_stats_.total_tasks_completed++;
    parallel_stats_.average_task_time_ms =
        (parallel_stats_.average_task_time_ms * (parallel_stats_.total_tasks_completed - 1) +
         task_time_ms) / parallel_stats_.total_tasks_completed;
}

size_t ParallelHMGSDP_CUDA::select_searcher_for_task() const {
    return 0;
}

void ParallelHMGSDP_CUDA::rebalance_searcher_load() {
}

#endif // USE_CUDA

}  // namespace search
}  // namespace go
}  // namespace spacego
