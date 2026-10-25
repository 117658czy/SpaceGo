#pragma once

#include "go/search/hmgs_dp_optimized.h"
#include "go/search/ca_moe.h"
#include "go/heterogeneous/heterogeneous_compute_engine.h"
#ifdef USE_CUDA
// CUDA相关头文件，仅在启用CUDA时包含
#include "go/board/go_board_cuda.h"
#include "go/neural/neural_network_cuda.h"
#endif
#include <memory>
#include <vector>
#include <queue>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace spacego {
namespace go {
namespace search {

// 前向声明空间拓扑加速器
class SpatialTopologyAccelerator;

#ifdef USE_CUDA
// ========================================================================
// CUDA加速的HMGS-DP搜索算法
// 仅在启用CUDA时编译，依赖GoBoardCUDA、NeuralNetworkCUDA等CUDA类型
// ========================================================================
class HMGSDP_CUDA : public HMGSDP_Optimized {
public:
    HMGSDP_CUDA();
    virtual ~HMGSDP_CUDA();
    
    // 初始化（覆盖基类方法）
    virtual void initialize(const std::string& model_path = "") override;
    
    // 搜索接口（CUDA加速版本）
    virtual std::pair<int, int> search(const std::string& board_state, 
                                       int max_iterations = 10000,
                                       double exploration_weight = 1.0) override;
    
    // 批量搜索（CUDA特有功能）
    std::vector<std::pair<int, int>> batch_search(
        const std::vector<std::string>& board_states,
        int max_iterations_per_board = 10000,
        double exploration_weight = 1.0);
    
    // 异步搜索
    struct AsyncSearchResult {
        std::string board_state;
        std::pair<int, int> best_move;
        double win_probability;
        uint64_t search_id;
        bool completed;
    };
    
    uint64_t start_async_search(const std::string& board_state,
                               int max_iterations = 10000,
                               double exploration_weight = 1.0);
    
    bool get_async_search_result(uint64_t search_id, AsyncSearchResult& result);
    
    // 性能优化设置
    void set_batch_size(size_t batch_size);
    void enable_cuda(bool enable);
    void set_cuda_stream_count(int stream_count);
    void set_parallel_simulation_count(int count);
    
    // 性能统计
    struct PerformanceStats {
        size_t total_simulations;
        size_t cuda_accelerated_simulations;
        double average_simulation_time_ms;
        double cuda_speedup_factor;
        double gpu_utilization;
        size_t batch_processing_count;
        double batch_efficiency;  // 实际批次利用率
        size_t topology_calculations;          // 拓扑计算次数
        double topology_avg_time_us;           // 拓扑计算平均耗时(微秒)
        
        PerformanceStats() 
            : total_simulations(0), cuda_accelerated_simulations(0),
              average_simulation_time_ms(0.0), cuda_speedup_factor(1.0),
              gpu_utilization(0.0), batch_processing_count(0),
              batch_efficiency(0.0), topology_calculations(0),
              topology_avg_time_us(0.0) {}
    };
    
    PerformanceStats get_performance_stats() const;
    void reset_performance_stats();
    
    // 获取/设置拓扑加速器
    void set_topology_accelerator(std::shared_ptr<class SpatialTopologyAccelerator> accelerator);
    std::shared_ptr<class SpatialTopologyAccelerator> get_topology_accelerator() const;

    // ========================================================================
    // 异构计算集成接口 - 支持HMGS-DP搜索仿真在CPU/GPU间智能拆分
    // ========================================================================

    /**
     * @brief 启用或禁用异构计算模式
     * @details 启用后，HMGS-DP搜索中的蒙特卡洛仿真可根据搜索树深度和
     *          仿真数量在CPU和GPU间智能拆分：
     *          - 浅层节点（深度<5）：小批次仿真，优先CPU
     *          - 深层节点（深度>=5）：大批次仿真，优先GPU
     *          - 异构引擎不可用时自动降级到原有CUDA/CPU路径
     * @param enabled true=启用异构拆分, false=使用原有搜索路径
     */
    void set_heterogeneous_enabled(bool enabled);

    /**
     * @brief 查询异构计算模式是否已启用
     * @return true=异构拆分模式已启用
     */
    bool is_heterogeneous_enabled() const { return enable_heterogeneous_; }

    /**
     * @brief 根据搜索树深度和仿真数将仿真任务拆分到CPU/GPU
     * @details 分析当前搜索状态的复杂度，决定CPU/GPU仿真分配比例。
     *          拆分策略：
     *          - 总仿真数 <= 256: 全部GPU（减少启动开销，GPU延迟优势）
     *          - 总仿真数 256~1024: CPU 25%, GPU 75%
     *          - 总仿真数 > 1024: CPU 40%, GPU 60%（大规模均衡）
     *          深度<5的浅层节点仿真优先在CPU执行（减少传输开销）。
     * @param total_simulations 总仿真数量
     * @param avg_depth 平均搜索树深度
     * @param cpu_simulations 输出：分配给CPU的仿真数
     * @param gpu_simulations 输出：分配给GPU的仿真数
     * @return true=拆分成功, false=回退到原有路径
     */
    bool split_simulations_for_heterogeneous(
        size_t total_simulations,
        int avg_depth,
        size_t& cpu_simulations,
        size_t& gpu_simulations) const;

protected:
    // 覆盖仿真方法（CUDA加速）
    virtual double simulate(HMGS_Node* node, const std::string& state) override;
    
    // 覆盖节点扩展方法
    virtual HMGS_Node* expand_node(HMGS_Node* node, const std::string& state) override;
    
    // 覆盖反向传播方法
    virtual void backpropagate(HMGS_Node* node, double reward) override;
    
private:
    // CUDA组件
    std::shared_ptr<go::GoBoardCUDA> go_board_cuda_;
    std::shared_ptr<neural::NeuralNetworkCUDA> neural_network_cuda_;
    std::shared_ptr<neural::NeuralNetworkBatchManager> batch_manager_;
    
    // 配置参数
    bool cuda_enabled_;
    bool enable_heterogeneous_;
    size_t batch_size_;
    int cuda_stream_count_;
    int parallel_simulation_count_;
    
    // 性能优化
    mutable PerformanceStats performance_stats_;
    mutable std::mutex stats_mutex_;
    
    // 批量处理队列
    struct BatchSimulationRequest {
        std::string board_state;
        HMGS_Node* node;
        std::promise<double> result_promise;
        uint64_t request_id;
    };
    
    std::queue<BatchSimulationRequest> simulation_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> processing_stopped_;
    std::thread batch_processing_thread_;
    
    // 异步搜索管理
    std::unordered_map<uint64_t, AsyncSearchResult> async_search_results_;
    std::mutex async_search_mutex_;
    std::atomic<uint64_t> next_search_id_{0};
    
    // 空间拓扑加速器
    std::shared_ptr<class SpatialTopologyAccelerator> topology_accelerator_;
    
    // 私有方法
    void initialize_cuda_components();
    void batch_processing_loop();
    
    // 批量仿真处理
    void process_simulation_batch(std::vector<BatchSimulationRequest>& batch);
    std::vector<double> batch_simulate_cuda(const std::vector<std::string>& board_states);
    
    // 节点管理（CUDA优化）
    struct CUDA_NodeData {
        HMGS_Node* node;
        std::string board_state;
        float prior_probability;
        float value_estimate;
        bool visited;
    };
    
    std::vector<CUDA_NodeData> cuda_node_cache_;
    std::unordered_map<HMGS_Node*, size_t> node_to_cuda_index_;
    
    // 棋盘状态缓存（减少字符串转换开销）
    mutable std::unordered_map<std::string, go::GoBoardCPU> board_cache_;
    mutable std::mutex board_cache_mutex_;
    
    // 工具方法
    go::GoBoardCPU get_cached_board(const std::string& state_str) const;
    std::vector<go::GoBoardCPU> convert_to_boards(const std::vector<std::string>& states) const;
    
    // 性能测量
    void update_performance_stats(size_t simulations, bool used_cuda, 
                                 double simulation_time_ms, size_t batch_size);
    
    // 调试和验证
    void validate_cuda_results(const std::vector<double>& cuda_results,
                              const std::vector<double>& cpu_results) const;
};

// CUDA加速的HMGS-DP搜索工厂
class HMGSDP_CUDA_Factory {
public:
    static std::unique_ptr<HMGSDP_Optimized> create_hmgs_dp_cuda(
        const std::string& model_path = "",
        bool enable_cuda = true,
        size_t batch_size = 256,
        int cuda_stream_count = 2);
    
    static std::unique_ptr<HMGSDP_Optimized> create_hmgs_dp_hybrid(
        const std::string& model_path = "",
        size_t cpu_threads = 4,
        size_t gpu_batch_size = 128);
};

// 性能监控器（用于HMGS-DP CUDA）
class HMGSDP_CUDA_Monitor {
public:
    struct MonitorConfig {
        int sampling_interval_ms;
        bool log_to_file;
        std::string log_file_path;
        bool enable_alerts;
        double performance_alert_threshold;  // 性能下降阈值（百分比）
    };
    
    HMGSDP_CUDA_Monitor(HMGSDP_CUDA* hmgs_dp, const MonitorConfig& config);
    ~HMGSDP_CUDA_Monitor();
    
    void start_monitoring();
    void stop_monitoring();
    
    struct MonitoringData {
        std::chrono::system_clock::time_point timestamp;
        HMGSDP_CUDA::PerformanceStats stats;
        double gpu_memory_usage_mb;
        double cpu_usage_percent;
        size_t active_search_count;
        double average_batch_utilization;
        
        // 性能指标
        double simulations_per_second;
        double effective_speedup;
        double cache_hit_rate;
    };
    
    std::vector<MonitoringData> get_monitoring_history() const;
    MonitoringData get_current_status() const;
    
    // 警报系统
    struct PerformanceAlert {
        enum class AlertType {
            PERFORMANCE_DEGRADATION,
            GPU_MEMORY_HIGH,
            BATCH_UTILIZATION_LOW,
            CACHE_EFFICIENCY_LOW,
            SEARCH_STALLED
        };
        
        AlertType type;
        std::string message;
        double severity;  // 0-1
        std::chrono::system_clock::time_point timestamp;
        std::vector<std::string> suggestions;
    };
    
    std::vector<PerformanceAlert> get_active_alerts() const;
    void acknowledge_alert(size_t alert_index);
    
    // 报告生成
    std::string generate_performance_report() const;
    void export_report_to_file(const std::string& file_path) const;
    
private:
    HMGSDP_CUDA* hmgs_dp_;
    MonitorConfig config_;
    
    std::thread monitoring_thread_;
    std::atomic<bool> monitoring_active_;
    mutable std::mutex data_mutex_;
    
    std::vector<MonitoringData> monitoring_history_;
    std::vector<PerformanceAlert> active_alerts_;
    
    // 性能基准
    double baseline_performance_;
    std::chrono::system_clock::time_point monitoring_start_time_;
    
    void monitoring_loop();
    void analyze_performance_data(const MonitoringData& data);
    void check_for_alerts(const MonitoringData& data);
    void log_monitoring_data(const MonitoringData& data);
    
    // 性能分析工具
    double calculate_performance_score(const MonitoringData& data) const;
    std::vector<std::string> generate_optimization_suggestions(
        const MonitoringData& data) const;
};

// CUDA加速的并行搜索管理器
class ParallelHMGSDP_CUDA {
public:
    struct SearchTask {
        std::string board_state;
        int max_iterations;
        double exploration_weight;
        std::promise<std::pair<int, int>> result_promise;
        uint64_t task_id;
    };
    
    ParallelHMGSDP_CUDA(size_t num_searchers = 4,
                       const std::string& model_path = "",
                       bool enable_cuda = true);
    ~ParallelHMGSDP_CUDA();
    
    // 任务提交
    uint64_t submit_search_task(const std::string& board_state,
                               int max_iterations = 10000,
                               double exploration_weight = 1.0);
    
    std::vector<uint64_t> submit_batch_search_tasks(
        const std::vector<std::string>& board_states,
        int max_iterations_per_board = 10000,
        double exploration_weight = 1.0);
    
    // 结果获取
    bool try_get_result(uint64_t task_id, std::pair<int, int>& result);
    std::pair<int, int> wait_for_result(uint64_t task_id);
    std::vector<std::pair<int, int>> wait_for_all_results();
    
    // 资源管理
    void set_concurrent_searches(size_t max_concurrent);
    void pause_searches();
    void resume_searches();
    void stop_all_searches();
    
    // 性能统计
    struct ParallelPerformanceStats {
        size_t total_tasks_completed;
        size_t tasks_in_progress;
        double average_task_time_ms;
        double system_utilization;  // 整体系统利用率
        std::vector<double> searcher_utilizations;
        
        // CUDA特定统计
        double gpu_utilization;
        size_t cuda_batch_count;
        double cuda_batch_efficiency;
    };
    
    ParallelPerformanceStats get_parallel_stats() const;
    
private:
    std::vector<std::unique_ptr<HMGSDP_CUDA>> searchers_;
    std::vector<std::thread> searcher_threads_;
    
    // 任务管理
    std::queue<SearchTask> task_queue_;
    std::unordered_map<uint64_t, SearchTask> completed_tasks_;
    mutable std::mutex task_mutex_;
    std::condition_variable task_cv_;
    
    // 线程控制
    std::atomic<bool> stop_searchers_;
    std::atomic<size_t> active_searcher_count_;
    std::atomic<uint64_t> next_task_id_{0};
    
    // 性能统计
    mutable ParallelPerformanceStats parallel_stats_;
    mutable std::mutex stats_mutex_;
    
    // 私有方法
    void searcher_worker(size_t searcher_index);
    void process_search_task(SearchTask& task, HMGSDP_CUDA& searcher);
    void update_parallel_stats(size_t searcher_index, double task_time_ms);
    
    // 负载均衡
    size_t select_searcher_for_task() const;
    void rebalance_searcher_load();
};

#endif // USE_CUDA

}  // namespace search
}  // namespace go
}  // namespace spacego