#pragma once

#include <string>
#include <memory>
#include <vector>
#include <utility>
#include <functional>
#include "framework/algorithm_base.h"
#include "network/algorithms/config/hmgs_config.h"
#include "network/algorithms/search_engine.h"

namespace spacego {
namespace algorithms {

// 前向声明
class HMGSNode;
class HMGSRootNode;

// 神经网络回调类型：接收状态字符串，返回策略概率和价值
using NeuralNetworkCallback = std::function<std::pair<std::vector<double>, double>(const std::string&)>;

// 注意: HMGSStatistics结构体定义已移至 config/hmgs_config.h
// 以避免搜索控制器(SearchController)与HMGSDP之间的循环依赖

/**
 * @brief HMGS-DP (Hierarchical Merged Graph Search with Diversity Protection) - 分层合并图搜索与多样性保护
 * @details 世界上最强的围棋AI主搜索算法，结合分层搜索、节点合并、多样性保护和回溯机制
 * 
 * 重构说明：
 * 本类已重构为门面模式（Facade Pattern），所有核心功能已委托给SearchEngine及其子组件。
 * HMGSDP类现在只负责：
 * 1. 提供与原有代码兼容的公共接口
 * 2. 管理AlgorithmBase生命周期（on_initialize, on_shutdown, on_execute, on_stop）
 * 3. 委托调用给内部的SearchEngine实例
 */
class HMGSDP : public spacego::framework::AlgorithmBase {
public:
    // 构造函数和析构函数
    explicit HMGSDP(const std::string& algorithm_name = "HMGS-DP");
    ~HMGSDP();

    // 配置管理
    void set_config(const HMGSConfig& config);
    HMGSConfig get_config() const;
    void update_parameter(const std::string& param_name, double value);
    void update_parameter(const std::string& param_name, size_t value);
    void update_parameter(const std::string& param_name, const std::string& value);
    void update_parameter(const std::string& param_name, bool value);

    // 搜索控制
    void start_search(const std::string& initial_state);
    void stop_search();
    void pause_search();
    void resume_search();
    
    // 搜索执行
    void run_simulation();
    void run_simulations(size_t count);
    std::string get_best_move() const;
    std::vector<std::pair<std::string, double>> get_move_probabilities() const;
    
    // 树管理
    void update_root(const std::string& move);
    void reset_tree();
    void clear_tree();
    
    // 状态查询
    bool is_searching() const;
    bool is_initialized() const;
    HMGSStatistics get_statistics() const;
    std::vector<std::string> get_search_debug_info() const;
    
    // 合并网络操作
    void force_merge_check();
    void recalculate_merge_probabilities();
    void rebuild_merge_network();
    
    // 多样性保护操作
    void identify_neglected_branches();
    void allocate_resources_to_neglected_branches(double total_resources);
    void adjust_diversity_parameters();
    
    // 回溯与再发现
    void perform_backtrack();
    void rediscover_neglected_branches();
    
    // 数据导出
    std::string export_search_tree() const;
    std::string export_merge_network() const;
    std::string export_diversity_map() const;
    void import_search_tree(const std::string& tree_data);
    void import_merge_network(const std::string& network_data);
    void import_diversity_map(const std::string& diversity_data);

    // 兼容旧版本导入函数
    void import_search_tree_v1(const std::string& tree_data);
    void import_merge_network_v1(const std::string& network_data);
    void import_diversity_map_v1(const std::string& diversity_data);

protected:
    // 重写AlgorithmBase的虚函数 - Override AlgorithmBase virtual functions
    bool on_initialize() override;
    bool on_shutdown() override;
    bool on_execute() override;
    void on_stop() override;

private:
    // 搜索引擎实例（包含所有专职组件）
    std::unique_ptr<SearchEngine> search_engine_;
};

} // namespace algorithms
} // namespace spacego
