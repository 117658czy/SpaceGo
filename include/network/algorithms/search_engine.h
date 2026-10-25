#pragma once

#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "network/algorithms/config/config_manager.h"
#include "network/algorithms/calculation/uct_calculator.h"
#include "network/algorithms/calculation/similarity_calculator.h"
#include "network/algorithms/calculation/diversity_calculator.h"
#include "network/algorithms/data/node_manager.h"
#include "network/algorithms/data/merge_network.h"
#include "network/algorithms/execution/simulation_executor.h"
#include "network/algorithms/control/search_controller.h"
#include "network/algorithms/persistence/data_exporter.h"
#include "network/algorithms/persistence/data_importer.h"

namespace spacego {
namespace algorithms {

class SearchEngine {
public:
    SearchEngine();
    ~SearchEngine() = default;
    
    // 配置管理
    ConfigManager* get_config_manager() { return config_manager_.get(); }
    const ConfigManager* get_config_manager() const { return config_manager_.get(); }
    
    // 计算组件
    UCTCalculator* get_uct_calculator() { return uct_calculator_.get(); }
    SimilarityCalculator* get_similarity_calculator() { return similarity_calculator_.get(); }
    DiversityCalculator* get_diversity_calculator() { return diversity_calculator_.get(); }
    
    // 数据管理
    NodeManager* get_node_manager() { return node_manager_.get(); }
    MergeNetwork* get_merge_network() { return merge_network_.get(); }
    
    // 执行组件
    SimulationExecutor* get_simulation_executor() { return simulation_executor_.get(); }
    SearchController* get_search_controller() { return search_controller_.get(); }
    
    // 持久化组件
    DataExporter* get_data_exporter() { return data_exporter_.get(); }
    DataImporter* get_data_importer() { return data_importer_.get(); }
    
    // 搜索控制（委托给SearchController）
    void start_search(const std::string& initial_state);
    void stop_search();
    void pause_search();
    void resume_search();
    
    // 搜索执行（委托给SearchController）
    void run_simulation();
    void run_simulations(size_t count);
    std::string get_best_move() const;
    std::vector<std::pair<std::string, double>> get_move_probabilities() const;
    
    // 树管理（委托给SearchController）
    void update_root(const std::string& move);
    void reset_tree();
    void clear_tree();
    
    // 状态查询（委托给SearchController）
    bool is_searching() const;
    bool is_initialized() const;
    HMGSStatistics get_statistics() const;
    std::vector<std::string> get_search_debug_info() const;
    
    // 合并网络操作（委托给MergeNetwork）
    void force_merge_check();
    void recalculate_merge_probabilities();
    void rebuild_merge_network();
    
    // 多样性保护操作（委托给DiversityCalculator和SearchController）
    void identify_neglected_branches();
    void allocate_resources_to_neglected_branches(double total_resources);
    void adjust_diversity_parameters();
    
    // 回溯与再发现（委托给SearchController）
    void perform_backtrack();
    void rediscover_neglected_branches();
    
    // 数据导出（委托给DataExporter）
    std::string export_search_tree() const;
    std::string export_merge_network() const;
    std::string export_diversity_map() const;
    
    // 数据导入（委托给DataImporter）
    void import_search_tree(const std::string& tree_data);
    void import_merge_network(const std::string& network_data);
    void import_diversity_map(const std::string& diversity_data);
    
    // 兼容旧版本导入函数
    void import_search_tree_v1(const std::string& tree_data);
    void import_merge_network_v1(const std::string& network_data);
    void import_diversity_map_v1(const std::string& diversity_data);

private:
    // 配置层
    std::unique_ptr<ConfigManager> config_manager_;
    
    // 计算层
    std::unique_ptr<UCTCalculator> uct_calculator_;
    std::unique_ptr<SimilarityCalculator> similarity_calculator_;
    std::unique_ptr<DiversityCalculator> diversity_calculator_;
    
    // 数据层
    std::unique_ptr<NodeManager> node_manager_;
    std::unique_ptr<MergeNetwork> merge_network_;
    
    // 执行层
    std::unique_ptr<SimulationExecutor> simulation_executor_;
    std::unique_ptr<SearchController> search_controller_;
    
    // 持久化层
    std::unique_ptr<DataExporter> data_exporter_;
    std::unique_ptr<DataImporter> data_importer_;
};

} // namespace algorithms
} // namespace spacego
