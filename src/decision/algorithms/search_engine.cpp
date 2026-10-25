#include "network/algorithms/search_engine.h"

namespace spacego {
namespace algorithms {

SearchEngine::SearchEngine() {
    // 创建配置管理器
    config_manager_ = std::make_unique<ConfigManager>();

    // 获取配置
    auto config = config_manager_->get_config();

    // 创建计算层组件
    uct_calculator_ = std::make_unique<UCTCalculator>(config);
    similarity_calculator_ = std::make_unique<SimilarityCalculator>();
    diversity_calculator_ = std::make_unique<DiversityCalculator>();
    diversity_calculator_->initialize(config);

    // 创建数据层组件
    node_manager_ = std::make_unique<NodeManager>();
    node_manager_->initialize(config);
    merge_network_ = std::make_unique<MergeNetwork>();
    merge_network_->initialize(config);

    // 创建执行层组件
    simulation_executor_ = std::make_unique<SimulationExecutor>();
    search_controller_ = std::make_unique<SearchController>();
    search_controller_->initialize(diversity_calculator_.get(),
                                   node_manager_.get(),
                                   merge_network_.get(),
                                   simulation_executor_.get(),
                                   config);

    // 创建持久化层组件
    data_exporter_ = std::make_unique<DataExporter>();
    data_importer_ = std::make_unique<DataImporter>();
}

// 搜索控制委托方法
void SearchEngine::start_search(const std::string& initial_state) {
    search_controller_->start_search(initial_state);
}

void SearchEngine::stop_search() {
    search_controller_->stop_search();
}

void SearchEngine::pause_search() {
    search_controller_->pause_search();
}

void SearchEngine::resume_search() {
    search_controller_->resume_search();
}

// 搜索执行委托方法
void SearchEngine::run_simulation() {
    search_controller_->run_simulation();
}

void SearchEngine::run_simulations(size_t count) {
    search_controller_->run_simulations(count);
}

std::string SearchEngine::get_best_move() const {
    return search_controller_->get_best_move();
}

std::vector<std::pair<std::string, double>> SearchEngine::get_move_probabilities() const {
    return search_controller_->get_move_probabilities();
}

// 树管理委托方法
void SearchEngine::update_root(const std::string& move) {
    search_controller_->update_root(move);
}

void SearchEngine::reset_tree() {
    search_controller_->reset_tree();
}

void SearchEngine::clear_tree() {
    search_controller_->clear_tree();
}

// 状态查询委托方法
bool SearchEngine::is_searching() const {
    return search_controller_->is_searching();
}

bool SearchEngine::is_initialized() const {
    return search_controller_->is_initialized();
}

HMGSStatistics SearchEngine::get_statistics() const {
    return search_controller_->get_statistics();
}

std::vector<std::string> SearchEngine::get_search_debug_info() const {
    return search_controller_->get_search_debug_info();
}

// 合并网络操作委托方法
void SearchEngine::force_merge_check() {
    merge_network_->force_merge_check(
        search_controller_->get_root_node(),
        search_controller_->get_node_registry(),
        search_controller_->get_config(),
        search_controller_->get_stats(),
        search_controller_->get_merge_probabilities(),
        search_controller_->get_unmerged_counts(),
        search_controller_->get_last_visit_times());
}

void SearchEngine::recalculate_merge_probabilities() {
    merge_network_->recalculate_merge_probabilities(
        search_controller_->get_node_registry(),
        search_controller_->get_config(),
        search_controller_->get_stats(),
        search_controller_->get_merge_probabilities(),
        search_controller_->get_merged_into_probabilities(),
        search_controller_->get_merge_relationships());
}

void SearchEngine::rebuild_merge_network() {
    merge_network_->rebuild_merge_network(
        search_controller_->get_node_registry(),
        search_controller_->get_config(),
        search_controller_->get_stats(),
        search_controller_->get_merge_probabilities(),
        search_controller_->get_merged_into_probabilities(),
        search_controller_->get_merge_relationships());
}

// 多样性保护操作委托方法
void SearchEngine::identify_neglected_branches() {
    search_controller_->identify_neglected_branches();
}

void SearchEngine::allocate_resources_to_neglected_branches(double total_resources) {
    search_controller_->allocate_resources_to_neglected_branches(total_resources);
}

void SearchEngine::adjust_diversity_parameters() {
    search_controller_->adjust_diversity_parameters();
}

// 回溯与再发现委托方法
void SearchEngine::perform_backtrack() {
    search_controller_->perform_backtrack();
}

void SearchEngine::rediscover_neglected_branches() {
    search_controller_->rediscover_neglected_branches();
}

// 数据导出委托方法
std::string SearchEngine::export_search_tree() const {
    return data_exporter_->export_search_tree(
        search_controller_->get_root_node(),
        search_controller_->get_node_registry(),
        search_controller_->get_main_branches(),
        search_controller_->get_stats());
}

std::string SearchEngine::export_merge_network() const {
    return data_exporter_->export_merge_network(
        search_controller_->get_node_registry(),
        search_controller_->get_merge_probabilities(),
        search_controller_->get_merged_into_probabilities(),
        search_controller_->get_merge_relationships(),
        search_controller_->get_stats());
}

std::string SearchEngine::export_diversity_map() const {
    return data_exporter_->export_diversity_map(
        search_controller_->get_node_registry(),
        search_controller_->get_main_branches(),
        search_controller_->get_root_node(),
        search_controller_->get_neglected_branches(),
        search_controller_->get_unmerged_counts(),
        search_controller_->get_last_visit_times(),
        search_controller_->get_stats());
}

// 数据导入委托方法
void SearchEngine::import_search_tree(const std::string& tree_data) {
    std::shared_ptr<HMGSNode> root_node = search_controller_->get_root_node();
    data_importer_->import_search_tree(
        tree_data,
        root_node,
        search_controller_->get_node_registry(),
        search_controller_->get_main_branches(),
        search_controller_->get_stats());
    search_controller_->set_root_node(std::dynamic_pointer_cast<HMGSRootNode>(root_node));
}

void SearchEngine::import_merge_network(const std::string& network_data) {
    data_importer_->import_merge_network(
        network_data,
        search_controller_->get_merge_probabilities(),
        search_controller_->get_merged_into_probabilities(),
        search_controller_->get_merge_relationships(),
        search_controller_->get_stats());
}

void SearchEngine::import_diversity_map(const std::string& diversity_data) {
    data_importer_->import_diversity_map(
        diversity_data,
        search_controller_->get_neglected_branches(),
        search_controller_->get_unmerged_counts(),
        search_controller_->get_last_visit_times(),
        search_controller_->get_stats());
}

// 兼容旧版本导入函数
void SearchEngine::import_search_tree_v1(const std::string& tree_data) {
    std::shared_ptr<HMGSNode> root_node = search_controller_->get_root_node();
    data_importer_->import_search_tree_v1(
        tree_data,
        root_node,
        search_controller_->get_node_registry(),
        search_controller_->get_stats());
    search_controller_->set_root_node(std::dynamic_pointer_cast<HMGSRootNode>(root_node));
}

void SearchEngine::import_merge_network_v1(const std::string& network_data) {
    data_importer_->import_merge_network_v1(
        network_data,
        search_controller_->get_merge_probabilities(),
        search_controller_->get_stats());
}

void SearchEngine::import_diversity_map_v1(const std::string& diversity_data) {
    data_importer_->import_diversity_map_v1(
        diversity_data,
        search_controller_->get_neglected_branches(),
        search_controller_->get_stats());
}

} // namespace algorithms
} // namespace spacego
