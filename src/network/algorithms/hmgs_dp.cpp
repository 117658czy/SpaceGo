/**
 * @file hmgs_dp.cpp
 * @brief HMGS-DP门面模式实现 - 所有功能委托给SearchEngine及其子组件
 * @details 重构后的HMGSDP类采用门面模式(Facade Pattern)，自身不再包含任何搜索逻辑，
 *          仅作为对外接口层，将所有调用委托给内部的SearchEngine实例。
 *          SearchEngine内部通过专职组件(ConfigManager, SearchController, MergeNetwork等)
 *          完成实际的搜索、计算和管理工作。
 */

#include "network/algorithms/hmgs_dp.h"
#include "network/algorithms/search_engine.h"
#include "framework/logger.h"

namespace spacego {
namespace algorithms {

// ============================================================================
// 构造与析构
// ============================================================================

HMGSDP::HMGSDP(const std::string& algorithm_name)
    : framework::AlgorithmBase(algorithm_name) {
    // 创建搜索引擎实例，所有专职组件在SearchEngine构造时自动初始化
    search_engine_ = std::make_unique<SearchEngine>();
}

HMGSDP::~HMGSDP() {
    // 基类析构函数会自动调用shutdown()，进而调用on_shutdown()
    // 此处无需额外操作
}

// ============================================================================
// AlgorithmBase生命周期回调
// ============================================================================

bool HMGSDP::on_initialize() {
    // SearchEngine在构造时已完成初始化，此处直接返回成功
    return true;
}

bool HMGSDP::on_shutdown() {
    // 停止搜索并释放搜索引擎资源
    if (search_engine_) {
        search_engine_->stop_search();
        search_engine_.reset();
    }
    return true;
}

bool HMGSDP::on_execute() {
    // 执行逻辑由外部通过start_search/run_simulation等方法驱动
    return true;
}

void HMGSDP::on_stop() {
    // 委托给搜索引擎停止搜索
    if (search_engine_) {
        search_engine_->stop_search();
    }
}

// ============================================================================
// 配置管理 - 委托给SearchEngine的ConfigManager
// ============================================================================

void HMGSDP::set_config(const HMGSConfig& config) {
    search_engine_->get_config_manager()->set_config(config);
}

HMGSConfig HMGSDP::get_config() const {
    return search_engine_->get_config_manager()->get_config();
}

void HMGSDP::update_parameter(const std::string& param_name, double value) {
    search_engine_->get_config_manager()->update_parameter(param_name, value);
}

void HMGSDP::update_parameter(const std::string& param_name, size_t value) {
    search_engine_->get_config_manager()->update_parameter(param_name, value);
}

void HMGSDP::update_parameter(const std::string& param_name, const std::string& value) {
    search_engine_->get_config_manager()->update_parameter(param_name, value);
}

void HMGSDP::update_parameter(const std::string& param_name, bool value) {
    search_engine_->get_config_manager()->update_parameter(param_name, value);
}

// ============================================================================
// 搜索控制 - 委托给SearchEngine（内部转发给SearchController）
// ============================================================================

void HMGSDP::start_search(const std::string& initial_state) {
    search_engine_->start_search(initial_state);
}

void HMGSDP::stop_search() {
    search_engine_->stop_search();
}

void HMGSDP::pause_search() {
    search_engine_->pause_search();
}

void HMGSDP::resume_search() {
    search_engine_->resume_search();
}

// ============================================================================
// 搜索执行 - 委托给SearchEngine（内部转发给SearchController）
// ============================================================================

void HMGSDP::run_simulation() {
    search_engine_->run_simulation();
}

void HMGSDP::run_simulations(size_t count) {
    search_engine_->run_simulations(count);
}

std::string HMGSDP::get_best_move() const {
    return search_engine_->get_best_move();
}

std::vector<std::pair<std::string, double>> HMGSDP::get_move_probabilities() const {
    return search_engine_->get_move_probabilities();
}

// ============================================================================
// 树管理 - 委托给SearchEngine（内部转发给SearchController）
// ============================================================================

void HMGSDP::update_root(const std::string& move) {
    search_engine_->update_root(move);
}

void HMGSDP::reset_tree() {
    search_engine_->reset_tree();
}

void HMGSDP::clear_tree() {
    search_engine_->clear_tree();
}

// ============================================================================
// 状态查询 - 委托给SearchEngine（内部转发给SearchController）
// ============================================================================

bool HMGSDP::is_searching() const {
    return search_engine_->is_searching();
}

bool HMGSDP::is_initialized() const {
    return search_engine_->is_initialized();
}

HMGSStatistics HMGSDP::get_statistics() const {
    return search_engine_->get_statistics();
}

std::vector<std::string> HMGSDP::get_search_debug_info() const {
    return search_engine_->get_search_debug_info();
}

// ============================================================================
// 合并网络操作 - 委托给SearchEngine（内部转发给MergeNetwork）
// ============================================================================

void HMGSDP::force_merge_check() {
    search_engine_->force_merge_check();
}

void HMGSDP::recalculate_merge_probabilities() {
    search_engine_->recalculate_merge_probabilities();
}

void HMGSDP::rebuild_merge_network() {
    search_engine_->rebuild_merge_network();
}

// ============================================================================
// 多样性保护操作 - 委托给SearchEngine（内部转发给DiversityCalculator和SearchController）
// ============================================================================

void HMGSDP::identify_neglected_branches() {
    search_engine_->identify_neglected_branches();
}

void HMGSDP::allocate_resources_to_neglected_branches(double total_resources) {
    search_engine_->allocate_resources_to_neglected_branches(total_resources);
}

void HMGSDP::adjust_diversity_parameters() {
    search_engine_->adjust_diversity_parameters();
}

// ============================================================================
// 回溯与再发现 - 委托给SearchEngine（内部转发给SearchController）
// ============================================================================

void HMGSDP::perform_backtrack() {
    search_engine_->perform_backtrack();
}

void HMGSDP::rediscover_neglected_branches() {
    search_engine_->rediscover_neglected_branches();
}

// ============================================================================
// 数据导出 - 委托给SearchEngine（内部转发给DataExporter）
// ============================================================================

std::string HMGSDP::export_search_tree() const {
    return search_engine_->export_search_tree();
}

std::string HMGSDP::export_merge_network() const {
    return search_engine_->export_merge_network();
}

std::string HMGSDP::export_diversity_map() const {
    return search_engine_->export_diversity_map();
}

// ============================================================================
// 数据导入 - 委托给SearchEngine（内部转发给DataImporter）
// ============================================================================

void HMGSDP::import_search_tree(const std::string& tree_data) {
    search_engine_->import_search_tree(tree_data);
}

void HMGSDP::import_merge_network(const std::string& network_data) {
    search_engine_->import_merge_network(network_data);
}

void HMGSDP::import_diversity_map(const std::string& diversity_data) {
    search_engine_->import_diversity_map(diversity_data);
}

// ============================================================================
// 兼容旧版本导入 - 委托给SearchEngine（内部转发给DataImporter）
// ============================================================================

void HMGSDP::import_search_tree_v1(const std::string& tree_data) {
    search_engine_->import_search_tree_v1(tree_data);
}

void HMGSDP::import_merge_network_v1(const std::string& network_data) {
    search_engine_->import_merge_network_v1(network_data);
}

void HMGSDP::import_diversity_map_v1(const std::string& diversity_data) {
    search_engine_->import_diversity_map_v1(diversity_data);
}

} // namespace algorithms
} // namespace spacego
