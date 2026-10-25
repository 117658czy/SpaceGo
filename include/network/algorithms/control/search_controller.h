#pragma once

// 搜索控制器头文件
// 职责：负责整个HMGS-DP搜索算法的流程控制和协调
// 包括搜索启动/停止、线程管理、分支管理、结果输出等核心功能

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <chrono>
#include <thread>
#include <future>
#include <functional>

#include "network/algorithms/config/hmgs_config.h"
#include "network/algorithms/hmgs_node.h"

// 前向声明其他专职类
namespace spacego {
namespace algorithms {
class DiversityCalculator;
class NodeManager;
class MergeNetwork;
class SimulationExecutor;
struct HMGSStatistics;
}
}

namespace spacego {
namespace algorithms {

// 搜索控制器类
class SearchController {
public:
    // 构造函数和析构函数
    SearchController();
    ~SearchController();

    // 初始化：注入依赖的其他专职类实例
    void initialize(
        DiversityCalculator* diversity_calc,
        NodeManager* node_mgr,
        MergeNetwork* merge_net,
        SimulationExecutor* sim_executor,
        const HMGSConfig& config);
    
    // 搜索生命周期管理方法
    
    // 启动搜索
    // 创建根节点、初始化并行分支、启动后台搜索和监控线程
    void start_search(const std::string& initial_state);
    
    // 停止搜索
    // 设置停止标志并等待所有线程安全退出
    void stop_search();
    
    // 暂停搜索
    // 设置暂停标志，搜索线程将进入等待状态
    void pause_search();
    
    // 恢复搜索
    // 清除暂停标志，搜索线程将继续执行
    void resume_search();
    
    // 核心搜索执行方法
    
    // 执行单次模拟
    // 选择一个分支并运行完整的模拟-扩展-回溯流程
    void run_simulation();
    
    // 执行多次模拟
    // 批量运行指定次数的模拟
    void run_simulations(size_t count);
    
    // 获取最佳走法
    // 从所有分支中选择价值最高的走法作为推荐结果
    std::string get_best_move() const;
    
    // 获取走法概率分布
    // 返回每个候选走法的访问概率，用于策略分析
    std::vector<std::pair<std::string, double>> get_move_probabilities() const;
    
    // 更新根节点
    // 在实际落子后更新搜索树的根节点到新状态
    void update_root(const std::string& move);
    
    // 重置搜索树
    // 清空树结构但保留根节点，重新初始化分支
    void reset_tree();
    
    // 清空搜索树
    // 完全清除所有节点和数据结构
    void clear_tree();
    
    // 后台线程方法
    
    // 后台搜索主线程
    // 持续执行模拟循环直到达到停止条件
    void background_search_thread();
    
    // 监控线程
    // 定期调整参数、重新计算概率等维护工作
    void monitoring_thread();
    
    // 并行分支管理方法
    
    // 初始化并行分支
    // 创建配置数量的主分支及其副本
    void initialize_parallel_branches();
    
    // 运行单个分支的模拟
    // 管理该分支下所有副本的模拟执行和结果合并
    void run_branch_simulation(size_t branch_index);
    
    // 运行单个副本的模拟
    // 执行指定次数的完整模拟并返回统计结果
    std::pair<double, bool> run_replica_simulation(size_t branch_index, size_t replica_index);
    
    // 合并副本结果
    // 将同一分支下所有副本的结果汇总到分支节点
    void merge_replica_results(size_t branch_index);
    
    // 管理所有分支
    // 定期检查分支质量、调整资源分配、触发合并和多样性保护
    void manage_branches();
    
    // 多样性和回溯方法
    
    // 执行回溯操作
    // 选择被忽视的节点并重置其统计信息以鼓励重新探索
    void perform_backtrack();
    
    // 重新发现被忽视分支
    // 对长期被忽视的分支进行恢复处理
    void rediscover_neglected_branches();
    
    // 识别被忽视的分支
    // 分析所有节点的访问情况找出需要保护的分支
    void identify_neglected_branches();
    
    // 为被忽视分支分配资源
    // 将额外的模拟资源分配给被忽视的分支
    void allocate_resources_to_neglected_branches(double total_resources);
    
    // 调整多样性参数
    // 基于当前多样性水平动态调整探索参数
    void adjust_diversity_parameters();
    
    // 选择回溯目标节点
    // 从候选节点中选择最适合回溯的节点
    std::shared_ptr<HMGSNode> select_backtrack_node() const;
    
    // 状态查询方法
    
    // 检查是否正在搜索
    bool is_searching() const;
    
    // 检查是否已初始化
    bool is_initialized() const;
    
    // 获取统计信息
    HMGSStatistics get_statistics() const;

    // 数据访问方法（供SearchEngine委托调用使用）
    std::shared_ptr<HMGSRootNode> get_root_node() const { return root_node_; }
    void set_root_node(std::shared_ptr<HMGSRootNode> root) { root_node_ = root; }

    std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& get_node_registry() { return node_registry_; }
    const std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& get_node_registry() const { return node_registry_; }

    std::vector<std::shared_ptr<HMGSNode>>& get_main_branches() { return main_branches_; }
    const std::vector<std::shared_ptr<HMGSNode>>& get_main_branches() const { return main_branches_; }

    std::unordered_map<std::string, double>& get_merge_probabilities() { return merge_probabilities_; }
    const std::unordered_map<std::string, double>& get_merge_probabilities() const { return merge_probabilities_; }

    std::unordered_map<std::string, double>& get_merged_into_probabilities() { return merged_into_probabilities_; }
    const std::unordered_map<std::string, double>& get_merged_into_probabilities() const { return merged_into_probabilities_; }

    std::unordered_map<std::string, std::vector<std::string>>& get_merge_relationships() { return merge_relationships_; }
    const std::unordered_map<std::string, std::vector<std::string>>& get_merge_relationships() const { return merge_relationships_; }

    std::unordered_map<std::string, size_t>& get_unmerged_counts() { return unmerged_counts_; }
    const std::unordered_map<std::string, size_t>& get_unmerged_counts() const { return unmerged_counts_; }

    std::unordered_map<std::string, std::chrono::steady_clock::time_point>& get_last_visit_times() { return last_visit_times_; }
    const std::unordered_map<std::string, std::chrono::steady_clock::time_point>& get_last_visit_times() const { return last_visit_times_; }

    std::vector<std::string>& get_neglected_branches() { return neglected_branches_; }
    const std::vector<std::string>& get_neglected_branches() const { return neglected_branches_; }

    HMGSStatistics& get_stats() { return stats_; }
    const HMGSStatistics& get_stats() const { return stats_; }

    const HMGSConfig& get_config() const { return config_; }

    // 获取搜索调试信息
    std::vector<std::string> get_search_debug_info() const;

private:
    // 依赖注入的专职类指针
    DiversityCalculator* diversity_calculator_;
    NodeManager* node_manager_;
    MergeNetwork* merge_network_;
    SimulationExecutor* simulation_executor_;
    
    // 配置
    HMGSConfig config_;
    
    // 搜索状态标志
    std::atomic<bool> initialized_{false};
    std::atomic<bool> searching_{false};
    std::atomic<bool> paused_{false};
    std::atomic<bool> shutdown_requested_{false};
    
    // 搜索树数据结构
    std::shared_ptr<HMGSRootNode> root_node_;                                          // 根节点
    std::unordered_map<std::string, std::shared_ptr<HMGSNode>> node_registry_;          // 全局节点注册表
    std::vector<std::shared_ptr<HMGSNode>> main_branches_;                              // 主分支列表
    
    // 合并网络数据结构
    std::unordered_map<std::string, double> merge_probabilities_;                       // 合并概率映射
    std::unordered_map<std::string, double> merged_into_probabilities_;                 // 被合并概率映射
    std::unordered_map<std::string, std::vector<std::string>> merge_relationships_;     // 合并关系记录
    std::unordered_map<std::string, size_t> unmerged_counts_;                           // 未合并计数
    
    // 多样性保护数据结构
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> last_visit_times_;  // 最后访问时间
    std::vector<std::string> neglected_branches_;                                       // 被忽视分支列表
    
    // 统计信息
    HMGSStatistics stats_;
    
    // 线程管理
    std::unique_ptr<std::thread> search_thread_;        // 后台搜索线程
    std::unique_ptr<std::thread> monitoring_thread_;     // 监控线程
    
    // 线程安全锁
    mutable std::mutex tree_mutex_;      // 搜索树锁
    mutable std::mutex config_mutex_;    // 配置锁
    mutable std::mutex stats_mutex_;     // 统计信息锁
};

} // namespace algorithms
} // namespace spacego
