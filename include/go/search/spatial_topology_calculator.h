#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cmath>
#include <array>

namespace spacego {
namespace go {
namespace search {

// 空间区域枚举
enum class SpatialRegion : uint8_t {
    TOP_LEFT = 0,
    TOP_RIGHT = 1,
    BOTTOM_LEFT = 2,
    BOTTOM_RIGHT = 3,
    CENTER = 4,
    TOP_EDGE = 5,
    BOTTOM_EDGE = 6,
    LEFT_EDGE = 7,
    RIGHT_EDGE = 8,
    UNKNOWN = 9
};

// 8分区立方体区域（对应文档中的8个初始立方体区域）
enum class CubeRegion : uint8_t {
    CUBE_0 = 0,  // 左上1
    CUBE_1 = 1,  // 左上2
    CUBE_2 = 2,  // 右上1
    CUBE_3 = 3,  // 右上2
    CUBE_4 = 4,  // 左下1
    CUBE_5 = 5,  // 左下2
    CUBE_6 = 6,  // 右下1
    CUBE_7 = 7   // 右下2
};

// 空间拓扑特征
struct SpatialTopologyFeatures {
    float connectivity;           // 连通性评分
    float centrality;             // 中心性评分
    float edge_proximity;         // 边缘接近度
    float region_importance;      // 区域重要性
    float cube_coherence;         // 立方体内聚性
    float inter_cube_distance;    // 立方体间距离
    float spatial_diversity;      // 空间多样性
    float strategic_value;        // 战略价值

    SpatialTopologyFeatures()
        : connectivity(0.0f)
        , centrality(0.0f)
        , edge_proximity(0.0f)
        , region_importance(0.0f)
        , cube_coherence(0.0f)
        , inter_cube_distance(0.0f)
        , spatial_diversity(0.0f)
        , strategic_value(0.0f) {}
};

// 立方体状态
struct CubeState {
    CubeRegion region;
    int stone_count;
    int black_stones;
    int white_stones;
    float influence;
    float territory_control;
    bool is_active;
    std::vector<std::pair<int, int>> key_points;

    CubeState()
        : region(CubeRegion::CUBE_0)
        , stone_count(0)
        , black_stones(0)
        , white_stones(0)
        , influence(0.0f)
        , territory_control(0.0f)
        , is_active(true) {}
};

// 前置声明空间拓扑加速器（避免循环依赖）
class SpatialTopologyAccelerator;

/**
 * @brief 空间拓扑结构计算器
 * @details 实现文档中设想的空间拓扑结构计算，包括8分区立方体架构
 */
class SpatialTopologyCalculator {
public:
    SpatialTopologyCalculator(int board_size = 19);
    ~SpatialTopologyCalculator();

    // 计算空间拓扑特征
    SpatialTopologyFeatures calculate_topology_features(const std::string& board_state);

    // 获取位置所属的空间区域
    SpatialRegion get_spatial_region(int x, int y) const;

    // 获取位置所属的立方体区域
    CubeRegion get_cube_region(int x, int y) const;

    // 计算8个立方体的状态
    std::array<CubeState, 8> calculate_cube_states(const std::string& board_state);

    // 计算立方体内聚性
    float calculate_cube_coherence(const CubeState& cube);

    // 计算立方体间距离
    float calculate_inter_cube_distance(const CubeState& cube1, const CubeState& cube2);

    // 计算连通性
    float calculate_connectivity(const std::string& board_state);

    // 计算中心性
    float calculate_centrality(int x, int y) const;

    // 计算边缘接近度
    float calculate_edge_proximity(int x, int y) const;

    // 计算区域重要性
    float calculate_region_importance(SpatialRegion region) const;

    // 计算空间多样性
    float calculate_spatial_diversity(const std::array<CubeState, 8>& cube_states);

    // 计算战略价值
    float calculate_strategic_value(const SpatialTopologyFeatures& features);

    // 获取立方体的边界坐标
    std::pair<std::pair<int, int>, std::pair<int, int>> get_cube_bounds(CubeRegion region) const;

    // 批量计算多个位置的拓扑特征
    std::vector<SpatialTopologyFeatures> batch_calculate_features(
        const std::vector<std::string>& board_states);

    // 重置计算器状态
    void reset();

    // 设置空间拓扑加速器（可选）
    // 设置后将由加速器处理批量计算请求
    void set_accelerator(std::shared_ptr<SpatialTopologyAccelerator> accelerator);

    // 获取当前加速器
    std::shared_ptr<SpatialTopologyAccelerator> get_accelerator() const;

    // 获取棋盘大小
    int get_board_size() const { return board_size_; }

    // 解析棋盘状态（供空间计算加速器使用）
    std::vector<std::vector<char>> parse_board_state(const std::string& board_state) const;

    // 检查连通性（使用简单的BFS，供空间计算加速器使用）
    bool check_connection(const std::vector<std::vector<char>>& board, 
                         int x1, int y1, int x2, int y2, char color) const;

private:
    int board_size_;
    int half_board_;
    int center_threshold_;

    // 空间拓扑加速器指针（可选，为null时使用原有CPU路径）
    std::shared_ptr<SpatialTopologyAccelerator> accelerator_;

    // 预计算的区域映射
    std::vector<std::vector<SpatialRegion>> region_cache_;
    std::vector<std::vector<CubeRegion>> cube_cache_;

    // 初始化区域缓存
    void initialize_region_caches();

    // 计算位置到中心的欧氏距离
    float distance_to_center(int x, int y) const;

    // 计算位置到最近边缘的距离
    float distance_to_edge(int x, int y) const;
};

/**
 * @brief 空间拓扑结构管理器
 * @details 管理整个搜索过程中的空间拓扑结构
 */
class SpatialTopologyManager {
public:
    SpatialTopologyManager(int board_size = 19);
    ~SpatialTopologyManager();

    // 更新拓扑结构
    void update_topology(const std::string& board_state);

    // 获取当前拓扑特征
    SpatialTopologyFeatures get_current_features() const;

    // 获取立方体状态
    const std::array<CubeState, 8>& get_cube_states() const;

    // 获取特定立方体的状态
    const CubeState& get_cube_state(CubeRegion region) const;

    // 计算位置的拓扑价值
    float calculate_position_value(int x, int y) const;

    // 推荐探索的立方体
    std::vector<CubeRegion> recommend_cubes_for_exploration() const;

    // 获取拓扑历史
    const std::vector<SpatialTopologyFeatures>& get_topology_history() const;

    // 清除历史
    void clear_history();

    // 获取计算器
    SpatialTopologyCalculator& get_calculator() { return calculator_; }
    const SpatialTopologyCalculator& get_calculator() const { return calculator_; }

private:
    SpatialTopologyCalculator calculator_;
    SpatialTopologyFeatures current_features_;
    std::array<CubeState, 8> current_cube_states_;
    std::vector<SpatialTopologyFeatures> topology_history_;
    size_t max_history_size_;
};

} // namespace search
} // namespace go
} // namespace spacego
