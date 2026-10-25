#define NOMINMAX
#include "go/search/spatial_topology_calculator.h"
#include "go/search/spatial_topology_accelerator.h"
#include "framework/chinese_logger.h"
#include <algorithm>
#include <numeric>
#include <queue>
#include <cstring>
#include <iostream>

#undef max
#undef min

namespace spacego {
namespace go {
namespace search {

// ==================== SpatialTopologyCalculator实现 ====================

SpatialTopologyCalculator::SpatialTopologyCalculator(int board_size)
    : board_size_(board_size)
    , half_board_(board_size / 2)
    , center_threshold_(board_size / 4) {
    
    initialize_region_caches();
}

SpatialTopologyCalculator::~SpatialTopologyCalculator() {
    // 释放加速器引用（shared_ptr自动管理生命周期）
    if (accelerator_) {
        accelerator_.reset();
    }
}

void SpatialTopologyCalculator::initialize_region_caches() {
    // 初始化空间区域缓存
    region_cache_.resize(board_size_, std::vector<SpatialRegion>(board_size_));
    
    // 初始化立方体区域缓存
    cube_cache_.resize(board_size_, std::vector<CubeRegion>(board_size_));
    
    for (int y = 0; y < board_size_; ++y) {
        for (int x = 0; x < board_size_; ++x) {
            // 计算空间区域
            float dist_center = distance_to_center(x, y);
            float dist_edge = distance_to_edge(x, y);
            
            if (dist_center < center_threshold_) {
                region_cache_[y][x] = SpatialRegion::CENTER;
            } else if (dist_edge < 2) {
                if (y < 3) {
                    region_cache_[y][x] = SpatialRegion::TOP_EDGE;
                } else if (y >= board_size_ - 3) {
                    region_cache_[y][x] = SpatialRegion::BOTTOM_EDGE;
                } else if (x < 3) {
                    region_cache_[y][x] = SpatialRegion::LEFT_EDGE;
                } else {
                    region_cache_[y][x] = SpatialRegion::RIGHT_EDGE;
                }
            } else {
                if (y < half_board_) {
                    region_cache_[y][x] = x < half_board_ ? SpatialRegion::TOP_LEFT : SpatialRegion::TOP_RIGHT;
                } else {
                    region_cache_[y][x] = x < half_board_ ? SpatialRegion::BOTTOM_LEFT : SpatialRegion::BOTTOM_RIGHT;
                }
            }
            
            // 计算立方体区域（8分区）
            int quarter = board_size_ / 4;
            int x_quad = x / quarter;
            int y_quad = y / quarter;
            
            // 限制在0-3范围内
            x_quad = std::min(3, std::max(0, x_quad));
            y_quad = std::min(3, std::max(0, y_quad));
            
            // 映射到8个立方体
            // 左上区域: CUBE_0, CUBE_1
            // 右上区域: CUBE_2, CUBE_3
            // 左下区域: CUBE_4, CUBE_5
            // 右下区域: CUBE_6, CUBE_7
            if (y_quad < 2) {
                if (x_quad < 2) {
                    cube_cache_[y][x] = (x_quad + y_quad) % 2 == 0 ? CubeRegion::CUBE_0 : CubeRegion::CUBE_1;
                } else {
                    cube_cache_[y][x] = (x_quad + y_quad) % 2 == 0 ? CubeRegion::CUBE_2 : CubeRegion::CUBE_3;
                }
            } else {
                if (x_quad < 2) {
                    cube_cache_[y][x] = (x_quad + y_quad) % 2 == 0 ? CubeRegion::CUBE_4 : CubeRegion::CUBE_5;
                } else {
                    cube_cache_[y][x] = (x_quad + y_quad) % 2 == 0 ? CubeRegion::CUBE_6 : CubeRegion::CUBE_7;
                }
            }
        }
    }
}

SpatialRegion SpatialTopologyCalculator::get_spatial_region(int x, int y) const {
    if (x < 0 || x >= board_size_ || y < 0 || y >= board_size_) {
        return SpatialRegion::UNKNOWN;
    }
    return region_cache_[y][x];
}

CubeRegion SpatialTopologyCalculator::get_cube_region(int x, int y) const {
    if (x < 0 || x >= board_size_ || y < 0 || y >= board_size_) {
        return CubeRegion::CUBE_0;
    }
    return cube_cache_[y][x];
}

std::pair<std::pair<int, int>, std::pair<int, int>> 
SpatialTopologyCalculator::get_cube_bounds(CubeRegion region) const {
    int quarter = board_size_ / 4;
    
    switch (region) {
        case CubeRegion::CUBE_0:
            return {{0, 0}, {quarter, quarter}};
        case CubeRegion::CUBE_1:
            return {{quarter, 0}, {2 * quarter, quarter}};
        case CubeRegion::CUBE_2:
            return {{2 * quarter, 0}, {3 * quarter, quarter}};
        case CubeRegion::CUBE_3:
            return {{3 * quarter, 0}, {board_size_, quarter}};
        case CubeRegion::CUBE_4:
            return {{0, quarter}, {quarter, 2 * quarter}};
        case CubeRegion::CUBE_5:
            return {{quarter, quarter}, {2 * quarter, 2 * quarter}};
        case CubeRegion::CUBE_6:
            return {{2 * quarter, quarter}, {3 * quarter, 2 * quarter}};
        case CubeRegion::CUBE_7:
            return {{3 * quarter, quarter}, {board_size_, 2 * quarter}};
        default:
            return {{0, 0}, {board_size_, board_size_}};
    }
}

std::vector<std::vector<char>> 
SpatialTopologyCalculator::parse_board_state(const std::string& board_state) const {
    std::vector<std::vector<char>> board(board_size_, std::vector<char>(board_size_, '.'));
    
    // 简单解析：假设board_state是按行优先的字符串
    size_t idx = 0;
    for (int y = 0; y < board_size_ && idx < board_state.size(); ++y) {
        for (int x = 0; x < board_size_ && idx < board_state.size(); ++x) {
            char c = board_state[idx++];
            if (c == 'B' || c == 'W' || c == '.') {
                board[y][x] = c;
            }
        }
    }
    
    return board;
}

float SpatialTopologyCalculator::distance_to_center(int x, int y) const {
    float cx = static_cast<float>(board_size_ - 1) / 2.0f;
    float cy = static_cast<float>(board_size_ - 1) / 2.0f;
    float dx = static_cast<float>(x) - cx;
    float dy = static_cast<float>(y) - cy;
    return std::sqrt(dx * dx + dy * dy);
}

float SpatialTopologyCalculator::distance_to_edge(int x, int y) const {
    float dist_top = static_cast<float>(y);
    float dist_bottom = static_cast<float>(board_size_ - 1 - y);
    float dist_left = static_cast<float>(x);
    float dist_right = static_cast<float>(board_size_ - 1 - x);
    
    return std::min({dist_top, dist_bottom, dist_left, dist_right});
}

SpatialTopologyFeatures 
SpatialTopologyCalculator::calculate_topology_features(const std::string& board_state) {
    SpatialTopologyFeatures features;
    
    // 计算连通性
    features.connectivity = calculate_connectivity(board_state);
    
    // 计算立方体状态
    auto cube_states = calculate_cube_states(board_state);
    
    // 计算平均立方体内聚性
    float total_coherence = 0.0f;
    for (const auto& cube : cube_states) {
        total_coherence += calculate_cube_coherence(cube);
    }
    features.cube_coherence = total_coherence / 8.0f;
    
    // 计算平均立方体间距离
    float total_distance = 0.0f;
    int count = 0;
    for (int i = 0; i < 8; ++i) {
        for (int j = i + 1; j < 8; ++j) {
            total_distance += calculate_inter_cube_distance(cube_states[i], cube_states[j]);
            count++;
        }
    }
    features.inter_cube_distance = count > 0 ? total_distance / static_cast<float>(count) : 0.0f;
    
    // 计算空间多样性
    features.spatial_diversity = calculate_spatial_diversity(cube_states);
    
    // 计算战略价值
    features.strategic_value = calculate_strategic_value(features);
    
    // 其他特征设为默认值（可以根据需要扩展）
    features.centrality = 0.5f;
    features.edge_proximity = 0.5f;
    features.region_importance = 0.5f;
    
    return features;
}

std::array<CubeState, 8> 
SpatialTopologyCalculator::calculate_cube_states(const std::string& board_state) {
    std::array<CubeState, 8> cube_states;
    auto board = parse_board_state(board_state);
    
    // 初始化每个立方体
    for (int i = 0; i < 8; ++i) {
        cube_states[i].region = static_cast<CubeRegion>(i);
    }
    
    // 统计每个立方体的棋子
    for (int y = 0; y < board_size_; ++y) {
        for (int x = 0; x < board_size_; ++x) {
            char c = board[y][x];
            if (c == 'B' || c == 'W') {
                CubeRegion region = get_cube_region(x, y);
                int idx = static_cast<int>(region);
                
                cube_states[idx].stone_count++;
                if (c == 'B') {
                    cube_states[idx].black_stones++;
                } else {
                    cube_states[idx].white_stones++;
                }
                
                // 记录关键点
                cube_states[idx].key_points.push_back({x, y});
            }
        }
    }
    
    // 计算影响力和领土控制
    for (auto& cube : cube_states) {
        if (cube.stone_count > 0) {
            float diff = static_cast<float>(cube.black_stones - cube.white_stones);
            cube.influence = std::tanh(diff / 10.0f);
            cube.territory_control = (static_cast<float>(cube.black_stones) - 
                                      static_cast<float>(cube.white_stones)) / 
                                     static_cast<float>(cube.stone_count);
        }
        cube.is_active = cube.stone_count > 0;
    }
    
    return cube_states;
}

float SpatialTopologyCalculator::calculate_cube_coherence(const CubeState& cube) {
    // 立方体内聚性：基于棋子分布的均匀性
    if (cube.key_points.size() < 2) {
        return 0.0f;
    }
    
    // 计算棋子之间的平均距离
    float total_dist = 0.0f;
    int count = 0;
    
    for (size_t i = 0; i < cube.key_points.size(); ++i) {
        for (size_t j = i + 1; j < cube.key_points.size(); ++j) {
            float dx = static_cast<float>(cube.key_points[i].first - cube.key_points[j].first);
            float dy = static_cast<float>(cube.key_points[i].second - cube.key_points[j].second);
            total_dist += std::sqrt(dx * dx + dy * dy);
            count++;
        }
    }
    
    if (count == 0) {
        return 0.0f;
    }
    
    float avg_dist = total_dist / static_cast<float>(count);
    // 距离越小，内聚性越高
    return 1.0f - std::min(1.0f, avg_dist / 10.0f);
}

float SpatialTopologyCalculator::calculate_inter_cube_distance(const CubeState& cube1, const CubeState& cube2) {
    // 获取两个立方体的中心点
    auto bounds1 = get_cube_bounds(cube1.region);
    auto bounds2 = get_cube_bounds(cube2.region);
    
    float cx1 = static_cast<float>(bounds1.first.first + bounds1.second.first) / 2.0f;
    float cy1 = static_cast<float>(bounds1.first.second + bounds1.second.second) / 2.0f;
    float cx2 = static_cast<float>(bounds2.first.first + bounds2.second.first) / 2.0f;
    float cy2 = static_cast<float>(bounds2.first.second + bounds2.second.second) / 2.0f;
    
    float dx = cx1 - cx2;
    float dy = cy1 - cy2;
    
    return std::sqrt(dx * dx + dy * dy) / static_cast<float>(board_size_);
}

float SpatialTopologyCalculator::calculate_connectivity(const std::string& board_state) {
    auto board = parse_board_state(board_state);
    
    // 简单的连通性计算：统计连通块数量
    std::vector<std::vector<bool>> visited(board_size_, std::vector<bool>(board_size_, false));
    int black_components = 0;
    int white_components = 0;
    
    for (int y = 0; y < board_size_; ++y) {
        for (int x = 0; x < board_size_; ++x) {
            if (!visited[y][x] && board[y][x] != '.') {
                char color = board[y][x];
                
                // BFS
                std::queue<std::pair<int, int>> q;
                q.push({x, y});
                visited[y][x] = true;
                
                while (!q.empty()) {
                    auto [cx, cy] = q.front();
                    q.pop();
                    
                    // 四个方向
                    const int dx[] = {-1, 1, 0, 0};
                    const int dy[] = {0, 0, -1, 1};
                    
                    for (int d = 0; d < 4; ++d) {
                        int nx = cx + dx[d];
                        int ny = cy + dy[d];
                        
                        if (nx >= 0 && nx < board_size_ && ny >= 0 && ny < board_size_ &&
                            !visited[ny][nx] && board[ny][nx] == color) {
                            visited[ny][nx] = true;
                            q.push({nx, ny});
                        }
                    }
                }
                
                if (color == 'B') {
                    black_components++;
                } else {
                    white_components++;
                }
            }
        }
    }
    
    // 连通性评分：连通块越少，连通性越高
    int total_components = black_components + white_components;
    float max_possible = static_cast<float>(board_size_ * board_size_);
    return 1.0f - static_cast<float>(total_components) / max_possible;
}

float SpatialTopologyCalculator::calculate_centrality(int x, int y) const {
    float dist = distance_to_center(x, y);
    float max_dist = static_cast<float>(board_size_) / 2.0f;
    return 1.0f - dist / max_dist;
}

float SpatialTopologyCalculator::calculate_edge_proximity(int x, int y) const {
    float dist = distance_to_edge(x, y);
    float max_dist = static_cast<float>(board_size_) / 2.0f;
    return 1.0f - dist / max_dist;
}

float SpatialTopologyCalculator::calculate_region_importance(SpatialRegion region) const {
    // 根据区域类型赋予不同的重要性
    switch (region) {
        case SpatialRegion::CENTER:
            return 1.0f;
        case SpatialRegion::TOP_LEFT:
        case SpatialRegion::TOP_RIGHT:
        case SpatialRegion::BOTTOM_LEFT:
        case SpatialRegion::BOTTOM_RIGHT:
            return 0.8f;
        case SpatialRegion::TOP_EDGE:
        case SpatialRegion::BOTTOM_EDGE:
        case SpatialRegion::LEFT_EDGE:
        case SpatialRegion::RIGHT_EDGE:
            return 0.6f;
        default:
            return 0.5f;
    }
}

float SpatialTopologyCalculator::calculate_spatial_diversity(const std::array<CubeState, 8>& cube_states) {
    // 计算空间多样性：基于各立方体的棋子数量分布
    std::vector<float> counts;
    for (const auto& cube : cube_states) {
        counts.push_back(static_cast<float>(cube.stone_count));
    }
    
    // 计算方差
    float mean = std::accumulate(counts.begin(), counts.end(), 0.0f) / 8.0f;
    float variance = 0.0f;
    for (float c : counts) {
        variance += (c - mean) * (c - mean);
    }
    variance /= 8.0f;
    
    // 方差越大，多样性越高
    float max_variance = static_cast<float>(board_size_ * board_size_) / 8.0f;
    return std::min(1.0f, variance / max_variance);
}

float SpatialTopologyCalculator::calculate_strategic_value(const SpatialTopologyFeatures& features) {
    // 综合各特征计算战略价值
    return 0.25f * features.connectivity +
           0.25f * features.cube_coherence +
           0.20f * features.spatial_diversity +
           0.15f * (1.0f - features.inter_cube_distance) +
           0.15f * features.centrality;
}

std::vector<SpatialTopologyFeatures> 
SpatialTopologyCalculator::batch_calculate_features(const std::vector<std::string>& board_states) {
    // 如果配置了加速器，委托给加速器处理
    if (accelerator_) {
        LOG_CN_INFO(spacego::framework::ChineseLogger::LogCategory::SEARCH_ALG,
                    "批量拓扑计算委托给加速器，批次大小: %zu", board_states.size());
        return accelerator_->calculate_batch(board_states);
    }

    // 原有CPU串行路径（向后兼容）
    std::vector<SpatialTopologyFeatures> results;
    results.reserve(board_states.size());
    
    for (const auto& state : board_states) {
        results.push_back(calculate_topology_features(state));
    }
    
    return results;
}

void SpatialTopologyCalculator::reset() {
    // 重置任何内部状态
}

void SpatialTopologyCalculator::set_accelerator(std::shared_ptr<SpatialTopologyAccelerator> accelerator) {
    accelerator_ = std::move(accelerator);
}

std::shared_ptr<SpatialTopologyAccelerator> SpatialTopologyCalculator::get_accelerator() const {
    return accelerator_;
}

bool SpatialTopologyCalculator::check_connection(
    const std::vector<std::vector<char>>& board, 
    int x1, int y1, int x2, int y2, char color) const {
    
    if (board[y1][x1] != color || board[y2][x2] != color) {
        return false;
    }
    
    // BFS检查连通性
    std::vector<std::vector<bool>> visited(board_size_, std::vector<bool>(board_size_, false));
    std::queue<std::pair<int, int>> q;
    q.push({x1, y1});
    visited[y1][x1] = true;
    
    const int dx[] = {-1, 1, 0, 0};
    const int dy[] = {0, 0, -1, 1};
    
    while (!q.empty()) {
        auto [cx, cy] = q.front();
        q.pop();
        
        if (cx == x2 && cy == y2) {
            return true;
        }
        
        for (int d = 0; d < 4; ++d) {
            int nx = cx + dx[d];
            int ny = cy + dy[d];
            
            if (nx >= 0 && nx < board_size_ && ny >= 0 && ny < board_size_ &&
                !visited[ny][nx] && board[ny][nx] == color) {
                visited[ny][nx] = true;
                q.push({nx, ny});
            }
        }
    }
    
    return false;
}

// ==================== SpatialTopologyManager实现 ====================

SpatialTopologyManager::SpatialTopologyManager(int board_size)
    : calculator_(board_size)
    , max_history_size_(100) {
    
    // 初始化立方体状态
    for (int i = 0; i < 8; ++i) {
        current_cube_states_[i].region = static_cast<CubeRegion>(i);
    }
}

SpatialTopologyManager::~SpatialTopologyManager() {
}

void SpatialTopologyManager::update_topology(const std::string& board_state) {
    current_features_ = calculator_.calculate_topology_features(board_state);
    current_cube_states_ = calculator_.calculate_cube_states(board_state);
    
    // 保存历史
    topology_history_.push_back(current_features_);
    if (topology_history_.size() > max_history_size_) {
        topology_history_.erase(topology_history_.begin());
    }
}

SpatialTopologyFeatures SpatialTopologyManager::get_current_features() const {
    return current_features_;
}

const std::array<CubeState, 8>& SpatialTopologyManager::get_cube_states() const {
    return current_cube_states_;
}

const CubeState& SpatialTopologyManager::get_cube_state(CubeRegion region) const {
    return current_cube_states_[static_cast<int>(region)];
}

float SpatialTopologyManager::calculate_position_value(int x, int y) const {
    float centrality = calculator_.calculate_centrality(x, y);
    float edge_proximity = calculator_.calculate_edge_proximity(x, y);
    SpatialRegion region = calculator_.get_spatial_region(x, y);
    float region_importance = calculator_.calculate_region_importance(region);
    
    // 综合评分
    return 0.4f * centrality + 0.3f * edge_proximity + 0.3f * region_importance;
}

std::vector<CubeRegion> SpatialTopologyManager::recommend_cubes_for_exploration() const {
    std::vector<std::pair<float, CubeRegion>> scores;
    
    for (int i = 0; i < 8; ++i) {
        const auto& cube = current_cube_states_[i];
        
        // 推荐策略：优先探索棋子少但潜力大的立方体
        float score = 0.0f;
        
        if (cube.stone_count < 5) {
            // 棋子少，优先级高
            score += 0.5f;
        }
        
        // 考虑区域重要性
        auto bounds = calculator_.get_cube_bounds(cube.region);
        float cx = static_cast<float>(bounds.first.first + bounds.second.first) / 2.0f;
        float cy = static_cast<float>(bounds.first.second + bounds.second.second) / 2.0f;
        float center_dist = std::sqrt(
            (cx - static_cast<float>(calculator_.get_board_size() - 1) / 2.0f) * 
            (cx - static_cast<float>(calculator_.get_board_size() - 1) / 2.0f) +
            (cy - static_cast<float>(calculator_.get_board_size() - 1) / 2.0f) * 
            (cy - static_cast<float>(calculator_.get_board_size() - 1) / 2.0f)
        );
        
        score += 0.3f * (1.0f - center_dist / static_cast<float>(calculator_.get_board_size()));
        
        // 活跃性
        if (cube.is_active) {
            score += 0.2f;
        }
        
        scores.push_back({-score, cube.region});  // 负号用于升序排序
    }
    
    // 按分数排序
    std::sort(scores.begin(), scores.end());
    
    std::vector<CubeRegion> result;
    for (const auto& [score, region] : scores) {
        result.push_back(region);
    }
    
    return result;
}

const std::vector<SpatialTopologyFeatures>& SpatialTopologyManager::get_topology_history() const {
    return topology_history_;
}

void SpatialTopologyManager::clear_history() {
    topology_history_.clear();
}

} // namespace search
} // namespace go
} // namespace spacego
