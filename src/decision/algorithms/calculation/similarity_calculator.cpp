#include "network/algorithms/calculation/similarity_calculator.h"
#include <algorithm>
#include <numeric>
#include <set>

namespace spacego {
namespace algorithms {

double SimilarityCalculator::calculate_state_similarity(const std::string& state1, const std::string& state2) const {
    const size_t BOARD_SIZE = 361; // 19x19
    
    if (state1.size() != BOARD_SIZE || state2.size() != BOARD_SIZE) {
        return calculate_state_similarity_fallback(state1, state2);
    }
    
    double position_similarity = 0.0;
    size_t same_position_count = 0;
    size_t valid_positions = 0;
    
    for (size_t i = 0; i < BOARD_SIZE; ++i) {
        char c1 = state1[i];
        char c2 = state2[i];
        
        if (c1 == 'X' || c2 == 'X') {
            continue;
        }
        
        valid_positions++;
        if (c1 == c2) {
            same_position_count++;
        }
    }
    
    if (valid_positions > 0) {
        position_similarity = static_cast<double>(same_position_count) / valid_positions;
    }
    
    double group_similarity = calculate_group_structure_similarity(state1, state2);
    double liberty_similarity = calculate_liberty_distribution_similarity(state1, state2);
    double strategic_similarity = calculate_strategic_position_similarity(state1, state2);
    
    double total_similarity = 
        0.4 * position_similarity +
        0.3 * group_similarity +
        0.2 * liberty_similarity +
        0.1 * strategic_similarity;
    
    total_similarity = std::pow(total_similarity, 0.8);
    
    return std::max(0.0, std::min(1.0, total_similarity));
}

double SimilarityCalculator::calculate_state_similarity_fallback(const std::string& state1, const std::string& state2) const {
    size_t min_len = std::min(state1.size(), state2.size());
    size_t max_len = std::max(state1.size(), state2.size());
    
    if (max_len == 0) {
        return 0.0;
    }
    
    size_t matches = 0;
    const size_t MAX_OFFSET = 3;
    
    for (size_t i = 0; i < min_len; ++i) {
        bool matched = false;
        
        if (state1[i] == state2[i]) {
            matched = true;
        } else {
            for (size_t offset = 1; offset <= MAX_OFFSET && !matched; ++offset) {
                if (i + offset < min_len && state1[i + offset] == state2[i]) {
                    matched = true;
                }
                if (i >= offset && state1[i - offset] == state2[i]) {
                    matched = true;
                }
            }
        }
        
        if (matched) {
            ++matches;
        }
    }
    
    double base_similarity = static_cast<double>(matches) / max_len;
    double length_ratio = static_cast<double>(min_len) / max_len;
    double length_similarity = 1.0 - 0.2 * (1.0 - length_ratio);
    
    return base_similarity * length_similarity;
}

double SimilarityCalculator::calculate_group_structure_similarity(const std::string& state1, const std::string& state2) const {
    auto groups1 = find_connected_groups(state1);
    auto groups2 = find_connected_groups(state2);
    
    if (groups1.empty() && groups2.empty()) {
        return 1.0;
    }
    
    if (groups1.empty() || groups2.empty()) {
        return 0.5;
    }
    
    size_t matching_groups = 0;
    const double POSITION_TOLERANCE = 2.0;
    
    for (const auto& group1 : groups1) {
        double center1_r = 0.0, center1_c = 0.0;
        for (const auto& pos : group1) {
            center1_r += pos.first;
            center1_c += pos.second;
        }
        center1_r /= group1.size();
        center1_c /= group1.size();
        
        for (const auto& group2 : groups2) {
            double center2_r = 0.0, center2_c = 0.0;
            for (const auto& pos : group2) {
                center2_r += pos.first;
                center2_c += pos.second;
            }
            center2_r /= group2.size();
            center2_c /= group2.size();
            
            double distance = std::sqrt(std::pow(center1_r - center2_r, 2) + 
                                        std::pow(center1_c - center2_c, 2));
            
            if (distance < POSITION_TOLERANCE && 
                std::abs(static_cast<int>(group1.size()) - static_cast<int>(group2.size())) <= 2) {
                matching_groups++;
                break;
            }
        }
    }
    
    double similarity = static_cast<double>(matching_groups) / 
                       std::max(groups1.size(), groups2.size());
    
    return similarity;
}

double SimilarityCalculator::calculate_value_compatibility(double value1, double value2) const {
    double diff = std::abs(value1 - value2);
    double compatibility = std::exp(-diff * 2.0);
    return compatibility;
}

double SimilarityCalculator::calculate_topology_consistency(double similarity, double value_compatibility) const {
    double consistency = 0.6 * similarity + 0.4 * value_compatibility;
    return consistency;
}

std::vector<std::vector<std::pair<int, int>>> SimilarityCalculator::find_connected_groups(const std::string& state) const {
    std::vector<std::vector<std::pair<int, int>>> groups;
    const int BOARD_DIM = 19;
    
    if (static_cast<int>(state.size()) != BOARD_DIM * BOARD_DIM) {
        return groups;
    }
    
    std::vector<std::vector<bool>> visited(BOARD_DIM, std::vector<bool>(BOARD_DIM, false));
    
    for (int r = 0; r < BOARD_DIM; ++r) {
        for (int c = 0; c < BOARD_DIM; ++c) {
            int idx = r * BOARD_DIM + c;
            
            if (visited[r][c] || state[idx] == '.' || state[idx] == 'X') {
                continue;
            }
            
            char color = state[idx];
            std::vector<std::pair<int, int>> group;
            std::vector<std::pair<int, int>> stack;
            stack.push_back({r, c});
            
            while (!stack.empty()) {
                auto [cr, cc] = stack.back();
                stack.pop_back();
                
                if (cr < 0 || cr >= BOARD_DIM || cc < 0 || cc >= BOARD_DIM) {
                    continue;
                }
                
                if (visited[cr][cc]) {
                    continue;
                }
                
                int cidx = cr * BOARD_DIM + cc;
                if (state[cidx] != color) {
                    continue;
                }
                
                visited[cr][cc] = true;
                group.push_back({cr, cc});
                
                stack.push_back({cr - 1, cc});
                stack.push_back({cr + 1, cc});
                stack.push_back({cr, cc - 1});
                stack.push_back({cr, cc + 1});
            }
            
            if (!group.empty()) {
                groups.push_back(group);
            }
        }
    }
    
    return groups;
}

double SimilarityCalculator::calculate_liberty_distribution_similarity(const std::string& state1, const std::string& state2) const {
    auto liberties1 = calculate_liberty_counts(state1);
    auto liberties2 = calculate_liberty_counts(state2);
    
    if (liberties1.empty() && liberties2.empty()) {
        return 1.0;
    }
    
    if (liberties1.empty() || liberties2.empty()) {
        return 0.3;
    }
    
    double mean1 = calculate_mean(liberties1);
    double mean2 = calculate_mean(liberties2);
    double stddev1 = calculate_stddev(liberties1, mean1);
    double stddev2 = calculate_stddev(liberties2, mean2);
    
    double mean_diff = std::abs(mean1 - mean2) / (std::max(mean1, mean2) + 1e-6);
    double stddev_diff = std::abs(stddev1 - stddev2) / (std::max(stddev1, stddev2) + 1e-6);
    
    double similarity = 1.0 - (0.7 * mean_diff + 0.3 * stddev_diff);
    
    return std::max(0.0, std::min(1.0, similarity));
}

std::vector<size_t> SimilarityCalculator::calculate_liberty_counts(const std::string& state) const {
    std::vector<size_t> liberty_counts;
    auto groups = find_connected_groups(state);
    const int BOARD_DIM = 19;
    
    for (const auto& group : groups) {
        std::set<std::pair<int, int>> liberties;
        
        for (const auto& [r, c] : group) {
            int dr[] = {-1, 1, 0, 0};
            int dc[] = {0, 0, -1, 1};
            
            for (int d = 0; d < 4; ++d) {
                int nr = r + dr[d];
                int nc = c + dc[d];
                
                if (nr >= 0 && nr < BOARD_DIM && nc >= 0 && nc < BOARD_DIM) {
                    int idx = nr * BOARD_DIM + nc;
                    if (state[idx] == '.') {
                        liberties.insert({nr, nc});
                    }
                }
            }
        }
        
        liberty_counts.push_back(liberties.size());
    }
    
    return liberty_counts;
}

double SimilarityCalculator::calculate_strategic_position_similarity(const std::string& state1, const std::string& state2) const {
    auto liberties1 = calculate_liberty_counts(state1);
    auto liberties2 = calculate_liberty_counts(state2);
    
    double position_agreement1 = calculate_position_agreement(state1, state2, liberties1);
    double position_agreement2 = calculate_position_agreement(state2, state1, liberties2);
    
    return (position_agreement1 + position_agreement2) / 2.0;
}

double SimilarityCalculator::calculate_position_agreement(const std::string& state1, const std::string& state2, 
                                                           const std::vector<size_t>& liberty_counts) const {
    const int BOARD_DIM = 19;
    const size_t BOARD_SIZE = BOARD_DIM * BOARD_DIM;
    
    if (state1.size() != BOARD_SIZE || state2.size() != BOARD_SIZE) {
        return 0.0;
    }
    
    size_t agreements = 0;
    size_t total_checked = 0;
    
    for (int r = 0; r < BOARD_DIM; ++r) {
        for (int c = 0; c < BOARD_DIM; ++c) {
            int idx = r * BOARD_DIM + c;
            
            if (state1[idx] == '.' || state1[idx] == 'X' || state2[idx] == '.' || state2[idx] == 'X') {
                continue;
            }
            
            bool is_corner_or_edge = (r <= 2 || r >= BOARD_DIM - 3 || c <= 2 || c >= BOARD_DIM - 3);
            bool is_center = (r >= 7 && r <= 11 && c >= 7 && c <= 11);
            
            if (!is_corner_or_edge && !is_center) {
                continue;
            }
            
            total_checked++;
            
            if (state1[idx] == state2[idx]) {
                agreements++;
            }
        }
    }
    
    if (total_checked == 0) {
        return 1.0;
    }
    
    return static_cast<double>(agreements) / total_checked;
}

double SimilarityCalculator::calculate_mean(const std::vector<size_t>& values) const {
    if (values.empty()) {
        return 0.0;
    }
    
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / values.size();
}

double SimilarityCalculator::calculate_stddev(const std::vector<size_t>& values, double mean) const {
    if (values.size() <= 1) {
        return 0.0;
    }
    
    double sum_sq_diff = 0.0;
    for (size_t val : values) {
        double diff = static_cast<double>(val) - mean;
        sum_sq_diff += diff * diff;
    }
    
    return std::sqrt(sum_sq_diff / (values.size() - 1));
}

} // namespace algorithms
} // namespace spacego
