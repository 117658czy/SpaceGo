#pragma once

#include <string>
#include <vector>
#include <utility>
#include <cmath>

namespace spacego {
namespace algorithms {

class SimilarityCalculator {
public:
    SimilarityCalculator() = default;
    
    double calculate_state_similarity(const std::string& state1, const std::string& state2) const;
    double calculate_state_similarity_fallback(const std::string& state1, const std::string& state2) const;
    double calculate_group_structure_similarity(const std::string& state1, const std::string& state2) const;
    double calculate_liberty_distribution_similarity(const std::string& state1, const std::string& state2) const;
    double calculate_strategic_position_similarity(const std::string& state1, const std::string& state2) const;
    
    double calculate_value_compatibility(double value1, double value2) const;
    double calculate_topology_consistency(double similarity, double value_compatibility) const;

private:
    std::vector<std::vector<std::pair<int, int>>> find_connected_groups(const std::string& state) const;
    std::vector<size_t> calculate_liberty_counts(const std::string& state) const;
    double calculate_position_agreement(const std::string& state1, const std::string& state2, 
                                        const std::vector<size_t>& liberty_counts) const;
    double calculate_mean(const std::vector<size_t>& values) const;
    double calculate_stddev(const std::vector<size_t>& values, double mean) const;
};

} // namespace algorithms
} // namespace spacego
