#pragma once

#include <memory>
#include <cmath>
#include <limits>
#include "network/algorithms/hmgs_node.h"
#include "network/algorithms/config/hmgs_config.h"

namespace spacego {
namespace algorithms {

class UCTCalculator {
public:
    explicit UCTCalculator(const HMGSConfig& config);
    
    double calculate_topology_uct(std::shared_ptr<HMGSNode> child, 
                                  std::shared_ptr<HMGSNode> parent, 
                                  size_t child_index) const;
    
    std::shared_ptr<HMGSNode> select_best_child(std::shared_ptr<HMGSNode> node) const;
    
    std::shared_ptr<HMGSNode> select_leaf_node(std::shared_ptr<HMGSNode> start_node) const;

private:
    HMGSConfig config_;
};

} // namespace algorithms
} // namespace spacego
