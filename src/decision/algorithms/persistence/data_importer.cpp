// 数据导入器实现文件
// 包含所有数据导入操作的具体实现
// 支持V2.0完整格式和V1.0简化格式的向后兼容

#include "network/algorithms/persistence/data_importer.h"
#include "framework/logger.h"
#include <sstream>
#include <algorithm>
#include <chrono>

namespace spacego {
namespace algorithms {

// 构造函数
DataImporter::DataImporter()
    : initialized_(false) {
}

// 析构函数
DataImporter::~DataImporter() {
}

// 初始化
void DataImporter::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    initialized_.store(true, std::memory_order_release);
    
    LOG_INFO("数据导入器初始化完成");
}

// 导入完整的搜索树结构（V2.0格式）
bool DataImporter::import_search_tree(
    const std::string& data,
    std::shared_ptr<HMGSNode>& root_node,
    std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
    std::vector<std::shared_ptr<HMGSNode>>& main_branches,
    HMGSStatistics& stats) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 清空现有数据结构以准备导入新数据
    node_registry.clear();
    main_branches.clear();
    root_node.reset();
    
    // 创建字符串流用于逐行解析
    std::stringstream ss(data);
    std::string line;
    
    // 状态变量用于跟踪解析进度
    bool in_nodes_section = false;
    bool in_main_branches_section = false;
    
    // 临时存储用于重建节点关系
    std::unordered_map<std::string, std::string> parent_map;  // 节点ID -> 父节点ID
    std::unordered_map<std::string, std::vector<std::string>> children_map;  // 节点ID -> 子节点ID列表
    std::unordered_map<std::string, std::vector<std::string>> similar_nodes_map;  // 相似节点映射
    
    // 当前正在构建的节点指针
    std::shared_ptr<HMGSNode> current_node = nullptr;
    
    // 逐行解析导出文件
    while (std::getline(ss, line)) {
        // 跳过空行
        if (line.empty()) continue;
        
        // 检测各个段的开始和结束标记
        if (line.find("METADATA_BEGIN") != std::string::npos) {
            // 元数据段开始，继续读取统计信息
            continue;
        }
        
        if (line.find("METADATA_END") != std::string::npos) {
            continue;
        }
        
        if (line.find("NODES_BEGIN") != std::string::npos) {
            in_nodes_section = true;
            continue;
        }
        
        if (line.find("NODES_END") != std::string::npos) {
            in_nodes_section = false;
            continue;
        }
        
        if (line.find("MAIN_BRANCHES_BEGIN") != std::string::npos) {
            in_main_branches_section = true;
            continue;
        }
        
        if (line.find("MAIN_BRANCHES_END") != std::string::npos) {
            in_main_branches_section = false;
            continue;
        }
        
        if (line.find("SEARCH_TREE_EXPORT_END") != std::string::npos) {
            break;  // 文件结束，停止解析
        }
        
        // 解析元数据（全局统计信息）
        if (!in_nodes_section && !in_main_branches_section) {
            if (line.find("TOTAL_SIMULATIONS:") == 0) {
                stats.total_simulations = std::stoull(line.substr(16));
            } else if (line.find("COMPLETED_SIMULATIONS:") == 0) {
                stats.completed_simulations = std::stoull(line.substr(21));
            } else if (line.find("SUCCESSFUL_MERGES:") == 0) {
                stats.successful_merges = std::stoull(line.substr(18));
            } else if (line.find("FAILED_MERGES:") == 0) {
                stats.failed_merges = std::stoull(line.substr(15));
            } else if (line.find("NEGLECTED_BRANCHES:") == 0) {
                stats.neglected_branches = std::stoull(line.substr(19));
            } else if (line.find("BACKTRACK_EVENTS:") == 0) {
                stats.backtrack_events = std::stoull(line.substr(17));
            } else if (line.find("ROOT_NODE:") == 0) {
                std::string root_id = line.substr(10);
                // 根节点ID稍后用于查找实际对象
            }
            continue;
        }
        
        // 解析主分支信息
        if (in_main_branches_section) {
            if (line.find("BRANCH:") == 0) {
                // 格式: "BRANCH:index:ROOT:node_id"
                auto parts = split_string(line.substr(7), ':');
                if (parts.size() >= 3) {
                    std::string branch_root_id = parts[2];
                    // 稍后从node_registry中查找并添加到main_branches
                    // 这里先记录ID，等所有节点创建后再建立引用
                }
            }
            continue;
        }
        
        // 解析节点信息
        if (in_nodes_section) {
            if (line.find("NODE_START:") == 0) {
                // 新节点开始，提取节点ID并创建节点对象
                std::string node_id = line.substr(11);
                
                // 创建新的节点实例
                current_node = std::make_shared<HMGSNode>("");
                
                // 注册到节点注册表
                node_registry[node_id] = current_node;
                
                continue;
            }
            
            if (line.find("NODE_END:") == 0) {
                // 当前节点信息结束，重置当前节点指针
                current_node = nullptr;
                continue;
            }
            
            // 如果没有当前节点，跳过属性设置
            if (!current_node) continue;
            
            // 解析节点的各种属性
            if (line.find("PARENT:") == 0) {
                parent_map[current_node->get_id()] = line.substr(7);
            } 
            else if (line.find("DEPTH:") == 0) {
                current_node->set_depth(std::stoull(line.substr(6)));
            } 
            else if (line.find("STATE:") == 0) {
                current_node->set_state(line.substr(6));
            } 
            else if (line.find("VISITS:") == 0) {
                current_node->set_visit_count(std::stoull(line.substr(7)));
            } 
            else if (line.find("TOTAL_VALUE:") == 0) {
                current_node->set_total_value(std::stod(line.substr(12)));
            } 
            else if (line.find("AVERAGE_VALUE:") == 0) {
                // 平均价值是只读的，通过total_value和visits自动计算
                // 这里不需要单独设置
            } 
            else if (line.find("MERGE_PROB:") == 0) {
                current_node->set_merge_prob(std::stod(line.substr(10)));
            } 
            else if (line.find("MERGED_INTO_PROB:") == 0) {
                current_node->set_merged_into_prob(std::stod(line.substr(17)));
            } 
            else if (line.find("CONTINUOUS_UNMERGED:") == 0) {
                current_node->set_continuous_unmerged(std::stoull(line.substr(20)));
            } 
            else if (line.find("NOVELTY_SCORE:") == 0) {
                current_node->set_novelty_score(std::stod(line.substr(14)));
            } 
            else if (line.find("BACKTRACK_COUNT:") == 0) {
                current_node->set_backtrack_count(std::stoull(line.substr(16)));
            } 
            else if (line.find("UNCERTAINTY:") == 0) {
                current_node->set_uncertainty(std::stod(line.substr(12)));
            } 
            else if (line.find("IS_EXPANDED:") == 0) {
                current_node->set_expanded(line.substr(13) == "true");
            } 
            else if (line.find("IS_REPLICA:") == 0) {
                current_node->set_is_replica(line.substr(12) == "true");
            } 
            else if (line.find("BRANCH_INDEX:") == 0) {
                current_node->set_branch_index(std::stoull(line.substr(14)));
            } 
            else if (line.find("REPLICA_INDEX:") == 0) {
                current_node->set_replica_index(std::stoull(line.substr(15)));
            } 
            else if (line.find("REPLICA_COUNT:") == 0) {
                current_node->set_replica_count(std::stoull(line.substr(14)));
            } 
            else if (line.find("BRANCH_QUALITY:") == 0) {
                current_node->set_branch_quality(std::stod(line.substr(15)));
            } 
            else if (line.find("EXPLORATION_BIAS:") == 0) {
                current_node->set_exploration_bias(std::stod(line.substr(17)));
            } 
            else if (line.find("TOPOLOGY_WEIGHT:") == 0) {
                current_node->set_topology_weight(std::stod(line.substr(16)));
            } 
            else if (line.find("NOVELTY_REWARD:") == 0) {
                current_node->set_novelty_reward(std::stod(line.substr(15)));
            } 
            else if (line.find("CHILDREN:") == 0) {
                // 解析子节点ID列表（逗号分隔）
                std::string children_str = line.substr(9);
                children_map[current_node->get_id()] = split_string(children_str, ',');
            } 
            else if (line.find("SIMILAR_NODES:") == 0) {
                // 解析相似节点ID列表
                std::string similar_str = line.substr(14);
                similar_nodes_map[current_node->get_id()] = split_string(similar_str, ',');
            }
        }
    }
    
    // 所有节点创建完成后，重建节点间的父子关系
    rebuild_node_relationships(node_registry, parent_map);
    
    // 建立子节点关系
    for (const auto& entry : children_map) {
        auto it = node_registry.find(entry.first);
        if (it != node_registry.end() && it->second) {
            for (const auto& child_id : entry.second) {
                auto child_it = node_registry.find(child_id);
                if (child_it != node_registry.end() && child_it->second) {
                    it->second->add_child(child_it->second);
                }
            }
        }
    }
    
    // 建立相似节点关系
    for (const auto& entry : similar_nodes_map) {
        auto it = node_registry.find(entry.first);
        if (it != node_registry.end() && it->second) {
            for (const auto& similar_id : entry.second) {
                auto similar_it = node_registry.find(similar_id);
                if (similar_it != node_registry.end() && similar_it->second) {
                    it->second->add_similar_node(similar_it->second);
                }
            }
        }
    }
    
    // 查找并设置根节点
    // 需要遍历找到根节点（在parent_map中没有条目或父节点为NULL的节点）
    for (const auto& entry : node_registry) {
        auto parent_it = parent_map.find(entry.first);
        if (parent_it == parent_map.end() || parent_it->second == "NULL") {
            // 尝试转换为根节点类型
            root_node = std::dynamic_pointer_cast<HMGSRootNode>(entry.second);
            if (root_node) {
                break;  // 找到根节点
            }
        }
    }
    
    // 重建主分支列表
    // 需要重新扫描分支信息（这里简化处理：使用已有的main_branches重建逻辑）
    // 实际应该重新解析MAIN_BRANCHES部分或根据branch_index属性收集
    
    LOG_INFO("搜索树导入完成: %zu个节点", node_registry.size());
    
    return !node_registry.empty();
}

// 导入合并网络信息
bool DataImporter::import_merge_network(
    const std::string& data,
    std::unordered_map<std::string, double>& merge_probabilities,
    std::unordered_map<std::string, double>& merged_into_probabilities,
    std::unordered_map<std::string, std::vector<std::string>>& merge_relationships,
    HMGSStatistics& stats) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 清空现有数据
    merge_probabilities.clear();
    merged_into_probabilities.clear();
    merge_relationships.clear();
    
    std::stringstream ss(data);
    std::string line;
    
    bool in_relationships_section = false;
    bool in_probs_section = false;
    bool in_merged_into_probs_section = false;
    bool in_history_section = false;
    
    while (std::getline(ss, line)) {
        if (line.empty()) continue;
        
        // 检测各段的开始/结束标记
        if (line.find("MERGE_RELATIONSHIPS_BEGIN") != std::string::npos) {
            in_relationships_section = true;
            continue;
        }
        if (line.find("MERGE_RELATIONSHIPS_END") != std::string::npos) {
            in_relationships_section = false;
            continue;
        }
        
        if (line.find("MERGE_PROBABILITIES_BEGIN") != std::string::npos) {
            in_probs_section = true;
            continue;
        }
        if (line.find("MERGE_PROBABILITIES_END") != std::string::npos) {
            in_probs_section = false;
            continue;
        }
        
        if (line.find("MERGED_INTO_PROBABILITIES_BEGIN") != std::string::npos) {
            in_merged_into_probs_section = true;
            continue;
        }
        if (line.find("MERGED_INTO_PROBABILITIES_END") != std::string::npos) {
            in_merged_into_probs_section = false;
            continue;
        }
        
        if (line.find("MERGE_HISTORY_BEGIN") != std::string::npos) {
            in_history_section = true;
            continue;
        }
        if (line.find("MERGE_HISTORY_END") != std::string::npos) {
            in_history_section = false;
            continue;
        }
        
        if (line.find("MERGE_NETWORK_EXPORT_END") != std::string::npos) {
            break;
        }
        
        // 解析合并关系记录
        if (in_relationships_section && line.find("NODE:") == 0) {
            // 格式: "NODE:node_id:MERGE_TARGETS:target1,target2,..."
            size_t first_colon = line.find(':', 5);  // 跳过"NODE:"
            size_t second_colon = line.find(':', first_colon + 1);
            
            if (first_colon != std::string::npos && second_colon != std::string::npos) {
                std::string node_id = line.substr(5, first_colon - 5);
                std::string targets_str = line.substr(second_colon + 1);  // "MERGE_TARGETS:"之后的内容
                
                // 提取实际的target列表（去掉"MERGE_TARGETS:"前缀）
                size_t targets_start = targets_str.find(':');
                if (targets_start != std::string::npos) {
                    std::string targets_list = targets_str.substr(targets_start + 1);
                    merge_relationships[node_id] = split_string(targets_list, ',');
                }
            }
            continue;
        }
        
        // 解析合并概率
        if (in_probs_section && line.find("NODE:") == 0) {
            // 格式: "NODE:node_id:PROB:value"
            auto parts = split_string(line.substr(5), ':');
            if (parts.size() >= 2) {
                merge_probabilities[parts[0]] = std::stod(parts[1]);
            }
            continue;
        }
        
        // 解析被合并概率
        if (in_merged_into_probs_section && line.find("NODE:") == 0) {
            // 格式: "NODE:node_id:MERGED_INTO_PROB:value"
            auto parts = split_string(line.substr(5), ':');
            if (parts.size() >= 2) {
                merged_into_probabilities[parts[0]] = std::stod(parts[parts.size() - 1]);  // 取最后一部分
            }
            continue;
        }
        
        // 解析合并历史统计
        if (in_history_section) {
            if (line.find("SUCCESSFUL_MERGES:") == 0) {
                stats.successful_merges = std::stoull(line.substr(18));
            } else if (line.find("FAILED_MERGES:") == 0) {
                stats.failed_merges = std::stoull(line.substr(15));
            }
        }
    }
    
    LOG_INFO("合并网络导入完成: %zu个概率, %zu个关系", 
             merge_probabilities.size(), merge_relationships.size());
    
    return !merge_probabilities.empty();
}

// 导入多样性保护地图
bool DataImporter::import_diversity_map(
    const std::string& data,
    std::vector<std::string>& neglected_branches,
    std::unordered_map<std::string, size_t>& unmerged_counts,
    std::unordered_map<std::string, std::chrono::steady_clock::time_point>& last_visit_times,
    HMGSStatistics& stats) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 清空现有数据
    neglected_branches.clear();
    unmerged_counts.clear();
    last_visit_times.clear();
    
    std::stringstream ss(data);
    std::string line;
    
    bool in_neglected_section = false;
    bool in_unmerged_section = false;
    bool in_visit_times_section = false;
    bool in_stats_section = false;
    
    while (std::getline(ss, line)) {
        if (line.empty()) continue;
        
        // 检测各段的标记
        if (line.find("NEGLECTED_BRANCHES_BEGIN") != std::string::npos) {
            in_neglected_section = true;
            continue;
        }
        if (line.find("NEGLECTED_BRANCHES_END") != std::string::npos) {
            in_neglected_section = false;
            continue;
        }
        
        if (line.find("UNMERGED_COUNTS_BEGIN") != std::string::npos) {
            in_unmerged_section = true;
            continue;
        }
        if (line.find("UNMERGED_COUNTS_END") != std::string::npos) {
            in_unmerged_section = false;
            continue;
        }
        
        if (line.find("LAST_VISIT_TIMES_BEGIN") != std::string::npos) {
            in_visit_times_section = true;
            continue;
        }
        if (line.find("LAST_VISIT_TIMES_END") != std::string::npos) {
            in_visit_times_section = false;
            continue;
        }
        
        if (line.find("DIVERSITY_STATISTICS_BEGIN") != std::string::npos) {
            in_stats_section = true;
            continue;
        }
        if (line.find("DIVERSITY_STATISTICS_END") != std::string::npos) {
            in_stats_section = false;
            continue;
        }
        
        if (line.find("DIVERSITY_MAP_EXPORT_END") != std::string::npos) {
            break;
        }
        
        // 解析被忽视分支列表
        if (in_neglected_section && line.find("BRANCH_ID:") == 0) {
            // 格式: "BRANCH_ID:id:..." 或 "BRANCH_ID:id:STATUS:..."
            auto parts = split_string(line.substr(10), ':');
            if (!parts.empty() && parts[0] != "STATUS") {
                neglected_branches.push_back(parts[0]);
            }
            continue;
        }
        
        // 解析未合并计数
        if (in_unmerged_section && line.find("NODE:") == 0) {
            // 格式: "NODE:node_id:UNMERGED_COUNT:count:..."
            auto parts = split_string(line.substr(5), ':');
            if (parts.size() >= 2) {
                std::string node_id = parts[0];
                // 查找UNMERGED_COUNT字段的位置
                for (size_t i = 0; i < parts.size(); ++i) {
                    if (parts[i] == "UNMERGED_COUNT" && i + 1 < parts.size()) {
                        unmerged_counts[node_id] = std::stoull(parts[i + 1]);
                        break;
                    }
                }
            }
            continue;
        }
        
        // 解析最后访问时间（注意：时间戳无法完全精确恢复，使用当前时间近似）
        if (in_visit_times_section && line.find("NODE:") == 0) {
            auto parts = split_string(line.substr(5), ':');
            if (!parts.empty()) {
                // 使用当前时间作为近似值（原始绝对时间戳可能已失效）
                last_visit_times[parts[0]] = std::chrono::steady_clock::now();
            }
            continue;
        }
        
        // 解析多样性统计信息
        if (in_stats_section) {
            if (line.find("NEGLECTED_BRANCHES_TOTAL:") == 0) {
                stats.neglected_branches = std::stoull(line.substr(23));
            } else if (line.find("BACKTRACK_EVENTS_TOTAL:") == 0) {
                stats.backtrack_events = std::stoull(line.substr(22));
            } else if (line.find("AVERAGE_BRANCH_VALUE:") == 0) {
                stats.average_branch_value = std::stod(line.substr(21));
            } else if (line.find("MAX_BRANCH_VALUE:") == 0) {
                stats.max_branch_value = std::stod(line.substr(17));
            } else if (line.find("MIN_BRANCH_VALUE:") == 0) {
                stats.min_branch_value = std::stod(line.substr(17));
            }
        }
    }
    
    LOG_INFO("多样性地图导入完成: %zu个被忽视分支", neglected_branches.size());
    
    return true;
}

// 导入V1.0格式的搜索树（向后兼容简化版）
bool DataImporter::import_search_tree_v1(
    const std::string& data,
    std::shared_ptr<HMGSNode>& root_node,
    std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
    HMGSStatistics& stats) {
    
    // V1.0格式是V2.0的子集，直接调用完整版本处理
    // V2.0导入器会自动忽略V1.0中不存在的字段
    std::vector<std::shared_ptr<HMGSNode>> dummy_branches;  // V1.0不支持主分支概念
    return import_search_tree(data, root_node, node_registry, dummy_branches, stats);
}

// 导入V1.0格式的合并网络（向后兼容简化版）
bool DataImporter::import_merge_network_v1(
    const std::string& data,
    std::unordered_map<std::string, double>& merge_probabilities,
    HMGSStatistics& stats) {
    
    // V1.0格式只支持基本的合并概率，不支持被合并概率和详细关系
    std::unordered_map<std::string, double> dummy_merged_into;
    std::unordered_map<std::string, std::vector<std::string>> dummy_relationships;
    
    return import_merge_network(data, merge_probabilities, dummy_merged_into, 
                              dummy_relationships, stats);
}

// 导入V1.0格式的多样性地图（向后兼容简化版）
bool DataImporter::import_diversity_map_v1(
    const std::string& data,
    std::vector<std::string>& neglected_branches,
    HMGSStatistics& stats) {
    
    // V1.0格式只支持基本的被忽视分支列表
    std::unordered_map<std::string, size_t> dummy_unmerged_counts;
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> dummy_last_visits;
    
    return import_diversity_map(data, neglected_branches, dummy_unmerged_counts, 
                               dummy_last_visits, stats);
}

// 辅助方法：解析键值对行
std::pair<std::string, std::string> DataImporter::parse_key_value(const std::string& line) const {
    size_t colon_pos = line.find(':');
    
    if (colon_pos == std::string::npos) {
        return {line, ""};  // 没有冒号，返回整行作为key
    }
    
    std::string key = line.substr(0, colon_pos);
    std::string value = line.substr(colon_pos + 1);
    
    return {key, value};
}

// 辅助方法：分割字符串为子串列表
std::vector<std::string> DataImporter::split_string(const std::string& str, char delimiter) const {
    std::vector<std::string> tokens;
    std::stringstream token_stream(str);
    std::string token;
    
    while (std::getline(token_stream, token, delimiter)) {
        if (!token.empty()) {  // 忽略空串
            tokens.push_back(token);
        }
    }
    
    return tokens;
}

// 辅助方法：重建节点间的父子关系
void DataImporter::rebuild_node_relationships(
    std::unordered_map<std::string, std::shared_ptr<HMGSNode>>& node_registry,
    const std::unordered_map<std::string, std::string>& parent_map) {
    
    // 遍历所有父子关系记录
    for (const auto& entry : parent_map) {
        const std::string& child_id = entry.first;
        const std::string& parent_id = entry.second;
        
        // 查找子节点
        auto child_it = node_registry.find(child_id);
        if (child_it == node_registry.end() || !child_it->second) {
            continue;  // 子节点不存在，跳过
        }
        
        // 处理特殊情况：父节点为NULL表示这是根节点
        if (parent_id == "NULL" || parent_id.empty()) {
            child_it->second->set_parent(nullptr);  // 无父节点
            continue;
        }
        
        // 查找父节点
        auto parent_it = node_registry.find(parent_id);
        if (parent_it != node_registry.end() && parent_it->second) {
            // 建立父子关系
            child_it->second->set_parent(parent_it->second);
            
            // 同时将子节点添加到父节点的children列表中
            parent_it->second->add_child(child_it->second);
        } else {
            // 父节点不存在于注册表中，记录警告
            LOG_WARNING("导入警告: 节点%s的父节点%s不存在", child_id.c_str(), parent_id.c_str());
        }
    }
    
    LOG_DEBUG("节点关系重建完成");
}

} // namespace algorithms
} // namespace spacego
