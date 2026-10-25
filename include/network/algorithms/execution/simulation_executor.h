#pragma once

// 模拟执行器头文件
// 职责：负责围棋对弈模拟的所有操作
// 包括完整模拟、合法走法生成、得分计算等核心功能

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <functional>

namespace spacego {
namespace algorithms {

// 模拟执行器类
class SimulationExecutor {
public:
    // 构造函数和析构函数
    SimulationExecutor();
    ~SimulationExecutor();

    // 初始化
    void initialize();
    
    // 核心模拟方法
    
    // 执行完整的围棋模拟
    // 从给定状态开始，进行随机或半随机对弈直到终局
    // 返回归一化的最终得分，范围[-1.0, 1.0]
    double perform_full_simulation(const std::string& start_state) const;
    
    // 生成合法落子位置列表
    // 使用围棋规则引擎检查所有可能的落子位置
    std::vector<std::string> generate_legal_moves(const std::string& state) const;
    
    // 为模拟生成合法落子（带玩家信息）
    // 专门用于模拟过程中的走法生成
    std::vector<std::string> generate_legal_moves_for_simulation(
        const std::string& state, 
        char player) const;
    
    // 检查落子是否合法（模拟专用）
    // 快速合法性检查，用于模拟过程中验证走法
    bool is_legal_move_for_simulation(
        const std::string& state, 
        size_t pos, 
        char player, 
        char opponent) const;
    
    // 执行提子操作（模拟专用）
    // 在指定位置落子并自动处理提子逻辑
    std::string capture_stones_for_simulation(
        const std::string& state, 
        size_t move_pos, 
        char player, 
        char opponent) const;
    
    // 计算最终得分（中国规则）
    // 基于地盘+活子的中国规则计算净胜分
    double calculate_final_score(const std::string& final_state) const;
    
    // 坐标转换工具方法
    // 将行列索引转换为标准围棋坐标字符串（如"K10"）
    std::string coordinate_to_string(int row, int col) const;

private:
    // 线程安全锁
    mutable std::mutex mutex_;
    
    // 是否已初始化
    std::atomic<bool> initialized_{false};
};

} // namespace algorithms
} // namespace spacego
