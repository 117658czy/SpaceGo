// 模拟执行器实现文件
// 包含所有围棋模拟操作的具体实现

#include "network/algorithms/execution/simulation_executor.h"
#include "go/board/go_board_cuda.h"
#include "framework/logger.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_set>
#include <queue>
#include <sstream>
#include <iomanip>

namespace spacego {
namespace algorithms {

// 构造函数
SimulationExecutor::SimulationExecutor()
    : initialized_(false) {
}

// 析构函数
SimulationExecutor::~SimulationExecutor() {
}

// 初始化
void SimulationExecutor::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    initialized_.store(true, std::memory_order_release);
    
    LOG_INFO("模拟执行器初始化完成");
}

// 执行完整的围棋模拟
double SimulationExecutor::perform_full_simulation(const std::string& start_state) const {
    
    // 模拟参数配置
    const size_t MAX_MOVES = 400;        // 最大步数（19x19棋盘约361个点）
    const size_t PASS_THRESHOLD = 2;      // 连续pass次数阈值（双方连续pass则终局）
    const double RANDOM_MOVE_PROB = 0.3;  // 随机探索概率
    
    // 初始化模拟状态
    std::string current_state = start_state;
    std::vector<std::string> move_history;
    std::unordered_set<std::string> position_history;  // 用于检测重复局面（劫）
    
    // 统计信息
    size_t consecutive_passes = 0;
    size_t move_count = 0;
    char current_player = 'B';  // 黑棋先行（默认假设）
    
    // 确定当前玩家（基于棋盘状态中的棋子数量判断）
    size_t black_count = 0, white_count = 0;
    for (char c : current_state) {
        if (c == 'B') black_count++;
        else if (c == 'W') white_count++;
    }
    current_player = (black_count <= white_count) ? 'B' : 'W';
    
    // 主模拟循环：持续对弈直到达到终止条件
    while (move_count < MAX_MOVES && consecutive_passes < PASS_THRESHOLD) {
        
        // 生成当前状态下所有合法落子
        std::vector<std::string> legal_moves = generate_legal_moves_for_simulation(
            current_state, current_player);
        
        if (legal_moves.empty()) {
            // 没有合法落子，只能选择pass
            move_history.push_back(current_state + "_pass");
            consecutive_passes++;
            
            // 切换到对方
            current_player = (current_player == 'B') ? 'W' : 'B';
            move_count++;
            continue;
        }
        
        // 使用随机探索策略选择落子
        static std::random_device rd;  // 随机设备（硬件随机数生成器）
        static std::mt19937 gen(rd());  // 梅森旋转算法随机数生成器
        
        std::string next_state;
        
        // 随机选择一个合法走法
        std::uniform_int_distribution<> move_dis(0, legal_moves.size() - 1);
        next_state = legal_moves[move_dis(gen)];
        
        // 检查重复局面（劫检测）：防止无限循环
        if (position_history.find(next_state) != position_history.end()) {
            // 检测到可能的劫争，尝试选择其他非重复的走法
            bool found_alternative = false;
            for (const auto& move : legal_moves) {
                if (move != next_state && 
                    position_history.find(move) == position_history.end()) {
                    next_state = move;
                    found_alternative = true;
                    break;
                }
            }
            
            if (!found_alternative) {
                // 所有合法落子都会导致重复局面，只能pass
                move_history.push_back(current_state + "_pass");
                consecutive_passes++;
                
                current_player = (current_player == 'B') ? 'W' : 'B';
                move_count++;
                continue;
            }
        }
        
        // 执行选定的落子
        move_history.push_back(next_state);
        position_history.insert(current_state);  // 记录当前状态用于劫检测
        current_state = next_state;
        
        // 重置连续pass计数（因为成功执行了落子）
        consecutive_passes = 0;
        
        // 切换玩家
        current_player = (current_player == 'B') ? 'W' : 'B';
        move_count++;
    }
    
    // 模拟结束，计算最终得分（使用中国规则：地盘+活子）
    double final_score = calculate_final_score(current_state);
    
    // 根据当前玩家视角归一化得分到[-1, 1]范围
    // 得分>0表示黑棋胜，<0表示白棋胜
    double normalized_score = final_score;
    
    // 确保结果在有效范围内
    normalized_score = std::max(-1.0, std::min(1.0, normalized_score));
    
    LOG_DEBUG("模拟完成: 步数=%zu, 最终得分=%.3f", move_count, normalized_score);
    
    return normalized_score;
}

// 生成合法落子位置列表
std::vector<std::string> SimulationExecutor::generate_legal_moves(const std::string& state) const {
    std::vector<std::string> legal_moves;
    
    // 验证状态字符串长度是否为标准19x19棋盘
    if (state.size() != 361) {
        return legal_moves;  // 无效状态，返回空列表
    }
    
    // 创建棋盘对象用于规则检查
    go::GoBoardCPU board(state);
    
    // 统计黑白棋子数量以确定当前轮到哪方
    size_t black_count = 0, white_count = 0;
    for (char c : state) {
        if (c == 'B') black_count++;
        else if (c == 'W') white_count++;
    }
    
    // 黑棋子数<=白棋子数时轮到黑棋下，否则轮到白棋
    go::StoneColor player = (black_count <= white_count) ? go::StoneColor::BLACK : go::StoneColor::WHITE;
    
    // 获取所有合法落子位置
    auto legal_positions = board.get_legal_moves(player);
    
    // 为每个合法位置生成新的状态字符串
    for (const auto& [x, y] : legal_positions) {
        go::GoBoardCPU new_board = board;  // 复制当前棋盘
        new_board.set_stone(x, y, player);  // 在该位置放置棋子（自动处理提子）
        legal_moves.push_back(new_board.to_string());  // 转换为状态字符串
    }
    
    // 始终添加pass选项
    legal_moves.push_back(state + "_pass");
    
    return legal_moves;
}

// 为模拟生成合法落子（带玩家信息）
std::vector<std::string> SimulationExecutor::generate_legal_moves_for_simulation(
    const std::string& state, 
    char player) const {
    
    std::vector<std::string> legal_moves;
    
    // 验证状态有效性
    if (state.size() != 361) {
        return legal_moves;
    }
    
    // 创建棋盘对象
    go::GoBoardCPU board(state);
    
    // 将字符标识转换为StoneColor枚举
    go::StoneColor stone_player = (player == 'B') ? go::StoneColor::BLACK : go::StoneColor::WHITE;
    
    // 获取合法落子位置
    auto legal_positions = board.get_legal_moves(stone_player);
    
    // 生成每个合法落子的新状态
    for (const auto& [x, y] : legal_positions) {
        go::GoBoardCPU new_board = board;
        new_board.set_stone(x, y, stone_player);
        legal_moves.push_back(new_board.to_string());
    }
    
    // 添加pass选项
    legal_moves.push_back(state + "_pass");
    
    return legal_moves;
}

// 检查落子是否合法（模拟专用快速检查）
bool SimulationExecutor::is_legal_move_for_simulation(
    const std::string& state, 
    size_t pos, 
    char player, 
    char opponent) const {
    
    // 基本边界和占用检查
    if (pos >= state.size() || state[pos] != 'E') {
        return false;  // 超出范围或位置已被占用
    }
    
    // 使用规则引擎进行完整合法性验证
    go::GoBoardCPU board(state);
    go::StoneColor stone_player = (player == 'B') ? go::StoneColor::BLACK : go::StoneColor::WHITE;
    
    // 将线性索引转换为行列坐标
    int x = pos % 19;   // 列坐标（0-18）
    int y = pos / 19;   // 行坐标（0-18）
    
    // 调用规则引擎的合法性检查方法
    return board.is_legal_move(x, y, stone_player);
}

// 执行提子操作（模拟专用）
std::string SimulationExecutor::capture_stones_for_simulation(
    const std::string& state, 
    size_t move_pos, 
    char player, 
    char opponent) const {
    
    // 创建棋盘对象并执行落子（自动处理提子）
    go::GoBoardCPU board(state);
    go::StoneColor stone_player = (player == 'B') ? go::StoneColor::BLACK : go::StoneColor::WHITE;
    
    // 转换坐标并执行落子
    int x = move_pos % 19;
    int y = move_pos / 19;
    
    board.set_stone(x, y, stone_player);  // 自动处理提子逻辑
    
    // 返回执行后的新状态
    return board.to_string();
}

// 计算最终得分（中国规则实现）
double SimulationExecutor::calculate_final_score(const std::string& final_state) const {
    
    // 中国规则计分：地盘 + 活子
    
    // 初始化数据结构
    std::vector<bool> visited(361, false);           // 访问标记数组
    std::vector<int> territory_owner(361, 0);         // 领地所有权: 0=未确定, 1=黑棋, -1=白棋
    const int directions[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};  // 四个方向偏移量
    
    // 第一步：识别所有活棋块及其气
    std::unordered_set<size_t> black_living_stones;   // 黑棋活子集合
    std::unordered_set<size_t> white_living_stones;   // 白棋活子集合
    
    // 遍历所有棋盘点位
    for (size_t pos = 0; pos < 361; ++pos) {
        if (!visited[pos] && final_state[pos] != 'E') {
            // 发现未访问的非空点，开始BFS搜索连通的棋块
            char stone_color = final_state[pos];
            std::queue<size_t> bfs_queue;
            std::unordered_set<size_t> group;  // 当前棋块的所有棋子位置
            
            // 初始化BFS
            bfs_queue.push(pos);
            visited[pos] = true;
            
            // 广度优先搜索找出整个连通棋块
            while (!bfs_queue.empty()) {
                size_t current = bfs_queue.front();
                bfs_queue.pop();
                
                group.insert(current);  // 加入当前棋块
                
                // 获取当前位置的行列坐标
                int row = current / 19;
                int col = current % 19;
                
                // 向四个方向扩展搜索同色棋子
                for (int d = 0; d < 4; ++d) {
                    int new_row = row + directions[d][0];
                    int new_col = col + directions[d][1];
                    
                    // 边界检查
                    if (new_row >= 0 && new_row < 19 && new_col >= 0 && new_col < 19) {
                        size_t neighbor_pos = static_cast<size_t>(new_row * 19 + new_col);
                        
                        // 只扩展相同颜色的相邻棋子
                        if (!visited[neighbor_pos] && final_state[neighbor_pos] == stone_color) {
                            visited[neighbor_pos] = true;
                            bfs_queue.push(neighbor_pos);
                        }
                    }
                }
            }
            
            // 检查该棋块是否有气（即是否为活棋）
            bool has_liberty_value = false;
            for (size_t stone_pos : group) {
                int row = stone_pos / 19;
                int col = stone_pos % 19;
                
                // 检查四个方向是否有空点（气）
                for (int d = 0; d < 4; ++d) {
                    int new_row = row + directions[d][0];
                    int new_col = col + directions[d][1];
                    
                    if (new_row >= 0 && new_row < 19 && new_col >= 0 && new_col < 19) {
                        size_t neighbor_pos = static_cast<size_t>(new_row * 19 + new_col);
                        
                        if (final_state[neighbor_pos] == 'E') {
                            has_liberty_value = true;  // 发现气
                            break;
                        }
                    }
                }
                
                if (has_liberty_value) break;  // 已确认有气，无需继续检查
            }
            
            // 根据是否有气将棋块分类为活棋或死棋
            if (has_liberty_value) {
                // 活棋：加入对应颜色的活子集合
                if (stone_color == 'B') {
                    black_living_stones.insert(group.begin(), group.end());
                } else {
                    white_living_stones.insert(group.begin(), group.end());
                }
            }
            // 死棋不计入得分（简化实现中不处理复杂死活判定）
        }
    }
    
    // 第二步：确定领地所有权（分析空点的归属）
    std::fill(visited.begin(), visited.end(), false);  // 重置访问标记
    
    for (size_t pos = 0; pos < 361; ++pos) {
        if (!visited[pos] && final_state[pos] == 'E') {
            // 发现未访问的空点，分析其所属的空白区域
            std::queue<size_t> bfs_queue;
            std::unordered_set<size_t> blank_area;       // 空白区域的所有空点
            std::unordered_set<char> adjacent_colors;     // 与该区域相邻的棋子颜色
            
            // 初始化BFS
            bfs_queue.push(pos);
            visited[pos] = true;
            
            // 广度优先搜索找出整个连通的空白区域
            while (!bfs_queue.empty()) {
                size_t current = bfs_queue.front();
                bfs_queue.pop();
                
                blank_area.insert(current);  // 加入空白区域
                
                // 获取坐标
                int row = current / 19;
                int col = current / 19;
                
                // 向四个方向扩展
                for (int d = 0; d < 4; ++d) {
                    int new_row = row + directions[d][0];
                    int new_col = col + directions[d][1];
                    
                    if (new_row >= 0 && new_row < 19 && new_col >= 0 && new_col < 19) {
                        size_t neighbor_pos = static_cast<size_t>(new_row * 19 + new_col);
                        char neighbor_color = final_state[neighbor_pos];
                        
                        if (neighbor_color == 'E' && !visited[neighbor_pos]) {
                            // 相邻的是空点，继续扩展空白区域
                            visited[neighbor_pos] = true;
                            bfs_queue.push(neighbor_pos);
                        } else if (neighbor_color == 'B' || neighbor_color == 'W') {
                            // 相邻的是棋子，记录颜色
                            adjacent_colors.insert(neighbor_color);
                        }
                    }
                }
            }
            
            // 确定领地所有权
            if (adjacent_colors.size() == 1) {
                // 只与一种颜色相邻 -> 完全属于该颜色的领地
                char owner = *adjacent_colors.begin();
                int owner_value = (owner == 'B') ? 1 : -1;  // 黑棋+1，白棋-1
                
                // 标记该空白区域的所有点位
                for (size_t territory_pos : blank_area) {
                    territory_owner[territory_pos] = owner_value;
                }
            }
            // 如果与两种颜色都相邻 -> 公共领地（如单官），不计入任何一方
        }
    }
    
    // 第三步：计算最终得分
    double black_score = 0.0;
    double white_score = 0.0;
    
    // 领地得分
    for (size_t pos = 0; pos < 361; ++pos) {
        if (territory_owner[pos] == 1) {
            black_score += 1.0;      // 黑棋领地
        } else if (territory_owner[pos] == -1) {
            white_score += 1.0;      // 白棋领地
        }
    }
    
    // 活子得分（中国规则特点：活子也计入得分）
    black_score += black_living_stones.size();
    white_score += white_living_stones.size();
    
    // 返回净胜分（黑棋视角：正值表示黑棋胜）
    return black_score - white_score;
}

// 坐标转换工具方法
std::string SimulationExecutor::coordinate_to_string(int row, int col) const {
    
    // 围棋坐标系统规范：
    // 列：A-T（跳过字母I以避免与数字1混淆）-> A=0, B=1, ..., H=7, J=8, ..., T=18
    // 行：1-19（1-based编号）-> 1=0, 2=1, ..., 19=18
    
    // 边界检查
    if (row < 0 || row >= 19 || col < 0 || col >= 19) {
        return "INVALID";  // 无效坐标
    }
    
    // 构建列字母（跳过I）
    char column_letter;
    if (col < 8) {
        // A-H范围（0-7）
        column_letter = 'A' + col;
    } else {
        // J-T范围（8-18），跳过I所以+1
        column_letter = 'A' + col + 1;
    }
    
    // 构建行号（转换为1-based）
    int row_number = row + 1;
    
    // 组合成标准坐标字符串（如"K10"、"D4"等）
    return std::string(1, column_letter) + std::to_string(row_number);
}

} // namespace algorithms
} // namespace spacego
