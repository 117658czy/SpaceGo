#pragma once

#include "go/board/board_state.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace spacego {
namespace go {

struct MoveResult {
    bool success;
    std::vector<int> captured_positions;
    int ko_position;
    bool is_ko;
    
    MoveResult() : success(false), ko_position(-1), is_ko(false) {}
};

class GoBoardCPU {
public:
    GoBoardCPU();
    explicit GoBoardCPU(const std::string& state_str);
    explicit GoBoardCPU(const uint64_t* black_bits, const uint64_t* white_bits);
    
    StoneColor get_stone(int x, int y) const;
    void set_stone(int x, int y, StoneColor color);
    bool is_empty(int x, int y) const;
    bool is_black(int x, int y) const;
    bool is_white(int x, int y) const;
    
    int calculate_liberties(int x, int y) const;
    std::vector<std::pair<int, int>> get_legal_moves(StoneColor player) const;
    bool is_legal_move(int x, int y, StoneColor player) const;
    
    std::string to_string() const;
    static GoBoardCPU from_string(const std::string& state_str);
    
    const uint64_t* get_black_bits() const { return black_bits_; }
    const uint64_t* get_white_bits() const { return white_bits_; }
    
    bool operator==(const GoBoardCPU& other) const;
    bool operator!=(const GoBoardCPU& other) const;
    
    size_t hash() const;
    
    bool is_ko_move(int x, int y, StoneColor player) const;
    int get_ko_position() const { return ko_position_; }
    void set_ko_position(int pos) { ko_position_ = pos; }
    bool has_ko() const { return ko_position_ >= 0 && ko_position_ < TOTAL_POINTS; }
    
    MoveResult make_move(int x, int y, StoneColor player);
    // 快速落子：跳过is_legal_move前置检查（调用者必须保证落子合法）
    // 用于搜索树展开等高频场景，避免每节点对每个候选着法重复做完整合法性检查（flood fill）
    MoveResult make_move_fast(int x, int y, StoneColor player);
    bool undo_move();
    
    std::vector<int> get_captured_stones(int x, int y, StoneColor player) const;
    std::vector<std::pair<int, int>> get_group(int x, int y) const;

    // 中国规则面积计点：计算黑/白双方最终得分（含贴目）
    // 返回 {黑分, 白分}；领地通过泛填充(flood fill)判定空区域归属
    std::pair<float, float> compute_area_score(float komi = 7.5f) const;
    
    void encode_to_tensor(float* output, int channels = 17) const;
    static GoBoardCPU decode_from_tensor(const float* input, int channels = 17);
    
    StoneColor get_current_player() const { return current_player_; }
    void set_current_player(StoneColor player) { current_player_ = player; }
    
    int get_move_count() const { return move_count_; }
    void set_move_count(int count) { move_count_ = count; }
    
    int get_capture_count(StoneColor player) const;
    
    GoBoardCPU copy() const;
    
    std::vector<std::pair<int, int>> get_all_stones(StoneColor color) const;
    
    uint64_t get_zobrist_hash() const;
    void update_zobrist_hash(int x, int y, StoneColor old_color, StoneColor new_color);

    static int xy_to_index(int x, int y);
    static std::pair<int, int> index_to_xy(int idx);

private:
    uint64_t black_bits_[BITBOARD_WORDS];
    uint64_t white_bits_[BITBOARD_WORDS];
    
    int ko_position_;
    StoneColor current_player_;
    int move_count_;
    int capture_count_[2];
    uint64_t zobrist_hash_;
    
    struct HistoryEntry {
        int move;
        std::vector<int> captured;
        int prev_ko;
        StoneColor prev_player;
        int prev_move_count;
        uint64_t prev_hash;
    };
    std::vector<HistoryEntry> history_;

    void find_connected_group(int x, int y, std::vector<bool>& visited, 
                             std::vector<std::pair<int, int>>& group) const;
    
    void remove_group(int x, int y);
    bool would_be_suicide(int x, int y, StoneColor player) const;
    bool would_capture(int x, int y, StoneColor player) const;
};

#ifdef USE_CUDA

// CUDA内核函数前向声明
namespace cuda_kernels {
    void launch_batch_calculate_liberties(
        const uint64_t* black_bits, const uint64_t* white_bits,
        int* liberty_counts, size_t batch_size);
    
    void launch_batch_check_legal_moves(
        const uint64_t* black_bits, const uint64_t* white_bits,
        const int* moves, bool* legal_flags,
        int player, size_t batch_size, size_t moves_per_board);
    
    void launch_batch_generate_legal_moves(
        const uint64_t* black_bits, const uint64_t* white_bits,
        int* legal_moves, int player, size_t batch_size);
}

class GoBoardCUDA {
public:
    GoBoardCUDA();
    explicit GoBoardCUDA(size_t batch_size);
    ~GoBoardCUDA();
    
    GoBoardCUDA(const GoBoardCUDA&) = delete;
    GoBoardCUDA& operator=(const GoBoardCUDA&) = delete;
    
    GoBoardCUDA(GoBoardCUDA&& other) noexcept;
    GoBoardCUDA& operator=(GoBoardCUDA&& other) noexcept;
    
    void upload_batch(const std::vector<GoBoardCPU>& boards);
    void download_batch(std::vector<GoBoardCPU>& boards) const;
    
    size_t get_batch_size() const { return batch_size_; }
    size_t get_gpu_memory_usage() const;
    
    uint64_t* black_bits_device_ptr() { return black_bits_device_; }
    uint64_t* white_bits_device_ptr() { return white_bits_device_; }
    const uint64_t* black_bits_device_ptr() const { return black_bits_device_; }
    const uint64_t* white_bits_device_ptr() const { return white_bits_device_; }
    
    std::vector<int> batch_calculate_liberties() const;
    std::vector<int> batch_generate_legal_moves(StoneColor player) const;
    std::vector<bool> batch_check_legal_moves(const std::vector<int>& moves, 
                                               StoneColor player) const;

    /**
     * @brief 批量提交棋盘状态，指定CPU/GPU处理数量
     * @details 接受一批棋盘数据指针，根据指定的cpu_count和gpu_count将数据分片：
     *          - 前 cpu_count 个棋盘交由CPU线程池处理
     *          - 后续 gpu_count 个棋盘交由GPU批量处理
     *          如果cpu_count + gpu_count != boards.size()，剩余的棋盘自动按设备负载
     *          动态分配（优先分配到空闲设备）。
     * @param boards 棋盘数据指针列表（void* 以支持多种棋盘表示格式）
     * @param cpu_count 指定CPU处理的棋盘数量
     * @param gpu_count 指定GPU处理的棋盘数量
     * @return true=提交成功，false=提交失败（部分数据无效或队列已满）
     */
    bool batch_submit_boards(const std::vector<void*>& boards,
                             size_t cpu_count,
                             size_t gpu_count);

    /**
     * @brief 将一批棋盘动态分配到CPU/GPU处理
     * @details 自动检测各设备当前负载，将棋盘数据动态分配到负载较低的设备。
     *          如果GPU队列已满，自动将任务重定向到CPU线程池；
     *          如果CPU线程池已满，阻塞等待直到有可用槽位。
     * @param boards 棋盘数据指针列表
     * @return 实际分配到CPU的棋盘数量
     */
    size_t batch_submit_boards_auto(const std::vector<void*>& boards);
    
private:
    size_t batch_size_;
    uint64_t* black_bits_device_;
    uint64_t* white_bits_device_;
};

class GoBoardCUDATools {
public:
    static bool initialize_cuda();
    static std::string get_gpu_info();
    
    static double benchmark_liberties_calculation(size_t batch_size, int iterations = 100);
    static double benchmark_move_generation(size_t batch_size, int iterations = 100);
    
    static size_t calculate_optimal_batch_size(size_t available_gpu_memory);
    static std::vector<std::string> get_optimization_suggestions();
};

#endif

}
}

namespace std {
template<>
struct hash<spacego::go::GoBoardCPU> {
    size_t operator()(const spacego::go::GoBoardCPU& board) const {
        return board.hash();
    }
};
}
