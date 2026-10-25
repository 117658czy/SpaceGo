// =============================================================================
// SpaceGo 围棋AI - 棋盘操作单元测试（GoBoardCPU）
// 测试内容：
//   1. 初始棋盘状态验证（19x19全空）
//   2. 落子后棋盘状态更新
//   3. 提子功能（一颗和多颗）
//   4. 劫争检测（立即提回判断）
//   5. Zobrist哈希一致性（相同局面哈希相同）
//   6. 棋盘复制正确性
//   7. 气计算（单块棋子的气数）
//   8. 边界坐标检查
// =============================================================================

#include <gtest/gtest.h>
#include "go/board/go_board_cuda.h"
#include "go/board/board_state.h"
#include <vector>
#include <string>

using namespace spacego::go;

// =============================================================================
// 测试夹具：GoBoardTest - 提供测试共用的初始化逻辑
// =============================================================================
class GoBoardTest : public ::testing::Test {
protected:
    // 每个测试用例开始前创建一个全新的空棋盘
    void SetUp() override {
        board = GoBoardCPU();
    }

    void TearDown() override {
        // 无需额外清理，GoBoardCPU 自动管理内存
    }

    // 辅助方法：确认某坐标上的棋子颜色
    void expect_empty(int x, int y) {
        EXPECT_TRUE(board.is_empty(x, y)) << "坐标 (" << x << "," << y << ") 应为空";
        EXPECT_EQ(board.get_stone(x, y), StoneColor::EMPTY);
    }

    void expect_black(int x, int y) {
        EXPECT_TRUE(board.is_black(x, y)) << "坐标 (" << x << "," << y << ") 应为黑子";
        EXPECT_EQ(board.get_stone(x, y), StoneColor::BLACK);
    }

    void expect_white(int x, int y) {
        EXPECT_TRUE(board.is_white(x, y)) << "坐标 (" << x << "," << y << ") 应为白子";
        EXPECT_EQ(board.get_stone(x, y), StoneColor::WHITE);
    }

    GoBoardCPU board;
};

// =============================================================================
// 测试1：初始棋盘状态验证 - 19x19全空
// =============================================================================
TEST_F(GoBoardTest, InitialBoardStateEmpty) {
    // 验证所有361个交叉点均为空
    for (int y = 0; y < GO_BOARD_SIZE; ++y) {
        for (int x = 0; x < GO_BOARD_SIZE; ++x) {
            expect_empty(x, y);
        }
    }
    // 验证初始玩家为黑方
    EXPECT_EQ(board.get_current_player(), StoneColor::BLACK);
    // 验证着法计数为0
    EXPECT_EQ(board.get_move_count(), 0);
    // 验证无劫争
    EXPECT_FALSE(board.has_ko());
    // 验证提子数为0
    EXPECT_EQ(board.get_capture_count(StoneColor::BLACK), 0);
    EXPECT_EQ(board.get_capture_count(StoneColor::WHITE), 0);
}

// =============================================================================
// 测试2：落子后棋盘状态更新
// =============================================================================
TEST_F(GoBoardTest, PlaceStoneUpdatesBoardState) {
    // 黑方落子于星位 (3,3)
    MoveResult result = board.make_move(3, 3, StoneColor::BLACK);
    EXPECT_TRUE(result.success);
    expect_black(3, 3);
    // 周围位置不应受影响
    expect_empty(2, 3);
    expect_empty(4, 3);
    expect_empty(3, 2);
    expect_empty(3, 4);
    // 着法计数应增加
    EXPECT_EQ(board.get_move_count(), 1);
    // 当前玩家应为白方
    EXPECT_EQ(board.get_current_player(), StoneColor::WHITE);

    // 白方落子于天元 (9,9)
    MoveResult result2 = board.make_move(9, 9, StoneColor::WHITE);
    EXPECT_TRUE(result2.success);
    expect_white(9, 9);
    expect_black(3, 3);  // 之前的黑子仍然存在
    EXPECT_EQ(board.get_move_count(), 2);

    // 多落几子验证
    board.make_move(15, 15, StoneColor::BLACK);
    board.make_move(15, 3, StoneColor::WHITE);
    expect_black(15, 15);
    expect_white(15, 3);
    EXPECT_EQ(board.get_move_count(), 4);
}

// =============================================================================
// 测试3：在已有棋子的位置落子应失败
// =============================================================================
TEST_F(GoBoardTest, CannotPlaceOnOccupiedPosition) {
    board.make_move(0, 0, StoneColor::BLACK);
    // 同一位置再次落子应失败
    MoveResult result = board.make_move(0, 0, StoneColor::WHITE);
    EXPECT_FALSE(result.success);
    // 棋子应保持不变
    expect_black(0, 0);
    EXPECT_EQ(board.get_move_count(), 1);
}

// =============================================================================
// 测试4：提子功能 - 提一颗子
// =============================================================================
TEST_F(GoBoardTest, CaptureSingleStone) {
    // 构造一个白子周围被黑子包围的局面
    // 在棋盘左上角 (1,0)、(0,1)、(1,1) 放黑子
    board.make_move(1, 0, StoneColor::BLACK);
    board.make_move(0, 1, StoneColor::BLACK);
    // 白子落于 (0,0)，被黑子部分包围（还有气）
    board.make_move(0, 0, StoneColor::WHITE);
    // 黑子落于 (1,1) 堵住最后一口气
    board.make_move(1, 1, StoneColor::BLACK);

    // 验证白子 (0,0) 已被提掉
    expect_empty(0, 0);
    // 验证黑子提子计数
    EXPECT_EQ(board.get_capture_count(StoneColor::BLACK), 1);

    // 验证通过 make_move 返回值
    // 重新构造局面来验证 capture 返回值
    GoBoardCPU board2;
    board2.make_move(1, 0, StoneColor::BLACK);  // 黑 (1,0)
    board2.make_move(0, 1, StoneColor::BLACK);  // 黑 (0,1)
    board2.make_move(0, 0, StoneColor::WHITE);  // 白 (0,0)
    MoveResult cap_result = board2.make_move(1, 1, StoneColor::BLACK);  // 黑 (1,1) 提吃
    EXPECT_TRUE(cap_result.success);
    EXPECT_EQ(cap_result.captured_positions.size(), 1);
    // 被提的坐标应为 (0,0)
    int captured_idx = GoBoardCPU::xy_to_index(0, 0);
    EXPECT_EQ(cap_result.captured_positions[0], captured_idx);
}

// =============================================================================
// 测试5：提子功能 - 提多颗子（一串棋子）
// =============================================================================
TEST_F(GoBoardTest, CaptureMultipleStones) {
    // 构造局面：白方两块棋分别被围
    // 坐标表示：第一块白棋 (0,1)-(1,0) 两子

    // 第1步：黑 (0,0) - 堵角
    board.make_move(0, 0, StoneColor::BLACK);
    // 第2步：白 (1,1) - 占天元方向（这是安全的）
    board.make_move(1, 1, StoneColor::WHITE);
    // 第3步：黑 (1,2) - 包围白
    board.make_move(1, 2, StoneColor::BLACK);
    // 第4步：白 (2,2) - 白回应
    board.make_move(2, 2, StoneColor::WHITE);
    // 第5步：黑 (0,2) - 包围
    board.make_move(0, 2, StoneColor::BLACK);
    // 第6步：白 (2,0) - 白回应
    board.make_move(2, 0, StoneColor::WHITE);

    // 现在验证 白 (1,1) 未被提（还有气）
    expect_white(1, 1);
}

// =============================================================================
// 测试6：劫争检测
// =============================================================================
TEST_F(GoBoardTest, KoDetection) {
    // 构造标准劫争局面
    // 棋盘左上角布局：
    //   0 1 2
    // 0 . B .
    // 1 B W B
    // 2 . B .
    // B可以吃W在(1,1)，然后W可以尝试立即提回(1,1)

    board.make_move(1, 0, StoneColor::BLACK);  // 黑 (1,0)
    board.make_move(2, 2, StoneColor::WHITE);  // 白 (2,2) 布置
    board.make_move(0, 1, StoneColor::BLACK);  // 黑 (0,1)
    board.make_move(0, 0, StoneColor::WHITE);  // 白 (0,0) 布置
    board.make_move(2, 1, StoneColor::BLACK);  // 黑 (2,1)
    board.make_move(1, 1, StoneColor::WHITE);  // 白 (1,1)

    // 白子 (1,1) 应该在棋盘上
    expect_white(1, 1);

    // 黑子吃白 (1,1)：放黑子在 (1,2)
    MoveResult cap_result = board.make_move(1, 2, StoneColor::BLACK);
    EXPECT_TRUE(cap_result.success);

    // 检查是否检测到劫争
    // 被提位置 (1,1) 应被标记为劫争位置
    int ko_pos = board.get_ko_position();
    int expected_ko = GoBoardCPU::xy_to_index(1, 1);
    EXPECT_EQ(ko_pos, expected_ko) << "劫争位置应为被提的白子位置 (1,1)";
    EXPECT_TRUE(board.has_ko()) << "应检测到劫争";

    // 验证白方不能立即提回
    bool is_ko = board.is_ko_move(1, 1, StoneColor::WHITE);
    EXPECT_TRUE(is_ko) << "白方立即提回应被标记为劫";
    MoveResult illegal_ko = board.make_move(1, 1, StoneColor::WHITE);
    EXPECT_FALSE(illegal_ko.success) << "劫争规则下白方不能立即提回";
}

// =============================================================================
// 测试7：Zobrist哈希一致性
// =============================================================================
TEST_F(GoBoardTest, ZobristHashConsistency) {
    // 两个独立构造的相同局面应有相同的哈希值
    GoBoardCPU board_a;
    board_a.make_move(3, 3, StoneColor::BLACK);
    board_a.make_move(15, 15, StoneColor::WHITE);
    board_a.make_move(3, 15, StoneColor::BLACK);
    board_a.make_move(15, 3, StoneColor::WHITE);
    uint64_t hash_a = board_a.get_zobrist_hash();

    GoBoardCPU board_b;
    board_b.make_move(3, 3, StoneColor::BLACK);
    board_b.make_move(15, 15, StoneColor::WHITE);
    board_b.make_move(3, 15, StoneColor::BLACK);
    board_b.make_move(15, 3, StoneColor::WHITE);
    uint64_t hash_b = board_b.get_zobrist_hash();

    EXPECT_EQ(hash_a, hash_b) << "相同局面应有相同的Zobrist哈希值";

    // 不同局面应有不同的哈希值
    GoBoardCPU board_c;
    board_c.make_move(3, 3, StoneColor::BLACK);
    board_c.make_move(15, 15, StoneColor::WHITE);
    uint64_t hash_c = board_c.get_zobrist_hash();

    EXPECT_NE(hash_a, hash_c) << "不同局面应有不同的Zobrist哈希值";
}

// =============================================================================
// 测试8：Zobrist哈希在悔棋后恢复
// =============================================================================
TEST_F(GoBoardTest, ZobristHashAfterUndo) {
    board.make_move(3, 3, StoneColor::BLACK);
    uint64_t hash_before = board.get_zobrist_hash();

    board.make_move(15, 15, StoneColor::WHITE);
    uint64_t hash_after_move = board.get_zobrist_hash();
    EXPECT_NE(hash_before, hash_after_move);

    board.undo_move();  // 撤销白棋着法
    uint64_t hash_after_undo = board.get_zobrist_hash();
    EXPECT_EQ(hash_before, hash_after_undo) << "悔棋后哈希值应恢复为之前的值";
}

// =============================================================================
// 测试9：棋盘复制正确性
// =============================================================================
TEST_F(GoBoardTest, BoardCopyCorrectness) {
    // 在棋盘上布置一些棋子
    board.make_move(3, 3, StoneColor::BLACK);
    board.make_move(15, 15, StoneColor::WHITE);
    board.make_move(9, 9, StoneColor::BLACK);

    // 复制棋盘
    GoBoardCPU copy = board.copy();

    // 验证复制后的棋盘与原始棋盘一致
    EXPECT_TRUE(board == copy) << "复制后的棋盘应与原始棋盘相等";
    EXPECT_EQ(board.hash(), copy.hash()) << "Hash值应相等";
    EXPECT_EQ(board.get_zobrist_hash(), copy.get_zobrist_hash());
    EXPECT_EQ(board.get_move_count(), copy.get_move_count());
    EXPECT_EQ(board.get_current_player(), copy.get_current_player());

    // 验证在复制棋盘上操作不会影响原棋盘
    copy.make_move(0, 0, StoneColor::WHITE);
    EXPECT_FALSE(board == copy) << "修改复制棋盘后不应影响原棋盘";
}

// =============================================================================
// 测试10：气计算 - 单块棋子的气数
// =============================================================================
TEST_F(GoBoardTest, LibertyCalculation) {
    // 在角部落一子，应有2气（角部只有两个相邻点）
    board.make_move(0, 0, StoneColor::BLACK);
    EXPECT_EQ(board.calculate_liberties(0, 0), 2) << "角部棋子应有2气";

    // 在边部落一子，应有3气
    board.make_move(0, 9, StoneColor::WHITE);
    EXPECT_EQ(board.calculate_liberties(0, 9), 3) << "边部棋子应有3气";

    // 在中央落一子，应有4气
    board.make_move(9, 9, StoneColor::BLACK);
    EXPECT_EQ(board.calculate_liberties(9, 9), 4) << "中央棋子应有4气";
}

// =============================================================================
// 测试11：气计算 - 多子连接后的气数变化
// =============================================================================
TEST_F(GoBoardTest, LibertyCalculationConnectedStones) {
    // 两个黑子连接，共6气（每个角部子2气，但共享2气）
    board.make_move(0, 0, StoneColor::BLACK);
    board.make_move(1, 0, StoneColor::WHITE);  // 白方隔开
    board.make_move(0, 1, StoneColor::BLACK);

    // (0,0) 和 (0,1) 是连接的，计算气数
    // (0,0) 的气: (1,0)白占据, (0,-1)边界, (0,1)同色 -> 只剩 (1,0)白占据,(0,-1)无
    // 实际 (0,0) 的气只有 (1,0)被白占了, 所以气为 (0,0)的邻居: (1,0)白, (-1,0)边界, (0,-1)边界, (0,1)黑
    // 所以(0,0)的气 = 0 (白和边界和同色都不算气)
    // (0,1) 的气: (1,1), (0,0)黑, (-1,1)边界, (0,2)
    // 连接后整体气: (1,1), (0,2) = 2气
    EXPECT_EQ(board.calculate_liberties(0, 0), 0) << "连接棋子 (0,0) 当前被白方堵";
}

// =============================================================================
// 测试12：气计算 - 一颗棋子在棋盘中央
// =============================================================================
TEST_F(GoBoardTest, LibertyCalculationCenter) {
    // 天元棋子应有4气
    board.make_move(9, 9, StoneColor::BLACK);
    EXPECT_EQ(board.calculate_liberties(9, 9), 4);

    // 堵住一气后应只剩3气
    board.make_move(10, 9, StoneColor::WHITE);
    EXPECT_EQ(board.calculate_liberties(9, 9), 3);
}

// =============================================================================
// 测试13：边界坐标检查
// =============================================================================
TEST_F(GoBoardTest, BoundaryCoordinateCheck) {
    // 负坐标应返回 BORDER
    EXPECT_EQ(board.get_stone(-1, 0), StoneColor::BORDER);
    EXPECT_EQ(board.get_stone(0, -1), StoneColor::BORDER);

    // 超出棋盘大小的坐标应返回 BORDER
    EXPECT_EQ(board.get_stone(GO_BOARD_SIZE, 0), StoneColor::BORDER);
    EXPECT_EQ(board.get_stone(0, GO_BOARD_SIZE), StoneColor::BORDER);

    // xy_to_index 对无效坐标应返回 -1
    EXPECT_EQ(GoBoardCPU::xy_to_index(-1, 0), -1);
    EXPECT_EQ(GoBoardCPU::xy_to_index(0, -1), -1);
    EXPECT_EQ(GoBoardCPU::xy_to_index(GO_BOARD_SIZE, 0), -1);
    EXPECT_EQ(GoBoardCPU::xy_to_index(0, GO_BOARD_SIZE), -1);

    // xy_to_index 对有效坐标应返回正确索引
    EXPECT_EQ(GoBoardCPU::xy_to_index(0, 0), 0);
    EXPECT_EQ(GoBoardCPU::xy_to_index(18, 18), TOTAL_POINTS - 1);
    EXPECT_EQ(GoBoardCPU::xy_to_index(9, 0), 9);
    EXPECT_EQ(GoBoardCPU::xy_to_index(0, 1), GO_BOARD_SIZE);
}

// =============================================================================
// 测试14：坐标转换往返测试
// =============================================================================
TEST_F(GoBoardTest, CoordinateRoundTrip) {
    for (int y = 0; y < GO_BOARD_SIZE; ++y) {
        for (int x = 0; x < GO_BOARD_SIZE; ++x) {
            int idx = GoBoardCPU::xy_to_index(x, y);
            auto [rx, ry] = GoBoardCPU::index_to_xy(idx);
            EXPECT_EQ(x, rx) << "坐标转换往返失败: x=" << x << " -> idx=" << idx << " -> rx=" << rx;
            EXPECT_EQ(y, ry) << "坐标转换往返失败: y=" << y << " -> idx=" << idx << " -> ry=" << ry;
        }
    }
}

// =============================================================================
// 测试15：undo_move 基本功能
// =============================================================================
TEST_F(GoBoardTest, UndoMoveBasic) {
    board.make_move(3, 3, StoneColor::BLACK);
    EXPECT_EQ(board.get_move_count(), 1);
    expect_black(3, 3);

    bool success = board.undo_move();
    EXPECT_TRUE(success);
    expect_empty(3, 3);  // 棋子应被移除
    EXPECT_EQ(board.get_current_player(), StoneColor::BLACK);  // 回到黑方
    EXPECT_EQ(board.get_move_count(), 0);
}

// =============================================================================
// 测试16：空棋盘 undo 应失败
// =============================================================================
TEST_F(GoBoardTest, UndoMoveOnEmptyBoardFails) {
    bool success = board.undo_move();
    EXPECT_FALSE(success) << "空棋盘无法悔棋";
}

// =============================================================================
// 测试17：to_string 和 from_string 往返测试
// =============================================================================
TEST_F(GoBoardTest, StringConversionRoundTrip) {
    board.make_move(0, 0, StoneColor::BLACK);
    board.make_move(18, 18, StoneColor::WHITE);
    board.make_move(9, 9, StoneColor::BLACK);

    std::string state_str = board.to_string();
    // 字符串长度应为 TOTAL_POINTS (361)
    EXPECT_EQ(state_str.size(), static_cast<size_t>(TOTAL_POINTS));

    GoBoardCPU restored = GoBoardCPU::from_string(state_str);
    EXPECT_TRUE(board == restored) << "字符串转换往返应保持一致性";
}

// =============================================================================
// 测试18：自杀着法应被拒绝
// =============================================================================
TEST_F(GoBoardTest, SuicideMoveRejected) {
    // 构造自杀局面：白子在角部被三面包围
    // (0,0) 位置的白子，周围 (0,1)空、(1,0)空、其余边界
    // 黑子占据 (0,1) 和 (1,0)
    board.make_move(0, 1, StoneColor::BLACK);  // 黑占一边
    board.make_move(4, 4, StoneColor::WHITE);  // 白走别处
    board.make_move(1, 0, StoneColor::BLACK);  // 黑占另一边

    // 现在白子落于 (0,0) 是自杀（无气且不能提子）
    bool is_legal = board.is_legal_move(0, 0, StoneColor::WHITE);
    EXPECT_FALSE(is_legal) << "角部自杀着法不应合法";
}

// =============================================================================
// 测试19：get_group 返回正确的连通组
// =============================================================================
TEST_F(GoBoardTest, GetGroupCorrectness) {
    // 从 (0,0) 开始放两个连接的黑子
    board.make_move(0, 0, StoneColor::BLACK);
    board.make_move(1, 1, StoneColor::WHITE);  // 白方回应
    board.make_move(0, 1, StoneColor::BLACK);
    board.make_move(2, 2, StoneColor::WHITE);  // 白方回应

    // 获取 (0,0) 的连通组
    auto group = board.get_group(0, 0);
    EXPECT_EQ(group.size(), 2);  // 应包含 (0,0) 和 (0,1)
    // 验证组内包含预期坐标
    bool has_00 = false, has_01 = false;
    for (const auto& [gx, gy] : group) {
        if (gx == 0 && gy == 0) has_00 = true;
        if (gx == 0 && gy == 1) has_01 = true;
    }
    EXPECT_TRUE(has_00);
    EXPECT_TRUE(has_01);
}

// =============================================================================
// 测试20：get_all_stones 返回正确棋子列表
// =============================================================================
TEST_F(GoBoardTest, GetAllStones) {
    board.make_move(0, 0, StoneColor::BLACK);
    board.make_move(1, 1, StoneColor::WHITE);
    board.make_move(18, 18, StoneColor::BLACK);
    board.make_move(17, 17, StoneColor::WHITE);

    auto black_stones = board.get_all_stones(StoneColor::BLACK);
    EXPECT_EQ(black_stones.size(), 2);

    auto white_stones = board.get_all_stones(StoneColor::WHITE);
    EXPECT_EQ(white_stones.size(), 2);
}