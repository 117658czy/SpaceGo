// =============================================================================
// SpaceGo 围棋AI - 搜索算法单元测试
// 测试内容：
//   1. 合法着法生成（空棋盘、有棋子、边界情况）
//   2. CA-MOE 前向传播（简单局面的价值评估）
//   3. HMGS-DP 搜索基础功能（简单局面的最佳着法）
//   4. 时间限制下搜索超时行为
// =============================================================================

#include <gtest/gtest.h>
#include "go/board/go_board_cuda.h"
#include "go/search/ca_moe.h"
#include "go/search/hmgs_dp_optimized.h"
#include "go/engine/go_engine.h"
#include <vector>
#include <string>
#include <chrono>
#include <thread>

using namespace spacego::go;

// =============================================================================
// 测试夹具：SearchTest - 提供搜索测试的共用设施
// =============================================================================
class SearchTest : public ::testing::Test {
protected:
    void SetUp() override {
        board = std::make_unique<GoBoardCPU>();
    }

    void TearDown() override {
        board.reset();
    }

    // 辅助方法：在棋盘上布置标准开局局面（中国流布局的简化版）
    void setup_opening_board() {
        // 黑方标准星位开局
        board->make_move(3, 3, StoneColor::BLACK);     // 左上星位
        board->make_move(3, 15, StoneColor::WHITE);    // 右上星位
        board->make_move(15, 3, StoneColor::BLACK);    // 左下星位
        board->make_move(15, 15, StoneColor::WHITE);   // 右下星位
    }

    std::unique_ptr<GoBoardCPU> board;
};

// =============================================================================
// 测试夹具：CAMOETest - CA-MOE 组件测试
// =============================================================================
class CAMOETest : public ::testing::Test {
protected:
    void SetUp() override {
        ca_moe = std::make_shared<CAMOE>();
        board = std::make_unique<GoBoardCPU>();
    }

    void TearDown() override {
        ca_moe.reset();
        board.reset();
    }

    std::shared_ptr<CAMOE> ca_moe;
    std::unique_ptr<GoBoardCPU> board;
};

// =============================================================================
// 测试1：空棋盘合法着法生成
// =============================================================================
TEST_F(SearchTest, LegalMovesGenerationEmptyBoard) {
    // 空棋盘黑方所有361个点都是合法着法
    auto legal_moves = board->get_legal_moves(StoneColor::BLACK);
    EXPECT_EQ(legal_moves.size(), static_cast<size_t>(TOTAL_POINTS))
        << "空棋盘应有361个合法着法";

    // 验证所有坐标都在有效范围内
    for (const auto& [x, y] : legal_moves) {
        EXPECT_GE(x, 0);
        EXPECT_LT(x, GO_BOARD_SIZE);
        EXPECT_GE(y, 0);
        EXPECT_LT(y, GO_BOARD_SIZE);
    }
}

// =============================================================================
// 测试2：有棋子后合法着法减少
// =============================================================================
TEST_F(SearchTest, LegalMovesWithStones) {
    // 黑方落子于(3,3)
    board->make_move(3, 3, StoneColor::BLACK);
    // 白方合法着法应减少（(3,3)被占）
    auto white_moves = board->get_legal_moves(StoneColor::WHITE);
    EXPECT_EQ(white_moves.size(), static_cast<size_t>(TOTAL_POINTS - 1))
        << "落一子后白方应有360个合法着法";

    // (3,3)不应在白方的合法着法中
    bool found_33 = false;
    for (const auto& [x, y] : white_moves) {
        if (x == 3 && y == 3) {
            found_33 = true;
            break;
        }
    }
    EXPECT_FALSE(found_33) << "(3,3)已被占据，不应是合法着法";
}

// =============================================================================
// 测试3：边界着法确认
// =============================================================================
TEST_F(SearchTest, LegalMovesAtBoundaries) {
    // 空棋盘所有角部和边部着法应合法
    EXPECT_TRUE(board->is_legal_move(0, 0, StoneColor::BLACK));
    EXPECT_TRUE(board->is_legal_move(0, 18, StoneColor::BLACK));
    EXPECT_TRUE(board->is_legal_move(18, 0, StoneColor::BLACK));
    EXPECT_TRUE(board->is_legal_move(18, 18, StoneColor::BLACK));

    // 超出边界的着法不合法
    EXPECT_FALSE(board->is_legal_move(-1, 0, StoneColor::BLACK));
    EXPECT_FALSE(board->is_legal_move(0, -1, StoneColor::BLACK));
    EXPECT_FALSE(board->is_legal_move(GO_BOARD_SIZE, 0, StoneColor::BLACK));
    EXPECT_FALSE(board->is_legal_move(0, GO_BOARD_SIZE, StoneColor::BLACK));
}

// =============================================================================
// 测试4：自杀着法不应在合法着法中
// =============================================================================
TEST_F(SearchTest, SuicideMovesNotInLegalMoves) {
    // 构造角部自杀局面：(0,0)放白子，黑子占(0,1)和(1,0)
    board->make_move(0, 1, StoneColor::BLACK);
    board->make_move(4, 4, StoneColor::WHITE);
    board->make_move(1, 0, StoneColor::BLACK);

    // 白方合法着法中不应包含(0,0)
    auto white_moves = board->get_legal_moves(StoneColor::WHITE);
    bool found_00 = false;
    for (const auto& [x, y] : white_moves) {
        if (x == 0 && y == 0) {
            found_00 = true;
            break;
        }
    }
    EXPECT_FALSE(found_00) << "角部自杀着法 (0,0) 不应在合法着法列表中";
}

// =============================================================================
// 测试5：CA-MOE forward_simple 前向传播
// =============================================================================
TEST_F(CAMOETest, CAMOEForwardSimpleEmptyBoard) {
    // 空棋盘上运行CA-MOE前向传播
    LinkNeuronOutput output = ca_moe->forward_simple(*board);

    // 验证输出不为空（价值估计在 [-1, 1] 范围内）
    EXPECT_GE(output.value, -1.0f);
    EXPECT_LE(output.value, 1.0f);

    // 验证先验概率分布维度
    EXPECT_EQ(output.prior_probs.size(), 8) << "先验概率分布应为8个方向";

    // 验证不确定性在 [0, 1] 范围内
    EXPECT_GE(output.uncertainty, 0.0f);
    EXPECT_LE(output.uncertainty, 1.0f);
}

// =============================================================================
// 测试6：CA-MOE forward_simple 在不同局面下
// =============================================================================
TEST_F(CAMOETest, CAMOEForwardSimpleWithStones) {
    // 布置简单棋子后再评估
    board->make_move(3, 3, StoneColor::BLACK);
    board->make_move(15, 15, StoneColor::WHITE);

    LinkNeuronOutput output1 = ca_moe->forward_simple(*board);
    EXPECT_GE(output1.value, -1.0f);
    EXPECT_LE(output1.value, 1.0f);

    // 添加更多棋子后再评估
    board->make_move(3, 15, StoneColor::BLACK);
    board->make_move(15, 3, StoneColor::WHITE);

    LinkNeuronOutput output2 = ca_moe->forward_simple(*board);
    EXPECT_GE(output2.value, -1.0f);
    EXPECT_LE(output2.value, 1.0f);
}

// =============================================================================
// 测试7：CA-MOE forward 完整前向传播
// =============================================================================
TEST_F(CAMOETest, CAMOEForwardFullEmptyBoard) {
    LinkNeuronOutput output = ca_moe->forward(*board);

    EXPECT_GE(output.value, -1.0f);
    EXPECT_LE(output.value, 1.0f);
    EXPECT_EQ(output.prior_probs.size(), 8);
    EXPECT_GE(output.uncertainty, 0.0f);
    EXPECT_LE(output.uncertainty, 1.0f);
}

// =============================================================================
// 测试8：GoEngine 搜索基本功能（genmove）
// =============================================================================
TEST_F(SearchTest, GoEngineGenmoveEmptyBoard) {
    // 创建引擎实例并初始化
    spacego::go::engine::GoEngine engine;
    engine.initialize();

    // 空棋盘上生成着法
    auto result = engine.genmove(StoneColor::BLACK);
    // 搜索应返回有效坐标
    EXPECT_GE(result.move_x, 0);
    EXPECT_LT(result.move_x, GO_BOARD_SIZE);
    EXPECT_GE(result.move_y, 0);
    EXPECT_LT(result.move_y, GO_BOARD_SIZE);

    // 移动字符串应不为空
    EXPECT_FALSE(result.move_string.empty());
}

// =============================================================================
// 测试11：GoEngine 有棋子时的搜索
// =============================================================================
TEST_F(SearchTest, GoEngineGenmoveWithStones) {
    spacego::go::engine::GoEngine engine;
    engine.initialize();

    // 布置几个棋子
    engine.play_move(3, 3, StoneColor::BLACK);
    engine.play_move(15, 15, StoneColor::WHITE);

    // 黑方生成着法
    auto result = engine.genmove(StoneColor::BLACK);
    EXPECT_GE(result.move_x, 0);
    EXPECT_LT(result.move_x, GO_BOARD_SIZE);
    EXPECT_GE(result.move_y, 0);
    EXPECT_LT(result.move_y, GO_BOARD_SIZE);

    // 不能走在已被占据的位置
    EXPECT_FALSE(result.move_x == 3 && result.move_y == 3)
        << "搜索不应返回已被占据的位置";
    EXPECT_FALSE(result.move_x == 15 && result.move_y == 15)
        << "搜索不应返回已被占据的位置";
}

// =============================================================================
// 测试12：HMGSDP_Optimized 独立搜索
// =============================================================================
TEST_F(SearchTest, HMGSDPOptimizedSearchEmptyBoard) {
    // 创建HMGS-DP优化搜索实例
    spacego::go::search::HMGSDP_Optimized hmgs;
    hmgs.initialize("");

    // 空棋盘状态字符串（361个'E'）
    std::string empty_state(TOTAL_POINTS, 'E');
    // 设置当前玩家（使用kanji棋字符串格式？不，使用我们自己的格式）
    // 直接使用GoBoardCPU::to_string()生成的字符串
    GoBoardCPU empty_board;
    std::string state = empty_board.to_string();

    // 运行搜索（减少迭代次数以快速完成测试）
    auto best_move = hmgs.search(state, 100, 1.0);
    EXPECT_GE(best_move.first, 0);
    EXPECT_LT(best_move.first, GO_BOARD_SIZE);
    EXPECT_GE(best_move.second, 0);
    EXPECT_LT(best_move.second, GO_BOARD_SIZE);
}

// =============================================================================
// 测试13：HMGSDP_Optimized 性能统计
// =============================================================================
TEST_F(SearchTest, HMGSDPOptimizedStatistics) {
    spacego::go::search::HMGSDP_Optimized hmgs;
    hmgs.initialize("");

    GoBoardCPU empty_board;
    std::string state = empty_board.to_string();

    // 执行搜索
    hmgs.search(state, 200, 1.0);

    // 获取性能统计
    auto stats = hmgs.get_search_statistics();
    EXPECT_GT(stats.total_simulations, 0) << "应进行了模拟";
    EXPECT_GT(stats.search_time_ms, 0.0) << "搜索应消耗时间";
}

// =============================================================================
// 测试14：时间限制下搜索超时行为
// =============================================================================
TEST_F(SearchTest, SearchTimeoutBehavior) {
    spacego::go::engine::GoEngine engine;
    engine.initialize();

    // 设置很短的超时时间 (100ms)
    engine.set_time_limit(100);

    auto start = std::chrono::high_resolution_clock::now();
    auto result = engine.genmove(StoneColor::BLACK);
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // 验证在合理时间内返回（不应超过超时时间太多）
    EXPECT_LT(elapsed.count(), 10000) << "搜索应在合理时间内完成（<10秒）";

    // 应返回有效着法
    EXPECT_GE(result.move_x, 0);
    EXPECT_LT(result.move_x, GO_BOARD_SIZE);
}

// =============================================================================
// 测试15：GoEngine 合法着法查询
// =============================================================================
TEST_F(SearchTest, GoEngineLegalMovesQuery) {
    spacego::go::engine::GoEngine engine;
    engine.initialize();

    // 空棋盘查询合法着法
    auto legal = engine.get_legal_moves(StoneColor::BLACK);
    EXPECT_EQ(legal.size(), static_cast<size_t>(TOTAL_POINTS));

    // 验证 is_legal_move
    EXPECT_TRUE(engine.is_legal_move(0, 0, StoneColor::BLACK));
    EXPECT_FALSE(engine.is_legal_move(-1, 0, StoneColor::BLACK));
    EXPECT_FALSE(engine.is_legal_move(GO_BOARD_SIZE, 0, StoneColor::BLACK));
}

// =============================================================================
// 测试16：GoEngine 局面分析
// =============================================================================
TEST_F(SearchTest, GoEngineAnalysis) {
    spacego::go::engine::GoEngine engine;
    engine.initialize();

    // 布置简单的局面
    engine.play_move(3, 3, StoneColor::BLACK);
    engine.play_move(15, 15, StoneColor::WHITE);

    // 分析局面
    auto result = engine.analyze(50);
    EXPECT_GE(result.move_x, 0);
    EXPECT_LT(result.move_x, GO_BOARD_SIZE);
    EXPECT_GE(result.win_probability, 0.0f);
    EXPECT_LE(result.win_probability, 1.0f);
}

// =============================================================================
// 测试17：GoEngine 彻底初始化与关闭
// =============================================================================
TEST_F(SearchTest, GoEngineInitializeShutdown) {
    spacego::go::engine::GoEngine engine;

    // 未初始化状态
    EXPECT_FALSE(engine.is_initialized());

    // 初始化
    bool init_ok = engine.initialize();
    EXPECT_TRUE(init_ok) << "引擎初始化应成功";
    EXPECT_TRUE(engine.is_initialized());

    // 关闭引擎
    engine.shutdown();
}

// =============================================================================
// 测试18：GoEngine 配置管理
// =============================================================================
TEST_F(SearchTest, GoEngineConfiguration) {
    spacego::go::engine::GoEngine engine;
    engine.initialize();

    // 默认配置
    const auto& config = engine.get_config();
    EXPECT_EQ(config.board_size, 19);

    // 修改配置
    engine.set_max_iterations(500);
    engine.set_exploration_weight(2.0f);
    engine.set_time_limit(1000);

    // 验证配置已更新
    EXPECT_EQ(engine.get_config().max_iterations, 500);
    EXPECT_FLOAT_EQ(engine.get_config().exploration_weight, 2.0f);
    EXPECT_EQ(engine.get_config().time_limit_ms, 1000);
}