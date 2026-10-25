// =============================================================================
// SpaceGo 围棋AI - 单元测试主入口
// Google Test 框架会自动生成 main() 函数（通过链接 gtest_main）
// =============================================================================
// 说明：
//   - 本文件仅包含全局测试环境配置
//   - 具体的测试用例分散在各个 test_*.cpp 文件中
//   - 使用 gtest_discover_tests 自动发现并注册测试
// =============================================================================

#include <gtest/gtest.h>
#include <iostream>

// =============================================================================
// 全局测试环境类 - 在所有测试之前/之后执行一次性的初始化与清理
// =============================================================================
class SpaceGoTestEnvironment : public ::testing::Environment {
public:
    // 所有测试开始前调用
    void SetUp() override {
        std::cout << "=== SpaceGo 测试框架初始化 ===" << std::endl;
        std::cout << "项目: SpaceGo 维空弈 围棋AI" << std::endl;
        std::cout << "测试框架: Google Test" << std::endl;
    }

    // 所有测试结束后调用
    void TearDown() override {
        std::cout << "=== SpaceGo 测试框架清理完成 ===" << std::endl;
    }
};

// =============================================================================
// main() - Google Test 入口
// 通过链接 gtest_main 库，GTEST 会自动提供 main() 函数
// 如果需要自定义 main()，可以取消下面注释并移除 gtest_main 链接
// =============================================================================
/*
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // 注册全局测试环境
    ::testing::AddGlobalTestEnvironment(new SpaceGoTestEnvironment());

    // 运行所有测试
    return RUN_ALL_TESTS();
}
*/