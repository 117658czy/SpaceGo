// =============================================================================
// SpaceGo 围棋AI - GTP协议单元测试
// 测试内容：
//   1. GTP命令解析（ID提取、命令名、参数列表）
//   2. 标准命令响应格式化
//   3. protocol_version/name/version 命令
//   4. boardsize/clear_board 命令
//   5. play/genmove 命令
//   6. undo 命令
//   7. 非法命令的错误处理
// =============================================================================

#include <gtest/gtest.h>
#include "network/protocol/gtp_protocol.h"
#include "go/engine/go_engine.h"
#include <sstream>
#include <string>
#include <vector>

using namespace spacego::network::protocol;

// =============================================================================
// 测试夹具：GTPProtocolTest - 提供GTP测试的共用设施
// =============================================================================
class GTPProtocolTest : public ::testing::Test {
protected:
    void SetUp() override {
        gtp = std::make_unique<GTPProtocol>();
    }

    void TearDown() override {
        gtp.reset();
    }

    // 辅助方法：验证响应格式
    // GTP成功响应格式: =[id] [output]\n\n
    // GTP错误响应格式: ?[id] [error]\n\n
    static bool is_success_response(const std::string& response) {
        return response.find('=') == 0;
    }

    static bool is_error_response(const std::string& response) {
        return response.find('?') == 0;
    }

    // 辅助方法：提取响应内容（去掉前缀和尾部的\n\n）
    static std::string extract_content(const std::string& response) {
        std::string content = response;
        // 移除尾部的 \n\n
        while (!content.empty() && content.back() == '\n') {
            content.pop_back();
        }
        // 移除前缀 = 或 ? 及可能的ID
        size_t space_pos = content.find(' ');
        if (space_pos != std::string::npos) {
            return content.substr(space_pos + 1);
        }
        return "";
    }

    std::unique_ptr<GTPProtocol> gtp;
};

// =============================================================================
// 测试1：命令解析 - 无ID的基本命令
// =============================================================================
TEST_F(GTPProtocolTest, ParseCommandNoId) {
    // 注册一个返回参数列表的回显命令
    gtp->register_command("echo_test", [](const std::vector<std::string>& args) -> GTPResult {
        std::string result;
        for (size_t i = 0; i < args.size(); ++i) {
            if (i > 0) result += ",";
            result += args[i];
        }
        return GTPResult::ok(result);
    });

    // 测试无ID、无参数
    std::string response = gtp->process_command("echo_test");
    EXPECT_TRUE(is_success_response(response));

    // 测试无ID、有一个参数
    response = gtp->process_command("echo_test hello");
    EXPECT_TRUE(is_success_response(response));
    EXPECT_NE(response.find("hello"), std::string::npos);

    // 测试无ID、有多个参数
    response = gtp->process_command("echo_test arg1 arg2 arg3");
    EXPECT_TRUE(is_success_response(response));
    EXPECT_NE(response.find("arg1"), std::string::npos);
    EXPECT_NE(response.find("arg2"), std::string::npos);
    EXPECT_NE(response.find("arg3"), std::string::npos);
}

// =============================================================================
// 测试2：命令解析 - 带ID的命令
// =============================================================================
TEST_F(GTPProtocolTest, ParseCommandWithId) {
    // 注册一个返回ID的回显命令
    gtp->register_command("id_echo", [](const std::vector<std::string>& args) -> GTPResult {
        std::string result = "args_count=" + std::to_string(args.size());
        return GTPResult::ok(result);
    });

    // 带ID 1的命令
    std::string response = gtp->process_command("1 id_echo");
    EXPECT_TRUE(is_success_response(response));
    EXPECT_NE(response.find("=1"), std::string::npos)
        << "响应应包含命令ID";

    // 带ID 42的命令和参数
    response = gtp->process_command("42 id_echo test");
    EXPECT_TRUE(is_success_response(response));
    EXPECT_NE(response.find("=42"), std::string::npos)
        << "响应应包含命令ID 42";
    EXPECT_NE(response.find("args_count=1"), std::string::npos);

    // 大ID值
    response = gtp->process_command("999 id_echo a b c");
    EXPECT_TRUE(is_success_response(response));
    EXPECT_NE(response.find("=999"), std::string::npos);
}

// =============================================================================
// 测试3：命令解析 - 空输入和注释
// =============================================================================
TEST_F(GTPProtocolTest, ParseCommandEdgeCases) {
    // 空命令应返回错误
    std::string response = gtp->process_command("");
    EXPECT_TRUE(is_error_response(response));

    // 仅空白字符的命令
    response = gtp->process_command("   ");
    EXPECT_TRUE(is_error_response(response));
}

// =============================================================================
// 测试4：响应格式化 - 成功响应
// =============================================================================
TEST_F(GTPProtocolTest, FormatSuccessResponse) {
    // 注册一个固定输出命令
    gtp->register_command("test_ok", [](const std::vector<std::string>&) -> GTPResult {
        return GTPResult::ok("success_value");
    });

    // 无ID响应
    std::string response = gtp->process_command("test_ok");
    EXPECT_TRUE(is_success_response(response));
    EXPECT_EQ(response.find("= "), 0) << "成功响应以 '= ' 开头";
    EXPECT_NE(response.find("success_value"), std::string::npos);

    // 带ID响应
    response = gtp->process_command("5 test_ok");
    EXPECT_TRUE(is_success_response(response));
    EXPECT_NE(response.find("=5"), std::string::npos);

    // 空输出
    gtp->register_command("test_ok_empty", [](const std::vector<std::string>&) -> GTPResult {
        return GTPResult::ok("");
    });
    response = gtp->process_command("test_ok_empty");
    EXPECT_TRUE(is_success_response(response));
}

// =============================================================================
// 测试5：响应格式化 - 错误响应
// =============================================================================
TEST_F(GTPProtocolTest, FormatErrorResponse) {
    // 注册一个固定错误命令
    gtp->register_command("test_err", [](const std::vector<std::string>&) -> GTPResult {
        return GTPResult::error("something went wrong");
    });

    // 无ID错误响应
    std::string response = gtp->process_command("test_err");
    EXPECT_TRUE(is_error_response(response));
    EXPECT_EQ(response.find("? "), 0) << "错误响应以 '? ' 开头";
    EXPECT_NE(response.find("something went wrong"), std::string::npos);

    // 带ID错误响应
    response = gtp->process_command("7 test_err");
    EXPECT_TRUE(is_error_response(response));
    EXPECT_NE(response.find("?7"), std::string::npos);
}

// =============================================================================
// 测试6：unknown command 错误
// =============================================================================
TEST_F(GTPProtocolTest, UnknownCommand) {
    std::string response = gtp->process_command("nonexistent_command");
    EXPECT_TRUE(is_error_response(response));
    EXPECT_NE(response.find("unknown command"), std::string::npos);
}

// =============================================================================
// 测试7：protocol_version 命令
// =============================================================================
TEST_F(GTPProtocolTest, ProtocolVersionCommand) {
    std::string response = gtp->process_command("protocol_version");
    EXPECT_TRUE(is_success_response(response));
    EXPECT_NE(response.find("2"), std::string::npos) << "GTP协议版本应为2";
}

// =============================================================================
// 测试8：name 命令
// =============================================================================
TEST_F(GTPProtocolTest, NameCommand) {
    std::string response = gtp->process_command("name");
    EXPECT_TRUE(is_success_response(response));
    EXPECT_NE(response.find("SpaceGo"), std::string::npos)
        << "引擎名称应为 SpaceGo";
}

// =============================================================================
// 测试9：version 命令
// =============================================================================
TEST_F(GTPProtocolTest, VersionCommand) {
    std::string response = gtp->process_command("version");
    EXPECT_TRUE(is_success_response(response));
    EXPECT_NE(response.find("1.0.0"), std::string::npos)
        << "引擎版本应为 1.0.0";
}

// =============================================================================
// 测试10：known_command 命令
// =============================================================================
TEST_F(GTPProtocolTest, KnownCommand) {
    // 已知命令应返回 true
    std::string response = gtp->process_command("known_command protocol_version");
    EXPECT_TRUE(is_success_response(response));
    EXPECT_NE(response.find("true"), std::string::npos);

    // 未知命令应返回 false
    response = gtp->process_command("known_command made_up_cmd");
    EXPECT_TRUE(is_success_response(response));
    EXPECT_NE(response.find("false"), std::string::npos);
}

// =============================================================================
// 测试11：list_commands 命令
// =============================================================================
TEST_F(GTPProtocolTest, ListCommands) {
    std::string response = gtp->process_command("list_commands");
    EXPECT_TRUE(is_success_response(response));

    // 验证包含基本命令
    EXPECT_NE(response.find("protocol_version"), std::string::npos);
    EXPECT_NE(response.find("name"), std::string::npos);
    EXPECT_NE(response.find("version"), std::string::npos);
    EXPECT_NE(response.find("boardsize"), std::string::npos);
    EXPECT_NE(response.find("clear_board"), std::string::npos);
    EXPECT_NE(response.find("play"), std::string::npos);
    EXPECT_NE(response.find("genmove"), std::string::npos);
}

// =============================================================================
// 测试12：boardsize 命令
// =============================================================================
TEST_F(GTPProtocolTest, BoardSizeCommand) {
    // 设置19路棋盘
    std::string response = gtp->process_command("boardsize 19");
    EXPECT_TRUE(is_success_response(response));

    // 非法棋盘大小
    response = gtp->process_command("boardsize 100");
    EXPECT_TRUE(is_error_response(response));

    // 缺少参数
    response = gtp->process_command("boardsize");
    EXPECT_TRUE(is_error_response(response));
}

// =============================================================================
// 测试13：clear_board 命令
// =============================================================================
TEST_F(GTPProtocolTest, ClearBoardCommand) {
    // 先设置棋盘大小
    gtp->process_command("boardsize 19");

    // 清空棋盘
    std::string response = gtp->process_command("clear_board");
    EXPECT_TRUE(is_success_response(response));
}

// =============================================================================
// 测试14：play 命令
// =============================================================================
TEST_F(GTPProtocolTest, PlayCommand) {
    // 确保棋盘已设置
    gtp->process_command("boardsize 19");
    gtp->process_command("clear_board");

    // 黑方下天元
    std::string response = gtp->process_command("play black k10");
    EXPECT_TRUE(is_success_response(response));

    // 白方下星位（使用字母坐标 C4 = 左上星位）
    response = gtp->process_command("play white d16");
    EXPECT_TRUE(is_success_response(response));

    // 非法着法（重复落子）
    response = gtp->process_command("play black k10");
    EXPECT_TRUE(is_error_response(response));

    // pass 着法
    response = gtp->process_command("play black pass");
    EXPECT_TRUE(is_success_response(response));
}

// =============================================================================
// 测试15：play 命令参数不足
// =============================================================================
TEST_F(GTPProtocolTest, PlayCommandMissingArgs) {
    gtp->process_command("boardsize 19");
    gtp->process_command("clear_board");

    std::string response = gtp->process_command("play");
    EXPECT_TRUE(is_error_response(response));

    response = gtp->process_command("play black");
    EXPECT_TRUE(is_error_response(response));
}

// =============================================================================
// 测试16：genmove 命令
// =============================================================================
TEST_F(GTPProtocolTest, GenmoveCommand) {
    gtp->process_command("boardsize 19");
    gtp->process_command("clear_board");

    // 生成黑方着法
    std::string response = gtp->process_command("genmove black");
    EXPECT_TRUE(is_success_response(response));

    // 提取着法坐标（响应格式为 "= coord"）
    std::string content = extract_content(response);
    EXPECT_FALSE(content.empty()) << "应返回着法坐标";

    // 生成白方着法
    response = gtp->process_command("genmove white");
    EXPECT_TRUE(is_success_response(response));
    content = extract_content(response);
    EXPECT_FALSE(content.empty());
}

// =============================================================================
// 测试17：undo 命令
// =============================================================================
TEST_F(GTPProtocolTest, UndoCommand) {
    gtp->process_command("boardsize 19");
    gtp->process_command("clear_board");

    // 下几步棋
    gtp->process_command("play black k10");
    gtp->process_command("play white d16");

    // 悔棋
    std::string response = gtp->process_command("undo");
    EXPECT_TRUE(is_success_response(response));

    // 再次悔棋
    response = gtp->process_command("undo");
    EXPECT_TRUE(is_success_response(response));

    // 第三步悔棋应失败（棋盘已空）
    response = gtp->process_command("undo");
    EXPECT_TRUE(is_error_response(response));
    EXPECT_NE(response.find("cannot undo"), std::string::npos);
}

// =============================================================================
// 测试18：quit 命令
// =============================================================================
TEST_F(GTPProtocolTest, QuitCommand) {
    std::string response = gtp->process_command("quit");
    EXPECT_TRUE(is_success_response(response));
    EXPECT_FALSE(gtp->is_running());
}

// =============================================================================
// 测试19：echo 命令
// =============================================================================
TEST_F(GTPProtocolTest, EchoCommand) {
    std::string response = gtp->process_command("echo hello world");
    EXPECT_TRUE(is_success_response(response));
    EXPECT_NE(response.find("hello world"), std::string::npos);

    response = gtp->process_command("echo");
    EXPECT_TRUE(is_success_response(response));
}

// =============================================================================
// 测试20：错误命令的异常处理
// =============================================================================
TEST_F(GTPProtocolTest, ExceptionHandling) {
    // 注册一个会抛出异常的命令处理器
    gtp->register_command("bad_handler", [](const std::vector<std::string>&) -> GTPResult {
        throw std::runtime_error("test exception occurred");
        return GTPResult::ok("never reached");
    });

    std::string response = gtp->process_command("bad_handler");
    EXPECT_TRUE(is_error_response(response));
    EXPECT_NE(response.find("test exception occurred"), std::string::npos);
}

// =============================================================================
// 测试21：自定义命令注册与调用
// =============================================================================
TEST_F(GTPProtocolTest, CustomCommandRegistration) {
    // 注册自定义命令
    gtp->register_command("spacego_custom", [](const std::vector<std::string>& args) -> GTPResult {
        if (args.empty()) {
            return GTPResult::error("need argument");
        }
        return GTPResult::ok("custom: " + args[0]);
    });

    // 调用自定义命令
    std::string response = gtp->process_command("spacego_custom test");
    EXPECT_TRUE(is_success_response(response));
    EXPECT_NE(response.find("custom: test"), std::string::npos);

    // 缺少参数
    response = gtp->process_command("spacego_custom");
    EXPECT_TRUE(is_error_response(response));
}

// =============================================================================
// 测试22：GTP命令名大小写不敏感
// =============================================================================
TEST_F(GTPProtocolTest, CommandCaseInsensitive) {
    // GTP协议规定命令名大小写不敏感
    std::string response = gtp->process_command("NAME");
    EXPECT_TRUE(is_success_response(response));

    response = gtp->process_command("Protocol_Version");
    EXPECT_TRUE(is_success_response(response));

    response = gtp->process_command("QuIt");
    EXPECT_TRUE(is_success_response(response));
}

// =============================================================================
// 测试23：get_supported_commands 返回正确命令列表
// =============================================================================
TEST_F(GTPProtocolTest, SupportedCommandsList) {
    auto commands = gtp->get_supported_commands();
    EXPECT_GT(commands.size(), 0) << "应有已注册的命令";

    // 检查包含基本命令
    bool has_version = false, has_play = false, has_genmove = false;
    for (const auto& cmd : commands) {
        if (cmd == "protocol_version") has_version = true;
        if (cmd == "play") has_play = true;
        if (cmd == "genmove") has_genmove = true;
    }
    EXPECT_TRUE(has_version);
    EXPECT_TRUE(has_play);
    EXPECT_TRUE(has_genmove);
}

// =============================================================================
// 测试24：factory 函数 create_gtp_protocol
// =============================================================================
TEST_F(GTPProtocolTest, FactoryCreation) {
    auto gtp1 = create_gtp_protocol();
    EXPECT_NE(gtp1, nullptr);

    auto engine = std::make_shared<spacego::go::engine::GoEngine>();
    auto gtp2 = create_gtp_protocol(engine);
    EXPECT_NE(gtp2, nullptr);
}

// =============================================================================
// 测试25：showboard 命令
// =============================================================================
TEST_F(GTPProtocolTest, ShowBoardCommand) {
    gtp->process_command("boardsize 19");
    gtp->process_command("clear_board");

    std::string response = gtp->process_command("showboard");
    EXPECT_TRUE(is_success_response(response));
}

// =============================================================================
// 测试26：GTPResult 静态工厂方法
// =============================================================================
TEST_F(GTPProtocolTest, GTPResultFactory) {
    // ok() 工厂方法
    GTPResult ok_result = GTPResult::ok("test output");
    EXPECT_TRUE(ok_result.success);
    EXPECT_EQ(ok_result.output, "test output");
    EXPECT_TRUE(ok_result.error_msg.empty());

    // error() 工厂方法
    GTPResult err_result = GTPResult::error("test error");
    EXPECT_FALSE(err_result.success);
    EXPECT_EQ(err_result.error_msg, "test error");
    EXPECT_TRUE(err_result.output.empty());
}