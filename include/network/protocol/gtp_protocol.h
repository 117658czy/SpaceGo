#pragma once

#include "go/engine/go_engine.h"
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <memory>
#include <istream>
#include <ostream>
#include <tuple>

namespace spacego {
namespace network {
namespace protocol {

// GTP命令结果
struct GTPResult {
    bool success;
    std::string output;
    std::string error_msg;
    
    GTPResult() : success(true) {}
    
    static GTPResult ok(const std::string& output = "") {
        GTPResult r;
        r.success = true;
        r.output = output;
        return r;
    }
    
    static GTPResult error(const std::string& err) {
        GTPResult r;
        r.success = false;
        r.error_msg = err;
        return r;
    }
};

// GTP命令处理器类型
using GTPCommandHandler = std::function<GTPResult(const std::vector<std::string>&)>;

// GTP协议处理器
class GTPProtocol {
public:
    // 构造函数
    GTPProtocol();
    explicit GTPProtocol(std::shared_ptr<go::engine::GoEngine> engine);
    ~GTPProtocol();
    
    // 设置引擎
    void set_engine(std::shared_ptr<go::engine::GoEngine> engine);
    
    // 注册命令处理器
    void register_command(const std::string& cmd, GTPCommandHandler handler);
    
    // 处理输入命令
    std::string process_command(const std::string& input);
    
    // 运行GTP主循环
    void run(std::istream& in, std::ostream& out);
    
    // 运行GTP主循环（标准输入输出）
    void run();
    
    // 停止运行
    void stop();
    
    // 检查是否正在运行
    bool is_running() const { return running_; }
    
    // 获取支持的命令列表
    std::vector<std::string> get_supported_commands() const;
    
    // 获取引擎
    std::shared_ptr<go::engine::GoEngine> get_engine() const { return engine_; }
    
private:
    std::shared_ptr<go::engine::GoEngine> engine_;
    std::map<std::string, GTPCommandHandler> command_handlers_;
    std::atomic<bool> running_;
    
    // 初始化标准GTP命令
    void initialize_standard_commands();
    
    // 解析命令，返回tuple
    std::tuple<int, std::string, std::vector<std::string>> parse_command(
        const std::string& input) const;
    
    // 格式化响应
    std::string format_response(int id, const GTPResult& result) const;
    
    // 标准GTP命令处理器
    GTPResult cmd_protocol_version(const std::vector<std::string>& args);
    GTPResult cmd_name(const std::vector<std::string>& args);
    GTPResult cmd_version(const std::vector<std::string>& args);
    GTPResult cmd_known_command(const std::vector<std::string>& args);
    GTPResult cmd_list_commands(const std::vector<std::string>& args);
    GTPResult cmd_quit(const std::vector<std::string>& args);
    
    // 棋盘设置命令
    GTPResult cmd_boardsize(const std::vector<std::string>& args);
    GTPResult cmd_clear_board(const std::vector<std::string>& args);
    GTPResult cmd_komi(const std::vector<std::string>& args);
    GTPResult cmd_play(const std::vector<std::string>& args);
    GTPResult cmd_genmove(const std::vector<std::string>& args);
    GTPResult cmd_pass(const std::vector<std::string>& args);
    GTPResult cmd_undo(const std::vector<std::string>& args);
    
    // 信息查询命令
    GTPResult cmd_showboard(const std::vector<std::string>& args);
    GTPResult cmd_legal_moves(const std::vector<std::string>& args);
    GTPResult cmd_is_legal(const std::vector<std::string>& args);
    GTPResult cmd_final_score(const std::vector<std::string>& args);
    GTPResult cmd_final_status_list(const std::vector<std::string>& args);
    
    // 分析命令
    GTPResult cmd_analyze(const std::vector<std::string>& args);
    GTPResult cmd_all_move_values(const std::vector<std::string>& args);
    
    // 时间控制命令
    GTPResult cmd_time_left(const std::vector<std::string>& args);
    GTPResult cmd_time_settings(const std::vector<std::string>& args);
    
    // 调试命令
    GTPResult cmd_debug(const std::vector<std::string>& args);
    GTPResult cmd_echo(const std::vector<std::string>& args);
    
    // SGF命令
    GTPResult cmd_loadsgf(const std::vector<std::string>& args);
    GTPResult cmd_savesgf(const std::vector<std::string>& args);
    
    // 引擎特定命令
    GTPResult cmd_set_max_iterations(const std::vector<std::string>& args);
    GTPResult cmd_set_exploration(const std::vector<std::string>& args);
    GTPResult cmd_get_stats(const std::vector<std::string>& args);
    GTPResult cmd_reset_stats(const std::vector<std::string>& args);
    
    // 辅助方法
    bool parse_vertex(const std::string& str, int& x, int& y) const;
    bool parse_color(const std::string& str, go::StoneColor& color) const;
    std::string vertex_to_string(int x, int y) const;
};

// 创建GTP协议处理器
std::unique_ptr<GTPProtocol> create_gtp_protocol();
std::unique_ptr<GTPProtocol> create_gtp_protocol(
    std::shared_ptr<go::engine::GoEngine> engine);

}  // namespace protocol
}  // namespace network
}  // namespace spacego
