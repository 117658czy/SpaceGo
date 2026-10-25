#include "network/protocol/gtp_protocol.h"
#include "framework/logger.h"
#include <sstream>
#include <algorithm>
#include <cctype>
#include <iostream>

namespace spacego {
namespace network {
namespace protocol {

// 构造函数
GTPProtocol::GTPProtocol()
    : engine_(std::make_shared<go::engine::GoEngine>())
    , running_(false) {
    initialize_standard_commands();
}

GTPProtocol::GTPProtocol(std::shared_ptr<go::engine::GoEngine> engine)
    : engine_(std::move(engine))
    , running_(false) {
    initialize_standard_commands();
}

// 析构函数
GTPProtocol::~GTPProtocol() {
    stop();
}

// 设置引擎
void GTPProtocol::set_engine(std::shared_ptr<go::engine::GoEngine> engine) {
    engine_ = std::move(engine);
}

// 注册命令处理器
void GTPProtocol::register_command(const std::string& cmd, GTPCommandHandler handler) {
    command_handlers_[cmd] = std::move(handler);
}

// 处理输入命令
std::string GTPProtocol::process_command(const std::string& input) {
    auto [id, command, args] = parse_command(input);
    
    auto it = command_handlers_.find(command);
    if (it == command_handlers_.end()) {
        return format_response(id, GTPResult::error("unknown command: " + command));
    }
    
    try {
        GTPResult result = it->second(args);
        return format_response(id, result);
    } catch (const std::exception& e) {
        return format_response(id, GTPResult::error(e.what()));
    }
}

// 运行GTP主循环
void GTPProtocol::run(std::istream& in, std::ostream& out) {
    running_ = true;
    std::string line;
    
    while (running_ && std::getline(in, line)) {
        // 跳过空行和注释
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // 移除行尾空白
        while (!line.empty() && std::isspace(line.back())) {
            line.pop_back();
        }
        
        if (line.empty()) {
            continue;
        }
        
        std::string response = process_command(line);
        // format_response 已含结尾双换行("\n\n")，直接输出即可，
        // 再加 endl 会多出一个空行，导致下游 GTP 客户端解析错位
        out << response;
        out.flush();
    }
}

// 运行GTP主循环（标准输入输出）
void GTPProtocol::run() {
    run(std::cin, std::cout);
}

// 停止运行
void GTPProtocol::stop() {
    running_ = false;
}

// 获取支持的命令列表
std::vector<std::string> GTPProtocol::get_supported_commands() const {
    std::vector<std::string> commands;
    for (const auto& [cmd, _] : command_handlers_) {
        commands.push_back(cmd);
    }
    std::sort(commands.begin(), commands.end());
    return commands;
}

// 初始化标准GTP命令
void GTPProtocol::initialize_standard_commands() {
    // 基本命令
    command_handlers_["protocol_version"] = [this](const auto& args) {
        return cmd_protocol_version(args);
    };
    command_handlers_["name"] = [this](const auto& args) {
        return cmd_name(args);
    };
    command_handlers_["version"] = [this](const auto& args) {
        return cmd_version(args);
    };
    command_handlers_["known_command"] = [this](const auto& args) {
        return cmd_known_command(args);
    };
    command_handlers_["list_commands"] = [this](const auto& args) {
        return cmd_list_commands(args);
    };
    command_handlers_["quit"] = [this](const auto& args) {
        return cmd_quit(args);
    };
    
    // 棋盘设置命令
    command_handlers_["boardsize"] = [this](const auto& args) {
        return cmd_boardsize(args);
    };
    command_handlers_["clear_board"] = [this](const auto& args) {
        return cmd_clear_board(args);
    };
    command_handlers_["komi"] = [this](const auto& args) {
        return cmd_komi(args);
    };
    command_handlers_["play"] = [this](const auto& args) {
        return cmd_play(args);
    };
    command_handlers_["genmove"] = [this](const auto& args) {
        return cmd_genmove(args);
    };
    command_handlers_["pass"] = [this](const auto& args) {
        return cmd_pass(args);
    };
    command_handlers_["undo"] = [this](const auto& args) {
        return cmd_undo(args);
    };
    
    // 信息查询命令
    command_handlers_["showboard"] = [this](const auto& args) {
        return cmd_showboard(args);
    };
    command_handlers_["legal_moves"] = [this](const auto& args) {
        return cmd_legal_moves(args);
    };
    command_handlers_["is_legal"] = [this](const auto& args) {
        return cmd_is_legal(args);
    };
    command_handlers_["final_score"] = [this](const auto& args) {
        return cmd_final_score(args);
    };
    command_handlers_["final_status_list"] = [this](const auto& args) {
        return cmd_final_status_list(args);
    };
    
    // 分析命令
    command_handlers_["analyze"] = [this](const auto& args) {
        return cmd_analyze(args);
    };
    command_handlers_["all_move_values"] = [this](const auto& args) {
        return cmd_all_move_values(args);
    };
    
    // 时间控制命令
    command_handlers_["time_left"] = [this](const auto& args) {
        return cmd_time_left(args);
    };
    command_handlers_["time_settings"] = [this](const auto& args) {
        return cmd_time_settings(args);
    };
    
    // 调试命令
    command_handlers_["debug"] = [this](const auto& args) {
        return cmd_debug(args);
    };
    command_handlers_["echo"] = [this](const auto& args) {
        return cmd_echo(args);
    };
    
    // SGF命令
    command_handlers_["loadsgf"] = [this](const auto& args) {
        return cmd_loadsgf(args);
    };
    command_handlers_["savesgf"] = [this](const auto& args) {
        return cmd_savesgf(args);
    };
    
    // 引擎特定命令
    command_handlers_["set_max_iterations"] = [this](const auto& args) {
        return cmd_set_max_iterations(args);
    };
    command_handlers_["set_exploration"] = [this](const auto& args) {
        return cmd_set_exploration(args);
    };
    command_handlers_["get_stats"] = [this](const auto& args) {
        return cmd_get_stats(args);
    };
    command_handlers_["reset_stats"] = [this](const auto& args) {
        return cmd_reset_stats(args);
    };
}

// 解析命令
std::tuple<int, std::string, std::vector<std::string>> GTPProtocol::parse_command(
    const std::string& input) const {
    
    std::istringstream iss(input);
    int id = -1;
    std::string command;
    std::vector<std::string> args;
    
    // 尝试读取ID
    std::string token;
    iss >> token;
    
    if (token.empty()) {
        return {-1, "", {}};
    }
    
    // 检查第一个token是否是数字
    bool is_number = !token.empty() && std::isdigit(token[0]);
    for (char c : token) {
        if (!std::isdigit(c)) {
            is_number = false;
            break;
        }
    }
    
    if (is_number) {
        id = std::stoi(token);
        iss >> command;
    } else {
        command = token;
    }
    
    // 读取参数
    while (iss >> token) {
        args.push_back(token);
    }
    
    // 转换命令为小写
    std::transform(command.begin(), command.end(), command.begin(), ::tolower);
    
    return {id, command, args};
}

// 格式化响应
std::string GTPProtocol::format_response(int id, const GTPResult& result) const {
    std::ostringstream oss;
    
    if (result.success) {
        if (id >= 0) {
            oss << "=" << id;
        } else {
            oss << "=";
        }
        
        if (!result.output.empty()) {
            oss << " " << result.output;
        }
    } else {
        if (id >= 0) {
            oss << "?" << id;
        } else {
            oss << "?";
        }
        
        oss << " " << result.error_msg;
    }
    
    // GTP 协议要求两个换行符表示响应结束
    oss << "\n\n";
    return oss.str();
}

// ==================== 标准GTP命令实现 ====================

GTPResult GTPProtocol::cmd_protocol_version(const std::vector<std::string>& args) {
    return GTPResult::ok("2");
}

GTPResult GTPProtocol::cmd_name(const std::vector<std::string>& args) {
    return GTPResult::ok(go::engine::GoEngine::get_engine_name());
}

GTPResult GTPProtocol::cmd_version(const std::vector<std::string>& args) {
    return GTPResult::ok(go::engine::GoEngine::get_engine_version());
}

GTPResult GTPProtocol::cmd_known_command(const std::vector<std::string>& args) {
    if (args.empty()) {
        return GTPResult::error("missing argument");
    }
    
    std::string cmd = args[0];
    std::transform(cmd.begin(), cmd.end(), cmd.begin(), ::tolower);
    
    bool known = command_handlers_.find(cmd) != command_handlers_.end();
    return GTPResult::ok(known ? "true" : "false");
}

GTPResult GTPProtocol::cmd_list_commands(const std::vector<std::string>& args) {
    auto commands = get_supported_commands();
    std::ostringstream oss;
    for (size_t i = 0; i < commands.size(); ++i) {
        if (i > 0) oss << "\n";
        oss << commands[i];
    }
    return GTPResult::ok(oss.str());
}

GTPResult GTPProtocol::cmd_quit(const std::vector<std::string>& args) {
    running_ = false;
    return GTPResult::ok();
}

GTPResult GTPProtocol::cmd_boardsize(const std::vector<std::string>& args) {
    if (args.empty()) {
        return GTPResult::error("missing argument");
    }
    
    try {
        int size = std::stoi(args[0]);
        if (!engine_->set_board_size(size)) {
            return GTPResult::error("unacceptable size");
        }
        return GTPResult::ok();
    } catch (...) {
        return GTPResult::error("invalid size");
    }
}

GTPResult GTPProtocol::cmd_clear_board(const std::vector<std::string>& args) {
    engine_->clear_board();
    return GTPResult::ok();
}

GTPResult GTPProtocol::cmd_komi(const std::vector<std::string>& args) {
    // 贴目设置（简化实现）
    return GTPResult::ok();
}

GTPResult GTPProtocol::cmd_play(const std::vector<std::string>& args) {
    if (args.size() < 2) {
        return GTPResult::error("missing argument");
    }
    
    go::StoneColor color;
    if (!parse_color(args[0], color)) {
        return GTPResult::error("invalid color");
    }
    
    int x, y;
    if (args[1] == "pass" || args[1] == "PASS") {
        engine_->set_current_player(color == go::StoneColor::BLACK ? 
                                     go::StoneColor::WHITE : go::StoneColor::BLACK);
        return GTPResult::ok();
    }
    
    if (!parse_vertex(args[1], x, y)) {
        return GTPResult::error("invalid vertex");
    }
    
    if (!engine_->play_move(x, y, color)) {
        return GTPResult::error("illegal move");
    }
    
    return GTPResult::ok();
}

GTPResult GTPProtocol::cmd_genmove(const std::vector<std::string>& args) {
    go::StoneColor color = engine_->get_current_player();
    
    if (!args.empty()) {
        if (!parse_color(args[0], color)) {
            return GTPResult::error("invalid color");
        }
    }
    
    auto result = engine_->genmove(color);
    return GTPResult::ok(result.move_string);
}

GTPResult GTPProtocol::cmd_pass(const std::vector<std::string>& args) {
    go::StoneColor current = engine_->get_current_player();
    engine_->set_current_player(current == go::StoneColor::BLACK ? 
                                 go::StoneColor::WHITE : go::StoneColor::BLACK);
    return GTPResult::ok();
}

GTPResult GTPProtocol::cmd_undo(const std::vector<std::string>& args) {
    if (!engine_->undo_move()) {
        return GTPResult::error("cannot undo");
    }
    return GTPResult::ok();
}

GTPResult GTPProtocol::cmd_showboard(const std::vector<std::string>& args) {
    std::string board_str = engine_->get_board_string();
    return GTPResult::ok("\n" + board_str);
}

GTPResult GTPProtocol::cmd_legal_moves(const std::vector<std::string>& args) {
    go::StoneColor color = engine_->get_current_player();
    
    if (!args.empty()) {
        if (!parse_color(args[0], color)) {
            return GTPResult::error("invalid color");
        }
    }
    
    auto moves = engine_->get_legal_moves(color);
    std::ostringstream oss;
    for (size_t i = 0; i < moves.size(); ++i) {
        if (i > 0) oss << " ";
        oss << vertex_to_string(moves[i].first, moves[i].second);
    }
    return GTPResult::ok(oss.str());
}

GTPResult GTPProtocol::cmd_is_legal(const std::vector<std::string>& args) {
    if (args.size() < 2) {
        return GTPResult::error("missing argument");
    }
    
    go::StoneColor color;
    if (!parse_color(args[0], color)) {
        return GTPResult::error("invalid color");
    }
    
    int x, y;
    if (!parse_vertex(args[1], x, y)) {
        return GTPResult::error("invalid vertex");
    }
    
    bool legal = engine_->is_legal_move(x, y, color);
    return GTPResult::ok(legal ? "true" : "false");
}

GTPResult GTPProtocol::cmd_final_score(const std::vector<std::string>& args) {
    auto [black, white] = engine_->count_territory();
    
    // 简化计分
    double score = black - white - 6.5;  // 贴目6.5
    
    std::ostringstream oss;
    if (score > 0) {
        oss << "B+" << score;
    } else {
        oss << "W+" << (-score);
    }
    
    return GTPResult::ok(oss.str());
}

GTPResult GTPProtocol::cmd_final_status_list(const std::vector<std::string>& args) {
    // 简化实现
    return GTPResult::ok();
}

GTPResult GTPProtocol::cmd_analyze(const std::vector<std::string>& args) {
    int iterations = 0;
    if (!args.empty()) {
        try {
            iterations = std::stoi(args[0]);
        } catch (...) {}
    }
    
    auto result = engine_->analyze(iterations);
    
    std::ostringstream oss;
    oss << "move: " << result.move_string 
        << " winrate: " << (result.win_probability * 100) << "%"
        << " visits: " << result.visits;
    
    return GTPResult::ok(oss.str());
}

GTPResult GTPProtocol::cmd_all_move_values(const std::vector<std::string>& args) {
    auto probs = engine_->get_move_probabilities();
    
    std::ostringstream oss;
    for (size_t i = 0; i < probs.size() && i < 10; ++i) {
        if (i > 0) oss << "\n";
        oss << probs[i].first << " " << (probs[i].second * 100) << "%";
    }
    
    return GTPResult::ok(oss.str());
}

GTPResult GTPProtocol::cmd_time_left(const std::vector<std::string>& args) {
    // 时间控制（简化实现）
    return GTPResult::ok();
}

GTPResult GTPProtocol::cmd_time_settings(const std::vector<std::string>& args) {
    // 时间设置（简化实现）
    return GTPResult::ok();
}

GTPResult GTPProtocol::cmd_debug(const std::vector<std::string>& args) {
    std::string info = go::engine::GoEngine::get_engine_info();
    return GTPResult::ok(info);
}

GTPResult GTPProtocol::cmd_echo(const std::vector<std::string>& args) {
    std::ostringstream oss;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0) oss << " ";
        oss << args[i];
    }
    return GTPResult::ok(oss.str());
}

GTPResult GTPProtocol::cmd_loadsgf(const std::vector<std::string>& args) {
    if (args.empty()) {
        return GTPResult::error("missing argument");
    }
    
    // TODO: 实现SGF加载
    return GTPResult::error("not implemented");
}

GTPResult GTPProtocol::cmd_savesgf(const std::vector<std::string>& args) {
    std::string sgf = engine_->to_sgf();
    return GTPResult::ok(sgf);
}

GTPResult GTPProtocol::cmd_set_max_iterations(const std::vector<std::string>& args) {
    if (args.empty()) {
        return GTPResult::error("missing argument");
    }
    
    try {
        int iterations = std::stoi(args[0]);
        engine_->set_max_iterations(iterations);
        return GTPResult::ok();
    } catch (...) {
        return GTPResult::error("invalid value");
    }
}

GTPResult GTPProtocol::cmd_set_exploration(const std::vector<std::string>& args) {
    if (args.empty()) {
        return GTPResult::error("missing argument");
    }
    
    try {
        float weight = std::stof(args[0]);
        engine_->set_exploration_weight(weight);
        return GTPResult::ok();
    } catch (...) {
        return GTPResult::error("invalid value");
    }
}

GTPResult GTPProtocol::cmd_get_stats(const std::vector<std::string>& args) {
    auto stats = engine_->get_stats();
    
    std::ostringstream oss;
    oss << "moves: " << stats.total_moves
        << " searches: " << stats.total_searches
        << " time: " << stats.total_time_ms << "ms"
        << " avg: " << stats.average_time_per_move_ms << "ms";
    
    return GTPResult::ok(oss.str());
}

GTPResult GTPProtocol::cmd_reset_stats(const std::vector<std::string>& args) {
    engine_->reset_stats();
    return GTPResult::ok();
}

// 辅助方法
bool GTPProtocol::parse_vertex(const std::string& str, int& x, int& y) const {
    if (str.empty() || str == "pass") {
        return false;
    }
    
    char col = std::toupper(str[0]);
    if (col < 'A' || col > 'T' || col == 'I') {
        return false;
    }
    
    y = col - 'A';
    if (col > 'I') {
        y--;
    }
    
    try {
        int row = std::stoi(str.substr(1));
        int board_size = engine_->get_board_size();
        x = board_size - row;
        
        if (x < 0 || x >= board_size || y < 0 || y >= board_size) {
            return false;
        }
        
        return true;
    } catch (...) {
        return false;
    }
}

bool GTPProtocol::parse_color(const std::string& str, go::StoneColor& color) const {
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == "black" || lower == "b") {
        color = go::StoneColor::BLACK;
        return true;
    }
    if (lower == "white" || lower == "w") {
        color = go::StoneColor::WHITE;
        return true;
    }
    
    return false;
}

std::string GTPProtocol::vertex_to_string(int x, int y) const {
    return go::engine::GoEngine::move_to_string(x, y);
}

// 工厂函数
std::unique_ptr<GTPProtocol> create_gtp_protocol() {
    return std::make_unique<GTPProtocol>();
}

std::unique_ptr<GTPProtocol> create_gtp_protocol(
    std::shared_ptr<go::engine::GoEngine> engine) {
    return std::make_unique<GTPProtocol>(engine);
}

}  // namespace protocol
}  // namespace network
}  // namespace spacego
