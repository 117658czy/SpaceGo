// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：game.h
// 描述：游戏模型
// 创建日期：2026-02-21
// 作者：SpaceGo Team
// ============================================================================

#pragma once
#include <vector>
#include <string>
#include "models/board.h"
#include "models/piece.h"

namespace SpaceGo {
namespace GUI {
namespace Models {

// 游戏状态
enum class GameState {
    Idle,
    Playing,
    Paused,
    Finished
};

// 游戏模式
enum class GameMode {
    PvP,
    PvAI,
    AIvAI
};

// 落子记录
struct MoveRecord {
    int x;
    int y;
    PieceColor color;
    int moveNumber;
};

class Game {
public:
    GameState state;
    GameMode mode;
    Board board;
    PieceColor currentPlayer;
    int moveCount;
    std::vector<MoveRecord> moveHistory;

    Game();
    ~Game();

    // 开始新游戏
    void StartNewGame(GameMode gm = GameMode::PvP, int boardSize = 19);

    // 落子
    bool MakeMove(int x, int y);

    // 悔棋
    bool UndoMove();

    // 弃权
    void Pass();

    // 暂停游戏
    void Pause();

    // 恢复游戏
    void Resume();

    // 结束游戏
    void Finish();

    // 重置游戏
    void Reset();

    // 获取当前玩家
    PieceColor GetCurrentPlayer() const;

    // 切换玩家
    void SwitchPlayer();

    // 获取游戏状态
    GameState GetState() const;

    // 获取落子历史
    const std::vector<MoveRecord>& GetMoveHistory() const;
};

} // namespace Models
} // namespace GUI
} // namespace SpaceGo
