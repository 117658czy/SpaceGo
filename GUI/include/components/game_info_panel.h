// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：game_info_panel.h
// 描述：对局信息面板
// 创建日期：2026-02-21
// 作者：SpaceGo Team
// ============================================================================

#pragma once

#include <windows.h>
#include <objidl.h>
#include <gdiplus.h>
#include <string>
#include "models/game.h"
#include "models/piece.h"

namespace SpaceGo {
namespace GUI {

class GameInfoPanel {
public:
    float x, y, width, height;
    bool visible;
    Models::PieceColor currentPlayer;
    int moveCount;
    std::wstring blackPlayerName;
    std::wstring whitePlayerName;
    int blackTime;
    int whiteTime;

    GameInfoPanel(float px, float py, float w, float h);
    ~GameInfoPanel();

    // 更新游戏信息
    void UpdateGameInfo(Models::PieceColor player, int count);

    // 设置玩家名称
    void SetPlayerNames(const std::wstring& black, const std::wstring& white);

    // 设置时间
    void SetTime(int blackSec, int whiteSec);

    // 设置位置大小
    void SetBounds(float nx, float ny, float nw, float nh);

    // 设置可见性
    void SetVisible(bool v);

    // 渲染
    void Render(Gdiplus::Graphics* g);

private:
    void DrawPlayerInfo(Gdiplus::Graphics* g, float px, float py, float pw, float ph,
                       const std::wstring& name, int time, bool isBlack, bool isCurrent);
};

} // namespace GUI
} // namespace SpaceGo
