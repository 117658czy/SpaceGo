// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：game_info_panel.cpp
// 描述：对局信息面板实现
// 创建日期：2026-02-21
// 作者：SpaceGo Team
// ============================================================================

#include "components/game_info_panel.h"
#include "utils/font.h"
#pragma comment(lib, "gdiplus.lib")

namespace SpaceGo {
namespace GUI {

GameInfoPanel::GameInfoPanel(float px, float py, float w, float h)
    : x(px), y(py), width(w), height(h), visible(true),
      currentPlayer(Models::PieceColor::Black), moveCount(0),
      blackPlayerName(L"黑方"), whitePlayerName(L"白方"),
      blackTime(600), whiteTime(600) {
}

GameInfoPanel::~GameInfoPanel() {
}

void GameInfoPanel::UpdateGameInfo(Models::PieceColor player, int count) {
    currentPlayer = player;
    moveCount = count;
}

void GameInfoPanel::SetPlayerNames(const std::wstring& black, const std::wstring& white) {
    blackPlayerName = black;
    whitePlayerName = white;
}

void GameInfoPanel::SetTime(int blackSec, int whiteSec) {
    blackTime = blackSec;
    whiteTime = whiteSec;
}

void GameInfoPanel::SetBounds(float nx, float ny, float nw, float nh) {
    x = nx;
    y = ny;
    width = nw;
    height = nh;
}

void GameInfoPanel::SetVisible(bool v) {
    visible = v;
}

void GameInfoPanel::DrawPlayerInfo(Gdiplus::Graphics* g, float px, float py, float pw, float ph,
                                  const std::wstring& name, int time, bool isBlack, bool isCurrent) {
    // 背景 - 新配色系统
    Gdiplus::Color bgColor = isCurrent ? 
        Gdiplus::Color(255, 52, 73, 94) : Gdiplus::Color(255, 42, 63, 84);
    Gdiplus::SolidBrush bgBrush(bgColor);
    
    // 绘制圆角矩形背景
    float radius = 10.0f;
    Gdiplus::GraphicsPath path;
    path.AddArc(px, py, radius * 2, radius * 2, 180, 90);
    path.AddArc(px + pw - radius * 2, py, radius * 2, radius * 2, 270, 90);
    path.AddArc(px + pw - radius * 2, py + ph - radius * 2, radius * 2, radius * 2, 0, 90);
    path.AddArc(px, py + ph - radius * 2, radius * 2, radius * 2, 90, 90);
    path.CloseFigure();
    g->FillPath(&bgBrush, &path);
    
    // 边框 - 深空蓝特色
    if (isCurrent) {
        Gdiplus::Pen borderPen(Gdiplus::Color(255, 74, 144, 164), 2.0f);
        g->DrawPath(&borderPen, &path);
    } else {
        Gdiplus::Pen borderPen(Gdiplus::Color(255, 61, 86, 110), 1.0f);
        g->DrawPath(&borderPen, &path);
    }
    
    // 棋子指示器
    float stoneRadius = 14.0f;
    float stoneX = px + 22;
    float stoneY = py + ph / 2;
    
    Gdiplus::Color stoneColor = isBlack ? 
        Gdiplus::Color(255, 26, 26, 26) : Gdiplus::Color(255, 245, 240, 235);
    Gdiplus::SolidBrush stoneBrush(stoneColor);
    g->FillEllipse(&stoneBrush, stoneX - stoneRadius, stoneY - stoneRadius, stoneRadius * 2, stoneRadius * 2);
    
    Gdiplus::Pen stonePen(isBlack ? Gdiplus::Color(255, 10, 10, 10) : Gdiplus::Color(255, 180, 180, 185), 1.2f);
    g->DrawEllipse(&stonePen, stoneX - stoneRadius, stoneY - stoneRadius, stoneRadius * 2, stoneRadius * 2);
    
    // 玩家名称 - 使用FontManager缓存
    Gdiplus::Font* nameFont = FontManager::GetInstance().GetFont(L"Microsoft YaHei", 14.0f, true);
    if (nameFont) {
        Gdiplus::SolidBrush textBrush(Gdiplus::Color(255, 236, 240, 241));
        Gdiplus::StringFormat format;
        format.SetAlignment(Gdiplus::StringAlignmentNear);
        format.SetLineAlignment(Gdiplus::StringAlignmentCenter);
        
        Gdiplus::RectF nameRect(px + 52, py, pw - 122, ph);
        g->DrawString(name.c_str(), -1, nameFont, nameRect, &format, &textBrush);
        
        // 时间
        int minutes = time / 60;
        int seconds = time % 60;
        wchar_t timeStr[32];
        swprintf_s(timeStr, 32, L"%02d:%02d", minutes, seconds);
        
        Gdiplus::Font* timeFont = FontManager::GetInstance().GetFont(L"Microsoft YaHei", 16.0f, true);
        if (timeFont) {
            Gdiplus::SolidBrush timeBrush(isCurrent ? 
                Gdiplus::Color(255, 243, 156, 18) : Gdiplus::Color(255, 236, 240, 241));
            Gdiplus::RectF timeRect(px + pw - 70, py, 60, ph);
            format.SetAlignment(Gdiplus::StringAlignmentFar);
            g->DrawString(timeStr, -1, timeFont, timeRect, &format, &timeBrush);
        }
    }
}

void GameInfoPanel::Render(Gdiplus::Graphics* g) {
    if (!visible) return;
    
    // 面板背景 - 深空蓝渐变
    Gdiplus::LinearGradientBrush bgBrush(
        Gdiplus::Point((int)x, (int)y),
        Gdiplus::Point((int)x, (int)(y + height)),
        Gdiplus::Color(255, 42, 63, 84),
        Gdiplus::Color(255, 30, 41, 59)
    );
    g->FillRectangle(&bgBrush, x, y, width, height);
    
    // 边框
    Gdiplus::Pen borderPen(Gdiplus::Color(255, 61, 86, 110), 1.5f);
    g->DrawRectangle(&borderPen, x, y, width - 1, height - 1);
    
    // 标题 - 使用FontManager缓存
    Gdiplus::Font* titleFont = FontManager::GetInstance().GetFont(L"Microsoft YaHei", 16.0f, true);
    if (titleFont) {
        Gdiplus::SolidBrush titleBrush(Gdiplus::Color(255, 243, 156, 18));
        Gdiplus::StringFormat format;
        format.SetAlignment(Gdiplus::StringAlignmentCenter);
        format.SetLineAlignment(Gdiplus::StringAlignmentCenter);
        
        Gdiplus::RectF titleRect(x, y + 10, width, 30);
        g->DrawString(L"对局信息", -1, titleFont, titleRect, &format, &titleBrush);
        
        // 手数
        Gdiplus::Font* infoFont = FontManager::GetInstance().GetFont(L"Microsoft YaHei", 12.0f);
        if (infoFont) {
            Gdiplus::SolidBrush infoBrush(Gdiplus::Color(255, 149, 165, 166));
            wchar_t moveStr[64];
            swprintf_s(moveStr, 64, L"手数: %d", moveCount);
            
            Gdiplus::RectF moveRect(x, y + 50, width, 25);
            format.SetAlignment(Gdiplus::StringAlignmentCenter);
            g->DrawString(moveStr, -1, infoFont, moveRect, &format, &infoBrush);
        }
    }
    
    // 玩家信息
    float playerHeight = 60;
    float playerSpacing = 15;
    float startY = y + 90;
    
    // 黑方信息
    DrawPlayerInfo(g, x + 10, startY, width - 20, playerHeight,
                   blackPlayerName, blackTime, true, 
                   currentPlayer == Models::PieceColor::Black);
    
    // 白方信息
    DrawPlayerInfo(g, x + 10, startY + playerHeight + playerSpacing, width - 20, playerHeight,
                   whitePlayerName, whiteTime, false, 
                   currentPlayer == Models::PieceColor::White);
}

} // namespace GUI
} // namespace SpaceGo
