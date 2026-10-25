// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：board.h
// 描述：围棋棋盘组件
// 创建日期：2026-02-20
// 作者：SpaceGo Team
// ============================================================================

#pragma once
#include <windows.h>
#include <gdiplus.h>
#include <vector>
#include <functional>
#include "utils/color.h"
#include "utils/math.h"

namespace SpaceGo {
namespace GUI {

// 棋盘位置回调
using BoardPositionCallback = std::function<void(int x, int y)>;

// 棋子类型
enum class StoneType {
    None,
    Black,
    White
};

class GoBoard {
public:
    float x, y, boardSize;
    int gridSize;
    std::vector<std::vector<StoneType>> stones;
    BoardPositionCallback onPositionClick;
    float hoverX, hoverY;
    bool showHover;
    bool visible;

    GoBoard(float px, float py, float size, int grid = 19);
    ~GoBoard();

    // 设置位置和大小
    void SetBounds(float nx, float ny, float size);

    // 放置棋子
    void PlaceStone(int x, int y, StoneType type);

    // 清除棋子
    void ClearStone(int x, int y);

    // 清空棋盘
    void ClearBoard();

    // 鼠标事件
    bool OnMouseMove(float mx, float my);
    bool OnMouseDown(float mx, float my);

    // 渲染
    void Render(Gdiplus::Graphics* g);

private:
    // 坐标转换
    void GridToScreen(int gx, int gy, float& sx, float& sy) const;
    bool ScreenToGrid(float sx, float sy, int& gx, int& gy) const;

    // 绘制棋盘
    void DrawBoard(Gdiplus::Graphics* g);
    void DrawGridLines(Gdiplus::Graphics* g);
    void DrawStarPoints(Gdiplus::Graphics* g);
    void DrawStones(Gdiplus::Graphics* g);
    void DrawStone(Gdiplus::Graphics* g, float cx, float cy, float radius, StoneType type);
    void DrawHoverIndicator(Gdiplus::Graphics* g);
};

} // namespace GUI
} // namespace SpaceGo
