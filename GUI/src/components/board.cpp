// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：board.cpp
// 描述：围棋棋盘组件实现
// 创建日期：2026-02-20
// 作者：SpaceGo Team
// ============================================================================

#include "components/board.h"
#pragma comment(lib, "gdiplus.lib")

namespace SpaceGo {
namespace GUI {

GoBoard::GoBoard(float px, float py, float size, int grid)
    : x(px), y(py), boardSize(size), gridSize(grid), 
      hoverX(0), hoverY(0), showHover(false), visible(true) {
    // 初始化棋盘
    stones.resize(gridSize);
    for (int i = 0; i < gridSize; ++i) {
        stones[i].resize(gridSize, StoneType::None);
    }
}

GoBoard::~GoBoard() {
}

void GoBoard::SetBounds(float nx, float ny, float size) {
    x = nx;
    y = ny;
    boardSize = size;
}

void GoBoard::PlaceStone(int gx, int gy, StoneType type) {
    if (gx >= 0 && gx < gridSize && gy >= 0 && gy < gridSize) {
        stones[gy][gx] = type;
    }
}

void GoBoard::ClearStone(int gx, int gy) {
    if (gx >= 0 && gx < gridSize && gy >= 0 && gy < gridSize) {
        stones[gy][gx] = StoneType::None;
    }
}

void GoBoard::ClearBoard() {
    for (int i = 0; i < gridSize; ++i) {
        for (int j = 0; j < gridSize; ++j) {
            stones[i][j] = StoneType::None;
        }
    }
}

void GoBoard::GridToScreen(int gx, int gy, float& sx, float& sy) const {
    float padding = boardSize * 0.05f;
    float cellSize = (boardSize - padding * 2) / (float)(gridSize - 1);
    sx = x + padding + gx * cellSize;
    sy = y + padding + gy * cellSize;
}

bool GoBoard::ScreenToGrid(float sx, float sy, int& gx, int& gy) const {
    float padding = boardSize * 0.05f;
    float cellSize = (boardSize - padding * 2) / (float)(gridSize - 1);
    
    float fx = (sx - x - padding) / cellSize;
    float fy = (sy - y - padding) / cellSize;
    
    gx = (int)(fx + 0.5f);
    gy = (int)(fy + 0.5f);
    
    return gx >= 0 && gx < gridSize && gy >= 0 && gy < gridSize;
}

bool GoBoard::OnMouseMove(float mx, float my) {
    if (!visible) return false;

    int gx, gy;
    if (ScreenToGrid(mx, my, gx, gy)) {
        hoverX = (float)gx;
        hoverY = (float)gy;
        showHover = true;
        return true;
    } else {
        showHover = false;
        return false;
    }
}

bool GoBoard::OnMouseDown(float mx, float my) {
    if (!visible) return false;

    int gx, gy;
    if (ScreenToGrid(mx, my, gx, gy)) {
        if (onPositionClick) {
            onPositionClick(gx, gy);
        }
        return true;
    }
    return false;
}

void GoBoard::DrawBoard(Gdiplus::Graphics* g) {
    // 棋盘木色背景
    Gdiplus::LinearGradientBrush boardBrush(
        Gdiplus::Point((int)x, (int)y),
        Gdiplus::Point((int)(x + boardSize), (int)(y + boardSize)),
        Gdiplus::Color(255, 200, 160, 100),
        Gdiplus::Color(255, 180, 140, 80)
    );
    
    // 绘制圆角矩形棋盘
    float radius = boardSize * 0.02f;
    Gdiplus::GraphicsPath path;
    path.AddArc(x, y, radius * 2, radius * 2, 180, 90);
    path.AddArc(x + boardSize - radius * 2, y, radius * 2, radius * 2, 270, 90);
    path.AddArc(x + boardSize - radius * 2, y + boardSize - radius * 2, radius * 2, radius * 2, 0, 90);
    path.AddArc(x, y + boardSize - radius * 2, radius * 2, radius * 2, 90, 90);
    path.CloseFigure();
    
    g->FillPath(&boardBrush, &path);
    
    // 绘制边框
    Gdiplus::Pen borderPen(Gdiplus::Color(255, 120, 80, 40), 3.0f);
    g->DrawPath(&borderPen, &path);
}

void GoBoard::DrawGridLines(Gdiplus::Graphics* g) {
    float padding = boardSize * 0.05f;
    float cellSize = (boardSize - padding * 2) / (float)(gridSize - 1);
    
    Gdiplus::Pen gridPen(Gdiplus::Color(200, 50, 40, 30), 1.0f);
    
    // 绘制横线
    for (int i = 0; i < gridSize; ++i) {
        float sy = y + padding + i * cellSize;
        g->DrawLine(&gridPen, x + padding, sy, x + boardSize - padding, sy);
    }
    
    // 绘制竖线
    for (int i = 0; i < gridSize; ++i) {
        float sx = x + padding + i * cellSize;
        g->DrawLine(&gridPen, sx, y + padding, sx, y + boardSize - padding);
    }
}

void GoBoard::DrawStarPoints(Gdiplus::Graphics* g) {
    float padding = boardSize * 0.05f;
    float cellSize = (boardSize - padding * 2) / (float)(gridSize - 1);
    float starRadius = boardSize * 0.008f;
    
    Gdiplus::SolidBrush starBrush(Gdiplus::Color(200, 50, 40, 30));
    
    // 星位点位置
    std::vector<std::pair<int, int>> starPoints;
    if (gridSize == 19) {
        starPoints = {{3,3}, {9,3}, {15,3}, {3,9}, {9,9}, {15,9}, {3,15}, {9,15}, {15,15}};
    } else if (gridSize == 13) {
        starPoints = {{3,3}, {9,3}, {6,6}, {3,9}, {9,9}};
    } else if (gridSize == 9) {
        starPoints = {{2,2}, {6,2}, {4,4}, {2,6}, {6,6}};
    }
    
    for (auto& p : starPoints) {
        float sx, sy;
        GridToScreen(p.first, p.second, sx, sy);
        g->FillEllipse(&starBrush, sx - starRadius, sy - starRadius, starRadius * 2, starRadius * 2);
    }
}

void GoBoard::DrawStone(Gdiplus::Graphics* g, float cx, float cy, float radius, StoneType type) {
    // 伦勃朗光源 - 右上45度
    float lightOffsetX = radius * 0.3f;
    float lightOffsetY = -radius * 0.3f;
    
    if (type == StoneType::Black) {
        // 黑子 - 深灰渐变
        Gdiplus::GraphicsPath path;
        path.AddEllipse(cx - radius, cy - radius, radius * 2, radius * 2);

        Gdiplus::PathGradientBrush gradientBrush(&path);
        gradientBrush.SetCenterColor(Gdiplus::Color(255, 80, 80, 80));
        Gdiplus::Color surroundBlack(255, 20, 20, 20);
        INT count1 = 1;
        gradientBrush.SetSurroundColors(&surroundBlack, &count1);

        // 高光
        Gdiplus::LinearGradientBrush highlightBrush(
            Gdiplus::Point((int)(cx - radius), (int)(cy - radius)),
            Gdiplus::Point((int)(cx + radius * 0.5f), (int)(cy + radius * 0.5f)),
            Gdiplus::Color(150, 120, 120, 120),
            Gdiplus::Color(0, 0, 0, 0)
        );

        g->FillEllipse(&gradientBrush, cx - radius, cy - radius, radius * 2, radius * 2);
        g->FillEllipse(&highlightBrush, cx - radius + lightOffsetX, cy - radius + lightOffsetY, radius, radius);

    } else if (type == StoneType::White) {
        // 白子 - 亮白渐变
        Gdiplus::GraphicsPath path;
        path.AddEllipse(cx - radius, cy - radius, radius * 2, radius * 2);

        Gdiplus::PathGradientBrush gradientBrush(&path);
        gradientBrush.SetCenterColor(Gdiplus::Color(255, 255, 255, 255));
        Gdiplus::Color surroundWhite(255, 200, 200, 200);
        INT count2 = 1;
        gradientBrush.SetSurroundColors(&surroundWhite, &count2);
        
        // 高光
        Gdiplus::LinearGradientBrush highlightBrush(
            Gdiplus::Point((int)(cx - radius), (int)(cy - radius)),
            Gdiplus::Point((int)(cx + radius * 0.5f), (int)(cy + radius * 0.5f)),
            Gdiplus::Color(200, 255, 255, 255),
            Gdiplus::Color(0, 255, 255, 255)
        );
        
        // 阴影
        Gdiplus::LinearGradientBrush shadowBrush(
            Gdiplus::Point((int)(cx - radius * 0.5f), (int)(cy - radius * 0.5f)),
            Gdiplus::Point((int)(cx + radius), (int)(cy + radius)),
            Gdiplus::Color(0, 0, 0, 0),
            Gdiplus::Color(80, 150, 150, 150)
        );
        
        g->FillEllipse(&gradientBrush, cx - radius, cy - radius, radius * 2, radius * 2);
        g->FillEllipse(&highlightBrush, cx - radius + lightOffsetX, cy - radius + lightOffsetY, radius, radius);
        g->FillEllipse(&shadowBrush, cx - radius, cy - radius, radius * 2, radius * 2);
    }
    
    // 边缘描边
    Gdiplus::Color edgeColor = (type == StoneType::Black) ? 
        Gdiplus::Color(255, 10, 10, 10) : Gdiplus::Color(255, 150, 150, 150);
    Gdiplus::Pen edgePen(edgeColor, 1.0f);
    g->DrawEllipse(&edgePen, cx - radius, cy - radius, radius * 2, radius * 2);
}

void GoBoard::DrawStones(Gdiplus::Graphics* g) {
    float padding = boardSize * 0.05f;
    float cellSize = (boardSize - padding * 2) / (float)(gridSize - 1);
    float stoneRadius = cellSize * 0.45f;
    
    for (int gy = 0; gy < gridSize; ++gy) {
        for (int gx = 0; gx < gridSize; ++gx) {
            if (stones[gy][gx] != StoneType::None) {
                float sx, sy;
                GridToScreen(gx, gy, sx, sy);
                DrawStone(g, sx, sy, stoneRadius, stones[gy][gx]);
            }
        }
    }
}

void GoBoard::DrawHoverIndicator(Gdiplus::Graphics* g) {
    if (!showHover) return;
    
    float sx, sy;
    GridToScreen((int)hoverX, (int)hoverY, sx, sy);
    
    float padding = boardSize * 0.05f;
    float cellSize = (boardSize - padding * 2) / (float)(gridSize - 1);
    float indicatorRadius = cellSize * 0.35f;
    
    // 半透明指示
    Gdiplus::Color hoverColor(100, 100, 200, 255);
    Gdiplus::SolidBrush hoverBrush(hoverColor);
    Gdiplus::Pen hoverPen(hoverColor, 2.0f);
    
    g->DrawEllipse(&hoverPen, sx - indicatorRadius, sy - indicatorRadius, 
                    indicatorRadius * 2, indicatorRadius * 2);
}

void GoBoard::Render(Gdiplus::Graphics* g) {
    if (!visible) return;
    
    DrawBoard(g);
    DrawGridLines(g);
    DrawStarPoints(g);
    DrawStones(g);
    DrawHoverIndicator(g);
}

} // namespace GUI
} // namespace SpaceGo
