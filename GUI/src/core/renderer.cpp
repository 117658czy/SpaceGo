// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：renderer.cpp
// 描述：渲染器系统实现
// 创建日期：2026-02-20
// 作者：SpaceGo Team
// ============================================================================

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "core/renderer.h"
#include "utils/color.h"

namespace SpaceGo {
namespace GUI {

// Renderer 类实现
Renderer::Renderer() : graphics(nullptr), width(0), height(0) {}

Renderer::~Renderer() {}

void Renderer::SetGraphics(Gdiplus::Graphics* g) {
    graphics = g;
}

void Renderer::SetSize(int w, int h) {
    width = w;
    height = h;
}

void Renderer::DrawLine(const Color& color, float x1, float y1, float x2, float y2, float width) {
    if (!graphics) return;
    Gdiplus::Pen pen(color.ToGdiplusColor(), width);
    pen.SetLineJoin(Gdiplus::LineJoinRound);
    graphics->DrawLine(&pen, x1, y1, x2, y2);
}

void Renderer::DrawRect(const Color& fill, const Color& border, float x, float y, float w, float h, float borderRadius) {
    if (!graphics) return;

    if (borderRadius > 0.1f) {
        Gdiplus::GraphicsPath path;
        DrawRoundedRect(path, x, y, w, h, borderRadius);
        
        if (fill.a > 0.01f) {
            Gdiplus::SolidBrush brush(fill.ToGdiplusColor());
            graphics->FillPath(&brush, &path);
        }
        if (border.a > 0.01f) {
            Gdiplus::Pen pen(border.ToGdiplusColor(), 1.0f);
            graphics->DrawPath(&pen, &path);
        }
    } else {
        if (fill.a > 0.01f) {
            Gdiplus::SolidBrush brush(fill.ToGdiplusColor());
            graphics->FillRectangle(&brush, x, y, w, h);
        }
        if (border.a > 0.01f) {
            Gdiplus::Pen pen(border.ToGdiplusColor(), 1.0f);
            graphics->DrawRectangle(&pen, x, y, w, h);
        }
    }
}

void Renderer::DrawCircle(const Color& fill, const Color& border, float cx, float cy, float radius) {
    if (!graphics) return;
    
    if (fill.a > 0.01f) {
        Gdiplus::SolidBrush brush(fill.ToGdiplusColor());
        graphics->FillEllipse(&brush, cx - radius, cy - radius, radius * 2, radius * 2);
    }
    if (border.a > 0.01f) {
        Gdiplus::Pen pen(border.ToGdiplusColor(), 1.0f);
        graphics->DrawEllipse(&pen, cx - radius, cy - radius, radius * 2, radius * 2);
    }
}

void Renderer::DrawText(const std::wstring& text, const Color& color, float x, float y, float fontSize, const std::wstring& fontName) {
    if (!graphics) return;
    
    Gdiplus::Font font(fontName.c_str(), fontSize, Gdiplus::FontStyleRegular, Gdiplus::UnitPixel);
    Gdiplus::SolidBrush brush(color.ToGdiplusColor());
    Gdiplus::StringFormat format;
    format.SetAlignment(Gdiplus::StringAlignmentNear);
    format.SetLineAlignment(Gdiplus::StringAlignmentNear);
    
    graphics->DrawString(text.c_str(), -1, &font, Gdiplus::PointF(x, y), &format, &brush);
}

void Renderer::DrawTextCentered(const std::wstring& text, const Color& color, float x, float y, float w, float h, float fontSize, const std::wstring& fontName) {
    if (!graphics) return;
    
    Gdiplus::Font font(fontName.c_str(), fontSize, Gdiplus::FontStyleRegular, Gdiplus::UnitPixel);
    Gdiplus::SolidBrush brush(color.ToGdiplusColor());
    Gdiplus::StringFormat format;
    format.SetAlignment(Gdiplus::StringAlignmentCenter);
    format.SetLineAlignment(Gdiplus::StringAlignmentCenter);
    
    Gdiplus::RectF rect(x, y, w, h);
    graphics->DrawString(text.c_str(), -1, &font, rect, &format, &brush);
}

void Renderer::DrawImage(Gdiplus::Image* image, float x, float y, float w, float h) {
    if (!graphics || !image) return;
    graphics->DrawImage(image, x, y, w, h);
}

void Renderer::DrawBlurRect(const Color& color, float x, float y, float w, float h, float blurRadius, float borderRadius) {
    DrawRect(color, color, x, y, w, h, borderRadius);
}

void Renderer::DrawGlassEffect(float x, float y, float w, float h, float opacity) {
    if (!graphics) return;
    
    ColorManager& cm = ColorManager::GetInstance();
    const ColorScheme& colors = cm.GetColors();
    
    Color glassColor = colors.surface.WithAlpha(opacity * 0.3f);
    DrawRect(glassColor, glassColor, x, y, w, h, 16.0f);
    
    Color borderColor = Color::FromRGB(255, 255, 255).WithAlpha(opacity * 0.5f);
    Gdiplus::Pen pen(borderColor.ToGdiplusColor(), 1.0f);
    
    Gdiplus::GraphicsPath path;
    DrawRoundedRect(path, x, y, w, h, 16.0f);
    graphics->DrawPath(&pen, &path);
}

void Renderer::DrawShadow(float x, float y, float w, float h, float blurRadius, const Color& shadowColor, float offsetX, float offsetY) {
    DrawRect(shadowColor, shadowColor, x + offsetX, y + offsetY, w, h, 16.0f);
}

void Renderer::DrawRoundedRect(Gdiplus::GraphicsPath& path, float x, float y, float w, float h, float radius) {
    if (radius <= 0.1f) {
        path.AddRectangle(Gdiplus::RectF(x, y, w, h));
        return;
    }
    
    radius = Math::Min(radius, Math::Min(w, h) / 2.0f);
    
    path.AddArc(x, y, radius * 2, radius * 2, 180, 90);
    path.AddArc(x + w - radius * 2, y, radius * 2, radius * 2, 270, 90);
    path.AddArc(x + w - radius * 2, y + h - radius * 2, radius * 2, radius * 2, 0, 90);
    path.AddArc(x, y + h - radius * 2, radius * 2, radius * 2, 90, 90);
    path.CloseFigure();
}

// BoardRenderer 类实现
BoardRenderer::BoardRenderer() {}

void BoardRenderer::DrawBoard(float boardSize, int gridSize) {
    if (!graphics) return;
    
    float cellSize = boardSize / gridSize;
    float margin = cellSize * 0.8f;
    float drawSize = boardSize - margin * 2;
    float step = drawSize / (gridSize - 1);
    
    float boardX = margin;
    float boardY = margin;
    float boardW = drawSize;
    float boardH = drawSize;
    
    // 棋盘背景 - Sabaki风格榧木色
    Color boardBg = Color::FromRGB(220, 179, 92);
    Color boardBorder = Color::FromRGB(139, 105, 20);
    
    // 绘制棋盘底色（带微妙渐变）
    Gdiplus::Rect boardRect((INT)boardX, (INT)boardY, (INT)boardW, (INT)boardH);
    Gdiplus::LinearGradientBrush boardBrush(
        Gdiplus::PointF(boardX, boardY),
        Gdiplus::PointF(boardX + boardW, boardY + boardH),
        Gdiplus::Color(255, 225, 185, 100),
        Gdiplus::Color(255, 215, 173, 84)
    );
    graphics->FillRectangle(&boardBrush, boardRect);
    
    // 绘制棋盘边框
    Gdiplus::Pen borderPen(boardBorder.ToGdiplusColor(), 3.0f);
    graphics->DrawRectangle(&borderPen, boardX, boardY, boardW, boardH);
    
    // 绘制网格线 - Sabaki风格深褐色
    Color gridColor = Color::FromRGB(74, 55, 40);
    Gdiplus::Pen gridPen(gridColor.ToGdiplusColor(), 1.0f);
    
    for (int i = 0; i < gridSize; i++) {
        float pos = margin + i * step;
        
        // 水平线
        Gdiplus::PointF hStart(margin, pos);
        Gdiplus::PointF hEnd(margin + drawSize, pos);
        graphics->DrawLine(&gridPen, hStart, hEnd);
        
        // 垂直线
        Gdiplus::PointF vStart(pos, margin);
        Gdiplus::PointF vEnd(pos, margin + drawSize);
        graphics->DrawLine(&gridPen, vStart, vEnd);
    }
    
    // 绘制星位
    Color starColor = Color::FromRGB(74, 55, 40);
    std::vector<std::pair<int, int>> starPoints;
    if (gridSize == 19) {
        starPoints = {{3,3}, {9,3}, {15,3}, {3,9}, {9,9}, {15,9}, {3,15}, {9,15}, {15,15}};
    } else if (gridSize == 13) {
        starPoints = {{3,3}, {9,3}, {6,6}, {3,9}, {9,9}};
    } else if (gridSize == 9) {
        starPoints = {{2,2}, {6,2}, {4,4}, {2,6}, {6,6}};
    }
    
    for (auto& sp : starPoints) {
        float x = margin + sp.first * step;
        float y = margin + sp.second * step;
        float starRadius = 4.0f;
        
        Gdiplus::SolidBrush starBrush(starColor.ToGdiplusColor());
        graphics->FillEllipse(&starBrush, x - starRadius, y - starRadius, 
                               starRadius * 2, starRadius * 2);
    }
}

void BoardRenderer::DrawStone(float x, float y, bool isBlack, float size, float offsetX, float offsetY) {
    if (!graphics) return;

    float sx = x + offsetX;
    float sy = y + offsetY;
    float stoneSize = size;

    if (isBlack) {
        // 黑子 - Sabaki风格深邃黑，带细腻高光
        Gdiplus::GraphicsPath path;
        path.AddEllipse(sx - stoneSize, sy - stoneSize, stoneSize * 2, stoneSize * 2);

        // 主体渐变
        Gdiplus::PathGradientBrush mainBrush(&path);
        mainBrush.SetCenterPoint(Gdiplus::PointF(sx - stoneSize * 0.35f, sy - stoneSize * 0.35f));
        mainBrush.SetCenterColor(Gdiplus::Color(255, 60, 60, 65));
        Gdiplus::Color mainSurround(255, 15, 15, 20);
        INT mainCount = 1;
        mainBrush.SetSurroundColors(&mainSurround, &mainCount);

        graphics->FillEllipse(&mainBrush, sx - stoneSize, sy - stoneSize, stoneSize * 2, stoneSize * 2);

        // 高光效果
        float highlightSize = stoneSize * 0.7f;
        float highlightX = sx - stoneSize * 0.5f;
        float highlightY = sy - stoneSize * 0.6f;
        
        Gdiplus::GraphicsPath highlightPath;
        highlightPath.AddEllipse(highlightX, highlightY, highlightSize * 2, highlightSize * 2);
        
        Gdiplus::PathGradientBrush highlightBrush(&highlightPath);
        highlightBrush.SetCenterPoint(Gdiplus::PointF(highlightX + highlightSize * 0.5f, highlightY + highlightSize * 0.5f));
        highlightBrush.SetCenterColor(Gdiplus::Color(120, 100, 100, 110));
        Gdiplus::Color highlightSurround(0, 60, 60, 65);
        INT highlightCount = 1;
        highlightBrush.SetSurroundColors(&highlightSurround, &highlightCount);
        
        graphics->FillEllipse(&highlightBrush, highlightX, highlightY, highlightSize * 2, highlightSize * 2);
        
        // 微妙边缘
        Gdiplus::Pen edgePen(Gdiplus::Color(60, 0, 0, 0), 1.0f);
        graphics->DrawEllipse(&edgePen, sx - stoneSize, sy - stoneSize, stoneSize * 2, stoneSize * 2);
        
    } else {
        // 白子 - Sabaki风格温润白，带柔和阴影
        Gdiplus::GraphicsPath path;
        path.AddEllipse(sx - stoneSize, sy - stoneSize, stoneSize * 2, stoneSize * 2);

        // 主体渐变（左上高光，右下阴影）
        Gdiplus::PathGradientBrush mainBrush(&path);
        mainBrush.SetCenterPoint(Gdiplus::PointF(sx - stoneSize * 0.25f, sy - stoneSize * 0.3f));
        mainBrush.SetCenterColor(Gdiplus::Color(255, 255, 255, 255));
        Gdiplus::Color mainSurround(255, 225, 220, 215);
        INT mainCount = 1;
        mainBrush.SetSurroundColors(&mainSurround, &mainCount);

        graphics->FillEllipse(&mainBrush, sx - stoneSize, sy - stoneSize, stoneSize * 2, stoneSize * 2);

        // 柔和高光
        float highlightSize = stoneSize * 0.5f;
        float highlightX = sx - stoneSize * 0.35f;
        float highlightY = sy - stoneSize * 0.45f;
        
        Gdiplus::GraphicsPath highlightPath;
        highlightPath.AddEllipse(highlightX, highlightY, highlightSize * 2, highlightSize * 2);
        
        Gdiplus::PathGradientBrush highlightBrush(&highlightPath);
        highlightBrush.SetCenterPoint(Gdiplus::PointF(highlightX + highlightSize * 0.5f, highlightY + highlightSize * 0.5f));
        highlightBrush.SetCenterColor(Gdiplus::Color(100, 255, 255, 255));
        Gdiplus::Color highlightSurround(0, 255, 255, 255);
        INT highlightCount = 1;
        highlightBrush.SetSurroundColors(&highlightSurround, &highlightCount);
        
        graphics->FillEllipse(&highlightBrush, highlightX, highlightY, highlightSize * 2, highlightSize * 2);

        // 细腻边缘
        Gdiplus::Pen edgePen(Gdiplus::Color(100, 180, 180, 185), 1.2f);
        graphics->DrawEllipse(&edgePen, sx - stoneSize, sy - stoneSize, stoneSize * 2, stoneSize * 2);
    }
}

void BoardRenderer::DrawHoverIndicator(float x, float y, bool isBlack) {
    // Sabaki风格半透明预览棋子
    float indicatorSize = 10.0f;
    
    if (isBlack) {
        // 黑子预览
        Gdiplus::Color fillColor(150, 40, 40, 45);
        Gdiplus::SolidBrush brush(fillColor);
        graphics->FillEllipse(&brush, x - indicatorSize, y - indicatorSize, 
                               indicatorSize * 2, indicatorSize * 2);
        
        Gdiplus::Color borderColor(180, 60, 60, 65);
        Gdiplus::Pen pen(borderColor, 1.0f);
        graphics->DrawEllipse(&pen, x - indicatorSize, y - indicatorSize, 
                               indicatorSize * 2, indicatorSize * 2);
    } else {
        // 白子预览
        Gdiplus::Color fillColor(150, 240, 240, 245);
        Gdiplus::SolidBrush brush(fillColor);
        graphics->FillEllipse(&brush, x - indicatorSize, y - indicatorSize, 
                               indicatorSize * 2, indicatorSize * 2);
        
        Gdiplus::Color borderColor(180, 200, 200, 205);
        Gdiplus::Pen pen(borderColor, 1.0f);
        graphics->DrawEllipse(&pen, x - indicatorSize, y - indicatorSize, 
                               indicatorSize * 2, indicatorSize * 2);
    }
}

void BoardRenderer::DrawHeatmap(const std::vector<float>& probabilities, int gridSize) {
}

void BoardRenderer::DrawLastMoveIndicator(float x, float y) {
    // Sabaki风格红点标记
    float outerRadius = 7.0f;
    float innerRadius = 3.0f;
    
    // 外环
    Color outerColor = Color::FromRGB(231, 76, 60).WithAlpha(0.8f);
    Gdiplus::SolidBrush outerBrush(outerColor.ToGdiplusColor());
    graphics->FillEllipse(&outerBrush, x - outerRadius, y - outerRadius, 
                           outerRadius * 2, outerRadius * 2);
    
    // 内点（稍浅）
    Color innerColor = Color::FromRGB(255, 150, 130);
    Gdiplus::SolidBrush innerBrush(innerColor.ToGdiplusColor());
    graphics->FillEllipse(&innerBrush, x - innerRadius, y - innerRadius, 
                           innerRadius * 2, innerRadius * 2);
}

// UIRenderer 类实现
UIRenderer::UIRenderer() {}

void UIRenderer::DrawButton(float x, float y, float w, float h, const std::wstring& text, 
                           const Color& bgColor, const Color& textColor, 
                           bool isHovered, bool isPressed, float borderRadius) {
    Color fillColor = bgColor;
    if (isPressed) {
        fillColor = bgColor.Darken(0.1f);
    } else if (isHovered) {
        fillColor = bgColor.Lighten(0.05f);
    }
    
    DrawRect(fillColor, fillColor, x, y, w, h, borderRadius);
    DrawTextCentered(text, textColor, x, y, w, h, 14.0f);
}

void UIRenderer::DrawTab(float x, float y, float w, float h, const std::wstring& text,
                        bool isActive, float borderRadius) {
    ColorManager& cm = ColorManager::GetInstance();
    const ColorScheme& colors = cm.GetColors();
    
    Color bgColor = isActive ? colors.primaryNormal : colors.surface;
    Color textColor = isActive ? colors.textPrimary : colors.textSecondary;
    
    DrawRect(bgColor, bgColor, x, y, w, h, borderRadius);
    DrawTextCentered(text, textColor, x, y, w, h, 14.0f);
}

void UIRenderer::DrawCard(float x, float y, float w, float h, float borderRadius) {
    ColorManager& cm = ColorManager::GetInstance();
    const ColorScheme& colors = cm.GetColors();
    DrawRect(colors.surface, colors.border, x, y, w, h, borderRadius);
}

void UIRenderer::DrawProgressBar(float x, float y, float w, float h, float progress, const Color& color) {
    ColorManager& cm = ColorManager::GetInstance();
    const ColorScheme& colors = cm.GetColors();
    
    DrawRect(colors.neutralLight, colors.neutralLight, x, y, w, h, h / 2.0f);
    
    float fillWidth = w * Math::Clamp(progress, 0.0f, 1.0f);
    DrawRect(color, color, x, y, fillWidth, h, h / 2.0f);
}

} // namespace GUI
} // namespace SpaceGo
