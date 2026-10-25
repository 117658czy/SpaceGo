// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：renderer.h
// 描述：渲染器系统
// 创建日期：2026-02-20
// 作者：SpaceGo Team
// ============================================================================

#pragma once

// Windows头文件配置（通过color.h已包含，但为确保安全仍定义）
#ifndef NOMINMAX
#define NOMINMAX
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#ifndef UNICODE
#define UNICODE
#endif

#ifndef _UNICODE
#define _UNICODE
#endif

// Windows头文件包含顺序非常重要
#include <windows.h>
#include <objbase.h>
#include <wingdi.h>
#include <gdiplus.h>

// GDI+库链接
#pragma comment(lib, "gdiplus.lib")
#pragma comment(lib, "ole32.lib")

#include "utils/color.h"
#include "utils/math.h"

namespace SpaceGo {
namespace GUI {

// 渲染器基类
class Renderer {
protected:
    Gdiplus::Graphics* graphics;
    int width, height;

public:
    Renderer();
    virtual ~Renderer();

    // 设置渲染目标
    void SetGraphics(Gdiplus::Graphics* g);
    void SetSize(int w, int h);

    // 基础绘图
    void DrawLine(const Color& color, float x1, float y1, float x2, float y2, float width = 1.0f);
    void DrawRect(const Color& fill, const Color& border, float x, float y, float w, float h, float borderRadius = 0.0f);
    void DrawCircle(const Color& fill, const Color& border, float cx, float cy, float radius);
    void DrawText(const std::wstring& text, const Color& color, float x, float y, float fontSize, const std::wstring& fontName = L"Segoe UI");
    void DrawTextCentered(const std::wstring& text, const Color& color, float x, float y, float w, float h, float fontSize, const std::wstring& fontName = L"Segoe UI");
    void DrawImage(Gdiplus::Image* image, float x, float y, float w, float h);

    // 高级效果
    void DrawBlurRect(const Color& color, float x, float y, float w, float h, float blurRadius, float borderRadius = 0.0f);
    void DrawGlassEffect(float x, float y, float w, float h, float opacity = 0.5f);
    void DrawShadow(float x, float y, float w, float h, float blurRadius, const Color& shadowColor, float offsetX = 0, float offsetY = 0);

protected:
    void DrawRoundedRect(Gdiplus::GraphicsPath& path, float x, float y, float w, float h, float radius);
};

// 棋盘渲染器
class BoardRenderer : public Renderer {
public:
    BoardRenderer();

    // 棋盘绘制
    void DrawBoard(float boardSize, int gridSize = 19);
    void DrawStone(float x, float y, bool isBlack, float size, float offsetX = 0, float offsetY = 0);
    void DrawHoverIndicator(float x, float y, bool isBlack);
    void DrawHeatmap(const std::vector<float>& probabilities, int gridSize);
    void DrawLastMoveIndicator(float x, float y);
};

// UI渲染器
class UIRenderer : public Renderer {
public:
    UIRenderer();

    // 控件绘制
    void DrawButton(float x, float y, float w, float h, const std::wstring& text, 
                    const Color& bgColor, const Color& textColor, 
                    bool isHovered = false, bool isPressed = false, float borderRadius = 8.0f);
    void DrawTab(float x, float y, float w, float h, const std::wstring& text,
                 bool isActive, float borderRadius = 12.0f);
    void DrawCard(float x, float y, float w, float h, float borderRadius = 16.0f);
    void DrawProgressBar(float x, float y, float w, float h, float progress, const Color& color);
};

} // namespace GUI
} // namespace SpaceGo
