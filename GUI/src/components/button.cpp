// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：button.cpp
// 描述：按钮控件实现
// 创建日期：2026-02-20
// 作者：SpaceGo Team
// ============================================================================

#include "components/button.h"
#include "utils/font.h"
#pragma comment(lib, "gdiplus.lib")

namespace SpaceGo {
namespace GUI {

Button::Button(float px, float py, float w, float h, const std::wstring& txt)
    : x(px), y(py), width(w), height(h), text(txt), state(ButtonState::Normal),
      cornerRadius(8.0f), visible(true), enabled(true),
      previousState(ButtonState::Normal), stateTransitionProgress(1.0f),
      stateTransitioning(false) {
}

Button::~Button() {
}

void Button::SetBounds(float nx, float ny, float nw, float nh) {
    x = nx;
    y = ny;
    width = nw;
    height = nh;
}

void Button::SetText(const std::wstring& txt) {
    text = txt;
}

void Button::SetVisible(bool v) {
    visible = v;
}

void Button::SetEnabled(bool e) {
    enabled = e;
    ButtonState newState = state;
    if (!enabled) {
        newState = ButtonState::Disabled;
    } else if (state == ButtonState::Disabled) {
        newState = ButtonState::Normal;
    }
    if (newState != state) {
        previousState = state;
        state = newState;
        stateTransitionProgress = 0.0f;
        stateTransitioning = true;
    }
}

void Button::Update(float deltaTime) {
    if (stateTransitioning) {
        stateTransitionProgress += deltaTime * 8.0f;
        if (stateTransitionProgress >= 1.0f) {
            stateTransitionProgress = 1.0f;
            stateTransitioning = false;
        }
    }
}

bool Button::ContainsPoint(float px, float py) const {
    return px >= x && px < x + width && py >= y && py < y + height;
}

bool Button::OnMouseMove(float mx, float my) {
    if (!visible || !enabled) return false;

    bool contains = ContainsPoint(mx, my);
    if (state == ButtonState::Normal && contains) {
        previousState = state;
        state = ButtonState::Hovered;
        stateTransitionProgress = 0.0f;
        stateTransitioning = true;
        return true;
    } else if (state == ButtonState::Hovered && !contains) {
        previousState = state;
        state = ButtonState::Normal;
        stateTransitionProgress = 0.0f;
        stateTransitioning = true;
        return true;
    }
    return false;
}

bool Button::OnMouseDown(float mx, float my) {
    if (!visible || !enabled) return false;

    if (ContainsPoint(mx, my)) {
        previousState = state;
        state = ButtonState::Pressed;
        stateTransitionProgress = 0.0f;
        stateTransitioning = true;
        return true;
    }
    return false;
}

bool Button::OnMouseUp(float mx, float my) {
    if (!visible || !enabled) return false;

    bool contains = ContainsPoint(mx, my);
    if (state == ButtonState::Pressed) {
        previousState = state;
        if (contains) {
            state = ButtonState::Hovered;
            if (onClick) {
                onClick();
            }
        } else {
            state = ButtonState::Normal;
        }
        stateTransitionProgress = 0.0f;
        stateTransitioning = true;
        return true;
    }
    return false;
}

Gdiplus::Color Button::GetBackgroundColor(ButtonState s) const {
    switch (s) {
        case ButtonState::Normal:
            return Gdiplus::Color(255, 42, 63, 84);
        case ButtonState::Hovered:
            return Gdiplus::Color(255, 52, 73, 94);
        case ButtonState::Pressed:
            return Gdiplus::Color(255, 35, 58, 84);
        case ButtonState::Disabled:
            return Gdiplus::Color(100, 42, 63, 84);
        default:
            return Gdiplus::Color(255, 42, 63, 84);
    }
}

Gdiplus::Color Button::GetTextColor(ButtonState s) const {
    switch (s) {
        case ButtonState::Disabled:
            return Gdiplus::Color(255, 100, 110, 115);
        default:
            return Gdiplus::Color(255, 236, 240, 241);
    }
}

Gdiplus::Color Button::GetBorderColor(ButtonState s) const {
    switch (s) {
        case ButtonState::Hovered:
            return Gdiplus::Color(255, 74, 144, 164);
        case ButtonState::Pressed:
            return Gdiplus::Color(255, 243, 156, 18);
        default:
            return Gdiplus::Color(255, 61, 86, 110);
    }
}

Gdiplus::Color LerpColor(Gdiplus::Color from, Gdiplus::Color to, float t) {
    BYTE a = (BYTE)(from.GetA() + (to.GetA() - from.GetA()) * t);
    BYTE r = (BYTE)(from.GetR() + (to.GetR() - from.GetR()) * t);
    BYTE g = (BYTE)(from.GetG() + (to.GetG() - from.GetG()) * t);
    BYTE b = (BYTE)(from.GetB() + (to.GetB() - from.GetB()) * t);
    return Gdiplus::Color(a, r, g, b);
}

void Button::Render(Gdiplus::Graphics* g) {
    if (!visible) return;

    // 缓动函数 - EaseOutQuint
    float t = stateTransitionProgress;
    float easedT = 1.0f + (--t) * t * t * t * t;

    // 获取当前插值颜色
    Gdiplus::Color bgColor, textColor, borderColor;
    if (stateTransitioning) {
        Gdiplus::Color prevBg = GetBackgroundColor(previousState);
        Gdiplus::Color currBg = GetBackgroundColor(state);
        Gdiplus::Color prevText = GetTextColor(previousState);
        Gdiplus::Color currText = GetTextColor(state);
        Gdiplus::Color prevBorder = GetBorderColor(previousState);
        Gdiplus::Color currBorder = GetBorderColor(state);

        bgColor = LerpColor(prevBg, currBg, easedT);
        textColor = LerpColor(prevText, currText, easedT);
        borderColor = LerpColor(prevBorder, currBorder, easedT);
    } else {
        bgColor = GetBackgroundColor(state);
        textColor = GetTextColor(state);
        borderColor = GetBorderColor(state);
    }

    // 按下效果 - 轻微缩放
    float scale = state == ButtonState::Pressed ? 0.98f : 1.0f;
    float drawX = x + width * (1.0f - scale) * 0.5f;
    float drawY = y + height * (1.0f - scale) * 0.5f;
    float drawW = width * scale;
    float drawH = height * scale;

    // 绘制圆角矩形背景
    Gdiplus::GraphicsPath path;
    float r = cornerRadius;
    path.AddArc(drawX, drawY, r * 2, r * 2, 180, 90);
    path.AddArc(drawX + drawW - r * 2, drawY, r * 2, r * 2, 270, 90);
    path.AddArc(drawX + drawW - r * 2, drawY + drawH - r * 2, r * 2, r * 2, 0, 90);
    path.AddArc(drawX, drawY + drawH - r * 2, r * 2, r * 2, 90, 90);
    path.CloseFigure();

    // 填充背景
    Gdiplus::SolidBrush bgBrush(bgColor);
    g->FillPath(&bgBrush, &path);

    // 绘制边框
    Gdiplus::Pen borderPen(borderColor, 1.5f);
    g->DrawPath(&borderPen, &path);

    // 绘制文字 - 使用FontManager缓存
    if (!text.empty()) {
        Gdiplus::Font* font = FontManager::GetInstance().GetFont(L"Microsoft YaHei", 14.0f);
        if (font) {
            Gdiplus::SolidBrush textBrush(textColor);
            Gdiplus::StringFormat format;
            format.SetAlignment(Gdiplus::StringAlignmentCenter);
            format.SetLineAlignment(Gdiplus::StringAlignmentCenter);

            Gdiplus::RectF rect(drawX, drawY, drawW, drawH);
            g->DrawString(text.c_str(), -1, font, rect, &format, &textBrush);
        }
    }
}

} // namespace GUI
} // namespace SpaceGo
