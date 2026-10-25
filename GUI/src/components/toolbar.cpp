// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：toolbar.cpp
// 描述：工具栏组件实现
// 创建日期：2026-02-21
// 作者：SpaceGo Team
// ============================================================================

#include "components/toolbar.h"
#pragma comment(lib, "gdiplus.lib")

namespace SpaceGo {
namespace GUI {

Toolbar::Toolbar(float px, float py, float w, float h)
    : x(px), y(py), width(w), height(h), visible(true) {
}

Toolbar::~Toolbar() {
}

Button* Toolbar::AddButton(const std::wstring& text, ToolbarCallback callback) {
    auto btn = std::make_unique<Button>(0, 0, 80, 50, text);
    Button* btnPtr = btn.get();
    btn->onClick = callback;
    buttons.push_back(std::move(btn));
    LayoutButtons();
    return btnPtr;
}

void Toolbar::SetBounds(float nx, float ny, float nw, float nh) {
    x = nx;
    y = ny;
    width = nw;
    height = nh;
    LayoutButtons();
}

void Toolbar::SetVisible(bool v) {
    visible = v;
}

void Toolbar::LayoutButtons() {
    if (buttons.empty()) return;

    float btnWidth = 80;
    float btnHeight = height - 20;
    float spacing = 10;
    float startY = y + 10;
    float currentY = startY;

    for (size_t i = 0; i < buttons.size(); ++i) {
        buttons[i]->SetBounds(x + 10, currentY, btnWidth, btnHeight);
        currentY += btnHeight + spacing;
    }
}

bool Toolbar::OnMouseMove(float mx, float my) {
    if (!visible) return false;

    bool needInvalidate = false;
    for (auto& btn : buttons) {
        needInvalidate |= btn->OnMouseMove(mx, my);
    }
    return needInvalidate;
}

bool Toolbar::OnMouseDown(float mx, float my) {
    if (!visible) return false;

    bool needInvalidate = false;
    for (auto& btn : buttons) {
        needInvalidate |= btn->OnMouseDown(mx, my);
    }
    return needInvalidate;
}

bool Toolbar::OnMouseUp(float mx, float my) {
    if (!visible) return false;

    bool needInvalidate = false;
    for (auto& btn : buttons) {
        needInvalidate |= btn->OnMouseUp(mx, my);
    }
    return needInvalidate;
}

void Toolbar::Render(Gdiplus::Graphics* g) {
    if (!visible) return;

    // 工具栏背景 - 深空蓝
    Gdiplus::LinearGradientBrush bgBrush(
        Gdiplus::Point((int)x, (int)y),
        Gdiplus::Point((int)(x + width), (int)y),
        Gdiplus::Color(255, 42, 63, 84),
        Gdiplus::Color(255, 30, 41, 59)
    );
    g->FillRectangle(&bgBrush, x, y, width, height);

    // 边框 - 灰蓝色
    Gdiplus::Pen borderPen(Gdiplus::Color(255, 61, 86, 110), 1.5f);
    g->DrawLine(&borderPen, x + width, y, x + width, y + height);

    // 渲染按钮
    for (auto& btn : buttons) {
        btn->Render(g);
    }
}

} // namespace GUI
} // namespace SpaceGo
