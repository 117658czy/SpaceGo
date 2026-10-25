// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：tab.cpp
// 描述：标签页控件实现
// 创建日期：2026-02-20
// 作者：SpaceGo Team
// ============================================================================

#include "components/tab.h"
#include "utils/font.h"
#pragma comment(lib, "gdiplus.lib")

namespace SpaceGo {
namespace GUI {

TabControl::TabControl(float px, float py, float w, float h)
    : x(px), y(py), width(w), height(h), activeIndex(0), hoveredIndex(-1),
      tabHeight(40.0f), visible(true),
      previousActiveIndex(0), tabTransitionProgress(1.0f), tabTransitioning(false) {
}

TabControl::~TabControl() {
}

void TabControl::AddTab(const std::wstring& name) {
    tabs.push_back(name);
}

void TabControl::SetActiveTab(int index) {
    if (index >= 0 && index < (int)tabs.size() && index != activeIndex) {
        previousActiveIndex = activeIndex;
        activeIndex = index;
        tabTransitionProgress = 0.0f;
        tabTransitioning = true;
        if (onTabChanged) {
            onTabChanged(index);
        }
    }
}

void TabControl::Update(float deltaTime) {
    if (tabTransitioning) {
        tabTransitionProgress += deltaTime * 4.0f;
        if (tabTransitionProgress >= 1.0f) {
            tabTransitionProgress = 1.0f;
            tabTransitioning = false;
        }
    }
}

void TabControl::SetBounds(float nx, float ny, float nw, float nh) {
    x = nx;
    y = ny;
    width = nw;
    height = nh;
}

void TabControl::GetTabRect(int index, float& tx, float& ty, float& tw, float& th) const {
    float tabWidth = width / (float)tabs.size();
    tx = x + index * tabWidth;
    ty = y;
    tw = tabWidth;
    th = tabHeight;
}

int TabControl::GetTabAtPoint(float px, float py) const {
    if (py < y || py >= y + tabHeight || px < x || px >= x + width) {
        return -1;
    }
    float tabWidth = width / (float)tabs.size();
    int index = (int)((px - x) / tabWidth);
    if (index >= 0 && index < (int)tabs.size()) {
        return index;
    }
    return -1;
}

bool TabControl::OnMouseMove(float mx, float my) {
    if (!visible) return false;

    int tabIndex = GetTabAtPoint(mx, my);
    if (tabIndex != hoveredIndex) {
        hoveredIndex = tabIndex;
        return true;
    }
    return false;
}

bool TabControl::OnMouseDown(float mx, float my) {
    if (!visible) return false;

    int tabIndex = GetTabAtPoint(mx, my);
    if (tabIndex >= 0) {
        SetActiveTab(tabIndex);
        return true;
    }
    return false;
}

void TabControl::Render(Gdiplus::Graphics* g) {
    if (!visible || tabs.empty()) return;

    // 绘制标签栏背景 - 深空蓝
    Gdiplus::SolidBrush tabBarBrush(Gdiplus::Color(255, 30, 41, 59));
    g->FillRectangle(&tabBarBrush, x, y, width, tabHeight);

    // 缓动函数 - EaseOutCubic
    float t = tabTransitionProgress;
    float easedT = (--t) * t * t + 1.0f;

    // 绘制每个标签
    for (size_t i = 0; i < tabs.size(); ++i) {
        float tx, ty, tw, th;
        GetTabRect((int)i, tx, ty, tw, th);

        // 确定颜色 - 带过渡动画
        Gdiplus::Color bgColor, textColor, borderColor;
        bool isActive = (int)i == activeIndex;
        bool wasActive = (int)i == previousActiveIndex;

        if (tabTransitioning && (isActive || wasActive)) {
            if (isActive) {
                // 新激活标签 - 淡入
                BYTE alpha = (BYTE)(easedT * 255);
                bgColor = Gdiplus::Color(alpha, 45, 74, 107);
                textColor = Gdiplus::Color(alpha, 236, 240, 241);
                borderColor = Gdiplus::Color(alpha, 74, 144, 164);
            } else if (wasActive) {
                // 旧激活标签 - 淡出
                BYTE alpha = (BYTE)((1.0f - easedT) * 255);
                bgColor = Gdiplus::Color(alpha, 45, 74, 107);
                textColor = Gdiplus::Color(alpha, 236, 240, 241);
                borderColor = Gdiplus::Color(alpha, 74, 144, 164);
            }
        } else if (isActive) {
            bgColor = Gdiplus::Color(255, 45, 74, 107);
            textColor = Gdiplus::Color(255, 236, 240, 241);
            borderColor = Gdiplus::Color(255, 74, 144, 164);
        } else if ((int)i == hoveredIndex) {
            bgColor = Gdiplus::Color(255, 52, 73, 94);
            textColor = Gdiplus::Color(255, 236, 240, 241);
            borderColor = Gdiplus::Color(255, 61, 86, 110);
        } else {
            bgColor = Gdiplus::Color(0, 0, 0, 0);
            textColor = Gdiplus::Color(255, 149, 165, 166);
            borderColor = Gdiplus::Color(0, 0, 0, 0);
        }

        // 填充背景
        Gdiplus::SolidBrush bgBrush(bgColor);
        g->FillRectangle(&bgBrush, tx, ty, tw, th);

        // 绘制边框（仅激活/悬停）
        if (isActive) {
            Gdiplus::Pen borderPen(borderColor, 2.0f);
            g->DrawRectangle(&borderPen, tx, ty, tw - 1, th - 1);
        }

        // 绘制文字 - 使用FontManager缓存
        Gdiplus::Font* font = FontManager::GetInstance().GetFont(L"Microsoft YaHei", 14.0f);
        if (font) {
            Gdiplus::SolidBrush textBrush(textColor);
            Gdiplus::StringFormat format;
            format.SetAlignment(Gdiplus::StringAlignmentCenter);
            format.SetLineAlignment(Gdiplus::StringAlignmentCenter);

            Gdiplus::RectF rect(tx, ty, tw, th);
            g->DrawString(tabs[i].c_str(), -1, font, rect, &format, &textBrush);
        }
    }

    // 绘制内容区域分隔线 - 灰蓝色
    Gdiplus::Pen separatorPen(Gdiplus::Color(255, 61, 86, 110), 1.5f);
    g->DrawLine(&separatorPen, x, y + tabHeight, x + width, y + tabHeight);
}

} // namespace GUI
} // namespace SpaceGo
