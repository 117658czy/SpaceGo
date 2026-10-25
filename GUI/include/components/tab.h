// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：tab.h
// 描述：标签页控件
// 创建日期：2026-02-20
// 作者：SpaceGo Team
// ============================================================================

#pragma once
#include <windows.h>
#include <gdiplus.h>
#include <string>
#include <vector>
#include <functional>
#include "utils/color.h"
#include "core/animation.h"

namespace SpaceGo {
namespace GUI {

// 标签页切换回调
using TabChangedCallback = std::function<void(int index)>;

class TabControl {
public:
    float x, y, width, height;
    std::vector<std::wstring> tabs;
    int activeIndex;
    int hoveredIndex;
    TabChangedCallback onTabChanged;
    float tabHeight;
    bool visible;

    // 动画相关
    int previousActiveIndex;
    float tabTransitionProgress;
    bool tabTransitioning;

    TabControl(float x, float y, float w, float h);
    ~TabControl();

    // 添加标签页
    void AddTab(const std::wstring& name);

    // 设置活动标签
    void SetActiveTab(int index);

    // 更新动画
    void Update(float deltaTime);

    // 设置位置大小
    void SetBounds(float nx, float ny, float nw, float nh);

    // 鼠标事件
    bool OnMouseMove(float mx, float my);
    bool OnMouseDown(float mx, float my);

    // 渲染
    void Render(Gdiplus::Graphics* g);

private:
    // 获取标签页的位置
    void GetTabRect(int index, float& tx, float& ty, float& tw, float& th) const;

    // 检查点在哪个标签页
    int GetTabAtPoint(float px, float py) const;
};

} // namespace GUI
} // namespace SpaceGo
