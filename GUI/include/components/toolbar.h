// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：toolbar.h
// 描述：工具栏组件
// 创建日期：2026-02-21
// 作者：SpaceGo Team
// ============================================================================

#pragma once
#include <windows.h>
#include <gdiplus.h>
#include <vector>
#include <functional>
#include <memory>
#include "components/button.h"

namespace SpaceGo {
namespace GUI {

// 工具栏按钮回调
using ToolbarCallback = std::function<void()>;

class Toolbar {
public:
    float x, y, width, height;
    std::vector<std::unique_ptr<Button>> buttons;
    bool visible;

    Toolbar(float px, float py, float w, float h);
    ~Toolbar();

    // 添加按钮
    Button* AddButton(const std::wstring& text, ToolbarCallback callback = nullptr);

    // 设置位置大小
    void SetBounds(float nx, float ny, float nw, float nh);

    // 设置可见性
    void SetVisible(bool v);

    // 鼠标事件
    bool OnMouseMove(float mx, float my);
    bool OnMouseDown(float mx, float my);
    bool OnMouseUp(float mx, float my);

    // 渲染
    void Render(Gdiplus::Graphics* g);

private:
    void LayoutButtons();
};

} // namespace GUI
} // namespace SpaceGo
