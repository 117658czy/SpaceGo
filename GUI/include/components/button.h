// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：button.h
// 描述：按钮控件
// 创建日期：2026-02-20
// 作者：SpaceGo Team
// ============================================================================

#pragma once
#include <windows.h>
#include <gdiplus.h>
#include <string>
#include <functional>
#include "utils/color.h"
#include "utils/math.h"

namespace SpaceGo {
namespace GUI {

// 按钮状态
enum class ButtonState {
    Normal,
    Hovered,
    Pressed,
    Disabled
};

// 按钮点击回调
using ButtonClickCallback = std::function<void()>;

class Button {
public:
    float x, y, width, height;
    std::wstring text;
    ButtonState state;
    ButtonClickCallback onClick;
    float cornerRadius;
    bool visible;
    bool enabled;

    // 动画相关
    ButtonState previousState;
    float stateTransitionProgress;
    bool stateTransitioning;

    Button(float x, float y, float w, float h, const std::wstring& txt);
    ~Button();

    // 设置位置和大小
    void SetBounds(float nx, float ny, float nw, float nh);

    // 设置文字
    void SetText(const std::wstring& txt);

    // 设置可见性
    void SetVisible(bool v);

    // 设置启用状态
    void SetEnabled(bool e);

    // 更新动画
    void Update(float deltaTime);

    // 鼠标事件
    bool OnMouseMove(float mx, float my);
    bool OnMouseDown(float mx, float my);
    bool OnMouseUp(float mx, float my);

    // 渲染
    void Render(Gdiplus::Graphics* g);

private:
    // 检查点是否在按钮内
    [[nodiscard]] bool ContainsPoint(float px, float py) const;

    // 获取指定状态的颜色
    [[nodiscard]] Gdiplus::Color GetBackgroundColor(ButtonState s) const;
    [[nodiscard]] Gdiplus::Color GetTextColor(ButtonState s) const;
    [[nodiscard]] Gdiplus::Color GetBorderColor(ButtonState s) const;
};

} // namespace GUI
} // namespace SpaceGo
