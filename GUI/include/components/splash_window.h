// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：splash_window.h
// 描述：启动窗口
// 创建日期：2026-02-20
// 作者：SpaceGo Team
// ============================================================================

#pragma once
#include "core/window.h"
#include "core/animation.h"
#include "core/renderer.h"
#include "utils/color.h"
#include <string>

namespace SpaceGo {
namespace GUI {

// 启动窗口
class SplashWindow : public Window {
private:
    float progress;               // 初始化进度 (0.0 - 1.0)
    float logoScale;            // Logo 缩放动画
    float fadeAlpha;            // 淡入淡出透明度
    bool isFadeOut;         // 是否正在淡出
    std::wstring currentStatus;   // 当前状态文本
    float animationTime;

public:
    SplashWindow(HINSTANCE hInstance);
    virtual ~SplashWindow();

    // 设置进度
    void SetProgress(float value);
    void SetStatusText(const std::wstring& text);

    // 开始淡出
    void StartFadeOut();

    // 是否完成淡出
    bool IsFadeOutComplete() const;

    // 更新动画
    void UpdateAnimation(float deltaTime);

protected:
    LRESULT HandleMessage(UINT msg, WPARAM wParam, LPARAM lParam) override;
    void OnPaint(Gdiplus::Graphics* graphics) override;
    void RenderContent(UIRenderer* renderer);
};

} // namespace GUI
} // namespace SpaceGo
