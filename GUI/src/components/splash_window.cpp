// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：splash_window.cpp
// 描述：启动窗口实现
// 创建日期：2026-02-20
// 作者：SpaceGo Team
// ============================================================================

#include "components/splash_window.h"
#include "core/renderer.h"
#include "utils/color.h"

namespace SpaceGo {
namespace GUI {

SplashWindow::SplashWindow(HINSTANCE hInstance)
    : Window(hInstance, L"SpaceGo | 维空弈", 600, 400)
    , progress(0.0f)
    , logoScale(1.0f)
    , fadeAlpha(1.0f)
    , isFadeOut(false)
    , animationTime(0.0f)
{
}

SplashWindow::~SplashWindow() {
}

void SplashWindow::SetProgress(float value) {
    progress = Math::Clamp(value, 0.0f, 1.0f);
    if (hwnd) {
        InvalidateRect(hwnd, NULL, FALSE);
    }
}

void SplashWindow::SetStatusText(const std::wstring& text) {
    currentStatus = text;
    if (hwnd) {
        InvalidateRect(hwnd, NULL, FALSE);
    }
}

void SplashWindow::StartFadeOut() {
    isFadeOut = true;
}

bool SplashWindow::IsFadeOutComplete() const {
    return isFadeOut && fadeAlpha <= 0.0f;
}

void SplashWindow::UpdateAnimation(float deltaTime) {
    animationTime += deltaTime;
    
    if (isFadeOut) {
        fadeAlpha -= deltaTime * 2.0f;
        if (fadeAlpha <= 0.0f) {
            fadeAlpha = 0.0f;
        }
    }
    
    logoScale = 1.0f + sinf(animationTime * 2.0f) * 0.05f;
    
    if (hwnd) {
        InvalidateRect(hwnd, NULL, FALSE);
    }
}

LRESULT SplashWindow::HandleMessage(UINT msg, WPARAM wParam, LPARAM lParam) {
    return Window::HandleMessage(msg, wParam, lParam);
}

void SplashWindow::OnPaint(Gdiplus::Graphics* graphics) {
    UIRenderer renderer;
    renderer.SetGraphics(graphics);
    renderer.SetSize(width, height);
    RenderContent(&renderer);
}

void SplashWindow::RenderContent(UIRenderer* renderer) {
    if (!renderer) return;

    ColorManager& cm = ColorManager::GetInstance();
    const ColorScheme& colors = cm.GetColors();

    float centerX = width / 2.0f;
    float centerY = height / 2.0f;

    renderer->DrawRect(colors.background, colors.background, 0, 0, width, height);

    std::wstring title = L"SpaceGo | 维空弈";
    float titleY = centerY - 80.0f;
    renderer->DrawTextCentered(title, colors.textPrimary, 0, titleY, width, 50.0f, 36.0f);

    std::wstring subtitle = L"围棋 AI 对弈系统";
    renderer->DrawTextCentered(subtitle, colors.textSecondary, 0, titleY + 50.0f, width, 30.0f, 18.0f);

    float progressBarWidth = 300.0f;
    float progressBarHeight = 8.0f;
    float progressBarX = centerX - progressBarWidth / 2.0f;
    float progressBarY = centerY + 60.0f;

    renderer->DrawRect(colors.neutralLight, colors.neutralLight, 
                       progressBarX, progressBarY, progressBarWidth, progressBarHeight, 4.0f);

    float fillWidth = progressBarWidth * progress;
    renderer->DrawRect(colors.primaryNormal, colors.primaryNormal, 
                       progressBarX, progressBarY, fillWidth, progressBarHeight, 4.0f);

    if (!currentStatus.empty()) {
        float statusY = progressBarY + 30.0f;
        renderer->DrawTextCentered(currentStatus, colors.textSecondary, 0, statusY, width, 25.0f, 14.0f);
    }

    std::wstring version = L"版本 1.0.0 | SpaceGo Team";
    renderer->DrawTextCentered(version, colors.textDisabled, 0, height - 50.0f, width, 25.0f, 12.0f);
}

} // namespace GUI
} // namespace SpaceGo
