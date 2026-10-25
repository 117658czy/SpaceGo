// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：new_game_dialog.h
// 描述：新游戏对话框
// 创建日期：2026-02-21
// 作者：SpaceGo Team
// ============================================================================

#pragma once
#include "core/window.h"
#include "components/button.h"
#include "models/game.h"

namespace SpaceGo {
namespace GUI {

class NewGameDialog : public Window {
public:
    NewGameDialog();
    virtual ~NewGameDialog();

    // 获取配置
    Models::GameMode GetGameMode() const;
    int GetBoardSize() const;
    std::wstring GetBlackPlayerName() const;
    std::wstring GetWhitePlayerName() const;
    bool IsAccepted() const;

protected:
    virtual void OnCreate() override;
    virtual void OnPaint(Gdiplus::Graphics* g) override;
    virtual void OnMouseMove(float x, float y) override;
    virtual void OnMouseDown(float x, float y) override;
    virtual void OnMouseUp(float x, float y) override;

private:
    Button* okBtn;
    Button* cancelBtn;
    Button* pvpBtn;
    Button* pvaiBtn;
    Button* aivaiBtn;
    Button* size9Btn;
    Button* size13Btn;
    Button* size19Btn;

    Models::GameMode selectedMode;
    int selectedSize;
    bool accepted;

    void LayoutComponents();
    void OnOkClicked();
    void OnCancelClicked();
};

} // namespace GUI
} // namespace SpaceGo
