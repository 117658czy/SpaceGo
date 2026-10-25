// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：main_window.h
// 描述：主窗口
// 创建日期：2026-02-20
// 作者：SpaceGo Team
// ============================================================================

#pragma once
#include <memory>
#include "core/window.h"
#include "components/tab.h"
#include "components/button.h"
#include "components/board.h"
#include "components/toolbar.h"
#include "components/game_info_panel.h"

namespace SpaceGo {
namespace GUI {

class MainWindow : public Window {
public:
    MainWindow();
    virtual ~MainWindow();

protected:
    virtual void OnCreate();
    virtual void OnPaint(Gdiplus::Graphics* g) override;
    virtual void OnUpdate(float deltaTime) override;
    virtual void OnMouseMove(float x, float y) override;
    virtual void OnMouseDown(float x, float y) override;
    virtual void OnMouseUp(float x, float y) override;
    virtual void OnResize(int width, int height) override;

private:
    std::unique_ptr<TabControl> tabControl;
    std::unique_ptr<Button> newGameBtn;
    std::unique_ptr<Button> loadGameBtn;
    std::unique_ptr<Button> settingsBtn;
    std::unique_ptr<GoBoard> goBoard;
    std::unique_ptr<Toolbar> toolbar;
    std::unique_ptr<GameInfoPanel> gameInfoPanel;
    bool turn;

    void LayoutComponents();
    void CreateSampleStones();
    void OnNewGame();
    void OnUndo();
    void OnPass();
    void OnPause();
    void OnSave();
    void OnLoad();
};

} // namespace GUI
} // namespace SpaceGo
