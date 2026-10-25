// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：main_window.cpp
// 描述：主窗口实现
// 创建日期：2026-02-20
// 作者：SpaceGo Team
// ============================================================================

#include "components/main_window.h"
#include <commdlg.h>
#pragma comment(lib, "gdiplus.lib")

namespace SpaceGo {
namespace GUI {

MainWindow::MainWindow()
    : Window(L"SpaceGo 维空弈", 1280, 800),
      turn(true) {
}

MainWindow::~MainWindow() {
}

void MainWindow::OnCreate() {
    // 创建标签页控件
    tabControl = std::make_unique<TabControl>(0, 0, 1280, 70);
    tabControl->AddTab(L"对弈");
    tabControl->AddTab(L"学习");
    tabControl->AddTab(L"分析");
    tabControl->AddTab(L"设置");

    // 创建工具栏
    toolbar = std::make_unique<Toolbar>(10, 80, 140, 600);
    toolbar->AddButton(L"新游戏", [this]() { OnNewGame(); });
    toolbar->AddButton(L"加载游戏", [this]() { OnLoad(); });
    toolbar->AddButton(L"悔棋", [this]() { OnUndo(); });
    toolbar->AddButton(L"弃权", [this]() { OnPass(); });
    toolbar->AddButton(L"暂停", [this]() { OnPause(); });
    toolbar->AddButton(L"保存", [this]() { OnSave(); });

    // 创建游戏信息面板
    gameInfoPanel = std::make_unique<GameInfoPanel>(1130, 80, 140, 600);
    gameInfoPanel->SetPlayerNames(L"黑方玩家", L"白方玩家");
    gameInfoPanel->SetTime(300, 300);

    // 创建棋盘
    float boardSize = 600;
    float boardX = 160;
    float boardY = 80;
    goBoard = std::make_unique<GoBoard>(boardX, boardY, boardSize, 19);

    // 设置棋盘回调
    goBoard->onPositionClick = [this](int x, int y) {
        StoneType type = turn ? StoneType::Black : StoneType::White;
        goBoard->PlaceStone(x, y, type);
        turn = !turn;
        gameInfoPanel->UpdateGameInfo(
            turn ? Models::PieceColor::White : Models::PieceColor::Black, 
            0
        );
        Invalidate();
    };

    // 创建示例棋子
    CreateSampleStones();

    LayoutComponents();
}

void MainWindow::CreateSampleStones() {
    // 放置一些示例棋子
    goBoard->PlaceStone(3, 3, StoneType::Black);
    goBoard->PlaceStone(15, 3, StoneType::White);
    goBoard->PlaceStone(3, 15, StoneType::White);
    goBoard->PlaceStone(15, 15, StoneType::Black);
    goBoard->PlaceStone(9, 9, StoneType::Black);
    goBoard->PlaceStone(10, 10, StoneType::White);
}

void MainWindow::OnNewGame() {
    if (goBoard) {
        goBoard->ClearBoard();
        turn = true;
        gameInfoPanel->UpdateGameInfo(Models::PieceColor::Black, 0);
        Invalidate();
    }
}

void MainWindow::OnUndo() {
    // 悔棋功能待实现
}

void MainWindow::OnPass() {
    // 弃权功能待实现
    turn = !turn;
    gameInfoPanel->UpdateGameInfo(
        turn ? Models::PieceColor::White : Models::PieceColor::Black, 
        0
    );
    Invalidate();
}

void MainWindow::OnPause() {
    // 暂停功能待实现
}

void MainWindow::OnSave() {
    // 保存功能待实现
}

void MainWindow::OnLoad() {
    // 打开文件对话框
    OPENFILENAMEW ofn = {0};
    wchar_t fileName[MAX_PATH] = L"";
    ofn.lStructSize = sizeof(OPENFILENAMEW);
    ofn.hwndOwner = NULL;
    ofn.lpstrFilter = L"SpaceGo 游戏文件 (*.sgf)\0*.sgf\0所有文件 (*.*)\0*.*\0";
    ofn.lpstrFile = fileName;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;
    ofn.lpstrDefExt = L"sgf";
    ofn.lpstrTitle = L"打开游戏文件";
    
    if (GetOpenFileNameW(&ofn)) {
        // 这里应该加载游戏文件
        // 暂时显示消息框
        MessageBoxW(NULL, L"游戏文件加载功能待实现", L"打开文件", MB_OK | MB_ICONINFORMATION);
    }
}

void MainWindow::LayoutComponents() {
    // 根据窗口大小重新布局
    if (tabControl) {
        tabControl->SetBounds(0, 0, (float)windowWidth, 70);
    }

    if (toolbar) {
        toolbar->SetBounds(10, 80, 140, (float)windowHeight - 90);
    }

    if (gameInfoPanel) {
        gameInfoPanel->SetBounds((float)windowWidth - 150, 80, 140, (float)windowHeight - 90);
    }

    if (goBoard) {
        float availableWidth = (float)windowWidth - 320;
        float availableHeight = (float)windowHeight - 100;
        float boardSize = Math::Min(availableWidth, availableHeight);
        boardSize = Math::Max(boardSize, 400.0f);
        float boardX = 160.0f + (availableWidth - boardSize) / 2.0f;
        float boardY = 80.0f + (availableHeight - boardSize) / 2.0f;
        goBoard->SetBounds(boardX, boardY, boardSize);
    }
}

void MainWindow::OnPaint(Gdiplus::Graphics* g) {
    // 背景渐变 - 深空蓝（SpaceGo特色）
    Gdiplus::LinearGradientBrush bgBrush(
        Gdiplus::Point(0, 0), Gdiplus::Point(0, windowHeight),
        Gdiplus::Color(255, 35, 48, 68),
        Gdiplus::Color(255, 25, 35, 50)
    );
    g->FillRectangle(&bgBrush, 0, 0, windowWidth, windowHeight);

    // 渲染组件
    if (tabControl) tabControl->Render(g);
    if (toolbar) toolbar->Render(g);
    if (gameInfoPanel) gameInfoPanel->Render(g);
    if (goBoard) goBoard->Render(g);
}

void MainWindow::OnMouseMove(float x, float y) {
    bool needInvalidate = false;
    if (tabControl) needInvalidate |= tabControl->OnMouseMove(x, y);
    if (toolbar) needInvalidate |= toolbar->OnMouseMove(x, y);
    if (goBoard) needInvalidate |= goBoard->OnMouseMove(x, y);
    
    if (needInvalidate) {
        Invalidate();
    }
}

void MainWindow::OnMouseDown(float x, float y) {
    bool needInvalidate = false;
    if (tabControl) needInvalidate |= tabControl->OnMouseDown(x, y);
    if (toolbar) needInvalidate |= toolbar->OnMouseDown(x, y);
    if (goBoard) needInvalidate |= goBoard->OnMouseDown(x, y);
    
    if (needInvalidate) {
        Invalidate();
    }
}

void MainWindow::OnMouseUp(float x, float y) {
    bool needInvalidate = false;
    if (toolbar) needInvalidate |= toolbar->OnMouseUp(x, y);
    
    if (needInvalidate) {
        Invalidate();
    }
}

void MainWindow::OnUpdate(float deltaTime) {
    bool needRedraw = false;

    // 更新TabControl动画
    if (tabControl) {
        tabControl->Update(deltaTime);
        if (tabControl->tabTransitioning) {
            needRedraw = true;
        }
    }

    // 更新Toolbar中的所有Button动画
    if (toolbar) {
        for (auto& btn : toolbar->buttons) {
            if (btn) {
                btn->Update(deltaTime);
                if (btn->stateTransitioning) {
                    needRedraw = true;
                }
            }
        }
    }

    // 如果需要，重绘窗口
    if (needRedraw) {
        Invalidate();
    }
}

void MainWindow::OnResize(int width, int height) {
    Window::OnResize(width, height);
    LayoutComponents();
    Invalidate();
}

} // namespace GUI
} // namespace SpaceGo
