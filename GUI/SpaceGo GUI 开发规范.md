# SpaceGo 维空弈 - GUI 开发规范

## 一、概述

### 1.1 设计目标
本规范定义了 SpaceGo 维空弈围棋 AI 系统的 GUI 开发标准，要求：
- **无第三方库依赖**：仅使用 Windows 原生 API
- **现代视觉风格**：融合 Sabaki 的简洁与 Trae 的柔和风格
- **高性能渲染**：使用 GDI+ 实现流畅动画和特效
- **模块化架构**：清晰的代码组织和接口设计

### 1.2 技术栈
- **图形渲染**：GDI+ (Windows Graphics Device Interface Plus)
- **窗口管理**：Win32 API
- **色彩系统**：HSL/RGB 转换算法
- **动画系统**：基于定时器的帧动画

---

## 二、项目架构

### 2.1 目录结构
```
GUI/
├── include/                    # 头文件
│   ├── core/                   # 核心模块
│   │   ├── window.h            # 窗口基类
│   │   ├── renderer.h          # 渲染器
│   │   └── animation.h         # 动画系统
│   ├── widgets/                # 控件
│   │   ├── button.h            # 按钮
│   │   ├── label.h             # 标签
│   │   ├── tab.h               # 标签页
│   │   └── dialog.h            # 对话框
│   ├── components/             # 组件
│   │   ├── board.h             # 棋盘组件
│   │   ├── sidebar.h           # 侧边栏
│   │   └── topbar.h            # 顶栏
│   └── utils/                  # 工具
│       ├── color.h             # 色彩系统
│       ├── font.h              # 字体管理
│       └── math.h              # 数学工具
├── src/                        # 源文件
│   ├── core/
│   ├── widgets/
│   ├── components/
│   └── utils/
└── resources/                  # 资源
    └── fonts/                  # 字体文件
```

### 2.2 核心类层次
```
Window (基类)
├── MainWindow (主窗口)
├── SplashWindow (启动窗口)
└── Dialog (对话框基类)
    ├── NewGameDialog (新对局对话框)
    └── SettingsDialog (设置对话框)

Renderer (渲染器)
├── BoardRenderer (棋盘渲染器)
└── UIRenderer (UI渲染器)

Animation (动画基类)
├── TransitionAnimation (转场动画)
├── ParticleAnimation (粒子动画)
└── EasingAnimation (缓动动画)
```

---

## 三、核心模块设计

### 3.1 窗口系统 (core/window.h)

```cpp
#pragma once
#include <windows.h>
#include <gdiplus.h>
#include <string>
#include <vector>
#include <functional>

#pragma comment(lib, "gdiplus.lib")

namespace SpaceGo {
namespace GUI {

// 窗口基类
class Window {
protected:
    HWND hwnd;
    HINSTANCE hinstance;
    std::wstring title;
    int width, height;
    bool isVisible;
    bool isDraggable;
    POINT dragOffset;

public:
    Window(HINSTANCE hInstance, const std::wstring& title, int w, int h);
    virtual ~Window();

    // 创建窗口
    virtual bool Create();
    
    // 显示/隐藏窗口
    void Show();
    void Hide();
    
    // 窗口位置
    void SetPosition(int x, int y);
    void GetPosition(int& x, int& y) const;
    
    // 窗口大小
    void SetSize(int w, int h);
    void GetSize(int& w, int& h) const;
    
    // 获取窗口句柄
    HWND GetHandle() const { return hwnd; }
    
    // 消息处理（子类重写）
    virtual LRESULT HandleMessage(UINT msg, WPARAM wParam, LPARAM lParam);
    
    // 渲染（子类重写）
    virtual void OnPaint(Gdiplus::Graphics* graphics);

protected:
    // 窗口过程回调
    static LRESULT CALLBACK WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
};

} // namespace GUI
} // namespace SpaceGo
```

### 3.2 渲染器系统 (core/renderer.h)

```cpp
#pragma once
#include <gdiplus.h>
#include "utils/color.h"

namespace SpaceGo {
namespace GUI {

// 渲染器基类
class Renderer {
protected:
    Gdiplus::Graphics* graphics;
    int width, height;

public:
    Renderer();
    virtual ~Renderer();

    // 设置渲染目标
    void SetGraphics(Gdiplus::Graphics* g);
    void SetSize(int w, int h);

    // 基础绘图
    void DrawLine(const Color& color, float x1, float y1, float x2, float y2, float width = 1.0f);
    void DrawRect(const Color& fill, const Color& border, float x, float y, float w, float h, float borderRadius = 0.0f);
    void DrawCircle(const Color& fill, const Color& border, float cx, float cy, float radius);
    void DrawText(const std::wstring& text, const Color& color, float x, float y, float fontSize, const std::wstring& fontName = L"HarmonyOS Sans");
    void DrawImage(Gdiplus::Image* image, float x, float y, float w, float h);

    // 高级效果
    void DrawBlurRect(const Color& color, float x, float y, float w, float h, float blurRadius, float borderRadius = 0.0f);
    void DrawGlassEffect(float x, float y, float w, float h, float opacity = 0.5f);
    void DrawShadow(float x, float y, float w, float h, float blurRadius, const Color& shadowColor, float offsetX = 0, float offsetY = 0);
};

// 棋盘渲染器
class BoardRenderer : public Renderer {
public:
    BoardRenderer();

    // 棋盘绘制
    void DrawBoard(float boardSize, int gridSize = 19);
    void DrawStone(float x, float y, bool isBlack, float size, float offsetX = 0, float offsetY = 0);
    void DrawHoverIndicator(float x, float y, bool isBlack);
    void DrawHeatmap(const std::vector<float>& probabilities, int gridSize);
    void DrawLastMoveIndicator(float x, float y);
};

// UI渲染器
class UIRenderer : public Renderer {
public:
    UIRenderer();

    // 控件绘制
    void DrawButton(float x, float y, float w, float h, const std::wstring& text, 
                    const Color& bgColor, const Color& textColor, 
                    bool isHovered = false, bool isPressed = false, float borderRadius = 8.0f);
    void DrawTab(float x, float y, float w, float h, const std::wstring& text,
                 bool isActive, float borderRadius = 12.0f);
    void DrawCard(float x, float y, float w, float h, float borderRadius = 16.0f);
    void DrawProgressBar(float x, float y, float w, float h, float progress, const Color& color);
};

} // namespace GUI
} // namespace SpaceGo
```

### 3.3 动画系统 (core/animation.h)

```cpp
#pragma once
#include <windows.h>
#include <functional>
#include <vector>
#include <cmath>

namespace SpaceGo {
namespace GUI {

// 缓动函数类型
enum class EasingType {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    EaseOutBack,
    EaseInOutBack,
    Bounce,
    Elastic
};

// 缓动函数
class Easing {
public:
    static float Apply(EasingType type, float t);
    
private:
    static float Linear(float t);
    static float EaseIn(float t);
    static float EaseOut(float t);
    static float EaseInOut(float t);
    static float EaseOutBack(float t);
    static float Bounce(float t);
};

// 动画基类
class Animation {
protected:
    float duration;        // 持续时间(ms)
    float elapsed;         // 已用时间
    bool isPlaying;
    bool isLooping;
    std::function<void(float)> onUpdate;
    std::function<void()> onComplete;

public:
    Animation(float durationMs);
    virtual ~Animation();

    void Start();
    void Stop();
    void Pause();
    void Resume();
    void SetLooping(bool loop);

    void SetOnUpdate(std::function<void(float)> callback);
    void SetOnComplete(std::function<void()> callback);

    virtual void Update(float deltaTime);
    float GetProgress() const;

protected:
    virtual void OnUpdate(float progress) {}
};

// 属性动画
class PropertyAnimation : public Animation {
private:
    float startValue;
    float endValue;
    float* target;
    EasingType easing;

public:
    PropertyAnimation(float durationMs, float start, float end, float* targetValue, 
                     EasingType easingType = EasingType::EaseOut);

protected:
    void OnUpdate(float progress) override;
};

// 粒子系统
struct Particle {
    float x, y;
    float vx, vy;
    float size;
    float life;
    float maxLife;
    Color color;
};

class ParticleSystem {
private:
    std::vector<Particle> particles;
    float centerX, centerY;
    float mouseX, mouseY;
    bool isMouseActive;

public:
    ParticleSystem(int count, float areaWidth, float areaHeight);
    
    void Update(float deltaTime);
    void SetMousePosition(float x, float y, bool active);
    void Render(Renderer* renderer);

private:
    void InitializeParticles(int count, float areaWidth, float areaHeight);
    void ApplyVortexForce(Particle& p);
};

// 动画管理器
class AnimationManager {
private:
    std::vector<Animation*> animations;
    static AnimationManager* instance;
    DWORD lastTime;

    AnimationManager();

public:
    static AnimationManager* GetInstance();
    
    void AddAnimation(Animation* anim);
    void RemoveAnimation(Animation* anim);
    void Update();
    
    static const int MAX_TRANSITION_TIME = 200;  // 转场动画最大200ms
};

} // namespace GUI
} // namespace SpaceGo
```

---

## 四、色彩系统设计

### 4.1 色彩工具 (utils/color.h)

```cpp
#pragma once
#include <gdiplus.h>
#include <string>
#include <vector>

namespace SpaceGo {
namespace GUI {

// 颜色类（支持RGB和HSL）
class Color {
public:
    float r, g, b, a;  // RGBA (0.0 - 1.0)
    float h, s, l;     // HSL (H:0-360, S:0-1, L:0-1)

    Color();
    Color(float red, float green, float blue, float alpha = 1.0f);
    
    // 从RGB创建
    static Color FromRGB(int r, int g, int b, int a = 255);
    static Color FromRGB(float r, float g, float b, float a = 1.0f);
    
    // 从HSL创建
    static Color FromHSL(float h, float s, float l, float a = 1.0f);
    
    // 转换
    void RGBToHSL();
    void HSLToRGB();
    
    // 获取GDI+颜色
    Gdiplus::Color ToGdiplusColor() const;
    
    // 颜色操作
    Color Lerp(const Color& other, float t) const;
    Color Lighten(float amount) const;
    Color Darken(float amount) const;
    Color Saturate(float amount) const;
    Color Desaturate(float amount) const;
    Color WithAlpha(float alpha) const;
    
    // 获取对比度
    float GetContrast(const Color& other) const;
    
    // 字符串表示
    std::string ToHexString() const;
};

// 色彩模式
enum class ColorSchemeType {
    Dark,       // 深色模式
    Light,      // 浅色模式
    Normal,     // 标准模式
    Custom      // 自定义模式
};

// 颜色方案
struct ColorScheme {
    // 主色组
    Color primaryLight;
    Color primaryNormal;
    Color primaryDark;
    
    // 辅助色组
    Color secondaryLight;
    Color secondaryNormal;
    Color secondaryDark;
    
    // 中性色组
    Color neutralLight;
    Color neutralNormal;
    Color neutralDark;
    
    // 语义色
    Color success;
    Color warning;
    Color error;
    Color info;
    
    // 背景色
    Color background;
    Color surface;
    Color surfaceElevated;
    
    // 文字色
    Color textPrimary;
    Color textSecondary;
    Color textDisabled;
    
    // 边框色
    Color border;
    Color borderFocus;
};

// 色彩管理器
class ColorManager {
private:
    ColorSchemeType currentScheme;
    ColorScheme currentColors;
    Color customBaseColor;
    static ColorManager* instance;

    ColorManager();

public:
    static ColorManager* GetInstance();
    
    // 设置色彩方案
    void SetScheme(ColorSchemeType type);
    void SetCustomBaseColor(const Color& baseColor);
    
    // 获取当前颜色
    const ColorScheme& GetColors() const;
    ColorSchemeType GetSchemeType() const;
    
    // 生成预设方案
    static ColorScheme GenerateDarkScheme();
    static ColorScheme GenerateLightScheme();
    static ColorScheme GenerateNormalScheme();
    static ColorScheme GenerateCustomScheme(const Color& baseColor);
    
    // 基准色预设
    static const std::vector<Color>& GetPresetBaseColors();
    
    // 确保对比度符合WCAG标准
    static Color EnsureContrast(const Color& foreground, const Color& background, float minContrast = 4.5f);
};

} // namespace GUI
} // namespace SpaceGo
```

### 4.2 预设基准色

```cpp
// 预设基准色定义
const Color PRESET_BLUE    = Color::FromRGB(52, 152, 219);   // 专业蓝
const Color PRESET_GREEN   = Color::FromRGB(39, 174, 96);    // 智慧绿
const Color PRESET_BROWN   = Color::FromRGB(139, 69, 19);    // 古典棕
const Color PRESET_PURPLE  = Color::FromRGB(155, 89, 182);   // 现代紫
const Color PRESET_ORANGE  = Color::FromRGB(230, 126, 34);   // 活力橙
const Color PRESET_GRAY    = Color::FromRGB(127, 140, 141);  // 沉静灰
```

---

## 五、控件设计规范

### 5.1 按钮控件 (widgets/button.h)

```cpp
#pragma once
#include "core/window.h"
#include "core/renderer.h"
#include <functional>

namespace SpaceGo {
namespace GUI {

class Button {
private:
    float x, y, width, height;
    std::wstring text;
    bool isHovered;
    bool isPressed;
    bool isEnabled;
    float borderRadius;
    std::function<void()> onClick;

public:
    Button(float x, float y, float w, float h, const std::wstring& text = L"");
    
    // 位置和大小
    void SetPosition(float px, float py);
    void SetSize(float w, float h);
    void GetBounds(float& px, float& py, float& pw, float& ph) const;
    
    // 状态
    void SetEnabled(bool enabled);
    bool IsEnabled() const;
    
    // 事件
    void SetOnClick(std::function<void()> callback);
    
    // 交互
    bool HitTest(float mx, float my) const;
    void OnMouseEnter();
    void OnMouseLeave();
    void OnMouseDown();
    void OnMouseUp();
    
    // 渲染
    void Render(UIRenderer* renderer);
};

} // namespace GUI
} // namespace SpaceGo
```

### 5.2 标签页控件 (widgets/tab.h)

```cpp
#pragma once
#include <vector>
#include <functional>
#include "core/renderer.h"

namespace SpaceGo {
namespace GUI {

class TabPage {
public:
    std::wstring title;
    int index;
    bool isActive;
    
    TabPage(const std::wstring& t, int idx);
    virtual void Render(UIRenderer* renderer, float x, float y, float w, float h) = 0;
};

class TabControl {
private:
    float x, y, width, height;
    std::vector<TabPage*> pages;
    int activeIndex;
    float tabHeight;
    float borderRadius;
    std::function<void(int)> onTabChanged;

public:
    TabControl(float x, float y, float w, float h);
    ~TabControl();
    
    // 添加/移除标签页
    void AddTab(TabPage* page);
    void RemoveTab(int index);
    
    // 活动标签
    void SetActiveTab(int index);
    int GetActiveTab() const;
    
    // 事件
    void SetOnTabChanged(std::function<void(int)> callback);
    
    // 交互
    bool HitTestTab(float mx, float my, int& tabIndex) const;
    
    // 渲染
    void Render(UIRenderer* renderer);
};

} // namespace GUI
} // namespace SpaceGo
```

### 5.3 对话框 (widgets/dialog.h)

```cpp
#pragma once
#include "core/window.h"
#include "widgets/button.h"
#include <vector>

namespace SpaceGo {
namespace GUI {

class Dialog : public Window {
protected:
    Window* parentWindow;
    bool isModal;
    std::vector<Button*> buttons;
    int result;

public:
    Dialog(HINSTANCE hInstance, const std::wstring& title, int w, int h, Window* parent = nullptr);
    virtual ~Dialog();
    
    // 显示模态对话框
    int ShowModal();
    
    // 关闭对话框
    void Close(int dialogResult);
    
    // 获取结果
    int GetResult() const;

protected:
    void OnPaint(Gdiplus::Graphics* graphics) override;
    virtual void RenderContent(UIRenderer* renderer, float x, float y, float w, float h) {}
};

// 新对局对话框
class NewGameDialog : public Dialog {
private:
    // 对弈类型选择
    int gameType;  // 0:人机, 1:人人, 2:机机
    
    // AI设置
    int aiStrength;
    int aiThinkingTime;
    std::wstring aiStyle;
    
    // 棋盘设置
    int boardSize;
    int handicap;
    bool isBlackFirst;
    bool isRandomFirst;

public:
    NewGameDialog(HINSTANCE hInstance, Window* parent = nullptr);
    
    // 获取设置
    int GetGameType() const;
    int GetAIStrength() const;
    int GetAIThinkingTime() const;
    std::wstring GetAIStyle() const;
    int GetBoardSize() const;
    int GetHandicap() const;
    bool IsBlackFirst() const;

protected:
    void RenderContent(UIRenderer* renderer, float x, float y, float w, float h) override;
};

} // namespace GUI
} // namespace SpaceGo
```

---

## 六、组件设计规范

### 6.1 棋盘组件 (components/board.h)

```cpp
#pragma once
#include "core/renderer.h"
#include <vector>
#include <functional>

namespace SpaceGo {
namespace GUI {

// 落子记录
struct MoveRecord {
    int x, y;
    bool isBlack;
    int moveNumber;
};

// 棋盘组件
class BoardComponent {
private:
    float x, y, size;
    int gridSize;
    float cellSize;
    float stoneSize;
    std::vector<MoveRecord> moves;
    int currentMoveIndex;
    float hoverX, hoverY;
    bool isHoverValid;
    bool isBlackTurn;
    
    // 视觉效果
    float lastMoveOffsetX, lastMoveOffsetY;
    float vibrationIntensity;
    bool isVibrating;
    
    // 回调
    std::function<void(int, int, bool)> onMovePlayed;

public:
    BoardComponent(float px, float py, float boardSize, int gridSize = 19);
    
    // 位置和大小
    void SetPosition(float px, float py);
    void SetSize(float newSize);
    
    // 棋盘状态
    void Reset();
    void PlayMove(int gx, int gy, bool isBlack);
    void UndoMove();
    void RedoMove();
    void GoToMove(int moveIndex);
    
    // 交互
    void SetHoverPosition(float mx, float my);
    bool HitTest(float mx, float my) const;
    void ScreenToGrid(float mx, float my, int& gx, int& gy) const;
    void GridToScreen(int gx, int gy, float& sx, float& sy) const;
    
    // 视觉效果
    void TriggerVibration(float intensity = 3.0f);
    void UpdateAnimation(float deltaTime);
    
    // 回调
    void SetOnMovePlayed(std::function<void(int, int, bool)> callback);
    
    // 渲染
    void Render(BoardRenderer* renderer);
    
    // 热力图（注意力可视化）
    void SetHeatmap(const std::vector<float>& probabilities);

private:
    float GetRandomOffset() const;
};

} // namespace GUI
} // namespace SpaceGo
```

### 6.2 右侧信息面板 (components/sidebar.h)

```cpp
#pragma once
#include "core/renderer.h"
#include "widgets/tab.h"
#include "utils/color.h"
#include <vector>

namespace SpaceGo {
namespace GUI {

// 玩家信息
struct PlayerInfo {
    std::wstring name;
    int captures;
    float timeRemaining;
    int byoyomiPeriods;
    bool isThinking;
};

// 胜率标签页
class WinRateTab : public TabPage {
private:
    std::vector<float> winRateHistory;
    std::vector<int> moveNumbers;

public:
    WinRateTab();
    
    void AddWinRate(float rate, int moveNum);
    void Clear();
    
    void Render(UIRenderer* renderer, float x, float y, float w, float h) override;
};

// 记录标签页
class RecordTab : public TabPage {
private:
    std::vector<std::wstring> moveRecords;
    int scrollOffset;
    int selectedIndex;

public:
    RecordTab();
    
    void AddMove(const std::wstring& record);
    void Clear();
    void SetSelected(int index);
    
    void Render(UIRenderer* renderer, float x, float y, float w, float h) override;
};

// 分析标签页
class AnalysisTab : public TabPage {
private:
    std::vector<float> attentionHeatmap;
    std::wstring explanationText;
    float miniBoardSize;

public:
    AnalysisTab();
    
    void SetHeatmap(const std::vector<float>& probabilities);
    void SetExplanation(const std::wstring& text);
    
    void Render(UIRenderer* renderer, float x, float y, float w, float h) override;
};

// 对话标签页
class ChatTab : public TabPage {
private:
    std::vector<std::pair<std::wstring, bool>> messages;
    std::wstring inputText;
    bool isInputFocused;

public:
    ChatTab();
    
    void AddMessage(const std::wstring& text, bool isUser);
    void Clear();
    
    void Render(UIRenderer* renderer, float x, float y, float w, float h) override;
};

// 右侧边栏组件
class RightSidebar {
private:
    float x, y, width, height;
    PlayerInfo blackPlayer;
    PlayerInfo whitePlayer;
    TabControl* middleTabs;
    TabControl* bottomTabs;
    WinRateTab* winRateTab;
    RecordTab* recordTab;
    AnalysisTab* analysisTab;
    ChatTab* chatTab;

public:
    RightSidebar(float px, float py, float w, float h);
    ~RightSidebar();
    
    // 玩家信息
    void SetBlackPlayer(const PlayerInfo& info);
    void SetWhitePlayer(const PlayerInfo& info);
    PlayerInfo GetBlackPlayer() const;
    PlayerInfo GetWhitePlayer() const;
    
    // 标签页访问
    WinRateTab* GetWinRateTab() { return winRateTab; }
    RecordTab* GetRecordTab() { return recordTab; }
    AnalysisTab* GetAnalysisTab() { return analysisTab; }
    ChatTab* GetChatTab() { return chatTab; }
    
    // 渲染
    void Render(UIRenderer* renderer);
};

} // namespace GUI
} // namespace SpaceGo
```

---

## 七、主窗口设计

### 7.1 主窗口 (main_window.h)

```cpp
#pragma once
#include "core/window.h"
#include "core/animation.h"
#include "components/board.h"
#include "components/sidebar.h"
#include "components/topbar.h"
#include "widgets/button.h"
#include "utils/color.h"
#include <vector>

namespace SpaceGo {
namespace GUI {

// 主窗口
class MainWindow : public Window {
private:
    // 组件
    BoardComponent* board;
    RightSidebar* rightSidebar;
    TopBar* topBar;
    ParticleSystem* particles;
    
    // 左侧工具栏按钮
    std::vector<Button*> leftToolbarButtons;
    
    // 动画状态
    bool showParticles;
    bool isTransitioning;
    float transitionProgress;
    
    // 布局
    float boardMargin;
    float sidebarWidth;
    float topbarHeight;
    float toolbarWidth;

public:
    MainWindow(HINSTANCE hInstance);
    virtual ~MainWindow();
    
    // 初始化
    bool Initialize();
    
    // 粒子系统
    void SetParticlesEnabled(bool enabled);
    bool IsParticlesEnabled() const;
    
    // 获取组件
    BoardComponent* GetBoard() { return board; }
    RightSidebar* GetSidebar() { return rightSidebar; }
    
    // 转场动画
    void StartTransition();
    
protected:
    LRESULT HandleMessage(UINT msg, WPARAM wParam, LPARAM lParam) override;
    void OnPaint(Gdiplus::Graphics* graphics) override;
    
    void OnResize(int newWidth, int newHeight);
    void OnMouseMove(float x, float y);
    void OnMouseDown(float x, float y, int button);
    void OnMouseUp(float x, float y, int button);
    
    void UpdateLayout();
    void CreateLeftToolbar();
};

} // namespace GUI
} // namespace SpaceGo
```

---

## 八、渲染实现规范

### 8.1 棋盘渲染实现

```cpp
// 在 BoardRenderer 中实现
void BoardRenderer::DrawBoard(float boardSize, int gridSize) {
    if (!graphics) return;
    
    ColorManager* cm = ColorManager::GetInstance();
    const ColorScheme& colors = cm->GetColors();
    
    float cellSize = boardSize / gridSize;
    float margin = cellSize * 0.8f;
    float drawSize = boardSize - margin * 2;
    float step = drawSize / (gridSize - 1);
    
    // 1. 绘制棋盘背景
    Color boardBg = Color::FromRGB(219, 178, 107);  // 木质色
    DrawRect(boardBg, boardBg, margin, margin, drawSize, drawSize, 4.0f);
    
    // 2. 绘制网格线
    Color gridColor = Color::FromRGB(80, 60, 40);
    for (int i = 0; i < gridSize; i++) {
        float pos = margin + i * step;
        
        // 水平线
        DrawLine(gridColor, margin, pos, margin + drawSize, pos, 1.0f);
        
        // 垂直线
        DrawLine(gridColor, pos, margin, pos, margin + drawSize, 1.0f);
    }
    
    // 3. 绘制星位
    std::vector<std::pair<int, int>> starPoints;
    if (gridSize == 19) {
        starPoints = {{3,3}, {9,3}, {15,3}, {3,9}, {9,9}, {15,9}, {3,15}, {9,15}, {15,15}};
    } else if (gridSize == 13) {
        starPoints = {{3,3}, {9,3}, {6,6}, {3,9}, {9,9}};
    } else if (gridSize == 9) {
        starPoints = {{2,2}, {6,2}, {4,4}, {2,6}, {6,6}};
    }
    
    for (auto& sp : starPoints) {
        float x = margin + sp.first * step;
        float y = margin + sp.second * step;
        DrawCircle(gridColor, gridColor, x, y, 3.5f);
    }
}

void BoardRenderer::DrawStone(float x, float y, bool isBlack, float size, float offsetX, float offsetY) {
    if (!graphics) return;
    
    float sx = x + offsetX;
    float sy = y + offsetY;
    
    // 伦勃朗光源 - 右上45度
    float lightX = sx + size * 0.3f;
    float lightY = sy - size * 0.3f;
    
    if (isBlack) {
        // 黑子 - 深灰渐变，带高光
        Gdiplus::GraphicsPath path;
        path.AddEllipse(sx - size, sy - size, size * 2, size * 2);
        
        Gdiplus::RadialGradientBrush brush(
            Gdiplus::PointF(lightX, lightY),
            Gdiplus::Color(255, 80, 80, 85),
            Gdiplus::Color(255, 20, 20, 25)
        );
        
        graphics->FillEllipse(&brush, sx - size, sy - size, size * 2, size * 2);
        
        // 高光
        Gdiplus::RadialGradientBrush highlightBrush(
            Gdiplus::PointF(lightX, lightY),
            Gdiplus::Color(150, 120, 120, 130),
            Gdiplus::Color(0, 80, 80, 85)
        );
        graphics->FillEllipse(&highlightBrush, sx - size * 0.8f, sy - size * 0.8f, size * 0.8f, size * 0.8f);
        
    } else {
        // 白子 - 灰白渐变
        Gdiplus::RadialGradientBrush brush(
            Gdiplus::PointF(lightX, lightY),
            Gdiplus::Color(255, 255, 255, 255),
            Gdiplus::Color(255, 210, 210, 215)
        );
        
        graphics->FillEllipse(&brush, sx - size, sy - size, size * 2, size * 2);
        
        // 阴影边缘
        Gdiplus::Pen pen(Gdiplus::Color(80, 150, 150, 150), 1.5f);
        graphics->DrawEllipse(&pen, sx - size, sy - size, size * 2, size * 2);
    }
}
```

### 8.2 毛玻璃效果实现

```cpp
void Renderer::DrawGlassEffect(float x, float y, float w, float h, float opacity) {
    if (!graphics) return;
    
    // 半透明背景
    Color glassColor = Color::FromRGB(255, 255, 255).WithAlpha(opacity * 0.3f);
    DrawRect(glassColor, glassColor, x, y, w, h, 16.0f);
    
    // 边框高光
    Color borderColor = Color::FromRGB(255, 255, 255).WithAlpha(opacity * 0.5f);
    Gdiplus::Pen pen(borderColor.ToGdiplusColor(), 1.0f);
    
    Gdiplus::GraphicsPath path;
    path.AddArc(x, y, 32, 32, 180, 90);
    path.AddArc(x + w - 32, y, 32, 32, 270, 90);
    path.AddArc(x + w - 32, y + h - 32, 32, 32, 0, 90);
    path.AddArc(x, y + h - 32, 32, 32, 90, 90);
    path.CloseFigure();
    
    graphics->DrawPath(&pen, &path);
}
```

---

## 九、性能优化规范

### 9.1 渲染优化

1. **双缓冲渲染**
   - 使用 `WS_EX_COMPOSITED` 或手动双缓冲
   - 避免频繁重绘

2. **无效区域优化**
   - 只重绘变化的区域
   - 使用 `InvalidateRect` 指定更新区域

3. **资源缓存**
   - 缓存字体、画笔、画刷
   - 预计算棋盘坐标

### 9.2 动画优化

1. **帧率控制**
   - 目标帧率：60fps
   - 使用高精度定时器（`timeBeginPeriod`）

2. **粒子系统优化**
   - 粒子数量限制：100-200个
   - 空间分区检测

3. **过渡动画**
   - 最大时间：200ms
   - 使用 `EaseOut` 缓动函数

---

## 十、编码规范

### 10.1 命名规范

- **类名**：大驼峰命名法 (PascalCase) - `MainWindow`, `Button`
- **函数名**：大驼峰命名法 - `OnPaint`, `SetPosition`
- **变量名**：小驼峰命名法 (camelCase) - `isHovered`, `mouseX`
- **成员变量**：小驼峰，无特殊前缀 - `width`, `height`
- **常量**：全大写下划线分隔 - `MAX_PARTICLES`, `DEFAULT_FONT`
- **命名空间**：全小写 - `spacego::gui`

### 10.2 注释规范

```cpp
// ============================================================================
// 类名：Button
// 描述：圆角按钮控件，支持悬停和按下状态
// 作者：SpaceGo Team
// ============================================================================
class Button {
public:
    // 创建按钮
    // 参数：
    //   x, y - 按钮位置
    //   w, h - 按钮大小
    //   text - 按钮文本
    Button(float x, float y, float w, float h, const std::wstring& text = L"");
    
    // 设置点击回调函数
    void SetOnClick(std::function<void()> callback);
    
private:
    float x, y;              // 按钮位置
    float width, height;      // 按钮大小
    bool isHovered;           // 鼠标悬停状态
};
```

### 10.3 文件头规范

每个文件开头必须包含：

```cpp
// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：[文件名]
// 描述：[文件描述]
// 创建日期：YYYY-MM-DD
// 作者：SpaceGo Team
// ============================================================================

#pragma once

// 包含头文件...

namespace SpaceGo {
namespace GUI {

// 代码...

} // namespace GUI
} // namespace SpaceGo
```

---

## 十一、实现检查清单

### 11.1 核心功能

- [ ] 启动窗口与初始化动画
- [ ] 主窗口布局与渲染
- [ ] 棋盘绘制与交互
- [ ] 棋子落子动画与震动效果
- [ ] 右侧信息面板
- [ ] 标签页系统
- [ ] 新对局对话框

### 11.2 视觉效果

- [ ] 毛玻璃半透明效果
- [ ] 圆角矩形控件
- [ ] 伦勃朗光源棋子
- [ ] 粒子系统背景
- [ ] 转场动画（≤200ms）
- [ ] 三种色彩方案

### 11.3 技术要求

- [ ] 无第三方库依赖
- [ ] 仅使用Win32 API和GDI+
- [ ] UTF-8编码支持中文
- [ ] 60fps流畅渲染
- [ ] 详细中文注释

---

## 十二、启动窗口设计

### 12.1 启动窗口 (components/splash_window.h)

```cpp
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
    Animation* progressAnimation;
    Animation* scaleAnimation;

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

protected:
    LRESULT HandleMessage(UINT msg, WPARAM wParam, LPARAM lParam) override;
    void OnPaint(Gdiplus::Graphics* graphics) override;
    void RenderContent(UIRenderer* renderer);
};

} // namespace GUI
} // namespace SpaceGo
```

### 12.2 启动窗口实现

```cpp
void SplashWindow::RenderContent(UIRenderer* renderer) {
    if (!renderer) return;

    ColorManager* cm = ColorManager::GetInstance();
    const ColorScheme& colors = cm->GetColors();

    float centerX = width / 2.0f;
    float centerY = height / 2.0f;

    // 1. 绘制背景
    renderer->DrawRect(colors.background, colors.background, 0, 0, width, height);

    // 2. 绘制 Logo（缩放动画
    float logoSize = 120.0f * logoScale;
    Color logoColor = colors.primaryNormal;

    // 绘制标题
    std::wstring title = L"SpaceGo | 维空弈";
    float titleY = centerY - 80.0f;
    renderer->DrawText(title, colors.textPrimary, centerX, titleY, 36.0f);

    // 绘制副标题
    std::wstring subtitle = L"围棋 AI 对弈系统";
    renderer->DrawText(subtitle, colors.textSecondary, centerX, titleY + 50.0f, 18.0f);

    // 3. 绘制进度条
    float progressBarWidth = 300.0f;
    float progressBarHeight = 6.0f;
    float progressBarX = centerX - progressBarWidth / 2.0f;
    float progressBarY = centerY + 60.0f;

    // 进度条背景
    renderer->DrawRect(colors.neutralLight, colors.neutralLight, progressBarX, progressBarY, progressBarWidth, progressBarHeight, 3.0f);

    // 进度条填充
    float fillWidth = progressBarWidth * progress;
    renderer->DrawRect(colors.primaryNormal, colors.primaryNormal, progressBarX, progressBarY, fillWidth, progressBarHeight, 3.0f);

    // 4. 绘制状态文本
    if (!currentStatus.empty()) {
        float statusY = progressBarY + 30.0f;
        renderer->DrawText(currentStatus, colors.textSecondary, centerX, statusY, 14.0f);
    }

    // 5. 绘制版本信息
    std::wstring version = L"版本 1.0.0 | SpaceGo Team";
    renderer->DrawText(version, colors.textDisabled, centerX, height - 40.0f, 12.0f);
}
```

---

## 十三、顶栏组件设计

### 13.1 顶栏组件 (components/topbar.h)

```cpp
#pragma once
#include "core/renderer.h"
#include "widgets/button.h"
#include "utils/color.h"
#include <vector>
#include <functional>
#include <string>

namespace SpaceGo {
namespace GUI {

// 菜单项
struct MenuItem {
    std::wstring title;
    int id;
    bool isSeparator;
    std::vector<MenuItem> children;
};

// 顶栏组件
class TopBar {
private:
    float x, y, width, height;
    std::vector<MenuItem> mainMenu;
    int hoveredMenuIndex;
    int activeMenuIndex;
    bool isMenuOpen;
    std::function<void(int)> onMenuClicked;
    float menuItemHeight;

public:
    TopBar(float px, float py, float w, float h);

    // 添加菜单项
    void AddMenuItem(const std::wstring& title, int id);
    void AddMenuSeparator();
    void AddSubMenuItem(int parentId, const std::wstring& title, int id);

    // 事件
    void SetOnMenuClicked(std::function<void(int)> callback);

    // 交互
    void SetHoveredMenu(float mx, float my);
    bool HitTest(float mx, float my, int& menuId);
    void OnMenuClick(int menuId);

    // 渲染
    void Render(UIRenderer* renderer);
};

} // namespace GUI
} // namespace SpaceGo
```

### 13.2 顶栏菜单项定义

```cpp
// 标准菜单结构
//
// ┌─────────────────────────────────────────────────────────┐
// │ 对局  棋谱  练习  设置  智能体  风格迁移  │
// └─────────────────────────────────────────────────────────┘
//
// 对局菜单：
//   - 打开对局
//   - 新开对局
//   ─────────
//   - 保存对局
//   - 导出对局
//   ─────────
//   - 退出

// 初始化标准菜单
void TopBar::InitializeStandardMenu() {
    // 对局菜单
    AddMenuItem(L"对局", 100);
    AddSubMenuItem(100, L"打开对局", 101);
    AddSubMenuItem(100, L"新开对局", 102);
    AddMenuSeparator();
    AddSubMenuItem(100, L"保存对局", 103);
    AddSubMenuItem(100, L"导出对局", 104);
    AddMenuSeparator();
    AddSubMenuItem(100, L"退出", 105);

    // 棋谱菜单
    AddMenuItem(L"棋谱", 200);
    AddSubMenuItem(200, L"新建棋谱", 201);
    AddSubMenuItem(200, L"打开棋谱", 202);
    AddSubMenuItem(200, L"保存棋谱", 203);

    // 练习菜单
    AddMenuItem(L"练习", 300);

    // 设置菜单
    AddMenuItem(L"设置", 400);
    AddSubMenuItem(400, L"界面设置", 401);
    AddSubMenuItem(400, L"AI 设置", 402);
    AddSubMenuItem(400, L"主题设置", 403);

    // 智能体菜单
    AddMenuItem(L"智能体", 500);
    AddSubMenuItem(500, L"管理智能体", 501);
    AddSubMenuItem(500, L"创建智能体", 502);

    // 风格迁移菜单
    AddMenuItem(L"风格迁移", 600);
}
```

---

## 十四、输入处理规范

### 14.1 鼠标输入处理

```cpp
// 在 Window 基类中实现
LRESULT Window::HandleMessage(UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_MOUSEMOVE: {
            float x = (float)(LOWORD(lParam));
            float y = (float)(HIWORD(lParam));
            OnMouseMove(x, y);
            return 0;
        }

        case WM_LBUTTONDOWN: {
            float x = (float)(LOWORD(lParam));
            float y = (float)(HIWORD(lParam));
            OnMouseDown(x, y, 1);
            return 0;
        }

        case WM_LBUTTONUP: {
            float x = (float)(LOWORD(lParam));
            float y = (float)(HIWORD(lParam));
            OnMouseUp(x, y, 1);
            return 0;
        }

        case WM_RBUTTONDOWN: {
            float x = (float)(LOWORD(lParam));
            float y = (float)(HIWORD(lParam));
            OnMouseDown(x, y, 2);
            return 0;
        }

        case WM_RBUTTONUP: {
            float x = (float)(LOWORD(lParam));
            float y = (float)(HIWORD(lParam));
            OnMouseUp(x, y, 2);
            return 0;
        }

        case WM_MOUSEWHEEL: {
            int delta = GET_WHEEL_DELTA_WPARAM(wParam);
            OnMouseWheel((float)delta / 120.0f);
            return 0;
        }

        case WM_SETCURSOR: {
            SetCursor(LoadCursor(NULL, IDC_ARROW));
            return TRUE;
        }

        default:
            return DefWindowProc(hwnd, msg, wParam, lParam);
    }
}
```

### 14.2 键盘输入处理

```cpp
// 键盘快捷键定义
//
// 游戏控制：
//   - Esc - 取消操作
//   - Ctrl+Z - 悔棋
//   - Ctrl+Y - 重做
//   - Ctrl+S - 保存
//   - F1 - 帮助
//   - 空格 - 暂停/继续

LRESULT Window::HandleMessage(UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_KEYDOWN: {
            bool ctrl = (GetKeyState(VK_CONTROL) & 0x8000) != 0;
            bool shift = (GetKeyState(VK_SHIFT) & 0x8000) != 0;

            switch (wParam) {
                case VK_ESCAPE:
                    OnEscapeKey();
                    break;
                case 'Z':
                    if (ctrl) OnUndo();
                    break;
                case 'Y':
                    if (ctrl) OnRedo();
                    break;
                case 'S':
                    if (ctrl) OnSave();
                    break;
                case VK_F1:
                    OnHelp();
                    break;
                case VK_SPACE:
                    OnSpaceKey();
                    break;
                case VK_LEFT:
                    OnLeftArrow();
                    break;
                case VK_RIGHT:
                    OnRightArrow();
                    break;
            }
            return 0;
        }

        case WM_CHAR: {
            OnCharInput((wchar_t)wParam);
            return 0;
        }
    }
}
```

---

## 十五、工具模块设计

### 15.1 字体管理 (utils/font.h)

```cpp
#pragma once
#include <gdiplus.h>
#include <string>
#include <map>

namespace SpaceGo {
namespace GUI {

// 字体管理器
class FontManager {
private:
    std::map<std::wstring, Gdiplus::Font*> fontCache;
    static FontManager* instance;

    FontManager();

public:
    static FontManager* GetInstance();

    // 获取字体
    Gdiplus::Font* GetFont(const std::wstring& familyName, float size, 
                             bool bold = false, bool italic = false);

    // 释放缓存
    void ClearCache();

    // 默认字体
    static const std::wstring DEFAULT_FONT_FAMILY;
    static const float DEFAULT_FONT_SIZE;
};

} // namespace GUI
} // namespace SpaceGo
```

### 15.2 数学工具 (utils/math.h)

```cpp
#pragma once
#include <cmath>

namespace SpaceGo {
namespace GUI {

namespace Math {

    // 常量
    const float PI = 3.14159265358979323846f;
    const float DEG_TO_RAD = PI / 180.0f;
    const float RAD_TO_DEG = 180.0f / PI;

    // 插值
    inline float Lerp(float a, float b, float t) {
        return a + (b - a) * t;
    }

    // 平滑插值 (Smoothstep)
    inline float Smoothstep(float edge0, float edge1, float x) {
        float t = Clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
        return t * t * (3.0f - 2.0f * t);
    }

    // 限制范围
    inline float Clamp(float value, float min, float max) {
        return value < min ? min : (value > max ? max : value);
    }

    // 距离计算
    inline float Distance(float x1, float y1, float x2, float y2) {
        float dx = x2 - x1;
        float dy = y2 - y1;
        return sqrtf(dx * dx + dy * dy);
    }

    // 点是否在矩形内
    inline bool PointInRect(float px, float py, float rx, float ry, float rw, float rh) {
        return px >= rx && px < rx + rw && py >= ry && py < ry + rh;
    }

    // 点是否在圆内
    inline bool PointInCircle(float px, float py, float cx, float cy, float radius) {
        return Distance(px, py, cx, cy) <= radius;
    }

    // 角度转换
    inline float DegToRad(float deg) { return deg * DEG_TO_RAD; }
    inline float RadToDeg(float rad) { return rad * RAD_TO_DEG; }

} // namespace Math
} // namespace GUI
} // namespace SpaceGo
```

---

## 十六、完整实现示例

### 16.1 主程序入口 (main.cpp)

```cpp
// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：main.cpp
// 描述：程序入口点
// 创建日期：2026-02-19
// 作者：SpaceGo Team
// ============================================================================

#include <windows.h>
#include <gdiplus.h>
#include "core/window.h"
#include "components/splash_window.h"
#include "components/main_window.h"
#include "core/animation.h"
#include "utils/color.h"

#pragma comment(lib, "gdiplus.lib")

using namespace SpaceGo::GUI;

// 全局变量
SplashWindow* g_splashWindow = nullptr;
MainWindow* g_mainWindow = nullptr;
bool g_initialized = false;

// GDI+ 初始化
ULONG_PTR g_gdiplusToken;

// 初始化 GDI+
void InitializeGDIPlus() {
    Gdiplus::GdiplusStartupInput input;
    Gdiplus::GdiplusStartup(&g_gdiplusToken, &input, NULL);
}

// 清理 GDI+
void ShutdownGDIPlus() {
    Gdiplus::GdiplusShutdown(g_gdiplusToken);
}

// 初始化过程
void DoInitialization() {
    // 模拟初始化过程
    for (int i = 0; i <= 100; i++) {
        float progress = (float)i / 100.0f;
        g_splashWindow->SetProgress(progress);

        // 更新状态文本
        if (i < 20) {
            g_splashWindow->SetStatusText(L"正在初始化渲染系统...");
        } else if (i < 40) {
            g_splashWindow->SetStatusText(L"正在加载神经网络...");
        } else if (i < 60) {
            g_splashWindow->SetStatusText(L"正在初始化 AI 引擎...");
        } else if (i < 80) {
            g_splashWindow->SetStatusText(L"正在准备界面...");
        } else {
            g_splashWindow->SetStatusText(L"准备就绪");
        }

        Sleep(20);
    }

    g_initialized = true;
}

// WinMain 入口
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    // 初始化 GDI+
    InitializeGDIPlus();

    // 初始化色彩管理器
    ColorManager::GetInstance()->SetScheme(ColorSchemeType::Normal);

    // 创建启动窗口
    g_splashWindow = new SplashWindow(hInstance);
    g_splashWindow->Create();
    g_splashWindow->Show();

    // 执行初始化
    DoInitialization();

    // 淡出启动窗口
    g_splashWindow->StartFadeOut();
    while (!g_splashWindow->IsFadeOutComplete()) {
        Sleep(10);
    }

    // 创建主窗口
    g_mainWindow = new MainWindow(hInstance);
    g_mainWindow->Initialize();
    g_mainWindow->Create();
    g_mainWindow->Show();

    // 关闭启动窗口
    delete g_splashWindow;
    g_splashWindow = nullptr;

    // 消息循环
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    // 清理
    delete g_mainWindow;
    ShutdownGDIPlus();

    return (int)msg.wParam;
}
```

### 16.2 窗口基类完整实现 (src/core/window.cpp)

```cpp
// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：window.cpp
// 描述：窗口基类实现
// 创建日期：2026-02-19
// 作者：SpaceGo Team
// ============================================================================

#include "core/window.h"

namespace SpaceGo {
namespace GUI {

Window::Window(HINSTANCE hInstance, const std::wstring& title, int w, int h)
    : hinstance(hInstance)
    , title(title)
    , width(w)
    , height(h)
    , hwnd(nullptr)
    , isVisible(false)
    , isDraggable(false)
{
}

Window::~Window() {
    if (hwnd) {
        DestroyWindow(hwnd);
    }
}

bool Window::Create() {
    // 注册窗口类
    WNDCLASSEX wc = {0};
    wc.cbSize        = sizeof(WNDCLASSEX);
    wc.style         = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc   = WindowProc;
    wc.hInstance     = hinstance;
    wc.hCursor       = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = L"SpaceGoWindow";

    RegisterClassEx(&wc);

    // 创建窗口
    hwnd = CreateWindowEx(
        WS_EX_COMPOSITED,  // 扩展样式：双缓冲
        L"SpaceGoWindow",
        title.c_str(),
        WS_POPUP | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT,
        width, height,
        NULL, NULL,
        hinstance,
        this
    );

    if (!hwnd) {
        return false;
    }

    // 存储 this 指针
    SetWindowLongPtr(hwnd, GWLP_USERDATA, (LONG_PTR)this);

    return true;
}

void Window::Show() {
    if (hwnd) {
        ShowWindow(hwnd, SW_SHOW);
        isVisible = true;
    }
}

void Window::Hide() {
    if (hwnd) {
        ShowWindow(hwnd, SW_HIDE);
        isVisible = false;
    }
}

void Window::SetPosition(int x, int y) {
    if (hwnd) {
        SetWindowPos(hwnd, NULL, x, y, 0, 0, SWP_NOSIZE);
    }
}

void Window::GetPosition(int& x, int& y) const {
    if (hwnd) {
        RECT rect;
        GetWindowRect(hwnd, &rect);
        x = rect.left;
        y = rect.top;
    }
}

void Window::SetSize(int w, int h) {
    width = w;
    height = h;
    if (hwnd) {
        SetWindowPos(hwnd, NULL, 0, 0, w, h, SWP_NOMOVE);
    }
}

void Window::GetSize(int& w, int& h) const {
    w = width;
    h = height;
}

LRESULT Window::HandleMessage(UINT msg, WPARAM wParam, LPARAM lParam) {
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

void Window::OnPaint(Gdiplus::Graphics* graphics) {
}

LRESULT CALLBACK Window::WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    Window* window = nullptr;

    if (msg == WM_NCCREATE) {
        CREATESTRUCT* cs = (CREATESTRUCT*)lParam;
        window = (Window*)cs->lpCreateParams;
        SetWindowLongPtr(hWnd, GWLP_USERDATA, (LONG_PTR)window;
        window->hwnd = hWnd;
    } else {
        window = (Window*)GetWindowLongPtr(hWnd, GWLP_USERDATA);
    }

    if (window) {
        if (msg == WM_PAINT) {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hWnd, &ps);

            Gdiplus::Graphics graphics(hdc);
            graphics.SetSmoothingMode(Gdiplus::SmoothingModeAntiAlias);
            graphics.SetTextRenderingHint(Gdiplus::TextRenderingHintClearTypeGridFit);

            window->OnPaint(&graphics);

            EndPaint(hWnd, &ps);
            return 0;
        }

        return window->HandleMessage(msg, wParam, lParam);
    }

    return DefWindowProc(hWnd, msg, wParam, lParam);
}

} // namespace GUI
} // namespace SpaceGo
```

---

## 十七、构建系统说明

### 17.1 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.15)
project(SpaceGoGUI)

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 源代码
set(SOURCES
    src/main.cpp
    src/core/window.cpp
    src/core/renderer.cpp
    src/core/animation.cpp
    src/widgets/button.cpp
    src/widgets/tab.cpp
    src/widgets/dialog.cpp
    src/components/board.cpp
    src/components/sidebar.cpp
    src/components/topbar.cpp
    src/components/splash_window.cpp
    src/components/main_window.cpp
    src/utils/color.cpp
    src/utils/font.cpp
)

# 头文件
set(HEADERS
    include/core/window.h
    include/core/renderer.h
    include/core/animation.h
    include/widgets/button.h
    include/widgets/label.h
    include/widgets/tab.h
    include/widgets/dialog.h
    include/components/board.h
    include/components/sidebar.h
    include/components/topbar.h
    include/components/splash_window.h
    include/components/main_window.h
    include/utils/color.h
    include/utils/font.h
    include/utils/math.h
)

# 可执行文件
add_executable(SpaceGoGUI WIN32 ${SOURCES} ${HEADERS})

# 包含目录
target_include_directories(SpaceGoGUI PRIVATE include)

# 链接 GDI+
target_link_libraries(SpaceGoGUI PRIVATE gdiplus)

# 设置输出目录
set_target_properties(SpaceGoGUI PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 安装规则
install(TARGETS SpaceGoGUI RUNTIME DESTINATION bin)
```

### 17.2 Visual Studio 项目配置

```
项目类型：Windows 桌面应用程序 (C++)

配置属性：
- C++ 语言标准：ISO C++17 标准 (/std:c++17)
- 字符集：使用 Unicode 字符集
- 运行库：多线程调试 DLL (/MDd) - Debug 模式
- 运行库：多线程 DLL (/MD) - Release 模式

链接器：
- 附加依赖项：gdiplus.lib;user32.lib;gdi32.lib;comctl32.lib

资源文件：
- 添加应用程序图标 (可选)
```

---

## 十八、调试与测试规范

### 18.1 调试技巧

```cpp
// 调试辅助宏
#ifdef _DEBUG
#define DEBUG_LOG(msg) OutputDebugStringA(msg)
#define DEBUG_LOGW(msg) OutputDebugStringW(msg)
#else
#define DEBUG_LOG(msg)
#define DEBUG_LOGW(msg)
#endif

// 性能计时
class PerformanceTimer {
private:
    LARGE_INTEGER start;
    LARGE_INTEGER frequency;

public:
    PerformanceTimer() {
        QueryPerformanceFrequency(&frequency);
        QueryPerformanceCounter(&start);
    }

    float GetElapsedMs() {
        LARGE_INTEGER end;
        QueryPerformanceCounter(&end);
        return (float)(end.QuadPart - start.QuadPart) * 1000.0f / frequency.QuadPart;
    }

    void Reset() {
        QueryPerformanceCounter(&start);
    }
};

// 使用示例
// PerformanceTimer timer;
// ... 执行代码 ...
// float elapsed = timer.GetElapsedMs();
// DEBUG_LOG(("耗时: " + std::to_string(elapsed) + "ms\n").c_str());
```

### 18.2 单元测试框架

```cpp
// 简单的测试宏
#define TEST_ASSERT(condition, message) \
    if (!(condition)) { \
        OutputDebugStringA(("测试失败: " + std::string(message) + "\n").c_str()); \
        return false; \
    }

// 颜色测试
bool TestColorConversions() {
    Color c = Color::FromRGB(255, 0, 0);
    TEST_ASSERT(c.r == 1.0f, "红色分量错误");
    TEST_ASSERT(c.g == 0.0f, "绿色分量错误");
    TEST_ASSERT(c.b == 0.0f, "蓝色分量错误");

    c.RGBToHSL();
    TEST_ASSERT(c.h == 0.0f, "色相转换错误");

    return true;
}

// 数学工具测试
bool TestMathUtils() {
    TEST_ASSERT(Math::Lerp(0.0f, 10.0f, 0.5f) == 5.0f, "Lerp 错误");
    TEST_ASSERT(Math::Clamp(15.0f, 0.0f, 10.0f) == 10.0f, "Clamp 错误");
    return true;
}
```

---

## 附录：参考资源

- Win32 API 文档：https://docs.microsoft.com/windows/win32/
- GDI+ 文档：https://docs.microsoft.com/windows/win32/gdiplus/
- WCAG 对比度标准：https://www.w3.org/TR/WCAG21/
- 色彩理论：https://en.wikipedia.org/wiki/Color_theory
- CMake 文档：https://cmake.org/documentation/

