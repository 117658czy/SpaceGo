// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：window.h
// 描述：窗口基类
// 创建日期：2026-02-20
// 作者：SpaceGo Team
// ============================================================================

#pragma once

// Windows头文件配置
#ifndef NOMINMAX
#define NOMINMAX
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#ifndef UNICODE
#define UNICODE
#endif

#ifndef _UNICODE
#define _UNICODE
#endif

// Windows头文件包含顺序非常重要
#include <windows.h>
#include <objbase.h>
#include <wingdi.h>
#include <gdiplus.h>

// GDI+库链接
#pragma comment(lib, "gdiplus.lib")
#pragma comment(lib, "ole32.lib")

#include <string>
#include <vector>
#include <functional>

namespace SpaceGo {
namespace GUI {

// 窗口基类
class Window {
protected:
    HWND hwnd;
    HINSTANCE hinstance;
    std::wstring title;
    int width, height;
    int windowWidth, windowHeight;
    bool isVisible;
    bool isDraggable;
    POINT dragOffset;

public:
    Window(const std::wstring& title, int w, int h);
    Window(HINSTANCE hInstance, const std::wstring& title, int w, int h);
    virtual ~Window();

    // 创建窗口
    virtual bool Create();

    // 窗口创建完成时调用（子类重写）
    virtual void OnCreate();

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

    // 更新动画（子类重写）
    virtual void OnUpdate(float deltaTime);

    // 输入事件（子类重写）
    virtual void OnMouseMove(float x, float y);
    virtual void OnMouseDown(float x, float y);
    virtual void OnMouseDown(float x, float y, int button);
    virtual void OnMouseUp(float x, float y);
    virtual void OnMouseUp(float x, float y, int button);
    virtual void OnMouseWheel(float delta);
    virtual void OnKeyDown(int key);
    virtual void OnKeyUp(int key);
    virtual void OnCharInput(wchar_t ch);
    virtual void OnEscapeKey();
    virtual void OnUndo();
    virtual void OnRedo();
    virtual void OnSave();
    virtual void OnHelp();
    virtual void OnSpaceKey();
    virtual void OnLeftArrow();
    virtual void OnRightArrow();
    virtual void OnResize(int newWidth, int newHeight);

    // 运行消息循环
    int Run();

    // 刷新窗口
    void Invalidate();

protected:
    // 窗口过程回调
    static LRESULT CALLBACK WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
};

} // namespace GUI
} // namespace SpaceGo
