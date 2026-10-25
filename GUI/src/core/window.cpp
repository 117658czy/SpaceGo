// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：window.cpp
// 描述：窗口基类实现
// 创建日期：2026-02-20
// 作者：SpaceGo Team
// ============================================================================

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "core/window.h"
#include "utils/math.h"

namespace SpaceGo {
namespace GUI {

Window::Window(const std::wstring& title, int w, int h)
    : hinstance(GetModuleHandle(NULL))
    , title(title)
    , width(w)
    , height(h)
    , windowWidth(w)
    , windowHeight(h)
    , hwnd(nullptr)
    , isVisible(false)
    , isDraggable(false)
{
}

Window::Window(HINSTANCE hInstance, const std::wstring& title, int w, int h)
    : hinstance(hInstance)
    , title(title)
    , width(w)
    , height(h)
    , windowWidth(w)
    , windowHeight(h)
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
    WNDCLASSEXW wc = {0};
    wc.cbSize        = sizeof(WNDCLASSEXW);
    wc.style         = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc   = WindowProc;
    wc.hInstance     = hinstance;
    wc.hCursor       = LoadCursorW(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = L"SpaceGoWindow";

    if (!RegisterClassExW(&wc)) {
        return false;
    }

    hwnd = CreateWindowExW(
        0,
        L"SpaceGoWindow",
        title.c_str(),
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        width, height,
        NULL, NULL,
        hinstance,
        this
    );

    if (!hwnd) {
        return false;
    }

    return true;
}

void Window::Show() {
    if (hwnd) {
        ShowWindow(hwnd, SW_SHOW);
        UpdateWindow(hwnd);
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

int Window::Run() {
    if (!hwnd) {
        if (!Create()) {
            return -1;
        }
    }

    // 初始化时间
    LARGE_INTEGER frequency, lastTime;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&lastTime);

    MSG msg;
    while (true) {
        // 使用PeekMessage来处理消息，这样可以在消息间隙更新动画
        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                return static_cast<int>(msg.wParam);
            }
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        // 计算delta time
        LARGE_INTEGER currentTime;
        QueryPerformanceCounter(&currentTime);
        float deltaTime = static_cast<float>(currentTime.QuadPart - lastTime.QuadPart) / 
                          static_cast<float>(frequency.QuadPart);
        lastTime = currentTime;

        // 限制delta time，避免跳帧
        deltaTime = Math::Min(deltaTime, 0.1f);

        // 更新动画
        OnUpdate(deltaTime);

        // 小休眠，降低CPU占用
        Sleep(1);
    }
}

void Window::Invalidate() {
    if (hwnd) {
        ::InvalidateRect(hwnd, NULL, FALSE);
    }
}

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
            OnMouseDown(x, y);
            OnMouseDown(x, y, 1);
            return 0;
        }

        case WM_LBUTTONUP: {
            float x = (float)(LOWORD(lParam));
            float y = (float)(HIWORD(lParam));
            OnMouseUp(x, y);
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

        case WM_KEYDOWN: {
            OnKeyDown((int)wParam);

            bool ctrl = (GetKeyState(VK_CONTROL) & 0x8000) != 0;

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

        case WM_KEYUP: {
            OnKeyUp((int)wParam);
            return 0;
        }

        case WM_CHAR: {
            OnCharInput((wchar_t)wParam);
            return 0;
        }

        case WM_SIZE: {
            int newWidth = LOWORD(lParam);
            int newHeight = HIWORD(lParam);
            windowWidth = newWidth;
            windowHeight = newHeight;
            OnResize(newWidth, newHeight);
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

void Window::OnPaint(Gdiplus::Graphics* graphics) {
}

void Window::OnUpdate(float deltaTime) {
}

void Window::OnCreate() {
}

void Window::OnMouseMove(float x, float y) {
}

void Window::OnMouseDown(float x, float y) {
}

void Window::OnMouseDown(float x, float y, int button) {
}

void Window::OnMouseUp(float x, float y) {
}

void Window::OnMouseUp(float x, float y, int button) {
}

void Window::OnMouseWheel(float delta) {
}

void Window::OnKeyDown(int key) {
}

void Window::OnKeyUp(int key) {
}

void Window::OnCharInput(wchar_t ch) {
}

void Window::OnEscapeKey() {
}

void Window::OnUndo() {
}

void Window::OnRedo() {
}

void Window::OnSave() {
}

void Window::OnHelp() {
}

void Window::OnSpaceKey() {
}

void Window::OnLeftArrow() {
}

void Window::OnRightArrow() {
}

void Window::OnResize(int newWidth, int newHeight) {
}

LRESULT CALLBACK Window::WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    Window* window = nullptr;

    if (msg == WM_NCCREATE) {
        CREATESTRUCT* cs = (CREATESTRUCT*)lParam;
        window = (Window*)cs->lpCreateParams;
        SetWindowLongPtr(hWnd, GWLP_USERDATA, (LONG_PTR)window);
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
