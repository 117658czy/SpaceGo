// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：main.cpp
// 描述：程序入口
// 创建日期：2026-02-20
// 作者：SpaceGo Team
// ============================================================================

#include <windows.h>
#include <gdiplus.h>
#include <memory>
#include "components/main_window.h"
#include "utils/font.h"
#include "utils/color.h"
#include "core/animation.h"
#include "utils/logger.h"
#pragma comment(lib, "gdiplus.lib")

using namespace SpaceGo::GUI;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    // 初始化日志系统
    GUI_LOG_INFO("SpaceGo GUI \u542f\u52a8\u4e2d...");
    GUI_LOG_INFO("\u521d\u59cb\u5316GDI+...");
    
    // \u521d\u59cb\u5316GDI+
    Gdiplus::GdiplusStartupInput gdiplusStartupInput;
    ULONG_PTR gdiplusToken;
    Gdiplus::Status gdiStatus = Gdiplus::GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);
    
    if (gdiStatus != Gdiplus::Ok) {
        GUI_LOG_ERROR("GDI+\u521d\u59cb\u5316\u5931\u8d25\uff0c\u9519\u8bef\u4ee3\u7801: " + std::to_string(gdiStatus));
        return -1;
    }
    GUI_LOG_INFO("GDI+\u521d\u59cb\u5316\u6210\u529f");

    try {
        GUI_LOG_INFO("\u521b\u5efa\u4e3b\u7a97\u53e3...");
        // \u521b\u5efa\u4e3b\u7a97\u53e3
        auto mainWindow = std::make_unique<MainWindow>();
        GUI_LOG_INFO("\u4e3b\u7a97\u53e3\u521b\u5efa\u6210\u529f");
        
        GUI_LOG_INFO("\u663e\u793a\u7a97\u53e3\u5e76\u8fd0\u884c\u6d88\u606f\u5faa\u73af...");
        // \u663e\u793a\u7a97\u53e3\u5e76\u8fd0\u884c\u6d88\u606f\u5faa\u73af
        mainWindow->Show();
        GUI_LOG_INFO("\u7a97\u53e3\u5df2\u663e\u793a\uff0c\u8fdb\u5165\u6d88\u606f\u5faa\u73af");
        
        int result = mainWindow->Run();
        GUI_LOG_INFO("\u6d88\u606f\u5faa\u73af\u9000\u51fa\uff0c\u8fd4\u56de\u503c: " + std::to_string(result));

        // \u5173\u95edGDI+
        GUI_LOG_INFO("\u5173\u95edGDI+...");
        Gdiplus::GdiplusShutdown(gdiplusToken);
        GUI_LOG_INFO("GDI+\u5df2\u5173\u95ed");

        GUI_LOG_INFO("SpaceGo GUI \u6b63\u5e38\u9000\u51fa");
        return result;
        
    } catch (const std::exception& e) {
        GUI_LOG_EXCEPTION("\u6355\u83b7\u5230\u5f02\u5e38: " + std::string(e.what()));
        Gdiplus::GdiplusShutdown(gdiplusToken);
        return -1;
    } catch (...) {
        GUI_LOG_EXCEPTION("\u6355\u83b7\u5230\u672a\u77e5\u5f02\u5e38");
        Gdiplus::GdiplusShutdown(gdiplusToken);
        return -1;
    }
}
