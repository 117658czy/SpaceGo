// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：font.h
// 描述：字体管理
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
#include <map>
#include <memory>

namespace SpaceGo {
namespace GUI {

// 字体管理器
class FontManager {
private:
    std::map<std::wstring, std::unique_ptr<Gdiplus::Font>> fontCache;

    FontManager();

public:
    static FontManager& GetInstance();
    ~FontManager();

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
