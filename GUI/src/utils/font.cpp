// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：font.cpp
// 描述：字体管理实现
// 创建日期：2026-02-20
// 作者：SpaceGo Team
// ============================================================================

#include "utils/font.h"
#pragma comment(lib, "gdiplus.lib")

namespace SpaceGo {
namespace GUI {

// 初始化静态成员
const std::wstring FontManager::DEFAULT_FONT_FAMILY = L"Microsoft YaHei";
const float FontManager::DEFAULT_FONT_SIZE = 12.0f;

// 构造函数
FontManager::FontManager() {
}

// 单例获取 - 使用Meyers' Singleton（线程安全）
FontManager& FontManager::GetInstance() {
    static FontManager instance;
    return instance;
}

// 析构函数
FontManager::~FontManager() {
    ClearCache();
}

// 获取字体
Gdiplus::Font* FontManager::GetFont(const std::wstring& familyName, float size, 
                                      bool bold, bool italic) {
    // 构建缓存键
    std::wstring key = familyName + L"_" + std::to_wstring(size);
    if (bold) key += L"_bold";
    if (italic) key += L"_italic";

    // 检查缓存
    auto it = fontCache.find(key);
    if (it != fontCache.end()) {
        return it->second.get();
    }

    // 创建新字体
    Gdiplus::FontFamily fontFamily(familyName.c_str());
    Gdiplus::FontStyle style = Gdiplus::FontStyleRegular;
    if (bold) style = static_cast<Gdiplus::FontStyle>(style | Gdiplus::FontStyleBold);
    if (italic) style = static_cast<Gdiplus::FontStyle>(style | Gdiplus::FontStyleItalic);

    auto font = std::make_unique<Gdiplus::Font>(&fontFamily, size, style, Gdiplus::UnitPixel);
    Gdiplus::Font* fontPtr = font.get();
    fontCache[key] = std::move(font);
    return fontPtr;
}

// 释放缓存
void FontManager::ClearCache() {
    fontCache.clear();
}

} // namespace GUI
} // namespace SpaceGo
