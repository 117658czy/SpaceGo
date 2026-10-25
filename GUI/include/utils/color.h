// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：color.h
// 描述：色彩系统
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

    ColorManager();

public:
    static ColorManager& GetInstance();
    
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
