// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：color.cpp
// 描述：色彩系统实现
// 创建日期：2026-02-20
// 作者：SpaceGo Team
// ============================================================================

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "utils/color.h"
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace SpaceGo {
namespace GUI {

// Color 类实现
Color::Color() : r(0), g(0), b(0), a(1.0f), h(0), s(0), l(0) {}

Color::Color(float red, float green, float blue, float alpha) 
    : r(red), g(green), b(blue), a(alpha), h(0), s(0), l(0) {
    RGBToHSL();
}

Color Color::FromRGB(int r, int g, int b, int a) {
    return Color(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f);
}

Color Color::FromRGB(float r, float g, float b, float a) {
    return Color(r, g, b, a);
}

Color Color::FromHSL(float h, float s, float l, float a) {
    Color c;
    c.h = h;
    c.s = s;
    c.l = l;
    c.a = a;
    c.HSLToRGB();
    return c;
}

void Color::RGBToHSL() {
    float max = std::max({r, g, b});
    float min = std::min({r, g, b});
    float delta = max - min;

    l = (max + min) / 2.0f;

    if (delta < 0.0001f) {
        h = 0;
        s = 0;
    } else {
        s = l > 0.5f ? delta / (2.0f - max - min) : delta / (max + min);

        if (max == r) {
            h = (g - b) / delta + (g < b ? 6.0f : 0.0f);
        } else if (max == g) {
            h = (b - r) / delta + 2.0f;
        } else {
            h = (r - g) / delta + 4.0f;
        }
        h *= 60.0f;
    }
}

void Color::HSLToRGB() {
    if (s < 0.0001f) {
        r = g = b = l;
    } else {
        auto hue2rgb = [](float p, float q, float t) {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1.0f/6.0f) return p + (q - p) * 6.0f * t;
            if (t < 1.0f/2.0f) return q;
            if (t < 2.0f/3.0f) return p + (q - p) * (2.0f/3.0f - t) * 6.0f;
            return p;
        };

        float q = l < 0.5f ? l * (1 + s) : l + s - l * s;
        float p = 2 * l - q;
        float hNorm = h / 360.0f;

        r = hue2rgb(p, q, hNorm + 1.0f/3.0f);
        g = hue2rgb(p, q, hNorm);
        b = hue2rgb(p, q, hNorm - 1.0f/3.0f);
    }
}

Gdiplus::Color Color::ToGdiplusColor() const {
    return Gdiplus::Color(
        (BYTE)(a * 255),
        (BYTE)(r * 255),
        (BYTE)(g * 255),
        (BYTE)(b * 255)
    );
}

Color Color::Lerp(const Color& other, float t) const {
    return Color(
        r + (other.r - r) * t,
        g + (other.g - g) * t,
        b + (other.b - b) * t,
        a + (other.a - a) * t
    );
}

Color Color::Lighten(float amount) const {
    Color c = *this;
    c.l = std::min(1.0f, c.l + amount);
    c.HSLToRGB();
    return c;
}

Color Color::Darken(float amount) const {
    Color c = *this;
    c.l = std::max(0.0f, c.l - amount);
    c.HSLToRGB();
    return c;
}

Color Color::Saturate(float amount) const {
    Color c = *this;
    c.s = std::min(1.0f, c.s + amount);
    c.HSLToRGB();
    return c;
}

Color Color::Desaturate(float amount) const {
    Color c = *this;
    c.s = std::max(0.0f, c.s - amount);
    c.HSLToRGB();
    return c;
}

Color Color::WithAlpha(float alpha) const {
    Color c = *this;
    c.a = alpha;
    return c;
}

float Color::GetContrast(const Color& other) const {
    auto luminance = [](const Color& c) {
        float r = c.r <= 0.03928f ? c.r / 12.92f : powf((c.r + 0.055f) / 1.055f, 2.4f);
        float g = c.g <= 0.03928f ? c.g / 12.92f : powf((c.g + 0.055f) / 1.055f, 2.4f);
        float b = c.b <= 0.03928f ? c.b / 12.92f : powf((c.b + 0.055f) / 1.055f, 2.4f);
        return 0.2126f * r + 0.7152f * g + 0.0722f * b;
    };

    float l1 = luminance(*this);
    float l2 = luminance(other);
    float lighter = std::max(l1, l2);
    float darker = std::min(l1, l2);
    return (lighter + 0.05f) / (darker + 0.05f);
}

std::string Color::ToHexString() const {
    std::stringstream ss;
    ss << "#"
       << std::hex << std::setfill('0') << std::setw(2) << (int)(r * 255)
       << std::hex << std::setfill('0') << std::setw(2) << (int)(g * 255)
       << std::hex << std::setfill('0') << std::setw(2) << (int)(b * 255);
    return ss.str();
}

// ColorManager 实现
ColorManager::ColorManager() : currentScheme(ColorSchemeType::Normal) {
    currentColors = GenerateNormalScheme();
}

ColorManager& ColorManager::GetInstance() {
    static ColorManager instance;
    return instance;
}

void ColorManager::SetScheme(ColorSchemeType type) {
    currentScheme = type;
    switch (type) {
        case ColorSchemeType::Dark:
            currentColors = GenerateDarkScheme();
            break;
        case ColorSchemeType::Light:
            currentColors = GenerateLightScheme();
            break;
        case ColorSchemeType::Normal:
        default:
            currentColors = GenerateNormalScheme();
            break;
    }
}

void ColorManager::SetCustomBaseColor(const Color& baseColor) {
    customBaseColor = baseColor;
    currentScheme = ColorSchemeType::Custom;
    currentColors = GenerateCustomScheme(baseColor);
}

const ColorScheme& ColorManager::GetColors() const {
    return currentColors;
}

ColorSchemeType ColorManager::GetSchemeType() const {
    return currentScheme;
}

ColorScheme ColorManager::GenerateDarkScheme() {
    ColorScheme scheme;
    scheme.primaryLight = Color::FromRGB(100, 180, 255);
    scheme.primaryNormal = Color::FromRGB(52, 152, 219);
    scheme.primaryDark = Color::FromRGB(30, 100, 150);
    
    scheme.secondaryLight = Color::FromRGB(80, 200, 140);
    scheme.secondaryNormal = Color::FromRGB(39, 174, 96);
    scheme.secondaryDark = Color::FromRGB(25, 120, 70);
    
    scheme.neutralLight = Color::FromRGB(80, 80, 85);
    scheme.neutralNormal = Color::FromRGB(50, 50, 55);
    scheme.neutralDark = Color::FromRGB(30, 30, 35);
    
    scheme.success = Color::FromRGB(46, 204, 113);
    scheme.warning = Color::FromRGB(241, 196, 15);
    scheme.error = Color::FromRGB(231, 76, 60);
    scheme.info = Color::FromRGB(52, 152, 219);
    
    scheme.background = Color::FromRGB(20, 20, 25);
    scheme.surface = Color::FromRGB(35, 35, 40);
    scheme.surfaceElevated = Color::FromRGB(45, 45, 50);
    
    scheme.textPrimary = Color::FromRGB(240, 240, 245);
    scheme.textSecondary = Color::FromRGB(180, 180, 185);
    scheme.textDisabled = Color::FromRGB(100, 100, 105);
    
    scheme.border = Color::FromRGB(60, 60, 65);
    scheme.borderFocus = scheme.primaryNormal;
    
    return scheme;
}

ColorScheme ColorManager::GenerateLightScheme() {
    ColorScheme scheme;
    scheme.primaryLight = Color::FromRGB(120, 200, 255);
    scheme.primaryNormal = Color::FromRGB(52, 152, 219);
    scheme.primaryDark = Color::FromRGB(30, 100, 150);
    
    scheme.secondaryLight = Color::FromRGB(100, 220, 160);
    scheme.secondaryNormal = Color::FromRGB(39, 174, 96);
    scheme.secondaryDark = Color::FromRGB(25, 120, 70);
    
    scheme.neutralLight = Color::FromRGB(240, 240, 245);
    scheme.neutralNormal = Color::FromRGB(220, 220, 225);
    scheme.neutralDark = Color::FromRGB(180, 180, 185);
    
    scheme.success = Color::FromRGB(46, 204, 113);
    scheme.warning = Color::FromRGB(241, 196, 15);
    scheme.error = Color::FromRGB(231, 76, 60);
    scheme.info = Color::FromRGB(52, 152, 219);
    
    scheme.background = Color::FromRGB(250, 250, 252);
    scheme.surface = Color::FromRGB(255, 255, 255);
    scheme.surfaceElevated = Color::FromRGB(245, 245, 250);
    
    scheme.textPrimary = Color::FromRGB(30, 30, 35);
    scheme.textSecondary = Color::FromRGB(80, 80, 85);
    scheme.textDisabled = Color::FromRGB(150, 150, 155);
    
    scheme.border = Color::FromRGB(200, 200, 205);
    scheme.borderFocus = scheme.primaryNormal;
    
    return scheme;
}

ColorScheme ColorManager::GenerateNormalScheme() {
    ColorScheme scheme;
    
    scheme.primaryLight = Color::FromRGB(74, 144, 164);
    scheme.primaryNormal = Color::FromRGB(45, 74, 107);
    scheme.primaryDark = Color::FromRGB(35, 58, 84);
    
    scheme.secondaryLight = Color::FromRGB(243, 176, 68);
    scheme.secondaryNormal = Color::FromRGB(243, 156, 18);
    scheme.secondaryDark = Color::FromRGB(200, 128, 15);
    
    scheme.neutralLight = Color::FromRGB(61, 86, 110);
    scheme.neutralNormal = Color::FromRGB(42, 63, 84);
    scheme.neutralDark = Color::FromRGB(30, 41, 59);
    
    scheme.success = Color::FromRGB(39, 174, 96);
    scheme.warning = Color::FromRGB(243, 156, 18);
    scheme.error = Color::FromRGB(231, 76, 60);
    scheme.info = Color::FromRGB(74, 144, 164);
    
    scheme.background = Color::FromRGB(30, 41, 59);
    scheme.surface = Color::FromRGB(42, 63, 84);
    scheme.surfaceElevated = Color::FromRGB(52, 73, 94);
    
    scheme.textPrimary = Color::FromRGB(236, 240, 241);
    scheme.textSecondary = Color::FromRGB(149, 165, 166);
    scheme.textDisabled = Color::FromRGB(100, 110, 115);
    
    scheme.border = Color::FromRGB(61, 86, 110);
    scheme.borderFocus = scheme.primaryLight;
    
    return scheme;
}

ColorScheme ColorManager::GenerateCustomScheme(const Color& baseColor) {
    ColorScheme scheme;
    Color base = baseColor;
    
    scheme.primaryLight = base.Lighten(0.15f);
    scheme.primaryNormal = base;
    scheme.primaryDark = base.Darken(0.15f);
    
    Color complementary = Color::FromHSL(fmodf(base.h + 180.0f, 360.0f), base.s, base.l);
    scheme.secondaryLight = complementary.Lighten(0.15f);
    scheme.secondaryNormal = complementary;
    scheme.secondaryDark = complementary.Darken(0.15f);
    
    scheme.neutralLight = Color::FromRGB(236, 240, 241);
    scheme.neutralNormal = Color::FromRGB(189, 195, 199);
    scheme.neutralDark = Color::FromRGB(149, 165, 166);
    
    scheme.success = Color::FromRGB(46, 204, 113);
    scheme.warning = Color::FromRGB(241, 196, 15);
    scheme.error = Color::FromRGB(231, 76, 60);
    scheme.info = base;
    
    scheme.background = Color::FromRGB(245, 246, 250);
    scheme.surface = Color::FromRGB(255, 255, 255);
    scheme.surfaceElevated = Color::FromRGB(248, 249, 252);
    
    scheme.textPrimary = Color::FromRGB(44, 62, 80);
    scheme.textSecondary = Color::FromRGB(127, 140, 141);
    scheme.textDisabled = Color::FromRGB(189, 195, 199);
    
    scheme.border = Color::FromRGB(200, 205, 210);
    scheme.borderFocus = scheme.primaryNormal;
    
    return scheme;
}

const std::vector<Color>& ColorManager::GetPresetBaseColors() {
    static std::vector<Color> presets = {
        Color::FromRGB(52, 152, 219),    // 专业蓝
        Color::FromRGB(39, 174, 96),     // 智慧绿
        Color::FromRGB(139, 69, 19),     // 古典棕
        Color::FromRGB(155, 89, 182),    // 现代紫
        Color::FromRGB(230, 126, 34),    // 活力橙
        Color::FromRGB(127, 140, 141)    // 沉静灰
    };
    return presets;
}

Color ColorManager::EnsureContrast(const Color& foreground, const Color& background, float minContrast) {
    Color result = foreground;
    float contrast = result.GetContrast(background);
    
    if (contrast < minContrast) {
        float luminance = result.l;
        float bgLuminance = background.l;
        
        if (bgLuminance > 0.5f) {
            result = result.Darken(0.3f);
        } else {
            result = result.Lighten(0.3f);
        }
    }
    
    return result;
}

} // namespace GUI
} // namespace SpaceGo
