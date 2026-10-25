// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：math.h
// 描述：数学工具库
// 创建日期：2026-02-20
// 作者：SpaceGo Team
// ============================================================================

#pragma once
#include <cmath>

namespace SpaceGo {
namespace GUI {

namespace Math {

    // 常量 (C++17 constexpr)
    constexpr float PI = 3.14159265358979323846f;
    constexpr float DEG_TO_RAD = PI / 180.0f;
    constexpr float RAD_TO_DEG = 180.0f / PI;

    // 限制范围
    [[nodiscard]] constexpr float Clamp(float value, float min, float max) noexcept {
        return value < min ? min : (value > max ? max : value);
    }

    // 插值
    [[nodiscard]] constexpr float Lerp(float a, float b, float t) noexcept {
        return a + (b - a) * t;
    }

    // 平滑插值 (Smoothstep)
    [[nodiscard]] inline float Smoothstep(float edge0, float edge1, float x) noexcept {
        float t = Clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
        return t * t * (3.0f - 2.0f * t);
    }

    // 距离计算
    [[nodiscard]] inline float Distance(float x1, float y1, float x2, float y2) noexcept {
        float dx = x2 - x1;
        float dy = y2 - y1;
        return sqrtf(dx * dx + dy * dy);
    }

    // 点是否在矩形内
    [[nodiscard]] constexpr bool PointInRect(float px, float py, float rx, float ry, float rw, float rh) noexcept {
        return px >= rx && px < rx + rw && py >= ry && py < ry + rh;
    }

    // 点是否在圆内
    [[nodiscard]] inline bool PointInCircle(float px, float py, float cx, float cy, float radius) noexcept {
        return Distance(px, py, cx, cy) <= radius;
    }

    // 最小/最大值
    [[nodiscard]] constexpr float Min(float a, float b) noexcept { return a < b ? a : b; }
    [[nodiscard]] constexpr float Max(float a, float b) noexcept { return a > b ? a : b; }

    // 角度转换
    [[nodiscard]] constexpr float DegToRad(float deg) noexcept { return deg * DEG_TO_RAD; }
    [[nodiscard]] constexpr float RadToDeg(float rad) noexcept { return rad * RAD_TO_DEG; }

} // namespace Math
} // namespace GUI
} // namespace SpaceGo
