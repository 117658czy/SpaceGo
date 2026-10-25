// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：animation.h
// 描述：动画系统
// 创建日期：2026-02-20
// 作者：SpaceGo Team
// ============================================================================

#pragma once
#include <functional>
#include <vector>
#include <memory>
#include "utils/math.h"

namespace SpaceGo {
namespace GUI {

// 缓动函数类型
enum class EasingType {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    EaseInQuad,
    EaseOutQuad,
    EaseInOutQuad,
    EaseInCubic,
    EaseOutCubic,
    EaseInOutCubic,
    EaseInQuart,
    EaseOutQuart,
    EaseInOutQuart,
    EaseInQuint,
    EaseOutQuint,
    EaseInOutQuint,
    EaseInSine,
    EaseOutSine,
    EaseInOutSine,
    EaseInExpo,
    EaseOutExpo,
    EaseInOutExpo,
    EaseInCirc,
    EaseOutCirc,
    EaseInOutCirc,
    EaseInBack,
    EaseOutBack,
    EaseInOutBack,
    Bounce,
    Elastic
};

// 动画回调类型
using AnimationCallback = std::function<void(float value)>;
using AnimationCompleteCallback = std::function<void()>;

// 动画类
class Animation {
public:
    float duration;
    float elapsed;
    float startValue;
    float endValue;
    EasingType easingType;
    AnimationCallback onUpdate;
    AnimationCompleteCallback onComplete;
    bool active;

    Animation(float dur, float start, float end, EasingType easing, AnimationCallback update);

    void Update(float deltaTime);
    float GetCurrentValue() const;
    bool IsComplete() const;

private:
    float ApplyEasing(float t) const;
};

// 动画管理器
class AnimationManager {
private:
    std::vector<std::unique_ptr<Animation>> animations;

    AnimationManager();

public:
    static AnimationManager& GetInstance();
    ~AnimationManager();

    // 添加动画
    Animation* AddAnimation(float duration, float startValue, float endValue,
                             EasingType easingType, AnimationCallback onUpdate);

    // 更新所有动画
    void Update(float deltaTime);

    // 清除所有动画
    void ClearAll();
};

} // namespace GUI
} // namespace SpaceGo
