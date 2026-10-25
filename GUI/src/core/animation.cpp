// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：animation.cpp
// 描述：动画系统实现
// 创建日期：2026-02-20
// 作者：SpaceGo Team
// ============================================================================

#include "core/animation.h"
#include <algorithm>

namespace SpaceGo {
namespace GUI {

// ==========================================
// Animation 类实现
// ==========================================

Animation::Animation(float dur, float start, float end, EasingType easing, AnimationCallback update)
    : duration(dur), elapsed(0), startValue(start), endValue(end), 
      easingType(easing), onUpdate(update), active(true) {
}

void Animation::Update(float deltaTime) {
    if (!active) return;

    elapsed += deltaTime;
    if (elapsed >= duration) {
        elapsed = duration;
        active = false;
        if (onUpdate) {
            onUpdate(endValue);
        }
        if (onComplete) {
            onComplete();
        }
    } else {
        if (onUpdate) {
            onUpdate(GetCurrentValue());
        }
    }
}

float Animation::GetCurrentValue() const {
    float t = Math::Clamp(elapsed / duration, 0.0f, 1.0f);
    float easedT = ApplyEasing(t);
    return Math::Lerp(startValue, endValue, easedT);
}

bool Animation::IsComplete() const {
    return !active;
}

float Animation::ApplyEasing(float t) const {
    switch (easingType) {
        case EasingType::Linear:
            return t;

        case EasingType::EaseIn:
        case EasingType::EaseInQuad:
            return t * t;

        case EasingType::EaseOut:
        case EasingType::EaseOutQuad:
            return t * (2.0f - t);

        case EasingType::EaseInOut:
        case EasingType::EaseInOutQuad:
            return t < 0.5f ? 2.0f * t * t : -1.0f + (4.0f - 2.0f * t) * t;

        case EasingType::EaseInCubic:
            return t * t * t;

        case EasingType::EaseOutCubic:
            return (--t) * t * t + 1.0f;

        case EasingType::EaseInOutCubic:
            return t < 0.5f ? 4.0f * t * t * t : (t - 1.0f) * (2.0f * t - 2.0f) * (2.0f * t - 2.0f) + 1.0f;

        case EasingType::EaseInQuart:
            return t * t * t * t;

        case EasingType::EaseOutQuart:
            return 1.0f - (--t) * t * t * t;

        case EasingType::EaseInOutQuart:
            return t < 0.5f ? 8.0f * t * t * t * t : 1.0f - 8.0f * (--t) * t * t * t;

        case EasingType::EaseInQuint:
            return t * t * t * t * t;

        case EasingType::EaseOutQuint:
            return 1.0f + (--t) * t * t * t * t;

        case EasingType::EaseInOutQuint:
            return t < 0.5f ? 16.0f * t * t * t * t * t : 1.0f + 16.0f * (--t) * t * t * t * t;

        case EasingType::EaseInSine:
            return 1.0f - cosf(t * Math::PI / 2.0f);

        case EasingType::EaseOutSine:
            return sinf(t * Math::PI / 2.0f);

        case EasingType::EaseInOutSine:
            return -(cosf(Math::PI * t) - 1.0f) / 2.0f;

        case EasingType::EaseInExpo:
            return t == 0.0f ? 0.0f : powf(2.0f, 10.0f * (t - 1.0f));

        case EasingType::EaseOutExpo:
            return t == 1.0f ? 1.0f : 1.0f - powf(2.0f, -10.0f * t);

        case EasingType::EaseInOutExpo:
            if (t == 0.0f) return 0.0f;
            if (t == 1.0f) return 1.0f;
            if (t < 0.5f) return powf(2.0f, 20.0f * t - 10.0f) / 2.0f;
            return (2.0f - powf(2.0f, -20.0f * t + 10.0f)) / 2.0f;

        case EasingType::EaseInCirc:
            return 1.0f - sqrtf(1.0f - t * t);

        case EasingType::EaseOutCirc:
            return sqrtf(1.0f - (--t) * t);

        case EasingType::EaseInOutCirc:
            if (t < 0.5f) return -(sqrtf(1.0f - 4.0f * t * t) - 1.0f) / 2.0f;
            return (sqrtf(1.0f - powf(-2.0f * t + 2.0f, 2.0f)) + 1.0f) / 2.0f;

        case EasingType::EaseInBack: {
            const float c1 = 1.70158f;
            const float c3 = c1 + 1.0f;
            return c3 * t * t * t - c1 * t * t;
        }

        case EasingType::EaseOutBack: {
            const float c1 = 1.70158f;
            const float c3 = c1 + 1.0f;
            return 1.0f + c3 * powf(t - 1.0f, 3.0f) + c1 * powf(t - 1.0f, 2.0f);
        }

        case EasingType::EaseInOutBack: {
            const float c1 = 1.70158f;
            const float c2 = c1 * 1.525f;
            if (t < 0.5f) return (powf(2.0f * t, 2.0f) * ((c2 + 1.0f) * 2.0f * t - c2)) / 2.0f;
            return (powf(2.0f * t - 2.0f, 2.0f) * ((c2 + 1.0f) * (t * 2.0f - 2.0f) + c2) + 2.0f) / 2.0f;
        }

        case EasingType::Bounce:
            if (t < 1.0f / 2.75f) {
                return 7.5625f * t * t;
            } else if (t < 2.0f / 2.75f) {
                t -= 1.5f / 2.75f;
                return 7.5625f * t * t + 0.75f;
            } else if (t < 2.5f / 2.75f) {
                t -= 2.25f / 2.75f;
                return 7.5625f * t * t + 0.9375f;
            } else {
                t -= 2.625f / 2.75f;
                return 7.5625f * t * t + 0.984375f;
            }

        case EasingType::Elastic: {
            const float c4 = (2.0f * Math::PI) / 3.0f;
            if (t == 0.0f) return 0.0f;
            if (t == 1.0f) return 1.0f;
            return -powf(2.0f, 10.0f * t - 10.0f) * sinf((t * 10.0f - 10.75f) * c4);
        }

        default:
            return t;
    }
}

// ==========================================
// AnimationManager 类实现
// ==========================================

AnimationManager::AnimationManager() {
}

AnimationManager& AnimationManager::GetInstance() {
    static AnimationManager instance;
    return instance;
}

AnimationManager::~AnimationManager() {
    ClearAll();
}

Animation* AnimationManager::AddAnimation(float duration, float startValue, float endValue,
                                           EasingType easingType, AnimationCallback onUpdate) {
    auto anim = std::make_unique<Animation>(duration, startValue, endValue, easingType, onUpdate);
    Animation* animPtr = anim.get();
    animations.push_back(std::move(anim));
    return animPtr;
}

void AnimationManager::Update(float deltaTime) {
    // 更新所有动画
    for (auto it = animations.begin(); it != animations.end(); ) {
        (*it)->Update(deltaTime);
        if ((*it)->IsComplete()) {
            it = animations.erase(it);
        } else {
            ++it;
        }
    }
}

void AnimationManager::ClearAll() {
    animations.clear();
}

} // namespace GUI
} // namespace SpaceGo
