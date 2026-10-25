#pragma once

#include <torch/torch.h>
#include <torch/optim/schedulers/lr_scheduler.h>
#include <vector>
#include <cmath>

namespace spacego {
namespace framework {

// 自定义AMP梯度缩放器
// LibTorch C++ API不提供torch::cuda::amp::GradScaler
// 此类提供兼容接口，用于混合精度训练中的梯度缩放
class AMPGradScaler {
public:
    explicit AMPGradScaler(double init_scale = 65536.0,
                           double growth_factor = 2.0,
                           double backoff_factor = 0.5,
                           int growth_interval = 2000,
                           bool enabled = true)
        : scale_(init_scale)
        , growth_factor_(growth_factor)
        , backoff_factor_(backoff_factor)
        , growth_interval_(growth_interval)
        , enabled_(enabled)
        , growth_tracker_(0)
        , found_inf_(false)
    {}

    // 缩放损失值，返回缩放后的损失
    torch::Tensor scale(const torch::Tensor& loss) {
        if (!enabled_) return loss;
        return loss * scale_;
    }

    // 反缩放优化器中的梯度
    void unscale_(torch::optim::Optimizer& optimizer) {
        if (!enabled_) return;
        found_inf_ = false;
        for (auto& param_group : optimizer.param_groups()) {
            for (auto& param : param_group.params()) {
                if (!param.grad().defined()) continue;
                auto grad = param.grad();
                // 检测inf/nan
                if (grad.is_floating_point()) {
                    found_inf_ = found_inf_ || grad.isinf().any().item<bool>();
                    found_inf_ = found_inf_ || grad.isnan().any().item<bool>();
                }
                // 反缩放梯度
                param.grad().mul_(1.0 / scale_);
            }
        }
    }

    // 执行优化器步骤，仅在梯度中无inf/nan时才更新参数
    void step(torch::optim::Optimizer& optimizer) {
        if (!enabled_) {
            optimizer.step();
            return;
        }
        if (!found_inf_) {
            optimizer.step();
        }
    }

    // 更新缩放因子
    void update() {
        if (!enabled_) return;
        if (found_inf_) {
            // 发现inf/nan，缩小缩放因子
            scale_ *= backoff_factor_;
            growth_tracker_ = 0;
        } else {
            // 正常增长
            growth_tracker_++;
            if (growth_tracker_ >= growth_interval_) {
                scale_ *= growth_factor_;
                growth_tracker_ = 0;
            }
        }
    }

    double get_scale() const { return scale_; }
    bool is_enabled() const { return enabled_; }

private:
    double scale_;
    double growth_factor_;
    double backoff_factor_;
    int growth_interval_;
    bool enabled_;
    int growth_tracker_;
    bool found_inf_;
};

// 自定义余弦退火学习率调度器
// LibTorch C++ API不提供CosineAnnealingLR
// 继承torch::optim::LRScheduler实现余弦退火策略
class CosineAnnealingLRScheduler : public torch::optim::LRScheduler {
public:
    CosineAnnealingLRScheduler(torch::optim::Optimizer& optimizer,
                               int64_t t_max,
                               double eta_min = 0.0)
        : torch::optim::LRScheduler(optimizer)
        , t_max_(t_max)
        , eta_min_(eta_min)
        , base_lrs_(get_current_lrs())
        , last_lrs_(base_lrs_)
    {}

    // 获取上一步的学习率
    std::vector<double> get_last_lr() const { return last_lrs_; }

protected:
    // 计算新的学习率（余弦退火公式）
    std::vector<double> get_lrs() override {
        std::vector<double> new_lrs;
        for (double base_lr : base_lrs_) {
            double new_lr = eta_min_ + (base_lr - eta_min_) *
                (1.0 + std::cos(M_PI * static_cast<double>(step_count_) / t_max_)) / 2.0;
            new_lrs.push_back(new_lr);
        }
        last_lrs_ = new_lrs;
        return new_lrs;
    }

private:
    int64_t t_max_;
    double eta_min_;
    std::vector<double> base_lrs_;
    mutable std::vector<double> last_lrs_;
};

} // namespace framework
} // namespace spacego
