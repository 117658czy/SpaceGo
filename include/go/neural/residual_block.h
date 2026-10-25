#pragma once

#ifdef USE_LIBTORCH

#include <torch/torch.h>
#include <memory>

namespace spacego {
namespace go {
namespace neural {

// 残差块实现
// 使用shared_ptr<Impl>代替ModuleHolder成员，避免MSVC过度实例化默认构造函数
struct ResidualBlockImpl : public torch::nn::Module {
    ResidualBlockImpl(int channels) {
        conv1 = std::make_shared<torch::nn::Conv2dImpl>(
            torch::nn::Conv2dOptions(channels, channels, 3).padding(1).bias(false));
        register_module("conv1", conv1);

        bn1 = std::make_shared<torch::nn::BatchNorm2dImpl>(channels);
        register_module("bn1", bn1);

        conv2 = std::make_shared<torch::nn::Conv2dImpl>(
            torch::nn::Conv2dOptions(channels, channels, 3).padding(1).bias(false));
        register_module("conv2", conv2);

        bn2 = std::make_shared<torch::nn::BatchNorm2dImpl>(channels);
        register_module("bn2", bn2);
    }

    torch::Tensor forward(torch::Tensor x) {
        auto out = torch::relu(bn1->forward(conv1->forward(x)));
        out = bn2->forward(conv2->forward(out));
        out = out + x;
        out = torch::relu(out);
        return out;
    }

    std::shared_ptr<torch::nn::Conv2dImpl> conv1, conv2;
    std::shared_ptr<torch::nn::BatchNorm2dImpl> bn1, bn2;
};

TORCH_MODULE(ResidualBlock);

} // namespace neural
} // namespace go
} // namespace spacego
#endif  // USE_LIBTORCH
