#pragma once

#include <MNN/expr/Module.hpp>

namespace MiniMind {

// 单头缩放点积自注意力，输入和输出均采用[B,E,F]布局。
class Attention final : public MNN::Express::Module {
public:
    Attention(int feature_count, int key_dim);

    // input: [batch, entity, feature_count], mask: [batch, entity] with 0/1 values.
    MNN::Express::VARP forward(
        const MNN::Express::VARP& input,
        const MNN::Express::VARP& mask);

    std::vector<MNN::Express::VARP> onForward(
        const std::vector<MNN::Express::VARP>& inputs) override;

    MNN::Express::Module* clone(CloneContext* context) const override;

private:
    Attention() = default;

    int feature_count_ = 0;
    int key_dim_ = 0;
    MNN::Express::VARP wq_;
    MNN::Express::VARP wk_;
    MNN::Express::VARP wv_;
};

}
