#pragma once

#include <MNN/expr/Module.hpp>

namespace MiniMind {

// 沿实体维分别计算均值和最大值，并按特征维拼接结果。
class StatePooling final : public MNN::Express::Module {
public:
    explicit StatePooling(int feature_count);

    std::vector<MNN::Express::VARP> onForward(
        const std::vector<MNN::Express::VARP>& inputs) override;

    MNN::Express::Module* clone(CloneContext* context) const override;

private:
    StatePooling() = default;

    int feature_count_ = 0;
};

} // MiniMind命名空间
