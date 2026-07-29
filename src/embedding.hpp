#pragma once

#include <MNN/expr/Module.hpp>

namespace MiniMind {

// 对每个实体独立应用同一组线性投影参数。
class Embedding final : public MNN::Express::Module {
public:
    Embedding(int input_feature_count, int output_feature_count);

    std::vector<MNN::Express::VARP> onForward(
        const std::vector<MNN::Express::VARP>& inputs) override;

    MNN::Express::Module* clone(CloneContext* context) const override;

private:
    Embedding() = default;

    int input_feature_count_ = 0;
    int output_feature_count_ = 0;
    MNN::Express::VARP weight_;
    MNN::Express::VARP bias_;
};

}
