#pragma once

#include <MNN/expr/Module.hpp>

namespace MiniMind {

// 无状态GRU单元；调用者显式传入并保存隐藏状态。
class GRU final : public MNN::Express::Module {
public:
    GRU(int input_size, int hidden_size);

    // 执行一个时间步，输入形状分别为[B,I]和[B,H]。
    MNN::Express::VARP forwardStep(
        const MNN::Express::VARP& input,
        const MNN::Express::VARP& hidden);

    std::vector<MNN::Express::VARP> onForward(
        const std::vector<MNN::Express::VARP>& inputs) override;

    MNN::Express::Module* clone(CloneContext* context) const override;

private:
    GRU() = default;

    int input_size_ = 0;
    int hidden_size_ = 0;

    MNN::Express::VARP wz_;
    MNN::Express::VARP wr_;
    MNN::Express::VARP wh_;
    MNN::Express::VARP uz_;
    MNN::Express::VARP ur_;
    MNN::Express::VARP uh_;
    MNN::Express::VARP bz_;
    MNN::Express::VARP br_;
    MNN::Express::VARP bh_;
};

} // namespace MiniMind
