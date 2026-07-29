#include "state_pooling.hpp"

#include <MNN/expr/ExprCreator.hpp>

#include <stdexcept>
#include <vector>

namespace MiniMind {

StatePooling::StatePooling(int feature_count)
    : feature_count_(feature_count) {
    if (feature_count <= 0) {
        throw std::invalid_argument(
            "StatePooling: feature_count必须为正整数");
    }
    setType("StatePooling");
}

std::vector<MNN::Express::VARP> StatePooling::onForward(
    const std::vector<MNN::Express::VARP>& inputs) {
    if (inputs.size() != 1 || inputs[0] == nullptr) {
        throw std::invalid_argument(
            "StatePooling: 必须提供一个非空输入");
    }
    const auto* info = inputs[0]->getInfo();
    if (info == nullptr || info->dim.size() != 3 || info->dim[0] <= 0 ||
        info->dim[1] <= 0 || info->dim[2] != feature_count_) {
        throw std::invalid_argument(
            "StatePooling: 输入形状必须为[batch,entity,feature_count]");
    }

    using namespace MNN::Express;
    // 原版对每个环境的每项特征分别沿实体集合计算均值和最大值。
    const auto mean = _ReduceMean(inputs[0], {1});
    const auto maximum = _ReduceMax(inputs[0], {1});
    return {_Concat({mean, maximum}, 1)};
}

MNN::Express::Module* StatePooling::clone(CloneContext* context) const {
    auto* module = new StatePooling;
    module->feature_count_ = feature_count_;
    return cloneBaseTo(context, module);
}

} // MiniMind命名空间
