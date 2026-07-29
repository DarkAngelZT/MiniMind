#include "gru.hpp"

#include <MNN/expr/ExprCreator.hpp>

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace MiniMind {
namespace {

MNN::Express::VARP make_weight(int output_size, int input_size) {
    // MiniBrain由调用方执行随机初始化；这里沿用项目的Xavier均匀初始化。
    const float bound =
        std::sqrt(6.0f / static_cast<float>(input_size + output_size));
    static thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<float> distribution(-bound, bound);
    std::vector<float> values(
        static_cast<std::size_t>(output_size * input_size));
    for (float& value : values) {
        value = distribution(generator);
    }
    return MNN::Express::_TrainableParam(
        values.data(), {output_size, input_size}, MNN::Express::NCHW);
}

MNN::Express::VARP make_bias(int hidden_size) {
    return MNN::Express::_TrainableParam(
        0.0f, {hidden_size}, MNN::Express::NCHW);
}

} // namespace

GRU::GRU(int input_size, int hidden_size)
    : input_size_(input_size), hidden_size_(hidden_size) {
    if (input_size <= 0 || hidden_size <= 0) {
        throw std::invalid_argument("GRU: input_size和hidden_size必须为正数");
    }

    // 参数使用[输出特征,输入特征]布局，前向时转置右矩阵。
    wz_ = make_weight(hidden_size_, input_size_);
    wr_ = make_weight(hidden_size_, input_size_);
    wh_ = make_weight(hidden_size_, input_size_);
    uz_ = make_weight(hidden_size_, hidden_size_);
    ur_ = make_weight(hidden_size_, hidden_size_);
    uh_ = make_weight(hidden_size_, hidden_size_);
    bz_ = make_bias(hidden_size_);
    br_ = make_bias(hidden_size_);
    bh_ = make_bias(hidden_size_);

    // 注册顺序与MiniBrain参数序列化顺序保持一致。
    addParameter(wz_);
    addParameter(wr_);
    addParameter(wh_);
    addParameter(uz_);
    addParameter(ur_);
    addParameter(uh_);
    addParameter(bz_);
    addParameter(br_);
    addParameter(bh_);
    setType("GRU");
}

MNN::Express::VARP GRU::forwardStep(
    const MNN::Express::VARP& input,
    const MNN::Express::VARP& hidden) {
    return onForward({input, hidden})[0];
}

std::vector<MNN::Express::VARP> GRU::onForward(
    const std::vector<MNN::Express::VARP>& inputs) {
    if (inputs.size() != 2 || inputs[0] == nullptr || inputs[1] == nullptr) {
        throw std::invalid_argument("GRU: 必须提供输入和上一隐藏状态");
    }
    const auto* input_info = inputs[0]->getInfo();
    const auto* hidden_info = inputs[1]->getInfo();
    if (input_info == nullptr || hidden_info == nullptr ||
        input_info->dim.size() != 2 || hidden_info->dim.size() != 2 ||
        input_info->dim[0] <= 0 ||
        input_info->dim[1] != input_size_ ||
        hidden_info->dim[0] != input_info->dim[0] ||
        hidden_info->dim[1] != hidden_size_) {
        throw std::invalid_argument(
            "GRU: 输入形状必须为[batch,input_size]，隐藏状态必须为[batch,hidden_size]");
    }

    using namespace MNN::Express;
    const auto& input = inputs[0];
    const auto& hidden = inputs[1];
    const auto z = _Sigmoid(
        _Add(_Add(_MatMul(input, wz_, false, true),
                  _MatMul(hidden, uz_, false, true)),
             bz_));
    const auto r = _Sigmoid(
        _Add(_Add(_MatMul(input, wr_, false, true),
                  _MatMul(hidden, ur_, false, true)),
             br_));
    const auto candidate = _Tanh(
        _Add(_Add(_MatMul(input, wh_, false, true),
                  _MatMul(_Multiply(r, hidden), uh_, false, true)),
             bh_));
    const auto one = _Scalar<float>(1.0f);

    // 严格保持MiniBrain约定：z控制候选状态，而不是旧隐藏状态。
    return {_Add(_Multiply(_Subtract(one, z), hidden),
                 _Multiply(z, candidate))};
}

MNN::Express::Module* GRU::clone(CloneContext* context) const {
    auto* module = new GRU;
    module->input_size_ = input_size_;
    module->hidden_size_ = hidden_size_;
    module->wz_ = wz_;
    module->wr_ = wr_;
    module->wh_ = wh_;
    module->uz_ = uz_;
    module->ur_ = ur_;
    module->uh_ = uh_;
    module->bz_ = bz_;
    module->br_ = br_;
    module->bh_ = bh_;
    // 按项目现有Module约定共享VARP，并重新注册相同的参数顺序。
    module->addParameter(module->wz_);
    module->addParameter(module->wr_);
    module->addParameter(module->wh_);
    module->addParameter(module->uz_);
    module->addParameter(module->ur_);
    module->addParameter(module->uh_);
    module->addParameter(module->bz_);
    module->addParameter(module->br_);
    module->addParameter(module->bh_);
    return cloneBaseTo(context, module);
}

} // namespace MiniMind
