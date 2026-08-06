#include "attention.hpp"

#include <MNN/expr/ExprCreator.hpp>

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace MiniMind {
namespace {

MNN::Express::VARP tile(
    const MNN::Express::VARP& input, const std::vector<int>& multiples) {
    using namespace MNN::Express;
    return _Tile(
        input,
        _Const(multiples.data(), {static_cast<int>(multiples.size())}, NCHW,
               halide_type_of<int>()));
}

MNN::Express::VARP make_weight(int output_size, int input_size) {
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

}

Attention::Attention(int feature_count, int key_dim)
    : feature_count_(feature_count), key_dim_(key_dim) {
    if (feature_count <= 0 || key_dim <= 0) {
        throw std::invalid_argument("Attention: feature_count和key_dim必须为正数");
    }

    // 参数形状沿用旧版的[输出特征,输入特征]，前向时转置右矩阵。
    wq_ = make_weight(key_dim_, feature_count_);
    wk_ = make_weight(key_dim_, feature_count_);
    wv_ = make_weight(feature_count_, feature_count_);
    addParameter(wq_);
    addParameter(wk_);
    addParameter(wv_);
    setType("Attention");
}

MNN::Express::VARP Attention::forward(
    const MNN::Express::VARP& input,
    const MNN::Express::VARP& mask) {
    return onForward({input, mask})[0];
}

std::vector<MNN::Express::VARP> Attention::onForward(
    const std::vector<MNN::Express::VARP>& inputs) {
    if (inputs.size() != 2 || inputs[0] == nullptr || inputs[1] == nullptr) {
        throw std::invalid_argument("Attention: input and mask are required");
    }
    const auto* info = inputs[0]->getInfo();
    const auto* mask_info = inputs[1]->getInfo();
    if (info == nullptr || info->dim.size() != 3 || info->dim[0] <= 0 ||
        info->dim[1] <= 0 || info->dim[2] != feature_count_) {
        throw std::invalid_argument(
            "Attention: input shape must be [batch,entity,feature_count]");
    }
    if (mask_info == nullptr || mask_info->dim.size() != 2 ||
        mask_info->dim[0] != info->dim[0] || mask_info->dim[1] != info->dim[1]) {
        throw std::invalid_argument(
            "Attention: mask shape must be [batch,entity]");
    }

    using namespace MNN::Express;
    const auto& input = inputs[0];
    const auto& mask = inputs[1];
    const int entity_count = info->dim[1];
    const auto query = _MatMul(input, wq_, false, true);
    const auto key = _MatMul(input, wk_, false, true);
    const auto value = _MatMul(input, wv_, false, true);
    const float scale = 1.0f / std::sqrt(static_cast<float>(key_dim_));
    const auto scores = _BatchMatMul(query, key, false, true) *
                        _Scalar<float>(scale);

    const auto valid_key = _Greater(mask, _Scalar<float>(0.0f));
    const auto key_mask = tile(_Unsqueeze(valid_key, {1}), {1, entity_count, 1});
    const auto masked_scores = _Select(
        key_mask, scores, _Scalar<float>(-std::numeric_limits<float>::infinity()));

    // An all-padding batch would otherwise softmax an all--infinity row.
    // Zero logits make that intermediate finite; output masking still returns zero.
    const auto has_valid_key = _Greater(
        _ReduceSum(mask, {1}, true), _Scalar<float>(0.0f));
    const auto score_row_mask = tile(_Unsqueeze(has_valid_key, {2}),
             {1, entity_count, entity_count});
    const auto stable_scores = _Select(
        score_row_mask, masked_scores, _Scalar<float>(0.0f));
    const auto attended = _BatchMatMul(_Softmax(stable_scores, -1), value);
    const auto output_mask = tile(_Unsqueeze(mask, {2}), {1, 1, feature_count_});
    return {_Multiply(attended, output_mask)};
}

MNN::Express::Module* Attention::clone(CloneContext* context) const {
    auto* module = new Attention;
    module->feature_count_ = feature_count_;
    module->key_dim_ = key_dim_;
    module->wq_ = wq_;
    module->wk_ = wk_;
    module->wv_ = wv_;
    // MNN内置训练Module的克隆会共享VARP，并按原顺序重新注册参数。
    module->addParameter(module->wq_);
    module->addParameter(module->wk_);
    module->addParameter(module->wv_);
    return cloneBaseTo(context, module);
}

}
