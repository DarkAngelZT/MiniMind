#include "embedding.hpp"

#include <MNN/expr/ExprCreator.hpp>

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace MiniMind {
namespace {

MNN::Express::VARP make_weight(int input_feature_count,
                               int output_feature_count) {
    const float bound = std::sqrt(
        6.0f / static_cast<float>(input_feature_count + output_feature_count));
    static thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<float> distribution(-bound, bound);
    std::vector<float> values(static_cast<std::size_t>(
        input_feature_count * output_feature_count));
    for (float& value : values) {
        value = distribution(generator);
    }
    return MNN::Express::_TrainableParam(
        values.data(), {input_feature_count, output_feature_count},
        MNN::Express::NCHW);
}

MNN::Express::VARP make_bias(int output_feature_count) {
    std::vector<float> values(static_cast<std::size_t>(output_feature_count),
                              0.0f);
    return MNN::Express::_TrainableParam(
        values.data(), {output_feature_count}, MNN::Express::NCHW);
}

}

Embedding::Embedding(int input_feature_count, int output_feature_count)
    : input_feature_count_(input_feature_count),
      output_feature_count_(output_feature_count) {
    if (input_feature_count <= 0 || output_feature_count <= 0) {
        throw std::invalid_argument("Embedding: 输入和输出特征数必须为正数");
    }

    // 参数形状与参考实现一致，依次注册权重[input_feature,out_feature]和偏置[out_feature]。
    weight_ = make_weight(input_feature_count_, output_feature_count_);
    bias_ = make_bias(output_feature_count_);
    addParameter(weight_);
    addParameter(bias_);
    setType("Embedding");
}

std::vector<MNN::Express::VARP> Embedding::onForward(
    const std::vector<MNN::Express::VARP>& inputs) {
    if (inputs.size() != 1 || inputs[0] == nullptr) {
        throw std::invalid_argument("Embedding: 必须提供一个输入");
    }
    const auto* info = inputs[0]->getInfo();
    if (info == nullptr || info->dim.size() != 3 || info->dim[0] <= 0 ||
        info->dim[1] <= 0 || info->dim[2] != input_feature_count_) {
        throw std::invalid_argument(
            "Embedding: 输入形状必须为[batch,entity,input_feature]");
    }
    if (info->type != halide_type_of<float>()) {
        throw std::invalid_argument("Embedding: 输入必须为float32 Tensor");
    }

    // MNN按行保存实体特征，因此该式等价于参考实现的weight转置乘以列向量。
    return {MNN::Express::_MatMul(inputs[0], weight_) + bias_};
}

MNN::Express::Module* Embedding::clone(CloneContext* context) const {
    auto* module = new Embedding;
    module->input_feature_count_ = input_feature_count_;
    module->output_feature_count_ = output_feature_count_;
    module->weight_ = weight_;
    module->bias_ = bias_;
    // 克隆模块共享参数，并保持权重、偏置的注册顺序。
    module->addParameter(module->weight_);
    module->addParameter(module->bias_);
    return cloneBaseTo(context, module);
}

}
