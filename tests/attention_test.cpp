#include "attention.hpp"

#include <MNN/expr/ExprCreator.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using MNN::Express::Module;
using MNN::Express::NCHW;
using MNN::Express::VARP;
using MNN::Express::_Const;

[[noreturn]] void fail(const std::string& message) {
    std::cerr << "Attention测试失败: " << message << '\n';
    std::exit(1);
}

void expect_near(float actual, float expected, const std::string& context,
                 float tolerance = 1.0e-4f) {
    if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) {
        fail(context + ": 期望 " + std::to_string(expected) +
             "，实际 " + std::to_string(actual));
    }
}

VARP constant(const float* values, const std::vector<int>& shape) {
    return _Const(values, shape, NCHW);
}

void set_identity_weights(MiniMind::Attention& attention) {
    auto parameters = attention.parameters();
    if (parameters.size() != 3) {
        fail("参数数量不是3");
    }
    const float identity[] = {1.0f, 0.0f, 0.0f, 1.0f};
    for (const auto& parameter : parameters) {
        float* data = parameter->writeMap<float>();
        for (int i = 0; i < 4; ++i) {
            data[i] = identity[i];
        }
    }
}

void test_fixed_weights_produce_exact_attention() {
    MiniMind::Attention attention(2, 2);
    set_identity_weights(attention);
    const float input_values[] = {1.0f, 0.0f,
                                  0.0f, 1.0f};
    auto output = attention.forward(constant(input_values, {1, 2, 2}));
    const auto* info = output->getInfo();
    if (info == nullptr || info->dim != std::vector<int>({1, 2, 2})) {
        fail("输出形状不是[1,2,2]");
    }

    const float diagonal = 0.66976155f;
    const float expected[] = {diagonal, 1.0f - diagonal,
                              1.0f - diagonal, diagonal};
    const float* actual = output->readMap<float>();
    for (int i = 0; i < 4; ++i) {
        expect_near(actual[i], expected[i], "固定权重前向结果");
    }
}

void test_batches_do_not_mix() {
    MiniMind::Attention attention(2, 2);
    set_identity_weights(attention);
    const float batch_values[] = {1.0f, 0.0f, 0.0f, 1.0f,
                                  2.0f, 1.0f, 1.0f, 2.0f};
    const float second_values[] = {2.0f, 1.0f, 1.0f, 2.0f};
    auto batch_output = attention.forward(constant(batch_values, {2, 2, 2}));
    auto single_output = attention.forward(constant(second_values, {1, 2, 2}));
    const float* batch_data = batch_output->readMap<float>();
    const float* single_data = single_output->readMap<float>();
    for (int i = 0; i < 4; ++i) {
        expect_near(batch_data[4 + i], single_data[i], "批次隔离");
    }
}

void test_parameters_are_registered_trainable_and_cloned() {
    MiniMind::Attention attention(3, 2);
    auto parameters = attention.parameters();
    const std::vector<std::vector<int>> expected_shapes = {
        {2, 3}, {2, 3}, {3, 3}};
    if (parameters.size() != expected_shapes.size()) {
        fail("注册参数数量错误");
    }
    for (std::size_t i = 0; i < parameters.size(); ++i) {
        if (parameters[i]->getInfo()->dim != expected_shapes[i]) {
            fail("注册参数形状错误");
        }
        if (parameters[i]->expr().first->inputType() != VARP::TRAINABLE) {
            fail("参数不是TRAINABLE");
        }
    }

    std::unique_ptr<Module, void (*)(Module*)> cloned(
        Module::clone(&attention), Module::destroy);
    if (!cloned || cloned->parameters().size() != 3) {
        fail("克隆后参数未注册");
    }
    for (std::size_t i = 0; i < parameters.size(); ++i) {
        if (cloned->parameters()[i].get() != parameters[i].get()) {
            fail("克隆未按Module约定共享参数");
        }
    }
}

void test_invalid_inputs_are_rejected() {
    MiniMind::Attention attention(2, 2);
    const float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto rank_two = constant(values, {2, 2});
    bool rank_threw = false;
    try {
        (void)attention.forward(rank_two);
    } catch (const std::invalid_argument&) {
        rank_threw = true;
    }
    if (!rank_threw) {
        fail("接受了非[B,E,F]输入");
    }

    const float wrong_feature_values[] = {1, 2, 3, 4, 5, 6};
    bool feature_threw = false;
    try {
        (void)attention.forward(constant(wrong_feature_values, {1, 2, 3}));
    } catch (const std::invalid_argument&) {
        feature_threw = true;
    }
    if (!feature_threw) {
        fail("接受了错误feature维度");
    }

    bool count_threw = false;
    try {
        (void)attention.onForward({rank_two, rank_two});
    } catch (const std::invalid_argument&) {
        count_threw = true;
    }
    if (!count_threw) {
        fail("接受了多个输入");
    }
}

}

int main() {
    test_fixed_weights_produce_exact_attention();
    test_batches_do_not_mix();
    test_parameters_are_registered_trainable_and_cloned();
    test_invalid_inputs_are_rejected();
    std::cout << "MiniMind Attention测试通过\n";
    return 0;
}
