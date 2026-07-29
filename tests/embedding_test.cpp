#include "embedding.hpp"

#include "OpGrad.hpp"
#include <MNN/expr/ExprCreator.hpp>

#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using MNN::Express::Module;
using MNN::Express::NCHW;
using MNN::Express::VARP;
using MNN::Express::_Const;
using MNN::Express::_ReduceSum;
using MNN::Express::_TrainableParam;

[[noreturn]] void fail(const std::string& message) {
    std::cerr << "Embedding测试失败: " << message << '\n';
    std::exit(1);
}

void expect_near(float actual, float expected, const std::string& context,
                 float tolerance = 1.0e-5f) {
    if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) {
        fail(context + "，期望 " + std::to_string(expected) + "，实际 " +
             std::to_string(actual));
    }
}

VARP constant(const float* values, const std::vector<int>& shape) {
    return _Const(values, shape, NCHW);
}

void fill(VARP parameter, const std::vector<float>& values) {
    const auto* info = parameter->getInfo();
    if (info == nullptr || info->size != values.size()) {
        fail("测试参数尺寸错误");
    }
    float* destination = parameter->writeMap<float>();
    for (std::size_t i = 0; i < values.size(); ++i) {
        destination[i] = values[i];
    }
}

void expect_invalid(const std::string& context,
                    const std::function<void()>& operation) {
    bool threw = false;
    try {
        operation();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    if (!threw) {
        fail(context);
    }
}

void test_fixed_parameters_match_reference_projection() {
    MiniMind::Embedding embedding(2, 3);
    const auto parameters = embedding.parameters();
    fill(parameters[0], {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    fill(parameters[1], {0.5f, -0.5f, 1.0f});

    const float input_values[] = {1.0f, 2.0f, -1.0f, 0.5f};
    auto output = embedding.forward(constant(input_values, {1, 2, 2}));
    if (output->getInfo()->dim != std::vector<int>({1, 2, 3})) {
        fail("输出形状不是[batch,entity,out_feature]");
    }

    const float expected[] = {9.5f, 11.5f, 16.0f, 1.5f, 0.0f, 1.0f};
    const float* actual = output->readMap<float>();
    for (int i = 0; i < 6; ++i) {
        expect_near(actual[i], expected[i], "固定参数前向结果");
    }
}

void test_batch_and_entity_are_independent() {
    MiniMind::Embedding embedding(2, 1);
    fill(embedding.parameters()[0], {2.0f, -3.0f});
    fill(embedding.parameters()[1], {0.25f});

    const float values[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        -1.0f, 5.0f, 0.5f, -2.0f,
    };
    auto output = embedding.forward(constant(values, {2, 2, 2}));
    if (output->getInfo()->dim != std::vector<int>({2, 2, 1})) {
        fail("批次前向输出形状错误");
    }
    const float expected[] = {-3.75f, -5.75f, -16.75f, 7.25f};
    const float* actual = output->readMap<float>();
    for (int i = 0; i < 4; ++i) {
        expect_near(actual[i], expected[i], "batch或entity发生混合");
    }
}

void test_parameters_and_clone() {
    MiniMind::Embedding embedding(4, 3);
    const auto parameters = embedding.parameters();
    if (parameters.size() != 2) {
        fail("参数数量或注册顺序错误");
    }
    const std::vector<std::vector<int>> expected_shapes = {{4, 3}, {3}};
    for (std::size_t i = 0; i < parameters.size(); ++i) {
        if (parameters[i]->getInfo()->dim != expected_shapes[i]) {
            fail("参数形状或注册顺序错误");
        }
        if (parameters[i]->expr().first->inputType() != VARP::TRAINABLE) {
            fail("参数不是可训练参数");
        }
    }

    std::unique_ptr<Module, void (*)(Module*)> cloned(
        Module::clone(&embedding), Module::destroy);
    if (!cloned || cloned->parameters().size() != 2) {
        fail("克隆后参数注册丢失");
    }
    for (std::size_t i = 0; i < parameters.size(); ++i) {
        if (cloned->parameters()[i].get() != parameters[i].get()) {
            fail("克隆未共享原模块参数");
        }
    }
}

void test_invalid_configuration_and_inputs_are_rejected() {
    expect_invalid("接受了零输入特征数",
                   [] { MiniMind::Embedding embedding(0, 2); });
    expect_invalid("接受了零输出特征数",
                   [] { MiniMind::Embedding embedding(2, 0); });

    MiniMind::Embedding embedding(2, 3);
    const float values[] = {1, 2, 3, 4, 5, 6, 7, 8};
    auto valid = constant(values, {2, 2, 2});
    expect_invalid("接受了空输入列表",
                   [&] { (void)embedding.onForward({}); });
    expect_invalid("接受了多个输入",
                   [&] { (void)embedding.onForward({valid, valid}); });
    expect_invalid("接受了空Tensor",
                   [&] { (void)embedding.onForward({nullptr}); });
    expect_invalid("接受了非三维输入", [&] {
        (void)embedding.forward(constant(values, {2, 4}));
    });
    expect_invalid("接受了错误的feature维度", [&] {
        (void)embedding.forward(constant(values, {1, 2, 4}));
    });
}

void test_gradients_exist_for_input_weight_and_bias() {
    MiniMind::Embedding embedding(2, 3);
    fill(embedding.parameters()[0],
         {0.2f, -0.1f, 0.3f, 0.4f, 0.5f, -0.2f});
    fill(embedding.parameters()[1], {0.1f, -0.2f, 0.05f});
    const float values[] = {0.5f, -1.0f, 2.0f, 0.25f};
    auto input = _TrainableParam(values, {1, 2, 2}, NCHW);
    auto loss = _ReduceSum(embedding.forward(input), {0, 1, 2});

    std::set<VARP> requested{input};
    for (const auto& parameter : embedding.parameters()) {
        requested.insert(parameter);
    }
    const auto gradients = MNN::OpGrad::grad(loss, requested);
    for (const auto& variable : requested) {
        const auto iterator = gradients.find(variable);
        if (iterator == gradients.end() || iterator->second == nullptr ||
            iterator->second->getInfo() == nullptr) {
            fail("输入或参数缺少梯度");
        }
        const float* data = iterator->second->readMap<float>();
        bool has_nonzero = false;
        for (std::size_t i = 0; i < iterator->second->getInfo()->size; ++i) {
            if (!std::isfinite(data[i])) {
                fail("梯度包含非有限值");
            }
            has_nonzero = has_nonzero || std::fabs(data[i]) > 1.0e-7f;
        }
        if (!has_nonzero) {
            fail("输入或参数梯度全为零");
        }
    }
}

}

int main() {
    test_fixed_parameters_match_reference_projection();
    test_batch_and_entity_are_independent();
    test_parameters_and_clone();
    test_invalid_configuration_and_inputs_are_rejected();
    test_gradients_exist_for_input_weight_and_bias();
    std::cout << "MiniMind Embedding测试通过\n";
    return 0;
}
