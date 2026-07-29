#include "state_pooling.hpp"

#include "OpGrad.hpp"
#include <MNN/expr/ExprCreator.hpp>

#include <cmath>
#include <cstdlib>
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
    std::cerr << "StatePooling测试失败: " << message << '\n';
    std::exit(1);
}

void expect_near(float actual, float expected, const std::string& context,
                 float tolerance = 1.0e-5f) {
    if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) {
        fail(context + ": 期望 " + std::to_string(expected) +
             "，实际 " + std::to_string(actual));
    }
}

VARP constant(const float* values, const std::vector<int>& shape) {
    return _Const(values, shape, NCHW);
}

void expect_values(const VARP& value, const std::vector<float>& expected,
                   const std::string& context) {
    const auto* info = value->getInfo();
    if (info == nullptr || info->size != expected.size()) {
        fail(context + "的元素数量错误");
    }
    const float* actual = value->readMap<float>();
    if (actual == nullptr) {
        fail(context + "读取失败");
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        expect_near(actual[i], expected[i],
                    context + "第" + std::to_string(i) + "项");
    }
}

void test_forward_matches_minibrain_mean_and_max_formula() {
    MiniMind::StatePooling pooling(2);
    const float input[] = {
        1.0f, 10.0f, 3.0f, 6.0f, -2.0f, 8.0f,
        4.0f, -1.0f, 0.0f, 5.0f, 2.0f, 3.0f};
    auto output = pooling.forward(constant(input, {2, 3, 2}));

    if (output->getInfo()->dim != std::vector<int>({2, 4})) {
        fail("输出形状不是[batch,2*feature]");
    }
    expect_values(output, {2.0f / 3.0f, 8.0f, 3.0f, 10.0f,
                           2.0f, 7.0f / 3.0f, 4.0f, 5.0f},
                  "固定输入前向结果");
}

void test_batch_isolation_and_entity_feature_semantics() {
    MiniMind::StatePooling pooling(3);
    const float batch_input[] = {
        1.0f, 100.0f, -4.0f, 5.0f, 20.0f, 8.0f,
        -3.0f, 6.0f, 9.0f, 7.0f, -2.0f, 3.0f};
    const float second_input[] = {-3.0f, 6.0f, 9.0f, 7.0f, -2.0f, 3.0f};
    auto batch = pooling.forward(constant(batch_input, {2, 2, 3}));
    auto second = pooling.forward(constant(second_input, {1, 2, 3}));

    if (batch->getInfo()->dim != std::vector<int>({2, 6}) ||
        second->getInfo()->dim != std::vector<int>({1, 6})) {
        fail("批次前向没有保持[batch,2*feature]布局");
    }
    const float* batch_data = batch->readMap<float>();
    const float* second_data = second->readMap<float>();
    for (int i = 0; i < 6; ++i) {
        expect_near(batch_data[6 + i], second_data[i],
                    "不同batch之间发生数据混合");
    }
    expect_values(second, {2.0f, 2.0f, 6.0f, 7.0f, 6.0f, 9.0f},
                  "entity轴池化和feature轴保留结果");
}

void test_has_no_parameters_and_clone_preserves_behavior() {
    MiniMind::StatePooling pooling(2);
    if (!pooling.parameters().empty()) {
        fail("无参数池化层注册了可训练参数");
    }
    std::unique_ptr<Module, void (*)(Module*)> cloned(
        Module::clone(&pooling), Module::destroy);
    if (!cloned || !cloned->parameters().empty()) {
        fail("克隆后的无参数约定错误");
    }
    const float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    expect_values(cloned->forward(constant(input, {1, 2, 2})),
                  {2.0f, 3.0f, 3.0f, 4.0f}, "克隆层前向结果");
}

void test_invalid_construction_and_inputs_are_rejected() {
    bool constructor_threw = false;
    try {
        MiniMind::StatePooling invalid(0);
    } catch (const std::invalid_argument&) {
        constructor_threw = true;
    }
    if (!constructor_threw) {
        fail("接受了非正数feature_count");
    }

    MiniMind::StatePooling pooling(2);
    const float values[] = {1, 2, 3, 4, 5, 6};
    const auto valid = constant(values, {1, 3, 2});
    for (const auto& inputs :
         std::vector<std::vector<VARP>>{{}, {valid, valid}, {nullptr}}) {
        bool threw = false;
        try {
            (void)pooling.onForward(inputs);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        if (!threw) {
            fail("接受了错误的输入数量或空输入");
        }
    }

    for (const auto& invalid :
         std::vector<VARP>{constant(values, {3, 2}),
                           constant(values, {1, 2, 3})}) {
        bool threw = false;
        try {
            (void)pooling.forward(invalid);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        if (!threw) {
            fail("接受了错误秩或错误feature维度");
        }
    }
}

void test_input_gradient_exists() {
    MiniMind::StatePooling pooling(2);
    const float input_values[] = {1.0f, 5.0f, 3.0f, -2.0f, 2.0f, 4.0f};
    auto input = _TrainableParam(input_values, {1, 3, 2}, NCHW);
    auto loss = _ReduceSum(pooling.forward(input), {0, 1});
    const auto gradients = MNN::OpGrad::grad(loss, std::set<VARP>{input});
    const auto found = gradients.find(input);
    if (found == gradients.end() || found->second == nullptr) {
        fail("输入缺少梯度");
    }
    const auto* info = found->second->getInfo();
    const float* values = found->second->readMap<float>();
    if (info == nullptr || info->dim != std::vector<int>({1, 3, 2}) ||
        values == nullptr) {
        fail("输入梯度形状或数据错误");
    }
    for (std::size_t i = 0; i < info->size; ++i) {
        if (!std::isfinite(values[i])) {
            fail("输入梯度包含非有限值");
        }
    }
}

} // 匿名命名空间

int main() {
    test_forward_matches_minibrain_mean_and_max_formula();
    test_batch_isolation_and_entity_feature_semantics();
    test_has_no_parameters_and_clone_preserves_behavior();
    test_invalid_construction_and_inputs_are_rejected();
    test_input_gradient_exists();
    std::cout << "MiniMind StatePooling测试通过\n";
    return 0;
}
