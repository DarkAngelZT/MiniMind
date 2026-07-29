#include "gru.hpp"

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
    std::cerr << "GRU测试失败: " << message << '\n';
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

void zero_all(MiniMind::GRU& gru) {
    for (const auto& parameter : gru.parameters()) {
        fill(parameter, std::vector<float>(parameter->getInfo()->size, 0.0f));
    }
}

void test_fixed_parameters_match_minibrain_formula() {
    MiniMind::GRU gru(2, 2);
    auto p = gru.parameters();
    fill(p[0], {0.2f, -0.3f, 0.4f, 0.1f});
    fill(p[1], {-0.1f, 0.5f, 0.3f, -0.2f});
    fill(p[2], {0.6f, -0.4f, -0.2f, 0.7f});
    fill(p[3], {0.1f, 0.2f, -0.3f, 0.4f});
    fill(p[4], {-0.2f, 0.3f, 0.5f, -0.1f});
    fill(p[5], {0.4f, -0.2f, 0.1f, 0.3f});
    fill(p[6], {0.05f, -0.1f});
    fill(p[7], {-0.2f, 0.15f});
    fill(p[8], {0.1f, -0.05f});

    const float x[] = {0.7f, -1.2f};
    const float h[] = {0.3f, -0.8f};
    auto output = gru.forwardStep(constant(x, {1, 2}), constant(h, {1, 2}));
    const float* actual = output->readMap<float>();

    auto sigmoid = [](float value) {
        return 1.0f / (1.0f + std::exp(-value));
    };
    const float z0 = sigmoid(0.7f * 0.2f + -1.2f * -0.3f +
                             0.3f * 0.1f + -0.8f * 0.2f + 0.05f);
    const float z1 = sigmoid(0.7f * 0.4f + -1.2f * 0.1f +
                             0.3f * -0.3f + -0.8f * 0.4f - 0.1f);
    const float r0 = sigmoid(0.7f * -0.1f + -1.2f * 0.5f +
                             0.3f * -0.2f + -0.8f * 0.3f - 0.2f);
    const float r1 = sigmoid(0.7f * 0.3f + -1.2f * -0.2f +
                             0.3f * 0.5f + -0.8f * -0.1f + 0.15f);
    const float rh0 = r0 * 0.3f;
    const float rh1 = r1 * -0.8f;
    const float c0 = std::tanh(0.7f * 0.6f + -1.2f * -0.4f +
                               rh0 * 0.4f + rh1 * -0.2f + 0.1f);
    const float c1 = std::tanh(0.7f * -0.2f + -1.2f * 0.7f +
                               rh0 * 0.1f + rh1 * 0.3f - 0.05f);
    expect_near(actual[0], (1.0f - z0) * 0.3f + z0 * c0,
                "固定参数第一个隐藏单元");
    expect_near(actual[1], (1.0f - z1) * -0.8f + z1 * c1,
                "固定参数第二个隐藏单元");
}

void test_update_direction_is_minibrain_specific() {
    MiniMind::GRU gru(1, 1);
    zero_all(gru);
    const float z = 0.8f;
    fill(gru.parameters()[6], {std::log(z / (1.0f - z))});
    const float x[] = {0.0f};
    const float h[] = {2.0f};
    const float actual =
        gru.forwardStep(constant(x, {1, 1}), constant(h, {1, 1}))
            ->readMap<float>()[0];
    expect_near(actual, 0.4f, "MiniBrain更新门方向");
    if (std::fabs(actual - 1.6f) < 1.0e-3f) {
        fail("错误采用了常见GRU的反向更新约定");
    }
}

void test_batch_rows_are_independent_and_shape_is_preserved() {
    MiniMind::GRU gru(2, 2);
    auto p = gru.parameters();
    for (std::size_t i = 0; i < p.size(); ++i) {
        fill(p[i], std::vector<float>(p[i]->getInfo()->size,
                                     0.03f * static_cast<float>(i + 1)));
    }
    const float x_batch[] = {1.0f, 2.0f, -3.0f, 0.5f};
    const float h_batch[] = {0.1f, 0.2f, -0.4f, 0.8f};
    const float x_second[] = {-3.0f, 0.5f};
    const float h_second[] = {-0.4f, 0.8f};
    auto batch = gru.onForward(
        {constant(x_batch, {2, 2}), constant(h_batch, {2, 2})})[0];
    auto single =
        gru.forwardStep(constant(x_second, {1, 2}),
                        constant(h_second, {1, 2}));
    if (batch->getInfo()->dim != std::vector<int>({2, 2})) {
        fail("输出形状不是[batch,hidden_size]");
    }
    const float* batch_data = batch->readMap<float>();
    const float* single_data = single->readMap<float>();
    expect_near(batch_data[2], single_data[0], "批次隔离第一个单元");
    expect_near(batch_data[3], single_data[1], "批次隔离第二个单元");
}

void test_parameters_and_clone() {
    MiniMind::GRU gru(3, 2);
    const auto parameters = gru.parameters();
    const std::vector<std::vector<int>> expected = {
        {2, 3}, {2, 3}, {2, 3}, {2, 2}, {2, 2},
        {2, 2}, {2},    {2},    {2}};
    if (parameters.size() != expected.size()) {
        fail("参数数量或注册顺序错误");
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (parameters[i]->getInfo()->dim != expected[i]) {
            fail("参数形状或注册顺序错误");
        }
        if (parameters[i]->expr().first->inputType() != VARP::TRAINABLE) {
            fail("参数不是可训练参数");
        }
    }
    std::unique_ptr<Module, void (*)(Module*)> cloned(
        Module::clone(&gru), Module::destroy);
    if (!cloned || cloned->parameters().size() != expected.size()) {
        fail("克隆后参数注册丢失");
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (cloned->parameters()[i].get() != parameters[i].get()) {
            fail("克隆未按项目约定共享参数");
        }
    }
}

void test_invalid_inputs_are_rejected() {
    MiniMind::GRU gru(2, 3);
    const float six[] = {0, 0, 0, 0, 0, 0};
    auto x = constant(six, {3, 2});
    auto h = constant(six, {2, 3});
    for (const auto& inputs :
         std::vector<std::vector<VARP>>{{x}, {x, h, h}}) {
        bool threw = false;
        try {
            (void)gru.onForward(inputs);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        if (!threw) {
            fail("接受了错误的输入数量");
        }
    }
    bool batch_threw = false;
    try {
        (void)gru.onForward({x, h});
    } catch (const std::invalid_argument&) {
        batch_threw = true;
    }
    if (!batch_threw) {
        fail("接受了不同batch大小的输入和隐藏状态");
    }
    bool rank_threw = false;
    try {
        (void)gru.forwardStep(constant(six, {1, 3, 2}),
                              constant(six, {1, 2, 3}));
    } catch (const std::invalid_argument&) {
        rank_threw = true;
    }
    if (!rank_threw) {
        fail("接受了非二维输入");
    }
}

void test_gradients_exist_for_inputs_hidden_and_parameters() {
    MiniMind::GRU gru(2, 2);
    const float x_values[] = {0.2f, -0.5f};
    const float h_values[] = {0.7f, -0.1f};
    auto x = _TrainableParam(x_values, {1, 2}, NCHW);
    auto h = _TrainableParam(h_values, {1, 2}, NCHW);
    auto output = gru.forwardStep(x, h);
    auto loss = _ReduceSum(output, {0, 1});
    std::set<VARP> requested{x, h};
    for (const auto& parameter : gru.parameters()) {
        requested.insert(parameter);
    }
    const auto gradients = MNN::OpGrad::grad(loss, requested);
    for (const auto& variable : requested) {
        const auto found = gradients.find(variable);
        if (found == gradients.end() || found->second == nullptr ||
            found->second->readMap<float>() == nullptr) {
            fail("输入、隐藏状态或参数缺少梯度");
        }
    }
}

} // namespace

int main() {
    test_fixed_parameters_match_minibrain_formula();
    test_update_direction_is_minibrain_specific();
    test_batch_rows_are_independent_and_shape_is_preserved();
    test_parameters_and_clone();
    test_invalid_inputs_are_rejected();
    test_gradients_exist_for_inputs_hidden_and_parameters();
    std::cout << "MiniMind GRU测试通过\n";
    return 0;
}
