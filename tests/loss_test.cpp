#include "loss.hpp"

#include "OpGrad.hpp"
#include <MNN/expr/ExprCreator.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

namespace {

using MNN::Express::NCHW;
using MNN::Express::VARP;
using MNN::Express::_Const;
using MNN::Express::_TrainableParam;

[[noreturn]] void fail(const std::string& message) {
    std::cerr << "Loss test failed: " << message << '\n';
    std::exit(1);
}

void expect_near(float actual, float expected, const std::string& context,
                 float tolerance = 1.0e-4f) {
    if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) {
        fail(context + ": expected " + std::to_string(expected) +
             ", got " + std::to_string(actual));
    }
}

float scalar_value(const VARP& value) {
    const float* data = value->readMap<float>();
    if (data == nullptr) {
        fail("MNN returned no scalar data");
    }
    return data[0];
}

VARP constant(const float* values, int batch, int classes) {
    return _Const(values, {batch, classes}, NCHW);
}

void test_mse_means_all_elements() {
    const float predictions[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float targets[] = {0.0f, 2.0f, 5.0f, 1.0f};
    expect_near(scalar_value(MiniMind::loss::mse(
                    constant(predictions, 2, 2), constant(targets, 2, 2))),
                3.5f, "MSE all-element mean");
}

void test_multi_label_cross_entropy_sums_labels_then_means_batch() {
    const float probabilities[] = {0.5f, 0.25f, 0.25f,
                                   0.2f, 0.3f, 0.5f};
    const float targets[] = {1.0f, 1.0f, 0.0f,
                             0.0f, 1.0f, 1.0f};
    expect_near(scalar_value(MiniMind::loss::cross_entropy_multi(
                    constant(probabilities, 2, 3), constant(targets, 2, 3))),
                1.9882808f, "multi-label cross entropy");
}

void test_cross_entropy_clips_zero_probability_to_epsilon() {
    const float probabilities[] = {0.0f, 1.0f};
    const float targets[] = {1.0f, 0.0f};
    expect_near(scalar_value(MiniMind::loss::cross_entropy_multi(
                    constant(probabilities, 1, 2), constant(targets, 1, 2))),
                16.1180957f, "zero-probability epsilon clipping");
}

void test_losses_reject_shape_mismatch() {
    const float four[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float six[] = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f};

    bool mse_threw = false;
    try {
        (void)MiniMind::loss::mse(constant(four, 2, 2), constant(six, 2, 3));
    } catch (const std::invalid_argument&) {
        mse_threw = true;
    }
    if (!mse_threw) {
        fail("MSE accepted mismatched shapes");
    }

    bool cross_entropy_threw = false;
    try {
        (void)MiniMind::loss::cross_entropy_multi(
            constant(four, 2, 2), constant(six, 2, 3));
    } catch (const std::invalid_argument&) {
        cross_entropy_threw = true;
    }
    if (!cross_entropy_threw) {
        fail("cross entropy accepted mismatched shapes");
    }
}

void test_mse_graph_supports_automatic_differentiation() {
    const float predictions[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float targets[] = {0.0f, 2.0f, 5.0f, 1.0f};
    auto parameter = _TrainableParam(predictions, {2, 2}, NCHW);
    auto loss = MiniMind::loss::mse(parameter, constant(targets, 2, 2));
    auto gradients = MNN::OpGrad::grad(loss, std::set<VARP>{parameter});
    auto found = gradients.find(parameter);
    if (found == gradients.end()) {
        fail("MSE graph produced no gradient");
    }
    const float expected[] = {0.5f, 0.0f, -1.0f, 1.5f};
    const float* actual = found->second->readMap<float>();
    for (int i = 0; i < 4; ++i) {
        expect_near(actual[i], expected[i],
                    "MSE gradient at index " + std::to_string(i));
    }
}

} // namespace

int main() {
    test_mse_means_all_elements();
    test_multi_label_cross_entropy_sums_labels_then_means_batch();
    test_cross_entropy_clips_zero_probability_to_epsilon();
    test_losses_reject_shape_mismatch();
    test_mse_graph_supports_automatic_differentiation();
    std::cout << "MiniMind loss tests passed\n";
    return 0;
}
