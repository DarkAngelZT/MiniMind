#include "loss.hpp"

#include <MNN/expr/ExprCreator.hpp>

#include <stdexcept>
#include <string>

namespace MiniMind::loss {
namespace {

void require_same_shape(const MNN::Express::VARP& lhs,
                        const MNN::Express::VARP& rhs,
                        const char* loss_name) {
    if (lhs == nullptr || rhs == nullptr) {
        throw std::invalid_argument(std::string(loss_name) +
                                    ": inputs must not be null");
    }
    const auto* lhs_info = lhs->getInfo();
    const auto* rhs_info = rhs->getInfo();
    if (lhs_info == nullptr || rhs_info == nullptr ||
        lhs_info->dim != rhs_info->dim) {
        throw std::invalid_argument(std::string(loss_name) +
                                    ": input shapes must match");
    }
}

} // namespace

MNN::Express::VARP mse(const MNN::Express::VARP& prediction,
                       const MNN::Express::VARP& target) {
    require_same_shape(prediction, target, "mse");
    return MNN::Express::_ReduceMean(
        MNN::Express::_Square(prediction - target), {});
}

MNN::Express::VARP cross_entropy_multi(const MNN::Express::VARP& probabilities,
                                       const MNN::Express::VARP& target) {
    require_same_shape(probabilities, target, "cross_entropy_multi");
    const auto* info = probabilities->getInfo();
    if (info->dim.size() != 2 || info->dim[0] <= 0 || info->dim[1] <= 0) {
        throw std::invalid_argument(
            "cross_entropy_multi: inputs must have shape [batch, classes]");
    }

    using namespace MNN::Express;
    const auto clipped = _Maximum(probabilities, _Scalar<float>(1.0e-7f));
    const auto total_loss = _Negative(_ReduceSum(target * _Log(clipped), {}));
    return total_loss / _Scalar<float>(static_cast<float>(info->dim[0]));
}

} // namespace MiniMind::loss
