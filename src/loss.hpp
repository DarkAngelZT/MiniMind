#pragma once

#include <MNN/expr/Expr.hpp>

namespace MiniMind::loss {

// Mean squared error over every element, matching PyTorch reduction="mean".
MNN::Express::VARP mse(const MNN::Express::VARP& prediction,
                       const MNN::Express::VARP& target);

// Multi-label categorical cross entropy for [batch, classes] probabilities
// and a same-shaped target. Positive labels are summed per sample, then
// batches are averaged; this intentionally is not binary cross entropy.
MNN::Express::VARP cross_entropy_multi(const MNN::Express::VARP& probabilities,
                                       const MNN::Express::VARP& target);

} // namespace MiniMind::loss
