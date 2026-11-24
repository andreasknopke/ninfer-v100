// The looped decoder's execution order; the contract is models/sensenova_loop/loop_plan.h.
// Pure host code: it is consulted while loading and while sizing state, and never on the GPU.

#include "models/sensenova_loop/loop_plan.h"

#include <stdexcept>
#include <string>

namespace ninfer::models::sensenova_loop {
namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("loop plan: ") + message); }
}

} // namespace

ExecutionPlan::ExecutionPlan(std::uint32_t physical_layers, LoopSettings loop,
                             std::vector<LayerKind> layer_kinds)
    : physical_layers_(physical_layers), loop_(loop), layer_kinds_(std::move(layer_kinds)) {
    require(physical_layers > 0, "a decoder needs at least one physical layer");
    require(layer_kinds_.size() == physical_layers,
            "layer kinds must cover every physical layer");
    require(loop.visits >= 1, "a looped range must run at least once");
    require(loop.start < loop.end, "the looped range must not be empty");
    require(loop.end <= physical_layers, "the looped range must stay inside the decoder");

    // Bound the table before allocating it: the slot count is the product of the range size and the
    // visit count, and a malformed config could otherwise ask for an absurd allocation.
    const std::uint64_t looped = static_cast<std::uint64_t>(loop.end - loop.start) * loop.visits;
    const std::uint64_t total  = looped + physical_layers - (loop.end - loop.start);
    require(total <= (1ULL << 32) - 1, "the execution order overflows a slot index");

    slots_.reserve(static_cast<std::size_t>(total));
    slot_kinds_.reserve(static_cast<std::size_t>(total));
    family_indices_.reserve(static_cast<std::size_t>(total));
    slots_by_layer_.resize(physical_layers);

    const auto append = [&](std::uint32_t layer, std::uint32_t visit) {
        const std::uint32_t index = static_cast<std::uint32_t>(slots_.size());
        const bool full           = layer_kinds_[layer] == LayerKind::FullAttention;
        slots_.push_back(ExecutionSlot{layer, visit});
        slot_kinds_.push_back(layer_kinds_[layer]);
        if (full) {
            ++full_attention_slots_;
            family_indices_.push_back(full_attention_slots_ - 1);
        } else {
            ++linear_attention_slots_;
            family_indices_.push_back(linear_attention_slots_ - 1);
        }
        slots_by_layer_[layer].push_back(index);
    };

    for (std::uint32_t layer = 0; layer < loop.start; ++layer) { append(layer, 0); }
    for (std::uint32_t visit = 1; visit <= loop.visits; ++visit) {
        for (std::uint32_t layer = loop.start; layer < loop.end; ++layer) { append(layer, visit); }
    }
    for (std::uint32_t layer = loop.end; layer < physical_layers; ++layer) { append(layer, 0); }
}

} // namespace ninfer::models::sensenova_loop
