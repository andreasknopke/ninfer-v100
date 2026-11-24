#pragma once

// The execution order of a looped decoder, and the state-plane counts that follow from it.
//
// A looped model runs a contiguous range of its physical layers more than once. The weights of a
// physical layer are materialized once and shared by every pass over it, but the state a layer
// accumulates while it runs -- KV planes, recurrent and convolution state, replay records,
// checkpoints -- belongs to the pass, not to the layer. Two passes over the same physical layer
// therefore need two independent state slots, and everything downstream that sizes or indexes a
// per-layer buffer has to work in slots rather than layers. This module is the single place that
// mapping is written down.

#include <cstdint>
#include <vector>

namespace ninfer::models::sensenova_loop {

// What a physical layer does. The two kinds keep separate state planes, so the plan has to know
// which kind each slot runs in order to report the plane counts.
enum class LayerKind : std::uint8_t { FullAttention, LinearAttention };

// The release `loop_settings` block reduced to what the execution order depends on: the physical
// layer range `[start, end)` that repeats, and how many times it runs.
struct LoopSettings {
    std::uint32_t start  = 0;
    std::uint32_t end    = 0;
    std::uint32_t visits = 1;
};

// One entry of the execution order.
struct ExecutionSlot {
    // Which physical layer's weights this slot runs.
    std::uint32_t physical_layer = 0;
    // Which pass over the looped range this slot is, 1-based. Zero for a layer outside the range,
    // where there is no pass number to report.
    std::uint32_t visit = 0;
};

// The execution order of one decoder, plus the per-family state-plane counts it implies. Immutable
// once built; every accessor is a table lookup.
class ExecutionPlan {
public:
    // `layer_kinds` is indexed by physical layer. Throws std::invalid_argument when the settings
    // cannot describe an execution order.
    ExecutionPlan(std::uint32_t physical_layers, LoopSettings loop,
                  std::vector<LayerKind> layer_kinds);

    [[nodiscard]] std::uint32_t slot_count() const noexcept {
        return static_cast<std::uint32_t>(slots_.size());
    }
    [[nodiscard]] const ExecutionSlot& slot(std::uint32_t index) const { return slots_.at(index); }

    [[nodiscard]] std::uint32_t physical_layer(std::uint32_t index) const {
        return slots_.at(index).physical_layer;
    }
    [[nodiscard]] std::uint32_t visit(std::uint32_t index) const { return slots_.at(index).visit; }
    [[nodiscard]] LayerKind kind(std::uint32_t index) const { return slot_kinds_.at(index); }
    [[nodiscard]] bool is_full_attention(std::uint32_t index) const {
        return slot_kinds_.at(index) == LayerKind::FullAttention;
    }

    // The kind of a physical layer, independent of any slot. Weights are indexed this way, while
    // state planes are indexed by slot.
    [[nodiscard]] LayerKind layer_kind(std::uint32_t layer) const {
        return layer_kinds_.at(layer);
    }

    // Index of this slot among the slots of its own kind. This is the index into that kind's state
    // planes, so two slots running the same physical layer get different plane indices.
    [[nodiscard]] std::uint32_t family_index(std::uint32_t index) const {
        return family_indices_.at(index);
    }

    [[nodiscard]] std::uint32_t full_attention_slots() const noexcept {
        return full_attention_slots_;
    }
    [[nodiscard]] std::uint32_t linear_attention_slots() const noexcept {
        return linear_attention_slots_;
    }

    [[nodiscard]] std::uint32_t physical_layers() const noexcept { return physical_layers_; }
    [[nodiscard]] const LoopSettings& loop() const noexcept { return loop_; }

    // Every slot that runs `layer`, in execution order. A layer inside the looped range yields
    // `visits` entries; any other layer yields exactly one.
    [[nodiscard]] const std::vector<std::uint32_t>& slots_for(std::uint32_t layer) const {
        return slots_by_layer_.at(layer);
    }

private:
    std::uint32_t physical_layers_ = 0;
    LoopSettings loop_;
    std::vector<ExecutionSlot> slots_;
    std::vector<LayerKind> layer_kinds_;
    std::vector<LayerKind> slot_kinds_;
    std::vector<std::uint32_t> family_indices_;
    std::vector<std::vector<std::uint32_t>> slots_by_layer_;
    std::uint32_t full_attention_slots_   = 0;
    std::uint32_t linear_attention_slots_ = 0;
};

} // namespace ninfer::models::sensenova_loop
