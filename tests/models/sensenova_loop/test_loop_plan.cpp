// The looped decoder's execution order, checked against the release geometry.
//
// Host-only: the plan is a table built from the config, so this runs with the GPU hidden and never
// touches CUDA. What it pins is the part everything else is sized from -- how many state planes
// exist and which pass of which layer each one belongs to -- because a wrong count here does not
// fail loudly, it silently mixes the state of two visits of the same layer.

#include "models/sensenova_loop/loop_plan.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::models::sensenova_loop::ExecutionPlan;
using ninfer::models::sensenova_loop::LayerKind;
using ninfer::models::sensenova_loop::LoopSettings;

int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        ++failures;
        std::printf("  FAIL: %s\n", what.c_str());
    }
}

template <typename T>
void check_eq(T actual, T expected, const std::string& what) {
    if (actual != expected) {
        ++failures;
        std::printf("  FAIL: %s -- got %lld, want %lld\n", what.c_str(),
                    static_cast<long long>(actual), static_cast<long long>(expected));
    }
}

// The release `text_config.layer_types`: full attention every fourth layer starting at 3.
std::vector<LayerKind> release_layer_types() {
    std::vector<LayerKind> kinds(32, LayerKind::LinearAttention);
    for (std::uint32_t layer : {3U, 7U, 11U, 15U, 19U, 23U, 27U, 31U}) {
        kinds[layer] = LayerKind::FullAttention;
    }
    return kinds;
}

LoopSettings release_loop() { return LoopSettings{4, 28, 2}; }

// The execution order the model reference documents for the release.
std::vector<std::uint32_t> release_order() {
    std::vector<std::uint32_t> order;
    for (std::uint32_t layer = 0; layer < 4; ++layer) { order.push_back(layer); }
    for (std::uint32_t pass = 0; pass < 2; ++pass) {
        for (std::uint32_t layer = 4; layer < 28; ++layer) { order.push_back(layer); }
    }
    for (std::uint32_t layer = 28; layer < 32; ++layer) { order.push_back(layer); }
    return order;
}

void test_release_geometry() {
    std::printf("release geometry\n");
    const ExecutionPlan plan(32, release_loop(), release_layer_types());

    // The three counts the whole runtime is sized from. They are asserted in the reference too, so
    // a change here means either the plan or the model changed.
    check_eq(plan.slot_count(), 56U, "slot count");
    check_eq(plan.full_attention_slots(), 14U, "full attention state planes");
    check_eq(plan.linear_attention_slots(), 42U, "linear attention state planes");
    check_eq(plan.physical_layers(), 32U, "physical layer count");

    const std::vector<std::uint32_t> want = release_order();
    check_eq(plan.slot_count(), static_cast<std::uint32_t>(want.size()), "order length");
    for (std::uint32_t slot = 0; slot < plan.slot_count(); ++slot) {
        check_eq(plan.physical_layer(slot), want[slot],
                 "execution order at slot " + std::to_string(slot));
    }
}

void test_visit_numbers() {
    std::printf("visit numbers\n");
    const ExecutionPlan plan(32, release_loop(), release_layer_types());

    // A layer outside the looped range runs once and has no pass number.
    for (std::uint32_t slot : {0U, 1U, 2U, 3U, 52U, 53U, 54U, 55U}) {
        check_eq(plan.visit(slot), 0U, "visit outside the looped range at slot " +
                                           std::to_string(slot));
    }
    // The first pass of the range is 1, the second is 2, and the pass number restarts at the range
    // boundary rather than counting up over the whole order.
    for (std::uint32_t slot = 4; slot < 28; ++slot) {
        check_eq(plan.visit(slot), 1U, "first pass at slot " + std::to_string(slot));
    }
    for (std::uint32_t slot = 28; slot < 52; ++slot) {
        check_eq(plan.visit(slot), 2U, "second pass at slot " + std::to_string(slot));
    }
}

void test_slot_isolation() {
    std::printf("two visits of one layer stay separate\n");
    const ExecutionPlan plan(32, release_loop(), release_layer_types());

    // Slot 4 and slot 28 both run physical layer 4. They share weights and must never share state,
    // which here means they must land on different planes of the same family.
    check_eq(plan.physical_layer(4), plan.physical_layer(28), "slot 4 and 28 share weights");
    check_eq(plan.kind(4), plan.kind(28), "slot 4 and 28 share a mixer kind");
    check(plan.family_index(4) != plan.family_index(28),
          "slot 4 and 28 use different state planes");

    // Same for a full attention layer: physical layer 7 runs at slots 7 and 31.
    check_eq(plan.physical_layer(7), 7U, "slot 7 runs layer 7");
    check_eq(plan.physical_layer(31), 7U, "slot 31 runs layer 7");
    check(plan.is_full_attention(7) && plan.is_full_attention(31),
          "both passes of layer 7 are full attention");
    check(plan.family_index(7) != plan.family_index(31),
          "the two passes of layer 7 use different KV planes");

    // A layer that runs once has exactly one slot, and a looped layer has exactly `visits`.
    for (std::uint32_t layer = 0; layer < 32; ++layer) {
        const std::size_t want = (layer >= 4 && layer < 28) ? 2 : 1;
        check_eq(plan.slots_for(layer).size(), want,
                 "slot count for physical layer " + std::to_string(layer));
    }
    // slots_for must agree with the order it was read off.
    for (std::uint32_t slot = 0; slot < plan.slot_count(); ++slot) {
        const auto& hits = plan.slots_for(plan.physical_layer(slot));
        bool found         = false;
        for (std::uint32_t hit : hits) { found = found || hit == slot; }
        check(found, "slots_for misses slot " + std::to_string(slot));
    }
}

void test_family_indices() {
    std::printf("family indices\n");
    const ExecutionPlan plan(32, release_loop(), release_layer_types());

    // Each family's plane indices must be exactly 0..count-1 with no gaps and no repeats, otherwise
    // two slots would write the same plane while another stayed unwritten.
    std::vector<char> full_seen(plan.full_attention_slots(), 0);
    std::vector<char> linear_seen(plan.linear_attention_slots(), 0);
    for (std::uint32_t slot = 0; slot < plan.slot_count(); ++slot) {
        const std::uint32_t index = plan.family_index(slot);
        if (plan.is_full_attention(slot)) {
            check(index < full_seen.size(), "full attention plane index in range");
            check(full_seen[index] == 0, "full attention plane index reused");
            full_seen[index] = 1;
        } else {
            check(index < linear_seen.size(), "linear attention plane index in range");
            check(linear_seen[index] == 0, "linear attention plane index reused");
            linear_seen[index] = 1;
        }
    }
    for (std::size_t index = 0; index < full_seen.size(); ++index) {
        check(full_seen[index] != 0, "full attention plane " + std::to_string(index) + " unused");
    }
    for (std::size_t index = 0; index < linear_seen.size(); ++index) {
        check(linear_seen[index] != 0,
              "linear attention plane " + std::to_string(index) + " unused");
    }

    // The plane counts must follow from the layer kinds, not be a second copy of the same fact.
    std::uint32_t full_layers = 0;
    for (std::uint32_t layer = 0; layer < 32; ++layer) {
        if (plan.layer_kind(layer) == LayerKind::FullAttention) {
            full_layers += (layer >= 4 && layer < 28) ? 2U : 1U;
        }
    }
    check_eq(plan.full_attention_slots(), full_layers, "full attention planes follow layer kinds");
}

void test_other_settings() {
    std::printf("other loop settings\n");

    // A single visit is the plain order: every layer once, no pass numbers, 32 slots.
    {
        const ExecutionPlan plan(32, LoopSettings{4, 28, 1}, release_layer_types());
        check_eq(plan.slot_count(), 32U, "single visit slot count");
        check_eq(plan.full_attention_slots(), 8U, "single visit full planes");
        check_eq(plan.linear_attention_slots(), 24U, "single visit linear planes");
        for (std::uint32_t slot = 0; slot < plan.slot_count(); ++slot) {
            check_eq(plan.physical_layer(slot), slot, "single visit order");
        }
    }

    // Three visits.
    {
        const ExecutionPlan plan(32, LoopSettings{4, 28, 3}, release_layer_types());
        check_eq(plan.slot_count(), 80U, "three visit slot count");
        check_eq(plan.full_attention_slots(), 20U, "three visit full planes");
        check_eq(plan.linear_attention_slots(), 60U, "three visit linear planes");
        check_eq(plan.physical_layer(52), 4U, "third visit starts at slot 52");
        check_eq(plan.visit(52), 3U, "third visit number");
    }

    // The whole decoder looped, including a range that starts and ends at the bounds.
    {
        const ExecutionPlan plan(4, LoopSettings{0, 4, 2},
                                 std::vector<LayerKind>(4, LayerKind::LinearAttention));
        check_eq(plan.slot_count(), 8U, "fully looped slot count");
        check_eq(plan.linear_attention_slots(), 8U, "fully looped linear planes");
        check_eq(plan.physical_layer(4), 0U, "second pass restarts at layer 0");
    }

    // A range of one layer.
    {
        const ExecutionPlan plan(3, LoopSettings{1, 2, 2},
                                 std::vector<LayerKind>(3, LayerKind::FullAttention));
        const std::array<std::uint32_t, 4> want{0, 1, 1, 2};
        check_eq(plan.slot_count(), static_cast<std::uint32_t>(want.size()),
                 "single layer range slot count");
        for (std::uint32_t slot = 0; slot < plan.slot_count(); ++slot) {
            check_eq(plan.physical_layer(slot), want[slot], "single layer range order");
        }
    }

    // Mixed kinds outside the looped range still land on the right family.
    {
        std::vector<LayerKind> kinds{LayerKind::FullAttention, LayerKind::LinearAttention,
                                     LayerKind::FullAttention, LayerKind::LinearAttention};
        const ExecutionPlan plan(4, LoopSettings{1, 3, 2}, kinds);
        // Order: 0 (full), 1 (linear), 2 (full), 1 (linear), 2 (full), 3 (linear).
        check_eq(plan.slot_count(), 6U, "mixed kind slot count");
        check_eq(plan.full_attention_slots(), 3U, "mixed kind full planes");
        check_eq(plan.linear_attention_slots(), 3U, "mixed kind linear planes");
        check_eq(plan.family_index(0), 0U, "first full plane");
        check_eq(plan.family_index(2), 1U, "second full plane");
        check_eq(plan.family_index(4), 2U, "third full plane");
        check_eq(plan.family_index(1), 0U, "first linear plane");
        check_eq(plan.family_index(3), 1U, "second linear plane");
        check_eq(plan.family_index(5), 2U, "third linear plane");
    }
}

void test_refusals() {
    std::printf("refusals\n");
    const auto kinds = release_layer_types();

    const std::array<std::pair<LoopSettings, const char*>, 4> bad{
        std::pair<LoopSettings, const char*>{LoopSettings{4, 4, 2}, "empty range"},
        std::pair<LoopSettings, const char*>{LoopSettings{28, 4, 2}, "reversed range"},
        std::pair<LoopSettings, const char*>{LoopSettings{4, 33, 2}, "range past the decoder"},
        std::pair<LoopSettings, const char*>{LoopSettings{4, 28, 0}, "zero visits"},
    };
    for (const auto& [loop, what] : bad) {
        bool threw = false;
        try {
            (void)ExecutionPlan(32, loop, kinds);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, std::string("refuses ") + what);
    }

    // A layer kind list that does not cover the decoder exactly.
    for (const std::size_t size : {std::size_t{31}, std::size_t{33}}) {
        bool threw = false;
        try {
            (void)ExecutionPlan(32, release_loop(), std::vector<LayerKind>(size));
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "refuses a layer kind list of size " + std::to_string(size));
    }

    // A decoder with no layers.
    bool threw = false;
    try {
        (void)ExecutionPlan(0, LoopSettings{0, 1, 2}, std::vector<LayerKind>{});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "refuses a decoder with no layers");
}

} // namespace

int main() {
    test_release_geometry();
    test_visit_numbers();
    test_slot_isolation();
    test_family_indices();
    test_other_settings();
    test_refusals();

    if (failures != 0) {
        std::printf("FAILED: %d loop plan check(s)\n", failures);
        return 1;
    }
    std::printf("PASS: loop execution plan\n");
    return 0;
}
