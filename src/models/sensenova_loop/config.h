#pragma once

// The SenseNova-Loop-9B release configuration, read straight from its `config.json`.
//
// The release describes itself with `model_type: sensenova_loop` and a text block whose own
// `model_type` is `qwen3_5_text`, so the text block alone does not identify the model -- the
// top-level architecture does. It also carries blocks no other model in this tree has (`loop_settings`
// and `mhc`) and RoPE parameters the Qwen3.5 reader does not know. Both readers are strict about
// unknown members on purpose, so the two cannot share an entry point: this one owns the release's
// field set and its constraints.

#include "models/sensenova_loop/loop_plan.h"

#include "artifact/schema.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::models::sensenova_loop {

// YaRN scaling of the rotary frequencies. The release omits `beta_fast`, `beta_slow` and
// `attention_factor`, which is not the same as setting them to zero: the ramp bounds fall back to
// 32 and 1, and the attention scale is derived from the factor instead of being given.
struct YarnConfig {
    float factor                              = 1.0F;
    std::uint32_t original_max_position_embeddings = 0;
    float beta_fast                           = 32.0F;
    float beta_slow                           = 1.0F;
    // Zero means "derive it from the factor", which is what the release asks for by omitting it.
    float attention_factor = 0.0F;
};

// The rotary layout shared with the Qwen3.5 text blocks: a partial rotary width and three
// interleaved MRoPE axis sections.
struct RopeConfig {
    float rope_theta            = 0.0F;
    float partial_rotary_factor = 0.0F;
    std::uint32_t rotary_dim    = 0;
    std::array<std::uint32_t, 3> mrope_section{};
    // Axis of each rotary pair, in execution order. Size is rotary_dim / 2.
    std::vector<std::uint8_t> pair_axes;
};

// The gated delta net a linear attention layer runs.
struct GdnConfig {
    std::uint32_t num_key_heads   = 0;
    std::uint32_t key_head_dim    = 0;
    std::uint32_t num_value_heads = 0;
    std::uint32_t value_head_dim  = 0;
    std::uint32_t conv_kernel_dim = 0;
};

struct AttentionConfig {
    std::uint32_t num_attention_heads = 0;
    std::uint32_t num_key_value_heads = 0;
    std::uint32_t head_dim            = 0;
    // The release attention writes an extra gate channel alongside the output. The program has to
    // allocate and consume it, so it is part of the shape rather than an ignored flag.
    bool output_gate = false;
};

struct LoopConfig {
    std::uint32_t start      = 0;
    std::uint32_t end        = 0;
    std::uint32_t visits     = 0;
    std::uint32_t eval_count = 0;
    bool use_gate            = false;
};

struct MhcConfig {
    std::uint32_t start          = 0;
    std::uint32_t end            = 0;
    std::uint32_t streams        = 0;
    std::uint32_t iterations     = 0;
    bool fixed_identity          = false;
    float pre_scale              = 1.0F;
};

struct TextConfig {
    std::string architecture;
    std::string model_type;
    std::uint32_t hidden_size                = 0;
    std::uint32_t vocab_size                 = 0;
    std::uint32_t num_hidden_layers          = 0;
    std::uint32_t max_position_embeddings    = 0;
    std::uint32_t intermediate_size          = 0;
    float rms_norm_eps                       = 0.0F;
    std::vector<LayerKind> layer_kinds;
    AttentionConfig attention;
    GdnConfig gdn;
    RopeConfig rope;
    YarnConfig yarn;
};

struct ModelConfig {
    TextConfig text;
    LoopConfig loop;
    MhcConfig mhc;
};

// Read a release `config.json`. Throws artifact::ArtifactError when the document does not describe a
// model this tree can run, including when it describes something this tree does not implement (an
// MTP block, a second looped range, a RoPE type other than YaRN) rather than quietly dropping it.
[[nodiscard]] ModelConfig parse_config(const artifact::Json& root);

// Recognition and loading for an artifact directory. The text component's config is the release
// document itself, so the two overloads below see the same field set the JSON overload does.
//
// There is deliberately no `models::resolve_architecture` entry for this model. That function is the
// Qwen3.5 reader's dispatcher, and the release's text block reports `model_type: qwen3_5_text`, so a
// pair here would make the two families indistinguishable at exactly the point where they diverge.
// DeepSeek-V4 set the precedent: a model with its own reader is recognised by its own predicate.
[[nodiscard]] bool is_sensenova_loop_config(const artifact::Directory& directory);
[[nodiscard]] ModelConfig parse_config(const artifact::Directory& directory);

// The execution order implied by a text block and the loop settings. Kept out of ModelConfig so the
// config stays a record of what the file said.
[[nodiscard]] ExecutionPlan build_plan(const ModelConfig& config);

} // namespace ninfer::models::sensenova_loop
