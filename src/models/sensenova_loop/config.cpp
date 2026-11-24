// The SenseNova-Loop-9B release configuration; the contract is models/sensenova_loop/config.h.
// Host code only: it runs while loading, and every constraint here exists because the runtime below
// assumes it rather than because the file happens to contain it.

#include "models/sensenova_loop/config.h"

#include <cmath>
#include <limits>
#include <string>
#include <string_view>

namespace ninfer::models::sensenova_loop {
namespace {

using artifact::ArtifactError;
using artifact::Json;
using artifact::require_members;

void fail(std::string_view message) { throw ArtifactError(std::string("sensenova_loop: ") + std::string(message)); }

std::uint32_t integer(const Json& value, std::string_view label, bool positive = true) {
    const auto n = artifact::require_u64(value, label, positive);
    if (n > std::numeric_limits<std::uint32_t>::max()) {
        throw ArtifactError(std::string(label) + ": config dimension exceeds u32");
    }
    return static_cast<std::uint32_t>(n);
}

std::uint32_t dimension(const Json& config, const char* name) {
    return integer(config.at(name), name);
}

std::uint32_t optional_dimension(const Json& config, const char* name, std::uint32_t fallback,
                                 bool positive = true) {
    if (!config.contains(name)) { return fallback; }
    return integer(config.at(name), name, positive);
}

float positive_float(const Json& config, const char* name) {
    const auto& value = config.at(name);
    if (!value.is_number()) { throw ArtifactError(std::string(name) + " must be a real value"); }
    const float result = value.get<float>();
    if (!std::isfinite(result) || result <= 0) {
        throw ArtifactError(std::string(name) + " must be positive finite FP32");
    }
    return result;
}

bool optional_bool(const Json& config, const char* name, bool fallback) {
    if (!config.contains(name)) { return fallback; }
    const auto& value = config.at(name);
    if (!value.is_boolean()) { throw ArtifactError(std::string(name) + " must be a boolean"); }
    return value.get<bool>();
}

std::string string_member(const Json& config, const char* name) {
    const auto& value = config.at(name);
    if (!value.is_string()) { throw ArtifactError(std::string(name) + " must be a string"); }
    return value.get<std::string>();
}

// The two architecture names the released checkpoints use. The BF16 repository says
// `SenseNovaLoopForConditionalGeneration`, the AWQ repository says `Qwen3_5LoopForConditionalGeneration`;
// both carry `model_type: sensenova_loop`, and the text block says `qwen3_5_text` either way, so the
// top-level name is the only thing that tells this model apart from a plain Qwen3.5.
bool known_architecture(const std::string& name) {
    return name == "SenseNovaLoopForConditionalGeneration" ||
           name == "Qwen3_5LoopForConditionalGeneration";
}

RopeConfig rope(const Json& value, std::uint32_t head_dim) {
    require_members(value,
                    {"rope_type", "factor", "rope_theta", "partial_rotary_factor", "mrope_section",
                     "original_max_position_embeddings"},
                    {"mrope_interleaved", "beta_fast", "beta_slow", "attention_factor", "truncate",
                     "mscale_all_dim"},
                    "text RoPE");
    if (string_member(value, "rope_type") != "yarn") { fail("only YaRN rotary scaling is supported"); }

    RopeConfig out;
    out.rope_theta            = positive_float(value, "rope_theta");
    out.partial_rotary_factor = positive_float(value, "partial_rotary_factor");
    if (out.partial_rotary_factor > 1) { fail("partial_rotary_factor exceeds one"); }
    out.rotary_dim = static_cast<std::uint32_t>(static_cast<double>(head_dim) *
                                                out.partial_rotary_factor);
    if (out.rotary_dim == 0 || out.rotary_dim % 2) {
        fail("rotary dimension must be positive and even");
    }

    // Only the interleaved axis assignment is implemented. The release uses it; a document that asks
    // for the other layout would silently get the wrong positions rather than a wrong shape.
    if (!optional_bool(value, "mrope_interleaved", true)) {
        fail("only interleaved MRoPE is supported");
    }
    const auto& sections = value.at("mrope_section");
    if (!sections.is_array() || sections.size() != 3) { fail("MRoPE requires three sections"); }
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < 3; ++i) {
        out.mrope_section[i] = integer(sections[i], "MRoPE section", false);
        sum += out.mrope_section[i];
    }
    if (sum != out.rotary_dim / 2) { fail("MRoPE sections differ from the rotary width"); }

    std::array<std::uint32_t, 3> counts{};
    for (std::uint32_t pair = 0; pair < out.rotary_dim / 2; ++pair) {
        const std::uint8_t axis = pair % 3 == 1 && pair < 3ULL * out.mrope_section[1]   ? 1
                                  : pair % 3 == 2 && pair < 3ULL * out.mrope_section[2] ? 2
                                                                                        : 0;
        out.pair_axes.push_back(axis);
        ++counts[axis];
    }
    if (counts != out.mrope_section) { fail("MRoPE sections exceed interleaved axis ranges"); }
    return out;
}

YarnConfig yarn(const Json& value) {
    YarnConfig out;
    out.factor = positive_float(value, "factor");
    if (out.factor < 1.0F) { fail("YaRN factor below one would raise the frequencies"); }
    out.original_max_position_embeddings = dimension(value, "original_max_position_embeddings");
    // Absent means the Hugging Face defaults, which the release relies on. A document that spells
    // them out is read as-is.
    out.beta_fast = value.contains("beta_fast") ? positive_float(value, "beta_fast") : 32.0F;
    out.beta_slow = value.contains("beta_slow") ? positive_float(value, "beta_slow") : 1.0F;
    if (out.beta_fast <= out.beta_slow) { fail("YaRN beta_fast must exceed beta_slow"); }
    out.attention_factor =
        value.contains("attention_factor") ? positive_float(value, "attention_factor") : 0.0F;
    if (value.contains("truncate") && value.at("truncate").is_boolean() &&
        value.at("truncate").get<bool>()) {
        fail("truncated YaRN ramp is not supported");
    }
    return out;
}

GdnConfig gdn(const Json& value) {
    GdnConfig out;
    out.num_key_heads   = dimension(value, "linear_num_key_heads");
    out.key_head_dim    = dimension(value, "linear_key_head_dim");
    out.num_value_heads = dimension(value, "linear_num_value_heads");
    out.value_head_dim  = dimension(value, "linear_value_head_dim");
    out.conv_kernel_dim = dimension(value, "linear_conv_kernel_dim");
    if (out.num_value_heads % out.num_key_heads) {
        fail("linear attention value heads must be a multiple of key heads");
    }
    return out;
}

std::vector<LayerKind> layer_kinds(const Json& config) {
    const auto& list = config.at("layer_types");
    if (!list.is_array() || list.empty()) { fail("layer_types must be a non-empty array"); }
    std::vector<LayerKind> out;
    out.reserve(list.size());
    for (const auto& entry : list) {
        if (!entry.is_string()) { fail("layer_types entries must be strings"); }
        const auto name = entry.get<std::string>();
        if (name == "full_attention") {
            out.push_back(LayerKind::FullAttention);
        } else if (name == "linear_attention") {
            out.push_back(LayerKind::LinearAttention);
        } else {
            fail("layer_types has an unknown mixer");
        }
    }
    return out;
}

TextConfig text(const Json& value) {
    require_members(
        value,
        {"model_type", "hidden_size", "vocab_size", "num_hidden_layers", "max_position_embeddings",
         "rms_norm_eps", "layer_types", "head_dim", "num_attention_heads", "num_key_value_heads",
         "intermediate_size", "partial_rotary_factor", "rope_parameters", "linear_num_key_heads",
         "linear_key_head_dim", "linear_num_value_heads", "linear_value_head_dim",
         "linear_conv_kernel_dim"},
        {"tie_word_embeddings", "attention_bias", "attention_dropout", "attn_output_gate", "dtype",
         "hidden_act", "initializer_range", "mamba_ssm_dtype", "mlp_only_layers",
         "mtp_num_hidden_layers", "mtp_use_dedicated_embeddings", "use_cache", "bos_token_id",
         "eos_token_id", "pad_token_id", "full_attention_interval"},
        "text config");

    TextConfig out;
    out.model_type   = artifact::require_id(value.at("model_type"), "text model_type");
    out.hidden_size  = dimension(value, "hidden_size");
    out.vocab_size   = dimension(value, "vocab_size");
    out.num_hidden_layers = dimension(value, "num_hidden_layers");
    out.max_position_embeddings = dimension(value, "max_position_embeddings");
    out.intermediate_size       = dimension(value, "intermediate_size");
    out.rms_norm_eps            = positive_float(value, "rms_norm_eps");

    // Multi-token prediction ships weights the loader would have to bind and the program would have
    // to run. Neither exists here, so a document that asks for it is refused rather than truncated.
    // Zero is the release's way of saying there is none, so zero has to be a legal value here.
    if (optional_dimension(value, "mtp_num_hidden_layers", 0, false) != 0) {
        fail("multi-token prediction is not implemented");
    }

    out.layer_kinds = layer_kinds(value);
    if (out.layer_kinds.size() != out.num_hidden_layers) {
        fail("layer_types differ from num_hidden_layers");
    }
    bool any_full = false;
    for (const auto kind : out.layer_kinds) {
        any_full = any_full || kind == LayerKind::FullAttention;
    }
    if (!any_full) { fail("a decoder without full attention cannot keep a KV cache"); }

    out.attention.num_attention_heads = dimension(value, "num_attention_heads");
    out.attention.num_key_value_heads = dimension(value, "num_key_value_heads");
    out.attention.head_dim            = dimension(value, "head_dim");
    if (out.attention.num_attention_heads % out.attention.num_key_value_heads) {
        fail("attention query heads must be divisible by KV heads");
    }
    if (out.attention.head_dim % 2) { fail("head_dim must be even"); }
    out.attention.output_gate = optional_bool(value, "attn_output_gate", false);

    out.gdn = gdn(value);

    // The release repeats partial_rotary_factor at the text level and inside rope_parameters. They
    // have to agree, otherwise the two readers in the wild disagree about the rotary width.
    const auto& parameters = value.at("rope_parameters");
    if (!parameters.is_object()) { fail("rope_parameters must be an object"); }
    if (parameters.contains("partial_rotary_factor")) {
        const auto nested = positive_float(parameters, "partial_rotary_factor");
        const auto outer  = positive_float(value, "partial_rotary_factor");
        if (nested != outer) { fail("partial_rotary_factor differs between its two homes"); }
    }
    out.rope = rope(parameters, out.attention.head_dim);
    out.yarn = yarn(parameters);
    return out;
}

LoopConfig loop(const Json& value, std::uint32_t num_layers) {
    require_members(value, {"start", "end", "counts"}, {"eval_count", "use_gate"}, "loop settings");
    LoopConfig out;
    out.start = integer(value.at("start"), "loop start", false);
    out.end   = integer(value.at("end"), "loop end", false);

    // `counts` is a list so that several separately-looped ranges could be described. The release
    // ships exactly one, and a second entry would need a plan this tree cannot build.
    const auto& counts = value.at("counts");
    if (!counts.is_array() || counts.size() != 1) {
        fail("exactly one looped range is supported");
    }
    out.visits = integer(counts[0], "loop count");

    out.eval_count = optional_dimension(value, "eval_count", out.visits);
    out.use_gate   = optional_bool(value, "use_gate", false);

    if (out.start >= out.end) { fail("the looped range must not be empty"); }
    if (out.end > num_layers) { fail("the looped range must stay inside the decoder"); }
    if (out.eval_count < 1 || out.eval_count > out.visits) {
        fail("loop eval_count must be between 1 and the visit count");
    }
    return out;
}

MhcConfig mhc(const Json& value, const LoopConfig& looped, std::uint32_t num_layers) {
    require_members(value, {"start", "end", "streams", "iterations"},
                    {"fixed_identity", "pre_scale"}, "mhc settings");
    MhcConfig out;
    out.start          = integer(value.at("start"), "mhc start", false);
    out.end            = integer(value.at("end"), "mhc end", false);
    out.streams        = integer(value.at("streams"), "mhc streams");
    out.iterations     = integer(value.at("iterations"), "mhc iterations");
    out.fixed_identity = optional_bool(value, "fixed_identity", false);
    out.pre_scale      = positive_float(value, "pre_scale");

    // The mHC op in this tree carries exactly two streams, which is what the release asks for.
    if (out.streams != 2) { fail("the mHC op carries exactly two streams"); }
    if (out.iterations == 0) { fail("mHC needs at least one Sinkhorn iteration"); }
    if (out.start >= out.end || out.end > num_layers) { fail("the mHC range is outside the decoder"); }
    // mHC and the loop are documented over the same block of layers. Running them over different
    // blocks would not change any shape, so nothing downstream would notice -- which is exactly why
    // it is refused here.
    if (out.start != looped.start || out.end != looped.end) {
        fail("the mHC range must match the looped range");
    }
    return out;
}

} // namespace

ModelConfig parse_config(const Json& root) {
    require_members(
        root, {"architectures", "model_type", "text_config", "loop_settings", "mhc"},
        {"vision_config", "patch_conv3d", "quantization_config", "attn_implementation", "dtype",
         "torch_dtype", "transformers_version", "use_cache", "tie_word_embeddings", "hidden_size",
         "bos_token_id", "eos_token_id", "pad_token_id", "image_token_id", "video_token_id",
         "vision_start_token_id", "vision_end_token_id", "auto_map", "model_name"},
        "sensenova_loop config");

    if (artifact::require_id(root.at("model_type"), "model_type") != "sensenova_loop") {
        fail("model_type is not sensenova_loop");
    }
    const auto& names = root.at("architectures");
    if (!names.is_array() || names.size() != 1) { fail("architectures must name one implementation"); }
    const auto architecture = artifact::require_id(names[0], "architecture");
    if (!known_architecture(architecture)) { fail("unknown architecture"); }

    if (!root.at("text_config").is_object()) { fail("text_config must be an object"); }
    ModelConfig out;
    out.text      = text(root.at("text_config"));
    out.text.architecture = architecture;
    out.loop      = loop(root.at("loop_settings"), out.text.num_hidden_layers);
    out.mhc       = mhc(root.at("mhc"), out.loop, out.text.num_hidden_layers);
    return out;
}

bool is_sensenova_loop_config(const artifact::Directory& directory) {
    if (!directory.components.contains("text")) { return false; }
    const auto& config = directory.component("text").config;
    return config.contains("model_type") && config.at("model_type") == "sensenova_loop";
}

ModelConfig parse_config(const artifact::Directory& directory) {
    return parse_config(directory.component("text").config);
}

ExecutionPlan build_plan(const ModelConfig& config) {
    return ExecutionPlan(config.text.num_hidden_layers,
                         LoopSettings{config.loop.start, config.loop.end, config.loop.visits},
                         config.text.layer_kinds);
}

} // namespace ninfer::models::sensenova_loop
