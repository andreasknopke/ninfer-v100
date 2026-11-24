// The SenseNova-Loop-9B release configuration reader.
//
// Host-only: no CUDA. The document under test is written out here as the release writes it, so the
// reader is checked against the field set it will actually meet, and against the refusals that keep
// a feature this tree does not implement from being dropped quietly. When NINFER_TEST_SENSENOVA_CONFIG
// names a real `config.json`, the same reader is run on it and the two must agree.

#include "models/sensenova_loop/config.h"

#include "artifact/schema.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::artifact::ArtifactError;
using ninfer::artifact::Json;
namespace artifact = ninfer::artifact;
using ninfer::models::sensenova_loop::ExecutionPlan;
using ninfer::models::sensenova_loop::LayerKind;
using ninfer::models::sensenova_loop::ModelConfig;
using ninfer::models::sensenova_loop::parse_config;

int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        ++failures;
        std::printf("  FAIL: %s\n", what.c_str());
    }
}

template <typename T>
void check_eq(const T& actual, const T& expected, const std::string& what) {
    if (!(actual == expected)) {
        ++failures;
        std::printf("  FAIL: %s -- values differ\n", what.c_str());
    }
}

void check_eq(std::uint64_t actual, std::uint64_t expected, const std::string& what) {
    if (actual != expected) {
        ++failures;
        std::printf("  FAIL: %s -- got %llu, want %llu\n", what.c_str(),
                    static_cast<unsigned long long>(actual),
                    static_cast<unsigned long long>(expected));
    }
}

void check_eq(float actual, float expected, const std::string& what) {
    if (actual != expected) {
        ++failures;
        std::printf("  FAIL: %s -- got %.9g, want %.9g\n", what.c_str(),
                    static_cast<double>(actual), static_cast<double>(expected));
    }
}

// The release `config.json` as it is published, with the fields the reader has to see. Values are
// copied from the AWQ repository's file; the optional fields that a Hugging Face export carries
// alongside are included too, because the reader is strict about unknown members and has to
// tolerate them.
Json release_document(const char* architecture = "Qwen3_5LoopForConditionalGeneration") {
    Json rope_parameters = {
        {"rope_type", "yarn"},
        {"factor", 2.0},
        {"rope_theta", 10000000},
        {"partial_rotary_factor", 0.25},
        {"mrope_section", {11, 11, 10}},
        {"mrope_interleaved", true},
        {"original_max_position_embeddings", 262144},
    };

    Json layer_types = Json::array();
    for (std::uint32_t layer = 0; layer < 32; ++layer) {
        layer_types.push_back(layer % 4 == 3 ? "full_attention" : "linear_attention");
    }

    Json text_config = {
        {"model_type", "qwen3_5_text"},
        {"hidden_size", 4096},
        {"vocab_size", 248320},
        {"num_hidden_layers", 32},
        {"max_position_embeddings", 262144},
        {"intermediate_size", 12288},
        {"rms_norm_eps", 1e-6},
        {"layer_types", layer_types},
        {"head_dim", 256},
        {"num_attention_heads", 16},
        {"num_key_value_heads", 4},
        {"partial_rotary_factor", 0.25},
        {"rope_parameters", rope_parameters},
        {"linear_num_key_heads", 16},
        {"linear_key_head_dim", 128},
        {"linear_num_value_heads", 32},
        {"linear_value_head_dim", 128},
        {"linear_conv_kernel_dim", 4},
        {"attn_output_gate", true},
        {"tie_word_embeddings", false},
        {"attention_bias", false},
        {"attention_dropout", 0.0},
        {"hidden_act", "silu"},
        {"dtype", "bfloat16"},
        {"full_attention_interval", 4},
        {"mlp_only_layers", Json::array()},
        {"mtp_num_hidden_layers", 0},
        {"mtp_use_dedicated_embeddings", false},
        {"use_cache", true},
    };

    Json quantization_config = {
        {"quant_method", "awq"},
        {"bits", 4},
        {"group_size", 128},
        {"zero_point", true},
        {"version", "gemm"},
        {"lm_head", true},
        {"modules_to_not_convert",
         {"model.visual", "model.language_model.in_proj_a", "model.language_model.in_proj_b",
          "model.language_model.layers.4.mhc_attn", "model.language_model.layers.4.mhc_ffn"}},
    };

    return Json{
        {"architectures", {architecture}},
        {"model_type", "sensenova_loop"},
        {"text_config", text_config},
        {"loop_settings",
         {{"start", 4}, {"end", 28}, {"counts", {2}}, {"eval_count", 2}, {"use_gate", false}}},
        {"mhc",
         {{"start", 4},
          {"end", 28},
          {"streams", 2},
          {"iterations", 20},
          {"fixed_identity", false},
          {"pre_scale", 1.0}}},
        {"quantization_config", quantization_config},
        {"vision_config", {{"model_type", "qwen3_5_vision"}}},
        {"patch_conv3d", {{"patch_size", 2}}},
        {"attn_implementation", "sdpa"},
        {"dtype", "bfloat16"},
        {"torch_dtype", "bfloat16"},
        {"transformers_version", "4.57.1"},
        {"use_cache", true},
        {"tie_word_embeddings", false},
        {"hidden_size", 4096},
        {"bos_token_id", 1},
        {"eos_token_id", 0},
        {"pad_token_id", 2},
        {"image_token_id", 3},
        {"video_token_id", 4},
        {"vision_start_token_id", 5},
        {"vision_end_token_id", 6},
    };
}

void check_release_text(const ModelConfig& config) {
    check_eq(config.text.architecture, std::string("Qwen3_5LoopForConditionalGeneration"),
             "architecture");
    check_eq(config.text.model_type, std::string("qwen3_5_text"), "text model_type");
    check_eq(config.text.hidden_size, 4096U, "hidden_size");
    check_eq(config.text.vocab_size, 248320U, "vocab_size");
    check_eq(config.text.num_hidden_layers, 32U, "num_hidden_layers");
    check_eq(config.text.max_position_embeddings, 262144U, "max_position_embeddings");
    check_eq(config.text.intermediate_size, 12288U, "intermediate_size");
    check_eq(config.text.rms_norm_eps, 1e-6F, "rms_norm_eps");
    check(config.text.attention.output_gate, "attention output gate");

    check_eq(config.text.attention.num_attention_heads, 16U, "query heads");
    check_eq(config.text.attention.num_key_value_heads, 4U, "KV heads");
    check_eq(config.text.attention.head_dim, 256U, "head_dim");

    check_eq(config.text.gdn.num_key_heads, 16U, "GDN key heads");
    check_eq(config.text.gdn.key_head_dim, 128U, "GDN key head dim");
    check_eq(config.text.gdn.num_value_heads, 32U, "GDN value heads");
    check_eq(config.text.gdn.value_head_dim, 128U, "GDN value head dim");
    check_eq(config.text.gdn.conv_kernel_dim, 4U, "GDN conv kernel");

    // 0.25 of a 256-wide head is a 64-wide rotary, and the three sections cover its 32 pairs.
    check_eq(config.text.rope.rope_theta, 10000000.0F, "rope_theta");
    check_eq(config.text.rope.partial_rotary_factor, 0.25F, "partial_rotary_factor");
    check_eq(config.text.rope.rotary_dim, 64U, "rotary_dim");
    check_eq(config.text.rope.mrope_section[0], 11U, "MRoPE section 0");
    check_eq(config.text.rope.mrope_section[1], 11U, "MRoPE section 1");
    check_eq(config.text.rope.mrope_section[2], 10U, "MRoPE section 2");
    check_eq(config.text.rope.pair_axes.size(), std::size_t{32}, "pair axis count");

    // The release omits the ramp bounds and the attention scale, so the defaults stand in. Zero in
    // the struct means "derive it", which is what the YaRN op expects to be handed.
    check_eq(config.text.yarn.factor, 2.0F, "YaRN factor");
    check_eq(config.text.yarn.original_max_position_embeddings, 262144U, "YaRN original width");
    check_eq(config.text.yarn.beta_fast, 32.0F, "YaRN beta_fast");
    check_eq(config.text.yarn.beta_slow, 1.0F, "YaRN beta_slow");
    check_eq(config.text.yarn.attention_factor, 0.0F, "YaRN attention factor is derived");
}

void check_release_loop_and_mhc(const ModelConfig& config) {
    check_eq(config.loop.start, 4U, "loop start");
    check_eq(config.loop.end, 28U, "loop end");
    check_eq(config.loop.visits, 2U, "loop visits");
    check_eq(config.loop.eval_count, 2U, "loop eval count");
    check(!config.loop.use_gate, "loop gate is off");

    check_eq(config.mhc.start, 4U, "mhc start");
    check_eq(config.mhc.end, 28U, "mhc end");
    check_eq(config.mhc.streams, 2U, "mhc streams");
    check_eq(config.mhc.iterations, 20U, "mhc iterations");
    check(!config.mhc.fixed_identity, "mHC is not fixed identity");
    check_eq(config.mhc.pre_scale, 1.0F, "mHC pre-scale");
}

void test_release_document() {
    std::printf("release document\n");
    const ModelConfig config = parse_config(release_document());
    check_release_text(config);
    check_release_loop_and_mhc(config);

    // The layer kinds must come out as the release states them, because the plan is built from them.
    check_eq(config.text.layer_kinds.size(), std::size_t{32}, "layer kind count");
    for (std::uint32_t layer = 0; layer < 32; ++layer) {
        const bool full = layer % 4 == 3;
        check(config.text.layer_kinds[layer] ==
                  (full ? LayerKind::FullAttention : LayerKind::LinearAttention),
              "layer kind at " + std::to_string(layer));
    }

    // And the plan the rest of the runtime sizes itself from has the release's counts.
    const ExecutionPlan plan = build_plan(config);
    check_eq(plan.slot_count(), 56U, "plan slot count");
    check_eq(plan.full_attention_slots(), 14U, "plan full planes");
    check_eq(plan.linear_attention_slots(), 42U, "plan linear planes");
}

void test_both_architecture_names() {
    std::printf("both released architecture names\n");
    // The BF16 and AWQ repositories disagree about the name and about two pieces of Hugging Face
    // metadata, and agree about everything that matters. Both have to load, and only the name may
    // differ in what comes back.
    Json bf16_document = release_document("SenseNovaLoopForConditionalGeneration");
    bf16_document["auto_map"]     = {{"AutoModel", "modeling_sensenova_loop.SenseNovaLoopForConditionalGeneration"}};
    bf16_document["model_name"]   = "SenseNova-Loop-9B";
    bf16_document.erase("quantization_config");
    bf16_document.erase("torch_dtype");
    bf16_document.erase("transformers_version");

    const ModelConfig awq  = parse_config(release_document());
    const ModelConfig bf16 = parse_config(bf16_document);
    check_eq(bf16.text.architecture, std::string("SenseNovaLoopForConditionalGeneration"),
             "BF16 architecture");
    check_eq(bf16.text.hidden_size, awq.text.hidden_size, "BF16 hidden_size");
    check_eq(bf16.text.yarn.factor, awq.text.yarn.factor, "BF16 YaRN factor");
    check_eq(build_plan(bf16).slot_count(), build_plan(awq).slot_count(), "BF16 slot count");
}

void test_refusals() {
    std::printf("refusals\n");

    const auto refuses = [&](const char* what, const std::function<void(Json&)>& mutate) {
        Json document = release_document();
        mutate(document);
        bool threw = false;
        try {
            (void)parse_config(document);
        } catch (const ArtifactError&) {
            threw = true;
        }
        check(threw, std::string("refuses ") + what);
    };

    refuses("an unknown architecture", [](Json& d) {
        d["architectures"] = {"LlamaForCausalLM"};
    });
    refuses("a plain Qwen3.5 document", [](Json& d) { d["model_type"] = "qwen3_5_text"; });
    refuses("a second looped range", [](Json& d) { d["loop_settings"]["counts"] = {2, 2}; });
    refuses("an empty looped range", [](Json& d) {
        d["loop_settings"]["start"] = 28;
        d["loop_settings"]["end"]   = 28;
    });
    refuses("a loop past the last layer", [](Json& d) { d["loop_settings"]["end"] = 33; });
    refuses("an eval_count above the visit count",
            [](Json& d) { d["loop_settings"]["eval_count"] = 3; });
    refuses("mHC over a different block than the loop", [](Json& d) {
        d["mhc"]["start"] = 8;
        d["mhc"]["end"]   = 28;
    });
    refuses("mHC with a stream count the op cannot carry", [](Json& d) {
        d["mhc"]["streams"] = 4;
    });
    refuses("a RoPE type other than YaRN", [](Json& d) {
        d["text_config"]["rope_parameters"]["rope_type"] = "default";
    });
    refuses("a YaRN factor below one", [](Json& d) {
        d["text_config"]["rope_parameters"]["factor"] = 0.5;
    });
    refuses("a truncated YaRN ramp", [](Json& d) {
        d["text_config"]["rope_parameters"]["truncate"] = true;
    });
    refuses("beta_fast not above beta_slow", [](Json& d) {
        d["text_config"]["rope_parameters"]["beta_fast"] = 1.0;
    });
    refuses("multi-token prediction", [](Json& d) {
        d["text_config"]["mtp_num_hidden_layers"] = 1;
    });
    refuses("an unknown mixer", [](Json& d) {
        d["text_config"]["layer_types"][0] = "sliding_attention";
    });
    refuses("layer types that disagree with the layer count", [](Json& d) {
        auto& types = d["text_config"]["layer_types"];
        types.erase(types.begin() + static_cast<std::ptrdiff_t>(types.size()) - 1);
    });
    refuses("MRoPE sections that miss the rotary width", [](Json& d) {
        d["text_config"]["rope_parameters"]["mrope_section"] = {11, 11, 11};
    });
    refuses("a non-interleaved MRoPE layout", [](Json& d) {
        d["text_config"]["rope_parameters"]["mrope_interleaved"] = false;
    });
    refuses("partial_rotary_factor that disagrees with itself", [](Json& d) {
        d["text_config"]["rope_parameters"]["partial_rotary_factor"] = 0.5;
    });
    refuses("a member it does not know", [](Json& d) { d["text_config"]["surprise"] = 1; });
    refuses("a top-level member it does not know", [](Json& d) { d["surprise"] = 1; });
    refuses("a missing mhc block", [](Json& d) { d.erase("mhc"); });
}

// Run the same reader over a real release file when one is available, and require it to agree with
// the document written out above. This is the check that the mirror above is still the release.
//
// The two published checkpoints differ in one way that matters here: the BF16 repository ships a
// multi-token prediction block, the AWQ repository does not. Since this tree has no MTP, the honest
// outcome for a BF16 file is the refusal, so the test asks for that instead of for agreement.
void test_against_real_file() {
    const char* path = std::getenv("NINFER_TEST_SENSENOVA_CONFIG");
    if (path == nullptr) {
        std::printf("  (no NINFER_TEST_SENSENOVA_CONFIG, real file check skipped)\n");
        return;
    }
    std::ifstream file(path);
    if (!file) {
        ++failures;
        std::printf("  FAIL: cannot open %s\n", path);
        return;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    const Json document = artifact::parse_json(buffer.str(), "release config");

    const auto& text_config = document.at("text_config");
    const bool has_mtp = text_config.contains("mtp_num_hidden_layers") &&
                         text_config.at("mtp_num_hidden_layers").get<std::uint64_t>() != 0;
    if (has_mtp) {
        bool refused = false;
        try {
            (void)parse_config(document);
        } catch (const ArtifactError& error) {
            refused = std::string(error.what()).find("multi-token prediction") != std::string::npos;
        }
        check(refused, "real file with an MTP block is refused for that reason");
        std::printf("  real file has an MTP block and is refused for it\n");
        return;
    }

    const ModelConfig real   = parse_config(document);
    const ModelConfig mirror = parse_config(release_document());

    check_eq(real.text.architecture, mirror.text.architecture, "real architecture");
    check_eq(real.text.hidden_size, mirror.text.hidden_size, "real hidden_size");
    check_eq(real.text.vocab_size, mirror.text.vocab_size, "real vocab_size");
    check_eq(real.text.num_hidden_layers, mirror.text.num_hidden_layers, "real layer count");
    check_eq(real.text.attention.head_dim, mirror.text.attention.head_dim, "real head_dim");
    check_eq(real.text.attention.output_gate, mirror.text.attention.output_gate,
             "real output gate");
    check_eq(real.text.rope.rotary_dim, mirror.text.rope.rotary_dim, "real rotary_dim");
    check_eq(real.text.yarn.factor, mirror.text.yarn.factor, "real YaRN factor");
    check_eq(real.text.yarn.original_max_position_embeddings,
             mirror.text.yarn.original_max_position_embeddings, "real YaRN original width");
    check_eq(real.text.yarn.beta_fast, mirror.text.yarn.beta_fast, "real beta_fast");
    check_eq(real.text.yarn.beta_slow, mirror.text.yarn.beta_slow, "real beta_slow");
    check_eq(real.text.yarn.attention_factor, mirror.text.yarn.attention_factor,
             "real attention factor");
    check_eq(real.loop.start, mirror.loop.start, "real loop start");
    check_eq(real.loop.end, mirror.loop.end, "real loop end");
    check_eq(real.loop.visits, mirror.loop.visits, "real loop visits");
    check_eq(real.mhc.iterations, mirror.mhc.iterations, "real mHC iterations");
    check_eq(build_plan(real).slot_count(), build_plan(mirror).slot_count(), "real slot count");
    std::printf("  real file agrees\n");
}

} // namespace

int main() {
    test_release_document();
    test_both_architecture_names();
    test_refusals();
    test_against_real_file();

    if (failures != 0) {
        std::printf("FAILED: %d config check(s)\n", failures);
        return 1;
    }
    std::printf("PASS: sensenova_loop config\n");
    return 0;
}
