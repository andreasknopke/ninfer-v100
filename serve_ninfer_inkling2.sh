#!/usr/bin/env bash
# ============================================================================
#  NInfer server launcher for INKLING-2-MINI (runs INSIDE WSL Ubuntu)
#
#  Usage: serve_ninfer_inkling2.sh [CTXSIZE] [SPEC] [VISION] [PORT] [KV] [CONCURRENCY] [THINKING]
#    CTXSIZE     logical context ceiling   (default 262144 = HARD MAX of the model)
#                Anything above is clamped to 262144 - Inkling-2-Mini has no
#                larger position range and the engine refuses a bigger context.
#    SPEC        dspark | none             (default dspark)
#                "none" omits --spec entirely ("--spec none" is not a valid
#                literal, the binary prints usage and exits 1).
#    VISION      1 | 0                     (default 1)
#    PORT        HTTP port                 (default 8040)
#    KV          int8 | fp16 | fp32 | rk8v4 | rk4v4   (default int8)
#                ninfer-serve restricts --kv-dtype for this model to
#                "fp16|fp32, or lossy opt-in global rings int8|rk8v4|rk4v4".
#                int8 is what the upstream 262144-token run used and is the only
#                one of these that makes the full context fit alongside DSpark +
#                vision, so it is the default. It is lossy in principle, but safe
#                by design here: the model clips k per head to |k| <= 22.63 and
#                RMS-normalises v to 1, so the cached values are bounded.
#                Upstream's greedy md5 was identical across plain/DSpark legs at
#                int8. Use fp16 if you want bit-exact KV and accept a smaller
#                context (roughly 196k with DSpark + vision).
#    CONCURRENCY parallel requests         (default 1)
#                DSpark is built for one request at a time. --max-concurrency
#                >1 switches the core to batched decode, which upstream measures
#                WITHOUT DSpark (198 tok/s aggregate at cc 8, but an individual
#                request falls back to plain-decode speed). Keep 1 for
#                interactive agent work; raise it only for real concurrent load.
#    THINKING    1 | 0                     (default 1)
#                0 passes --no-thinking, which maps to the chat template's
#                effort "none" ("Thinking effort level: 0").
#
#  Env overrides: NINFER_DIR, NINFER_MODEL, EXTRA_ARGS, DEFAULT_MAX_TOKENS,
#                 MAX_PENDING_REQUESTS, PENDING_TIMEOUT_MS, PREFILL_CHUNK,
#                 DRAFT_TOKENS, DRAFT_MIN_P, DSPARK_ADAPTIVE, TCP_TIMEOUT_MS
#
#  ---------------------------------------------------------------------------
#  WHY A SEPARATE SOURCE TREE?
#  ---------------------------------------------------------------------------
#  NINFER_DIR defaults to /home/andre/ninfer-inkling, a dedicated checkout of
#  upstream master. The Qwen build in /home/andre/ninfer (rev 158f49d from
#  2026-09-13) contains NO Inkling-2 code at all, and updating it in place would
#  put the working Qwen profiles (port 8084) at risk. Artifacts are shared: the
#  .ninfer file lives in /home/andre/ninfer/models/ either way.
#
#  ---------------------------------------------------------------------------
#  WHAT THIS MODEL IS NOT (requests the route refuses outright)
#  ---------------------------------------------------------------------------
#  Clients that depend on any of these must not be pointed at this endpoint:
#    * logprobs
#    * output constraints (JSON schema / ABNF grammar) and
#      tool_choice "required" or a named tool   -> there is no grammar engine
#    * video input
#    * continuing a final assistant message
#    * LoRA / graft adapters other than the two opt-in uncensoring adapters
#  Tool calls are free generation: a malformed tool-call JSON comes back as
#  content, it is neither repaired nor rejected.
#  Thinking is reported in `reasoning_content`, the answer in `content`.
#
#  ---------------------------------------------------------------------------
#  PREFILL CHUNK (gemessen 2026-11-24 auf dieser V100, ctx 262144,
#  --spec dspark --vision --kv-dtype int8, Prompt 7,5k Tokens)
#  ---------------------------------------------------------------------------
#  upstream schreibt, Chunks 4096 und 8192 passten bei 262144 mit DSpark +
#  Vision nicht - das gilt nur fuer FP16 KV. Mit int8 KV (unser Default)
#  passen sie sehr wohl:
#
#    chunk   Prefill tok/s   VRAM frei    Prefix-Reuse ab
#    2048        2.463        1.938 MiB        2.048   <- Engine-Wahl
#    4096        2.686        1.438 MiB        4.096   <- Default hier
#    8192        2.951          438 MiB        8.192   <- Maximal, knapp
#
#  8192 ist der schnellste Wert (ueber upstreams 2.727), laesst aber nur
#  438 MiB frei - riskant, sobald der Desktop oder ein zweites Modell VRAM
#  will. PREFILL_CHUNK=8192 also bewusst opt-in.
#
#  ---------------------------------------------------------------------------
#  PREFIX REUSE (matters a lot for agent prompts)
#  ---------------------------------------------------------------------------
#  Reuse is CHUNK-grained: a snapshot is taken at the last whole multiple of
#  --prefill-chunk, so a second turn resumes only if it shares at least that
#  many tokens. Upstream measured at 262144 with DSpark + vision + int8 KV:
#      chunk 2048: 2nd turn at ~8k -> TTFT 1.03 s (6,144 reused) vs 3.93 s cold
#      chunk 4096: 2nd turn at ~8k -> TTFT 1.77 s (4,096 reused) vs 3.38 s cold
#      chunk 8192: 2nd turn at ~8k -> TTFT 3.04 s (NOTHING reused)
#  So a bigger chunk trades cold-prefill speed against warm-turn latency.
#  4096 is the balance: +9 % cold prefill, reuse still covers a typical
#  multi-turn agent prompt. Set PREFILL_CHUNK=2048 if your chats are short and
#  mostly repeat a small prefix, or 8192 for maximum cold throughput.
#
#  NOTE ON MEASURED "tok/s" FOR PREFILL: prefill throughput is dominated by
#  fixed cost on short prompts. Measured on this card, same server:
#      52 tok -> 426 tok/s     473 tok -> 1.578 tok/s
#     116 tok -> 651 tok/s   1.759 tok -> 2.490 tok/s
#     252 tok -> 1.135 tok/s  7.033 tok -> 2.463 tok/s  (plateau)
#  A client that reports ~200-400 tok/s is timing a SHORT prompt (or including
#  queue wait), not a slow engine. Compare only at >=2k prompt tokens.
# ============================================================================
set -uo pipefail

CTX="${1:-262144}"
SPEC="${2:-dspark}"
VISION="${3:-1}"
PORT="${4:-8040}"
KV="${5:-int8}"
CONCURRENCY="${6:-1}"
THINKING="${7:-1}"

# --- hard ceiling ------------------------------------------------------------
# Inkling-2-Mini ships no RoPE; positions enter only through a learned relative
# bias (extent 1024) plus log scaling. 262144 is the supported ceiling.
MODEL_MAX_CTX=262144
if [ "$CTX" -gt "$MODEL_MAX_CTX" ]; then
    echo "NOTE: context $CTX exceeds the model ceiling - clamping to $MODEL_MAX_CTX."
    CTX="$MODEL_MAX_CTX"
fi

NINFER_DIR="${NINFER_DIR:-/home/andre/ninfer-inkling}"
MODEL="${NINFER_MODEL:-/home/andre/ninfer/models/inkling2-mini-q8.ninfer}"

DEFAULT_MAX_TOKENS="${DEFAULT_MAX_TOKENS:-32768}"
MAX_PENDING_REQUESTS="${MAX_PENDING_REQUESTS:-16}"
PENDING_TIMEOUT_MS="${PENDING_TIMEOUT_MS:-600000}"
PREFILL_CHUNK="${PREFILL_CHUNK:-4096}"
DRAFT_TOKENS="${DRAFT_TOKENS:-3}"
DRAFT_MIN_P="${DRAFT_MIN_P:-0.5}"
DSPARK_ADAPTIVE="${DSPARK_ADAPTIVE:-on}"

# The local ninfer-tcp-user-timeout.patch raises the kernel's TCP_USER_TIMEOUT
# from 15 s to 10 min so a long non-streaming generation is not reset by the
# Windows TCP stack. The patch also honours this env var.
TCP_TIMEOUT_MS="${TCP_TIMEOUT_MS:-600000}"
export NINFER_TCP_USER_TIMEOUT_MS="$TCP_TIMEOUT_MS"

BIN="$NINFER_DIR/build-v100/apps/ninfer-serve"

if [ ! -x "$BIN" ]; then
    echo "FATAL: $BIN not found or not executable."
    echo "Build it first:  bash /mnt/d/GitHub/Beellama/ninfer/setup_ninfer_inkling2.sh"
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "FATAL: artifact not found: $MODEL"
    echo "Download it first:"
    echo "  bash /mnt/d/GitHub/Beellama/ninfer/download_artifact.sh \\"
    echo "    WarlaxZ/Inkling-2-Mini-NInfer-V100 inkling2-mini-q8.ninfer \\"
    echo "    2167c0c0ff2ebb5d9dcf7ea5c8289ce13239bdc5104aac053af79a3b8cf009c1"
    exit 1
fi

ARGS=(
    --host 0.0.0.0
    --port "$PORT"
    --max-context "$CTX"
    --kv-dtype "$KV"
    --max-concurrency "$CONCURRENCY"
    --max-pending-requests "$MAX_PENDING_REQUESTS"
    --pending-timeout-ms "$PENDING_TIMEOUT_MS"
    --default-max-tokens "$DEFAULT_MAX_TOKENS"
)

# Deliberately NOT passed (unsupported or harmful on this route):
#   --host-state-slots / --host-kv-mib  -> no context store / host cache tiers
#   --kv-capacity                        -> let the engine size KV to the context
#   --device-state-slots                 -> engine default matches concurrency
#   --lm-head-draft                      -> Qwen MTP only
#   --prefill-chunk                      -> engine picks 2048 (best reuse/fit mix)
[ -n "$PREFILL_CHUNK" ] && ARGS+=(--prefill-chunk "$PREFILL_CHUNK")

case "${SPEC,,}" in
    dspark)
        ARGS+=(--spec dspark --draft-tokens "$DRAFT_TOKENS"
               --draft-min-p "$DRAFT_MIN_P" --dspark-adaptive "$DSPARK_ADAPTIVE")
        SPEC_DESC="dspark (up to $DRAFT_TOKENS drafts, adaptive $DSPARK_ADAPTIVE, floor $DRAFT_MIN_P)"
        ;;
    none|"")
        SPEC_DESC="off (plain decode)"
        ;;
    *)
        echo "FATAL: unknown spec '$SPEC' (use dspark|none)"
        exit 1
        ;;
esac

if [ "$VISION" = "1" ]; then
    ARGS+=(--vision)
else
    SPEC_DESC="$SPEC_DESC, text-only"
fi

if [ "$THINKING" = "0" ]; then
    ARGS+=(--no-thinking)
    THINK_DESC="off (template effort none)"
else
    THINK_DESC="on (reasoning in reasoning_content)"
fi

echo "=============================================="
echo " NInfer - Inkling-2-Mini on V100"
echo "=============================================="
echo " tree     : $NINFER_DIR"
echo " artifact : $(basename "$MODEL") ($(du -h "$MODEL" | cut -f1))"
echo " context  : $CTX"
echo " spec     : $SPEC_DESC"
echo " vision   : $VISION"
echo " thinking : $THINK_DESC"
echo " kv       : $KV"
echo " out-limit: $DEFAULT_MAX_TOKENS tokens when the client omits max_tokens"
echo " parallel : $CONCURRENCY request(s) at once; queue up to $MAX_PENDING_REQUESTS"
echo " queue-t/o: $((PENDING_TIMEOUT_MS / 1000))s wait for admission before HTTP 503"
echo " tcp-t/o  : $((TCP_TIMEOUT_MS / 1000))s TCP_USER_TIMEOUT"
echo " API      : http://127.0.0.1:$PORT/v1"
echo "=============================================="
echo

cd "$NINFER_DIR" || exit 1
exec "$BIN" "$MODEL" "${ARGS[@]}" ${EXTRA_ARGS:-}
