#!/usr/bin/env bash
# ============================================================================
#  NInfer server launcher for SenseNova-Loop-9B (runs INSIDE WSL Ubuntu)
#  Usage: serve_ninfer_sensenova.sh [CTXSIZE] [PORT] [KV] [CONCURRENCY]
#    CTXSIZE     logical context ceiling   (default 131072)
#    PORT        HTTP port                 (default 8041)
#    KV          int8 | fp16 | bf16        (default int8)
#    CONCURRENCY parallel requests         (default 1)
#
#  Why the context default is 131072 and not 262144:
#    The loop states the model as 56 blocks over 32 physical layers, and 14 of
#    those blocks are full-attention. Each needs its own KV plane, so the cache
#    is 75% larger than the same model without the loop. On a 32 GB card that
#    is the binding constraint, not the 17 GiB of weights.
#
#  No --spec: the release ships no MTP head (mtp_num_hidden_layers is 0), so
#  there is nothing to draft with. No --vision: the converter writes the text
#  tower only, the vision weights in the checkpoint are not in the artifact.
#
#  Env overrides: NINFER_DIR, NINFER_MODEL, EXTRA_ARGS, DEFAULT_MAX_TOKENS,
#                 MAX_PENDING_REQUESTS, PENDING_TIMEOUT_MS
# ============================================================================
set -uo pipefail

CTX="${1:-131072}"
PORT="${2:-8041}"
KV="${3:-int8}"
CONCURRENCY="${4:-1}"
DEFAULT_MAX_TOKENS="${DEFAULT_MAX_TOKENS:-32768}"
MAX_PENDING_REQUESTS="${MAX_PENDING_REQUESTS:-16}"
PENDING_TIMEOUT_MS="${PENDING_TIMEOUT_MS:-600000}"

NINFER_DIR="${NINFER_DIR:-/home/andre/ninfer-sensenova}"
MODEL="${NINFER_MODEL:-$NINFER_DIR/../models/sensenova-loop-9b-q45.ninfer}"
MODEL="$(realpath -m "$MODEL")"

BIN="$NINFER_DIR/build-v100/apps/ninfer-serve"

if [ ! -x "$BIN" ]; then
    echo "FATAL: $BIN not found or not executable."
    echo "Build it:  cmake --build $NINFER_DIR/build-v100 --target ninfer-serve -j"
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "FATAL: artifact not found: $MODEL"
    echo "Convert it (~5 min, CPU only):  bash /mnt/d/GitHub/Beellama/_port/conv/convert_bf16.sh"
    exit 1
fi

ARGS=(
    --host 0.0.0.0
    --port "$PORT"
    --max-context "$CTX"
    --kv-dtype "$KV"
    --max-concurrency "$CONCURRENCY"
    --device-state-slots "$CONCURRENCY"
    --max-pending-requests "$MAX_PENDING_REQUESTS"
    --pending-timeout-ms "$PENDING_TIMEOUT_MS"
    --default-max-tokens "$DEFAULT_MAX_TOKENS"
    # The loop's 14 KV planes are the tight allocation, so ask for exactly the
    # context asked for instead of leaving the 1 GiB the "auto" sizing reserves.
    --kv-capacity "$CTX"
    # The Q4/Q5 cuBLAS prefill route dequantises each weight once per chunk and
    # runs the GEMM over the tokens in it, so a wide chunk amortises the dequant.
    # 4096 is where the MLP + projection promotion pays off (prefill ~2.4k tok/s
    # vs ~200 on the emulated-MMA path); matches the Inkling launcher.
    --prefill-chunk 4096
)

echo "=============================================="
echo " NInfer - SenseNova-Loop-9B on V100"
echo "=============================================="
echo " artifact : $MODEL"
echo " context  : $CTX"
echo " kv       : $KV"
echo " parallel : $CONCURRENCY request(s) at once; queue up to $MAX_PENDING_REQUESTS"
echo " out-limit: $DEFAULT_MAX_TOKENS tokens when the client omits max_tokens"
echo " spec     : none (the release has no MTP head)"
echo " vision   : off (the artifact carries the text tower only)"
echo " API      : http://127.0.0.1:$PORT/v1"
echo "=============================================="
echo

cd "$NINFER_DIR" || exit 1
exec "$BIN" "$MODEL" "${ARGS[@]}" ${EXTRA_ARGS:-}
