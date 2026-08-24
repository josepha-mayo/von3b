#!/usr/bin/env bash
# Download the ADTC laptop GGUF (Q8_0 only). Public URL, no credentials.
# Output path must match metadata.json _runtime.model_path.
#
# Intended llama.cpp flags after download:
#   llama-cli -m model/von3b-Q8_0.gguf -c 65536 -ctk q4_0 -ctv q4_0 -ngl 0

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$HERE/model"
MODEL_FILE="$MODEL_DIR/von3b-Q8_0.gguf"
EXPECTED_BYTES="${VON_GGUF_BYTES:-3285475488}"
MODEL_URL="${VON_GGUF_URL:-https://huggingface.co/josephmayo/von3b/resolve/main/von3b-Q8_0.gguf?download=true}"

mkdir -p "$MODEL_DIR"

if [[ -f "$MODEL_FILE" ]]; then
  have="$(wc -c < "$MODEL_FILE" | tr -d ' ')"
  if [[ "$have" == "$EXPECTED_BYTES" ]]; then
    echo "model already present at $MODEL_FILE ($have bytes) — skipping download"
    echo "run: llama-cli -m $MODEL_FILE -c 65536 -ctk q4_0 -ctv q4_0 -ngl 0"
    exit 0
  fi
  echo "incomplete model at $MODEL_FILE ($have bytes, want $EXPECTED_BYTES) — redownloading"
  rm -f "$MODEL_FILE"
fi

echo "downloading $MODEL_URL -> $MODEL_FILE"
rm -f "$MODEL_FILE.partial"

if command -v curl > /dev/null 2>&1; then
  curl -L --fail --retry 5 --retry-delay 2 \
    -A "adtc-download/1.0" \
    --progress-bar \
    -o "$MODEL_FILE.partial" \
    "$MODEL_URL"
elif command -v wget > /dev/null 2>&1; then
  wget --tries=5 --user-agent="adtc-download/1.0" --show-progress \
    -O "$MODEL_FILE.partial" \
    "$MODEL_URL"
else
  echo "error: neither curl nor wget found" >&2
  exit 1
fi

have="$(wc -c < "$MODEL_FILE.partial" | tr -d ' ')"
if [[ "$have" != "$EXPECTED_BYTES" ]]; then
  echo "error: downloaded $have bytes, expected $EXPECTED_BYTES" >&2
  rm -f "$MODEL_FILE.partial"
  exit 1
fi

mv "$MODEL_FILE.partial" "$MODEL_FILE"
echo "done: $MODEL_FILE ($have bytes)"
echo "run: llama-cli -m $MODEL_FILE -c 65536 -ctk q4_0 -ctv q4_0 -ngl 0"
