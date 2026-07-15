#!/bin/bash

set -eof pipefail

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <input_cleaned.csv> <output_gbbs.adj>"
    exit 1
fi

INPUT_CSV="$1"
OUTPUT_ADJ="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GBBS_DIR="$(realpath "$SCRIPT_DIR/../gbbs-core-decomposition")"

ABS_OUTPUT=$(realpath "$OUTPUT_ADJ" 2>/dev/null || readlink -f "$OUTPUT_ADJ")
ABS_INPUT=$(realpath "$INPUT_CSV" 2>/dev/null || readlink -f "$INPUT_CSV")
TEMP_SNAP=$(realpath "$(mktemp "$(pwd)/gbbs_stream_XXXXXX.txt")")
trap 'echo "Cleaning up temporary stream file..."; rm -f "$TEMP_SNAP"' EXIT

tail -n +2 "$ABS_INPUT" | tr ',' '\t' > "$TEMP_SNAP"

(
  cd $GBBS_DIR
  ./bazel-bin/utils/snap_converter -s -i "$TEMP_SNAP" -o "$ABS_OUTPUT"
)

echo "Done! Successfully generated $OUTPUT_ADJ"
