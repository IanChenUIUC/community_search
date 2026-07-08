#!/bin/bash

set -eof pipefail

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <input_cleaned.csv> <output_gbbs.adj>"
    exit 1
fi

INPUT_CSV="$1"
OUTPUT_ADJ="$2"
ABS_OUTPUT=$(realPath "$OUTPUT_ADJ" 2>/dev/null || readlink -f "$OUTPUT_ADJ")

TEMP_SNAP=$(mktemp "$(pwd)/gbbs_stream_XXXXXX.txt")
trap 'echo "Cleaning up temporary stream file..."; rm -f "$TEMP_SNAP"' EXIT

tail -n +2 "$INPUT_CSV" | tr ',' '\t' > "$TEMP_SNAP"
bazel run \
  --disk_cache= \
  --repository_cache= \
  --nocache_test_results \
  --spawn_strategy=local \
  --copt=-march=x86-64 \
  //utils:snap_converter -- -s -i "$TEMP_SNAP" -o "$ABS_OUTPUT"

echo "Done! Successfully generated $OUTPUT_ADJ"
