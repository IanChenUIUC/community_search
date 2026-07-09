#!/bin/bash

# Check if an input file was provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <input_graph.csv> <output_graph.csv>"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"

echo "Processing CSV graph..."

# Use awk to find the max node ID and total edges in one pass, skipping the CSV header
HEADER=$(awk -F',' '
    NR == 1 { next } # Skip the "source,target" text header line
    {
        if ($1 > max) max = $1;
        if ($2 > max) max = $2;
        edges++
    }
    END {
        # Total nodes = max_id + 1 (since 0-indexed)
        print (max + 1) "," edges
    }
' "$INPUT_FILE")

# Write the metadata header to the output file
echo "$HEADER" > "$OUTPUT_FILE"

# Append the original edgelist data
tail -n +2 "$INPUT_FILE" >> "$OUTPUT_FILE"

echo "Done! Saved to $OUTPUT_FILE"
echo "Header added (nodes,edges): $HEADER"
