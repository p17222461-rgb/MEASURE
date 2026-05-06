#!/bin/bash
INPUT_DIR="$1"
OUTPUT_DIR="$2"
mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/*; do
    if [ -f "$file" ]; then
        surya_ocr "$file" --output_dir "$OUTPUT_DIR"
    fi
done