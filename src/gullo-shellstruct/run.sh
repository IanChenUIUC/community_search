#!/usr/bin/env bash
# Usage: ./run.sh <fully.qualified.MainClass> [args...]
# Example: ./run.sh main.BuildIndex_PKDDJ Datasets/mygraph.txt
set -e
cd "$(dirname "$0")"          # anchor to the project dir so the relative classpath resolves
CP="target/classes:target/dependency/*"
MAIN_CLASS="$1"
shift
java \
  -cp "$CP" \
  --add-exports java.base/sun.nio.ch=ALL-UNNAMED \
  --add-opens java.base/sun.nio.ch=ALL-UNNAMED \
  "$MAIN_CLASS" "$@"
