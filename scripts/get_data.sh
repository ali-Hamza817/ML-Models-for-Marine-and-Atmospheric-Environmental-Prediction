#!/usr/bin/env bash
# Download the authors' processed datasets from Zenodo (10.5281/zenodo.19510440) into repo/data/processed.
set -euo pipefail
cd "$(dirname "$0")/.."
URL="https://zenodo.org/api/records/19510440/files/marine-ml-benchmark.zip/content"
ZIP=zenodo_marine-ml-benchmark.zip
if [ ! -f "$ZIP" ]; then
  if command -v aria2c >/dev/null; then aria2c -x 8 -s 8 --max-tries=0 -o "$ZIP" "$URL"; else curl -L --retry 5 -C - -o "$ZIP" "$URL"; fi
fi
unzip -q -o "$ZIP" "marine-ml-benchmark/data/processed/*" -d /tmp/mmlb
mkdir -p repo/data && rm -rf repo/data/processed && mv /tmp/mmlb/marine-ml-benchmark/data/processed repo/data/processed
rm -rf /tmp/mmlb
echo "Data ready in repo/data/processed"
