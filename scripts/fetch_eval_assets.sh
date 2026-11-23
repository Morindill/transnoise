#!/usr/bin/env bash
# Download the files the evaluation needs that are not Python packages: the DeepFilterNet3 binary + ONNX model and the DNSMOS model.
# Existing files are skipped; any failed step (including a 404) aborts the whole script. Run from the project root: bash scripts/fetch_eval_assets.sh
set -euo pipefail
cd "$(dirname "$0")/.."

fetch() {  # fetch <output path> <URL>
    if [ -s "$1" ]; then echo "已存在，跳过: $1"; return; fi
    mkdir -p "$(dirname "$1")"
    echo "下载: $1"
    curl -fL --retry 3 -o "$1.part" "$2" && mv "$1.part" "$1"
}

DF=https://github.com/Rikorose/DeepFilterNet
fetch models/dfn3/bin/deep-filter "$DF/releases/download/v0.5.6/deep-filter-0.5.6-x86_64-unknown-linux-musl"
chmod +x models/dfn3/bin/deep-filter
fetch models/dfn3/DeepFilterNet3_onnx.tar.gz "https://raw.githubusercontent.com/Rikorose/DeepFilterNet/main/models/DeepFilterNet3_onnx.tar.gz"
fetch data/eval/assets/dnsmos/sig_bak_ovr.onnx "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"

echo "完成。deep-filter 版本: $(models/dfn3/bin/deep-filter --version 2>&1 | head -1)"
