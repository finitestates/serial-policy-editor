#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
llama_root="${SPE_LLAMA_CPP_ROOT:-${repo_root}/../llama.cpp}"
llama_build="${SPE_LLAMA_CPP_BUILD:-${llama_root}/build}"
output="${1:-${repo_root}/build/spe-llama-worker}"

if [[ ! -f "${llama_root}/include/llama.h" ]]; then
    echo "llama.cpp headers not found under ${llama_root}" >&2
    exit 2
fi
if [[ ! -f "${llama_build}/bin/libllama.so" && ! -f "${llama_build}/bin/libllama.so.0.3.0" ]]; then
    echo "llama.cpp build library not found under ${llama_build}/bin" >&2
    exit 2
fi

mkdir -p "$(dirname "${output}")"

c++ -std=c++17 -O2 \
    -I"${llama_root}/include" \
    -I"${llama_root}/ggml/include" \
    "${repo_root}/tools/spe_llama_worker.cpp" \
    -L"${llama_build}/bin" \
    -Wl,-rpath,"${llama_build}/bin" \
    -lllama -lllama-common -lggml-cpu -lggml-base -lggml \
    -o "${output}"

echo "built ${output}" >&2
