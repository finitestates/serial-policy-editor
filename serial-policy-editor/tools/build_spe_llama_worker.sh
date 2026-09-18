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
if [[ ! -f "${llama_root}/src/llama-ext.h" ]]; then
    echo "the SPE worker requires the llama.cpp staging header ${llama_root}/src/llama-ext.h" >&2
    exit 2
fi
if ! command -v c++ >/dev/null 2>&1; then
    echo "a C++17 compiler named c++ is required to build the SPE worker" >&2
    exit 2
fi

for library in llama llama-common ggml-cpu ggml-base ggml; do
    if [[ ! -e "${llama_build}/bin/lib${library}.so" && ! -e "${llama_build}/bin/lib${library}.so.0" ]]; then
        echo "llama.cpp library lib${library} not found under ${llama_build}/bin" >&2
        exit 2
    fi
done

if [[ ! -f "${repo_root}/tools/spe_llama_worker.cpp" ]]; then
    echo "SPE worker source is missing: ${repo_root}/tools/spe_llama_worker.cpp" >&2
    exit 2
fi

mkdir -p "$(dirname "${output}")"

c++ -std=c++17 -O2 \
    -I"${llama_root}/include" \
    -I"${llama_root}/src" \
    -I"${llama_root}/ggml/include" \
    "${repo_root}/tools/spe_llama_worker.cpp" \
    -L"${llama_build}/bin" \
    -Wl,-rpath,"${llama_build}/bin" \
    -lllama -lllama-common -lggml-cpu -lggml-base -lggml \
    -o "${output}"

llama_revision="$(git -C "${llama_root}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "built ${output} against llama.cpp revision ${llama_revision}" >&2
