#!/usr/bin/env bash
# Build skrrt-engine, the llama.cpp side of skrrt's GGUF backend.
#
#   llamacpp/build.sh                 CUDA build if a CUDA toolkit is found, else CPU
#   SKRRT_CUDA=0 llamacpp/build.sh    CPU-only build
#   SKRRT_CUDA=1 llamacpp/build.sh    CUDA build, fail if there is no toolkit
#
# It clones llama.cpp at the pinned commit into llamacpp/llama.cpp, applies every patches/*.patch,
# copies skrrt-engine/ into tools/ and builds only that target (statically linked):
#   llamacpp/build/cuda/bin/skrrt-engine   or   llamacpp/build/cpu/bin/skrrt-engine
# The two builds live side by side; the Python backend picks the one that fits n_gpu_layers.
#
# Other knobs: SKRRT_LLAMA_REPO (git URL or local clone to fetch from), SKRRT_LLAMA_COMMIT,
# SKRRT_BUILD_DIR, SKRRT_CUDA_ARCH (e.g. 120; default: asked from nvidia-smi), SKRRT_JOBS,
# CMAKE_ARGS (extra cmake configure arguments).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${SKRRT_LLAMA_REPO:-https://github.com/ggml-org/llama.cpp.git}"
COMMIT="${SKRRT_LLAMA_COMMIT:-29b52d398b294509746ec0a3f28948a3107c0213}"
SRC="$HERE/llama.cpp"
JOBS="${SKRRT_JOBS:-$(nproc 2>/dev/null || echo 4)}"
say() { echo "[skrrt build] $*"; }

# ---- source tree at the pinned commit
if [ ! -d "$SRC/.git" ]; then
    say "fetching llama.cpp $COMMIT"
    git init -q "$SRC"
    git -C "$SRC" remote add origin "$REPO"
fi
if [ "$(git -C "$SRC" rev-parse -q --verify HEAD 2>/dev/null || true)" != "$COMMIT" ]; then
    git -C "$SRC" fetch -q --depth 1 origin "$COMMIT"
    git -C "$SRC" checkout -q -f FETCH_HEAD
fi

# ---- patches (skipped when already applied)
shopt -s nullglob
for p in "$HERE"/patches/*.patch; do
    if git -C "$SRC" apply --reverse --check "$p" 2>/dev/null; then
        say "patch already applied: $(basename "$p")"
    else
        git -C "$SRC" apply "$p"
        say "applied patch: $(basename "$p")"
    fi
done
shopt -u nullglob

# ---- the tool
mkdir -p "$SRC/tools/skrrt-engine"
for f in CMakeLists.txt skrrt-engine.cpp; do
    cmp -s "$HERE/skrrt-engine/$f" "$SRC/tools/skrrt-engine/$f" || cp "$HERE/skrrt-engine/$f" "$SRC/tools/skrrt-engine/$f"
done
grep -qxF 'add_subdirectory(skrrt-engine)' "$SRC/tools/CMakeLists.txt" || echo 'add_subdirectory(skrrt-engine)' >> "$SRC/tools/CMakeLists.txt"

# ---- CUDA: only what is detected gets passed on
want="${SKRRT_CUDA:-auto}"
nvcc=""
if [ "$want" != "0" ]; then
    for c in "$(command -v nvcc 2>/dev/null || true)" "${CUDA_HOME:-}/bin/nvcc" "${CUDA_PATH:-}/bin/nvcc" /usr/local/cuda/bin/nvcc; do
        if [ -n "$c" ] && [ -x "$c" ]; then nvcc="$c"; break; fi
    done
    if [ -z "$nvcc" ]; then
        if [ "$want" = "1" ]; then echo "SKRRT_CUDA=1 but no nvcc found (PATH, CUDA_HOME, /usr/local/cuda)" >&2; exit 1; fi
        say "no CUDA toolkit found, building CPU-only"
    fi
fi

args=(-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=OFF
      -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF)
if [ -n "$nvcc" ]; then
    kind=cuda
    export PATH="$(dirname "$nvcc"):$PATH"
    args+=(-DGGML_CUDA=ON "-DCMAKE_CUDA_COMPILER=$nvcc")
    arch="${SKRRT_CUDA_ARCH:-}"
    if [ -z "$arch" ] && command -v nvidia-smi >/dev/null 2>&1; then
        # compute capability of the first GPU, "12.0" -> 120 (a read-only query)
        arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. \r' || true)"
    fi
    if [[ "$arch" =~ ^[0-9]+$ ]]; then args+=("-DCMAKE_CUDA_ARCHITECTURES=$arch"); fi
else
    kind=cpu
    args+=(-DGGML_CUDA=OFF)
fi
BUILD="${SKRRT_BUILD_DIR:-$HERE/build/$kind}"

say "configuring ($kind) in $BUILD"
# shellcheck disable=SC2086
cmake -S "$SRC" -B "$BUILD" "${args[@]}" ${CMAKE_ARGS:-}
say "building skrrt-engine with $JOBS jobs"
cmake --build "$BUILD" --target skrrt-engine -j "$JOBS"
say "done: $BUILD/bin/skrrt-engine"
