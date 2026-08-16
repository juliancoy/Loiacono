#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT_DIR/build/asan"
APP_BIN="$BUILD_DIR/loiacono_spectrogram"

CMAKE_GENERATOR="${CMAKE_GENERATOR:-Ninja}"
BUILD_TYPE="${BUILD_TYPE:-Debug}"

mkdir -p "$BUILD_DIR"

qt_cmake_args=()
if [[ -n "${QT_PREFIX:-}" ]]; then
  qt_cmake_args+=("-DLOIACONO_QT_PREFIX=$QT_PREFIX")
fi

# Build with ASAN (Address Sanitizer)
cmake --fresh -S "$ROOT_DIR" -B "$BUILD_DIR" -G "$CMAKE_GENERATOR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  "${qt_cmake_args[@]}" \
  -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address"

cmake --build "$BUILD_DIR"

echo "ASAN build complete: $APP_BIN"
echo "Run with: ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 $APP_BIN"
