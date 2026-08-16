#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT_DIR/build/root_qt"
APP_BIN="$BUILD_DIR/loiacono_spectrogram"

CMAKE_GENERATOR="${CMAKE_GENERATOR:-Ninja}"
BUILD_TYPE="${BUILD_TYPE:-Debug}"

run_app=true
if [[ "${1:-}" == "--build-only" ]]; then
  run_app=false
fi

mkdir -p "$BUILD_DIR"

qt_cmake_args=()
if [[ -n "${QT_PREFIX:-}" ]]; then
  qt_cmake_args+=("-DLOIACONO_QT_PREFIX=$QT_PREFIX")
fi

cmake --fresh -S "$ROOT_DIR" -B "$BUILD_DIR" -G "$CMAKE_GENERATOR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  "${qt_cmake_args[@]}"
cmake --build "$BUILD_DIR"

if [[ "$run_app" == true ]]; then
  exec "$APP_BIN"
fi
