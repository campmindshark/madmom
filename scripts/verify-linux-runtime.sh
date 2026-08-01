#!/usr/bin/env bash

# Prove that a staged Linux runtime can be copied to an unrelated path, keeps
# its interpreter links internal, loads PortAudio dynamically, and executes the
# real DBN tracker. This is separate from build.sh so release jobs can recheck a
# runtime after copying it into the final archive layout.

set -euo pipefail

if (($# < 1 || $# > 2)); then
  printf 'Usage: scripts/verify-linux-runtime.sh RUNTIME [SAMPLE_WAV]\n' >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
madmom_root=$(cd -- "$script_dir/.." && pwd)
source_runtime=$(realpath -- "$1")
sample=${2:-"$madmom_root/tests/data/audio/sample.wav"}
sample=$(realpath -- "$sample")

if [[ $(uname -s) != Linux ]]; then
  printf 'Runtime relocation verification must run on Linux.\n' >&2
  exit 2
fi
if [[ ! -x "$source_runtime/bin/python" ]]; then
  printf 'Runtime has no executable bin/python: %s\n' "$source_runtime" >&2
  exit 2
fi
if [[ ! -f "$source_runtime/bin/DBNBeatTracker" ]]; then
  printf 'Runtime has no DBNBeatTracker: %s\n' "$source_runtime" >&2
  exit 2
fi
if [[ ! -f "$sample" ]]; then
  printf 'Smoke-test audio file is missing: %s\n' "$sample" >&2
  exit 2
fi

temp_root=$(mktemp -d "${TMPDIR:-/tmp}/spectrum-madmom-runtime.XXXXXX")
cleanup() {
  rm -rf -- "$temp_root"
}
trap cleanup EXIT

cp -a -- "$source_runtime" "$temp_root/runtime"
runtime_python="$temp_root/runtime/bin/python"
resolved_python=$(realpath -- "$runtime_python")
case "$resolved_python" in
  "$temp_root"/*) ;;
  *)
    printf 'Portable Python resolves outside its runtime: %s\n' \
      "$resolved_python" >&2
    exit 1
    ;;
esac

output=$(
  "$runtime_python" \
    "$temp_root/runtime/bin/DBNBeatTracker" \
    --host_api_name auto single "$sample" 2>&1
)
beat_count=$(grep -c '^BEAT:' <<<"$output")
if [[ "$beat_count" -ne 8 ]]; then
  printf '%s\n' "$output" >&2
  printf 'Expected 8 DBN smoke-test beats; found %s.\n' "$beat_count" >&2
  exit 1
fi

pcm_output=$(
  ffmpeg -loglevel error -i "$sample" \
    -f s16le -acodec pcm_s16le -ar 44100 -ac 1 - 2>/dev/null |
    "$runtime_python" \
      "$temp_root/runtime/bin/DBNBeatTracker" \
      --pcm_stdin online 2>&1
)
pcm_beat_count=$(grep -c '^BEAT:' <<<"$pcm_output")
if [[ "$pcm_beat_count" -ne 5 ]]; then
  printf '%s\n' "$pcm_output" >&2
  printf 'Expected 5 PCM-stdin DBN smoke-test beats; found %s.\n' \
    "$pcm_beat_count" >&2
  exit 1
fi

pyaudio_module=$(
  "$runtime_python" -c \
    'import pyaudio; print(pyaudio._portaudio.__file__)'
)
if ! ldd "$pyaudio_module" | grep -q 'libportaudio\.so\.2'; then
  ldd "$pyaudio_module" >&2
  printf 'PyAudio is not linked to libportaudio.so.2.\n' >&2
  exit 1
fi

printf 'Relocated Linux runtime: OK (%s file beats, %s PCM beats)\n' \
  "$beat_count" "$pcm_beat_count"
