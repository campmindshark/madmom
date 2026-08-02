#!/usr/bin/env bash

# Build Spectrum's Linux Madmom wheel and, optionally, a relocatable CPython
# runtime for the headless release. This intentionally mirrors build.ps1 while
# keeping every generated path inside the checkout so cleanup is bounded.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
madmom_root=$(cd -- "$script_dir/.." && pwd)
repository_root=$(cd -- "$madmom_root/.." && pwd)

python_version="3.11.15"
python_path=""
environment_directory="$madmom_root/.build-env"
wheel_directory="$madmom_root/dist"
portable_runtime_directory=""
skip_tests=false

usage() {
  cat <<'EOF'
Usage: scripts/build.sh [options]

Options:
  --python PATH                    Use an existing CPython 3.11 interpreter.
  --python-version VERSION         Managed CPython version (default: 3.11.15).
  --environment-directory PATH     Build virtual environment.
  --wheel-directory PATH           Wheel output directory.
  --portable-runtime-directory PATH
                                   Stage a relocatable CPython runtime.
  --skip-tests                     Skip pytest; wheel/runtime smoke tests remain.
  -h, --help                       Show this help.
EOF
}

while (($#)); do
  case "$1" in
    --python)
      python_path=${2:?"--python requires a path"}
      shift 2
      ;;
    --python-version)
      python_version=${2:?"--python-version requires a value"}
      shift 2
      ;;
    --environment-directory)
      environment_directory=${2:?"--environment-directory requires a path"}
      shift 2
      ;;
    --wheel-directory)
      wheel_directory=${2:?"--wheel-directory requires a path"}
      shift 2
      ;;
    --portable-runtime-directory)
      portable_runtime_directory=${2:?"--portable-runtime-directory requires a path"}
      shift 2
      ;;
    --skip-tests)
      skip_tests=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

step() {
  printf '\n==> %s\n' "$1"
}

absolute_path() {
  realpath -m -- "$1"
}

assert_workspace_path() {
  local candidate
  candidate=$(absolute_path "$1")
  case "$candidate" in
    "$repository_root"/*) ;;
    *)
      printf 'Refusing to modify a path outside the repository: %s\n' \
        "$candidate" >&2
      exit 2
      ;;
  esac
  if [[ "$candidate" == "$repository_root" ]]; then
    printf 'Refusing to modify the repository root.\n' >&2
    exit 2
  fi
}

reset_directory() {
  local target
  target=$(absolute_path "$1")
  assert_workspace_path "$target"
  rm -rf -- "$target"
  mkdir -p -- "$target"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf '%s is required.\n' "$1" >&2
    exit 2
  fi
}

get_managed_python() {
  local install_directory="$madmom_root/.python"
  mkdir -p -- "$install_directory"
  UV_PYTHON_INSTALL_DIR="$install_directory" \
    uv python install --no-config --no-bin "$python_version" >&2
  UV_PYTHON_INSTALL_DIR="$install_directory" \
    uv python find --no-config --python-preference only-managed \
      "$python_version"
}

assert_python() {
  "$1" -c \
    "import platform, struct, sys; assert sys.version_info[:2] == (3, 11), sys.version; assert struct.calcsize('P') == 8, 'x64 Python required'; assert platform.machine().lower() in ('x86_64', 'amd64'), platform.machine(); print(sys.version)"
}

assert_native_imports() {
  "$1" -c \
    "from importlib import util; from pathlib import Path; import madmom, numpy, pyaudio, scipy; from madmom import models; from madmom.audio import comb_filters; from madmom.features import beats_crf; from madmom.ml import hmm; from madmom.ml.nn import layers; native = (comb_filters, beats_crf, hmm, layers); assert all(Path(module.__file__).suffix == '.so' for module in native), [module.__file__ for module in native]; assert len(models.BEATS_LSTM) == 8; assert not models.BEATS_BLSTM; assert not models.BEATS_TCN; assert util.find_spec('mido') is None; assert util.find_spec('madmom.evaluation') is None; assert util.find_spec('madmom.piracy') is None; print('native runtime imports: OK'); print(*(module.__file__ for module in native), sep='\n')"
}

dbn_smoke_test() {
  local interpreter=$1
  local scripts_directory=$2
  local tracker="$scripts_directory/DBNBeatTracker"
  local sample="$madmom_root/tests/data/audio/sample.wav"
  local output

  if [[ ! -f "$tracker" ]]; then
    printf 'The installed wheel did not provide DBNBeatTracker: %s\n' \
      "$tracker" >&2
    exit 1
  fi
  output=$(
    "$interpreter" "$tracker" --host_api_name auto online "$sample" 2>&1
  )
  if ! grep -q '^BEAT:' <<<"$output"; then
    printf '%s\n' "$output" >&2
    printf 'DBNBeatTracker completed without emitting any BEAT events.\n' >&2
    exit 1
  fi
  printf 'DBN smoke test: %s beat events\n' \
    "$(grep -c '^BEAT:' <<<"$output")"
}

dbn_pcm_smoke_test() {
  local interpreter=$1
  local scripts_directory=$2
  local tracker="$scripts_directory/DBNBeatTracker"
  local sample="$madmom_root/tests/data/audio/sample.wav"
  local output
  local beat_count

  output=$(
    ffmpeg -loglevel error -i "$sample" \
      -f s16le -acodec pcm_s16le -ar 44100 -ac 1 - 2>/dev/null |
      "$interpreter" "$tracker" --pcm_stdin online 2>&1
  )
  beat_count=$(grep -c '^BEAT:' <<<"$output")
  if [[ "$beat_count" -ne 5 ]]; then
    printf '%s\n' "$output" >&2
    printf 'Expected 5 PCM-stdin DBN smoke-test beats; found %s.\n' \
      "$beat_count" >&2
    exit 1
  fi
  printf 'DBN PCM-stdin smoke test: %s beat events\n' "$beat_count"
}

require_command realpath
require_command uv
require_command gcc

if [[ $(uname -s) != Linux ]]; then
  printf 'scripts/build.sh builds the Linux runtime and must run on Linux.\n' >&2
  exit 2
fi

environment_directory=$(absolute_path "$environment_directory")
wheel_directory=$(absolute_path "$wheel_directory")
assert_workspace_path "$environment_directory"
assert_workspace_path "$wheel_directory"
if [[ -n "$portable_runtime_directory" ]]; then
  portable_runtime_directory=$(absolute_path "$portable_runtime_directory")
  assert_workspace_path "$portable_runtime_directory"
fi

export UV_CACHE_DIR="$madmom_root/.uv-cache"
export PYTHONNOUSERSITE=1
mkdir -p -- "$UV_CACHE_DIR"

if [[ -n "$python_path" ]]; then
  build_python=$(absolute_path "$python_path")
  if [[ ! -x "$build_python" ]]; then
    printf 'Python interpreter is not executable: %s\n' "$build_python" >&2
    exit 2
  fi
else
  step "Installing managed CPython $python_version"
  build_python=$(get_managed_python)
fi
assert_python "$build_python"

required_model="$madmom_root/madmom/models/beats/2016/beats_lstm_1.pkl"
if [[ ! -f "$required_model" ]]; then
  printf '%s\n' \
    'Madmom model files are missing. Initialize submodules recursively.' >&2
  exit 2
fi

step 'Creating the Python 3.11 development environment'
uv venv --no-config --clear --python "$build_python" \
  "$environment_directory"
environment_python="$environment_directory/bin/python"
uv pip install --no-config --python "$environment_python" --no-deps \
  --requirements "$madmom_root/requirements-dev.txt"

build_directory="$madmom_root/build"
reset_directory "$build_directory"
reset_directory "$wheel_directory"

# Old ABI-tagged files can make a source checkout appear healthier than it is.
find "$madmom_root/madmom" -type f -name '*.so' -delete

step 'Compiling the Linux native extensions in place'
(
  cd -- "$madmom_root"
  "$environment_python" setup.py build_ext --inplace --force
)

step 'Building the CPython 3.11 Linux wheel'
(
  cd -- "$madmom_root"
  "$environment_python" -m pip wheel \
    --no-build-isolation \
    --no-deps \
    --no-cache-dir \
    --wheel-dir "$wheel_directory" \
    .
)

mapfile -t wheels < <(find "$wheel_directory" -maxdepth 1 -type f -name '*.whl')
if ((${#wheels[@]} != 1)); then
  printf 'Expected one Madmom wheel in %s; found %s.\n' \
    "$wheel_directory" "${#wheels[@]}" >&2
  exit 1
fi
wheel=${wheels[0]}

step 'Installing and checking the wheel'
uv pip install --no-config --python "$environment_python" --no-deps \
  --force-reinstall "$wheel"
uv pip check --no-config --python "$environment_python"

if [[ "$skip_tests" != true ]]; then
  step 'Running the Madmom test suite'
  (
    cd -- "$madmom_root"
    "$environment_python" -m pytest -q
  )
fi

step 'Testing the wheel in a fresh environment'
wheel_test_environment="$build_directory/wheel-test-env"
uv venv --no-config --clear --python "$build_python" \
  "$wheel_test_environment"
wheel_test_python="$wheel_test_environment/bin/python"
uv pip install --no-config --python "$wheel_test_python" --no-deps \
  --requirements "$madmom_root/requirements-runtime.txt" \
  "$wheel"
uv pip check --no-config --python "$wheel_test_python"
(
  cd -- "$build_directory"
  assert_native_imports "$wheel_test_python"
)
dbn_smoke_test "$wheel_test_python" "$wheel_test_environment/bin"
dbn_pcm_smoke_test "$wheel_test_python" "$wheel_test_environment/bin"

if [[ -n "$portable_runtime_directory" ]]; then
  step 'Staging the relocatable Linux Python runtime'
  managed_python=$(get_managed_python)
  managed_python=$(realpath -- "$managed_python")
  portable_base_directory=$(dirname -- "$(dirname -- "$managed_python")")

  reset_directory "$portable_runtime_directory"
  cp -a -- "$portable_base_directory/." "$portable_runtime_directory/"

  runtime_python="$portable_runtime_directory/bin/python"
  if [[ ! -x "$runtime_python" ]]; then
    printf 'Managed CPython copy has no bin/python: %s\n' \
      "$portable_runtime_directory" >&2
    exit 1
  fi
  if [[ $(realpath -- "$runtime_python") != "$portable_runtime_directory"/* ]]; then
    printf 'Portable Python resolves outside its runtime directory.\n' >&2
    exit 1
  fi

  uv pip install --no-config --python "$runtime_python" \
    --break-system-packages \
    --no-deps \
    --requirements "$madmom_root/requirements-runtime.txt" \
    "$wheel"
  uv pip check --no-config --python "$runtime_python"
  (
    cd -- "$portable_runtime_directory"
    assert_native_imports "$runtime_python"
  )
  dbn_smoke_test "$runtime_python" "$portable_runtime_directory/bin"
  dbn_pcm_smoke_test "$runtime_python" "$portable_runtime_directory/bin"

  step 'Verifying the runtime after relocation'
  bash "$script_dir/verify-linux-runtime.sh" \
    "$portable_runtime_directory" \
    "$madmom_root/tests/data/audio/sample.wav"

  # Release archives do not need bytecode caches left by the verification run.
  find "$portable_runtime_directory" -type d -name __pycache__ \
    -prune -exec rm -rf -- {} +
  find "$portable_runtime_directory" -type f \
    \( -name '*.pyc' -o -name '*.pyo' \) -delete
fi

printf '\nPython build complete.\n'
printf 'Environment: %s\n' "$environment_directory"
printf 'Wheel: %s\n' "$wheel"
if [[ -n "$portable_runtime_directory" ]]; then
  printf 'Portable runtime: %s\n' "$portable_runtime_directory"
fi
