[CmdletBinding()]
param(
  [string]$PythonPath,
  [string]$PythonVersion = "3.11.15",
  [string]$EnvironmentDirectory,
  [string]$WheelDirectory,
  [string]$PortableRuntimeDirectory,
  [switch]$SkipTests
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$MadmomRoot = [System.IO.Path]::GetFullPath(
  (Join-Path $PSScriptRoot "..")
)
$RepositoryRoot = [System.IO.Path]::GetFullPath(
  (Join-Path $MadmomRoot "..")
)

if (-not $EnvironmentDirectory) {
  $EnvironmentDirectory = Join-Path $MadmomRoot ".build-env"
}
if (-not $WheelDirectory) {
  $WheelDirectory = Join-Path $MadmomRoot "dist"
}

$EnvironmentDirectory = [System.IO.Path]::GetFullPath($EnvironmentDirectory)
$WheelDirectory = [System.IO.Path]::GetFullPath($WheelDirectory)
if ($PortableRuntimeDirectory) {
  $PortableRuntimeDirectory = [System.IO.Path]::GetFullPath(
    $PortableRuntimeDirectory
  )
}

function Write-Step {
  param([string]$Message)
  Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Invoke-Checked {
  param(
    [string]$FilePath,
    [string[]]$Arguments
  )

  & $FilePath @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
  }
}

function Assert-WorkspacePath {
  param([string]$Path)

  $fullPath = [System.IO.Path]::GetFullPath($Path)
  $rootWithSeparator = $RepositoryRoot.TrimEnd('\') + '\'
  if (-not $fullPath.StartsWith(
      $rootWithSeparator,
      [System.StringComparison]::OrdinalIgnoreCase
    )) {
    throw "Refusing to modify a path outside the repository: $fullPath"
  }
  if ($fullPath.Equals(
      $RepositoryRoot,
      [System.StringComparison]::OrdinalIgnoreCase
    )) {
    throw "Refusing to modify the repository root."
  }
}

function Remove-WorkspaceDirectory {
  param([string]$Path)

  Assert-WorkspacePath $Path
  if (Test-Path -LiteralPath $Path) {
    Remove-Item -LiteralPath $Path -Recurse -Force
  }
}

function Get-ManagedPython {
  param(
    [string]$UvPath,
    [string]$Version
  )

  $installDirectory = Join-Path $MadmomRoot ".python"
  $candidate = Join-Path $installDirectory (
    "cpython-$Version-windows-x86_64-none\python.exe"
  )
  if (-not (Test-Path -LiteralPath $candidate)) {
    Write-Step "Installing local CPython $Version"
    New-Item -ItemType Directory -Path $installDirectory -Force | Out-Null
    Invoke-Checked -FilePath $UvPath -Arguments @(
      "python", "install",
      "--no-config",
      "--install-dir", $installDirectory,
      "--no-bin",
      "--no-registry",
      $Version
    )
  }
  if (-not (Test-Path -LiteralPath $candidate)) {
    throw "uv completed, but the expected interpreter was not found: $candidate"
  }
  return [System.IO.Path]::GetFullPath($candidate)
}

function Enter-MsvcEnvironment {
  $vswhereCandidates = @(
    (Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"),
    (Join-Path $env:ProgramFiles "Microsoft Visual Studio\Installer\vswhere.exe")
  )
  $vswhere = $vswhereCandidates |
    Where-Object { Test-Path -LiteralPath $_ } |
    Select-Object -First 1
  if (-not $vswhere) {
    throw "Visual Studio Installer's vswhere.exe was not found. Install the Desktop development with C++ workload."
  }

  $vsInstallPath = & $vswhere `
    -latest `
    -products "*" `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
  if ($LASTEXITCODE -ne 0 -or -not $vsInstallPath) {
    throw "No Visual Studio installation with the x64 C++ toolchain was found."
  }
  $vsInstallPath = ($vsInstallPath | Select-Object -First 1).Trim()

  $devShellModule = Join-Path $vsInstallPath (
    "Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
  )
  if (-not (Test-Path -LiteralPath $devShellModule)) {
    throw "Visual Studio developer-shell module was not found: $devShellModule"
  }

  Import-Module $devShellModule -Force
  Enter-VsDevShell `
    -VsInstallPath $vsInstallPath `
    -SkipAutomaticLocation `
    -DevCmdArguments "-arch=x64 -host_arch=x64" | Out-Null

  $compiler = Get-Command cl.exe -ErrorAction SilentlyContinue
  if (-not $compiler) {
    throw "Visual Studio was found, but cl.exe is not available after developer-shell activation."
  }
  Write-Host "MSVC: $($compiler.Source)"
}

function Invoke-DbnSmokeTest {
  param(
    [string]$Interpreter,
    [string]$ScriptsDirectory
  )

  $tracker = Join-Path $ScriptsDirectory "DBNBeatTracker"
  $sample = Join-Path $MadmomRoot "tests\data\audio\sample.wav"
  if (-not (Test-Path -LiteralPath $tracker)) {
    throw "The installed wheel did not provide DBNBeatTracker: $tracker"
  }

  $output = & $Interpreter $tracker "--host_api" "single" $sample 2>&1
  if ($LASTEXITCODE -ne 0) {
    $output | ForEach-Object { Write-Host $_ }
    throw "DBNBeatTracker smoke test failed with exit code $LASTEXITCODE."
  }
  $beatLines = @($output | Where-Object { "$_" -like "BEAT:*" })
  if ($beatLines.Count -eq 0) {
    $output | ForEach-Object { Write-Host $_ }
    throw "DBNBeatTracker completed without emitting any BEAT: events."
  }
  Write-Host "DBN smoke test: $($beatLines.Count) beat events"
}

$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCommand) {
  throw "uv is required. Install it with: winget install --id astral-sh.uv -e"
}
$uv = $uvCommand.Source

$uvCache = Join-Path $MadmomRoot ".uv-cache"
New-Item -ItemType Directory -Path $uvCache -Force | Out-Null
$env:UV_CACHE_DIR = $uvCache
$env:PYTHONNOUSERSITE = "1"

if ($PythonPath) {
  $buildPython = (Resolve-Path -LiteralPath $PythonPath).Path
} else {
  $buildPython = Get-ManagedPython -UvPath $uv -Version $PythonVersion
}

Invoke-Checked -FilePath $buildPython -Arguments @(
  "-c",
  "import struct, sys; assert sys.version_info[:2] == (3, 11), sys.version; assert struct.calcsize('P') == 8, 'x64 Python required'; print(sys.version)"
)

$requiredModel = Join-Path $MadmomRoot (
  "madmom\models\beats\2016\beats_lstm_1.pkl"
)
if (-not (Test-Path -LiteralPath $requiredModel)) {
  throw "Madmom model files are missing. Initialize submodules with: git submodule update --init --recursive"
}

Assert-WorkspacePath $EnvironmentDirectory
Assert-WorkspacePath $WheelDirectory
if ($PortableRuntimeDirectory) {
  Assert-WorkspacePath $PortableRuntimeDirectory
}

Write-Step "Creating the Python 3.11 development environment"
Invoke-Checked -FilePath $uv -Arguments @(
  "venv", "--no-config", "--clear",
  "--python", $buildPython,
  $EnvironmentDirectory
)
$environmentPython = Join-Path $EnvironmentDirectory "Scripts\python.exe"
Invoke-Checked -FilePath $uv -Arguments @(
  "pip", "install", "--no-config",
  "--python", $environmentPython,
  "--no-deps",
  "--requirements", (Join-Path $MadmomRoot "requirements-dev.txt")
)

Write-Step "Activating the Visual C++ compiler"
Enter-MsvcEnvironment

$buildDirectory = Join-Path $MadmomRoot "build"
Remove-WorkspaceDirectory $buildDirectory
Remove-WorkspaceDirectory $WheelDirectory
New-Item -ItemType Directory -Path $buildDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $WheelDirectory -Force | Out-Null

# Old ABI-tagged files can make a source checkout appear healthier than it is.
Get-ChildItem -LiteralPath (Join-Path $MadmomRoot "madmom") `
  -Recurse `
  -Filter "*.pyd" |
  ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }

Write-Step "Compiling the native extensions in place"
Push-Location $MadmomRoot
try {
  Invoke-Checked -FilePath $environmentPython -Arguments @(
    "setup.py", "build_ext", "--inplace", "--force"
  )

  Write-Step "Building the CPython 3.11 wheel"
  Invoke-Checked -FilePath $environmentPython -Arguments @(
    "-m", "pip", "wheel",
    "--no-build-isolation",
    "--no-deps",
    "--no-cache-dir",
    "--wheel-dir", $WheelDirectory,
    "."
  )
} finally {
  Pop-Location
}

$wheels = @(Get-ChildItem -LiteralPath $WheelDirectory -Filter "*.whl")
if ($wheels.Count -ne 1) {
  throw "Expected one Madmom wheel in $WheelDirectory; found $($wheels.Count)."
}
$wheel = $wheels[0].FullName

Write-Step "Installing and checking the wheel"
Invoke-Checked -FilePath $uv -Arguments @(
  "pip", "install", "--no-config",
  "--python", $environmentPython,
  "--no-deps",
  "--force-reinstall",
  $wheel
)
Invoke-Checked -FilePath $environmentPython -Arguments @("-m", "pip", "check")

if (-not $SkipTests) {
  Write-Step "Running the Madmom test suite"
  Push-Location $MadmomRoot
  try {
    Invoke-Checked -FilePath $environmentPython -Arguments @("-m", "pytest", "-q")
  } finally {
    Pop-Location
  }
}

Write-Step "Testing the wheel in a fresh environment"
$wheelTestEnvironment = Join-Path $buildDirectory "wheel-test-env"
Invoke-Checked -FilePath $uv -Arguments @(
  "venv", "--no-config", "--clear",
  "--python", $buildPython,
  $wheelTestEnvironment
)
$wheelTestPython = Join-Path $wheelTestEnvironment "Scripts\python.exe"
Invoke-Checked -FilePath $uv -Arguments @(
  "pip", "install", "--no-config",
  "--python", $wheelTestPython,
  "--no-deps",
  "--requirements", (Join-Path $MadmomRoot "requirements-runtime.txt"),
  $wheel
)
Invoke-Checked -FilePath $uv -Arguments @(
  "pip", "check", "--no-config", "--python", $wheelTestPython
)

Push-Location $buildDirectory
try {
  Invoke-Checked -FilePath $wheelTestPython -Arguments @(
    "-c",
    "from madmom import models; from madmom.audio import comb_filters; from madmom.features import beats_crf; from madmom.ml import hmm; from madmom.ml.nn import layers; assert len(models.BEATS_LSTM) == 8; assert len(models.BEATS_TCN) == 8; print('wheel imports and models: OK')"
  )
} finally {
  Pop-Location
}
Invoke-DbnSmokeTest `
  -Interpreter $wheelTestPython `
  -ScriptsDirectory (Join-Path $wheelTestEnvironment "Scripts")

if ($PortableRuntimeDirectory) {
  Write-Step "Staging the portable Python runtime"
  $portableBasePython = Get-ManagedPython -UvPath $uv -Version $PythonVersion
  $portableBaseDirectory = Split-Path -Parent $portableBasePython

  Remove-WorkspaceDirectory $PortableRuntimeDirectory
  New-Item -ItemType Directory -Path $PortableRuntimeDirectory -Force | Out-Null
  Copy-Item `
    -Path (Join-Path $portableBaseDirectory "*") `
    -Destination $PortableRuntimeDirectory `
    -Recurse `
    -Force

  $runtimePython = Join-Path $PortableRuntimeDirectory "python.exe"
  Invoke-Checked -FilePath $uv -Arguments @(
    "pip", "install", "--no-config",
    "--python", $runtimePython,
    # This is a private copy made specifically for the release artifact.
    "--break-system-packages",
    "--no-deps",
    "--requirements", (Join-Path $MadmomRoot "requirements-runtime.txt"),
    $wheel
  )
  Invoke-Checked -FilePath $uv -Arguments @(
    "pip", "check", "--no-config", "--python", $runtimePython
  )

  Push-Location $PortableRuntimeDirectory
  try {
    Invoke-Checked -FilePath $runtimePython -Arguments @(
      "-c",
      "import madmom, numpy, scipy, pyaudio; from madmom import models; assert len(models.BEATS_LSTM) == 8; print('portable runtime:', madmom.__version__)"
    )
  } finally {
    Pop-Location
  }
  Invoke-DbnSmokeTest `
    -Interpreter $runtimePython `
    -ScriptsDirectory (Join-Path $PortableRuntimeDirectory "Scripts")
}

Write-Host "`nPython build complete." -ForegroundColor Green
Write-Host "Environment: $EnvironmentDirectory"
Write-Host "Wheel: $wheel"
if ($PortableRuntimeDirectory) {
  Write-Host "Portable runtime: $PortableRuntimeDirectory"
}
