Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$MadmomRoot = [System.IO.Path]::GetFullPath(
  (Join-Path $PSScriptRoot "..")
)
$EnvironmentPython = Join-Path $MadmomRoot (
  ".build-env\Scripts\python.exe"
)

if (-not (Test-Path -LiteralPath $EnvironmentPython -PathType Leaf)) {
  $Message = @"
The Madmom test environment does not exist.
Create it, compile the native extensions, and run the initial suite with:

  .\scripts\build.ps1
"@
  Write-Error -Message $Message -ErrorAction Continue
  exit 2
}

$Preflight = @'
import struct
import sys

if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        "Madmom tests require CPython 3.11; found " + sys.version.split()[0]
    )
if struct.calcsize("P") != 8:
    raise SystemExit("Madmom tests require a 64-bit Python interpreter")

from madmom import models
from madmom.ml import hmm
from madmom.ml.nn import layers

if not models.BEATS_LSTM:
    raise SystemExit(
        "Madmom model files are missing; initialize the model submodule"
    )
'@

$PytestArguments = @($args)
if ($PytestArguments.Count -eq 0) {
  $PytestArguments = @("-q", "tests/test_spectrum_boundary.py")
}

Push-Location $MadmomRoot
try {
  $Preflight | & $EnvironmentPython -
  if ($LASTEXITCODE -ne 0) {
    $Message = @"
The Madmom test environment failed its preflight check.
Rebuild it with:

  .\scripts\build.ps1
"@
    Write-Error -Message $Message -ErrorAction Continue
    exit 2
  }

  & $EnvironmentPython -m pytest @PytestArguments
  $TestExitCode = $LASTEXITCODE
} finally {
  Pop-Location
}

exit $TestExitCode
