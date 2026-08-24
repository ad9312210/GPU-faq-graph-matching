# Build and Run Script for SCGM CUDA/CSR/LAPJV Backend
# Windows + NVIDIA CUDA

param(
    [switch]$Clean,
    [switch]$Build,
    [switch]$Test,
    [switch]$Run,
    [switch]$All,
    [string]$BuildType = "Release",
    [string]$CudaArch = "native"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = Split-Path -Parent $scriptDir
$buildDir = Join-Path $projectDir "build"

Write-Host "SCGM CUDA/CSR/LAPJV Build Script" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host "Project: $projectDir"
Write-Host "Build:   $buildDir"
Write-Host "Type:    $BuildType"
Write-Host "CUDA Arch: $CudaArch"
Write-Host ""

if ($All) {
    $Clean = $true
    $Build = $true
    $Test = $true
    $Run = $true
}

if ($Clean) {
    Write-Host "Cleaning build directory..." -ForegroundColor Yellow
    if (Test-Path $buildDir) {
        Remove-Item -Recurse -Force $buildDir
    }
}

if ($Build -or -not (Test-Path $buildDir)) {
    Write-Host "Configuring CMake..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
    Push-Location $buildDir

    cmake $projectDir `
        -DCMAKE_BUILD_TYPE=$BuildType `
        -DCMAKE_CUDA_ARCHITECTURES=$CudaArch

    Write-Host "Building..." -ForegroundColor Yellow
    cmake --build . --config $BuildType --parallel

    Pop-Location
}

if ($Test) {
    Write-Host ""
    Write-Host "Running Tests..." -ForegroundColor Yellow
    Write-Host "=================" -ForegroundColor Yellow

    Push-Location $buildDir

    $testExes = @(
        "test_features",
        "test_permutation",
        "test_csr",
        "test_topk",
        "test_lapjv",
        "test_cpu_gpu_parity",
        "test_toy_scgm"
    )

    $totalPassed = 0
    $totalFailed = 0

    foreach ($test in $testExes) {
        $exePath = Join-Path $buildDir "$BuildType\$test.exe"
        if (-not (Test-Path $exePath)) {
            $exePath = Join-Path $buildDir "$test.exe"
        }
        if (-not (Test-Path $exePath)) {
            # Try Debug folder
            $exePath = Join-Path $buildDir "Debug\$test.exe"
        }

        if (Test-Path $exePath) {
            Write-Host ""
            Write-Host "--- $test ---" -ForegroundColor Cyan
            & $exePath
            if ($LASTEXITCODE -eq 0) {
                $totalPassed++
            } else {
                $totalFailed++
                Write-Host "$test FAILED (exit code $LASTEXITCODE)" -ForegroundColor Red
            }
        } else {
            Write-Host "WARNING: $test not found" -ForegroundColor Yellow
        }
    }

    Write-Host ""
    Write-Host "Test Summary: $totalPassed passed, $totalFailed failed" -ForegroundColor $(if ($totalFailed -eq 0) { "Green" } else { "Red" })

    Pop-Location
}

if ($Run) {
    Write-Host ""
    Write-Host "Running Toy Example..." -ForegroundColor Yellow
    Write-Host "======================" -ForegroundColor Yellow

    $exePath = Join-Path $buildDir "$BuildType\scgm_cuda.exe"
    if (-not (Test-Path $exePath)) {
        $exePath = Join-Path $buildDir "scgm_cuda.exe"
    }
    if (-not (Test-Path $exePath)) {
        $exePath = Join-Path $buildDir "Debug\scgm_cuda.exe"
    }

    if (Test-Path $exePath) {
        & $exePath --toy
    } else {
        Write-Host "ERROR: scgm_cuda.exe not found" -ForegroundColor Red
    }
}