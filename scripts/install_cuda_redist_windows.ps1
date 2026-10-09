$ErrorActionPreference = "Stop"

$cudaRoot = Join-Path $env:RUNNER_TEMP "cuda-12.8"
$downloadRoot = Join-Path $env:RUNNER_TEMP "cuda-redist-12.8"
New-Item -ItemType Directory -Force -Path $cudaRoot, $downloadRoot | Out-Null

# These are the CUDA 12.8.1 redistributable components used to build and bundle
# llama-cpp-python. Pin both NVIDIA's archive URLs and hashes so this step stays
# smaller than installing the full CUDA toolkit and fails closed on drift.
$components = @(
    @{
        Name = "CUDA compiler"
        File = "cuda_nvcc-windows-x86_64-12.8.93-archive.zip"
        Url = "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/windows-x86_64/cuda_nvcc-windows-x86_64-12.8.93-archive.zip"
        Sha256 = "9fdc70b4271ed9aad4d64cd7076a7d96ec36512d074b9995fe638de669197391"
    },
    @{
        Name = "CUDA runtime and headers"
        File = "cuda_cudart-windows-x86_64-12.8.90-archive.zip"
        Url = "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.8.90-archive.zip"
        Sha256 = "4a39058fd8519444a81cfc7ae055d136f48d1a31ffa41ae255b35b2edd61e13b"
    },
    @{
        Name = "CUDA CCCL headers"
        File = "cuda_cccl-windows-x86_64-12.8.90-archive.zip"
        Url = "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cccl/windows-x86_64/cuda_cccl-windows-x86_64-12.8.90-archive.zip"
        Sha256 = "bd8548fa1ae82f92910bebc3079e14bd58c5a92aa64596d46bd610a478cb39d7"
    },
    @{
        Name = "cuBLAS runtime and headers"
        File = "libcublas-windows-x86_64-12.8.4.1-archive.zip"
        Url = "https://developer.download.nvidia.com/compute/cublas/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-12.8.4.1-archive.zip"
        Sha256 = "57a470112cec7e112c95253dde8b3c7184d795dbd92b0bde77a4cb7f8c94c8aa"
    },
    @{
        Name = "Visual Studio integration"
        File = "visual_studio_integration-windows-x86_64-12.8.90-archive.zip"
        Url = "https://developer.download.nvidia.com/compute/cuda/redist/visual_studio_integration/windows-x86_64/visual_studio_integration-windows-x86_64-12.8.90-archive.zip"
        Sha256 = "f41d12a0e49b7848ed35e8a15b58926b83f635c723cab7e9952bc633e3c1f200"
    }
)

foreach ($component in $components) {
    $archivePath = Join-Path $downloadRoot $component.File
    $extractPath = Join-Path $downloadRoot ($component.File -replace "\.zip$", "")
    Write-Host "Downloading $($component.Name) from NVIDIA's pinned CUDA 12.8.1 redistributable"

    & curl.exe --fail --location --retry 5 --retry-all-errors --connect-timeout 30 `
        --progress-bar --output $archivePath $component.Url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed for $($component.Name) with curl exit code $LASTEXITCODE"
    }

    $actualHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $component.Sha256) {
        throw "SHA-256 mismatch for $($component.Name): expected $($component.Sha256), received $actualHash"
    }

    Expand-Archive -LiteralPath $archivePath -DestinationPath $extractPath -Force
    $items = @(Get-ChildItem -LiteralPath $extractPath -Force)
    if ($items.Count -eq 1 -and $items[0].PSIsContainer) {
        $packageRoot = $items[0].FullName
    } else {
        $packageRoot = $extractPath
    }

    & robocopy.exe $packageRoot $cudaRoot /E /COPY:DAT /DCOPY:DAT /R:1 /W:1 /NFL /NDL /NJH /NJS /NP
    if ($LASTEXITCODE -ge 8) {
        throw "Could not merge $($component.Name) into the CUDA toolkit root (robocopy exit code $LASTEXITCODE)"
    }
    Remove-Item -LiteralPath $archivePath, $extractPath -Recurse -Force
}

$requiredFiles = @(
    "bin\nvcc.exe",
    "include\cuda_runtime.h",
    "include\cublas_v2.h",
    "lib\x64\cudart.lib",
    "lib\x64\cublas.lib",
    "bin\cudart64_12.dll",
    "bin\cublas64_12.dll"
)
foreach ($relativePath in $requiredFiles) {
    $requiredPath = Join-Path $cudaRoot $relativePath
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Pinned CUDA components did not provide required file: $requiredPath"
    }
}

Add-Content -LiteralPath $env:GITHUB_ENV -Value "CUDA_PATH=$cudaRoot"
Add-Content -LiteralPath $env:GITHUB_ENV -Value "CUDA_PATH_V12_8=$cudaRoot"
Add-Content -LiteralPath $env:GITHUB_PATH -Value (Join-Path $cudaRoot "bin")
Write-Host "Pinned CUDA 12.8.1 build components installed under $cudaRoot"
