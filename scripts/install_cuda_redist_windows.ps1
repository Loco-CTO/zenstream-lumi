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
        Url = "https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-12.8.4.1-archive.zip"
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

# Extracting the NVIDIA Visual Studio integration archive does not register its
# MSBuild customization files with Visual Studio. The Visual Studio CMake
# generator needs those files in BuildCustomizations to enable the CUDA toolset.
$cudaProps = Get-ChildItem -LiteralPath $cudaRoot -Filter "CUDA 12.8.props" -File -Recurse |
    Select-Object -First 1
if (-not $cudaProps) {
    throw "Pinned CUDA Visual Studio integration did not provide CUDA 12.8.props"
}

$integrationPath = $cudaProps.DirectoryName
foreach ($fileName in @("CUDA 12.8.props", "CUDA 12.8.targets")) {
    if (-not (Test-Path -LiteralPath (Join-Path $integrationPath $fileName) -PathType Leaf)) {
        throw "Pinned CUDA Visual Studio integration is missing $fileName"
    }
}

$vsInstallPath = $null
$vsRoots = @()
if (-not [string]::IsNullOrWhiteSpace($env:ProgramFiles)) {
    $vsRoots += Join-Path $env:ProgramFiles "Microsoft Visual Studio\2022"
}
if (-not [string]::IsNullOrWhiteSpace(${env:ProgramFiles(x86)})) {
    $vsRoots += Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\2022"
}

foreach ($vsRoot in $vsRoots) {
    if (-not (Test-Path -LiteralPath $vsRoot -PathType Container)) {
        continue
    }
    foreach ($candidate in Get-ChildItem -LiteralPath $vsRoot -Directory) {
        $candidateMsvcPath = Join-Path $candidate.FullName "VC\Tools\MSVC"
        if (Test-Path -LiteralPath $candidateMsvcPath -PathType Container) {
            $vsInstallPath = $candidate.FullName
            break
        }
    }
    if ($vsInstallPath) {
        break
    }
}

if (-not $vsInstallPath) {
    $vswherePath = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $vswherePath -PathType Leaf) {
        $vsInstallPath = (& $vswherePath -latest -products "*" `
            -property installationPath | Select-Object -First 1)
    }
}
if ([string]::IsNullOrWhiteSpace($vsInstallPath)) {
    throw "Could not locate the installed Visual Studio 2022 C++ toolchain"
}

$msvcToolchainPath = Join-Path $vsInstallPath "VC\Tools\MSVC"
if (-not (Test-Path -LiteralPath $msvcToolchainPath -PathType Container)) {
    throw "Visual Studio does not contain the MSVC toolchain: $msvcToolchainPath"
}
Write-Host "Using Visual Studio 2022 at $vsInstallPath"

$buildCustomizationsPath = Join-Path $vsInstallPath "MSBuild\Microsoft\VC\v170\BuildCustomizations"
New-Item -ItemType Directory -Force -Path $buildCustomizationsPath | Out-Null
Copy-Item -Path (Join-Path $integrationPath "*") -Destination $buildCustomizationsPath -Force

$cudaGeneratorToolset = "cuda=$cudaRoot"
Add-Content -LiteralPath $env:GITHUB_ENV -Value "CMAKE_GENERATOR_TOOLSET=$cudaGeneratorToolset"

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
# Robocopy returns nonzero codes for successful copies; validations above decide success.
exit 0
