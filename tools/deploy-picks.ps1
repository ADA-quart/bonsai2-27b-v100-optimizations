# Same as deploy-all.ps1 but sourced from the oripoin cherry-pick worktree
# (work\picks-llama\build-picks\bin) instead of the prod tree's build-v2.
$ErrorActionPreference = "Stop"
$ROOT  = "<REPO>"
$SRC   = Join-Path $ROOT "work\picks-llama\build-picks\bin"
$DST   = Join-Path $ROOT "work\bonsai-demo\bin\cuda"
$STAMP = Get-Date -Format "yyyyMMdd-HHmmss"
$BAK   = Join-Path $ROOT "work\bin-backup\$STAMP-full"
New-Item -ItemType Directory -Force -Path $BAK | Out-Null

$files = @(
    "llama-server.exe", "llama-server-impl.dll",
    "llama-cli.exe", "llama-cli-impl.dll",
    "llama-bench.exe", "llama-bench-impl.dll",
    "llama-perplexity.exe", "llama-perplexity-impl.dll",
    "llama-quantize.exe", "llama-quantize-impl.dll",
    "llama.dll", "llama-common.dll",
    "ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll",
    "mtmd.dll"
)

$mismatch = @()
foreach ($f in $files) {
    $s = Join-Path $SRC $f
    $d = Join-Path $DST $f
    if (-not (Test-Path $s)) { Write-Warning "skip (not in build): $f"; continue }
    if (Test-Path $d) { Copy-Item -LiteralPath $d -Destination (Join-Path $BAK $f) -Force }
    Copy-Item -LiteralPath $s -Destination $d -Force
    $hs = (Get-FileHash $s -Algorithm SHA256).Hash
    $hd = (Get-FileHash $d -Algorithm SHA256).Hash
    if ($hs -ne $hd) { $mismatch += $f }
    Write-Host ("deployed {0,-26} {1,10:N0} bytes" -f $f, (Get-Item $d).Length)
}

if ($mismatch.Count -gt 0) {
    throw ("hash mismatch after copy: " + ($mismatch -join ", "))
}
Write-Host ""
Write-Host ("all files verified by SHA256; backup of the previous runtime: " + $BAK)
