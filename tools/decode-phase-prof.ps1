# Run the agent-turn client with SPC_DECODE_PROF=1 and print the per-ubatch phase breakdown,
# which is where the ~457 ms of non-kernel time per turn lives (RESEARCH 24.4).
$ROOT  = "<REPO>"
$EXE   = Join-Path $ROOT "work\picks-llama\build-picks\bin\llama-server.exe"
$MODEL = Join-Path $ROOT "work\bonsai-demo\models\bonsai2-gguf\27B\Ternary-Bonsai-2-27B-PTQ1_0-f16ssm-MTP.gguf"
$CLI   = Join-Path $ROOT "work\tmp\prefill-turn-client.py"
$PORT  = 8244
$ERR   = Join-Path $ROOT "work\tmp\decode-phase-prof.err.log"

Get-NetTCPConnection -LocalPort $PORT -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
Stop-Process -Name llama-server -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 6

$env:SPC_DECODE_PROF = "1"
$env:GGML_CUDA_GRAPH_SHAPECACHE = "1"
$a = @(
    "-m", $MODEL, "-ngl", "99", "-c", "131072",
    "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--flash-attn", "on",
    "-b", "8192", "-ub", "1024", "--host", "127.0.0.1", "--port", "$PORT",
    "--jinja", "-np", "1", "--kv-offload", "--no-warmup",
    "--spec-type", "draft-mtp", "--spec-draft-n-max", "2", "-ctkd", "q8_0", "-ctvd", "q8_0"
)
Start-Process -FilePath $EXE -ArgumentList $a -RedirectStandardOutput ($ERR + ".out") -RedirectStandardError $ERR -WindowStyle Hidden | Out-Null
for ($i = 0; $i -lt 200; $i++) {
    Start-Sleep -Seconds 2
    try { if ((Invoke-WebRequest -Uri "http://127.0.0.1:$PORT/health" -TimeoutSec 3 -UseBasicParsing).StatusCode -eq 200) { break } } catch { }
}
python $CLI $PORT 2>&1 | Select-Object -Last 4
Get-NetTCPConnection -LocalPort $PORT -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2
python (Join-Path $ROOT "work\tmp\grep-lines.py") $ERR "dec-prof"
