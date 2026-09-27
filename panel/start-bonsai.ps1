# 一键启用 Bonsai 2 27B：启动控制台（后台）→ 按已保存参数加载模型 → 打开浏览器
# 用法：powershell -ExecutionPolicy Bypass -File start-bonsai.ps1  [-NoBrowser]

[CmdletBinding()]
param(
    [int]$Port = 8090,
    [int]$ServerPort = 8080,
    [switch]$NoBrowser
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$ErrorActionPreference = "Stop"

$here      = Split-Path -Parent $MyInvocation.MyCommand.Path
$dashboard = Join-Path $here "bonsai-dashboard.py"
$panelUrl  = "http://127.0.0.1:$Port"
$logPath   = Join-Path (Split-Path -Parent $here) "work\bonsai-start.log"

function Say([string]$Message, [string]$Color = "Gray") {
    Write-Host $Message -ForegroundColor $Color
    try { Add-Content -LiteralPath $logPath -Value ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message) } catch { }
}

$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python) { $python = (Get-Command py -ErrorAction SilentlyContinue).Source }
if (-not $python) {
    Write-Host "找不到 Python（需要 3.10+）" -ForegroundColor Red
    exit 1
}

# ---------- 1) 控制台 ----------
$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if (-not $listener) {
    Start-Process -FilePath $python `
        -ArgumentList "`"$dashboard`"", "--port", "$Port", "--server-port", "$ServerPort" `
        -WindowStyle Hidden
    for ($i = 0; $i -lt 24; $i++) {
        Start-Sleep -Milliseconds 500
        try { $null = Invoke-RestMethod "$panelUrl/api/status" -TimeoutSec 2; break } catch { }
    }
    Say "[面板] 已启动 $panelUrl" "Green"
} else {
    Say "[面板] 已在运行 $panelUrl" "DarkGray"
}

# ---------- 2) 按保存的参数加载模型 ----------
try {
    $body = @{ action = "start" } | ConvertTo-Json
    $r = Invoke-RestMethod "$panelUrl/api/loadconfig" -Method Post `
        -ContentType "application/json; charset=utf-8" `
        -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 900
    if ($r.ok) { Say "[模型] $($r.message)" "Green" }
    else {
        Say "[模型] 加载失败：$($r.message)" "Red"
        $r.log_tail | Select-Object -Last 6 | ForEach-Object { Say "        $_" "DarkGray" }
    }
} catch {
    Say "[模型] 调用失败：$($_.Exception.Message)" "Yellow"
}

# ---------- 3) 状态 ----------
try {
    $s = Invoke-RestMethod "$panelUrl/api/status" -TimeoutSec 10
    Say ("[状态] 在线={0}  上下文={1}  每槽={2}  FA={3}  KV={4}" -f `
        $s.runtime.online, $s.config.ctx, $s.derived.ctx_per_slot, $s.derived.flash, $s.derived.kv_label) "Green"
    if ($s.gpu) {
        Say ("[显卡] {0}  利用率 {1}%  温度 {2}°C  功耗 {3}W  显存 {4:N2}/{5:N1} GB" -f `
            $s.gpu.name, $s.gpu.util, $s.gpu.temp_c, [math]::Round($s.gpu.power_w), `
            ($s.gpu.mem_used_mb / 1024), ($s.gpu.mem_total_mb / 1024)) "Green"
    }
} catch { }

if (-not $NoBrowser) { Start-Process $panelUrl }

