param(
    [int]$ContextSize = 8192,
    [int]$Port = 11435
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$server = Join-Path $projectRoot 'local_runtime/bin/llama-server.exe'
$model = Join-Path $projectRoot 'local_runtime/Qwen3-8B-Q4_K_M.gguf'

if (-not (Test-Path -LiteralPath $server)) {
    throw "llama-server.exe missing: $server"
}
if (-not (Test-Path -LiteralPath $model)) {
    throw "Qwen GGUF missing: $model"
}

try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
    if ($health.status -eq 'ok') {
        Write-Host "Local Qwen server is already running on port $Port."
        return
    }
} catch {
    # No healthy server is listening yet.
}

& $server -m $model --alias 'Qwen/Qwen3-8B-GGUF:Q4_K_M' `
    --host 127.0.0.1 --port $Port --ctx-size $ContextSize `
    --gpu-layers 99 --parallel 1 --reasoning off --jinja
