param(
    [int]$Port = 11435,
    [string]$RuntimeRoot = 'E:\LET_ME_COOK\Medical_Generative_AI_v2'
)

$ErrorActionPreference = 'Stop'
$server = Join-Path $RuntimeRoot 'local_runtime\bin\llama-server.exe'
$model = Join-Path $RuntimeRoot 'local_runtime\Qwen3-8B-Q4_K_M.gguf'
$contextSize = 16384
$alias = 'Qwen/Qwen3-8B-GGUF:Q4_K_M'

foreach ($path in @($server, $model)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing Qwen runtime file: $path"
    }
}

try {
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/models" -TimeoutSec 2
    $running = @($response.data) | Where-Object { $_.id -eq $alias } | Select-Object -First 1
    if ($null -eq $running) {
        throw "Port $Port already serves a different model. Stop that server first."
    }
    if ([int]$running.meta.n_ctx -ne $contextSize) {
        throw "Qwen is already running with context $($running.meta.n_ctx). Stop it and rerun this script for 16384."
    }
    Write-Host "Qwen 8B is already running on port $Port with context $contextSize."
    return
} catch [System.Net.WebException] {
    # No server is listening; start one below.
} catch [System.Net.Http.HttpRequestException] {
    # No server is listening; start one below.
}

& $server -m $model --alias $alias `
    --host 127.0.0.1 --port $Port --ctx-size $contextSize `
    --gpu-layers 99 --parallel 1 --reasoning off --jinja
