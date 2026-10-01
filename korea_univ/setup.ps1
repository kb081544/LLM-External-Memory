# ReasoningBank 작업 환경 세팅
# 사용법:  D:\reasoning-bank\setup.ps1
# (실행 정책 오류 시)  powershell -ExecutionPolicy Bypass -File D:\reasoning-bank\setup.ps1

$ErrorActionPreference = "Continue"

Write-Host "`n=== 1. 가상환경 활성화 ===" -ForegroundColor Cyan
if ($env:CONDA_DEFAULT_ENV) { conda deactivate }
& "D:\reasoning-bank\.venv\Scripts\Activate.ps1"
Set-Location "D:\reasoning-bank\WebArena"

$py = python -c "import sys; print(sys.executable)"
Write-Host "  python: $py"
if ($py -notlike "*\.venv\*") {
    Write-Host "  [경고] .venv가 아닙니다. conda deactivate 후 다시 시도하세요." -ForegroundColor Red
}

Write-Host "`n=== 2. 환경변수 ===" -ForegroundColor Cyan
$env:GOOGLE_API_KEY    = [Environment]::GetEnvironmentVariable("GOOGLE_API_KEY", "User")
$env:PYTHONUTF8        = "1"
# WebArena 채점기(fuzzy_match)가 OpenAI를 부른다. 없으면 태스크가 1스텝에 죽음.
$env:OPENAI_API_KEY    = "ollama"
$env:OPENAI_BASE_URL   = "http://localhost:11434/v1"
$env:WA_SHOPPING       = "http://localhost:7770"
$env:WA_SHOPPING_ADMIN = "http://localhost:7780/admin"
$env:WA_REDDIT         = "http://localhost:9999"
$env:WA_GITLAB         = "http://localhost:8023"
$env:WA_WIKIPEDIA      = "http://localhost:8888"
$env:WA_MAP            = "http://localhost:3000"
$env:WA_HOMEPAGE       = "http://localhost:4399"
# Vertex AI 경로로 새지 않도록 반드시 제거
Remove-Item Env:\GOOGLE_GENAI_USE_VERTEXAI -ErrorAction SilentlyContinue

if ([string]::IsNullOrWhiteSpace($env:GOOGLE_API_KEY)) {
    Write-Host "  [경고] GOOGLE_API_KEY 없음. 아래로 등록하세요:" -ForegroundColor Red
    Write-Host '  [Environment]::SetEnvironmentVariable("GOOGLE_API_KEY","키","User")'
} else {
    $head = $env:GOOGLE_API_KEY.Substring(0, [Math]::Min(6, $env:GOOGLE_API_KEY.Length))
    Write-Host "  GOOGLE_API_KEY: $head... (길이 $($env:GOOGLE_API_KEY.Length))"
}

Write-Host "`n=== 3. Ollama (WebArena 채점기용) ===" -ForegroundColor Cyan
$ollamaOk = $false
try {
    $tags = Invoke-RestMethod "http://localhost:11434/api/tags" -TimeoutSec 5
    $ollamaOk = $true
    $names = ($tags.models | ForEach-Object { $_.name }) -join ", "
    Write-Host "  Ollama 응답 OK — 모델: $names" -ForegroundColor Green
} catch {
    Write-Host "  Ollama 응답 없음 → 시작 시도" -ForegroundColor Yellow
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 5
    try {
        Invoke-RestMethod "http://localhost:11434/api/tags" -TimeoutSec 5 | Out-Null
        $ollamaOk = $true
        Write-Host "  Ollama 시작됨" -ForegroundColor Green
    } catch {
        Write-Host "  [경고] Ollama를 띄우지 못했습니다." -ForegroundColor Red
        Write-Host "  이 상태로 돌리면 일부 태스크가 1스텝 만에 중단됩니다." -ForegroundColor Red
        Write-Host "  (WebArena 채점기가 OPENAI_API_KEY를 요구 → env.step() 전체가 예외로 종료)"
    }
}

Write-Host "`n=== 4. Docker 컨테이너 ===" -ForegroundColor Cyan
$running = (docker ps --format "{{.Names}}") -split "`n" | Where-Object { $_ }
foreach ($name in @("shopping", "shopping_admin")) {
    if ($running -contains $name) {
        Write-Host "  $name : Up" -ForegroundColor Green
    } else {
        Write-Host "  $name : 정지됨 → 시작 시도" -ForegroundColor Yellow
        docker start $name 2>&1 | Out-Null
    }
}

Write-Host "`n  응답 확인 (최대 60초 대기)..."
foreach ($p in @(7770, 7780)) {
    $ok = $false
    for ($i = 0; $i -lt 12; $i++) {
        try {
            $r = Invoke-WebRequest "http://localhost:$p" -UseBasicParsing -TimeoutSec 5
            if ($r.StatusCode -eq 200) { $ok = $true; break }
        } catch { Start-Sleep -Seconds 5 }
    }
    if ($ok) { Write-Host "  localhost:$p : 200 OK" -ForegroundColor Green }
    else     { Write-Host "  localhost:$p : 응답 없음" -ForegroundColor Red }
}

Write-Host "`n=== 5. 진행 상황 ===" -ForegroundColor Cyan
$dirs = Get-ChildItem "D:\results\" -Directory -Filter "webarena.*" -ErrorAction SilentlyContinue
if ($dirs) {
    $ids = $dirs | ForEach-Object { [int]($_.Name -replace 'webarena\.', '') } | Sort-Object
    $last = $ids | Select-Object -Last 1
    Write-Host "  완료 태스크: $($dirs.Count)개"
    Write-Host "  마지막 번호: $last"
    Write-Host "  → 이어서 하려면 --prev_id $($last - 1)" -ForegroundColor Yellow
} else {
    Write-Host "  결과 없음 (처음 시작)"
}

$memf = "memories_reasoningbank\shopping.jsonl"
if (Test-Path $memf) {
    $n = (Get-Content $memf | Measure-Object -Line).Lines
    Write-Host "  축적 메모리: $n 건"
} else {
    Write-Host "  메모리 파일 없음"
}

Write-Host "`n=== 준비 완료 ===" -ForegroundColor Cyan
Write-Host @"

실행 예시:
  python pipeline_memory.py --website "shopping" --output_dir "D:\results" ``
    --model "gemini-3.5-flash-lite" --prev_id <번호> ``
    2>&1 | Tee-Object -FilePath D:\log_shopping.txt -Append

리포트:
  python make_report.py --results_dir "D:\results" ``
    --memory_file "memories_reasoningbank\shopping.jsonl" ``
    --log_file "D:\log_shopping.txt" --out "D:\report\index.html" --max_cases 16

"@