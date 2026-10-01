# ReasoningBank 작업 재개 체크리스트

새 PowerShell 터미널을 열었을 때 순서대로 실행하면 됩니다.
`D:\reasoning-bank\setup.ps1` 로 저장해두면 나중에 한 줄로 불러올 수 있습니다.

---

## 0. 한 번만 해두면 되는 것 (영구 등록)

이미 하셨다면 건너뛰세요. 새 PC이거나 환경변수가 날아갔을 때만 필요합니다.

```powershell
[Environment]::SetEnvironmentVariable("GOOGLE_API_KEY", "여기에_AI_Studio_키", "User")
[Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "User")

# WebArena 내부 채점기(fuzzy_match)가 OpenAI를 호출한다. 키가 없으면 태스크가 통째로 중단됨.
# Ollama의 OpenAI 호환 엔드포인트로 대체한다.
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "ollama", "User")
[Environment]::SetEnvironmentVariable("OPENAI_BASE_URL", "http://localhost:11434/v1", "User")
[Environment]::SetEnvironmentVariable("WA_SHOPPING", "http://localhost:7770", "User")
[Environment]::SetEnvironmentVariable("WA_SHOPPING_ADMIN", "http://localhost:7780/admin", "User")
[Environment]::SetEnvironmentVariable("WA_REDDIT", "http://localhost:9999", "User")
[Environment]::SetEnvironmentVariable("WA_GITLAB", "http://localhost:8023", "User")
[Environment]::SetEnvironmentVariable("WA_WIKIPEDIA", "http://localhost:8888", "User")
[Environment]::SetEnvironmentVariable("WA_MAP", "http://localhost:3000", "User")
[Environment]::SetEnvironmentVariable("WA_HOMEPAGE", "http://localhost:4399", "User")
```

등록 후 **새 터미널을 열어야** 반영됩니다.

---

## 1. 매번 하는 세팅 (복사해서 붙여넣기)

```powershell
# 가상환경 활성화 (conda base 빠져나온 뒤)
conda deactivate
D:\reasoning-bank\.venv\Scripts\Activate.ps1
cd D:\reasoning-bank\WebArena

# 환경변수 (영구 등록했어도 세션에 확실히 밀어넣기)
$env:GOOGLE_API_KEY      = [Environment]::GetEnvironmentVariable("GOOGLE_API_KEY","User")
$env:PYTHONUTF8          = "1"
$env:OPENAI_API_KEY      = "ollama"
$env:OPENAI_BASE_URL     = "http://localhost:11434/v1"
$env:WA_SHOPPING         = "http://localhost:7770"
$env:WA_SHOPPING_ADMIN   = "http://localhost:7780/admin"
$env:WA_REDDIT           = "http://localhost:9999"
$env:WA_GITLAB           = "http://localhost:8023"
$env:WA_WIKIPEDIA        = "http://localhost:8888"
$env:WA_MAP              = "http://localhost:3000"
$env:WA_HOMEPAGE         = "http://localhost:4399"
Remove-Item Env:\GOOGLE_GENAI_USE_VERTEXAI -ErrorAction SilentlyContinue
```

**주의 1:** `GOOGLE_GENAI_USE_VERTEXAI`는 절대 설정하면 안 됩니다.
켜지면 Vertex AI를 찾다가 인증 오류가 납니다.

**주의 2:** `OPENAI_API_KEY`가 없으면 **태스크가 1스텝 만에 죽습니다.**
WebArena 채점기가 일부 태스크에서 `llm_fuzzy_match` / `llm_ua_match`를 호출하는데,
이 예외가 `env.step()` 전체를 무너뜨립니다. 실제로 이것 때문에 106건 중 29건(27%)이
모델 성능과 무관하게 소실됐습니다. Ollama가 떠 있는지도 함께 확인하세요.

---

## 2. 상태 점검

```powershell
python -c "import sys; print(sys.executable)"     # .venv 경로여야 함
$env:GOOGLE_API_KEY.Substring(0,6)                 # 키 앞자리 확인
docker ps                                          # shopping, shopping_admin 둘 다 Up
curl.exe http://localhost:11434/api/tags           # Ollama 응답 확인
```

**Ollama가 죽어 있으면** 시작 메뉴에서 Ollama를 실행하거나:
```powershell
Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
```
(`bind: Only one usage...` 오류는 이미 떠 있다는 뜻이니 정상입니다.)

모델이 없으면:
```powershell
ollama pull qwen3:8b
```

**Docker가 꺼져 있으면:**
```powershell
docker start shopping shopping_admin
Start-Sleep -Seconds 60
```

**컨테이너 자체가 없으면** (재생성 후 base URL 재설정 필요):
```powershell
docker run --name shopping -p 7770:80 -d shopping_final_0712
docker run --name shopping_admin -p 7780:80 -d shopping_admin_final_0719
Start-Sleep -Seconds 60

docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="http://localhost:7770"
docker exec shopping mysql -u magentouser -pMyPassword magentodb -e "UPDATE core_config_data SET value='http://localhost:7770/' WHERE path = 'web/secure/base_url';"
docker exec shopping /var/www/magento2/bin/magento cache:flush

docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url="http://localhost:7780"
docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e "INSERT INTO core_config_data (scope, scope_id, path, value) VALUES ('default', 0, 'web/secure/base_url', 'http://localhost:7780/') ON DUPLICATE KEY UPDATE value='http://localhost:7780/';"
docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_is_forced 0
docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_lifetime 0
docker exec shopping_admin /var/www/magento2/bin/magento cache:flush
```

브라우저 확인: `http://localhost:7770` / `http://localhost:7780/admin`
(관리자 로그인: **admin / admin1234**, 잠기면
`docker exec shopping_admin php /var/www/magento2/bin/magento admin:user:unlock admin`)

---

## 3. API 키 동작 확인

```powershell
cd D:\reasoning-bank
python test_embedding.py
cd D:\reasoning-bank\WebArena
```

`OK`가 나오면 정상. 401이면 키 문제, 429면 쿼터 소진입니다.

---

## 4. 어디까지 진행됐는지 확인

```powershell
# 마지막으로 끝난 태스크 번호
Get-ChildItem D:\results\ -Directory -Filter "webarena.*" |
  ForEach-Object { [int]($_.Name -replace 'webarena\.','') } |
  Sort-Object | Select-Object -Last 1

# 완료 개수 / 메모리 축적량
Get-ChildItem D:\results\ -Directory -Filter "webarena.*" | Measure-Object
Get-Content memories_reasoningbank\shopping.jsonl | Measure-Object -Line
```

---

## 5. 이어서 실행

위에서 나온 마지막 번호에서 **1을 뺀 값**을 `--prev_id`에 넣습니다.
(마지막 태스크가 쿼터 소진으로 중간에 끊겼을 수 있으므로)

```powershell
python pipeline_memory.py --website "shopping" --output_dir "D:\results" `
  --model "gemini-3.5-flash-lite" --prev_id 351 `
  2>&1 | Tee-Object -FilePath D:\log_shopping.txt -Append
```

**절대 `memories_reasoningbank\*.jsonl`을 지우지 마세요.** 지금까지 쌓인 메모리가 이어집니다.

shopping이 끝나면 admin으로:
```powershell
python pipeline_memory.py --website "shopping_admin" --output_dir "D:\results_admin" `
  --model "gemini-3.5-flash-lite" `
  2>&1 | Tee-Object -FilePath D:\log_admin.txt -Append
```

---

## 6. 분석 · 리포트

```powershell
python analyze_results.py --results_dir "D:\results" `
  --memory_file "memories_reasoningbank\shopping.jsonl" --log_file "D:\log_shopping.txt"

python make_report.py --results_dir "D:\results" `
  --memory_file "memories_reasoningbank\shopping.jsonl" `
  --log_file "D:\log_shopping.txt" --out "D:\report\index.html" --max_cases 16

Invoke-Item D:\report\index.html
```

---

## 자주 만나는 문제

| 증상 | 원인 | 해결 |
|---|---|---|
| `ImportError: cannot import name 'genai'` | conda base 파이썬으로 실행됨 | 1번 활성화 다시 |
| `Environment variable WA_XXX missing` | 환경변수 미설정 | 1번 블록 붙여넣기 |
| `429 RESOURCE_EXHAUSTED` + `quotaValue: 500` | 일일 한도 소진 | 한국시간 오후 4~5시 이후 재개, 또는 다른 계정 키로 교체 |
| `OPENAI_API_KEY environment variable must be set` | WebArena 채점기가 OpenAI 호출 | Ollama 기동 + `OPENAI_API_KEY`/`OPENAI_BASE_URL` 설정. **태스크가 1스텝에 죽는 주범** |
| 태스크가 계속 1스텝에서 끝남 | 위와 동일 | 같은 조치 후 해당 태스크 재실행 |
| `ERR_CONNECTION_REFUSED` | Magento 컨테이너 다운 | `docker restart shopping` |
| `401 UNAUTHENTICATED` | 키 문제 | 다른 Google 계정으로 키 재발급 |
| `404 no longer available` | 모델 단종 | AI Studio에서 현재 모델명 확인 후 `choices`·`CLIENT_DICT` 수정 |
| PowerShell에 빨간 `NativeCommandError` | `2>&1`로 stderr 합쳐서 생기는 표시 | 정상, 무시 |

---

## 쿼터 계산

- 무료 티어: **모델별·프로젝트별 하루 500 요청**
- 태스크당 약 12회 소모 (에이전트 8 + 평가 1 + 메모리추출 1 + 임베딩 2)
- **하루 약 40개** 처리 가능 → shopping 187개는 약 5일
- 계정을 바꾸면 쿼터가 새로 생김 (같은 계정의 다른 키는 소용없음)
- 모델을 바꿔도 쿼터가 별개지만, 결과 해석이 섞이므로 권장하지 않음

---

## 브라우저 창 보기 / 숨기기

`pipeline_memory.py`의 `run_cmd` 리스트에서:

```python
"--headless", "False",   # 있으면 창이 뜸, 지우면 headless
"--slow_mo", "1000",     # 액션 사이 대기 (ms). 클수록 천천히
```