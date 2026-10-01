# LLM별 WebArena 50+50 태스크 비교

`run_model_comparison.py`는 `shopping`과 `shopping_admin`에서 각각 50개를 무작위로 뽑아
동일한 태스크 집합으로 여러 LLM을 비교한다. 기본 비교 대상은 Lite 계열인
`gemini-3.1-flash-lite`와 `gemini-3.5-flash-lite`다. 기본 actor는
`gemini-3.5-flash-lite`이며 `--model`로 둘 중 하나를 선택한다.

`gemini-2.5-flash-lite`는 일부 기존 프로젝트에는 남아 있지만 신규 사용자 키에서는
`generateContent`가 404를 반환할 수 있다. `gemini-3.6-flash-lite`라는 모델명 대신
공식 Lite 모델명은 `gemini-3.1-flash-lite` 또는 `gemini-3.5-flash-lite`다.

## 1. 환경 설정

PowerShell에서 다음 환경 변수를 준비한다.

```powershell
cd D:\reasoning-bank\WebArena
$env:GOOGLE_API_KEY = "AI Studio에서 발급한 키"
Remove-Item Env:\GOOGLE_GENAI_USE_VERTEXAI -ErrorAction SilentlyContinue

$env:WA_SHOPPING = "http://localhost:7770"
$env:WA_SHOPPING_ADMIN = "http://localhost:7780/admin"
$env:WA_HOMEPAGE = "http://localhost:4399"
```

일부 WebArena GT evaluator는 actor와 별개로 OpenAI 호환 API를 호출한다. 비용 없이
실행하려면 로컬 Ollama 같은 OpenAI 호환 endpoint를 설정할 수 있다.

```powershell
$env:OPENAI_API_KEY = "ollama"
$env:OPENAI_BASE_URL = "http://localhost:11434/v1"
```

## 2. 태스크 확인 후 실행

API를 호출하지 않고 선택된 태스크만 확인한다.

```powershell
python run_model_comparison.py --dry-run
```

Gemini로 50+50개를 실행한다.

```powershell
python run_model_comparison.py --model gemini-3.5-flash-lite
```

기본 실행은 브라우저 창을 표시하고 각 행동 사이에 500ms 지연을 둔다. 창을 숨기려면
`--headless`, 더 느리게 보려면 예를 들어 `--slow-mo 1000`을 사용한다. 영상도
남기려면 `--record-video`를 추가한다. 영상은 용량이 크므로 기본값은 꺼져 있다.

무료 일일 쿼터에 맞춰 나누어 실행하려면 다음처럼 사용한다. 같은 명령을 다시 실행하면
`summary_info.json`이 있는 완료 태스크는 자동으로 건너뛴다.

```powershell
python run_model_comparison.py --model gemini-3.5-flash-lite --max-tasks-this-run 10
```

## 3. Gemini 3.1 Flash-Lite와 3.5 Flash-Lite 비교

같은 `--output-root`, `--seed`, `--limit-per-site`를 사용하면 최초에 저장된 manifest를
재사용하므로 모델마다 태스크가 바뀌지 않는다.

```powershell
python run_model_comparison.py --model gemini-3.1-flash-lite

# shopping 및 shopping_admin 서버를 최초 snapshot으로 복구한 뒤 실행
python run_model_comparison.py --model gemini-3.5-flash-lite
```

실행기는 브라우저를 시작하기 전에 작은 `generateContent` 요청을 한 번 보내 모델 접근
가능 여부를 검사한다. 접근 불가 모델이면 어떤 태스크도 시작하지 않고 종료한다.

모델을 바꿔 재실행하기 전에 shopping 및 shopping_admin 서버를 같은 초기 snapshot으로
복구해야 한다. 수정형 태스크가 서버 상태를 바꾸므로 초기화하지 않으면 모델 간 비교가
공정하지 않다.

기본 비교표에는 두 모델이 항상 나타난다. 아직 실행하지 않은 모델은 `pending`으로
표시된다. OpenAI나 Claude 등 다른 모델을 추가하려면 비교 대상도 함께 지정한다.

```powershell
python run_model_comparison.py --model openai/gpt-4.1-mini `
  --comparison-models gemini-3.1-flash-lite gemini-3.5-flash-lite openai/gpt-4.1-mini
```

이 저장소에서 `openai/*`는 `OPENAI_API_KEY`를, `claude*`는 Vertex AI의
`GOOGLE_CLOUD_PROJECT`와 `GOOGLE_CLOUD_LOCATION`을 사용한다.

## 4. 결과

기본 출력 위치는 다음과 같다.

```text
results_model_comparison/
  manifests/                         # 모든 모델이 공유하는 50+50 task IDs
  models/gemini-3.5-flash-lite/
    shopping_shopping_admin_50_each_seed_42/
      shopping/webarena.<id>/
      shopping_admin/webarena.<id>/
      summary.json                   # 코드에서 읽기 좋은 모델별 집계
      summary.md                     # 모델별 표 형태 요약
      task_results.csv               # 태스크별 reward/status/error
      review.html                    # 전체 태스크 스크린샷/상태 검토 화면
      run_state.json                 # 중단/오류 후 재개 상태
  comparison_shopping_shopping_admin_50_each_seed_42.csv
  comparison_shopping_shopping_admin_50_each_seed_42.md  # 전체 모델 비교표
```

집계만 다시 만들 때는 API 호출 없이 다음 명령을 쓴다.

```powershell
python run_model_comparison.py --model gemini-3.5-flash-lite --report-only
```

성공은 기본적으로 `cum_reward >= 1`이다. `success`와 `fail`만 성공률의 분모에 들어가며,
API 오류나 실행 중단으로 reward가 없는 태스크는 `error`로 따로 센다.

각 `webarena.<id>/` 폴더에는 다음 검토 파일도 저장된다.

- `screenshot_step_<N>.png`: BrowserGym이 저장한 매 step 화면
- `trajectory.json`: 압축 pickle에서 추출한 행동과 저장된 reasoning
- `task_report.html`: intent, success/fail/error, 행동, 스크린샷을 시간순으로 표시
- `video.webm` 계열: `--record-video`를 사용했을 때만 저장

현재 `run_model_comparison.py`는 모델 자체를 공정하게 비교하는 actor-only 실행기다.
ReasoningBank 메모리 검색과 memory induction을 실행하지 않으므로 메모리 항목은 생성되지
않으며, HTML에도 `Memory: disabled`로 명시된다.
