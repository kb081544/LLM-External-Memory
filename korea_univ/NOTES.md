# NOTES — EFM 작업 중 판단 기록

`EFM_TASK.md` 1.5절("불확실한 점은 가장 합리적인 선택을 하고 여기에 기록")에 따라 작성.

## 2026-10-01 — `baseline-original` 태그 범위

`git tag baseline-original`을 찍기 전, 작업 디렉터리엔 이미 35개 파일의 미커밋 변경사항이 있었음
(Vertex AI→API 키 전환, watchdog/재시도 하드닝, MaTTS, Mind2Web 벤치마크 전체, 오늘 낮에 만든
`reasoningbank_pruned` 실험 등 — 전부 `google-research/reasoning-bank` 원문 릴리스엔 없던 로컬 추가/수정).

**판단**: 이 상태 그대로를 커밋(`3936d27`)하고 `baseline-original`로 태깅함. "원본"을 논문 공식
릴리스로 되돌리지 않고 **"EFM 작업 시작 직전, 지금 실제로 돌아가고 있던 코드 그대로"**를 기준점으로
삼음 — 사용자가 오늘 하루 종일 검증해온 상태(임베딩 백엔드, 재시도 로직 등)를 EFM 비교의 진짜 출발점으로
보는 게 맞다고 판단. `reasoningbank_pruned`는 EFM과 무관한 별개 실험이라 코드는 같이 보존되지만,
EFM의 `baseline` 비교 대상은 `reasoningbank_pruned`가 아니라 **원래의 `reasoningbank` 모드**임(둘 다
현재 코드에 공존, 서로 분기로 격리됨).

## 2026-10-01 — 백본 LLM 전송 경로 변경 (모델 자체는 아님)

`EFM_TASK.md` 0절/7절은 "백본 모델을 바꾸지 않는다"고 명시. 오늘 Gemini API 무료 티어 쿼터를 하루 종일
소진한 뒤, **API 키 직접 호출 대신 Antigravity CLI(`agy -p`, `claude-sonnet-4-6`)로 전송 경로만 교체**하기로
결정. baseline/efm 두 팔 모두 동일하게 이 경로를 쓰므로 "arm 간 모델 일관성"이라는 공정성 원칙 자체는
유지됨 — 다만 이건 원 논문/이전 실험들이 써온 Gemini 계열과는 다른 모델이라, **이번 EFM 비교 결과는
이전에 쌓아온 Gemini 기반 결과(8월 baseline 데이터, 오늘의 `reasoningbank_pruned` 결과)와 직접 비교할
수 없고, 오직 이번에 새로 도는 no_memory/reasoningbank/efm 세 팔 사이에서만 비교 유효**함.

## 2026-10-01 — 동등성 테스트(5.1) 전용 파라미터: `efm_k_cand=1`, `efm_n_inject=5`

문서 5.1은 `TAU_D=TAU_C=1.01, ETA=0, T_STALE=R_MIN=10^9`만 명시하지만, 이것만으로는 baseline과
정확히 동일한 주입 결과가 나오지 않음. baseline(`run.py`의 `select_memory(n=1, ...)`)은 **가장 가까운
experience 단 하나**를 후보로 삼고, 그 experience에 속한 **모든 memory_items를 개수 제한 없이 전부**
주입함(주입 개수 상한 없음, extraction 단계 자체가 태스크당 최대 3개 items를 생성하므로 사실상
상한은 3). EFM의 기본값(`k_cand=5`)은 후보 experience 풀을 5개로 넓히므로, TAU/ETA/T_STALE만
degenerate하게 맞춰도 후보 집합 자체가 baseline과 달라져 동등성 비교가 성립하지 않음.

**판단**: 동등성 테스트에 한해 추가로 `--efm_k_cand 1`(baseline처럼 최근접 experience 1개만 후보로),
`--efm_n_inject 5`(baseline의 실질 상한인 3개를 여유있게 넘는 값 — 예산 때문에 밀려나는 항목이 없도록)
를 사용. 이 두 값은 동등성 테스트 전용 판단이며, TAU/ETA/T_STALE처럼 실제 벤치마크 실행 시에는
`EFMHParams` 기본값(3.7절 표)으로 되돌아감. `k_cand=1, n_inject=5` 조건에서는 group/conflict 로직도
사실상 무력화됨(후보가 1개 experience의 items뿐이라 서로 충돌/그룹 관계가 나타나도 baseline과 동일하게
전부 주입되므로 결과 비교에는 영향 없음).

## 2026-10-01 — `efm/retrieve.py` 쿼리 임베딩 비대칭성 수정 (동등성 테스트 중 발견)

동등성 테스트를 준비하며 `memory_management.screening()`을 다시 읽다가 발견: baseline은 캐시에
저장하는 과거 쿼리 임베딩은 **plain** `embed_query_with_gemini(query)`이지만, 현재 쿼리를 채점할
때는 `"Given the prior web navigation queries..."` instruction을 덧씌운 **instruction-wrapped**
임베딩을 사용함(비대칭 query-vs-document 임베딩 패턴). `efm/retrieve.py`는 원래 양쪽 다 plain
임베딩을 썼음 — `TAU=1.01` 등으로 후보 집합 로직을 동일하게 맞춰도, 이 임베딩 방식 차이 때문에
유사도 순위 자체가 baseline과 달라져 동등성 테스트가 "알고리즘 버그"가 아닌 "임베딩 방식 불일치"로
실패할 수 있었음.

**조치**: `efm/store.py`에 `embed_query_for_scoring()`을 추가해 baseline과 동일한 instruction-wrap을
재현하고, `efm/retrieve.py`의 experience-cosine(1단계)과 description-cosine(4단계 그룹 collapse) 모두
이 함수로 전환. 이건 테스트 전용 땜질이 아니라 **실제 운영 시에도 적용되는 수정**임 — baseline이 의도적으로
튜닝한 쿼리 표현 방식을 EFM도 동일하게 쓰는 게 더 올바른 선택이라 판단(문서 0절 "동일 임베딩 모델" 제약의
자연스러운 확장). 캐시에 저장되는 문서 쪽 임베딩은 여전히 plain(변경 없음) — baseline의 캐시 포맷과
100% 호환 유지. 수정 후 단위테스트 28/28 유지 확인(순수 함수/mock 경로라 이 변경의 영향을 받지 않음).

## 2026-10-01 — 임베딩은 Gemini 무료 API 그대로, 실행 규모는 187→80으로 축소 (1차)

동등성 테스트 중 임베딩(`gemini-embedding-001`) 쪽에서 분당 100회 무료 쿼터 벽을 실제로 확인함
(하루 단위 하드 캡은 아니고 분당 속도 제한 — `gemini_retry.py`의 지수 백오프로 결국 통과는 하지만
느려짐). Antigravity CLI(`agy`)는 임베딩 API가 없어 이 부분만은 옮길 수 없음(텍스트 생성 쪽은 전부
`agy-claude-sonnet-4-6`).

**사용자 결정**: 유료 전환 없이 무료 쿼터 + 백오프로 버티는 것으로 확정. 다만 187개 태스크 × 3-arm
전체 실행은 무리일 수 있다고 보고, **1차로 80개 태스크 × 3-arm(no_memory/reasoningbank/efm)을 먼저
실행 → 정상적으로 끝까지 돌면 그 다음 187개 전체로 확장**하는 2단계 규모로 축소.

**참고**: 동등성 테스트 스크립트가 유독 느린 건 기존 185개 reasoningbank 항목을 한 번에 EFM 스토어로
일괄 변환하며 임베딩을 연속 호출하기 때문(테스트 전용 백필 스크립트의 특성). 실제 파이프라인
(`pipeline_memory.py`)은 태스크당 임베딩 호출이 1~3회뿐이고 그 사이에 브라우저 에이전트 실행/judge/
추출로 수 분씩 자연스러운 간격이 생기므로, 실제 80태스크×3-arm 실행이 이 정도로 쿼터에 몰릴 가능성은
낮다고 판단 — 다만 실측 전까지는 가정일 뿐, 10-task smoke test에서 실제 속도를 확인할 예정.

**추가 발견**: 동등성 테스트가 실제로 멈춘 원인은 분당 제한이 아니라 **일일 무료 쿼터(1000회/일,
`EmbedContentRequestsPerDayPerUserPerProjectPerModel-FreeTier`)** 소진이었음(160/185 처리 시점에 확정
에러 메시지로 확인). 사용자가 다른 구글 계정으로 새 API 키를 발급해 `GOOGLE_API_KEY`를 교체, 쿼터를
리셋함.

이 과정에서 `efm/insert.py`의 실제 비효율 버그도 함께 발견·수정: 새 아이템의 description/content를
관계 분류용으로 한 번 임베딩(`d_vec`/`c_vec`)해놓고도, 캐시에 저장할 때 `store.embed_and_cache()`를
다시 호출해 **같은 텍스트를 한 번 더 임베딩**하고 있었음 — 아이템 하나당 실제 필요한 2회가 아니라 4회의
임베딩 API 호출을 쓰고 있었던 것. `store.append_cache(vec, ...)`를 새로 추가해 이미 계산된 벡터를
재임베딩 없이 캐시에만 기록하도록 수정(`efm/store.py`, `efm/insert.py`). 이건 테스트만이 아니라 **실제
운영 시 EFM의 임베딩 호출량을 정확히 절반으로 줄이는 실질적 수정**이라 쿼터가 빠듯한 지금 상황에 중요함.
수정 후 단위테스트 28/28 유지 확인.

## 2026-10-01 — 동등성 테스트에서 실제 버그 발견: `efm/retrieve.py`가 모든 검색에서 0개 주입

쿼터 문제 해결 후 동등성 테스트를 끝까지 돌렸더니, EFM 스토어 구축(185개 → 194 아이템, tau=1.01이라
전부 Novel)까지는 정상이었지만 **20개 태스크 전부 `efm_injected=0`**으로 나왔음(baseline은 전부
정상적으로 1개 소스를 찾음). 단위테스트(28/28)는 순수 함수(`collapse_groups`/`pair_conflicts`/
`allocate_budget`)만 커버해서 이 버그를 못 잡았음 — 버그는 `retrieve()`를 감싸는 오케스트레이션
코드 자체에 있었음.

**원인**: 쿼리 임베딩 캐시(`query_embeddings.jsonl`)의 `id` 필드는 순수 task_id(예: `"21"`) —
`run.py`/테스트 스크립트 모두 `embed_and_cache(query, ..., task_id)`로 이렇게 저장함. 반면
`experiences` dict는 어디서나 `f"ex_{tid}"`(예: `"ex_21"`) 키로 저장됨(`pipeline_memory.py`,
테스트 스크립트). `efm/retrieve.py` 2단계가 `experiences.get(ex_id)`로 **prefix 없이** 조회해서
항상 `None`을 받고 `continue` → `item_best_cos`가 영원히 비어서 매번 빈 리스트를 반환하고 있었음.

**영향**: 이 버그가 안 잡혔다면 EFM arm은 실제로 `no_memory`와 동일하게 동작(메모리를 전혀 주입 못함)
했을 것이고, SR 차이가 "EFM이 효과 없다"로 잘못 해석됐을 것 — 5.1 동등성 테스트를 생략하지 않고
먼저 돌린 이유가 정확히 이런 걸 잡기 위함이었음.

**수정**: `efm/retrieve.py`에서 `experiences.get(f"ex_{ex_id}")`로 prefix를 맞춰서 조회하도록 수정.
단위테스트 28/28 유지 확인. 재실행 결과 **20/20 태스크 모두 baseline과 동일한 소스 task_id, 동일한
주입 개수(task 124: 둘 다 2개)로 일치** — 5.1 동등성 테스트 통과.

## 2026-10-01 — 5.3 스모크 테스트에서 실제 `agy` 연동 버그 2건 발견·수정

10태스크×3-arm 스모크 테스트를 처음 돌리자마자 실제 파이프라인(단위테스트/동등성 테스트로는 못 잡는
real subprocess 경로)에서 버그 2개가 바로 터짐:

1. **`count_tokens`가 `agy-claude-sonnet-4-6`를 HuggingFace 모델로 오인**: `agents/legacy/utils/
   llm_utils.py`의 `count_tokens()`는 `gemini`/`claude` 접두사만 분기 처리하고 나머지는
   `AutoTokenizer.from_pretrained(model_name)`로 떨어짐 — `"agy-claude-sonnet-4-6"`은 HF 저장소가
   아니라서 매 스텝(`dynamic_prompting.fit_tokens`)마다 즉시 크래시. **수정**: `model.startswith(
   "agy-")`일 때 `tiktoken`의 범용 `gpt-4` 인코딩으로 근사 처리(토큰 예산 가드용이라 정밀할 필요
   없음, Vertex AI `gcloud auth` 경로는 더 이상 쓰지 않으므로 그쪽으로도 못 감).

2. **`agy` 서브프로세스 호출이 `WinError 2`(파일 없음)로 실패**: 이 세션(Claude Code Bash 도구)의
   인터랙티브 쉘에서 `agy`는 PATH에 없고(`which agy` 실패), 실제로는
   `C:\Users\user\AppData\Local\agy\bin\agy.exe`에 설치돼 있음. `subprocess.run(["agy", ...])`가
   파이프라인(비-인터랙티브) 프로세스에서 PATH로 못 찾아 실패 — judge(`autoeval`) 단계에서 첫 실패가
   드러남. **수정**: `utils/agy_client.py`에 `_resolve_agy_bin()`(먼저 `shutil.which("agy")`
   시도, 안 되면 알려진 설치 경로로 폴백) 추가, `AgyClient`/`ChatAgy` 둘 다 하드코딩된 `"agy"` 대신
   이 절대경로를 사용하도록 수정.

두 버그 모두 수정 후 `AgyClient`/`ChatAgy`/`count_tokens` 각각 직접 호출로 재검증 완료. 실패한
스모크 테스트의 부분 산출물(`results_smoke_*`, `memories_smoke_*`, `autoeval/logs_*_shopping`)은
삭제하고 처음부터 재실행.

**추가로 발견된 버그 2개 (재실행 중)**:

3. **`agy -p <prompt>`가 Windows 명령줄 길이 제한에 걸림 (`WinError 206`)**: judge 단계는 trajectory
   전체(axtree 텍스트 등)를 프롬프트에 포함하는데, 이걸 `-p` argv 값으로 넘기면 Windows의
   `CreateProcess` 명령줄 길이 한도를 넘어 실패. **수정**: `agy`의 `--input-format stream-json`
   모드로 전환 — 프롬프트를 argv가 아니라 stdin으로 NDJSON 한 줄(`{"event":"user","message":
   {"role":"user","content":...}}`)로 전달. `--print=`(빈 값 명시, 안 그러면 다음 flag를 프롬프트로
   먹어버림), `--output-format stream-json`도 같이 필요. 출력은 `init`/`step_update`/`result`
   이벤트 스트림이라 마지막 `"event":"result"` 줄만 파싱해서 쓰면 기존 `{status, response, usage}`
   구조와 동일. `utils/agy_client.py`(`_parse_result_event` 추가)와 `chat_api.py`의 `ChatAgy._call`
   양쪽 다 수정. 65000자 프롬프트로 직접 재현·검증 완료(이전엔 실패, 수정 후 정상 응답).

4. **WebArena 자체 ground-truth 채점기(`llm_ua_match`/`llm_fuzzy_match`)가 `OPENAI_API_KEY` 요구**:
   `--judge` 설정과 무관하게 `task.validate()`가 매 스텝 내부적으로 OpenAI를 직접 호출함(CLAUDE.md에
   이미 문서화된 사항, 오늘 처음 실제로 부딪힘). 확인해보니 **로컬 Ollama가 이미 떠 있고, 이 하드코딩된
   모델 이름(`gpt-4-1106-preview` 등)과 정확히 일치하는 이름으로 모델 태그를 미리 pull해둔 상태**였음
   (`gpt-3.5-turbo`, `gpt-4`, `gpt-4o`, `gpt-4-1106-preview` 전부 `qwen3:8b` 기반) — 예전에 이런
   용도로 준비해둔 것으로 보임. **조치**: `OPENAI_API_KEY`(더미 값, Ollama는 검증 안 함)와
   `OPENAI_BASE_URL=http://localhost:11434/v1`를 스모크 테스트 실행 환경에 설정. 완전 무료, agy/Gemini
   쿼터와 무관. (참고: thinking 모델이라 `max_tokens`가 너무 작으면 reasoning만 쓰고 `content`가 빈
   문자열로 끝날 수 있음 — WebArena 하드코딩값 `max_tokens=768`은 짧은 same/different 분류 작업엔
   충분함을 확인.)

## 2026-10-01 — Claude/GPT 계정 쿼터 완전 소진 + EFM 디렉터리 생성 버그

스모크 테스트 도중 Antigravity 계정 대시보드를 확인해보니 Claude+GPT 모델이 **같은 쿼터 풀**을
공유하고, 5시간 윈도우 쿼터가 9%만 남은 상태였음(주간 쿼터는 66% 남음). Gemini 모델은 별도 풀로
100% 남아있었음. Gemini로 전환을 검토했으나, 실제 추론이 필요한 프롬프트로 테스트해보니 Gemini(low/
medium/high 전부)가 **텍스트로 답하는 대신 도구 호출(RunCommand)을 시도하다 권한 거부당해 빈 응답만
반환** — effort 레벨 문제가 아니라 agy의 에이전트형 틀 안에서 Gemini가 도구를 선호하는 구조적 문제로
판단, Gemini 전환은 보류하고 Claude 유지 + 쿼터 페이싱으로 방향을 잡음. 이후 실제로 5시간 쿼터가
완전히 소진됨("Individual quota reached... Resets in 3h59m46s") — no_memory arm은 마지막 태스크
직전까지 끝났고(10/10 완료, exit=0), reasoningbank/efm arm은 첫 호출부터 막힘.

이 와중에 **새 버그 발견**: `run.py`의 `--efm_dir` 분기가 `ensure_file(args.memory_path)`만 하고
`args.efm_dir` 자체(예: `memories_smoke_efm/shopping/`) 디렉터리는 안 만들어서, 그 안에 저장되는
`query_embeddings.jsonl` 등을 처음 쓰는 시점(`store.load_cache` → `memory_management.
load_cached_embeddings`의 "파일 없으면 빈 파일 생성" 폴백)에 `FileNotFoundError`로 죽음 — 지금까지
동등성 테스트는 스크립트에서 수동으로 `os.makedirs(OUT_DIR, ...)`를 해놔서 안 드러났던, **fresh
efm_dir로 시작하는 모든 실행(80개 본 실행 포함)에 영향을 미치는 버그**였음. **수정**: `run.py`의
`if args.efm_dir:` 분기 맨 앞에 `os.makedirs(args.efm_dir, exist_ok=True)` 추가.

**현재 상태**: Claude/GPT 쿼터 리셋까지 약 4시간 대기 필요. 그동안 reasoningbank/efm arm의 스모크
테스트는 보류. no_memory arm은 10/10 완료된 상태로 남아있음(재사용 가능).

## 2026-10-01 — Claude Code CLI(`claude`)로 백본 전환 (Antigravity 대체, 쿼터 리셋 안 기다리기로 결정)

사용자가 4시간 대기 대신 Claude Code CLI(이 세션 자체의 구독)로 전환해서 바로 진행하기로 결정.
**트레이드오프 인지하고 결정함**: `claude -p ... --output-format stream-json --verbose` 호출 중
`rate_limit_event`로 이 세션 자체의 실시간 쿼터가 노출됨 — 7일 윈도우 utilization 0.76(75% 경고선
이미 넘음), 5시간 윈도우 0.25. 즉 이 전환은 "이 세션(나)"의 할당량을 reasoningbank/efm arm(20개
태스크)에 추가로 씀. 사용자가 이 사실을 확인하고도 "그래도 Claude CLI로 진행" 선택.

**구현**: `utils/claude_cli_client.py`(`ClaudeCliClient`) 신규 — `agy_client.py`와 동일 인터페이스/
패턴. 주요 차이:
- `claude`는 네이티브 `--system-prompt` 플래그가 있어 agy처럼 system+user를 한 프롬프트에 욱여넣을
  필요 없음.
- argv 로 프롬프트를 넘기면 agy와 동일하게 Windows 명령줄 길이 제한(`WinError 206`)에 걸림을 직접
  재현·확인 — 동일하게 `--input-format stream-json`(stdin NDJSON) + `--output-format stream-json
  --verbose`로 전환. NDJSON 스키마는 agy와 다름: `{"type":"user","message":{"role":"user",
  "content":"..."}}` (agy는 `{"event":"user",...}`). 최종 결과는 `"type":"result"` 줄의
  `result`/`is_error` 필드.
- **`--restricted` 플래그 필수**: 이게 없으면 claude도 Gemini처럼 Bash 등 도구를 호출하려다 막혀서
  빈 응답이 나올 위험이 있음(Gemini 테스트에서 실제로 겪은 문제, 동일 원인으로 재발 방지 차원에서
  선제 적용) — Gemini의 "list bid actions" 테스트 프롬프트로 직접 재현 검증, `--restricted` 적용
  후 정상적으로 `click("14")\nfill("15","2")\n...` 형태의 깔끔한 텍스트 답변 확인.
- `agents/legacy/utils/chat_api.py`에 `ChatClaudeCli`(langchain, 에이전트 루프용) 추가, `make_chat_model`
  dispatch에 `model_name.startswith("ccli-")` 분기 추가(`"claude"` 분기보다 먼저 와야 함, agy와 동일한
  이유).
- CLIENT_DICT 키: `"ccli-sonnet"` — `utils/clients.py`, `autoeval/clients.py` 양쪽에 등록.
  `count_tokens`(`llm_utils.py`)에 `ccli-` 접두사도 agy와 동일하게 tiktoken 근사 처리 추가.
  `pipeline_memory.py`/`induce_memory.py`/`autoeval/evaluate_trajectory.py`의 `--model` choices에
  `"ccli-sonnet"` 추가.
- `ClaudeCliClient.one_step_chat`/`ChatClaudeCli._call` 둘 다 작은 프롬프트, 65000자 큰 프롬프트,
  단위테스트(28/28 유지) 전부 직접 호출로 검증 완료.

## 2026-10-01 — 파이프라인 전용 별도 Claude 계정 분리 (`CLAUDE_CONFIG_DIR`)

이 세션 자체의 Claude Code 구독 쿼터(7일 76% 소진)를 파이프라인이 더 깎아먹지 않도록, 사용자가 보유한
**두 번째 계정**으로 분리하기로 함. API 키(`--bare` + `ANTHROPIC_API_KEY`, 종량제)는 처음에 시도했다가
사용자가 "API 키는 없고 계정만 있다"고 정정 — `--bare`는 되돌림.

**실제로 동작한 방법**: `claude` CLI는 `CLAUDE_CONFIG_DIR` 환경변수로 로그인/설정 저장 위치를 바꿀 수
있음(빈 디렉터리 지정 시 "Not logged in"이 뜨는 걸로 확인 — 완전히 독립된 인증 컨텍스트). 사용자가
본인 터미널에서 `CLAUDE_CONFIG_DIR=C:\Users\user\.claude-pipeline-alt`를 설정하고 두 번째 계정으로
`/login` 완료. 이후 같은 `CLAUDE_CONFIG_DIR`을 파이프라인 실행 환경(래퍼 스크립트)에 넣어주면
`ClaudeCliClient`/`ChatClaudeCli`는 코드 변경 없이(둘 다 `subprocess.run`이 부모 프로세스 env를
그대로 물려받음) 자동으로 그 계정으로 인증됨. 실측: 새 계정 쿼터 5시간 0%, 7일 25% — 이 세션(76%)과
완전히 분리된 걸 확인.

reasoningbank arm을 태스크 23부터(21,22는 이미 완료) 이 분리된 계정으로 재개.

## 2026-10-01 — 진짜 버그: EFM이 Description을 빼고 주입하고 있었음 (공정성 위반)

reasoningbank arm에서만 반복적으로 "거부(refusal)"가 발생하는 걸 조사하다가 발견한 더 근본적인 문제.
**EFM과 reasoningbank가 같은 10개 태스크를 도는데 efm은 거부 0회, reasoningbank는 거부가 반복됨** —
웹 콘텐츠 차이가 아니라 두 arm의 프롬프트 주입 형태 차이가 원인일 가능성이 높다고 보고 비교하다가,
`efm/parse.py`의 `format_item_for_prompt()`가 **`## Description`을 아예 안 보여주고 Title+Content만
주입하고 있었다는 걸 발견**함. 코드 주석엔 "baseline과 동일한 형태"라고 적혀 있었는데, 실제 baseline
(`run.py`의 `elif args.memory_path:` 분기)은 `# Memory Item i\n## Title\n## Description\n## Content`
전체를 그대로 주입함 — **주석의 근거 자체가 틀렸음**. `EFM_TASK.md`를 다시 찾아봐도 "주입 시
Description 제외"를 요구하는 근거는 어디에도 없음(description은 3.3절 그룹 유사도 계산에만 쓰임).

**영향**: 이건 단순 버그가 아니라 EFM_TASK.md 0절의 "동일 조건 비교" 원칙을 위반한 것 — EFM이
baseline보다 더 적은 정보(언제 이 메모리를 써야 하는지 설명)를 에이전트에게 주고 있었음. 오늘
돌린 efm 스모크 테스트(10/10, 전부 success)는 이 잘못된(Description 없는) 형식으로 나온 결과라
**baseline과 공정하게 비교할 수 없는 상태** — 참고용으로만 보존.

**수정**: `efm/parse.py`의 `format_item_for_prompt()`가 `## Description`을 포함하도록 수정, baseline과
완전히 동일한 포맷(`# Memory Item i\n## Title\n## Description\n## Content`)으로 맞춤. 이 설계를
전제로 "Description 없음"을 검증하던 단위테스트도 "Description 있음"으로 같이 수정(`test_efm_units.py`).
수정 후 단위테스트 28/28 유지 확인.

**거부율에 대한 가설**: Description을 추가하면 efm도 이제 reasoningbank와 비슷한 길이/구조의 프롬프트를
주입하게 되므로, efm에서도 거부가 나타나기 시작할 수 있음 — 다음에 efm을 다시 돌릴 때 확인 필요.
(이 수정 전 efm 스모크 테스트 결과의 "거부 0회"가 설계 자체의 우수성이 아니라 단순히 프롬프트가 더
짧았기 때문일 수 있다는 뜻 — 재확인 전까지는 결론 내리지 않음.)

## 2026-10-01 — 거부(refusal) 원인 조사 중단, 80개 본 실행으로 진행

reasoningbank arm에서 반복되는 거부의 정확한 트리거를 격리 테스트로 찾아보려 했으나(메모리 유무,
`<think>` 태그 요구 유무 등 조합 테스트), 테스트에 쓴 프롬프트 문구가 실제 파이프라인의 phrasing과
달라(명령조 vs few-shot 예시) 결론을 확정하지 못함. 사용자가 "테스트로 넣어서 이상하게 하지 말고
원본 기준으로 다음 단계로 넘어가자"고 결정 — 추가 격리 테스트 중단. 실측 기준으로는:
- no_memory arm: 거부 0건
- reasoningbank/efm arm: 거부 다수 발생하지만 태스크 단위 재시도로 전부 복구됨(유실 태스크 0개)

80개 본 실행에서도 이 현상이 반복될 수 있음 — arm별 거부/재시도 횟수와 완전 유실된 태스크 수를
최종 `results/summary.md`에 같이 집계하기로 함(원인 불명이어도 영향 범위는 투명하게 기록).

## 2026-10-01 — 거부가 확률적이라는 걸 확인 → 재시도 전략 수정(태스크 재시작 대신 즉시 재호출)

80개 본 실행 중 no_memory arm에서도 거부가 반복 발생(8개 중 4개꼴)하는 걸 보고 재검토. 에러 메시지
자체가 `"This sometimes happens with safe, normal conversations"`라고 명시 — Anthropic이 공식적으로
인정하는 **확률적 오탐(false positive)**이지 결정적 콘텐츠 차단이 아님. 실제로 동일 프롬프트를 바로
재호출하면 대부분 성공함을 확인(`claude-sonnet-5-5`에서 관측, 제 세션 계정의 `claude-sonnet-5`와
다른 버전).

**기존 설계의 문제**: "거부=결정적"이라고 잘못 가정하고 즉시 실패 처리 → 태스크 단위 재시도(브라우저
새로 열고 처음부터)에 맡겼었음. 근데 확률적이라면 그 자리에서 바로 다시 호출하는 게 훨씬 빠르고
이미 진행된 트래젝토리를 안 버려도 됨.

**수정**: `RefusalError`를 더 이상 즉시 상위로 던지지 않고, **같은 호출을 그 자리에서 즉시(sleep 없이)
최대 4번까지 재시도**하도록 변경(`utils/claude_cli_client.py`의 `ClaudeCliClient.chat()`,
`agents/legacy/utils/chat_api.py`의 `ChatClaudeCli._call()` 둘 다). 4번 다 거부되면 그때만 상위(태스크
재시도)로 전파. 단위테스트 28/28 유지 확인. 다음 태스크부터(새 서브프로세스) 자동 적용됨.

**임계값 보정 체크(무료, API 호출 없이 기존 캐시 재사용)**: `efm_equiv_test/`에 이미 계산된 194개
아이템의 description/content 임베딩으로 쌍별 코사인 유사도 분포 확인 — `TAU_D`/`TAU_C` 기본값 0.85가
대략 p90~95 구간에 해당함(너무 느슨하지도 빡빡하지도 않음). 보정 없이 기본값 그대로 사용하기로 결정.

**`--efm_n_inject` 실측값 반영**: 스모크 테스트 reasoningbank 데이터(10개 태스크)에서 태스크당
memory_items 개수 실측 — [3,3,3,2,3,3,3,3,3,3], 평균 2.9개(추출 상한 3개에 근접). baseline은 매칭된
experience의 아이템을 개수 제한 없이 전부 주입하므로, 이 실측 평균/상한에 맞춰 80개 본 실행의
`--efm_n_inject`를 기존 placeholder 2에서 **3**으로 올림.

## 2026-10-01 — 80개 × 3-arm 본 실행 시작

전부 빈 메모리에서 새로 시작(스모크 테스트나 이전 축적 데이터 재사용 안 함 — `--memory_dir
memories_real_*`로 분리). 백본: Claude Code CLI, 분리된 두 번째 계정(`CLAUDE_CONFIG_DIR`). 순서:
no_memory → efm → reasoningbank(80개씩, 총 240개 태스크). 거부(refusal) 현상은 원인 규명 중단하고
태스크 재시도로 흡수, 최종 집계 시 arm별 거부/유실 횟수 같이 기록하기로 함(위 항목 참고).

## 2026-10-01 16:29 — 두 번째 계정도 세션 한도 도달, 일시 중단

no_memory arm이 61/80까지 진행된 시점에 두 번째 계정도 `429` 세션 한도에 걸림(`"You've hit your
session limit · resets 7pm (Asia/Seoul)"`). 재시도만 낭비되는 걸 확인(같은 에러 45회 누적, 완료
카운트 정체)하고 전체 프로세스 중단.

**현재 상태**: no_memory 61/80 완료(저장됨, 재개 시 `--prev_id`로 이어서), efm/reasoningbank는
둘 다 태스크 1개도 못 끝내고 중단(디렉터리 자체가 안 생김, 정리할 것도 없음). 19시(한국시간) 이후
재개 예정.

## 2026-10-01 — 80/80/80 실행 권장 + 논문 레퍼런스 수치

사용자 결정: 세 arm 전부 **80개씩**으로 통일(쿼터/시간 감안). 평가 지표는 **success rate(SR)의
arm 간 증가분**으로 본다.

**원 논문(ReasoningBank, ICLR 2026) Table 1 참고 수치** — WebArena Shopping 서브셋(n=187, 이번
실험과 동일한 태스크 풀):

| Backbone | No Memory SR | ReasoningBank SR | 증가분 |
|---|---|---|---|
| Gemini-2.5-flash | 39.0 | 49.7 | +10.7 |
| Gemini-2.5-pro | 45.5 | 51.9 | +6.4 |
| Claude-3.7-sonnet | 38.5 | 44.9 | +6.4 |

지금 실험은 Claude 계열 백본(`ccli-sonnet`)을 쓰므로, 논문의 Claude-3.7-sonnet 행(+6.4 SR)이 가장
가까운 참고치. 단, 모델 버전이 다르고(Claude-3.7 vs 지금 쓰는 Sonnet), n=80(논문은 n=187)이라
직접 비교는 참고용일 뿐 — `ReasoningBank → EFM`의 개선폭이 이 수준(±6~10 SR) 안팎인지 보는 용도로만
쓸 것.

## 2026-10-01 — Mind2Web으로 EFM 3-arm 포팅 (Docker 불필요, 공유용)

사용자가 다른 협업자(고려대)와 공유하려는데 WebArena는 Docker 호스팅이 필요해서 현실적으로 어려움 —
Mind2Web은 오프라인 리플레이라 Docker 없이 `download_data.py`만으로 재현 가능해서 전환 결정.

**중요 발견**: 루트 `CLAUDE.md`가 "Mind2Web은 Actor-only, ReasoningBank 메모리 미연동"이라고 적어뒀는데
이미 stale함 — `Mind2Web/memory.py`에 reasoningbank 단일 arm 메모리 루프(검색/judge/추출)가 이미
구현·연동(`--memory-mode reasoningbank`)되어 있었음. 그래서 EFM을 처음부터 새로 만든 게 아니라,
이미 동작하는 `none`/`reasoningbank` 인프라 위에 `efm` arm 하나를 추가하는 작업이었음.

**구현**: `Mind2Web/efm/`(WebArena/efm/ 포팅 — 알고리즘 자체는 도메인 무관이라 거의 그대로, 임베딩만
`memory.py`의 `embed_query()`를 감싸는 `store.embed_text()`로 교체), `Mind2Web/utils/{agy_client.py,
claude_cli_client.py}` 추가 + `clients.py`에 등록(`agy-claude-sonnet-4-6`, `ccli-sonnet`), `run.py`의
`process_task()`/`main()`에 `efm_ctx` 분기 추가(`--memory-mode efm`). 단위테스트 28/28 포팅·통과.

**검증 상태**: 3-태스크 생티티 체크 시도했으나 두 번째 Claude 계정의 세션 한도(19시 리셋)가 아직
안 풀려서 judge/추출 호출이 막힘(코드 버그 아님, graceful하게 에러 로깅하고 안 죽음 — 확인됨). 쿼터
리셋 후 재검증 필요.
