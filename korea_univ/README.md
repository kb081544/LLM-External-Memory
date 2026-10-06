# LLM External Memory (EFM) Experiment

ReasoningBank 논문 코드(`google-research/reasoning-bank`)를 베이스로, **EFM(Edit-Free Memory
Lifecycle)** — 삽입 시점에 설명(D)/내용(C) 임베딩 유사도로 아이템을 Duplicate/Group/Conflict/Novel로
분류하고, 검색 시점에 그룹 collapse + 충돌 co-injection을 하고, 사용 증거 기반으로 오래되거나
안 쓰이는 메모리를 삭제(forgetting)하는 메모리 관리 기법 — 을 구현하고 baseline(ReasoningBank의
단순 누적 주입)과 비교하는 연구 코드입니다.

**6가지 메모리 기법**을 비교합니다 — 코드는 최상위에 **방법론별로** 정리돼 있고(`no_memory/`,
`synapse/`, `awm/`, `reasoningbank/`, `ace/`, `efm/`), 각 폴더 안에 벤치마크별 실행 스크립트가
있습니다. 벤치마크(WebArena/Mind2Web)의 실제 코드·데이터는 각자의 폴더에 그대로 있고, 아래
방법론 폴더들은 거기로 들어가서 올바른 플래그로 호출해주는 역할만 합니다.

| 폴더 | 방식 | WebArena | Mind2Web |
|---|---|---|---|
| `no_memory/` | 메모리 없음(하한선) | ✅ | ✅ |
| `synapse/` | 성공 궤적을 통째로 저장(distillation 없음) | ✅ | ✅ |
| `awm/` (Agent Workflow Memory) | 성공 사례에서 반복 워크플로 추출 | ✅ | ✅ |
| `reasoningbank/` | **baseline** — 성공/실패 모두 distill, 가장 가까운 experience 통째로 주입 | ✅ | ✅ |
| `ace/` (Agentic Context Engineering) | Reflector가 bullet 추출 → helpful/harmful 투표로 playbook 진화 (재구현, `NOTES.md` 참고) | ✅ | ✅ |
| `efm/` (Edit-Free Memory Lifecycle) | **우리 연구** — D/C 유사도 기반 분류·그룹·충돌 관리 + 사용 증거 기반 삭제 | ✅ | ✅ |

이제 6가지 기법 전부 **두 벤치마크 모두에서** 바로 돌릴 수 있습니다(`synapse`/`awm`/`ace`의
Mind2Web 포팅은 WebArena 쪽 프롬프트·알고리즘을 그대로 재사용 — `NOTES.md` 참고).

```bash
cd efm
./run_webarena.sh     # Docker 쇼핑몰 사이트 필요
./run_mind2web.sh     # Docker 불필요, download_data.py만 하면 바로 실행
```

각 방법론 폴더의 `README.md`에 벤치마크별 지원 여부와 정확한 실행 명령이 있습니다.

같은 비교를 **두 벤치마크**에서 돌릴 수 있습니다 — 상황에 맞게 고르세요:

| | WebArena | Mind2Web |
|---|---|---|
| 평가 대상 | Shopping 서브셋(187개 태스크, 원 논문 Table 1과 동일한 풀) | Shopping 도메인(HuggingFace에서 다운로드) |
| 인프라 | **Docker로 쇼핑몰 사이트를 직접 호스팅해야 함** | 없음 — 이미 캡처된 스텝을 오프라인 리플레이 |
| 적합한 상황 | 이미 Docker 환경이 떠 있는 경우 | **다른 곳에서 빠르게 돌려보고 싶을 때(추천)** — `python download_data.py`만 하면 바로 실행 가능 |
| 지표 | task success rate(SR, autoeval judge 기준) | task success rate(SR, 요소/동작/값 매칭 기준) |

두 벤치마크의 코드는 독립적이지만(`WebArena/efm/` ↔ `Mind2Web/efm/`), EFM 알고리즘·하이퍼파라미터·
주입 포맷은 동일합니다 — 자세한 실행 명령은 아래 각 섹션 참고.

## 문서

- **`DATA_AND_TASK.md`** — **먼저 읽으세요.** 어떤 모델을 쓰는지, 어떤 데이터가 있는지, WebArena
  태스크/메모리 뱅크의 실제 JSON 모양, 그리고 지금 해야 할 작업이 뭔지 정리돼 있습니다.
- **`EFM_TASK.md`** — EFM 알고리즘 명세(원본 작업 지시서)
- **`CODE_MAP.md`** — 어떤 컴포넌트가 어느 파일/함수에 있는지 지도
- **`NOTES.md`** — 작업 중 내린 판단과 이유, 발견한 버그와 수정 내역을 시간순으로 전부 기록한
  상세 로그(부록 성격). 코드를 깊게 고치거나 과거 버그(임베딩 쿼터, 백본 모델 전환, 프롬프트 공정성
  버그 등)를 재현하기 전에 검색해서 참고하는 용도 — 처음 훑어볼 땐 몰라도 됨.

## 환경 설정

```bash
# Python 3.12, uv로 의존성 관리
uv sync
```

### 필요한 API 키/환경변수

```bash
# 임베딩(검색 유사도 계산용, gemini-embedding-001) — 무료 티어 쿼터 있음(분당 100회, 일일 1000회)
export GOOGLE_API_KEY="..."

# WebArena 자체의 ground-truth 평가기(llm_fuzzy_match/llm_ua_match)가 내부적으로 OpenAI를 직접 호출함
# (--judge 설정과 무관하게 항상 필요). 로컬 Ollama로 대체 가능 — gpt-4-1106-preview 등 이름으로
# 모델을 pull해두고 아래처럼 설정하면 무료로 돌아감.
export OPENAI_API_KEY="아무-더미-값"
export OPENAI_BASE_URL="http://localhost:11434/v1"   # Ollama 쓸 경우

# WebArena 쇼핑몰/어드민/레딧 등 사이트 URL (자체 호스팅한 Docker 환경)
export WA_SHOPPING="http://..."
export WA_SHOPPING_ADMIN="http://.../admin"
# ... (나머지는 루트 CLAUDE.md 참고)
```

### 백본 LLM: Claude Code CLI 경유

이 코드는 백본 LLM 호출을 **API 키 직접 호출이 아니라 `claude` CLI(Claude Code CLI)를 서브프로세스로
호출**하는 방식을 씁니다(`--model ccli-sonnet`). 이유와 세부 구현은 `NOTES.md`에 상세히 기록돼
있습니다(요약: Gemini/Antigravity API 무료 쿼터를 반복적으로 소진해서, 구독 기반 CLI로 전환).

사용하려면 [Claude Code](https://claude.com/claude-code)가 설치·로그인돼 있어야 합니다(`claude`
명령이 PATH에 있어야 함). **다른 계정으로 분리해서 돌리고 싶으면** `CLAUDE_CONFIG_DIR` 환경변수로
별도 로그인 프로필을 지정하세요:

```bash
export CLAUDE_CONFIG_DIR="/path/to/separate/profile"
claude   # 최초 1회, 해당 프로필로 로그인
```

다른 백본(Gemini API 직접 호출, Antigravity CLI 등)을 쓰고 싶으면 `--model` 값만 바꾸면 됩니다 —
`WebArena/utils/clients.py`의 `CLIENT_DICT`에 등록된 키 목록 참고.

## 공통 옵션

- WebArena 쪽 스크립트: `--end_index 80`(처음 80개, 187개 전체면 이 옵션 빼기), `--prev_id N`(중단 후
  재개), `--efm_n_inject`/`--efm_*`(EFM 하이퍼파라미터, `WebArena/efm/hparams.py` 참고).
- Mind2Web 쪽 스크립트: `--limit 80`(시드 고정, `results_*/manifest.json`에 기록돼 재현 가능),
  `--report-only`(LLM 호출 없이 리포트만 재생성), `--efm-n-inject`/`--efm-*`(하이픈 표기, 기본값은
  WebArena와 동일).
- 각 방법론 폴더의 스크립트는 전부 `"$@"`로 추가 플래그를 그대로 전달합니다 — 예:
  `./run_webarena.sh --end_index 187`.
- 같은 벤치마크 안에서 여러 방법론을 돌려도 결과/메모리 디렉터리가 전부 분리돼 있어 서로 안
  섞입니다(단, WebArena는 같은 Docker 사이트를 공유하므로 동시 병렬 실행은 피하세요).

## 결과 집계

```bash
# WebArena: 각 결과 디렉터리에서 <model>_autoeval.json의 "rm" 필드(성공/실패)를 집계
# Mind2Web: 각 결과 디렉터리에서 trajectory.json의 "task_success" 필드를 집계
#           (또는 summary.json에 arm 전체 집계가 이미 매 태스크마다 갱신되어 있음)
```

평가는 **task success rate(SR)의 arm 간 증감**으로 봅니다 — 원 논문(Table 1, WebArena Shopping
서브셋 n=187)에서는 `No Memory → ReasoningBank`가 백본 모델에 따라 **+6.4 ~ +10.7 SR**의 개선을
보였습니다. 이번 실험에서 `ReasoningBank → EFM`의 개선폭을 그 기준과 비교해서 해석하면 됩니다.

주의: 샘플 수(n)가 작을수록(예: n=80) 신뢰구간이 넓어집니다 — 자세한 통계적 해석은 `NOTES.md`의
관련 항목을 참고하세요.
