# LLM External Memory (EFM) Experiment

ReasoningBank 논문 코드(`google-research/reasoning-bank`)를 베이스로, **EFM(Edit-Free Memory
Lifecycle)** — 삽입 시점에 설명(D)/내용(C) 임베딩 유사도로 아이템을 Duplicate/Group/Conflict/Novel로
분류하고, 검색 시점에 그룹 collapse + 충돌 co-injection을 하고, 사용 증거 기반으로 오래되거나
안 쓰이는 메모리를 삭제(forgetting)하는 메모리 관리 기법 — 을 구현하고 baseline(ReasoningBank의
단순 누적 주입)과 비교하는 연구 코드입니다.

세 가지 arm(조건)을 비교합니다:

| arm | 메모리 사용 | 설명 |
|---|---|---|
| `no_memory` | 없음 | 메모리 없이 순수 에이전트 성능 (하한선) |
| `reasoningbank` | baseline | 원 논문의 방식 — 가장 가까운 과거 experience의 메모리를 전부, 수정 없이 누적 주입 |
| `efm` | 우리 연구 | D/C 유사도 기반 분류·그룹·충돌 관리 + 사용 증거 기반 삭제 |

평가 대상: WebArena Shopping 서브셋(187개 태스크, 원 논문 Table 1과 동일한 풀). 지표는 **task
success rate(SR, autoeval judge 기준)**.

## 상세 설계·결정 기록

- **`EFM_TASK.md`** — EFM 알고리즘 명세(원본 작업 지시서)
- **`CODE_MAP.md`** — 어떤 컴포넌트가 어느 파일/함수에 있는지 지도
- **`NOTES.md`** — 작업 중 내린 모든 판단과 이유, 발견한 버그와 수정 내역(시간순). **코드를 고치거나
  재현하기 전에 꼭 한번 읽어보세요** — 같은 문제를 다시 겪지 않도록 임베딩 쿼터, 백본 모델 전환,
  프롬프트 공정성 버그 등이 전부 기록돼 있습니다.

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

## 실행 방법 — 세 arm을 각각 돌리기

세 arm은 **완전히 분리된 결과/메모리 디렉터리**를 쓰므로 서로 간섭하지 않습니다. 순서 상관없이
독립적으로 실행 가능합니다(단, 같은 Docker 쇼핑몰 사이트를 공유하므로 동시에 병렬 실행은 피하세요 —
사이트 상태가 섞일 수 있습니다).

```bash
cd WebArena

# 1) no_memory arm — 메모리 없음, 하한선
python pipeline_memory.py --website shopping --model ccli-sonnet --memory_mode no_memory \
    --output_dir results_no_memory --end_index 80

# 2) reasoningbank arm — baseline (원 논문 방식)
python pipeline_memory.py --website shopping --model ccli-sonnet --memory_mode reasoningbank \
    --memory_dir memories_reasoningbank_run --output_dir results_reasoningbank --end_index 80

# 3) efm arm — 우리 연구
python pipeline_memory.py --website shopping --model ccli-sonnet --memory_mode efm \
    --memory_dir memories_efm_run --output_dir results_efm --end_index 80 --efm_n_inject 3
```

- `--end_index 80`: 전체 187개 중 처음 80개만(쿼터 상황에 따라 조절). 187개 전체를 돌리려면 이 옵션을
  빼세요.
- `--prev_id N`: 중간에 중단됐을 때 태스크 id가 N 이하인 것들을 건너뛰고 이어서 재개(이미 완료된 건
  안 날아감).
- 각 arm의 결과는 `--output_dir`로 지정한 디렉터리에 `webarena.<task_id>/` 폴더별로 쌓입니다
  (스텝 기록, judge 판정 포함).
- EFM 하이퍼파라미터(`TAU_D`, `TAU_C`, `ETA`, `T_STALE` 등)는 `--efm_*` 플래그로 조절 가능 —
  `WebArena/efm/hparams.py` 참고. 기본값은 `EFM_TASK.md` 3.7절 표와 동일.

## 결과 집계

```bash
cd WebArena
# 각 결과 디렉터리에서 <model>_autoeval.json의 "rm" 필드(성공/실패)를 집계하면 arm별 SR이 나옵니다.
```

평가는 **task success rate(SR)의 arm 간 증감**으로 봅니다 — 원 논문(Table 1, WebArena Shopping
서브셋 n=187)에서는 `No Memory → ReasoningBank`가 백본 모델에 따라 **+6.4 ~ +10.7 SR**의 개선을
보였습니다. 이번 실험에서 `ReasoningBank → EFM`의 개선폭을 그 기준과 비교해서 해석하면 됩니다.

주의: 샘플 수(n)가 작을수록(예: n=80) 신뢰구간이 넓어집니다 — 자세한 통계적 해석은 `NOTES.md`의
관련 항목을 참고하세요.
