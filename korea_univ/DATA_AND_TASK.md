# 데이터와 작업 개요

## 모델

| 역할 | 모델 | 비고 |
|---|---|---|
| 에이전트(백본) | Claude (Claude Code CLI 경유, `--model ccli-sonnet`) | 서브프로세스로 `claude -p` 호출. API 키 아님 — 구독 계정 로그인 사용 |
| Judge (autoeval) | 위와 동일 백본 | `--judge autoeval`, 같은 `ccli-sonnet` |
| 임베딩 (검색/유사도) | `gemini-embedding-001` (Google GenAI API, 3072차원) | `GOOGLE_API_KEY` 필요, 무료 티어(분당 100회/일 1000회) |
| WebArena 자체 ground-truth 평가기 | OpenAI 호환 엔드포인트 (`llm_fuzzy_match`/`llm_ua_match`) | `--judge` 설정과 무관하게 내부적으로 항상 호출됨. 로컬 Ollama로 대체 가능(README 참고) |

원 논문은 Gemini-2.5-flash/pro, Claude-3.7-sonnet 세 백본으로 실험했음. 이번 실험은 Claude 계열
백본 하나로만 진행(쿼터 사정, NOTES.md 참고).

## 존재하는 데이터

- **`WebArena/config_files/*.json`** (gitignored, `generate_config_files.py`로 생성) — WebArena
  태스크 1개당 JSON 1개. Shopping 사이트 태스크가 총 **187개** 있음(원 논문 Table 1의 Shopping
  열과 동일한 n).
- **`WebArena/memories_reasoningbank/shopping.jsonl`** — 이미 reasoningbank 모드로 추출된 메모리
  185개 항목(2026-08-05 기준 축적분). 새로 80개/187개를 돌리면 별도 디렉터리에 새로 쌓임(기존
  축적분과 섞지 않음 — README의 `--memory_dir` 사용법 참고).
- **`"ReasoningBank Scaling Agent Self-Evolving with Reasoning Memory.pdf"`** / `_paper_text.txt` —
  원 논문 원문(텍스트 추출본 포함).

## WebArena 데이터 모양

### 태스크 설정 (`config_files/21.json`)

```json
{
  "sites": ["shopping"],
  "task_id": 21,
  "start_url": "http://localhost:7770/6s-wireless-headphones-....html",
  "intent": "List out reviewers, if exist, who mention about ear cups being small",
  "eval": {
    "eval_types": ["string_match"],
    "reference_answers": {
      "must_include": ["Joseph Brzezinski", "Catso", "Dibbins", "Anglebert Dinkherhump", "Michelle Davis"]
    }
  },
  "intent_template_id": 222
}
```

- `intent`가 에이전트에게 주어지는 실제 자연어 지시문.
- `eval`은 WebArena 자체 ground-truth 채점 기준(지금 실험에선 참고용, 실제 성공 판정은 autoeval judge의
  `rm` 필드를 씀).
- `intent_template_id`는 같은 패턴(예: "List out reviewers who mention about {{X}}")의 다른 인스턴스를
  식별하는 템플릿 ID — 메모리 중복/일반화 분석에 유용.

### 메모리 뱅크 1건 (`memories_reasoningbank/shopping.jsonl`의 한 줄)

```json
{
  "task_id": "21",
  "query": "List out reviewers, if exist, who mention about ear cups being small",
  "status": "success",
  "template_id": 222,
  "memory_items": [
    "# Memory Item 1\n## Title Locating Specific Feedback in Customer Reviews\n## Description Use this strategy when searching for particular complaints or product features mentioned by users in review sections.\n## Content Scan the customer review pages or use text filtering/search features to locate keywords related to the user's query..."
  ]
}
```

- `memory_items`는 문자열 리스트, 각 원소가 `# Memory Item i\n## Title\n## Description\n## Content`
  형식 — 이 형식 그대로 다음 태스크의 에이전트 프롬프트에 주입됨(baseline/efm 둘 다 동일 형식 사용).
- `status`가 `success`/`fail` — 성공한 궤적뿐 아니라 **실패한 궤적에서도** 메모리를 추출하는 게
  ReasoningBank의 핵심 아이디어(실패 원인을 다음에 피하는 법을 기록).

### 실행 결과 (`results_*/webarena.<id>/`)

- `step_N.pkl.gz` — 스텝별 `agent_info`(`think`/`action`), `obs`(axtree 텍스트 등)
- `summary_info.json` — `cum_reward`(WebArena 자체 ground-truth 보상)
- `<model>_autoeval.json` — `[{"rm": true/false, "thoughts": "...", ...}]`, **이게 실제 성공/실패
  판정**

## 해야 할 일

1. 세 arm(`no_memory`/`reasoningbank`/`efm`)을 각각 **80개 태스크**(또는 전체 187개)로 돌린다 —
   README.md의 실행 명령 참고.
2. 각 arm의 `results_*/webarena.*/​<model>_autoeval.json`에서 `rm` 필드를 모아 **arm별 success
   rate(SR)**를 계산한다.
3. `ReasoningBank → EFM`의 SR 증가분을 `No Memory → ReasoningBank`의 증가분(원 논문 기준 +6.4,
   Claude-3.7-sonnet 백본, 이 저장소의 NOTES.md에 표로 정리돼 있음)과 비교해서 EFM이 baseline
   대비 추가 개선을 내는지 확인한다.
4. (선택) efm arm의 `memories_*/shopping/items.json`을 보면 아이템별 `n_retrieved`/`n_used`/
   `group_id`/`conflict_with`/`deleted_reason`이 기록돼 있어, "어떤 메모리가 실제로 쓰였는지",
   "어떤 메모리가 삭제됐는지"를 사후 분석할 수 있다.

더 자세한 설계 배경과 중간에 발견된 버그/판단 이유는 `NOTES.md`(시간순 작업 로그)를 참고.
