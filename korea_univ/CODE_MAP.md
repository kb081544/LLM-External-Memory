# CODE_MAP — EFM 작업용 (WebArena/shopping 기준)

`EFM_TASK.md` 2절 요구사항에 따라 작성. 전부 `baseline-original` 태그 시점(=이 브랜치 분기 시점) 코드 기준.

## 태스크 루프

- **파일**: `WebArena/pipeline_memory.py`
- **함수**: `main()`
- 태스크 순서: `config_files/*.json`을 `--website`로 필터링 후 `task_id` 오름차순 정렬, `task_ids[start:end]` 순회.
- 한 태스크당 3단계 서브프로세스: ① `run.py`(검색+에이전트 실행) → ② `autoeval.evaluate_trajectory`(judge) →
  ③ `induce_memory.py`(추출). 메모리 갱신(파일에 새 줄 append)은 **③ 끝난 시점**.
- `--memory_mode` choices: 기존 `no_memory`/`reasoningbank`/`awm`/`synapse`/`reasoningbank_pruned`
  (마지막은 오늘 낮에 만든 별개 실험, 이번 EFM 비교엔 안 씀). **`efm` 추가 예정.**

## 메모리 저장 형식

- **경로**: `memories_<mode>/<website>.jsonl` (한 줄 = 한 태스크), `memories_<mode>/<website>_embeddings.jsonl`
  (쿼리 임베딩 캐시, 별도 파일).
- **JSON 구조** (`WebArena/induce_memory.py` `main()` L176-186):
  `{task_id, query, think_list, action_list, status, memory_items: [str, ...], template_id}`
- **`memory_items` 문자열 포맷** — `WebArena/prompts/memory_instruction.py`의 `SUCCESSFUL_SI`/`FAILED_SI`가
  강제하는 출력 형식:
  ```
  # Memory Item i
  ## Title <제목>
  ## Description <한 문장: 언제/언제 안 쓸지>
  ## Content <1-3문장 인사이트>
  ```
  **→ EFM의 title/description/content 스키마가 이미 그대로 나옴. 파싱만 하면 됨(LLM 재호출 불필요).**

## 검색 (retrieval)

- **파일**: `WebArena/memory_management.py`
- **함수**: `select_memory(n, reasoning_bank, cur_query, task_id, cache_path, prefer_model)` L116,
  `screening(cur_query, cache_path, task_id, prefer_model)` L150
- 쿼리 임베딩 cosine 유사도로 상위 n개 **experience**(태스크) 선택. `screening()`이 부수효과로 현재 쿼리
  임베딩을 캐시 파일에 append(누적).
- **호출부**: `WebArena/run.py` `main()` L185 — `select_memory(n=1, ...)`. **n=1**이므로 baseline은
  가장 가까운 experience **하나**의 `memory_items`를 통째로 주입(평균 주입 개수는 그 experience의
  `memory_items` 길이 — 5.1 sanity check에서 처음 20개 태스크로 실측 예정, 가정하지 않음).
- 임베딩 모델: `embed_query_with_gemini()`(L78), `gemini-embedding-001`, 3072차원.

## 메모리 추출 (extraction)

- **파일**: `WebArena/induce_memory.py`, 프롬프트 `WebArena/prompts/memory_instruction.py`
- **함수**: `main()` L117 — `reward == 1`(gt의 `cum_reward` 또는 autoeval의 `rm`)로 성공/실패 분기,
  `SUCCESSFUL_SI`/`FAILED_SI` 중 하나로 `llm_client.one_step_chat()` 1회 호출.
- 출력은 `"\n\n"`로 split해 `memory_items` 리스트로 저장(L184).

## Judge (성공/실패 판정)

- **파일**: `WebArena/autoeval/evaluate_trajectory.py`(엔트리) + `WebArena/autoeval/evaluator.py`
  (`class Evaluator`, `eval_text()`)
- **프롬프트**: `WebArena/prompts/autoeval_prompts.py` `build_text_eval_prompt()` (기본, `--prompt text`)
- 결과: `results_<dir>/webarena.<id>/<model>_autoeval.json` → `[{"rm": bool, "gt": float, ...}]`

## 임베딩 모델

- `gemini-embedding-001` (via `google.genai`), `WebArena/memory_management.py`의
  `embed_query_with_gemini()`/`get_embeddings()`. EFM의 description/content 임베딩도 **같은 함수 재사용**.

## 사용 선언 (EFM 3.4에 필요한 것 — 이미 존재함)

- **파일**: `WebArena/agents/legacy/agent.py` `get_action()` L134-137
- `memory_path`가 설정돼 있으면, 시스템 프롬프트에 *"...please first explicitly discuss if you want to
  use each memory item or not, and then take action."*가 이미 삽입됨(baseline도 동일하게 받음 — 프롬프트
  변경 없음, EFM은 이 기존 지시에 대한 에이전트의 응답(`<think>`, `step_N.pkl.gz`의 `agent_info['think']`)을
  **사후 파싱**만 하면 됨).
- **주의**: 현재 baseline은 n=1이라 한 experience의 아이템들만 주입되므로 "Memory Item N" 번호가 안
  겹침. EFM은 여러 experience/아이템을 섞어 주입하므로, 사용 선언 파싱을 위해 **주입 시점에 아이템을
  1..K로 재번호**해야 함(Mind2Web `memory.py`의 기존 재번호 로직과 동일 패턴, 별도 파일에 구현).

## 평가 (SR 계산)

- `results_<dir>/webarena.<id>/<model>_autoeval.json`의 `rm` 필드를 전 태스크에 대해 집계
  (오늘 하루 비교 작업에서 반복 사용한 패턴, 별도 스크립트 없음 — `results/summary.md` 생성 시 재사용).

## 실행 스크립트

```
python pipeline_memory.py --website shopping --memory_mode <no_memory|reasoningbank|efm> \
  --model <model> --output_dir <dir>
```
기존 진입점 그대로, `efm` 분기만 추가.

## 백본 LLM 호출 경로 (오늘 확정)

에이전트/judge/추출/EFM의 NLI·사용여부 분류 전부 `Antigravity CLI`(`agy -p ... --model claude-sonnet-4-6`)
경유. 새 클라이언트: `WebArena/utils/agy_client.py`, 기존 `CLIENT_DICT` 3곳
(`utils/clients.py`, `autoeval/clients.py`, `agents/legacy/utils/chat_api.py`)에 동일 패턴으로 등록.
