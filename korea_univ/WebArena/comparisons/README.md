# 메모리 기법 비교 (6가지)

각 폴더는 **하나의 메모리 기법**을 나타내고, 그 안의 `run.sh`로 바로 돌릴 수 있습니다. 전부 같은
파이프라인(`pipeline_memory.py`)을 `--memory_mode` 플래그로만 다르게 호출하는 것이라 코드는
공유되지만, 결과/메모리 디렉터리는 전부 분리돼 있어서 서로 섞이지 않습니다.

| 폴더 | 방식 | 코드 상태 |
|---|---|---|
| `no_memory/` | 메모리 없음(하한선) | 기존 baseline |
| `synapse/` | 성공한 궤적을 통째로 "메모리"로 저장(distillation 없음) | 기존 baseline |
| `awm/` (Agent Workflow Memory) | 성공 사례들에서 반복되는 워크플로를 추출 | 기존 baseline |
| `reasoningbank/` | 성공/실패 모두에서 추론 전략을 distill, 가장 가까운 과거 experience를 통째로 주입 | 원 논문 baseline |
| `ace/` (Agentic Context Engineering) | Reflector가 통찰을 bullet로 추출, 기존 playbook과 임베딩 유사도로 비교해 merge(helpful/harmful 투표) 또는 신규 추가. 전체 playbook을 매 태스크 주입(크기 상한 적용) | **이번에 새로 재구현**(원 코드 `ace-agent/ace`는 단일 QA 형식이라 우리 멀티스텝 구조에 안 맞아 알고리즘만 재현 — `WebArena/ace/`, `NOTES.md` 참고) |
| `efm/` (Edit-Free Memory Lifecycle) | **우리 연구** — 삽입 시 D/C 임베딩 유사도로 Duplicate/Group/Conflict/Novel 분류, 검색 시 그룹 collapse+충돌 co-injection, 사용 증거 기반 삭제 | `WebArena/efm/` |

## 공통 실행법

```bash
cd comparisons/<method>
./run.sh                 # 기본값: shopping, 80개 태스크, ccli-sonnet
./run.sh --end_index 187  # 전체 187개로 바꾸고 싶을 때처럼 pipeline_memory.py 옵션을 그대로 덧붙일 수 있음
```

각 `run.sh`는 `memories_<method>_cmp/`, `results_<method>_cmp/`로 결과를 저장합니다(기존에
축적된 `memories_reasoningbank/` 등과 섞이지 않음 — 전부 이 비교 실험 전용 새 디렉터리).

## 결과 집계

모든 방식 공통으로 `results_<method>_cmp/webarena.<id>/<model>_autoeval.json`의 `rm` 필드가
성공/실패 판정입니다. `ace`는 추가로 `memories_ace_cmp/shopping/playbook.json`에서
`helpful`/`harmful` 투표가 쌓인 플레이북을 직접 열어볼 수 있고, `efm`은
`memories_efm_cmp/shopping/items.json`에서 `n_used`/`n_retrieved`/`group_id`/`conflict_with`/
`deleted_reason`을 볼 수 있습니다.
