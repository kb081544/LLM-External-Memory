# TASK: ReasoningBank에 Edit-Free Memory Lifecycle(EFM) 적용 및 Success Rate 비교

이 문서는 Claude Code에게 주는 작업 명세입니다. 처음부터 끝까지 읽은 뒤 순서대로 진행하세요.

## 0. 목표

기존 ReasoningBank(이하 baseline)의 메모리 관리 방식을 이 문서의 EFM 방식으로 바꾼 버전을 만들고, **같은 조건에서 두 버전을 실행해 success rate(SR) 차이를 보고**한다.

- 원본 코드와 수정 코드를 **모두 보존**한다.
- 메모리 **관리**(저장, 검색 후보 정리, 삭제)만 바꾼다. 에이전트 프롬프트, 백본 모델, judge, 임베딩 모델, 메모리 추출(extraction) 프롬프트는 **절대 바꾸지 않는다.**
- 최종 산출물: `results/summary.md`의 비교표 (baseline SR, EFM SR, 차이).

---

## 1. 작업 규칙

1. **원본 보존**
   - 시작 전 현재 상태를 커밋하고 태그를 단다: `git tag baseline-original`
   - 새 브랜치에서 작업한다: `git checkout -b efm`
   - git이 없으면 저장소 전체를 `../<repo>_original/`로 복사해 두고 거기는 건드리지 않는다.
2. **한 코드베이스에서 두 모드 실행**: 수정 브랜치에서 `--memory_mode baseline|efm` 플래그로 두 방식을 모두 돌릴 수 있게 만든다. `baseline` 모드는 원본과 **완전히 동일하게** 동작해야 한다.
3. **새 로직은 새 파일에**: `efm/` 디렉터리(또는 저장소 관례에 맞는 위치)에 모듈을 만들고, 기존 파일에는 분기 호출만 최소한으로 추가한다.
4. **비용이 큰 실행 전 보고**: 전체 벤치마크 실행 전에 아래 5절의 sanity check 결과와 예상 API 호출 수를 사용자에게 보고하고 승인받는다.
5. 불확실한 점이 있어도 멈추지 말고, 가장 합리적인 선택을 한 뒤 `NOTES.md`에 **무엇을 왜 그렇게 했는지** 기록한다. 단, 2절의 코드 탐색에서 핵심 구성요소(검색, 추출, 저장)를 찾지 못하면 멈추고 보고한다.

---

## 2. 먼저 할 일: 코드 탐색

코드를 수정하기 전에 다음을 찾아 `CODE_MAP.md`에 파일 경로, 함수명, 한두 줄 설명으로 정리한다.

| 찾을 것 | 확인할 내용 |
|---|---|
| 태스크 루프 | 태스크 순서, 한 태스크 처리 후 메모리 갱신 시점 |
| 메모리 저장 형식 | JSON 구조, 임베딩 저장 위치(별도 파일인지) |
| 검색(retrieval) | query 임베딩, cosine, top-k 값, 주입되는 필드(title+content) |
| 메모리 추출 | trajectory → memory items 변환, 성공/실패 분기 |
| judge | 성공/실패 판정 함수 |
| 임베딩 모델 | 모델명, 호출 함수 |
| 사용 선언 | 에이전트에게 "각 memory item 사용 여부를 명시적으로 논하라"고 지시하는 프롬프트와, 그 응답이 로그 어디에 남는지 |
| 평가 | SR 계산 방식, 결과 저장 위치 |
| 실행 스크립트 | 논문 결과 재현에 쓰는 명령어와 설정값 |

baseline이 **태스크당 평균 몇 개의 memory item을 프롬프트에 주입하는지**도 확인해 기록한다. EFM도 같은 개수를 주입해야 공정하다(3.2절).

---

## 3. 알고리즘 명세

### 3.1 핵심 원칙

- 저장된 아이템의 `title`, `description`, `content`는 **생성 후 절대 수정하지 않는다.**
- 메모리 관리는 **유지, 삭제, 표시** 세 가지 연산만 쓴다.
- 관리용 정보는 아이템의 별도 필드(관리 층)에 두고, **에이전트 프롬프트에는 절대 넣지 않는다.**

### 3.2 데이터 구조

baseline은 experience(쿼리 + trajectory + items) 단위로 저장한다. EFM은 **아이템 테이블과 experience 테이블을 분리**한다. 그래야 새 experience가 기존 아이템을 참조(재확인)할 수 있다.

```jsonc
// items.json : item_id -> item
{
  "it_0001": {
    // --- content layer (immutable) ---
    "title": "...",
    "description": "...",
    "content": "...",
    "source_label": "success",        // 추출 당시 judge 결과, 기록용

    // --- management layer (mutable) ---
    "created_at": 12,                 // 태스크 카운터(3.6절)
    "last_active": 40,                // 마지막 사용 또는 재확인 시점
    "n_retrieved": 5,                 // 주입된 태스크 수
    "n_used": 3,                      // 에이전트가 사용한다고 선언한 태스크 수
    "n_reconfirmed": 1,               // 중복으로 재추출된 횟수
    "group_id": null,
    "conflict_with": [],
    "pinned": false,
    "deleted": false,
    "deleted_reason": null            // "stale" | "declined"
  }
}

// experiences.json : 검색 키
{
  "ex_0001": {
    "query": "...",
    "trajectory_ref": "...",
    "item_ids": ["it_0001", "it_0002"]   // 기존 아이템 참조 포함 가능
  }
}

// 임베딩은 baseline처럼 별도 파일에 저장:
//   query 임베딩 (experience 단위), description 임베딩, content 임베딩 (item 단위)
```

### 3.3 검색 (태스크 시작 시)

1. baseline과 같은 방식으로 현재 query와 experience의 query 임베딩 cosine을 계산해 **후보 experience를 `K_CAND`개** 가져온다.
2. 후보 experience의 아이템을 펼치고, `deleted=true`인 아이템을 제외한다.
3. 각 아이템 점수: `score = cos(q, q_exp) + ETA * w`, 여기서 `w = GAMMA ** (now - last_active)`. 한 아이템이 여러 experience에 걸쳐 있으면 가장 높은 cos를 쓴다.
4. **그룹 정리**: 같은 `group_id`를 가진 아이템은 현재 query와 description 임베딩 cosine이 가장 높은 하나만 남긴다.
5. **충돌 쌍은 함께 주입**: `conflict_with` 관계인 두 아이템이 모두 후보에 남아 있으면 **둘 다 주입**한다. 어느 쪽을 따를지는 에이전트가 고르고, 그 선택이 3.4절의 used/declined로 기록되어 3.6절의 조기 삭제가 충돌을 판가름한다. 한쪽만 후보에 있으면 평소처럼 단독으로 다룬다. 둘 중 하나를 recency로 골라 버리면 새로 들어온 쪽(w=1)이 항상 이기고 기존 쪽은 비교될 기회 없이 사라지므로, 이 규칙을 바꾸지 않는다.
6. 점수순으로 **baseline 평균 주입 개수와 같은 수**만큼 주입한다. 충돌 쌍은 두 아이템 중 높은 점수를 기준으로 순위를 매기고 **예산 2개**를 차지한다. 쌍의 차례에 남은 예산이 1개뿐이면 그 쌍을 건너뛰고 다음 단독 아이템으로 채우며, 단독 아이템이 없으면 쌍 중 점수가 높은 하나만 주입한다. 주입 형식(title + content)은 baseline과 동일하게 유지하며, 충돌이라는 사실을 프롬프트에 따로 표시하지 않는다.
7. 주입된 모든 아이템에 `n_retrieved += 1`.

### 3.4 사용 이력 기록 (태스크 종료 시)

1. 에이전트의 추론 로그에서 주입된 각 아이템을 **사용했는지(used) 거부했는지(declined)** 판정한다.
   - 추론 로그가 아이템 번호나 제목을 언급하는 형식이면 규칙 기반으로 파싱한다.
   - 그렇지 않으면 백본 LLM에 한 번 호출해 JSON으로 분류한다: `{"it_0001": "used", "it_0002": "declined"}`.
   - 파싱이 불가능하면 `NOTES.md`에 기록하고, 모든 주입 아이템을 used로 간주하며 조기 삭제(3.6절 규칙 2)를 비활성화한다.
2. 한 태스크 안에서 여러 번 언급돼도 **태스크당 한 번만** 센다.
3. used인 아이템: `n_used += 1`, `last_active = now`.
4. 충돌 쌍이 함께 주입된 태스크는 `efm_events.jsonl`에 `{"type": "conflict_exposure", "pair": [id_a, id_b], "used": [...], "declined": [...], "judge": label}`로 기록한다. 논문의 사례 분석에 쓴다.

### 3.5 삽입: 관계 분류 (메모리 추출 직후)

baseline의 추출 로직으로 새 아이템들을 얻은 뒤, **각 새 아이템을 저장하기 전에** 다음을 수행한다.

1. 새 아이템의 description, content를 각각 임베딩한다(baseline과 같은 임베딩 모델).
2. 삭제되지 않은 기존 아이템 중 `s_C` 상위 `M_CAND`개와 `s_D` 상위 `M_CAND`개의 합집합을 후보로 삼는다. 같은 배치의 앞선 새 아이템도 후보에 포함한다.
3. 후보 각각과 `s_D = cos(e_D_new, e_D_old)`, `s_C = cos(e_C_new, e_C_old)`를 계산하고, 아래 순서로 **처음 해당하는 관계 하나**를 적용한다.

| 우선순위 | 조건 | 처리 |
|---|---|---|
| 1 | `s_D >= TAU_D` and `s_C >= TAU_C` (Duplicate) | 새 아이템 **저장 안 함**. 기존 아이템 `n_reconfirmed += 1`, `last_active = now`. 이번 experience의 `item_ids`에 기존 아이템 ID 추가 |
| 2 | `s_D < TAU_D` and `s_C >= TAU_C` (Group) | 새 아이템 저장. 기존 아이템에 `group_id`가 있으면 공유, 없으면 새로 만들어 둘 다에 부여 |
| 3 | `s_D >= TAU_D` and `s_C < TAU_C` (Conflict 후보) | NLI 판정(아래). `contradiction`이면 저장 후 서로의 `conflict_with`에 ID 추가. `complementary`면 Novel로 처리 |
| 4 | 어느 후보와도 해당 없음 (Novel) | 새 아이템 저장 |

Duplicate가 여러 후보와 성립하면 `s_C`가 가장 높은 하나에만 적용한다. Group과 Conflict는 해당하는 후보 모두에 적용한다.

새로 저장하는 아이템의 관리 필드 초기값: `created_at = last_active = now`, `n_retrieved = n_used = n_reconfirmed = 0`.

**NLI 판정 프롬프트** (백본 LLM, temperature 0):

```
Two strategies apply to the same situation.
Situation A: {desc_old}
Strategy A: {content_old}
Situation B: {desc_new}
Strategy B: {content_new}
Do the two strategies contradict each other (following one means not following the other),
or are they complementary (both can be followed or either works)?
Answer with JSON only: {"relation": "contradiction"} or {"relation": "complementary"}
```

### 3.6 망각 (각 태스크 종료 후, 사용 이력 기록 다음)

**시간 단위는 처리한 태스크 수**다. 전역 카운터 `now`를 태스크마다 1씩 올린다. (벤치마크에는 실제 시간 개념이 없기 때문이다.)

삭제되지 않았고 `pinned=false`이며 `now - created_at >= GRACE`인 아이템에 대해:

1. **Stale 삭제**: `now - last_active > T_STALE` 이면 삭제 (`deleted_reason="stale"`).
2. **조기 삭제**: `n_retrieved >= R_MIN` 이고 `n_used / n_retrieved < RHO` 이면 삭제 (`deleted_reason="declined"`).

충돌 해소에 별도 규칙은 없다. 함께 주입될 때마다 거부당하는 쪽이 규칙 2에 걸려 삭제되고, 비교 기회 자체가 없어진 쪽(오랫동안 해당 쿼리가 오지 않음)은 규칙 1로 정리된다. 충돌 쌍의 한쪽이 삭제되면 `efm_events.jsonl`에 `{"type": "conflict_resolved", "kept": id, "deleted": id, "reason": ..., "kept_is_newer": bool}`을 기록한다.

삭제는 `deleted=true`로 표시하고 파일에서 지우지 않는다(분석용). 삭제된 아이템을 가리키는 `conflict_with`는 정리하고, 그룹에 멤버가 하나만 남으면 `group_id`를 비운다. `item_ids`가 모두 삭제된 experience는 검색 후보에서 제외한다.

**제한적 품질 삭제**는 이번 실험에서는 구현만 하고 기본값은 끈다(`--efm_quality_delete false`): `n_used >= 5` 이고 used였던 최근 3개 태스크가 모두 judge 실패면 삭제.

### 3.7 하이퍼파라미터 기본값

모두 CLI 플래그로 노출하고, 실행한 값은 결과 파일에 함께 기록한다.

| 이름 | 기본값 | 설명 |
|---|---|---|
| `TAU_D` | 0.85 | description 유사도 임계값 |
| `TAU_C` | 0.85 | content 유사도 임계값 |
| `M_CAND` | 10 | 관계 분류 후보 수 |
| `K_CAND` | 5 | 검색 후보 experience 수 |
| `ETA` | 0.1 | 검색 점수에서 recency 가중치 |
| `GAMMA` | 0.98 | 태스크당 감쇠율 |
| `T_STALE` | 60 | stale 기준(태스크 수) |
| `GRACE` | 20 | 신규 아이템 보호 기간(태스크 수) |
| `R_MIN` | 3 | 조기 삭제 최소 검색 횟수 |
| `RHO` | 0.34 | 조기 삭제 사용률 기준 |

**임계값 보정**: 본 실험 전에 baseline 실행에서 쌓인 메모리로 모든 아이템 쌍의 `s_D`, `s_C` 분포를 계산해 `results/similarity_hist.png`와 백분위수(50/90/95/99)를 저장한다. 0.85가 분포상 지나치게 낮거나 높으면(예: 쌍의 30% 이상이 Duplicate로 분류됨) 95~99 백분위 근처로 조정하고 `NOTES.md`에 근거를 적는다. 단, **조정은 본 실험 전에 한 번만** 하고, 결과를 보고 다시 조정하지 않는다.

### 3.8 의사코드

```python
now = 0
for task in tasks:                      # baseline과 같은 순서
    items = efm.retrieve(task.query, now)          # 3.3
    traj = agent.run(task, items)                  # 변경 없음
    label = judge(traj)                            # 변경 없음
    efm.record_usage(items, traj, now)             # 3.4
    new_items = extract(traj, label)               # 변경 없음
    efm.insert(task.query, new_items, now)         # 3.5
    efm.forget(now)                                # 3.6
    now += 1
```

---

## 4. 실험 프로토콜

1. **벤치마크**: 저장소의 기본 설정을 따른다. 비용이 크면 먼저 ReasoningBank 논문이 scaling 분석에 쓴 **WebArena-Shopping 서브셋**으로 진행한다.
2. **공정성**: 두 모드 모두 같은 태스크 순서, 같은 모델, 같은 temperature/seed, 같은 임베딩 모델, 빈 메모리에서 시작, 같은 주입 개수.
3. **실행 목록** (우선순위 순):
   - `baseline` 1회
   - `efm` 1회 (기본값)
   - 예산이 허락하면 ablation: `efm --no_relation`(관계 분류 끄고 append만, 망각은 켬), `efm --no_forgetting`(관계 분류만)
   - 예산이 허락하면 각 모드를 다른 seed로 1~2회 더 반복
4. MaTTS는 이번에 사용하지 않는다(k=1).

---

## 5. Sanity check (본 실행 전 필수)

1. **동등성 테스트**: `efm` 모드에서 `TAU_D=TAU_C=1.01`(관계 분류가 절대 발동하지 않음), `ETA=0`, `T_STALE=R_MIN=10**9`로 설정하면, 처음 20개 태스크에서 **baseline과 주입되는 아이템이 동일해야 한다.** 다르면 원인을 찾아 고친다.
2. **단위 테스트**: 관계 분류 네 가지 경우, 망각 두 규칙, grace 보호, 그룹 정리를 가짜 임베딩으로 테스트한다. 충돌은 다음을 따로 확인한다: 쌍이 둘 다 후보이면 둘 다 주입되는지, 예산이 1개 남았을 때 건너뛰기 규칙이 맞게 동작하는지, 한쪽이 반복 거부되면 조기 삭제되고 `conflict_with`가 정리되는지.
3. **소규모 실행**: 두 모드를 10개 태스크로 끝까지 돌려 로그와 결과 파일이 정상 생성되는지 확인한다.
4. 결과와 예상 API 호출 수(태스크당 NLI 판정 수, 사용 판정 호출 수 포함)를 사용자에게 보고하고 승인받은 뒤 본 실행한다.

---

## 6. 산출물

```
results/
  summary.md               # 아래 표 + 사용한 하이퍼파라미터 + 실행 명령어
  baseline/                # 태스크별 결과, 메모리 최종 스냅샷
  efm/                     # 태스크별 결과, 메모리 최종 스냅샷
  efm_events.jsonl         # 태스크별 관계 분류 결과, 삭제 이벤트, 사용 판정
  similarity_hist.png
CODE_MAP.md
NOTES.md
README_EFM.md              # 두 모드 실행 방법
```

`results/summary.md`의 표:

| Mode | SR | ΔSR vs baseline | Avg. steps | Final #items (active) | #Dup | #Group | #Conflict | #Novel | #Deleted (stale / declined) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | | – | | | – | – | – | – | – |
| efm | | | | | | | | | |

추가로 태스크 진행에 따른 **활성 아이템 수 곡선**(baseline vs efm)을 `results/memory_size.png`로 저장한다.

충돌 통계도 `summary.md`에 적는다: 생성된 충돌 쌍 수, 해소된 쌍 수, 해소 시 새 아이템이 남은 비율과 기존 아이템이 남은 비율, 미해소 쌍 수.

---

## 7. 하지 말 것

- 에이전트 프롬프트, 추출 프롬프트, judge, 모델, 임베딩 모델 변경
- 저장된 아이템의 title/description/content를 수정하거나 LLM으로 병합·요약하는 코드
- 관리 층 필드를 에이전트 프롬프트에 노출
- 결과를 본 뒤 하이퍼파라미터를 다시 조정해 재실행 (조정했다면 반드시 `NOTES.md`에 명시)
- `baseline-original` 태그 또는 원본 복사본 수정