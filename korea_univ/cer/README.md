# cer — Contextual Experience Replay

- 논문: [Contextual Experience Replay for Self-Improvement of Language Agents](https://arxiv.org/abs/2506.06698)
  (arXiv:2506.06698, ACL 2025)
- 코드: **공개 저장소 없음** → 논문 본문(3.1~3.3)과 **부록 A.1의 프롬프트 원문(Figure 3~6)**을
  그대로 옮겨 재구현했습니다. 프롬프트는 `WebArena/prompts/memory_instruction.py`의 `CER_*`
  상수에 있고, Mind2Web 쪽(`Mind2Web/memory_instruction.py`)은 바이트 단위로 동일한 복사본입니다.
- 구현: `WebArena/cer/`, `Mind2Web/cer/` (`buffer.py`, `distill.py`, `retrieve.py`)

## 왜 비교 대상인가

이 비교군 중 **유일하게 WebArena에서 직접 평가된 논문**입니다(논문 Table 1: CER_hybrid 평균 36.7,
GPT-4o+BrowserGym 베이스라인 24.3) — 도메인이 우리와 정확히 일치합니다.

구조적으로도 다른 arm들과 뚜렷이 구분됩니다:

| | reasoningbank / efm / reme / memp | **cer** |
|---|---|---|
| 경험 표현 | 단일 종류(아이템/경험/절차) | **dynamics**(페이지 요약+URL) + **skills**(하위목표 절차) 2종, 각자 별도 모듈 |
| 검색 | 임베딩 코사인 유사도 top-k | **LLM이 버퍼 전체를 보고 top-k를 직접 선택** |
| 중복 제거 | 임베딩 유사도 임계값 | **프롬프트 수준** — 기존 버퍼를 보여주고 "Summarized before"로 답하게 함 |

검색이 임베딩이 아니라 LLM 선택이라는 점이 핵심 차이이므로, 공용 임베딩 검색기로 바꾸지 않고
논문 그대로 두었습니다. 대신 태스크당 LLM 호출이 추출 2회 + 검색 2회 늘어납니다.

## 설정 선택

논문에는 offline / online / hybrid 세 설정이 있습니다. 여기서는 **online**을 씁니다 — 다른 arm들이
전부 자기 실행 궤적만으로 온라인 자기개선을 하므로, 사람이 주석한 gold trajectory를 추가로 쓰는
offline/hybrid는 동일 조건 비교가 아닙니다. 성공/실패 궤적 **둘 다**에서 추출합니다(논문의 기본
설정 — 성공만 쓰는 건 별도 ablation `CER_success`, §5.6).

## 벤치마크별 차이 (의도된 것)

- **WebArena**: dynamics + skills 전체 (k_d = k_s = 5, 논문 4.1.1과 동일).
- **Mind2Web**: **skills만.** 이미 캡처된 스텝을 오프라인 리플레이하는 구조라 스텝별 URL이 없어서
  dynamics 경험이 가리킬 대상 자체가 없습니다. 논문 자신의 **"CER − dynamics" ablation**(§5.7,
  Forum split에서 35.1 vs 전체 37.7)과 같은 구성이라 임의 변형이 아닙니다.

## 알려진 수정 사항 (충실도 관련)

논문 부록의 프롬프트는 스크래치 사고 공간을 `<<think>>\nthink step by step\n<</think>>` 블록으로
요구합니다. 이 조합(`think`라는 태그 안에 단계적 사고를 명시적으로 요구)이 Claude CLI 백본에서
**결정론적으로 거부**됩니다(`stop_reason: refusal`, 분류기 태그 `reasoning_extraction` — 격리
테스트로 8/8 거부, 블록 제거 후 8/8 통과 확인). 네 프롬프트 전부에서 이 블록을 포맷 지시와 예시
양쪽에서 제거했습니다 — `cer/distill.py`가 애초에 `<<think>>` 내용을 파싱하지 않으므로, 실제
추출 결과(스킬/스텝, URL/페이지요약)는 전혀 손실되지 않습니다. 다른 백본(OpenAI, Gemini 등)을
쓴다면 이 문제가 재현되지 않을 수 있습니다.

## 실행

```bash
./run_webarena.sh     # Docker 쇼핑몰 사이트 필요
./run_mind2web.sh     # Docker 불필요
```

결과 확인: `memories_cer_cmp/<site>/buffer.json`에 `dynamics`/`skills` 섹션이 쌓이고, 각 항목의
`n_retrieved`로 실제로 몇 번 replay됐는지 볼 수 있습니다.
