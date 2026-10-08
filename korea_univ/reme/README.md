# reme (Remember Me, Refine Me)

arXiv:2512.10696의 알고리즘을 이 코드베이스에 맞게 재구현(원 코드 `agentscope-ai/ReMe`는
BFCL-V3/AppWorld tool-call 루프 전용이라 WebArena/Mind2Web에 그대로 꽂을 수 없어 재구현 —
`NOTES.md` 참고). 성공/실패/비교(comparative) 세 갈래로 경험을 추출하고, 검색은 각 경험의
**usage scenario** 임베딩으로 색인, 사용 통계(`u`/`f`) 기반으로 utility가 낮은 경험을 삭제(본문은
수정하지 않음). 지원 벤치마크: **WebArena / Mind2Web** 둘 다.

```bash
./run_webarena.sh     # Docker 쇼핑몰 사이트 필요
./run_mind2web.sh     # Docker 불필요
```

결과는 `memories_reme_cmp/shopping/items.json`에서 경험별 `n_retrieved`(논문의 f),
`n_success`(u), `facet`(success/failure/comparative), `deleted_reason`을 확인할 수 있습니다.
`*_events.jsonl`에 credit/retrieve/delete 이벤트가 시간순으로 기록됩니다.

재구현 범위에서 제외한 것(둘 다 원 논문도 optional이거나, arm 간 실행 예산을 오염시키는 요소):
rerank/rewrite, 실패 시 최대 3회 self-reflection 재시도, 그리고 comparative 짝짓기는 원 논문의
"같은 태스크 8회 샘플링" 대신 "같은 태스크군의 가장 최근 반대 결과"로 단순화.
