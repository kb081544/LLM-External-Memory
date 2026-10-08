# memp (Memp)

arXiv:2508.06433의 알고리즘을 이 코드베이스에 맞게 재구현(원 코드 `zjunlp/MemP`는
ALFWorld/TravelPlanner 루프 전용이라 WebArena/Mind2Web에 그대로 꽂을 수 없어 재구현 —
`NOTES.md` 참고). 절차(procedure) 메모리 — BUILD(script / trajectory / proceduralization) ×
UPDATE(vanilla / validation / adjustment) 두 축을 플래그로 전환. 기본값은 논문이 가장 좋다고
보고한 조합: BUILD=proceduralization, UPDATE=adjustment(실패 시 주입됐던 1위 절차를 **in-place로
재작성**). 지원 벤치마크: **WebArena / Mind2Web** 둘 다.

```bash
./run_webarena.sh     # Docker 쇼핑몰 사이트 필요
./run_mind2web.sh     # Docker 불필요
```

결과는 `memories_memp_cmp/shopping/procedures.json`에서 `n_revisions`/`revised_by`(어느
태스크가 그 절차를 in-place로 고쳤는지)를 확인할 수 있습니다. `*_events.jsonl`에
append/adjust/retrieve 이벤트가 시간순으로 기록됩니다.

재구현 범위에서 제외한 것: Facts/AveFact 검색 키(기본값인 Key=Query만 포팅).
