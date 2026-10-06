# efm (Edit-Free Memory Lifecycle) — 우리 연구

삽입 시점에 설명(D)/내용(C) 임베딩 유사도로 아이템을 Duplicate/Group/Conflict/Novel로 분류하고,
검색 시점에 그룹 collapse + 충돌 co-injection을 하고, 사용 증거 기반으로 오래되거나 안 쓰이는
메모리를 삭제(forgetting). 지원 벤치마크: **WebArena**, **Mind2Web**.

```bash
./run_webarena.sh
./run_mind2web.sh
```

결과는 `memories_efm_cmp/<site>/items.json`에서 아이템별 `n_used`/`n_retrieved`/`group_id`/
`conflict_with`/`deleted_reason`을 직접 확인할 수 있습니다.
