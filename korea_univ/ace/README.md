# ace (Agentic Context Engineering)

arXiv:2510.04618의 알고리즘을 이 코드베이스에 맞게 재구현(원 코드 `ace-agent/ace`는 단일 QA
형식 전용이라 WebArena의 멀티스텝 액션 루프에 그대로 꽂을 수 없어 재구현 — `NOTES.md` 참고).
Reflector가 통찰을 bullet로 추출 → 기존 playbook과 임베딩 유사도(EFM과 동일한 방식)로 비교해
merge(helpful/harmful 투표) 또는 신규 추가 → 전체 playbook을 매 태스크 주입(크기 상한 적용).
지원 벤치마크: **WebArena / Mind2Web** 둘 다.

```bash
./run_webarena.sh     # Docker 쇼핑몰 사이트 필요
./run_mind2web.sh     # Docker 불필요
```

결과는 `WebArena/memories_ace_cmp/shopping/playbook.json` (또는 Mind2Web 쪽
`Mind2Web/memories_ace_cmp/shopping/playbook.json`)에서 각 bullet의 `helpful`/`harmful` 투표
수를 직접 확인할 수 있습니다.
