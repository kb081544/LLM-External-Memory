import glob, json, os

ms = [json.loads(l) for l in open("memories_reasoningbank/shopping.jsonl", encoding="utf-8")]
print(f"{'task':>6} {'act':>4} {'think':>6} {'shot':>5}  diff")
odd = 0
for m in ms:
    tid = m["task_id"]
    n_act = len(m.get("action_list", []))
    n_th = sum(1 for t in m.get("think_list", []) if t.strip())
    n_sh = len(glob.glob(f"D:/results/webarena.{tid}/screenshot_step_*.png"))
    d = n_sh - n_act
    flag = "" if d == 1 else "  <-- 확인"
    if d != 1:
        odd += 1
    print(f"{tid:>6} {n_act:>4} {n_th:>6} {n_sh:>5}  {d:+d}{flag}")
print(f"\n총 {len(ms)}건 중 diff!=1 인 것: {odd}건")