import glob, json, os

ms = [json.loads(l) for l in open("memories_reasoningbank/shopping.jsonl", encoding="utf-8")]
one, multi = [], []
for m in ms:
    tid = m["task_id"]
    n = len(m.get("action_list", []))
    ok = m.get("status") == "success"
    (one if n <= 1 else multi).append(ok)

f = lambda g: f"{sum(g)}/{len(g)} = {sum(g)/len(g):.1%}" if g else "없음"
print("1스텝 태스크 성공률 :", f(one))
print("2스텝 이상 성공률   :", f(multi))