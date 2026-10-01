import json

ms = [json.loads(l) for l in open("memories_reasoningbank/shopping.jsonl", encoding="utf-8")]
one = [m for m in ms if len(m.get("action_list", [])) <= 1]
print(f"1스텝 태스크 {len(one)}건\n")
for m in one[:8]:
    th = " ".join(m.get("think_list", []))
    print(f"--- task {m['task_id']} ---")
    print("Q:", m.get("query", "")[:100])
    print("사고:", th[:400].replace("\n", " "))
    print("액션:", (m.get("action_list") or [""])[0][:150])
    print()