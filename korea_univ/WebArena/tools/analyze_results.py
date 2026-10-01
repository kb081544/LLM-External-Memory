"""ReasoningBank WebArena 결과 분석 스크립트.

사용법:
    python analyze_results.py --results_dir "D:\\results" --memory_file "memories_reasoningbank/shopping.jsonl"
"""

import argparse
import json
import os
import re
from collections import Counter


def load_task_results(results_dir):
    """각 태스크 폴더에서 gt, rm, 스텝 수를 수집."""
    rows = []
    for name in os.listdir(results_dir):
        if not name.startswith("webarena."):
            continue
        path = os.path.join(results_dir, name)
        if not os.path.isdir(path):
            continue

        tid = int(name.split(".")[-1])
        row = {"task_id": tid, "gt": None, "rm": None, "steps": None, "thoughts": ""}

        # 스텝 수: step_N.pkl.gz 파일 개수
        row["steps"] = len([f for f in os.listdir(path)
                            if re.match(r"step_\d+\.pkl\.gz$", f)])

        # autoeval 결과 (모델명이 파일명에 들어가므로 패턴으로 탐색)
        for f in os.listdir(path):
            if f.endswith("_autoeval.json"):
                try:
                    data = json.load(open(os.path.join(path, f), encoding="utf-8"))
                    if isinstance(data, list) and data:
                        row["gt"] = data[0].get("gt")
                        row["rm"] = data[0].get("rm")
                        row["thoughts"] = data[0].get("thoughts") or ""
                except Exception as e:
                    print(f"  [warn] {name}/{f} 읽기 실패: {e}")
                break

        rows.append(row)

    return sorted(rows, key=lambda r: r["task_id"])


def load_memories(memory_file):
    """메모리 jsonl 로드."""
    if not os.path.exists(memory_file):
        print(f"[warn] 메모리 파일 없음: {memory_file}")
        return []
    out = []
    with open(memory_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def truthy(v):
    """rm 값이 성공을 뜻하는지 판정 (True / 1 / 1.0 / 'true' 모두 허용)."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v == 1
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "success")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--memory_file", default="memories_reasoningbank/shopping.jsonl")
    ap.add_argument("--log_file", default=None, help="run_log.txt (메모리 재사용 집계용)")
    args = ap.parse_args()

    rows = load_task_results(args.results_dir)
    mems = load_memories(args.memory_file)

    print("=" * 60)
    print(f"완료 태스크: {len(rows)}개   축적 메모리: {len(mems)}개")
    print("=" * 60)

    # ---------- 1. 성공률 ----------
    rm_vals = [truthy(r["rm"]) for r in rows]
    rm_ok = sum(1 for v in rm_vals if v is True)
    rm_bad = sum(1 for v in rm_vals if v is False)
    rm_null = sum(1 for v in rm_vals if v is None)

    gt_vals = [r["gt"] for r in rows if r["gt"] is not None]
    gt_ok = sum(1 for v in gt_vals if v == 1.0)

    print("\n[1] 성공률")
    if rm_ok + rm_bad:
        print(f"  autoeval(rm) : {rm_ok}/{rm_ok + rm_bad} = {rm_ok / (rm_ok + rm_bad):.1%}")
    if gt_vals:
        print(f"  규칙기반(gt)  : {gt_ok}/{len(gt_vals)} = {gt_ok / len(gt_vals):.1%}")
    if rm_null:
        print(f"  rm 결측       : {rm_null}개 (평가 실패)")

    # ---------- 2. 전반 vs 후반 (메모리 누적 효과) ----------
    valid = [r for r in rows if truthy(r["rm"]) is not None]
    if len(valid) >= 10:
        half = len(valid) // 2
        first, second = valid[:half], valid[half:]

        def rate(g):
            n = sum(1 for r in g if truthy(r["rm"]))
            return n / len(g)

        def avg_steps(g, only_success=False):
            sel = [r for r in g if r["steps"] and (not only_success or truthy(r["rm"]))]
            return sum(r["steps"] for r in sel) / len(sel) if sel else 0

        print("\n[2] 전반부 vs 후반부 (메모리 누적 효과)")
        print(f"  전반 {len(first)}개: 성공률 {rate(first):.1%}, "
              f"평균 {avg_steps(first):.1f}스텝 (성공만 {avg_steps(first, True):.1f})")
        print(f"  후반 {len(second)}개: 성공률 {rate(second):.1%}, "
              f"평균 {avg_steps(second):.1f}스텝 (성공만 {avg_steps(second, True):.1f})")

    # ---------- 3. gt vs rm 불일치 ----------
    mismatch = [r for r in rows
                if r["gt"] is not None and truthy(r["rm"]) is not None
                and (r["gt"] == 1.0) != truthy(r["rm"])]
    print(f"\n[3] gt-rm 불일치: {len(mismatch)}건")
    for r in mismatch[:5]:
        print(f"  task {r['task_id']}: gt={r['gt']} rm={r['rm']}")
        if r["thoughts"]:
            print(f"    판정근거: {r['thoughts'][:150]}...")

    # ---------- 4. 메모리 내용 ----------
    print("\n[4] 메모리 분석")
    status_cnt = Counter(m.get("status") for m in mems)
    print(f"  상태 분포: {dict(status_cnt)}")

    titles = []
    for m in mems:
        for item in m.get("memory_items", []):
            mt = re.search(r"## Title\s*(.+)", item)
            if mt:
                titles.append(mt.group(1).strip())
    print(f"  추출된 메모리 아이템: {len(titles)}개")

    print("\n  자주 등장한 키워드:")
    words = Counter()
    for t in titles:
        for w in re.findall(r"[A-Za-z]{4,}", t.lower()):
            words[w] += 1
    for w, c in words.most_common(12):
        print(f"    {w:20s} {c}")

    print("\n  최근 메모리 제목 5개:")
    for t in titles[-5:]:
        print(f"    - {t}")

    # ---------- 5. 빈 think_list 비율 ----------
    empty_think = sum(1 for m in mems
                      if all(not t.strip() for t in m.get("think_list", [])))
    if mems:
        print(f"\n[5] think_list가 비어있는 궤적: {empty_think}/{len(mems)} "
              f"({empty_think / len(mems):.0%})")
        if empty_think > len(mems) * 0.5:
            print("    → 사고 과정 파싱이 대부분 실패. 모델 출력 형식 확인 필요.")

    # ---------- 6. 메모리 재사용 (로그 기반) ----------
    if args.log_file and os.path.exists(args.log_file):
        try:
            text = open(args.log_file, encoding="utf-8", errors="ignore").read()
        except Exception:
            text = ""
        uses = len(re.findall(r"Memory Item \d+", text))
        print(f"\n[6] 로그 내 'Memory Item N' 언급: {uses}회")

    # ---------- 7. 스텝 분포 ----------
    steps = [r["steps"] for r in rows if r["steps"]]
    if steps:
        steps_sorted = sorted(steps)
        print(f"\n[7] 스텝 수  최소 {min(steps)} / 중앙값 "
              f"{steps_sorted[len(steps) // 2]} / 최대 {max(steps)}")
        long_tasks = [r for r in rows if r["steps"] and r["steps"] >= 15]
        if long_tasks:
            ids = ", ".join(str(r["task_id"]) for r in long_tasks[:10])
            print(f"  15스텝 이상 (헤맨 태스크): {ids}")


if __name__ == "__main__":
    main()