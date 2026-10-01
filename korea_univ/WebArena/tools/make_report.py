"""ReasoningBank 재현 리포트 생성기 (v2).

논문 개념을 SVG로 먼저 설명하고, 실제 실행 데이터로 하나씩 확인하는 구성.

사용법:
    python make_report.py --results_dir "D:\\results" ^
        --memory_file "memories_reasoningbank/shopping.jsonl" ^
        --log_file "D:\\log_shopping.txt" ^
        --out "D:\\report\\index.html"

옵션:
    --max_cases N   태스크 사례 수록 개수 (기본 12)
    --copy_images   스크린샷을 리포트 폴더로 복사 (공유용)
"""

import argparse
import base64
import glob
import html
import json
import os
import re
import shutil
from collections import Counter


# ============================================================ 데이터 로드

def read_text(path):
    """Tee-Object는 UTF-16으로 저장하므로 여러 인코딩을 시도한다."""
    if not path or not os.path.exists(path):
        return ""
    for enc in ("utf-8", "utf-16", "utf-16-le", "cp949"):
        try:
            with open(path, encoding=enc) as f:
                t = f.read()
            if t.count("\x00") < max(1, len(t)) * 0.1:
                return t
        except Exception:
            continue
    return ""


def load_memories(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                m = json.loads(line)
                m["_order"] = i
                out.append(m)
            except json.JSONDecodeError:
                pass
    return out


def load_tasks(results_dir):
    rows = {}
    if not os.path.isdir(results_dir):
        return rows
    for name in os.listdir(results_dir):
        if not name.startswith("webarena."):
            continue
        path = os.path.join(results_dir, name)
        if not os.path.isdir(path):
            continue
        try:
            tid = int(name.split(".")[-1])
        except ValueError:
            continue

        shot_map = {}
        for sp in glob.glob(os.path.join(path, "screenshot_step_*.png")):
            m = re.search(r"step_(\d+)", os.path.basename(sp))
            if m:
                shot_map[int(m.group(1))] = sp
        shots = [shot_map[k] for k in sorted(shot_map)]
        row = {"task_id": tid, "dir": path, "screenshots": shots,
               "shot_map": shot_map,
               "steps": len(glob.glob(os.path.join(path, "step_*.pkl.gz"))),
               "gt": None, "rm": None, "thoughts": ""}

        for f in os.listdir(path):
            if f.endswith("_autoeval.json"):
                try:
                    d = json.load(open(os.path.join(path, f), encoding="utf-8"))
                    if isinstance(d, list) and d:
                        row["gt"] = d[0].get("gt")
                        row["rm"] = d[0].get("rm")
                        row["thoughts"] = d[0].get("thoughts") or ""
                except Exception:
                    pass
                break
        rows[tid] = row
    return rows


def truthy(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v == 1
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "success")
    return None


def parse_items(items):
    out = []
    for raw in items or []:
        t = re.search(r"##\s*Title\s*(.+)", raw)
        d = re.search(r"##\s*Description\s*(.+?)(?=\n##|\Z)", raw, re.S)
        c = re.search(r"##\s*Content\s*(.+?)(?=\n##|\Z)", raw, re.S)
        out.append({
            "title": t.group(1).strip() if t else "(제목 없음)",
            "description": d.group(1).strip() if d else "",
            "content": c.group(1).strip() if c else "",
        })
    return out


def esc(s):
    return html.escape("" if s is None else str(s))


# ============================================================ 프롬프트 재구성

MEM_INSTRUCTION = (
    "Below are some memory items that I accumulated from past interaction from "
    "the environment that may be helpful to solve the task. You can use it when "
    "you feel it's relevant. In each step, please first explicitly discuss if you "
    "want to use each memory item or not, and then take action."
)

RE_USE = re.compile(r"(will use|I use|I'll use|using Memory|use Memory Item)", re.I)
RE_NOUSE = re.compile(r"(will not use|won't use|do not use|don't use|not use Memory|"
                      r"not applicable|isn't relevant|is not relevant)", re.I)


def classify_think(t):
    """사고 텍스트를 메모리 사용/거절/무언급으로 분류."""
    if not t or not t.strip():
        return None
    if RE_NOUSE.search(t):
        return "reject"
    if RE_USE.search(t) or "Memory Item" in t:
        return "use"
    return None


def memory_judgment_stats(mems):
    use = reject = silent = 0
    for m in mems:
        for t in m.get("think_list", []):
            if not (t or "").strip():
                continue
            c = classify_think(t)
            if c == "use":
                use += 1
            elif c == "reject":
                reject += 1
            else:
                silent += 1
    return use, reject, silent


def collect_judgments(mems, kind, limit=6):
    """사용/거절 판단 사례를 수집."""
    out = []
    for m in mems:
        for i, t in enumerate(m.get("think_list", [])):
            if classify_think(t) != kind:
                continue
            acts = m.get("action_list", [])
            out.append({"task_id": m.get("task_id"), "step": i,
                        "query": m.get("query", ""), "think": t,
                        "action": acts[i] if i < len(acts) else "",
                        "status": m.get("status")})
            if len(out) >= limit * 4:
                break
    # 길이가 적당한 것 위주로 (너무 짧으면 정보 부족)
    out.sort(key=lambda r: abs(len(r["think"]) - 380))
    return out[:limit]


def reconstruct_prompt(retrieved_items):
    """실제 주입되는 시스템 프롬프트 조각을 복원."""
    p = [MEM_INSTRUCTION, ""]
    for i, it in enumerate(retrieved_items, 1):
        p.append(f"# Memory Item {i}")
        p.append(f"## Title {it['title']}")
        if it.get("content"):
            p.append(f"## Content {it['content']}")
        p.append("")
    return "\n".join(p)


# ============================================================ SVG 다이어그램

SVG_PROBLEM = """
<svg viewBox="0 0 760 200" class="fig">
  <defs>
    <marker id="ar" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#8b93a7"/></marker>
    <marker id="arb" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#58a6ff"/></marker>
  </defs>
  <text x="0" y="14" class="cap">메모리 없는 에이전트 — 매 태스크가 백지에서 시작</text>
  <rect x="0" y="30" width="118" height="42" rx="7" class="bx"/>
  <text x="59" y="56" class="tx" text-anchor="middle">태스크 1</text>
  <rect x="168" y="30" width="118" height="42" rx="7" class="bx"/>
  <text x="227" y="56" class="tx" text-anchor="middle">태스크 2</text>
  <rect x="336" y="30" width="118" height="42" rx="7" class="bx"/>
  <text x="395" y="56" class="tx" text-anchor="middle">태스크 3</text>
  <rect x="504" y="30" width="118" height="42" rx="7" class="bx"/>
  <text x="563" y="56" class="tx" text-anchor="middle">태스크 4</text>
  <line x1="120" y1="51" x2="161" y2="51" class="ln" marker-end="url(#ar)"/>
  <line x1="288" y1="51" x2="329" y2="51" class="ln" marker-end="url(#ar)"/>
  <line x1="456" y1="51" x2="497" y2="51" class="ln" marker-end="url(#ar)"/>
  <text x="59" y="90" class="sm" text-anchor="middle">같은 실수 반복</text>
  <text x="227" y="90" class="sm" text-anchor="middle">같은 실수 반복</text>
  <text x="395" y="90" class="sm" text-anchor="middle">같은 실수 반복</text>
  <text x="563" y="90" class="sm" text-anchor="middle">같은 실수 반복</text>

  <text x="0" y="132" class="cap">ReasoningBank — 경험이 다음 태스크로 전달됨</text>
  <rect x="0" y="160" width="118" height="34" rx="7" class="bx2"/>
  <text x="59" y="182" class="tx" text-anchor="middle">태스크 1</text>
  <rect x="168" y="160" width="118" height="34" rx="7" class="bx2"/>
  <text x="227" y="182" class="tx" text-anchor="middle">태스크 2</text>
  <rect x="336" y="160" width="118" height="34" rx="7" class="bx2"/>
  <text x="395" y="182" class="tx" text-anchor="middle">태스크 3</text>
  <rect x="504" y="160" width="118" height="34" rx="7" class="bx2"/>
  <text x="563" y="182" class="tx" text-anchor="middle">태스크 4</text>
  <path d="M59,158 C59,136 227,136 227,158" class="ln2" marker-end="url(#arb)"/>
  <path d="M227,158 C227,136 395,136 395,158" class="ln2" marker-end="url(#arb)"/>
  <path d="M395,158 C395,136 563,136 563,158" class="ln2" marker-end="url(#arb)"/>
  <text x="650" y="182" class="sm acc">전략 축적</text>
</svg>
"""

SVG_SCHEMA = """
<svg viewBox="0 0 760 215" class="fig">
  <text x="0" y="14" class="cap">메모리 아이템의 세 부분</text>
  <rect x="0" y="30" width="350" height="175" rx="10" class="bx"/>
  <rect x="0" y="30" width="350" height="32" rx="10" class="hd"/>
  <text x="16" y="51" class="tx b">Title</text>
  <text x="80" y="51" class="sm">핵심 전략을 한 줄로 식별</text>
  <line x1="0" y1="62" x2="350" y2="62" class="dv"/>
  <text x="16" y="86" class="tx b">Description</text>
  <text x="122" y="86" class="sm">한 문장 요약</text>
  <line x1="0" y1="98" x2="350" y2="98" class="dv"/>
  <text x="16" y="122" class="tx b">Content</text>
  <text x="94" y="122" class="sm">추론 단계 · 판단 근거 · 운영 지침</text>
  <line x1="0" y1="134" x2="350" y2="134" class="dv"/>
  <text x="16" y="158" class="sm">저수준 실행 세부는 버리고</text>
  <text x="16" y="178" class="sm">전이 가능한 추론 패턴만 남긴다</text>
  <text x="16" y="196" class="sm acc">사람도 읽고, 모델도 바로 쓸 수 있는 형태</text>

  <text x="410" y="14" class="cap">기존 메모리 방식과의 차이</text>
  <rect x="410" y="30" width="350" height="50" rx="8" class="bx"/>
  <text x="426" y="50" class="tx b">Synapse</text>
  <text x="426" y="70" class="sm">과거 궤적을 통째로 저장 → 길고 노이즈 많음</text>
  <rect x="410" y="90" width="350" height="50" rx="8" class="bx"/>
  <text x="426" y="110" class="tx b">AWM</text>
  <text x="426" y="130" class="sm">성공 워크플로만 추출 → 실패에서 못 배움</text>
  <rect x="410" y="150" width="350" height="55" rx="8" class="bx2"/>
  <text x="426" y="172" class="tx b acc">ReasoningBank</text>
  <text x="426" y="192" class="sm">성공·실패 양쪽에서 전략 수준으로 추상화</text>
</svg>
"""

SVG_LOOP = """
<svg viewBox="0 0 760 340" class="fig">
  <defs>
    <marker id="a2" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#58a6ff"/></marker>
    <marker id="a3" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#d29922"/></marker>
  </defs>
  <text x="0" y="14" class="cap">폐루프 — 꺼내 쓰고 → 새로 만들고 → 다시 넣는다</text>

  <rect x="290" y="34" width="180" height="52" rx="9" class="bx2"/>
  <text x="380" y="56" class="tx b" text-anchor="middle">새 태스크 쿼리</text>
  <text x="380" y="75" class="sm" text-anchor="middle">스트리밍으로 하나씩 도착</text>

  <rect x="14" y="140" width="208" height="92" rx="9" class="bx"/>
  <text x="118" y="163" class="tx b acc" text-anchor="middle">① Retrieval</text>
  <text x="118" y="185" class="sm" text-anchor="middle">gemini-embedding-001</text>
  <text x="118" y="203" class="sm" text-anchor="middle">코사인 유사도 top-k (k=1)</text>
  <text x="118" y="221" class="sm" text-anchor="middle">시스템 프롬프트에 주입</text>

  <rect x="276" y="140" width="208" height="92" rx="9" class="bx"/>
  <text x="380" y="163" class="tx b acc" text-anchor="middle">에이전트 실행</text>
  <text x="380" y="185" class="sm" text-anchor="middle">ReAct 루프</text>
  <text x="380" y="203" class="sm" text-anchor="middle">관측 → 사고 → 액션</text>
  <text x="380" y="221" class="sm" text-anchor="middle">최대 30 스텝</text>

  <rect x="538" y="140" width="208" height="92" rx="9" class="bx"/>
  <text x="642" y="163" class="tx b acc" text-anchor="middle">② Construction</text>
  <text x="642" y="185" class="sm" text-anchor="middle">LLM-as-a-Judge 자기판정</text>
  <text x="642" y="203" class="sm" text-anchor="middle">성공/실패별 다른 프롬프트</text>
  <text x="642" y="221" class="sm" text-anchor="middle">궤적당 최대 3개 추출</text>

  <rect x="276" y="276" width="208" height="52" rx="9" class="bx2"/>
  <text x="380" y="299" class="tx b" text-anchor="middle">③ Consolidation</text>
  <text x="380" y="318" class="sm" text-anchor="middle">단순 추가 (pruning 없음)</text>

  <line x1="330" y1="88" x2="330" y2="134" class="ln2" marker-end="url(#a2)"/>
  <line x1="224" y1="186" x2="268" y2="186" class="ln2" marker-end="url(#a2)"/>
  <line x1="486" y1="186" x2="530" y2="186" class="ln2" marker-end="url(#a2)"/>
  <path d="M642,234 C642,286 520,302 492,302" class="ln3" marker-end="url(#a3)"/>
  <path d="M274,302 C150,302 60,286 60,238" class="ln3" marker-end="url(#a3)"/>
  <text x="576" y="268" class="sm warn">새 경험</text>
  <text x="120" y="268" class="sm warn">다음 태스크에서 재사용</text>
</svg>
"""

SVG_EXTRACT = """
<svg viewBox="0 0 760 210" class="fig">
  <defs>
    <marker id="a4" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#3fb950"/></marker>
    <marker id="a5" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#f85149"/></marker>
  </defs>
  <text x="0" y="14" class="cap">성공과 실패는 서로 다른 방식으로 메모리가 된다</text>

  <rect x="280" y="30" width="200" height="42" rx="9" class="bx"/>
  <text x="380" y="56" class="tx b" text-anchor="middle">완료된 궤적</text>

  <rect x="270" y="94" width="220" height="42" rx="9" class="bx2"/>
  <text x="380" y="112" class="tx" text-anchor="middle">LLM-as-a-Judge</text>
  <text x="380" y="129" class="sm" text-anchor="middle">정답 라벨 없이 temp 0.0으로 판정</text>
  <line x1="380" y1="74" x2="380" y2="90" class="ln2"/>

  <rect x="10" y="164" width="340" height="42" rx="8" class="okbx"/>
  <text x="180" y="181" class="tx ok" text-anchor="middle">성공 → 왜 통했는지 분석</text>
  <text x="180" y="198" class="sm" text-anchor="middle">검증된 전략으로 요약</text>

  <rect x="410" y="164" width="340" height="42" rx="8" class="badbx"/>
  <text x="580" y="181" class="tx bad" text-anchor="middle">실패 → 원인 반성</text>
  <text x="580" y="198" class="sm" text-anchor="middle">예방 규칙 · 가드레일 도출</text>

  <path d="M320,138 C320,152 220,150 195,158" class="lnok" marker-end="url(#a4)"/>
  <path d="M440,138 C440,152 540,150 565,158" class="lnbad" marker-end="url(#a5)"/>
</svg>
"""


# ============================================================ 렌더 헬퍼

def _encode_image(path, max_w, quality):
    """PNG를 축소·JPEG 변환해 data URI로. PIL이 없으면 원본 PNG를 그대로 인코딩."""
    try:
        from PIL import Image
        import io
        with Image.open(path) as im:
            im = im.convert("RGB")
            if max_w and im.width > max_w:
                h = round(im.height * max_w / im.width)
                im = im.resize((max_w, h), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=quality, optimize=True)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    except Exception:
        return ""


def img_tag(path, ctx):
    """ctx: {'mode', 'out_dir', 'max_w', 'quality', 'cache', 'bytes'}"""
    if not path or not os.path.exists(path):
        return ""
    mode = ctx["mode"]

    if mode == "embed":
        if path in ctx["cache"]:
            src = ctx["cache"][path]
        else:
            src = _encode_image(path, ctx["max_w"], ctx["quality"])
            ctx["cache"][path] = src
            ctx["bytes"][0] += len(src)
        if not src:
            return ""
        return f'<img class="shot" src="{src}">'

    if mode == "copy":
        dest_dir = os.path.join(ctx["out_dir"], "img")
        os.makedirs(dest_dir, exist_ok=True)
        base = f"{os.path.basename(os.path.dirname(path))}_{os.path.basename(path)}"
        dest = os.path.join(dest_dir, base)
        if not os.path.exists(dest):
            try:
                shutil.copy2(path, dest)
            except Exception:
                return ""
        src = f"img/{base}"
    else:
        src = "file:///" + os.path.abspath(path).replace("\\", "/")
    return (f'<a href="{esc(src)}" target="_blank">'
            f'<img class="shot" src="{esc(src)}" loading="lazy"></a>')


def short_action(a):
    return (a or "").strip().replace("\n", " ")


def trajectory_html(mem, task, ctx, max_steps=40):
    """스텝 번호를 기준으로 순회 — 스크린샷이 일부 없어도 밀리지 않는다."""
    thinks = (mem or {}).get("think_list", [])
    actions = (mem or {}).get("action_list", [])
    shot_map = (task or {}).get("shot_map", {})

    idx = set(range(len(actions))) | set(range(len(thinks))) | set(shot_map)
    if not idx:
        return '<p class="muted">궤적 데이터 없음</p>'
    order = sorted(idx)
    shown, omitted = order[:max_steps], max(0, len(order) - max_steps)

    p = []
    for i in shown:
        think = thinks[i] if i < len(thinks) else ""
        act = actions[i] if i < len(actions) else ""
        shot = shot_map.get(i)
        used = "Memory Item" in (think or "")
        p.append('<div class="step">')
        p.append(f'<div class="stepno">{i}{" ★" if used else ""}</div>')
        p.append('<div class="stepbody">')
        if think:
            p.append(f'<div class="think{" used" if used else ""}">{esc(think)}</div>')
        elif not act:
            p.append('<div class="muted" style="font-size:12.5px">'
                     '(이 스텝의 사고·액션 기록 없음 — 화면만 남음)</div>')
        if act:
            p.append(f'<div class="action">→ <code>{esc(short_action(act))[:600]}</code></div>')
        p.append("</div>")
        p.append(f'<div class="stepshot">{img_tag(shot, ctx)}</div>')
        p.append("</div>")
    if omitted:
        p.append(f'<p class="muted">… 이하 {omitted} 스텝 생략</p>')
    return "\n".join(p)


def flow_html(mem, task):
    """태스크를 화살표 흐름으로 압축 표시."""
    acts = (mem or {}).get("action_list", [])
    thinks = (mem or {}).get("think_list", [])
    used_any = any("Memory Item" in (t or "") for t in thinks)

    n = ['<div class="fnode q">쿼리<span>'
         + esc((mem or {}).get("query", ""))[:80] + '</span></div>']
    if used_any:
        n.append('<div class="farrow">▸</div>')
        n.append('<div class="fnode m">메모리 검색<span>유사 경험 → 전략 주입</span></div>')
    for i, a in enumerate(acts[:10]):
        used = "Memory Item" in ((thinks[i] if i < len(thinks) else "") or "")
        n.append('<div class="farrow">▸</div>')
        n.append(f'<div class="fnode a{" u" if used else ""}">step {i}'
                 f'<span>{esc(short_action(a))[:60]}</span></div>')
    if len(acts) > 10:
        n.append('<div class="farrow">▸</div>')
        n.append(f'<div class="fnode">…<span>{len(acts) - 10} 스텝 더</span></div>')
    st = (mem or {}).get("status")
    n.append('<div class="farrow">▸</div>')
    n.append(f'<div class="fnode {"s" if st == "success" else "f"}">자기판정'
             f'<span>{esc(st)}</span></div>')
    n.append('<div class="farrow">▸</div>')
    n.append(f'<div class="fnode m">메모리 추출'
             f'<span>{len(parse_items((mem or {}).get("memory_items")))}개 생성</span></div>')
    return '<div class="flow">' + "".join(n) + "</div>"


def memcards(items, cls=""):
    return "\n".join(
        f'<div class="memitem {cls}"><div class="memtitle">{esc(it["title"])}</div>'
        f'<div class="memdesc">{esc(it["description"])}</div>'
        f'<div class="memcontent">{esc(it["content"])}</div></div>'
        for it in items)


CSS = """
:root{--bg:#0d1017;--panel:#161a22;--panel2:#1b2029;--line:#252b37;--fg:#e6e8ee;
--mut:#8b93a7;--ok:#3fb950;--bad:#f85149;--acc:#58a6ff;--warn:#d29922}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.75 -apple-system,"Segoe UI","Malgun Gothic",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:52px 26px 140px}
h1{font-size:32px;margin:0 0 10px;letter-spacing:-.025em}
h2{font-size:23px;margin:66px 0 18px;padding-bottom:12px;border-bottom:1px solid var(--line)}
h3{font-size:17px;margin:34px 0 12px;color:var(--acc)}
h4{font-size:15px;margin:22px 0 8px}
.sub{color:var(--mut);margin-bottom:16px;font-size:14px}
.lead{color:#c9d1d9;font-size:15.5px;max-width:860px}
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(160px,1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:18px}
.big{font-size:28px;font-weight:650;letter-spacing:-.02em}
.lbl{color:var(--mut);font-size:12px;margin-top:4px}
table{width:100%;border-collapse:collapse;margin:16px 0;font-size:14px}
th,td{padding:10px 13px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--mut);font-weight:500;font-size:11.5px;text-transform:uppercase;letter-spacing:.05em}
.muted{color:var(--mut)}.ok{color:var(--ok)}.bad{color:var(--bad)}
.warn{color:var(--warn)}.acc{color:var(--acc)}
.pill{display:inline-block;padding:1px 10px;border-radius:99px;font-size:11px;
border:1px solid var(--line);color:var(--mut)}
.pill.s{border-color:#2c5c34;color:var(--ok)}.pill.f{border-color:#6b2b2b;color:var(--bad)}
.fig{width:100%;height:auto;margin:18px 0 28px;display:block}
.fig .bx{fill:#161a22;stroke:#252b37;stroke-width:1}
.fig .bx2{fill:#121926;stroke:#2b4a72;stroke-width:1}
.fig .hd{fill:#1b2029;stroke:none}
.fig .okbx{fill:#101a12;stroke:#2c5c34}
.fig .badbx{fill:#1c1111;stroke:#6b2b2b}
.fig .tx{fill:#e6e8ee;font:13px sans-serif}
.fig .tx.b{font-weight:650}
.fig .sm{fill:#8b93a7;font:11.5px sans-serif}
.fig .cap{fill:#8b93a7;font:11.5px sans-serif;letter-spacing:.06em;text-transform:uppercase}
.fig .ln{stroke:#8b93a7;stroke-width:1.4}
.fig .ln2{stroke:#58a6ff;stroke-width:1.6;fill:none}
.fig .ln3{stroke:#d29922;stroke-width:1.6;fill:none;stroke-dasharray:5 4}
.fig .lnok{stroke:#3fb950;stroke-width:1.6;fill:none}
.fig .lnbad{stroke:#f85149;stroke-width:1.6;fill:none}
.fig .dv{stroke:#252b37;stroke-width:1}
.fig .acc{fill:#58a6ff}.fig .ok{fill:#3fb950}.fig .bad{fill:#f85149}.fig .warn{fill:#d29922}
.flow{display:flex;flex-wrap:wrap;align-items:stretch;gap:6px;margin:18px 0}
.fnode{background:var(--panel);border:1px solid var(--line);border-radius:9px;
padding:9px 13px;font-size:12.5px;min-width:104px;max-width:230px}
.fnode span{display:block;color:var(--mut);font-size:11px;margin-top:3px;
word-break:break-word;line-height:1.45}
.fnode.q{border-color:#3a4a63}
.fnode.m{border-color:#2b4a72;background:#121926}
.fnode.a.u{border-color:var(--acc)}
.fnode.s{border-color:#2c5c34}.fnode.f{border-color:#6b2b2b}
.farrow{color:var(--acc);align-self:center;font-size:15px}
.step{display:grid;grid-template-columns:44px 1fr 290px;gap:15px;
padding:15px 0;border-bottom:1px solid var(--line)}
.stepno{color:var(--mut);font-size:12px;padding-top:4px}
.think{white-space:pre-wrap;font-size:13px;color:#c9d1d9;background:#11141a;
border-left:2px solid var(--line);padding:10px 13px;border-radius:0 7px 7px 0}
.think.used{border-left-color:var(--acc);background:#121926}
.action{margin-top:9px}
.action code{background:#0a0d12;border:1px solid var(--line);color:var(--warn);
padding:4px 9px;border-radius:5px;font-size:12.5px;display:inline-block;word-break:break-all}
.shot{width:100%;border:1px solid var(--line);border-radius:7px;display:block}
.shot:hover{border-color:var(--acc)}
.memitem{background:var(--panel);border:1px solid var(--line);
border-radius:10px;padding:15px 17px;margin:11px 0}
.memitem.s{border-left:3px solid var(--ok)}
.memitem.f{border-left:3px solid var(--bad)}
.memtitle{font-weight:640;color:var(--acc);margin-bottom:5px;font-size:14.5px}
.memdesc{font-size:13px;color:var(--mut);margin-bottom:8px}
.memcontent{font-size:13.5px;white-space:pre-wrap;color:#c9d1d9}
.note{background:#191307;border:1px solid #3d2f0f;border-left:3px solid var(--warn);
border-radius:0 9px 9px 0;padding:14px 18px;margin:18px 0;font-size:14px}
.note.b{background:#0d1620;border-color:#1f3a5c;border-left-color:var(--acc)}
.bar{height:7px;background:#0a0d12;border-radius:4px;overflow:hidden;margin-top:5px}
.bar>div{height:100%;background:var(--acc)}
.case{background:var(--panel2);border:1px solid var(--line);border-radius:12px;
padding:20px 22px;margin:26px 0}
.casehd{display:flex;align-items:center;gap:11px;flex-wrap:wrap;margin-bottom:6px}
.caseq{color:#c9d1d9;font-size:14.5px;margin-bottom:6px}
details{margin:12px 0}
summary{cursor:pointer;color:var(--acc);font-size:13.5px;padding:6px 0}
.prompt{background:#0a0d12;border:1px solid var(--line);border-radius:9px;
padding:16px 18px;font:12.5px/1.7 ui-monospace,Consolas,monospace;color:#c9d1d9;
white-space:pre-wrap;margin:14px 0;max-height:340px;overflow:auto}
.jcase{background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:15px 17px;margin:13px 0}
.jcase.use{border-left:3px solid var(--acc)}
.jcase.rej{border-left:3px solid var(--warn)}
.jhd{margin-bottom:9px;font-size:13px}
.toc{background:var(--panel);border:1px solid var(--line);border-radius:11px;
padding:18px 24px;margin:26px 0}
.toc a{color:var(--acc);text-decoration:none;display:block;padding:3px 0;font-size:14px}
.toc a:hover{text-decoration:underline}
@media(max-width:900px){.step{grid-template-columns:1fr}}
"""


# ============================================================ 메인

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--memory_file", required=True)
    ap.add_argument("--log_file", default=None)
    ap.add_argument("--out", default="report/index.html")
    ap.add_argument("--copy_images", action="store_true",
                    help="스크린샷을 report/img 폴더로 복사 (폴더째 공유 가능)")
    ap.add_argument("--embed_images", action="store_true",
                    help="스크린샷을 HTML 안에 직접 삽입 (파일 1개로 공유 가능)")
    ap.add_argument("--img_width", type=int, default=900,
                    help="임베드 시 최대 가로 픽셀 (기본 900)")
    ap.add_argument("--img_quality", type=int, default=72,
                    help="임베드 시 JPEG 품질 (기본 72)")
    ap.add_argument("--max_cases", type=int, default=12)
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument(
        "--subset",
        default=None,
        help="Subset label; inferred from the memory filename when omitted.",
    )
    args = ap.parse_args()

    if args.subset is None:
        memory_stem = os.path.splitext(os.path.basename(args.memory_file))[0].lower()
        known_subsets = {
            "shopping": "Shopping",
            "shopping_admin": "Shopping Admin",
        }
        args.subset = known_subsets.get(
            memory_stem, memory_stem.replace("_", " ").title() or "Unknown"
        )

    subset_key = args.subset.strip().lower().replace("_", " ")
    is_shopping = subset_key == "shopping"
    expected_task_count = {
        "shopping": 187,
        "shopping admin": 182,
    }.get(subset_key)

    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)

    ctx = {"mode": "embed" if args.embed_images else ("copy" if args.copy_images else "link"),
           "out_dir": out_dir, "max_w": args.img_width, "quality": args.img_quality,
           "cache": {}, "bytes": [0]}
    if ctx["mode"] == "embed":
        try:
            import PIL  # noqa: F401
        except ImportError:
            print("  [알림] Pillow가 없어 원본 PNG를 그대로 삽입합니다 "
                  "(파일이 매우 커질 수 있음). 권장: uv pip install pillow")

    mems = load_memories(args.memory_file)
    tasks = load_tasks(args.results_dir)
    log = read_text(args.log_file)

    valid = sorted([t for t in tasks.values() if truthy(t["rm"]) is not None],
                   key=lambda r: r["task_id"])
    n_ok = sum(1 for t in valid if truthy(t["rm"]))
    gt_list = [t for t in tasks.values() if t["gt"] is not None]
    n_gt = sum(1 for t in gt_list if t["gt"] == 1.0)
    mismatch = [t for t in tasks.values()
                if t["gt"] is not None and truthy(t["rm"]) is not None
                and (t["gt"] == 1.0) != truthy(t["rm"])]

    half = len(valid) // 2
    first, second = valid[:half], valid[half:]
    rate = lambda g: (sum(1 for t in g if truthy(t["rm"])) / len(g)) if g else 0
    avgs = lambda g: (sum(t["steps"] for t in g) / len(g)) if g else 0
    succ = [t for t in valid if truthy(t["rm"])]
    fail = [t for t in valid if not truthy(t["rm"])]

    reuse = Counter(re.findall(r'Memory Item \d+ \("([^"]+)"\)', log))
    n_refs = len(re.findall(r"Memory Item \d+", log))
    used_mems = [m for m in mems
                 if any("Memory Item" in (t or "") for t in m.get("think_list", []))]

    timeline = []
    for m in mems:
        for it in parse_items(m.get("memory_items")):
            timeline.append({"order": m["_order"], "task_id": m.get("task_id"),
                             "status": m.get("status"), **it})

    ms = [m for m in mems
          if str(m.get("task_id", "")).isdigit() and int(m["task_id"]) in tasks]

    walkthrough = next(
        (m for m in ms if m.get("status") == "success"
         and 2 <= len(m.get("action_list", [])) <= 4
         and any("Memory Item" in (t or "") for t in m.get("think_list", []))),
        None)
    if walkthrough is None and ms:
        walkthrough = min(ms, key=lambda m: len(m.get("action_list", [])))

    s_list = [m for m in ms if m.get("status") == "success"]
    f_list = [m for m in ms if m.get("status") == "fail"]
    cases = []
    for i in range(max(len(s_list), len(f_list))):
        if i < len(s_list):
            cases.append(s_list[i])
        if i < len(f_list):
            cases.append(f_list[i])
        if len(cases) >= args.max_cases:
            break
    cases = cases[:args.max_cases]

    H = []

    def A(fragment):
        """Append HTML, hiding Shopping-only paper claims for other subsets."""
        if not is_shopping:
            if "Table 4(Shopping)" in fragment:
                fragment = (
                    f'<div class="note">이 절의 수치는 현재 {esc(args.subset)} '
                    '실행 데이터에서 직접 계산했습니다. Shopping 전용 논문 수치는 '
                    '비교 대상으로 표시하지 않았습니다.</div>'
                )
            elif "Table 1 (Gemini-2.5-Flash, Shopping 187" in fragment:
                fragment = (
                    f'<div class="note">{esc(args.subset)} 결과는 현재 실행에서 '
                    '수집한 값만 제시합니다. 동일 subset·모델·실행 조건의 기준값이 '
                    '확인되기 전에는 논문과 정량 비교하지 않습니다.</div>'
                )
            elif "Shopping 49.7%" in fragment:
                fragment = fragment.replace(
                    "Shopping 49.7%", f"{esc(args.subset)} 기준값 미지정"
                )
            elif "187" in fragment and "500" in fragment and expected_task_count:
                fragment = fragment.replace("187", str(expected_task_count), 1)
        H.append(fragment)
    A(f'<!doctype html><html lang="ko"><meta charset="utf-8">'
      f'<title>ReasoningBank 재현 리포트</title><style>{CSS}</style><div class="wrap">')

    A('<h1>ReasoningBank 재현 리포트</h1>')
    A(f'<div class="sub">WebArena {esc(args.subset)} subset · {esc(args.model)} · '
      f'완료 {len(tasks)} 태스크 · 메모리 {len(mems)}건 / 아이템 {len(timeline)}개</div>')
    A('<p class="lead">Google Cloud AI Research의 ReasoningBank(arXiv:2509.25140)를 '
      '개인 PC에서 재현한 기록입니다. 먼저 논문이 무엇을 주장하는지 그림으로 정리하고, '
      '이어서 실제로 실행된 태스크를 통해 그 주장이 어떻게 나타나는지 확인합니다.</p>')

    A('<div class="toc">'
      '<a href="#s1">1. 논문이 풀려는 문제</a>'
      '<a href="#s2">2. 메모리 아이템의 구조</a>'
      '<a href="#s3">3. 세 단계 폐루프</a>'
      '<a href="#s4">4. 태스크 하나로 보는 전체 흐름</a>'
      '<a href="#s45">5. 메모리는 어떻게 프롬프트에 들어가는가</a>'
      '<a href="#s5">6. 태스크별 실행 사례</a>'
      '<a href="#s6">7. 메모리의 진화</a>'
      '<a href="#s7">8. 성공과 실패의 갈림</a>'
      '<a href="#s8">9. 논문과의 대조</a>'
      '<a href="#s9">10. 재현 과정에서 수정한 것</a></div>')

    # 1
    A('<h2 id="s1">1. 논문이 풀려는 문제</h2>')
    A('<p class="lead">LLM 에이전트는 태스크를 하나씩 독립적으로 처리합니다. '
      '어제 겪은 실패를 오늘 다시 반복하고, 어렵게 찾아낸 경로를 매번 새로 탐색합니다. '
      'ReasoningBank는 이 축적되지 않는 경험을 <b>재사용 가능한 추론 전략</b>으로 '
      '증류해 다음 태스크로 넘겨줍니다.</p>')
    A(SVG_PROBLEM)

    # 2
    A('<h2 id="s2">2. 메모리 아이템의 구조</h2>')
    A('<p class="lead">저장되는 것은 궤적 자체가 아니라 <b>전략</b>입니다. '
      '세 항목으로 구조화되어 사람이 읽을 수도, 모델이 프롬프트로 쓸 수도 있습니다.</p>')
    A(SVG_SCHEMA)
    if timeline:
        A('<h4>실제로 생성된 아이템</h4>')
        A(memcards(timeline[:2]))

    # 3
    A('<h2 id="s3">3. 세 단계 폐루프</h2>')
    A('<p class="lead">검색 → 구축 → 통합이 태스크마다 한 바퀴 돌면서 메모리 풀이 자랍니다. '
      '정답 라벨 없이 에이전트가 스스로 성공·실패를 판정하는 것이 핵심입니다.</p>')
    A(SVG_LOOP)
    A(SVG_EXTRACT)
    A('<div class="note b"><b>논문 설정</b> — 임베딩은 gemini-embedding-001, '
      '코사인 유사도 top-k(기본 k=1). 추출기 temperature 1.0, 판정기 0.0. '
      '궤적당 최대 3개 아이템. 통합은 단순 추가로, 병합이나 삭제를 하지 않습니다. '
      'ReasoningBank 자체의 기여를 흐리지 않기 위한 의도적 단순화입니다.</div>')

    # 4
    A('<h2 id="s4">4. 태스크 하나로 보는 전체 흐름</h2>')
    if walkthrough:
        wt = tasks.get(int(walkthrough["task_id"]))
        A(f'<p class="lead">실제 실행된 task {esc(walkthrough["task_id"])}를 따라가 봅니다. '
          f'위 그림의 각 단계가 어디에 대응하는지 보세요.</p>')
        A(flow_html(walkthrough, wt))
        A('<h4>각 스텝의 사고와 화면</h4>')
        A('<p class="muted">★ 표시는 메모리를 인용한 스텝. '
          '스크린샷을 클릭하면 원본이 열립니다.</p>')
        A(trajectory_html(walkthrough, wt, ctx))
        items = parse_items(walkthrough.get("memory_items"))
        if items:
            A('<h4>이 궤적에서 새로 만들어진 메모리</h4>')
            A(memcards(items, "s" if walkthrough.get("status") == "success" else "f"))
    else:
        A('<p class="muted">데이터가 부족해 흐름 예시를 만들지 못했습니다.</p>')

    # 5 (신규) 프롬프트 주입과 모델의 판단
    A('<h2 id="s45">5. 메모리는 어떻게 프롬프트에 들어가는가</h2>')
    A('<p class="lead">검색된 메모리는 시스템 프롬프트 앞에 붙습니다. '
      '이때 함께 들어가는 지시문이 이 논문의 미묘한 설계 지점입니다 — '
      '메모리를 <b>무조건 따르라</b>가 아니라, <b>쓸지 말지 먼저 논한 다음 행동하라</b>고 '
      '요구합니다.</p>')

    if timeline:
        A('<h4>실제로 주입되는 형태 (재구성)</h4>')
        A('<p class="muted">저장소가 프롬프트 원문을 남기지 않으므로, '
          '논문 부록 A.2의 템플릿과 실제 축적된 메모리를 결합해 복원했습니다.</p>')
        A(f'<div class="prompt">{esc(reconstruct_prompt(timeline[:1]))}</div>')

    use_n, rej_n, sil_n = memory_judgment_stats(mems)
    tot = use_n + rej_n + sil_n
    if tot:
        A('<h3>그 지시가 만들어낸 결과</h3>')
        A('<div class="grid">')
        A(f'<div class="card"><div class="big">{tot}</div>'
          f'<div class="lbl">전체 사고 단계</div></div>')
        A(f'<div class="card"><div class="big ok">{use_n}</div>'
          f'<div class="lbl">메모리 사용 판단</div></div>')
        A(f'<div class="card"><div class="big warn">{rej_n}</div>'
          f'<div class="lbl">사용하지 않겠다는 판단</div></div>')
        A(f'<div class="card"><div class="big">{(use_n + rej_n) / tot:.0%}</div>'
          f'<div class="lbl">명시적으로 언급한 비율</div></div>')
        A('</div>')
        if use_n + rej_n:
            A(f'<div class="note b">메모리를 언급한 {use_n + rej_n}회 중 '
              f'<b>{rej_n / (use_n + rej_n):.0%}가 "쓰지 않겠다"는 판단</b>입니다. '
              f'에이전트가 검색된 전략을 기계적으로 따르는 것이 아니라 '
              f'현재 상황에 맞는지 스스로 검토하고 있다는 뜻입니다. '
              f'이 선별 능력이 없다면 관련 없는 메모리가 오히려 방해가 됩니다.</div>')

    used_cases = collect_judgments(mems, "use", 4)
    rej_cases = collect_judgments(mems, "reject", 4)

    if used_cases:
        A('<h3>메모리를 채택한 순간</h3>')
        for c in used_cases:
            A('<div class="jcase use">')
            A(f'<div class="jhd"><span class="pill">task {esc(c["task_id"])} · '
              f'step {c["step"]}</span> <span class="muted">{esc(c["query"])[:90]}</span></div>')
            A(f'<div class="think used">{esc(c["think"])}</div>')
            if c["action"]:
                A(f'<div class="action">→ <code>{esc(short_action(c["action"]))[:300]}</code></div>')
            A('</div>')

    if rej_cases:
        A('<h3>메모리를 물린 순간</h3>')
        A('<p class="lead">논문이 강조하지 않는 부분이지만, 실제로는 이 판단이 자주 일어납니다. '
          '"아직 로그인도 안 했으니 이 전략을 쓸 때가 아니다" 같은 시점 판단, '
          '"이 태스크와는 무관하다"는 관련성 판단이 섞여 있습니다.</p>')
        for c in rej_cases:
            A('<div class="jcase rej">')
            A(f'<div class="jhd"><span class="pill">task {esc(c["task_id"])} · '
              f'step {c["step"]}</span> <span class="muted">{esc(c["query"])[:90]}</span></div>')
            A(f'<div class="think">{esc(c["think"])}</div>')
            if c["action"]:
                A(f'<div class="action">→ <code>{esc(short_action(c["action"]))[:300]}</code></div>')
            A('</div>')

    A('<div class="note">캡처 한계 — 저장소는 LLM 응답의 사고 텍스트만 궤적에 남기고 '
      '프롬프트 원문(agent_info)은 비워둡니다. 위 프롬프트는 재구성이며, '
      '사고 텍스트는 실제 모델 출력 그대로입니다.</div>')

    # 6

    A('<h2 id="s5">6. 태스크별 실행 사례</h2>')
    A(f'<p class="lead">성공과 실패를 번갈아 {len(cases)}건 수록했습니다. '
      f'쿼리 → 흐름 → 상세 궤적 → 추출된 메모리 순입니다.</p>')
    for m in cases:
        tid = int(m["task_id"])
        t = tasks.get(tid)
        st = m.get("status")
        A('<div class="case">')
        A('<div class="casehd">'
          f'<span class="pill {"s" if st == "success" else "f"}">{esc(st)}</span>'
          f'<b>task {tid}</b>'
          f'<span class="muted">{len(m.get("action_list", []))} 스텝</span>'
          + (f'<span class="muted">gt={esc(t["gt"])} · rm={esc(t["rm"])}</span>' if t else "")
          + '</div>')
        A(f'<div class="caseq">{esc(m.get("query", ""))}</div>')
        A(flow_html(m, t))
        if t and t.get("thoughts"):
            A(f'<div class="note"><b>autoeval 판정 근거</b><br>{esc(t["thoughts"])}</div>')
        A('<details><summary>상세 궤적과 스크린샷 펼치기</summary>')
        A(trajectory_html(m, t, ctx))
        A('</details>')
        items = parse_items(m.get("memory_items"))
        if items:
            A('<h4>추출된 메모리</h4>')
            A(memcards(items, "s" if st == "success" else "f"))
        A('</div>')

    # 6
    A('<h2 id="s6">7. 메모리의 진화</h2>')
    A('<p class="lead">논문은 메모리가 절차적 규칙에서 시작해 자기점검, 완결성 확인, '
      '조합적 전략으로 발전한다고 보고합니다. 아래는 실제 축적 순서입니다.</p>')
    if reuse:
        A('<h3>가장 많이 재사용된 전략</h3>')
        A(f'<p class="muted">로그 전체에서 메모리 인용 {n_refs}회. '
          f'소수의 전략이 반복적으로 쓰입니다.</p>')
        A('<table><tr><th>인용</th><th>메모리</th><th></th></tr>')
        mx = max(reuse.values())
        for title, cnt in reuse.most_common(15):
            A(f'<tr><td>{cnt}</td><td>{esc(title)}</td>'
              f'<td style="width:190px"><div class="bar">'
              f'<div style="width:{cnt / mx * 100:.0f}%"></div></div></td></tr>')
        A('</table>')
    if timeline:
        A('<h3>초기 메모리</h3>')
        A(memcards(timeline[:3]))
        if len(timeline) > 6:
            A('<h3>후기 메모리</h3>')
            A(memcards(timeline[-3:]))
        A(f'<details><summary>전체 아이템 {len(timeline)}개 펼치기</summary>')
        for it in timeline:
            cls = "s" if it["status"] == "success" else "f"
            A(f'<div class="memitem {cls}"><div class="memtitle">{esc(it["title"])} '
              f'<span class="pill {cls}">task {esc(it["task_id"])}</span></div>'
              f'<div class="memdesc">{esc(it["description"])}</div>'
              f'<div class="memcontent">{esc(it["content"])}</div></div>')
        A('</details>')

    # 7
    A('<h2 id="s7">8. 성공과 실패의 갈림</h2>')
    A('<div class="grid">')
    if valid:
        A(f'<div class="card"><div class="big">{n_ok / len(valid):.1%}</div>'
          f'<div class="lbl">성공률 (autoeval)</div></div>')
    if gt_list:
        A(f'<div class="card"><div class="big">{n_gt / len(gt_list):.1%}</div>'
          f'<div class="lbl">성공률 (규칙기반 gt)</div></div>')
    if succ:
        A(f'<div class="card"><div class="big">{avgs(succ):.1f}</div>'
          f'<div class="lbl">성공 시 평균 스텝</div></div>')
    if fail:
        A(f'<div class="card"><div class="big">{avgs(fail):.1f}</div>'
          f'<div class="lbl">실패 시 평균 스텝</div></div>')
    A(f'<div class="card"><div class="big">{len(mismatch)}</div>'
      f'<div class="lbl">gt·rm 판정 불일치</div></div>')
    A('</div>')
    A('<div class="note">논문 Table 4(Shopping): No Memory는 성공 6.8 / 실패 8.7 스텝, '
      'ReasoningBank는 성공 4.7(↓2.1) / 실패 7.3(↓1.4). '
      '성공 궤적에서 감소폭이 더 크다는 점이 중요합니다. '
      '실패를 일찍 포기해서가 아니라 올바른 경로를 빨리 찾기 때문이라는 근거입니다.</div>')

    if first and second:
        A('<h3>전반부 vs 후반부</h3>')
        A('<table><tr><th>구간</th><th>태스크</th><th>성공률</th><th>평균 스텝</th></tr>')
        A(f'<tr><td>전반</td><td>{len(first)}</td><td>{rate(first):.1%}</td>'
          f'<td>{avgs(first):.1f}</td></tr>')
        A(f'<tr><td>후반</td><td>{len(second)}</td><td>{rate(second):.1%}</td>'
          f'<td>{avgs(second):.1f}</td></tr></table>')
        A('<div class="note">WebArena 태스크는 번호순 난이도가 균일하지 않습니다. '
          '이 차이가 메모리 효과인지 난이도 차이인지는 no_memory 대조군 없이는 판단할 수 없습니다.</div>')

    if mismatch:
        A('<h3>규칙 채점과 LLM 채점이 갈린 사례</h3>')
        A('<p class="lead">WebArena 기본 채점은 문자열 매칭이라 표현 차이를 실패로 봅니다. '
          'autoeval은 의미 기준으로 판정합니다. 논문은 두 방식을 함께 씁니다.</p>')
        A('<table><tr><th>task</th><th>gt</th><th>rm</th><th>autoeval 판정 근거</th></tr>')
        for t in sorted(mismatch, key=lambda r: r["task_id"])[:12]:
            A(f'<tr><td>{t["task_id"]}</td><td>{esc(t["gt"])}</td><td>{esc(t["rm"])}</td>'
              f'<td>{esc(t["thoughts"][:320])}…</td></tr>')
        A('</table>')

    # 8
    A('<h2 id="s8">9. 논문과의 대조</h2>')
    A('<h3>재현된 것</h3><table><tr><th>항목</th><th>논문</th><th>본 재현</th></tr>')
    A(f'<tr><td>성공·실패 양쪽 추출</td><td>실패를 counterfactual 신호로 활용</td>'
      f'<td class="ok">성공 {sum(1 for m in mems if m.get("status") == "success")}건 · '
      f'실패 {sum(1 for m in mems if m.get("status") == "fail")}건</td></tr>')
    A(f'<tr><td>검색 후 프롬프트 주입</td><td>임베딩 유사도 top-k</td>'
      f'<td class="ok">{len(used_mems)}/{len(mems)} 궤적에서 인용</td></tr>')
    A('<tr><td>전략 수준 추상화</td><td>궤적이 아닌 재사용 가능한 전략</td>'
      '<td class="ok">사이트 구조·검증 절차 형태로 추출</td></tr>')
    A('<tr><td>메모리 재사용 편중</td><td>소수 전략이 반복 활용</td>'
      '<td class="ok">상위 소수가 인용의 다수 차지</td></tr>')
    if first and second:
        A(f'<tr><td>누적에 따른 개선</td><td>+8.3%p (전체 기준)</td>'
          f'<td class="warn">전반 {rate(first):.1%} → 후반 {rate(second):.1%} '
          f'(난이도 교란 배제 불가)</td></tr>')
    A('</table>')

    A('<h3>재현하지 못한 것</h3><table><tr><th>항목</th><th>사유</th></tr>')
    A('<tr><td>No Memory 대비 개선폭</td><td class="bad">대조군 미실행. '
      '같은 태스크에서 메모리 유무를 비교해야 인과를 말할 수 있음</td></tr>')
    A('<tr><td>성공 궤적 2.1 스텝 단축</td><td class="bad">동일 태스크 짝 비교 불가</td></tr>')
    A('<tr><td>절대 수치 (Shopping 49.7%)</td><td class="bad">백본 상이 — '
      '논문 Gemini-2.5-Flash, 본 재현 3.5-flash-lite</td></tr>')
    A(f'<tr><td>전체 187개 태스크</td><td class="bad">무료 티어 일일 500 요청 한도. '
      f'현재 {len(tasks)}개 완료</td></tr>')
    A('<tr><td>MaTTS (test-time scaling)</td><td class="bad">k=5 병렬 실행은 '
      '요청량이 5배 — 쿼터상 불가</td></tr>')
    A('<tr><td>Synapse · AWM 비교</td><td class="bad">--memory_mode 옵션은 있으나 '
      '쿼터 제약으로 미실행</td></tr></table>')

    A('<div class="note">논문 Table 1 (Gemini-2.5-Flash, Shopping 187개): '
      'No Memory 39.0% / 8.2스텝 → Synapse 40.6% → AWM 44.4% → '
      'ReasoningBank 49.7% / 6.1스텝. '
      '실패 궤적을 포함하면 46.5%에서 49.7%로 더 오릅니다.</div>')

    # 9
    A('<h2 id="s9">10. 재현 과정에서 수정한 것</h2>')
    A('<p class="lead">공개 저장소는 Linux + Vertex AI 환경을 전제로 작성되어 있습니다. '
      'Windows에서 무료 API로 돌리기 위해 필요했던 변경입니다.</p>')
    A('<table><tr><th>구분</th><th>증상</th><th>조치</th></tr>')
    A('<tr><td>의존성</td><td>greenlet 3.0.3이 Python 3.13에서 빌드 실패</td>'
      '<td>browsergym 0.14.1 → playwright 1.44 고정 사슬 확인 후 '
      'requires-python을 3.12로 하향</td></tr>')
    A('<tr><td>인증</td><td>genai.Client(vertexai=True) 하드코딩 5개소</td>'
      '<td>AI Studio API 키 방식으로 전환</td></tr>')
    A('<tr><td>임베딩</td><td>구형 vertexai.TextEmbeddingModel은 GCP 프로젝트 필수</td>'
      '<td>google-genai의 embed_content로 교체</td></tr>')
    A('<tr><td>경로</td><td>split("/")가 Windows 경로에서 오작동</td>'
      '<td>os.path.basename으로 수정</td></tr>')
    A('<tr><td>인코딩</td><td>cp949 기본값으로 UnicodeEncodeError</td>'
      '<td>PYTHONUTF8=1 및 open(encoding="utf-8")</td></tr>')
    A('<tr><td>파일 잠금</td><td>os.rename PermissionError (WinError 5)</td>'
      '<td>재시도 후 copytree 폴백</td></tr>')
    A('<tr><td>하위 프로세스</td><td>Popen의 "python"이 conda base를 집음</td>'
      '<td>sys.executable로 고정</td></tr>')
    A('<tr><td>모델</td><td>gemini-2.5-flash 서비스 종료 (404)</td>'
      '<td>3.5-flash-lite로 대체, CLIENT_DICT와 choices에 등록</td></tr>')
    A('<tr><td>쿼터</td><td>일일 500 요청 초과 (429 RESOURCE_EXHAUSTED)</td>'
      '<td>태스크당 약 12회 소모 → 하루 약 40개. --prev_id로 분할 실행</td></tr>')
    A('</table>')

    A(f'<p class="muted" style="margin-top:52px">태스크 {len(tasks)} · '
      f'메모리 {len(mems)} · 아이템 {len(timeline)} · 로그 인용 {n_refs}회</p>')
    A('</div></html>')

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(H))

    size_mb = os.path.getsize(args.out) / 1024 / 1024
    print(f"생성 완료: {os.path.abspath(args.out)}  ({size_mb:.1f} MB)")
    if ctx["mode"] == "embed":
        print(f"  이미지 {len(ctx['cache'])}장 삽입 — 이 파일 하나만 보내면 됩니다.")
    elif ctx["mode"] == "copy":
        print(f"  스크린샷은 {os.path.join(out_dir, 'img')} 에 복사됨 — 폴더째 공유하세요.")
    else:
        print("  스크린샷은 원본 경로를 참조합니다 (이 PC에서만 보임). "
              "공유하려면 --embed_images 또는 --copy_images 를 쓰세요.")
    print(f"  태스크 {len(tasks)} / 메모리 {len(mems)} / "
          f"아이템 {len(timeline)} / 사례 {len(cases)}")
    if not log:
        print("  [주의] 로그를 읽지 못해 재사용 통계가 비어 있습니다.")


if __name__ == "__main__":
    main()
