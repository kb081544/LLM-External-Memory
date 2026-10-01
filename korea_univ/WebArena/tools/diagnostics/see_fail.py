import io, re
enc_try = ("utf-16", "utf-8", "cp949")
for e in enc_try:
    try:
        t = open("D:/log_shopping.txt", encoding=e).read()
        if t.count("\x00") < len(t) * 0.1:
            break
    except Exception:
        continue

lines = t.splitlines()
start = next(i for i, l in enumerate(lines) if "webarena.50_33" in l)
end = next((i for i, l in enumerate(lines[start:], start)
            if "Saving summary" in l), start + 120)
for l in lines[start:end + 25]:
    print(l[:220])