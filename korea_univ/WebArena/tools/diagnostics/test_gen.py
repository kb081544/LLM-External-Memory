import os
from google import genai

client = genai.Client()
for model in ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.5-flash"]:
    try:
        r = client.models.generate_content(model=model, contents="Say ok")
        print(f"{model:26s} → OK  {r.text[:40]!r}")
    except Exception as e:
        msg = str(e)
        code = "429" if "429" in msg else "503" if "503" in msg else "404" if "404" in msg else "?"
        print(f"{model:26s} → {code}  {msg[:90]}")