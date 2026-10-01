import os, openai

c = openai.OpenAI()
r = c.chat.completions.create(
    model="qwen3:8b",
    messages=[{"role": "user", "content": "Reply with exactly: ok"}],
    max_tokens=200,
)
m = r.choices[0].message
print("content:", repr(m.content))
print("reasoning:", repr(getattr(m, "reasoning_content", None))[:200])