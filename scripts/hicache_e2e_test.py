#!/usr/bin/env python3
"""HiCache end-to-end verification for hybrid GDN models on SGLang.

Method: cold-prefill a long needle prefix -> push >1x GPU-pool of distinct long
prompts to force eviction from GPU L1 -> re-request the prefix. If the host-memory
L2 (HiMambaRadixCache: full-attention KV + GDN mamba checkpoints + draft KV) works,
TTFT drops to a PCIe reload and the needle answer stays correct.

Usage: python3 hicache_e2e_test.py   (server on 127.0.0.1:8000, HiCache enabled)
"""
import json, random, re, time, urllib.request

BASE = "http://127.0.0.1:8000"
KEY = "sk-CHANGE-ME"
NEEDLE = "884213"


def ask(content, max_tokens=8, timeout=900):
    body = json.dumps(
        {"model": "default",
         "messages": [{"role": "user", "content": content}],
         "max_tokens": max_tokens, "temperature": 0},
        ensure_ascii=False).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=timeout))
    dt = time.time() - t
    u = d.get("usage", {})
    msg = d["choices"][0]["message"]
    text = (msg.get("content") or "") + "|" + (msg.get("reasoning_content") or "")
    return dt, u, text


def metrics():
    req = urllib.request.Request(BASE + "/metrics",
                                 headers={"Authorization": f"Bearer {KEY}"})
    m = urllib.request.urlopen(req, timeout=30).read().decode()
    out = {}
    for name in ("hicache_host_used_tokens", "hicache_host_total_tokens", "cache_hit_rate"):
        mm = re.search(rf"sglang:{name}\{{[^}}]*\}} ([0-9.e+]+)", m)
        out[name] = float(mm.group(1)) if mm else None
    return out


def nonce(n=16):
    return "".join(random.choice("abcdefghij0123456789") for _ in range(n)) + "。"


def main():
    print(f"[metrics-start] {metrics()}", flush=True)

    # ~31.6k-token needle prefix (fills + needle + question)
    A = (nonce()
         + "今天天气不错。" * 3500
         + f"请记住这个秘密数字：{NEEDLE}。"
         + "我们继续闲聊吧。" * 3500
         + "问题：我前面让你记住的秘密数字是什么？请直接回答数字。")
    dt0, u0, c0 = ask(A, 400)
    print(f"[A-cold] {dt0:.2f}s pt={u0.get('prompt_tokens')} needle={NEEDLE in c0}", flush=True)

    # Eviction pressure: 14 x 50k-125k distinct long prompts (>> GPU pool 657,733 tokens)
    fillers = ["山高水长", "风起云涌", "海阔天空", "云淡风轻", "柳暗花明", "波澜壮阔", "繁花似锦",
               "星光灿烂", "碧波荡漾", "层峦叠嶂", "烟雨朦胧", "鸟语花香", "金碧辉煌", "曲径通幽"]
    for i, f in enumerate(fillers):
        dt, u, _ = ask(nonce() + f * 25000 + "。这段话里哪个字出现最多？", 1)
        print(f"[pressure-{i}] {dt:.2f}s pt={u.get('prompt_tokens')}", flush=True)
    print(f"[metrics-after-pressure] {metrics()}", flush=True)

    # Reload: prefix now lives only in host RAM
    dt1, u1, c1 = ask(A, 400)
    cached = (u1.get("prompt_tokens_details") or {}).get("cached_tokens")
    print(f"[A-reload] {dt1:.2f}s cached={cached} needle={NEEDLE in c1}", flush=True)
    print(f"RESULT cold={dt0:.2f}s reload={dt1:.2f}s speedup={dt0 / max(dt1, 0.01):.1f}x "
          f"needle_cold={NEEDLE in c0} needle_reload={NEEDLE in c1}", flush=True)


if __name__ == "__main__":
    main()
