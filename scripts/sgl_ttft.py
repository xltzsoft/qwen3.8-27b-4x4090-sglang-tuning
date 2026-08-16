#!/usr/bin/env python3
"""Prefill TTFT bench with cache-busting salt.
Usage: python3 sgl_ttft.py <tag> [tokens ...]   (default 50000 100000)
"""
import urllib.request, json, time, sys, random, string, os

BASE = 'http://127.0.0.1:8001'
KEY = 'sk-CHANGE-ME'
OUTDIR = '/root/gdn-opt/results'

PARA = ("人工智能的发展经历了多次浪潮。从符号主义到连接主义，再到深度学习的大模型时代，"
        "每一次范式转移都伴随着算力、数据与算法的共同进步。在实际工程落地中，推理效率"
        "往往决定了产品的用户体验与成本结构，因此量化、投机解码、KV缓存优化等技术成为"
        "业界关注的焦点。The quick brown fox jumps over the lazy dog. "
        "Batch size 与 latency 的权衡贯穿 serving 系统的各个层次。\n")

def make_prompt(target_tokens, salt):
    text = salt + '\n'
    while len(text) < target_tokens * 3.2:
        text += PARA
    return text

def run_once(tokens):
    salt = ''.join(random.choices(string.ascii_letters, k=24))
    q = make_prompt(tokens, salt) + "\n\n请用一句话总结上文。"
    payload = {'model': 'default', 'messages': [{'role': 'user', 'content': q}],
               'max_tokens': 8, 'temperature': 0.0, 'stream': True,
               'stream_options': {'include_usage': True}}
    t0 = time.time()
    resp = urllib.request.urlopen(urllib.request.Request(
        BASE + '/v1/chat/completions', data=json.dumps(payload).encode(),
        headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'}), timeout=900)
    first_t = None
    usage = {}
    buf = b''
    while True:
        chunk = resp.read(8192)
        if not chunk:
            break
        buf += chunk
        while b'\n' in buf:
            line, buf = buf.split(b'\n', 1)
            line = line.strip()
            if not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if data == b'[DONE]':
                break
            try:
                j = json.loads(data)
            except Exception:
                continue
            if j.get('usage'):
                usage = j['usage']
            ch = j.get('choices') or []
            if ch:
                delta = ch[0].get('delta') or {}
                if (delta.get('content') or delta.get('reasoning_content')) and first_t is None:
                    first_t = time.time()
    wall = time.time() - t0
    ttft = (first_t - t0) if first_t else None
    ptok = usage.get('prompt_tokens', 0)
    return {'target': tokens, 'prompt_tokens': ptok, 'ttft': round(ttft, 2) if ttft else None,
            'prefill_tps': round(ptok / ttft, 1) if (ttft and ptok) else None,
            'wall': round(wall, 2)}

def main():
    tag = sys.argv[1]
    toks = [int(x) for x in sys.argv[2:]] or [50000, 100000]
    os.makedirs(OUTDIR, exist_ok=True)
    res = {'tag': tag, 'ts': time.strftime('%F %T'), 'runs': []}
    for t in toks:
        for rep in range(2):
            r = run_once(t)
            r['rep'] = rep
            res['runs'].append(r)
            print(r, flush=True)
    path = os.path.join(OUTDIR, 'ttft-' + tag + '.json')
    with open(path, 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print('SAVED', path)

if __name__ == '__main__':
    main()
