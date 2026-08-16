#!/usr/bin/env python3
"""Unified bench for manual instances on :8001.
Usage: SGL_LOG=/root/gdn-opt/server-xxx.log python3 sgl_bench3.py <tag> [--skip-quality] [--skip-50k]
Server-side tps parsed from the instance log, split by context size.
"""
import urllib.request, json, time, sys, os, subprocess, re, statistics

BASE = 'http://127.0.0.1:8001'
KEY = 'sk-CHANGE-ME'
OUTDIR = '/root/gdn-opt/results'
LOG = os.environ.get('SGL_LOG', '/root/gdn-opt/server-nospec.log')

def chat_stream(messages, max_tokens=300, temperature=0.0):
    payload = {'model': 'default', 'messages': messages, 'max_tokens': max_tokens,
               'temperature': temperature, 'stream': True,
               'stream_options': {'include_usage': True}}
    t0 = time.time()
    resp = urllib.request.urlopen(urllib.request.Request(
        BASE + '/v1/chat/completions', data=json.dumps(payload).encode(),
        headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'}), timeout=900)
    first_t = None
    text = ''
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
                c = delta.get('content') or delta.get('reasoning_content') or ''
                if c:
                    if first_t is None:
                        first_t = time.time()
                    text += c
    return time.time(), (first_t - t0) if first_t else None, text, usage

def log_stats(min_tok=0, max_tok=10**9, since_marker=0):
    out = subprocess.run("grep 'Decode batch' %s" % LOG,
                         shell=True, capture_output=True, text=True).stdout
    lines = out.strip().splitlines()[since_marker:]
    rows = []
    for ln in lines:
        mt = re.search(r'#full token: (\d+)', ln)
        mp = re.search(r'gen throughput \(token/s\): ([\d.]+)', ln)
        ma = re.search(r'accept len\w*: ([\d.]+)', ln)
        if not (mt and mp):
            continue
        tok = int(mt.group(1))
        tps = float(mp.group(1))
        if min_tok <= tok <= max_tok and tps > 3:
            rows.append((tok, tps, float(ma.group(1)) if ma else None))
    # The first decode log line after a prefill straddles the whole prefill
    # window (throughput is tokens/wall-since-last-log) -> drop it as artifact.
    tps_rows = rows[1:] if len(rows) > 1 else rows
    tps = [r[1] for r in tps_rows]
    acc = [r[2] for r in rows if r[2]]
    return {
        'tps_median': round(statistics.median(tps), 2) if tps else None,
        'tps_min': round(min(tps), 2) if tps else None,
        'tps_max': round(max(tps), 2) if tps else None,
        'accept_len': round(statistics.median(acc), 2) if acc else None,
        'n': len(rows),
    }

def client_tps(wall, ttft, usage):
    """Decode tps from client wall time excluding prefill (TTFT)."""
    try:
        ct = usage.get('completion_tokens', 0)
        if ct and ttft:
            return round(ct / max(wall - ttft, 1e-6), 2)
    except Exception:
        pass
    return None

def log_line_count():
    out = subprocess.run("grep -c 'Decode batch' %s" % LOG,
                         shell=True, capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else 0

def make_long_prompt(target_tokens):
    para = ("人工智能的发展经历了多次浪潮。从符号主义到连接主义，再到深度学习的大模型时代，"
            "每一次范式转移都伴随着算力、数据与算法的共同进步。在实际工程落地中，推理效率"
            "往往决定了产品的用户体验与成本结构，因此量化、投机解码、KV缓存优化等技术成为"
            "业界关注的焦点。The quick brown fox jumps over the lazy dog. "
            "Batch size 与 latency 的权衡贯穿 serving 系统的各个层次。\n")
    needle = "本文档中隐藏的秘密数字是 884213，请记住它。\n"
    text = ''
    while len(text) < target_tokens * 3.2:
        text += para
        if len(text) % 12000 < 3000:
            text += needle
    return text

QUALITY_PROMPTS = [
    "计算 12345 乘以 6789，给出精确结果和计算过程。",
    "求解方程 x^2 - 5x + 6 = 0，写出完整步骤。",
    "用 Python 写一个函数，判断字符串是否为回文，要求时间复杂度 O(n)，并给出三个测试用例。",
    "解释量子纠缠是什么，以及它为什么不能用于超光速通信。",
    "把这段话翻译成英文：'深度学习模型的推理优化需要同时考虑计算、显存带宽和通信开销。'",
    "写一个 SQL 查询：表 orders(id, user_id, amount, created_at)，找出每个用户最近一笔订单的金额。",
    "甲乙丙三人中有一人说谎。甲说：乙说谎。乙说：丙说谎。丙说：甲和乙都没说谎。推理谁说谎。",
    "用 C++ 实现一个线程安全的单例模式，并说明为什么这样写是正确的。",
]

def main():
    tag = sys.argv[1]
    skip_quality = '--skip-quality' in sys.argv
    skip_50k = '--skip-50k' in sys.argv
    os.makedirs(OUTDIR, exist_ok=True)
    res = {'tag': tag, 'ts': time.strftime('%F %T'), 'log': LOG}

    m0 = log_line_count()
    wall, ttft, _, usage = chat_stream(
        [{'role': 'user', 'content': '请详细介绍一下 Transformer 架构的演进历史，从 2017 年到现在的主要里程碑。'}],
        max_tokens=600)
    s = log_stats(0, 100000, m0)
    s['ttft'] = ttft
    s['client_tps'] = client_tps(wall, ttft, usage)
    res['short'] = s
    print(f"[short] {s}", flush=True)

    if not skip_50k:
        q = make_long_prompt(50000) + "\n\n问题：文档中隐藏的秘密数字是什么？请先给出数字，然后详细总结这段文字反复论述的主题，不少于四百字。"
        m0 = log_line_count()
        wall, ttft, text, usage = chat_stream([{'role': 'user', 'content': q}], max_tokens=600)
        s = log_stats(50000, 10**9, m0)
        s['ttft'] = ttft
        s['client_tps'] = client_tps(wall, ttft, usage)
        s['needle_ok'] = '884213' in text
        res['long50k'] = s
        print(f"[50k] {s}", flush=True)

    if not skip_quality:
        outs = []
        for i, p in enumerate(QUALITY_PROMPTS):
            _, _, text, _ = chat_stream([{'role': 'user', 'content': p}], max_tokens=800)
            outs.append({'prompt': p, 'output': text})
            print(f"[quality {i+1}/8] len={len(text)}", flush=True)
        res['quality'] = outs

    path = os.path.join(OUTDIR, tag + '.json')
    with open(path, 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print('SAVED', path)

if __name__ == '__main__':
    main()
