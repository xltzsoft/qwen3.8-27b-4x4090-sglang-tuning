#!/usr/bin/env python3
"""Profile 50k-context spec decode and analyze kernel time distribution."""
import urllib.request, json, time, subprocess, sys, gzip, glob, os, collections

BASE = 'http://127.0.0.1:8001'
KEY = 'sk-CHANGE-ME'
PROFDIR = '/root/gdn-opt/prof'

def post(ep, payload, timeout=180):
    r = urllib.request.Request(BASE + ep, data=json.dumps(payload).encode(),
                               headers={'Authorization': 'Bearer ' + KEY,
                                        'Content-Type': 'application/json'})
    return urllib.request.urlopen(r, timeout=timeout).read().decode()

def make_long_prompt(target_tokens):
    para = ("人工智能的发展经历了多次浪潮。从符号主义到连接主义，再到深度学习的大模型时代，"
            "每一次范式转移都伴随着算力、数据与算法的共同进步。在实际工程落地中，推理效率"
            "往往决定了产品的用户体验与成本结构，因此量化、投机解码、KV缓存优化等技术成为"
            "业界关注的焦点。The quick brown fox jumps over the lazy dog. "
            "Batch size 与 latency 的权衡贯穿 serving 系统的各个层次。\n")
    text = ''
    while len(text) < target_tokens * 3.2:
        text += para
    return text

def main():
    warmup = '--no-warmup' not in sys.argv
    os.makedirs(PROFDIR, exist_ok=True)
    for f in glob.glob(PROFDIR + '/*'):
        os.remove(f)

    q = make_long_prompt(50000) + "\n\n请详细总结这段文字反复论述的主题，至少写三百字。"
    payload = {'model': 'default', 'messages': [{'role': 'user', 'content': q}],
               'max_tokens': 400, 'temperature': 0.0, 'stream': False}

    if warmup:
        # warm: prefill + some decode, ensures radix cache hit on 2nd run
        r0 = urllib.request.urlopen(urllib.request.Request(
            BASE + '/v1/chat/completions', data=json.dumps(payload).encode(),
            headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'}), timeout=900).read()
        print('warmup done', flush=True)
        time.sleep(2)

    print(post('/start_profile', {'output_dir': PROFDIR, 'with_stack': False}), flush=True)
    time.sleep(1)
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(urllib.request.Request(
        BASE + '/v1/chat/completions', data=json.dumps(payload).encode(),
        headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'}), timeout=900).read())
    wall = time.time() - t0
    print('gen done, tokens=', resp['usage']['completion_tokens'], f'{wall:.1f}s', flush=True)
    print(post('/stop_profile', {}), flush=True)
    time.sleep(10)

    files = sorted(glob.glob(PROFDIR + '/*TP-0*.json.gz') or glob.glob(PROFDIR + '/*.json.gz'))
    print('trace files:', [os.path.basename(f) for f in files], flush=True)
    f0 = files[0]
    op = gzip.open if f0.endswith('.gz') else open
    with op(f0, 'rt') as fh:
        trace = json.load(fh)
    evs = trace['traceEvents'] if isinstance(trace, dict) else trace
    agg = collections.defaultdict(float)
    cnt = collections.defaultdict(int)
    total = 0.0
    for e in evs:
        if e.get('ph') != 'X':
            continue
        if e.get('cat', '') in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            agg[e['name']] += e['dur']
            cnt[e['name']] += 1
            total += e['dur']
    print(f'total kernel time: {total/1000:.1f} ms (wall {wall*1000:.0f} ms)')
    print(f'{"kernel":<88} {"ms":>9} {"cnt":>6} {"%":>6}')
    for name, dur in sorted(agg.items(), key=lambda x: -x[1])[:30]:
        print(f'{name[:88]:<88} {dur/1000:>9.2f} {cnt[name]:>6} {dur/total*100:>5.1f}%')

if __name__ == '__main__':
    main()
