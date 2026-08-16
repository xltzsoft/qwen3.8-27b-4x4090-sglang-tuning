#!/usr/bin/env python3
"""Profile decode: start_profile -> generate -> stop_profile -> analyze chrome trace."""
import urllib.request, json, time, subprocess, sys, gzip, glob, os, collections

BASE = 'http://127.0.0.1:8000'
KEY = 'sk-CHANGE-ME'
PROFDIR = '/root/gdn-opt/prof'

def post(ep, payload):
    r = urllib.request.Request(BASE + ep, data=json.dumps(payload).encode(),
                               headers={'Authorization': 'Bearer ' + KEY,
                                        'Content-Type': 'application/json'})
    return urllib.request.urlopen(r, timeout=120).read().decode()

def main():
    os.makedirs(PROFDIR, exist_ok=True)
    for f in glob.glob(PROFDIR + '/*'):
        os.remove(f)
    print(post('/start_profile', {'output_dir': PROFDIR, 'with_stack': False}), flush=True)
    time.sleep(1)

    # decode workload: short prompt, 300 tokens
    prompt = '请详细介绍一下 Transformer 架构的演进历史。'
    payload = {'model': 'default', 'messages': [{'role': 'user', 'content': prompt}],
               'max_tokens': 300, 'temperature': 0.0, 'stream': False}
    t0 = time.time()
    r = urllib.request.Request(BASE + '/v1/chat/completions', data=json.dumps(payload).encode(),
                               headers={'Authorization': 'Bearer ' + KEY,
                                        'Content-Type': 'application/json'})
    resp = json.loads(urllib.request.urlopen(r, timeout=600).read())
    print('gen done, tokens=', resp['usage']['completion_tokens'], f'{time.time()-t0:.1f}s', flush=True)

    print(post('/stop_profile', {}), flush=True)
    time.sleep(8)  # wait for trace flush

    # analyze: aggregate kernel time by name (TP0 trace only)
    files = sorted(glob.glob(PROFDIR + '/*TP0*.json.gz') or glob.glob(PROFDIR + '/*.json.gz') or glob.glob(PROFDIR + '/*.json'))
    print('trace files:', files, flush=True)
    f0 = files[0]
    op = gzip.open if f0.endswith('.gz') else open
    with op(f0, 'rt') as fh:
        trace = json.load(fh)
    evs = trace['traceEvents'] if isinstance(trace, dict) else trace
    agg = collections.defaultdict(float)
    cnt = collections.defaultdict(int)
    memcpy_t = 0.0
    total_kernel = 0.0
    for e in evs:
        if e.get('ph') != 'X':
            continue
        cat = e.get('cat', '')
        if cat in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            agg[e['name']] += e['dur']
            cnt[e['name']] += 1
            total_kernel += e['dur']
    print(f'total kernel time: {total_kernel/1000:.1f} ms')
    print(f'{"kernel":<90} {"ms":>9} {"cnt":>6} {"%":>6}')
    for name, dur in sorted(agg.items(), key=lambda x: -x[1])[:35]:
        print(f'{name[:90]:<90} {dur/1000:>9.2f} {cnt[name]:>6} {dur/total_kernel*100:>5.1f}%')

if __name__ == '__main__':
    main()
