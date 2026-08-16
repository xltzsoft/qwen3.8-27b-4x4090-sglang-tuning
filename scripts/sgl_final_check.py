#!/usr/bin/env python3
"""Final validation on :8001 (or :8000): 100k needle, 4-way concurrency, mm smoke.
Usage: python3 sgl_final_check.py <tag> [port]
"""
import urllib.request, json, time, sys, threading, os, base64, io, struct, zlib

PORT = sys.argv[2] if len(sys.argv) > 2 else '8001'
BASE = f'http://127.0.0.1:{PORT}'
KEY = 'sk-CHANGE-ME'
OUTDIR = '/root/gdn-opt/results'

PARA = ("人工智能的发展经历了多次浪潮。从符号主义到连接主义，再到深度学习的大模型时代，"
        "每一次范式转移都伴随着算力、数据与算法的共同进步。在实际工程落地中，推理效率"
        "往往决定了产品的用户体验与成本结构，因此量化、投机解码、KV缓存优化等技术成为"
        "业界关注的焦点。The quick brown fox jumps over the lazy dog. "
        "Batch size 与 latency 的权衡贯穿 serving 系统的各个层次。\n")

def chat(messages, max_tokens=300, temperature=0.0, timeout=900):
    payload = {'model': 'default', 'messages': messages, 'max_tokens': max_tokens,
               'temperature': temperature}
    t0 = time.time()
    r = urllib.request.urlopen(urllib.request.Request(
        BASE + '/v1/chat/completions', data=json.dumps(payload).encode(),
        headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'}), timeout=timeout)
    j = json.loads(r.read())
    wall = time.time() - t0
    ch = j['choices'][0]['message']
    text = (ch.get('content') or '') + (ch.get('reasoning_content') or '')
    return wall, text, j.get('usage', {})

def make_long(target_tokens, needle=True):
    text = ''
    nd = "本文档中隐藏的秘密数字是 884213，请记住它。\n"
    while len(text) < target_tokens * 3.2:
        text += PARA
        if needle and len(text) % 12000 < 3000:
            text += nd
    return text

def tiny_png_b64():
    # 64x64 red square PNG
    w = h = 64
    raw = b''.join(b'\x00' + b'\xff\x00\x00' * w for _ in range(h))
    def chunk(t, d):
        c = t + d
        return struct.pack('>I', len(d)) + c + struct.pack('>I', zlib.crc32(c))
    ihdr = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b'')
    return base64.b64encode(png).decode()

def main():
    tag = sys.argv[1]
    os.makedirs(OUTDIR, exist_ok=True)
    res = {'tag': tag, 'ts': time.strftime('%F %T'), 'port': PORT}

    # 1) 100k needle
    q = make_long(100000) + "\n\n问题：文档中隐藏的秘密数字是什么？请先给出数字，然后详细总结这段文字反复论述的主题，不少于四百字。"
    wall, text, usage = chat([{'role': 'user', 'content': q}], max_tokens=600)
    res['needle100k'] = {'ok': '884213' in text, 'wall': round(wall, 1),
                         'prompt_tokens': usage.get('prompt_tokens'),
                         'completion_tokens': usage.get('completion_tokens')}
    print('[100k needle]', res['needle100k'], flush=True)

    # 2) 4-way concurrency
    def worker(i, out):
        out[i] = chat([{'role': 'user', 'content': f'用三句话介绍分布式训练中的流水线并行（请求编号{i}）。'}], max_tokens=300)
    out = [None] * 4
    t0 = time.time()
    ths = [threading.Thread(target=worker, args=(i, out)) for i in range(4)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    agg_wall = time.time() - t0
    toks = sum(o[2].get('completion_tokens', 0) for o in out)
    res['conc4'] = {'wall': round(agg_wall, 1), 'total_completion_tokens': toks,
                    'agg_tps': round(toks / agg_wall, 1),
                    'per_req_ok': all(len(o[1]) > 50 for o in out)}
    print('[conc4]', res['conc4'], flush=True)

    # 3) mm smoke
    try:
        img = tiny_png_b64()
        wall, text, usage = chat([{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + img}},
            {'type': 'text', 'text': '这张图片是什么颜色？一句话回答。'}]}], max_tokens=200)
        res['mm'] = {'ok': len(text) > 0, 'wall': round(wall, 1), 'answer_has_red': ('红' in text or 'red' in text.lower())}
    except Exception as e:
        res['mm'] = {'ok': False, 'error': str(e)[:200]}
    print('[mm]', res['mm'], flush=True)

    path = os.path.join(OUTDIR, 'final-' + tag + '.json')
    with open(path, 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print('SAVED', path)

if __name__ == '__main__':
    main()
