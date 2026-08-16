#!/usr/bin/env python3
"""Compare quality sets across bench result files: check key answers + rough similarity."""
import json, sys, re, difflib

KEYS = {
    "12345": "83810205",          # 12345*6789
    "x^2": None,                   # roots 2 and 3, check both present
    "SQL": None,
    "说谎": None,
}

def check(prompt, out):
    issues = []
    if "12345" in prompt and "83810205" not in out:
        issues.append("math_wrong")
    if "x^2" in prompt:
        if not re.search(r'x\s*=\s*2|x_1\s*=\s*2|x\s*=\s*3|x_2\s*=\s*3|为\s*2|为\s*3', out):
            issues.append("roots_missing")
        if "2" not in out or "3" not in out:
            issues.append("roots_absent")
    if "说谎" in prompt:
        # correct answer: 丙说谎 (assume each tells truth/lie consistent)
        pass
    if "回文" in prompt and "def " not in out:
        issues.append("no_code")
    if "单例" in prompt and ("static" not in out and "instance" not in out.lower()):
        issues.append("singleton_missing")
    if "翻译成英文" in prompt and "inference" not in out.lower():
        issues.append("translation_suspect")
    if "SQL" in prompt and "select" not in out.lower():
        issues.append("sql_missing")
    return issues

def main():
    tags = sys.argv[1:]
    data = {}
    for t in tags:
        with open(f'/root/gdn-opt/results/{t}.json') as f:
            data[t] = json.load(f)
    n = len(data[tags[0]]['quality'])
    for i in range(n):
        p = data[tags[0]]['quality'][i]['prompt']
        print(f"--- Q{i+1}: {p[:40]}")
        ref_out = None
        for t in tags:
            out = data[t]['quality'][i]['output']
            iss = check(p, out)
            sim = ''
            if ref_out is not None:
                r = difflib.SequenceMatcher(None, ref_out, out).ratio()
                sim = f" sim_vs_{tags[0]}={r:.2f}"
            else:
                ref_out = out
            print(f"  [{t}] len={len(out)} issues={iss or 'none'}{sim}")
    for t in tags:
        d = data[t]
        print(f"[{t}] short={d['short']['server_tps']} long50k={d.get('long50k',{}).get('server_tps')} needle={d.get('long50k',{}).get('needle_ok')}")

if __name__ == '__main__':
    main()
