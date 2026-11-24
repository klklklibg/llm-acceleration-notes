#!/usr/bin/env python3
"""Generic llama-server benchmark: tag = argv[1], bin = argv[2], model = argv[3], port = argv[4], extra args after."""
import json, os, subprocess, sys, time, urllib.request

tag, BIN, MODEL, PORT = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
extra = sys.argv[5:]

base = [os.path.join(BIN, "llama-server.exe"), "-m", MODEL, "-ngl", "99",
        "-c", "8192", "-b", "1024", "-ub", "512", "-t", "6", "-fa", "on",
        "-np", "1", "--host", "127.0.0.1", "--port", str(PORT)]
args = base + extra
logp = os.environ.get("BENCH_LOG_DIR", ".") + os.sep + f"bench_{tag}.log"
log = open(logp, "w", encoding="utf-8", errors="replace")
env = dict(os.environ)
env["GGML_CUDA_NO_PINNED"] = "0"
flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW on Windows
p = subprocess.Popen(args, cwd=BIN, stdout=log, stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, creationflags=flags, env=env)
ok = False
t0 = time.time()
try:
    for _ in range(300):
        time.sleep(1)
        if p.poll() is not None:
            break
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read()
            ok = True
            break
        except Exception:
            pass
    print(f"[{tag}] ready={ok} load={time.time()-t0:.0f}s exit={p.poll()}", flush=True)
    if not ok:
        tail = "".join(open(logp, encoding="utf-8", errors="replace").readlines()[-25:])
        print(tail, flush=True)
        raise SystemExit(1)
    LONG = ("请阅读下面的材料并回答问题。\n材料：" +
            "人工智能在医疗影像、自动驾驶、金融风控、教育等领域广泛应用，也带来隐私与就业问题。" * 60 +
            "\n问题：用三句话总结。")
    for i in range(2):
        payload = {"messages": [{"role": "user", "content": LONG}], "max_tokens": 128,
                   "temperature": 0, "stream": False}
        body = json.dumps(payload).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                     data=body, headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=900).read())
        t = r["timings"]
        msg = r["choices"][0]["message"]
        txt = (msg.get("content") or "") or (msg.get("reasoning_content") or "")
        print(f"[{tag}] run{i+1}: prompt_n={t['prompt_n']} pp={t['prompt_per_second']:.2f} "
              f"tg={t['predicted_per_second']:.2f} out={t['predicted_n']}", flush=True)
        if i == 0:
            print(f"[{tag}] text: {txt[:150].replace(chr(10),' ')}", flush=True)
        time.sleep(2)
finally:
    try:
        p.terminate(); p.wait(timeout=20)
    except Exception:
        try: p.kill()
        except Exception: pass
    log.close()
