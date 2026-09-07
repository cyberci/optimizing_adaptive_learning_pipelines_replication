"""Closed-loop load generator over the HTTP service.

Reports the end-to-end latency distribution and the deadline-satisfaction rate
against the 3 s asynchronous budget, which the client-side polling loop enforces
as ten polls at 300 ms."""

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
from repro_paths import (MODELS, RESULTS, FIGURES, WORK, DATA,   # noqa: E402
                         results, model, figure, work, data)     # noqa: E402

import asyncio, json, sys, time
import numpy as np, httpx

URL = "http://127.0.0.1:8111/games/x/suggest"
DEADLINE_MS = 3000          # 10 polls x 300 ms, the asynchronous deadline
INTERTASK_MS = 600          # synchronous inter-task window


async def worker(client, n, out, rng):
    for _ in range(n):
        body = {"abilities": list(rng.normal(1, .15, 10)),
                "prev_params": {"number_of_medusas": 10, "time_limit": 3},
                "result": bool(rng.random() < .5)}
        t = time.perf_counter()
        try:
            r = await client.post(URL, json=body, timeout=60.0)
            ok = r.status_code == 200
        except Exception:
            ok = False
        out.append(((time.perf_counter() - t) * 1e3, ok))


async def level(c, per_worker):
    rng = np.random.default_rng(1234 + c)
    out = []
    limits = httpx.Limits(max_connections=c + 8, max_keepalive_connections=c + 8)
    async with httpx.AsyncClient(limits=limits) as client:
        t0 = time.perf_counter()
        await asyncio.gather(*[worker(client, per_worker, out, rng) for _ in range(c)])
        wall = time.perf_counter() - t0
    lat = np.array([x[0] for x in out]); ok = np.array([x[1] for x in out])
    return dict(concurrency=c, n=len(out), wall_s=round(wall, 2),
                throughput_rps=round(len(out) / wall, 2),
                ok_rate=float(ok.mean()),
                p50=round(float(np.percentile(lat, 50)), 1),
                p90=round(float(np.percentile(lat, 90)), 1),
                p95=round(float(np.percentile(lat, 95)), 1),
                p99=round(float(np.percentile(lat, 99)), 1),
                max=round(float(lat.max()), 1),
                within_3s=float((lat <= DEADLINE_MS).mean()),
                within_600ms=float((lat <= INTERTASK_MS).mean()))


DEFAULT_LEVELS = "1,2,4,8,16,32,64,100"
DEFAULT_PER = 12                 # requests per worker at each level
DEFAULT_TAG = "V4"
OUT_TEMPLATE = results("results_load_{tag}.json")


async def main():
    """usage: loadtest.py [levels] [requests-per-worker] [tag]

    Defaults reproduce the cross-check reported in the article."""
    argv = sys.argv[1:]
    levels = [int(x) for x in (argv[0] if len(argv) > 0 else DEFAULT_LEVELS).split(",")]
    per = int(argv[1]) if len(argv) > 1 else DEFAULT_PER
    tag = argv[2] if len(argv) > 2 else DEFAULT_TAG
    rows = []
    async with httpx.AsyncClient() as c:                      # warm the server
        for _ in range(3):
            await c.post(URL, json={"abilities": [1.0] * 10,
                                    "prev_params": {"number_of_medusas": 10, "time_limit": 3},
                                    "result": True}, timeout=120.0)
    for c_ in levels:
        r = await level(c_, per)
        rows.append(r)
        print(f"c={c_:4d} n={r['n']:5d} thr={r['throughput_rps']:7.2f} rps  "
              f"p50={r['p50']:8.1f} p95={r['p95']:9.1f} p99={r['p99']:9.1f} ms  "
              f"<=600ms {r['within_600ms']:5.1%}  <=3s {r['within_3s']:5.1%}", flush=True)
    json.dump(rows, open(OUT_TEMPLATE.format(tag=tag), "w"), indent=2)


asyncio.run(main())
