
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
from repro_paths import (MODELS, RESULTS, FIGURES, WORK, DATA,   # noqa: E402
                         results, model, figure, work, data)     # noqa: E402

import os, json, time, pickle
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3")
import numpy as np
from tensorflow import keras
import engine, cognitive_sim as cs

rng = np.random.default_rng(7)
REPS = {"V0":12, "V1":12, "V2":12, "V3":40, "V4":200}
prev = {"number_of_medusas":10, "time_limit":3}

def pct(a,q): return float(np.percentile(a,q))

out = {}
for v, reps in REPS.items():
    e = engine.Engine(v)
    e.suggest(rng.normal(1,.15,10), prev, True)          # discard first
    lat=[]
    for _ in range(reps):
        ab = rng.normal(1,.15,10)
        t=time.perf_counter(); e.suggest(ab, prev, bool(rng.random()<.5)); lat.append((time.perf_counter()-t)*1e3)
    lat=np.array(lat)
    out[v] = {"n":reps,"mean":float(lat.mean()),"p50":pct(lat,50),"p90":pct(lat,90),
              "p95":pct(lat,95),"p99":pct(lat,99),"max":float(lat.max())}
    print(f"{v:3s} n={reps:3d} mean={lat.mean():8.1f}  p50={pct(lat,50):8.1f}  p95={pct(lat,95):8.1f}  p99={pct(lat,99):8.1f} ms")

# ---- cost decomposition ----
d={}
t=time.perf_counter()
for _ in range(5):
    keras.models.load_model(model("medusa", "abilities_model.h5"),compile=False)
    keras.models.load_model(model("medusa", "games_model.h5"),compile=False)
d["model_load_ms"]=(time.perf_counter()-t)/5*1e3

e3=engine.Engine("V3"); m1,m2,meta=e3.cache["medusa"]
x1=np.zeros((1,12),np.float32); x2=[np.zeros((1,3),np.float32),np.zeros((1,2),np.float32)]
m1.predict(x1,verbose=0); m1(x1,training=False)
t=time.perf_counter()
for _ in range(30): m1.predict(x1,verbose=0); m2.predict(x2,verbose=0)
d["predict_api_per_eval_ms"]=(time.perf_counter()-t)/30*1e3
t=time.perf_counter()
for _ in range(300): m1(x1,training=False); m2(x2,training=False)
d["direct_call_per_eval_ms"]=(time.perf_counter()-t)/300*1e3
xb1=np.zeros((80,12),np.float32); xb2=[np.zeros((80,3),np.float32),np.zeros((80,2),np.float32)]
m1(xb1,training=False)
t=time.perf_counter()
for _ in range(300): m1(xb1,training=False); m2(xb2,training=False)
d["batch80_call_ms"]=(time.perf_counter()-t)/300*1e3
d["batch80_per_eval_ms"]=d["batch80_call_ms"]/80

import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
t=time.perf_counter()
for _ in range(10):
    s=optuna.create_study(direction="minimize")
    s.optimize(lambda tr:(tr.suggest_int("a",5,20)+tr.suggest_int("b",1,5))*0.0+0.1, n_trials=20)
d["tpe_overhead_20trials_ms"]=(time.perf_counter()-t)/10*1e3
out["decomposition"]={k:round(v,4) for k,v in d.items()}
for k,v in d.items(): print(f"  {k:28s} {v:9.3f} ms")
json.dump(out, open(results("results_latency.json"),"w"), indent=2)
