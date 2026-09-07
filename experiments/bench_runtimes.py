"""Is the fixed per-invocation cost a framework artefact?

The same network is re-expressed as an ONNX graph and as a plain NumPy forward
pass, both built directly from the trained weights, and the cost model
c(n) = c_fixed + n * c_unit is fitted for each runtime from a batch-size sweep.
The fitted constants then predict the enumeration/search crossover analytically.
The comparison covers the framework's high-level prediction API, a direct call to
the same model object, ONNX Runtime and a dependency-free NumPy reference.
"""

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
from repro_paths import (MODELS, RESULTS, FIGURES, WORK, DATA,   # noqa: E402
                         results, model, figure, work, data)     # noqa: E402

import json, os, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
from tensorflow import keras

MODEL = model("medusa_B_corrected")
SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536]
REPS = {1: 60, 2: 60, 4: 60, 8: 60, 16: 60, 32: 60, 64: 40, 128: 40, 256: 30,
        512: 30, 1024: 20, 4096: 12, 16384: 8, 65536: 4}


BLOCKS = 5


def timeit(fn, x, reps, blocks=BLOCKS):
    """Per-call time in ms, as the minimum over `blocks` timing blocks.

    The mean over a single block is not a stable estimator on a shared machine:
    a co-tenant scheduled during the block inflates it, and repeated runs of
    this script then disagree by more than the differences it is trying to
    measure. The minimum block is the standard robust choice for a
    microbenchmark whose true cost is a floor with additive noise above it."""
    fn(x)                                   # warm up at this exact shape
    best = float("inf")
    for _ in range(blocks):
        t = time.perf_counter()
        for _ in range(reps):
            fn(x)
        best = min(best, (time.perf_counter() - t) / reps * 1e3)
    return best


def numpy_mlp(weights):
    """Plain NumPy forward pass for the 64-32-1 ability model."""
    (W1, b1), (W2, b2), (W3, b3) = weights

    def f(x):
        h = np.maximum(x @ W1 + b1, 0.0)
        h = np.maximum(h @ W2 + b2, 0.0)
        return 1.0 / (1.0 + np.exp(-(h @ W3 + b3)))
    return f



def build_onnx_mlp(weights, path):
    """Write the dense stack as an ONNX graph, straight from the Keras weights.

    The network is Dense(relu) x2 followed by Dense(sigmoid); each layer becomes
    a Gemm plus its activation, with the weights carried as initialisers."""
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    acts = ["Relu"] * (len(weights) - 1) + ["Sigmoid"]
    nodes, inits, cur = [], [], "x"
    for i, ((Wi, bi), act) in enumerate(zip(weights, acts)):
        wn, bn = f"W{i}", f"b{i}"
        inits += [numpy_helper.from_array(Wi.astype(np.float32), wn),
                  numpy_helper.from_array(bi.astype(np.float32), bn)]
        nodes.append(helper.make_node("Gemm", [cur, wn, bn], [f"g{i}"]))
        nodes.append(helper.make_node(act, [f"g{i}"], [f"a{i}"]))
        cur = f"a{i}"
    graph = helper.make_graph(
        nodes, "abilities_mlp",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT,
                                       [None, weights[0][0].shape[0]])],
        [helper.make_tensor_value_info(cur, TensorProto.FLOAT,
                                       [None, weights[-1][0].shape[1]])],
        initializer=inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    onnx.save(m, path)
    return path


def fit_cost(sizes, times):
    """Least squares fit of c(n) = c_fixed + n*c_unit over the linear regime."""
    A = np.vstack([np.ones(len(sizes)), np.asarray(sizes, float)]).T
    sol, *_ = np.linalg.lstsq(A, np.asarray(times, float), rcond=None)
    return float(sol[0]), float(sol[1])


def main():
    m1 = keras.models.load_model(f"{MODEL}/abilities_model.h5", compile=False)
    W = [(l.get_weights()[0], l.get_weights()[1]) for l in m1.layers if l.get_weights()]
    d = W[0][0].shape[0]
    npf = numpy_mlp(W)

    runtimes = {}
    agreement = {}
    xc = np.random.default_rng(0).normal(1.0, 0.2, (64, d)).astype(np.float32)
    ref = np.asarray(m1(xc, training=False))
    agreement["numpy"] = float(np.abs(npf(xc) - ref).max())
    print(f"NumPy forward pass agrees with Keras to {agreement['numpy']:.1e}")

    # ---- Keras high-level predict() -------------------------------------
    rows = []
    for n in [1, 8, 64, 512, 2048, 8192]:
        x = np.zeros((n, d), np.float32)
        rows.append((n, timeit(lambda a: m1.predict(a, verbose=0), x, 8, blocks=3)))
    runtimes["keras_predict"] = rows

    # ---- Keras direct call ----------------------------------------------
    rows = []
    for n in SIZES:
        x = np.zeros((n, d), np.float32)
        rows.append((n, timeit(lambda a: np.asarray(m1(a, training=False)), x, REPS[n])))
    runtimes["keras_call"] = rows

    # ---- ONNX Runtime ----------------------------------------------------
    # The graph is built directly from the weights of the loaded Keras model
    # rather than through a converter, so the arm depends only on onnx and
    # onnxruntime and is not exposed to converter/Keras version drift. The
    # output is checked against Keras before anything is timed.
    try:
        import onnxruntime as ort
        onnx_path = work("abilities_model.onnx")
        build_onnx_mlp(W, onnx_path)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 0
        sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        gap = float(np.abs(sess.run(None, {iname: xc})[0] - ref).max())
        if gap > 1e-5:
            raise RuntimeError(f"ONNX graph disagrees with Keras by {gap:.2e}")
        print(f"ONNX graph agrees with Keras to {gap:.1e}")
        agreement["onnxruntime"] = gap
        rows = []
        for n in SIZES:
            x = np.zeros((n, d), np.float32)
            rows.append((n, timeit(lambda a: sess.run(None, {iname: a})[0], x, REPS[n])))
        runtimes["onnxruntime"] = rows
    except Exception as e:
        print("ONNX arm skipped:", repr(e)[:200])

    # ---- NumPy reference -------------------------------------------------
    rows = []
    for n in SIZES:
        x = np.zeros((n, d), np.float32)
        rows.append((n, timeit(npf, x, REPS[n])))
    runtimes["numpy"] = rows

    out = {}
    print(f"{'runtime':16s} {'c_fixed (ms)':>13s} {'c_unit (ms)':>12s} {'ratio':>9s} "
          f"{'t(1)':>8s} {'t(64)':>8s} {'t(65536)':>10s}")
    for name, rows in runtimes.items():
        sizes = [r[0] for r in rows]
        times = [r[1] for r in rows]
        lin = [(s, t) for s, t in rows if s <= 1024]
        cf, cu = fit_cost([s for s, _ in lin], [t for _, t in lin])
        cu = max(cu, 1e-6)
        # marginal unit cost in the large-batch regime
        big = [(s, t) for s, t in rows if s >= 1024]
        cu_big = ((big[-1][1] - big[0][1]) / (big[-1][0] - big[0][0])) if len(big) > 1 else cu
        out[name] = dict(sizes=sizes, times=[round(t, 4) for t in times],
                         c_fixed=round(cf, 4), c_unit_small=round(cu, 6),
                         c_unit_large=round(max(cu_big, 1e-9), 6))
        g = dict(rows)
        print(f"{name:16s} {cf:13.3f} {cu:12.5f} {cf/max(cu,1e-9):9.0f} "
              f"{g.get(1, float('nan')):8.3f} {g.get(64, float('nan')):8.3f} "
              f"{g.get(65536, float('nan')):10.1f}")

    # ---- analytic crossover ---------------------------------------------
    # exhaustive:  c_fixed + |G| * c_unit_large
    # T-trial search: T*(c_fixed + c_unit) + T*c_book
    C_BOOK = 35.1 / 20.0     # measured Optuna bookkeeping per trial, ms
    print(f"\nanalytic crossover |G|* = [T (c_fixed + c_unit + c_book) - c_fixed] / c_unit_large")
    cross = {}
    for name in out:
        cf, cu = out[name]["c_fixed"], out[name]["c_unit_large"]
        cross[name] = {}
        for T in (20, 50, 100, 200):
            g = (T * (cf + out[name]["c_unit_small"] + C_BOOK) - cf) / cu
            cross[name][T] = int(max(g, 0))
        print(f"  {name:16s} " + "  ".join(f"T={T}: {cross[name][T]:>10,d}" for T in (20, 50, 100, 200)))
    out["_crossover"] = cross
    out["_agreement_vs_keras"] = agreement
    out["_c_book_ms"] = C_BOOK
    json.dump(out, open(results("results_runtimes.json"), "w"), indent=2)
    print("\nwrote results_runtimes.json")


main()
