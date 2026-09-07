#!/usr/bin/env bash
#
# Reproduce the article end to end.
#
#   ./run_all.sh            everything except the optional HTTP cross-check
#   ./run_all.sh train      re-train the two ensemble variants only
#   ./run_all.sh bench      run the measurement scripts only
#   ./run_all.sh figures    redraw the figures from the existing results/
#   ./run_all.sh loadtest   the optional HTTP cross-check (starts a local server)
#
# Every script writes its own JSON into results/ and its own log into logs/.
# The timing measurements are sensitive to load, so run this on an otherwise
# idle machine; see docs/ENVIRONMENT.md.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p logs work
PY="${PYTHON:-python3}"
export TF_CPP_MIN_LOG_LEVEL=3
# Matplotlib stamps a creation date into every PDF it writes, which would make
# the figures differ byte-for-byte between otherwise identical runs. Pinning
# SOURCE_DATE_EPOCH makes the rendered figures bit-reproducible.
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1767225600}"

run() {
    local script="$1"
    local name
    name="$(basename "$script" .py)"
    printf '%-28s ' "$name"
    local t0 t1
    t0=$(date +%s)
    if "$PY" "$script" >"logs/${name}.log" 2>&1; then
        t1=$(date +%s)
        printf 'ok  %5ds\n' "$((t1 - t0))"
    else
        t1=$(date +%s)
        printf 'FAILED after %ds -- see logs/%s.log\n' "$((t1 - t0))" "$name"
        return 1
    fi
}

stage_train() {
    echo "== training =================================================="
    run experiments/train_models.py
    run experiments/train_variants.py
}

stage_bench() {
    echo "== measurement ==============================================="
    run experiments/bench_oracle.py
    run experiments/bench_latency.py
    run experiments/bench_ensemble_cost.py
    run experiments/bench_runtimes.py
    run experiments/bench_planner.py
    run experiments/bench_baselines.py
    run experiments/bench_baselines_warm.py
    run experiments/bench_concurrency_inproc.py
    run experiments/bench_tiers.py
    run experiments/bench_tiers_alpha.py
    run experiments/bench_residual.py
    run experiments/bench_residual2.py
    run experiments/bench_degradation.py
}

stage_figures() {
    echo "== figures ==================================================="
    run figures/make_diagrams.py
    run figures/make_figures.py
    run figures/make_figures_cost.py
    run figures/make_fig6.py
}

stage_loadtest() {
    echo "== HTTP cross-check =========================================="
    echo "starting the service on 127.0.0.1:8111"
    ( cd src && SUGGEST_VARIANT=V4 "$PY" -m uvicorn service:app \
        --host 127.0.0.1 --port 8111 --log-level warning ) >logs/service.log 2>&1 &
    local pid=$!
    trap 'kill '"$pid"' 2>/dev/null || true' EXIT
    for _ in $(seq 1 60); do
        if "$PY" -c "import socket,sys; s=socket.socket(); \
            sys.exit(0 if s.connect_ex(('127.0.0.1',8111))==0 else 1)"; then break; fi
        sleep 1
    done
    run experiments/loadtest.py
    kill "$pid" 2>/dev/null || true
    trap - EXIT
}

case "${1:-all}" in
    all)      stage_train; stage_bench; stage_figures ;;
    train)    stage_train ;;
    bench)    stage_bench ;;
    figures)  stage_figures ;;
    loadtest) stage_loadtest ;;
    *) echo "usage: $0 [all|train|bench|figures|loadtest]" >&2; exit 2 ;;
esac

echo
echo "done. results in results/, figures in figures/output/, logs in logs/"
