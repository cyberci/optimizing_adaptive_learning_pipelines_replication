# What produces what

Every number in the article comes from one of the scripts below. Each writes a
JSON file into `results/`; the figure scripts read those files and write into
`figures/output/`. Nothing is transcribed by hand.

## Tables

| Table | Content | Produced by | Result file |
|---|---|---|---|
| 1 | The five task descriptors and their candidate-grid sizes | input data, not computed | `data/descriptors.json` |
| 2 | Two training protocols against the true latent success probability | `experiments/train_variants.py` | `results_training.json` |
| 3 | Flow-band retention against training protocol and ensemble weight | `experiments/bench_tiers_alpha.py` | `results_tiers_alpha.json` |
| 4 | Recommendation latency of the five runtime variants V0–V4 | `experiments/bench_latency.py` | `results_latency.json` |
| 5 | Cost model fitted under four serving runtimes, and the predicted crossover | `experiments/bench_runtimes.py` | `results_runtimes.json` |
| 6 | Search cost and attained objective for four samplers at four budgets | `experiments/bench_baselines.py` | `results_baselines.json` |
| 7 | Strategies for grids at and beyond the crossover | `experiments/bench_baselines_warm.py` | `results_baselines_warm.json` |

## Figures

| Figure | Content | Produced by | Reads |
|---|---|---|---|
| 1 | Architecture of the platform | `figures/make_diagrams.py` | — |
| 2 | Request sequence and the two deadlines | `figures/make_diagrams.py` | — |
| 3 | Flow-band retention by predictor configuration, and the sweep of the ensemble weight | `figures/make_fig6.py` | `results_tiers.json`, `results_tiers_alpha.json` |
| 4 | Latency of the five variants and the cost decomposition | `figures/make_figures.py` | `results_latency.json` |
| 5 | Descriptor-driven planner selection | `figures/make_figures_cost.py` | `results_planner.json`, `results_ensemble_cost.json`, `data/descriptors.json` |
| 6 | Invocation cost under four runtimes, and the predicted crossover | `figures/make_figures_cost.py` | `results_runtimes.json` |
| 7 | Closed-loop concurrency on 2 vCPU | `figures/make_figures.py` | `results_concurrency.json` |
| 8 | Flow-band retention against availability of the ML tier | `figures/make_figures.py` | `results_degradation.json`, `results_tiers.json` |

Figure files are named after the order in which they were written, not after
their number in the article: Figure 3 is `figures/output/fig6_tiers.pdf`,
Figure 4 is `fig3_latency.pdf`, Figure 5 is `fig4_planner.pdf`, Figure 6 is
`fig8_runtimes.pdf`, Figure 7 is `fig5_concurrency.pdf` and Figure 8 is
`fig7_degradation.pdf`. Figures 1 and 2 are `fig1_architecture.pdf` and
`fig2_sequence.pdf`. Each figure has exactly one producing script.

## Numbers quoted in the text but not tabulated

| Claim in the article | Produced by | Result file |
|---|---|---|
| Oracle bound of 93.7 %, and the grid-refinement control | `experiments/bench_oracle.py` | `results_oracle.json` |
| Oracle, rule-tier and static-default retention | `experiments/bench_tiers.py` | `results_tiers.json` |
| Cost model of the ensemble scoring call; the crossover of Eq. (1) | `experiments/bench_ensemble_cost.py` | `results_ensemble_cost.json` |
| Enumeration versus search across grid size | `experiments/bench_planner.py` | `results_planner.json` |
| Residual predictor error; the optimiser's-curse test | `experiments/bench_residual.py` | `results_residual.json` |
| Per-network error decomposition; the one-parameter recalibration | `experiments/bench_residual2.py` | `results_residual2.json` |
| Concurrency and deadline attainment | `experiments/bench_concurrency_inproc.py` | `results_concurrency.json` |
| HTTP cross-check of the in-process concurrency substitution | `src/service.py` + `experiments/loadtest.py` | `results_load_V4.json` |
| Degradation of the tier chain in the domain metric | `experiments/bench_degradation.py` | `results_degradation.json` |
| Non-monotone robustness of the planner | `experiments/bench_baselines.py` | `results_baselines.json` |

## A note on the identifier `A_as_deployed`

The two training protocols are tagged `A_as_deployed` and `B_corrected` in the
code, in the model directory names under `data/models/` and in the result JSON.
`A_as_deployed` is the variant the article calls **as implemented**: the label
thresholded at 0.5 and two training epochs. The tag is a historical identifier
for that configuration and carries no claim that it was ever used to serve
learners; Coglica is an experimental research platform, and every result in this
package is a simulation. Renaming the tag would have invalidated the shipped
model artefacts and result files, so it was left alone.

## Dependency order

```
train_models.py                     writes data/models/medusa, work/sim_results.csv
  └── train_variants.py             writes data/models/medusa_{A_as_deployed,B_corrected}
        ├── bench_latency.py
        ├── bench_ensemble_cost.py
        ├── bench_runtimes.py
        ├── bench_planner.py
        ├── bench_baselines.py
        ├── bench_baselines_warm.py
        ├── bench_concurrency_inproc.py
        ├── bench_tiers.py  ──► bench_tiers_alpha.py
        ├── bench_residual.py ──► bench_residual2.py
        └── bench_degradation.py
bench_oracle.py                     independent of the trained models

figures/make_diagrams.py            independent of all results
figures/make_figures.py             needs latency, concurrency, degradation, tiers
figures/make_figures_cost.py      needs planner, ensemble_cost, runtimes, descriptors
figures/make_fig6.py                needs tiers, tiers_alpha
```

The trained artefacts are shipped in `data/models/`, so the benchmark scripts can
be run without re-running the two training steps. Re-running them regenerates the
artefacts from the same seeds.
