# DIE — Dual-Intervention Exploration for Vision-Language Domain Adaptation

Official code for the ICASSP submission

> **Beyond Reliable Samples: Dual-Intervention Exploration for Vision-Language Domain Adaptation**

DIE is a transductive vision-language domain adaptation (VLDA) method built on a
frozen CLIP. Instead of trusting only high-confidence pseudo-labels, DIE runs a
**class-conditioned residual audit** that scores every target sample against
every candidate class, then applies an **admission-disparity logit
intervention** that boosts under-represented classes. The two interventions
define a risk-stratified training set — a conservative **core** (hard
pseudo-labels, certificate-weighted) and an exploratory **frontier** (audit
rejected but consistently re-classified, KL soft targets with a per-class
quota).

This repository is designed to be **clean, reproducible, and GitHub-ready**:
YAML-driven experiments, a `src/` package with one module per mechanism,
mock-backend smoke tests that run on CPU without downloading any weights, and
per-transfer/per-seed JSON archives.

## Highlights

- **No pseudo-label thresholding.** Both interventions are per-class and
  audited, not confidence-gated.
- **Certificate-driven training.** Core samples are weighted by the
  class-conditioned residual certificate `c_i(k,γ')`; frontier samples use
  stop-gradient KL soft targets with a per-class quota
  `⌈β·max(1,|C_k|)⌉`.
- **Label-shift robustness.** The logit offset `b_k` is estimated from
  admission rates (`A_k`), not from prediction frequency, so frequency bias
  does not leak into the offset.
- **Reproducible by construction.** Frozen CLIP ViT-B/16, only
  `V, B_s, B_t` and the projection trained; 3 seeds; every number written to
  `outputs/**/results.json`.

## Repository layout

```text
DIE/
├── main.py                      # benchmark / run / diagnose / extract / count_params
├── data.py                      # Office-Home, VisDA-2017, DomainNet-126 loaders + cache
├── configs/
│   └── die/
│       ├── office_home.yaml     # 12 transfers, 65 classes
│       ├── visda.yaml           # synthetic -> real, 12 classes
│       ├── domainnet.yaml       # 126-class subset, 12 transfers
│       ├── diagnostics.yaml     # single transfer for mechanism analysis
│       └── ablation/            # Table ablation + label-shift variants
├── src/
│   ├── model.py                 # SemanticFactorization (V, B_s, B_t), SourceMemory
│   ├── factorization.py         # alternating block solver + E_d row prox
│   ├── audit.py                 # class-conditioned residual audit (certificate)
│   ├── intervention.py          # EMA admission stats, logit offset b, core/frontier split
│   ├── losses.py                # source, decomposition, core CE, frontier KL
│   ├── trainer.py               # warm-up + E audit/optimization rounds
│   ├── eval.py                  # benchmark driver, per-seed archives
│   ├── diagnostics.py           # admission-accuracy, round curves, label-shift
│   ├── clip_backend.py          # frozen CLIP (transformers) + mock backend
│   └── utils.py                 # seeds, logging
├── tests/test_smoke.py          # CPU smoke test (mock backend, no downloads)
├── tools/
│   ├── prepare_data.py          # layout / validation / DomainNet lists
│   ├── run_all.sh               # all experiments
│   └── analyze_results.py       # aggregate results.json -> summary
├── docs/REPRODUCE.md            # environment, data, commands, known choices
├── requirements.txt
└── LICENSE
```

## Quick start

```bash
# 1. Environment (Python 3.12+)
pip install -r requirements.txt

# 2. Data (see docs/REPRODUCE.md for official download sources)
python3 tools/prepare_data.py --data-root ./data        # validates layout
python3 tools/prepare_data.py --data-root ./data --domainnet-lists

# 3. CPU sanity check (mock backend; no weights, no images)
python3 tests/test_smoke.py

# 4. Feature cache (extracts frozen CLIP features once; ~30-60 min on GPU)
python3 main.py extract --config configs/die/office_home.yaml

# 5. A single transfer, 3 seeds
python3 main.py run --config configs/die/office_home.yaml --src Art --tgt Clipart

# 6. Full benchmark
python3 main.py benchmark --config configs/die/office_home.yaml
# or everything:
bash tools/run_all.sh
```

## Diagnostics

```bash
python3 main.py diagnose --config configs/die/diagnostics.yaml --kind admission,rounds,label_shift
```

writes `outputs/**/diagnostics/{admission,rounds,label_shift}.json`:

- `admission`: per-class admission rate vs per-class accuracy (Fig. admission)
- `rounds`: per-round core/frontier fractions and raw admission (Fig. curves)
- `label_shift`: seeded Dirichlet target subsets for the label-shift table

## Citation

```bibtex
@article{die2026,
  title={Beyond Reliable Samples: Dual-Intervention Exploration for
         Vision-Language Domain Adaptation},
  author={TBD},
  journal={TBD},
  year={2026}
}
```

## License

Apache-2.0 (see `LICENSE`). Benchmark data and CLIP weights are external assets.
# Dual-INTERVENTION-EXPLORATION
