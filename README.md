# Beyond Reliable Samples: Dual-Intervention Exploration for Vision-Language Domain Adaptation

<img width="2186" height="1040" alt="DIEframework" src="https://github.com/user-attachments/assets/0d554b56-415b-43ce-88c4-8dbea751bb08" />


DIE is a transductive vision-language domain adaptation (VLDA) method built on a frozen CLIP. 
Instead of trusting only high-confidence pseudo-labels, DIE runs a class-conditioned residual audit that scores every target sample against every candidate class, then applies an admission-disparity logit intervention that boosts under-represented classes. 
The two interventions define a risk-stratified training set — a conservative core (hard pseudo-labels, certificate-weighted) and an exploratory frontier (audit rejected but consistently re-classified, KL soft targets with a per-class quota).
This repository：YAML-driven experiments, a `src/` package with one module per mechanism, mock-backend smoke tests that run on CPU without downloading any weights, and per-transfer/per-seed JSON archives.


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

# 3. Feature cache (extracts frozen CLIP features once; ~30-60 min on GPU)
python3 main.py extract --config configs/die/office_home.yaml

# 4. Full benchmark
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
