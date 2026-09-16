# Reproducibility guide (DIE)

This document records exactly how to re-run every experiment in the paper, what
the code assumes, and which implementation choices are faithful to the paper
versus pragmatic surrogates. If a number differs from the paper, start here.

## 1. Environment

Verified on Python 3.12.11 / pip 25.0.1:

```bash
pip install -r requirements.txt
```

Key pins (what was actually verified in this repo's development):

| package | version used |
|---|---|
| torch | 2.14.0+cpu (any 2.1+ works) |
| torchvision | 0.29.0+cpu |
| transformers | 5.17.0 |
| scikit-learn | 1.6 |
| PyYAML | 6.x |

`openai/clip-vit-base-patch16` weights are downloaded by
`transformers` on first use (HF hub). The code contains a compatibility fix for
the current transformers text-encoder forward pass (position embeddings are
added manually and a 4D bidirectional attention mask is used); this is in
`src/clip_backend.py`.

## 2. Data

Official sources (not bundled):

| benchmark | source | classes | layout |
|---|---|---|---|
| Office-Home | https://www.hemanthdv.org/officeHomeDataset.html | 65 | `office_home/{Art,Clipart,Product,Real_World}/<cls>/*.jpg` |
| VisDA-2017 | http://ai.bu.edu/visda-2017/ | 12 | `visda/{synthetic,real}/<cls>/*.jpg` |
| DomainNet | https://ai.bu.edu/M3SDA/ | 126 (subset) | `domainnet/{clipart,painting,real,sketch}/<cls>/*.jpg` |

DomainNet uses the widely adopted **126-class subset** (SSDA-DomainNet
convention). Provide the class list at `data/domainnet/labels_126.txt` (one
class name per line, matching directory names). If the file is absent, the code
**falls back to all directory classes (345)** and logs a warning — do not
compare those numbers against the paper's DomainNet-126 column.

```bash
python3 tools/prepare_data.py --data-root ./data                # validate layout
python3 tools/prepare_data.py --data-root ./data --domainnet-lists  # official lists
```

Frozen CLIP features are extracted once and cached under
`data/.feat_cache/` (`*.pt` + metadata `*.json`). Delete the cache to
re-extract (e.g. after changing the backbone).

## 3. Protocol

- Frozen CLIP ViT-B/16; **only** `V, B_s, B_t`, and the learnable projection
  (logit scale) are trained. Prompts are frozen by default
  (`train_prompt: false`).
- AdamW lr=1e-3, cosine schedule over 20 epochs (3 warm-up + 17 adaptation),
  batch 64.
- Audit rounds `E = 5`, warm-up 3 epochs (offset `b = 0` during warm-up).
- Strong augmentation for core/frontier views: RandAugment(2 ops, magnitude 9)
  on reloaded images (`strong_aug: image`, faithful to the paper); weak
  augmentation (CLIP preprocess only) for the audited features.
- 3 seeds `[0, 1, 2]`; per-transfer/per-seed JSON under `outputs/<exp>/<src>-><tgt>/seed<k>/`.

## 4. Commands

```bash
python3 main.py benchmark --config configs/die/office_home.yaml
python3 main.py benchmark --config configs/die/visda.yaml
python3 main.py benchmark --config configs/die/domainnet.yaml
# ablation + label shift (Table ablation, Table shift):
for f in configs/die/ablation/*.yaml; do python3 main.py benchmark --config "$f"; done
# aggregate:
python3 tools/analyze_results.py --root outputs
# diagnostics (Fig. admission / Fig. curves / Table shift setup):
python3 main.py diagnose --config configs/die/diagnostics.yaml --kind admission,rounds,label_shift
```

Config overrides: `--set key=value` (e.g. `--set rounds=3`).

## 5. Table / figure mapping

| paper artifact | command / config | output |
|---|---|---|
| Table 1 (main results) | benchmark on `office_home.yaml`, `visda.yaml`, `domainnet.yaml` | `outputs/*/results.json` |
| Table 2 (ablation) | `configs/die/ablation/*.yaml` | `outputs/office_home_ablation/**` |
| Table 3 (label shift) | `configs/die/ablation/label_shift_a{1.0,0.5,0.1}.yaml` + diagnostics `label_shift` | subsets + accuracy |
| Fig. 2 (round curves) | `diagnose --kind rounds` | per-round core/frontier/admission |
| Fig. 3 (admission-accuracy) | `diagnose --kind admission` | per-class rates |

## 6. Known implementation choices (please read)

These are deliberate and documented deviations-or-fill-ins; the paper's text is
authoritative. If you finalize a hyper-parameter or change a mechanism,
update this section.

1. **Candidate window.** The audit scores the top-`n_candidates` classes of
   `q_0` (default 10) instead of all K classes. For DomainNet (126 classes)
   this keeps the audit tractable; classes outside the window get certificate
   zero, so they cannot enter the frontier. The paper's "every (target,
   candidate-class) pair" is implemented as this windowed version.
2. **Frontier consistency H.** The paper requires two consecutive rounds of
   re-classification agreement; at round 0 the previous mask is taken as all
   false (no historical agreement yet). Rounds are indexed 0..E-1.
3. **λ_F ramp.** Implemented as `λ_F(e) = λ_F · (e+1)/E` over audit rounds
   (0 at the start of round 0). If the final paper instead wants a ramp over
   optimization steps, set `ramp_lambda_F: false` for a constant λ_F.
4. **Audit speed.** The audit pass is a per-(target, class) donor scan with a
   CPU loop (each pair draws M=16 donors). For DomainNet (126 classes) one
   round may take tens of minutes on CPU; use GPU (`device: gpu`) for
   benchmarks. The loop is intentionally simple and deterministic (seeded).
5. **Strong augmentation cost.** `strong_aug: image` reloads images and
   re-runs the frozen ViT for core/frontier samples each round. On CPU this is
   the dominant cost; `strong_aug: feature` substitutes feature-space Gaussian
   noise (`feature_aug_std=0.02`) — a **surrogate**, not the paper's protocol.
   Use `image` for reported numbers.
6. **Warm-up periodization.** Epochs: 3 warm-up + 17 adaptation, split evenly
   into 5 rounds (last round takes the remainder). This reproduces "20 epochs
   total, E=5".
7. **`iterations_per_epoch: 1000`** is a size-independent knob: one epoch = 1000
   optimizer steps. This matches TCRT's convention and keeps training length
   comparable across datasets; it does not depend on dataset size.
8. **Batch normalization of losses.** Core/frontier losses are normalized by
   the batch size `|B_t|` (not by the number of selected samples), as in the
   paper's objective.
9. **Paper's default hyper-parameters** are used verbatim where given
   (M=16, ρ=0.5, δ=0.5, κ_r=4, Δ=2, b_max=2, β=0.3, λ_dec=1, λ_C=1, η=0.1).
   The paper marks some of these as "to be finalized by a small validation
   split"; the yaml files expose every one of them.
10. **No data fabrication.** All tables/figures in the paper are placeholders
    (`\tabph`); this repository contains no fake numbers. `results.json` only
    ever contains numbers produced by actual runs.

## 7. Extending / adding a method

- New intervention: implement in `src/intervention.py`, register in
  `DIETrainer._audit_round`.
- New audit variant: `src/audit.py::audit_batch` already exposes
  `tailmean`, `use_margin` switches (used by the ablation configs).
- New dataset: add a branch in `data.py::load_split` and a config; keep the
  class-vocabulary-alignment assertion.

## 8. Troubleshooting

- `FileNotFoundError: missing domain directory` — the expected layout is in
  `tools/prepare_data.py --data-root ./data` output; fix paths or pass
  `--set data_root=...`.
- Slow audit on CPU — see item 4; use GPU.
- HF hub blocked — set `HF_ENDPOINT=https://hf-mirror.com` and re-run; the
  weights are cached after the first successful download.
