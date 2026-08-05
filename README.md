# PA-STT: Persistence-Anchored Short-Sequence Temporal Transformer

Code, configurations, manifests and archived results for:

> Mohy, A.A. and Elgendi, E.O. *Short-Term Multi-Building Electricity Demand Forecasting
> via a Persistence-Anchored Residual Transformer for Energy Management.*
> Manuscript ENB-D-26-05298, Energy and Buildings.

PA-STT is a compact Transformer encoder whose zero-initialised head predicts the change
in load from the most recent reading, so the untrained network reproduces persistence
exactly and training learns only a bounded correction.

## Layout

| Path | Contents |
|---|---|
| `requirements.txt` | Python package dependencies for reviewers |
| `src/energy_forecasting/` | model, data, feature, baseline, metric and pipeline modules |
| `scripts/` | experiment runners, validation suites, and results processing scripts |
| `configs/` | experiment configuration used for the main run |
| `tools/` | cohort profiling and selection utilities |
| `manifests/` | cohort-selection manifest, run manifests with hashes, configs as executed |
| `results/main_run/` | one-hour-ahead metrics, leaderboard, diagnostics, leakage audit |
| `results/extended_validation/` | multi-horizon, cross-site, component-ablation and ramp-sensitivity metrics |
| `results/multi_seed_validation/` | three-seed validation (42, 52, 62) and significance tests |
| `results/seed_replication/` | six-seed replication of PA-STT at one hour (42, 52, 62, 72, 82, 92) with its manifest |
| `results/anchored_boosting/` | anchored gradient boosting at one hour and at every horizon; the source of the two anchored rows of Table 8 and of Supplementary Table S10 |
| `results/claim_audits/` | independent re-derivations of the headline claims, the cross-site scaler-leakage audit and the horizon-margin analysis |
| `results/paper_tables/` | every table of the paper and the supplement as published, named by its number in the paper; see `README_tables.md` there for provenance |
| `results/paper_tables_superseded_numbering/` | the earlier table CSVs, which used the pre-revision numbering; retained for reference only |

## Reconstructing the cohort

The evaluation cohort is a seeded, site-stratified draw of five buildings from each of
the 19 Building Data Genome Project 2 sites, taken after three quality thresholds fixed
before selection (history at least 1,000 h; non-zero fraction at least 0.20; mean load at
least 1 kWh), which retain 1,536 eligible buildings.

- `manifests/cohort_selected_buildings.csv` — the 95 selected buildings
- `manifests/supplementary_label_to_bdg2_identifier.csv` — maps every supplementary
  label `B01`–`B95` and `Site 1`–`Site 19` to its BDG2 identifier, so the tables in the
  supplement can be tied back to the public dataset
- `manifests/cohort_selection_summary.json` — algorithm, seed (20260730) and
  representativeness statistics
- `manifests/cohort_selection_audit.json` — source-file SHA-256 hashes and thresholds

## Data

The meter, weather and metadata files are not redistributed here. They are the public
Building Data Genome Project 2 release (Miller et al., 2020, *Scientific Data* 7, 368),
available at https://github.com/buds-lab/building-data-genome-project-2 . The SHA-256
hashes of the exact files used are recorded in `manifests/cohort_selection_audit.json`.

## Reproducibility notes

- Seeds 42, 52, and 62 for training validation, seed 20260730 for the cohort draw.
- Fixed seeds are recorded as evidence, not as a guarantee of bitwise determinism:
  mixed-precision kernels are not deterministic across driver versions. The standard
  deviation in test RMSE across the three training seeds is 0.10 kWh.
- Two seed campaigns exist and they give different noise floors. Read them together.
  `results/multi_seed_validation/` refits the whole pipeline at seeds 42, 52 and 62 and
  gives a PA-STT test RMSE of 17.30 +/- 0.10 kWh. `results/seed_replication/` refits
  PA-STT alone at six seeds (42, 52, 62, 72, 82, 92) on the same archived windows and
  gives 17.36 +/- 0.28 kWh, range 16.98 to 17.83. The wider figure is the applicable
  noise floor for any single-seed comparison in the paper, and on it the 0.19 kWh margin
  over the anchored LightGBM comparator is not resolved (one-sample t against the seed
  distribution, p = 0.39; the comparator falls inside the PA-STT seed range).
- The three-seed suite refitted the `main` variant only: `suite_manifest.json` records
  `variants: ["main"]` and `suite_ablation_summary.csv` is empty, because the suite was
  invoked with `--skip-ablations`. The `validation.ablations` block of
  `configs/revision_main.yaml` was therefore never exercised.
- The anchor ablation was closed separately by `scripts/r2_plain_transformer_seeds.py`,
  which trains the unanchored variant at the same three seeds on the same archived windows,
  changing only `residual_feature_index` (0 -> None). Pairing by seed removes the seed as a
  source of between-model variance. The anchor lowers both metrics in every seed:

  | seed | PA-STT RMSE | plain RMSE | d | PA-STT MAE | plain MAE | d |
  |---|---|---|---|---|---|---|
  | 42 | 17.204 | 17.345 | +0.141 | 5.784 | 6.643 | +0.859 |
  | 52 | 17.402 | 18.484 | +1.083 | 5.960 | 7.988 | +2.028 |
  | 62 | 17.298 | 19.336 | +2.039 | 5.786 | 10.392 | +4.606 |

  Mean effect 5.9% of RMSE and 30.0% of MAE; seed SD 0.099 anchored against 0.999
  unanchored, a tenfold reduction. With n = 3 the paired one-sided t-test does not reach
  0.05 (p = 0.093 and 0.076), so the direction is consistent and the magnitude is not
  precisely estimated. Seed 42 reproduces the published 17.3450 kWh exactly, which is the
  harness check for this comparison.
- The ramp variants and the anchored boosting variants were each fitted once, so no seed
  spread exists for them.
- At the shared seed, that suite reproduces PA-STT and all seven tabular baselines at the
  precision reported in the paper, but returns 18.74 and 17.59 kWh for the plain and
  anchored temporal convolutional networks against the 20.21 and 17.74 kWh of
  `results/main_run/`. Both convolutional baselines stop early inside a six-epoch budget
  (`patience: 2`), so the epoch early stopping selects moves their reported accuracy.
  Section 5.6 of the paper records this.
- Summary window statistics are accumulated in double precision. Recomputing them in
  single precision shifts boosting-baseline test RMSE by about 0.34 kWh, so differences
  of that order between tabular baselines should not be read as robust.

## Citation

Please cite the Energy and Buildings article. The dataset must be cited separately as
Miller et al. (2020).
