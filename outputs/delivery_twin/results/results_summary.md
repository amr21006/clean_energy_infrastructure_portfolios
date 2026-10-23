# Results: multistate digital twin of clean-infrastructure delivery

## Cohort and data integrity

The continuous archive contained 129 monthly snapshots and 10,262 ever-planned generating units after identity harmonization. The prespecified clean-infrastructure cohort contained 7,172 generating units grouped into 6,713 plant-development clusters. Of these, 5,249 had at least 24 months of potential follow-up.

At the May 2026 inventory, 4,739 units had entered operation or later retired, 529 were canceled or indefinitely postponed, 1,751 remained planned, and 153 could not be reconciled to a terminal inventory identity. Unresolved identities were treated as censored, not as failures or completions.

## Delivery-state dynamics

The analysis identified 118,142 strictly immediate monthly transitions. State-specific persistence, progression, reversal, operation, cancellation, and transition entropy are reported in `tables/transition_metrics.csv`; the complete transition matrix is reported in `tables/transition_probabilities.csv`.

- From state P, the next-month probabilities were 96.2% remaining, 2.9% advancing, 0.0% reversing, 0.1% entering operation, and 0.8% canceling.
- From state L, the next-month probabilities were 94.2% remaining, 4.7% advancing, 0.2% reversing, 0.1% entering operation, and 0.8% canceling.
- From state T, the next-month probabilities were 90.6% remaining, 8.2% advancing, 0.4% reversing, 0.3% entering operation, and 0.6% canceling.
- From state U, the next-month probabilities were 86.5% remaining, 11.1% advancing, 0.3% reversing, 2.0% entering operation, and 0.1% canceling.
- From state V, the next-month probabilities were 82.6% remaining, 6.2% advancing, 0.2% reversing, 10.9% entering operation, and 0.0% canceling.
- From state TS, the next-month probabilities were 76.4% remaining, 0.0% advancing, 0.9% reversing, 22.7% entering operation, and 0.0% canceling.
- From state OTHER, the next-month probabilities were 0.0% remaining, 0.0% advancing, 0.0% reversing, 1.7% entering operation, and 1.7% canceling.
- From state O, the next-month probabilities were 0.0% remaining, 0.0% advancing, 0.0% reversing, 0.0% entering operation, and 0.0% canceling.

## Schedule and pathway instability

Technology- and outcome-specific schedule revision, cumulative slippage, absolute schedule movement, and state reversal estimates are reported in `tables/schedule_and_reversal_summary.csv`. The most frequent collapsed process paths are reported in `tables/top_pathways.csv`. These quantities describe reported administrative trajectories and are not contractual delay measures.

## Held-out temporal-test performance

13 named model specifications were compared under the same cohort, feature-origin, and temporal-split rules (excluding the equal-weight and no-Markov ablations). CatBoost, XGBoost, TabM, and the custom neural networks used CUDA; LightGBM used CPU after a matched local runtime profile showed that backend to be faster on the available hardware.

For next-month state prediction, catboost_gpu had the lowest observed test log loss (0.333), macro-F1 of 0.743, and multiclass Brier score of 0.166. Performance by state, including rare cancellation transitions, is reported separately to prevent overall accuracy from concealing minority-class failure.

The earlier delivery-aware postprocessing pool is retained as an audited benchmark, not as the standalone proposed algorithm. Its test log loss was 0.334 for next-state prediction, 0.397 for 12-month operation, and 0.195 for 24-month cancellation. Paired plant-development-cluster bootstrap differences against every comparator are in the `*_proposed_paired_comparisons.csv` files; an interval spanning zero is not interpreted as evidence of superiority.

Study-specific deep-learning models were evaluated as frozen exploratory specifications: a Markov-residual multistate network, a coherent three-class competing-risk network, and a standalone sparse MoE family. Their development followed inspection of the benchmark test results; consequently, their test comparisons are not confirmatory evidence of novelty or general superiority. The state network's test log loss was 0.375 (macro-F1 0.718), versus 0.377 without its Markov prior. The competing-risk network's calibrated test log loss was 0.410 for operation and 0.182 for cancellation; its cancellation ROC-AUC was 0.785 and average precision was 0.162. Paired exploratory intervals are saved in the `*_custom_deep_paired_comparisons.csv` files.

The proposed standalone Delivery-PathMoE does not pool comparator predictions. Its transition head uses one feature-conditioned sparse-MoE transition law, and its horizon risks are obtained by state-duration path rollout with audited probability-mass conservation. The transition MoE's test log loss was 0.342 (macro-F1 0.729). Its direct competing-risk head had operation/cancellation log losses of 0.415/0.183; coherent path rollout produced 0.411/0.186. These observed results are compared with every benchmark in the `*_delivery_sparse_moe_paired_comparisons.csv` and `*_delivery_path_moe_paired_comparisons.csv` files. The method is a candidate domain-specific integration, not an established globally novel or universally superior algorithm.

For operation within 12 months at first report, proposed_delivery_ensemble had the lowest test Brier score (0.133); its ROC-AUC was 0.891, average precision 0.821, calibration intercept 0.205, and calibration slope 0.878.

For cancellation within 24 months, delivery_competing_risk_net_gpu had the lowest test Brier score (0.048); its ROC-AUC was 0.785, average precision 0.162, calibration intercept 1.352, and calibration slope 1.516. Because cancellation is uncommon, average precision, calibration, and decision curves should be emphasized over accuracy.

Cluster-bootstrap 95% confidence intervals are available in the corresponding `*_bootstrap_ci.csv` files. Calibration bins, decision curves, temporal test predictions, and subgroup generalization results are exported separately.

## Interpretation boundaries

The results support prediction and process characterization. They do not identify causal effects of technology, sector, organization, or geography. Entity ID is a reporting-organization identifier and is not assumed to identify an EPC contractor. Generating units are the source observation; inverse plant-development-cluster weights and cluster bootstrap intervals reduce the influence of multi-unit projects.

EIA-860M values are preliminary and may be corrected in subsequent inventories. Operation outcomes were reconciled to annual/final inventory fields, but unobserved early-stage proposals and contractual milestones are outside the dataset's scope.

## Reproducibility and validation

All 18 implemented validation checks passed. The validation evidence is recorded in `results/validation_checks.csv` and `results/validation_checks.json`. Model selection used the validation cohort; the 2023-May 2024 entry cohort was retained for temporal testing of the prespecified benchmarks. The custom networks were developed after those test results were inspected and are therefore explicitly exploratory. No automated test-set hyperparameter optimization is implemented.
