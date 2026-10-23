# Analysis protocol and candidate methodological contribution

## Scope and estimands

The source observation is an EIA generating unit. The primary population is
utility-scale solar, wind, storage, hydro, and other renewable units first
observed after a 12-month archive burn-in. Co-developed units are grouped by
plant, technology group, and initially reported operation month. Analyses use
inverse development-cluster-size weights where supported and resample whole
development clusters for uncertainty.

The three predictive estimands are:

1. the next reported monthly state among P, L, T, U, V, TS, O, C, and OTHER;
2. operation within 12 months of first report, among units with observable
   follow-up or an observed event; and
3. cancellation or indefinite postponement within 24 months of first report,
   under the analogous observability rule.

Features at an origin month contain only information reported through that
month. Entry cohorts are split temporally: training through December 2021,
validation during 2022, and held-out testing from January 2023 through May
2024. No automated procedure optimizes hyperparameters on test outcomes.

## Comparator set

The same stable predictors and temporal cohorts are used for the following
named specifications where the endpoint permits: empirical first-order Markov,
status/schedule logistic, full logistic, random forest, extremely randomized
trees, histogram gradient boosting, LightGBM, XGBoost, CatBoost, TabM, and the
audited convex pool. Exploratory study-specific specifications add a CUDA
delivery-state residual network, a CUDA competing-risk network, and the
standalone Delivery-PathMoE candidate. Markov is
specific to next-state prediction; the two logistic specifications occupy the
analogous transparent-baseline roles for binary landmarks. Equal-weight pooling
and removal of the neural transition prior are ablations, not additional
algorithm families.

CatBoost, XGBoost, TabM, and the custom networks use CUDA device 0. A matched
local timing profile found LightGBM CPU markedly faster than its GPU build on
the available Quadro T1000, so LightGBM uses CPU. The run manifest records
versions, hardware, and backend policy.

## Audited postprocessing pool (not the standalone proposed model)

Let `p_k(y | x)` be the probability vector from learner `k`. The global weights
are the nonnegative simplex solution

`w_global = argmin_w LogLoss(y_validation, sum_k w_k p_k)`,

subject to `w_k >= 0` and `sum_k w_k = 1`. A local simplex solution is estimated
for each reported delivery state when at least 100 validation observations and
both relevant outcome classes are available. Local weights are partially pooled
toward the global solution:

`w_state = r_state w_local + (1 - r_state) w_global`,

where `r_state = n_state / (n_state + 750)`. Sparse or unseen states therefore
fall back toward the global pool. The final probability is the convex mixture
under the state-specific weight vector. Convexity guarantees nonnegative,
normalized probabilities; automated validation checks enforce these
constraints.

For next-state prediction, the empirical Markov learner supplies a structural
transition prior within the pool. For binary landmarks, Platt calibration is
fitted in the validation cohort for each base learner before pooling. Thresholds
are also chosen in validation and are not used for proper scoring rules.

This pool is retained as an audited benchmark and ablation. It is not the
standalone proposed model and is not presented as an algorithmic novelty.

## Exploratory study-specific deep networks and Delivery-PathMoE

The delivery-state residual network embeds categorical features, standardizes
numeric features, applies gated residual blocks, and predicts the next reported
state. Its primary logits can include a positive learned multiple of a
training-only empirical Markov log-prior. An auxiliary head classifies remaining,
progressing/operating, reversing, canceling, or other transitions. A no-Markov
version is retained as an ablation.

The landmark network shares a tabular residual backbone and uses a three-class
softmax for neither event, operation within 12 months, or cancellation within
24 months. This enforces coherent mutually exclusive raw probabilities; each
binary endpoint is then calibrated using validation data under the same rule as
the other flexible learners. Two fixed candidates per network are selected by
validation log loss, and the test cohort is predicted only after checkpoint
selection.

Delivery-PathMoE is one jointly trained model and does not consume predictions
from any comparator. A piecewise-numeric tabular encoder feeds four neural
experts. A learned feature-conditioned gate activates the top two experts, and
a load-balancing penalty discourages routing collapse. The transition head
adds a learned positive multiple of a training-only empirical transition prior.
Validation-only temperature scaling is applied to the next-state probabilities.

For horizon prediction, the selected transition checkpoint is rolled forward
over reported state and state-duration probability paths. Operation and
cancellation are absorbing states. Deterministic clock features advance, while
future cumulative administrative features are frozen at the prediction origin
to prevent leakage. The operation-within-12-month and
cancellation-within-24-month probabilities therefore arise from one coherent
transition law. Automated checks enforce conservation of total path probability.

These networks, including Delivery-PathMoE, were developed after the original comparator test results had
been inspected. Their architectures are frozen for the reported run, but their
test-set comparisons are necessarily exploratory. They cannot support a
confirmatory superiority or novelty claim without evaluation on a new temporal
cohort or an external dataset. Inferior results are retained and reported.
The accompanying prior-art audit explicitly recognizes parameter-efficient
tabular ensembling, learned retrieval, generic tabular MoE, discrete-time
survival MoE, and neural semi-Markov transition models as established work.
Accordingly, Delivery-PathMoE is described only as a candidate domain-specific
methodological integration unless a systematic review and independent evidence
support a stronger novelty statement.

## Evaluation and claim rules

Multiclass evaluation reports log loss, multiclass Brier score, macro-F1,
weighted F1, per-state precision/recall/F1, and cluster-weighted counterparts.
Binary evaluation reports ROC-AUC, average precision, Brier score, log loss,
calibration intercept/slope, threshold metrics, calibration bins, and decision
curves. Fit and prediction times are saved with the backend.

Uncertainty is estimated by resampling plant-development clusters. In addition
to model-specific confidence intervals, paired bootstrap differences compare
the proposed method with every comparator. A proposed-model win is supported
only when the metric direction is favorable and the paired 95% interval excludes
zero. Intervals spanning zero are described as inconclusive. If different
models lead different metrics, the result is reported as a trade-off rather
than universal dominance.

Adjusted logistic models are descriptive associations, not causal effects.
They use three delivery phases, collapsed sparse technology/sector levels, and
entity-clustered robust covariance. Exact collinearity at entry is removed and
convergence/finite-inference checks are mandatory.
