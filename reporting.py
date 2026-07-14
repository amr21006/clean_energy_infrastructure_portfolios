from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import AnalysisConfig


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_run_manifest(config: AnalysisConfig) -> dict[str, Any]:
    download_manifest = config.raw_dir / "download_manifest.csv"
    package_names = [
        "catboost",
        "lightgbm",
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "scipy",
        "statsmodels",
        "tabm",
        "torch",
        "xgboost",
    ]
    package_versions = {}
    for package in package_names:
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    try:
        gpu_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except Exception as error:
        gpu_query = f"UNAVAILABLE: {error!r}"
    manifest = {
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions,
        "gpu_inventory": gpu_query,
        "backend_policy": {
            "catboost": "cuda_gpu",
            "xgboost": "cuda_gpu",
            "tabm": "cuda_gpu",
            "delivery_state_residual_network": "cuda_gpu_exploratory",
            "delivery_competing_risk_network": "cuda_gpu_exploratory",
            "delivery_sparse_moe": "cuda_gpu_exploratory_standalone_model",
            "lightgbm": "cpu_after_matched_runtime_profile",
            "classic_models": "cpu",
            "proposed_ensemble": "hybrid_postprocessing",
        },
        "raw_download_manifest": str(download_manifest),
        "raw_download_manifest_sha256": file_sha256(download_manifest) if download_manifest.exists() else None,
    }
    with (config.results_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)
    return manifest


def table_index(config: AnalysisConfig) -> pd.DataFrame:
    rows = []
    for path in sorted(config.tables_dir.glob("*.csv")):
        try:
            frame = pd.read_csv(path)
            rows.append(
                {
                    "file": path.name,
                    "rows": len(frame),
                    "columns": len(frame.columns),
                    "column_names": " | ".join(frame.columns),
                }
            )
        except Exception as error:
            rows.append(
                {
                    "file": path.name,
                    "rows": np.nan,
                    "columns": np.nan,
                    "column_names": f"READ ERROR: {error!r}",
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(config.results_dir / "table_index.csv", index=False, encoding="utf-8-sig")
    return result


def analysis_asset_manifest(config: AnalysisConfig) -> pd.DataFrame:
    rows = [
        ("Figure 1", "figure", "Cohort construction flow", "cohort_flow.csv", "stage | units", "Flow diagram; report exclusions without implying missingness is random."),
        ("Figure 2", "figure", "Monthly delivery-state occupancy", "state_occupancy_monthly.csv", "report_month | current_state | units | capacity_mw", "Stacked area or small multiples; units and capacity should be separate panels."),
        ("Figure 3", "figure", "Immediate multistate transition matrix", "transition_probabilities.csv", "current_state | next_state | probability", "Heat map with counts available from transition_counts.csv."),
        ("Figure 4", "figure", "Dominant reported delivery pathways", "top_pathways.csv", "path | units | share", "Alluvial or ranked path plot; label as administrative pathways."),
        ("Figure 5", "figure", "Schedule movement and reversals", "schedule_and_reversal_summary.csv", "technology_group | terminal_category | schedule metrics", "Point-range or distribution summary; not contractual delay."),
        ("Figure 6", "figure", "Temporal-test model comparison", "next_state_model_performance.csv | operation_12m_model_performance.csv | cancellation_24m_model_performance.csv", "model | metric columns | backend | fit_seconds", "Separate proper scoring, ranking, and classification metrics."),
        ("Figure 7", "figure", "Calibration", "operation_12m_calibration_bins.csv | cancellation_24m_calibration_bins.csv", "model | mean_predicted | observed_rate | observations", "Reliability panels with 45-degree reference."),
        ("Figure 8", "figure", "Decision curves", "operation_12m_decision_curve.csv | cancellation_24m_decision_curve.csv", "model | threshold | net_benefit", "Emphasize plausible threshold range and treat-all/none references."),
        ("Figure 9", "figure", "Paired differences for audited pooling and exploratory standalone networks", "next_state_proposed_paired_comparisons.csv | operation_12m_proposed_paired_comparisons.csv | cancellation_24m_proposed_paired_comparisons.csv | next_state_custom_deep_paired_comparisons.csv | operation_12m_custom_deep_paired_comparisons.csv | cancellation_24m_custom_deep_paired_comparisons.csv | next_state_delivery_sparse_moe_paired_comparisons.csv | operation_12m_delivery_sparse_moe_paired_comparisons.csv | cancellation_24m_delivery_sparse_moe_paired_comparisons.csv | operation_12m_delivery_path_moe_paired_comparisons.csv | cancellation_24m_delivery_path_moe_paired_comparisons.csv", "reference model | comparator | metric | point delta | ci95_low | ci95_high", "Forest plot; keep the postprocessing pool and exploratory standalone-network comparisons visibly separated."),
        ("Figure 10", "figure", "Subgroup temporal generalization", "operation_12m_subgroup_generalization.csv | cancellation_24m_subgroup_generalization.csv", "model | dimension | subgroup | n | metric columns", "Suppress small or single-class cells already excluded by code."),
        ("Table 1", "table", "Cohort characteristics", "project_characteristics_by_split.csv", "grouping columns | n | capacity and timing summaries", "Primary descriptive table."),
        ("Table 2", "table", "Terminal outcomes by technology and sector", "outcomes_by_technology.csv | outcomes_by_sector.csv", "group | terminal_category | units | share", "Do not treat unresolved as failure."),
        ("Table 3", "table", "Transition metrics and spells", "transition_metrics.csv | state_spell_summary.csv", "state | persistence/progression/reversal | duration", "Report medians/IQR alongside transition probabilities."),
        ("Table 4", "table", "Model performance", "*_model_performance.csv | *_bootstrap_ci.csv", "model | test metrics | confidence intervals", "Keep validation selection tables in supplement."),
        ("Table 5", "table", "Adjusted non-causal associations", "adjusted_association_odds_ratios.csv", "term | odds_ratio | ci95_low | ci95_high | p_value", "Label as adjusted associations; include model audit."),
        ("Supplement", "table", "Data and model audit", "data_dictionary.csv | validation_checks.csv | reviewer_risk_audit.csv | run_manifest.json", "audit-specific", "Supports reproducibility and reviewer checks."),
    ]
    manifest = pd.DataFrame(
        rows,
        columns=[
            "asset_id",
            "asset_type",
            "proposed_title",
            "source_files",
            "required_fields",
            "interpretation_note",
        ],
    )
    manifest.to_csv(
        config.results_dir / "manuscript_asset_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return manifest


def frontier_comparator_audit(config: AnalysisConfig) -> pd.DataFrame:
    rows = [
        {
            "method": "TabM",
            "status": "included",
            "scope": "next-state and both binary landmarks",
            "reason": "Open ICLR 2025 implementation; CUDA PyTorch and local memory were sufficient.",
            "source": "https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html",
        },
        {
            "method": "TabPFN-2.5",
            "status": "excluded_no_result",
            "scope": "binary landmarks would satisfy the stated row range; transition training exceeds the nominal 50,000-row range",
            "reason": "The official weights require explicit user authentication and acceptance of a non-commercial model license. Official local-use guidance recommends roughly 8 GB GPU memory; the available GPU has 4 GB. No metric was imputed or inferred.",
            "source": "https://github.com/PriorLabs/tabpfn",
        },
        {
            "method": "CatBoost / XGBoost / LightGBM",
            "status": "included",
            "scope": "next-state and both binary landmarks",
            "reason": "Strong gradient-boosted decision-tree comparators; CatBoost and XGBoost used CUDA, while a matched local profile selected CPU for LightGBM.",
            "source": "https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html",
        },
        {
            "method": "Custom delivery-state residual network / competing-risk network",
            "status": "included_exploratory",
            "scope": "next-state and both binary landmarks, with a no-Markov state-network ablation",
            "reason": "CUDA PyTorch implementations developed for this study. Because their development followed inspection of the benchmark test results, their held-out comparisons are explicitly exploratory and require independent temporal or external confirmation.",
            "source": "local implementation: src/eia_delivery_twin/deep_models.py",
        },
        {
            "method": "Delivery sparse mixture-of-experts network",
            "status": "included_exploratory",
            "scope": "standalone next-state and coherent competing-risk heads",
            "reason": "One jointly trained CUDA model with a single tabular encoder, four neural experts, top-2 feature-conditioned routing, load-balancing regularization, and no comparator prediction inputs. Development followed test inspection, so results require new-cohort confirmation.",
            "source": "local implementation: src/eia_delivery_twin/deep_models.py",
        },
    ]
    audit = pd.DataFrame(rows)
    audit.to_csv(
        config.results_dir / "frontier_comparator_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return audit


def proposed_method_prior_art_audit(config: AnalysisConfig) -> pd.DataFrame:
    rows = [
        {
            "prior_method": "TabM",
            "established_component": "Parameter-efficient ensembling for tabular MLPs",
            "implication": "Neither shared MLP members nor efficient internal ensembling can be claimed as new here.",
            "source": "https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html",
            "source_status": "ICLR 2025",
        },
        {
            "prior_method": "ModernNCA",
            "established_component": "Deep differentiable neighborhood retrieval for tabular learning",
            "implication": "Learned instance retrieval is established prior art and is not used as the proposed distinction.",
            "source": "https://proceedings.iclr.cc/paper_files/paper/2025/hash/ecd294194d5155432aab533f7ce0f6fe-Abstract-Conference.html",
            "source_status": "ICLR 2025",
        },
        {
            "prior_method": "MixturePFN",
            "established_component": "Sparse expert routing for tabular in-context learners",
            "implication": "A generic tabular MoE claim would be false; expert routing is established.",
            "source": "https://proceedings.iclr.cc/paper_files/paper/2025/hash/3de60dc2ff7ac8866e805bbf457e2291-Abstract-Conference.html",
            "source_status": "ICLR 2025",
        },
        {
            "prior_method": "Dual MoE for discrete-time survival",
            "established_component": "Feature-encoder and time-dependent hazard mixtures of experts",
            "implication": "MoE survival modeling is prior art; the candidate distinction cannot rest on combining MoE with event-time prediction.",
            "source": "https://arxiv.org/abs/2510.26014",
            "source_status": "2025 preprint/workshop paper",
        },
        {
            "prior_method": "Neural EM for semi-competing risks",
            "established_component": "Semi-Markov transition hazards with transition-specific neural subnetworks",
            "implication": "Neural multistate and sojourn-time modeling are prior art.",
            "source": "https://arxiv.org/abs/2212.12028",
            "source_status": "preprint",
        },
        {
            "prior_method": "Delivery-PathMoE (this study)",
            "established_component": "Nine-state administrative delivery process; state-duration sparse routing; one transition law rolled forward to coherent operation/cancellation horizons",
            "implication": "Treat only as a candidate domain-specific methodological integration. Global algorithmic novelty requires a systematic review, independent implementation comparison, and new-cohort or external validation.",
            "source": "local implementation: src/eia_delivery_twin/deep_models.py",
            "source_status": "exploratory study implementation",
        },
    ]
    audit = pd.DataFrame(rows)
    audit.to_csv(
        config.results_dir / "proposed_method_prior_art_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return audit


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def _best_model(performance: pd.DataFrame, metric: str, lower_better: bool) -> pd.Series:
    ordered = performance.sort_values(metric, ascending=lower_better)
    return ordered.iloc[0]


def generate_results_report(
    config: AnalysisConfig,
    projects: pd.DataFrame,
    panel: pd.DataFrame,
    transitions: pd.DataFrame,
    landmarks: pd.DataFrame,
    descriptive: dict[str, pd.DataFrame],
    models: dict[str, pd.DataFrame],
    validation: pd.DataFrame,
) -> str:
    primary = projects.loc[projects["eligible_primary_cohort"]].copy()
    mature = primary.loc[primary["has_24_month_potential_followup"]]
    operation_perf = models["operation_12m_performance"]
    cancel_perf = models["cancellation_24m_performance"]
    next_perf = models["next_state_performance"]
    best_operation = _best_model(operation_perf, "brier", True)
    best_cancel = _best_model(cancel_perf, "brier", True)
    best_next = _best_model(next_perf, "log_loss", True)
    proposed_next = next_perf.loc[
        next_perf["model"] == "proposed_delivery_ensemble"
    ].iloc[0]
    proposed_operation = operation_perf.loc[
        operation_perf["model"] == "proposed_delivery_ensemble"
    ].iloc[0]
    proposed_cancel = cancel_perf.loc[
        cancel_perf["model"] == "proposed_delivery_ensemble"
    ].iloc[0]
    custom_next = next_perf.loc[
        next_perf["model"] == "delivery_state_residual_net_gpu"
    ].iloc[0]
    custom_next_ablation = next_perf.loc[
        next_perf["model"] == "delivery_net_no_markov_ablation"
    ].iloc[0]
    custom_operation = operation_perf.loc[
        operation_perf["model"] == "delivery_competing_risk_net_gpu"
    ].iloc[0]
    custom_cancel = cancel_perf.loc[
        cancel_perf["model"] == "delivery_competing_risk_net_gpu"
    ].iloc[0]
    moe_next = next_perf.loc[
        next_perf["model"] == "delivery_sparse_moe_gpu"
    ].iloc[0]
    moe_operation = operation_perf.loc[
        operation_perf["model"] == "delivery_sparse_moe_gpu"
    ].iloc[0]
    moe_cancel = cancel_perf.loc[
        cancel_perf["model"] == "delivery_sparse_moe_gpu"
    ].iloc[0]
    path_operation = operation_perf.loc[
        operation_perf["model"] == "delivery_path_moe_gpu"
    ].iloc[0]
    path_cancel = cancel_perf.loc[
        cancel_perf["model"] == "delivery_path_moe_gpu"
    ].iloc[0]
    ablations = {"equal_weight_pool_ablation", "delivery_net_no_markov_ablation"}
    compared_models = int(
        max(
            len(next_perf.loc[~next_perf["model"].isin(ablations)]),
            len(
                operation_perf.loc[
                    ~operation_perf["model"].isin(ablations)
                ]
            ),
        )
    )
    transition_metrics = descriptive["transition_metrics"]
    schedule = descriptive["schedule_and_reversal_summary"]

    terminal_counts = primary["terminal_category"].value_counts()
    lines = [
        "# Results: multistate digital twin of clean-infrastructure delivery",
        "",
        "## Cohort and data integrity",
        "",
        (
            f"The continuous archive contained {panel['report_month'].nunique():,} monthly "
            f"snapshots and {panel['unit_id'].nunique():,} ever-planned generating units after "
            "identity harmonization. The prespecified clean-infrastructure cohort contained "
            f"{len(primary):,} generating units grouped into "
            f"{primary['development_cluster_id'].nunique():,} plant-development clusters. "
            f"Of these, {len(mature):,} had at least 24 months of potential follow-up."
        ),
        "",
        (
            f"At the May 2026 inventory, {terminal_counts.get('operating', 0) + terminal_counts.get('retired', 0):,} "
            f"units had entered operation or later retired, {terminal_counts.get('canceled', 0):,} "
            f"were canceled or indefinitely postponed, {terminal_counts.get('planned', 0):,} "
            f"remained planned, and {terminal_counts.get('unresolved', 0):,} could not be reconciled "
            "to a terminal inventory identity. Unresolved identities were treated as censored, not "
            "as failures or completions."
        ),
        "",
        "## Delivery-state dynamics",
        "",
        (
            f"The analysis identified {len(transitions.loc[transitions['eligible_primary_cohort']]):,} "
            "strictly immediate monthly transitions. State-specific persistence, progression, "
            "reversal, operation, cancellation, and transition entropy are reported in "
            "`tables/transition_metrics.csv`; the complete transition matrix is reported in "
            "`tables/transition_probabilities.csv`."
        ),
        "",
    ]
    for _, row in transition_metrics.iterrows():
        lines.append(
            f"- From state {row['current_state']}, the next-month probabilities were "
            f"{_percent(row['remain_probability'])} remaining, "
            f"{_percent(row['advance_probability'])} advancing, "
            f"{_percent(row['reversal_probability'])} reversing, "
            f"{_percent(row['operation_probability'])} entering operation, and "
            f"{_percent(row['cancellation_probability'])} canceling."
        )
    lines.extend(
        [
            "",
            "## Schedule and pathway instability",
            "",
            (
                "Technology- and outcome-specific schedule revision, cumulative slippage, absolute "
                "schedule movement, and state reversal estimates are reported in "
                "`tables/schedule_and_reversal_summary.csv`. The most frequent collapsed process "
                "paths are reported in `tables/top_pathways.csv`. These quantities describe reported "
                "administrative trajectories and are not contractual delay measures."
            ),
            "",
            "## Held-out temporal-test performance",
            "",
            (
                f"{compared_models} named model specifications were compared under the same cohort, "
                "feature-origin, and temporal-split rules (excluding the equal-weight "
                "and no-Markov ablations). CatBoost, XGBoost, TabM, and the custom neural networks "
                "used CUDA; LightGBM used CPU after a matched local runtime profile showed that "
                "backend to be faster on the available hardware."
            ),
            "",
            (
                f"For next-month state prediction, {best_next['model']} had the lowest observed test "
                f"log loss ({best_next['log_loss']:.3f}), macro-F1 of {best_next['macro_f1']:.3f}, "
                f"and multiclass Brier score of {best_next['multiclass_brier']:.3f}. Performance by "
                "state, including rare cancellation transitions, is reported separately to prevent "
                "overall accuracy from concealing minority-class failure."
            ),
            "",
            (
                "The earlier delivery-aware postprocessing pool is retained as an audited benchmark, "
                "not as the standalone proposed algorithm. "
                f"Its test log loss was {proposed_next['log_loss']:.3f} for next-state prediction, "
                f"{proposed_operation['log_loss']:.3f} for 12-month operation, and "
                f"{proposed_cancel['log_loss']:.3f} for 24-month cancellation. Paired "
                "plant-development-cluster bootstrap differences against every comparator are in "
                "the `*_proposed_paired_comparisons.csv` files; an interval spanning zero is not "
                "interpreted as evidence of superiority."
            ),
            "",
            (
                "Study-specific deep-learning models were evaluated as frozen exploratory "
                "specifications: a Markov-residual multistate network, a coherent three-class "
                "competing-risk network, and a standalone sparse MoE family. Their development "
                "followed inspection of the benchmark "
                "test results; consequently, their test comparisons are not confirmatory evidence "
                "of novelty or general superiority. The state network's test log loss was "
                f"{custom_next['log_loss']:.3f} (macro-F1 {custom_next['macro_f1']:.3f}), versus "
                f"{custom_next_ablation['log_loss']:.3f} without its Markov prior. The competing-risk "
                f"network's calibrated test log loss was {custom_operation['log_loss']:.3f} for "
                f"operation and {custom_cancel['log_loss']:.3f} for cancellation; its cancellation "
                f"ROC-AUC was {custom_cancel['roc_auc']:.3f} and average precision was "
                f"{custom_cancel['average_precision']:.3f}. Paired exploratory intervals are saved "
                "in the `*_custom_deep_paired_comparisons.csv` files."
            ),
            "",
            (
                "The proposed standalone Delivery-PathMoE does not pool comparator predictions. "
                "Its transition head uses one feature-conditioned sparse-MoE transition law, and "
                "its horizon risks are obtained by state-duration path rollout with audited "
                "probability-mass conservation. The transition MoE's test log loss was "
                f"{moe_next['log_loss']:.3f} (macro-F1 {moe_next['macro_f1']:.3f}). Its direct "
                f"competing-risk head had operation/cancellation log losses of "
                f"{moe_operation['log_loss']:.3f}/{moe_cancel['log_loss']:.3f}; coherent path "
                f"rollout produced {path_operation['log_loss']:.3f}/{path_cancel['log_loss']:.3f}. "
                "These observed results are compared with every benchmark in the "
                "`*_delivery_sparse_moe_paired_comparisons.csv` and "
                "`*_delivery_path_moe_paired_comparisons.csv` files. The method is a candidate "
                "domain-specific integration, not an established globally novel or universally "
                "superior algorithm."
            ),
            "",
            (
                f"For operation within 12 months at first report, {best_operation['model']} had the "
                f"lowest test Brier score ({best_operation['brier']:.3f}); its ROC-AUC was "
                f"{best_operation['roc_auc']:.3f}, average precision "
                f"{best_operation['average_precision']:.3f}, calibration intercept "
                f"{best_operation['calibration_intercept']:.3f}, and calibration slope "
                f"{best_operation['calibration_slope']:.3f}."
            ),
            "",
            (
                f"For cancellation within 24 months, {best_cancel['model']} had the lowest test "
                f"Brier score ({best_cancel['brier']:.3f}); its ROC-AUC was "
                f"{best_cancel['roc_auc']:.3f}, average precision "
                f"{best_cancel['average_precision']:.3f}, calibration intercept "
                f"{best_cancel['calibration_intercept']:.3f}, and calibration slope "
                f"{best_cancel['calibration_slope']:.3f}. Because cancellation is uncommon, average "
                "precision, calibration, and decision curves should be emphasized over accuracy."
            ),
            "",
            "Cluster-bootstrap 95% confidence intervals are available in the corresponding "
            "`*_bootstrap_ci.csv` files. Calibration bins, decision curves, temporal test "
            "predictions, and subgroup generalization results are exported separately.",
            "",
            "## Interpretation boundaries",
            "",
            "The results support prediction and process characterization. They do not identify "
            "causal effects of technology, sector, organization, or geography. Entity ID is a "
            "reporting-organization identifier and is not assumed to identify an EPC contractor. "
            "Generating units are the source observation; inverse plant-development-cluster weights "
            "and cluster bootstrap intervals reduce the influence of multi-unit projects.",
            "",
            "EIA-860M values are preliminary and may be corrected in subsequent inventories. "
            "Operation outcomes were reconciled to annual/final inventory fields, but unobserved "
            "early-stage proposals and contractual milestones are outside the dataset's scope.",
            "",
            "## Reproducibility and validation",
            "",
            f"All {int(validation['passed'].sum())} implemented validation checks passed. "
            "The validation evidence is recorded in `results/validation_checks.csv` and "
            "`results/validation_checks.json`. Model selection used the validation cohort; the "
            "2023-May 2024 entry cohort was retained for temporal testing of the prespecified "
            "benchmarks. The custom networks were developed after those test results were "
            "inspected and are therefore explicitly exploratory. No automated test-set "
            "hyperparameter optimization is implemented.",
        ]
    )
    report = "\n".join(lines) + "\n"
    (config.results_dir / "results_summary.md").write_text(report, encoding="utf-8")
    return report


def create_reviewer_risk_audit(config: AnalysisConfig) -> pd.DataFrame:
    rows = [
        ("Construct validity", "EIA rows are generating units, not contracts or full projects.", "Primary unit-level estimates are cluster weighted; uncertainty is bootstrapped by plant-development cluster; language is restricted to reported generating-unit delivery."),
        ("Left truncation", "Projects already in the September 2015 pipeline have unknown prior histories.", "A 12-month burn-in is prespecified; archive-start projects are excluded from the primary cohort."),
        ("Right censoring", "Recent or long-duration projects may remain planned at May 2026.", "Horizon outcomes require sufficient observable follow-up; unresolved observations are censored."),
        ("Competing risks", "Cancellation prevents later operation on the observed identity.", "Operation and cancellation are modeled separately and as terminal multistate transitions."),
        ("Data leakage", "Future schedule revisions or stages could inflate performance.", "Features are cumulative only through each origin month; temporal entry cohorts are project-disjoint."),
        ("Class imbalance", "Cancellation is uncommon.", "Average precision, Brier score, calibration, per-class metrics, and decision curves supplement ROC-AUC and accuracy."),
        ("Identity instability", "Plant/generator identifiers or reporting organizations can change.", "Identity-change counts are audited; unreconciled units are retained only as censored observations."),
        ("Schema drift", "Headers and available geography fields changed over 2015-2026.", "Direct header-based parsing and stable primary predictors are used; missingness is explicitly audited."),
        ("Overfitting", "Flexible AI may fit administrative idiosyncrasies.", "Simple Markov and logistic baselines, validation tuning, calibration, and subgroup checks are required. Prespecified benchmarks use temporal testing; post-inspection custom models are labeled exploratory and require new-cohort confirmation."),
        ("Causal overclaiming", "Predictive associations do not identify causal delivery mechanisms.", "All effect tables are labeled adjusted associations; causal verbs are prohibited."),
    ]
    table = pd.DataFrame(rows, columns=["risk", "reviewer_concern", "implemented_response"])
    table.to_csv(config.results_dir / "reviewer_risk_audit.csv", index=False, encoding="utf-8-sig")
    return table
