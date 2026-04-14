from __future__ import annotations

import hashlib
import shutil
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

THIS_FILE = Path(__file__).resolve()
PROJ_ROOT = THIS_FILE.parents[4]
EXPERIMENTS_ROOT = THIS_FILE.parents[3]
PROGRAM_ROOT = THIS_FILE.parents[2]
for p in (PROJ_ROOT, EXPERIMENTS_ROOT, PROGRAM_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.analysis.aggregate import aggregate_curve, build_summary_table, evaluate_predictions_long
from src.analysis.plot_learning_curves import plot_metric_curve
from src.analysis.repro_checks import run_repro_checks
from src.common.config import load_all_configs, project_root, write_json
from src.common.metrics import EvalPolicy
from src.data.dataset import load_prompts_dataset
from src.data.splits import build_master_split, build_sample_map, sample_map_as_json_ready
from src.runners.run_llm_icl import run_llm_family
from src.runners.run_neural import run_neural_family
from src.runners.run_symbolic import run_symbolic_family


def _resolve_path(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs).expanduser()
    if p.is_absolute():
        return p
    return project_root() / p


def _staging_pair(protocol: Dict, run_id: str) -> Tuple[Path, Path, bool]:
    """
    Returns (active_out_root, archive_out_root, use_staging).
    When use_staging: all run artifacts go to active (e.g. SSD); on success caller copies to archive then removes active.
    """
    paths_cfg = protocol.get("paths", {})
    archive_base = _resolve_path(str(paths_cfg["output_root"]))
    archive_out = (archive_base / f"run_{run_id}").resolve()
    raw = paths_cfg.get("staging_output_root")
    if raw is None:
        return archive_out, archive_out, False
    s = str(raw).strip()
    if s == "" or s.lower() == "null":
        return archive_out, archive_out, False
    staging_base = _resolve_path(s).resolve()
    active_out = (staging_base / f"run_{run_id}").resolve()
    if active_out == archive_out:
        raise ValueError("paths.staging_output_root resolves to the same directory as output_root; pick a different SSD path.")
    return active_out, archive_out, True


def _write_methods_report(out_path: Path, protocol: Dict) -> None:
    prompts_path = protocol.get("data", {}).get("prompts_path", "")
    lines = [
        "# Methods and Reproducibility",
        "",
        "## Core protocol",
        f"- Target mode: `{protocol['experiment']['target_mode']}`",
        f"- Sample sizes: `{protocol['sampling']['sample_sizes']}`",
        f"- Seeds: `{protocol['sampling']['seeds']}`",
        f"- Test size: `{protocol['data']['test_size']}`",
        f"- Prompts JSONL (relative to repo root): `{prompts_path}`",
        "",
        "## Data freeze (split_manifest.json)",
        "- Each `05_program` run writes `split_manifest.json` with: `train_indices`, `test_indices`, `sample_map` (per N, seed), `prompts_sha256`, `target_mode`.",
        "- Text QLoRA (`experiments/06_qlora_text/train_qlora.py`) must consume that manifest so train/test pools match ICL/neural runs.",
        "",
        "## LLM parse failure policy",
        f"- `llm_parse_fail_fill`: `{protocol['evaluation'].get('llm_parse_fail_fill')}` "
        "(`null` = main metrics on **parse_ok ∧ finite pred** only; numeric = impute failures for main metrics).",
        "- When imputation is enabled, sensitivity metrics on parse_ok-only subset are still reported in `sens_*` columns.",
        "",
        "## LLM prompting",
        f"- ICL deduplicated template (Option A/B blocks only per example): `{protocol.get('llm', {}).get('dedup_icl', True)}`",
        "",
        "## LLM task wording vs evaluation (construct validity)",
        "- Some JSONL versions (e.g. `en2_preference`) phrase the decision as a discrete **Option A vs Option B** answer in the body text.",
        "- The evaluation pipeline still asks the model (via retry text in `parse_output.py`) for a **single probability** `P_B` in [0,1] (or one numeric token), aligned with the **bRate** label.",
        "- Interpret results as **probability prediction under this elicitation protocol**, not a verbatim reproduction of the original two-alternative forced-choice wording. Prefer prompt versions whose instructions match `P_B` (e.g. `instruction_en_v2` / v3) for tighter construct alignment.",
        "",
        "## Metrics: cross-entropy with continuous labels",
        "- `cross_entropy` uses `-mean(y*log(p) + (1-y)*log(1-p))` with `y` in [0,1] (`experiments/model_base.py`). For non-binary `y`, this is the **average pointwise Bernoulli negative log-likelihood** treating each `y` as a Bernoulli parameter, not multinomial classification CE.",
        "",
        "## LLM parse fallback and generation cap",
        f"- `llm_parse_fail_fill` is `{protocol['evaluation'].get('llm_parse_fail_fill')}`; see policy section above.",
        f"- `llm.num_predict` caps completion length; document it if outputs are truncated.",
        "",
        "## Learning-curve axis (ICL)",
        "- For the LLM family, **N equals the number of in-context exemplars** drawn from the train pool for that `(N, seed)` (same index set as neural/symbolic train subset size), not a separate notion of shots unless stated.",
        "",
        "## Text QLoRA (reviewer: few-shot fine-tuning)",
        "- Implementation: `experiments/06_qlora_text/` — 4-bit LoRA on a causal LM + scalar head; MSE on `y_target` from `load_prompts_dataset` (same as 05_program).",
        "- **Train/val mask**: within each `sample_map` train index list, the holdout fraction for validation uses the **same random seed `seed` as `run_neural.py`’s `_split_train_val`**, and `val_fraction` in `06_qlora_text/config.yaml` should match neural `VAL_FRACTION` (0.2) for comparable early-stopping behavior.",
        "- Training budget: `max_epochs` in `06_qlora_text/config.yaml` (default 6), early stopping on validation MSE, `n_checkpoints` save intervals derived from estimated total steps.",
        "- **Input vs ICL**: QLoRA appends a short **regression task suffix** to the raw `prompt` (see `regression_suffix` in config); ICL uses few-shot `P_B=` exemplars. The two routes are not identical input distributions; compare them as distinct protocols.",
        "- Checkpoints: `checkpoint_index.json` + `val_curve.csv` under each QLoRA output dir; test metrics use `load_best_model_at_end` weights only once on the frozen test index list.",
        "- Merge into unified tables: `python experiments/06_qlora_text/merge_qlora_predictions.py --base .../predictions_long.csv --qlora .../predictions_long.csv --out ...`",
        "",
        "## Reviewer comment mapping (training / fine-tuning)",
        "| Item | What we did |",
        "|------|-------------|",
        "| Few-shot QLoRA | `06_qlora_text` + manifest-locked indices |",
        "| 10 checkpoints + epoch budget | Uniform `save_steps` over estimated max steps; cap `max_epochs` (default 6) + early stopping |",
        "| 1 epoch + 0.2…1.0 continuation | Optional follow-up (not default); avoid mixing with primary checkpoint narrative |",
        "| Pre-built dataset | Prompts JSONL + `prompts_sha256` in manifest; no runtime rebuild of splits |",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 05_program unified experiment")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny smoke configuration")
    parser.add_argument("--neural-epochs", type=int, default=1000)
    parser.add_argument("--neural-patience", type=int, default=100)
    parser.add_argument("--quiet", action="store_true", help="Disable progress logs")
    args = parser.parse_args()
    verbose = not args.quiet

    cfg_dir = PROGRAM_ROOT / "configs"
    cfg = load_all_configs(cfg_dir)
    protocol = cfg["protocol"]
    if args.smoke:
        protocol["sampling"]["sample_sizes"] = [8, 32]
        protocol["sampling"]["seeds"] = [42, 43]
        protocol["llm"]["dry_run"] = True
    if verbose:
        print(
            f"[05_program] start | target={protocol['experiment']['target_mode']} "
            f"N={protocol['sampling']['sample_sizes']} seeds={protocol['sampling']['seeds']} "
            f"llm_dry_run={protocol['llm']['dry_run']}",
            flush=True,
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root, archive_out_root, use_staging = _staging_pair(protocol, run_id)
    if verbose:
        if use_staging:
            print(
                f"[05_program] staging on fast disk -> {out_root}\n"
                f"[05_program] will archive after success -> {archive_out_root}",
                flush=True,
            )
    pred_dir = out_root / "predictions"
    metric_dir = out_root / "metrics"
    fig_dir = out_root / "figures"
    table_dir = out_root / "tables"
    report_dir = out_root / "reports"
    cfg_snap_dir = out_root / "configs_snapshot"
    for d in (pred_dir, metric_dir, fig_dir, table_dir, report_dir, cfg_snap_dir):
        d.mkdir(parents=True, exist_ok=True)

    prompts_path = _resolve_path(protocol["data"]["prompts_path"])
    target_mode = str(protocol["experiment"]["target_mode"])
    data = load_prompts_dataset(path=prompts_path, target_mode=target_mode, max_outcomes=9)

    split = build_master_split(
        n_rows=len(data.df),
        master_seed=int(protocol["data"]["master_seed"]),
        test_size=float(protocol["data"]["test_size"]),
        split_random_state_base=int(protocol["data"]["split_random_state_base"]),
    )
    sample_map = build_sample_map(
        train_idx=split.train_idx,
        sample_sizes=protocol["sampling"]["sample_sizes"],
        seeds=protocol["sampling"]["seeds"],
    )
    if verbose:
        print(
            f"[05_program] data ready | n_total={len(data.df)} n_train={len(split.train_idx)} "
            f"n_test={len(split.test_idx)} combos={len(sample_map)}",
            flush=True,
        )

    pred_rows = []
    if verbose:
        print("[05_program] symbolic phase...", flush=True)
    pred_rows.extend(
        run_symbolic_family(
            model_defs=cfg["symbolic"]["models"],
            standardized_records=data.standardized_records,
            y_target=data.y_target,
            y_brate=data.y_brate,
            test_idx=split.test_idx,
            sample_map=sample_map,
            verbose=verbose,
        )
    )
    if verbose:
        print("[05_program] neural phase...", flush=True)
    pred_rows.extend(
        run_neural_family(
            model_defs=cfg["neural"]["models"],
            enc_a=data.enc_A,
            enc_b=data.enc_B,
            enc_full=data.enc_full,
            y_target=data.y_target,
            test_idx=split.test_idx,
            sample_map=sample_map,
            epochs=int(args.neural_epochs),
            patience=int(args.neural_patience),
            verbose=verbose,
        )
    )
    if verbose:
        print("[05_program] llm phase...", flush=True)
    pred_rows.extend(
        run_llm_family(
            model_defs=cfg["llm"]["models"],
            df=data.df,
            y_target=data.y_target,
            train_idx=split.train_idx,
            test_idx=split.test_idx,
            sample_map=sample_map,
            llm_cfg=protocol["llm"],
            verbose=verbose,
        )
    )

    pred_df = pd.DataFrame(pred_rows).sort_values(
        ["family", "model_id", "N", "seed", "row_index"]
    ).reset_index(drop=True)
    pred_path = pred_dir / "predictions_long.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8")
    if verbose:
        print(f"[05_program] predictions saved: {pred_path}", flush=True)

    _fill = protocol["evaluation"].get("llm_parse_fail_fill")
    policy = EvalPolicy(
        ce_eps=float(protocol["evaluation"]["ce_eps"]),
        llm_parse_fail_fill=None if _fill is None else float(_fill),
    )
    metric_df = evaluate_predictions_long(pred_df, policy=policy)
    metric_path = metric_dir / "metrics_by_model_n_seed.csv"
    metric_df.to_csv(metric_path, index=False, encoding="utf-8")
    if verbose:
        print(f"[05_program] metrics saved: {metric_path}", flush=True)

    curve_ce = aggregate_curve(
        metric_df=metric_df,
        metric="cross_entropy",
        ci_level=float(protocol["evaluation"]["ci_level"]),
        bootstrap_samples=int(protocol["evaluation"]["bootstrap_samples"]),
    )
    curve_mae = aggregate_curve(
        metric_df=metric_df,
        metric="mae",
        ci_level=float(protocol["evaluation"]["ci_level"]),
        bootstrap_samples=int(protocol["evaluation"]["bootstrap_samples"]),
    )
    curve_df = pd.concat([curve_ce, curve_mae], ignore_index=True)
    curve_path = metric_dir / "curve_aggregated.csv"
    curve_df.to_csv(curve_path, index=False, encoding="utf-8")

    plot_metric_curve(curve_df, metric="cross_entropy", out_path=fig_dir / "learning_curve_cross_entropy.png")
    plot_metric_curve(curve_df, metric="mae", out_path=fig_dir / "learning_curve_mae.png")
    if verbose:
        print(f"[05_program] figures saved: {fig_dir}", flush=True)

    summary_df = build_summary_table(
        metric_df=metric_df,
        n_target_metric=str(protocol["evaluation"]["n_target_metric"]),
        n_target_value=float(protocol["evaluation"]["n_target_value"]),
    )
    summary_df.to_csv(table_dir / "summary_by_model.csv", index=False, encoding="utf-8")

    repro = run_repro_checks(pred_df=pred_df, metric_df=metric_df, expected_n_test=len(split.test_idx))
    write_json(report_dir / "repro_checks.json", repro)

    split_meta = {
        "master_seed": int(protocol["data"]["master_seed"]),
        "split_random_state": int(split.random_state),
        "train_size": int(len(split.train_idx)),
        "test_size": int(len(split.test_idx)),
        "train_indices": [int(x) for x in split.train_idx.tolist()],
        "test_indices": [int(x) for x in split.test_idx.tolist()],
        "prompts_path": str(prompts_path.as_posix() if hasattr(prompts_path, "as_posix") else prompts_path),
        "prompts_sha256": _sha256_file(prompts_path),
        "target_mode": target_mode,
        "sample_map": sample_map_as_json_ready(sample_map),
    }
    write_json(out_root / "split_manifest.json", split_meta)

    for name in ("protocol.yaml", "models_symbolic.yaml", "models_neural.yaml", "models_llm.yaml"):
        src = cfg_dir / name
        (cfg_snap_dir / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    _write_methods_report(report_dir / "methods_and_reproducibility.md", protocol=protocol)

    if use_staging:
        try:
            archive_out_root.parent.mkdir(parents=True, exist_ok=True)
            if archive_out_root.exists():
                shutil.rmtree(archive_out_root)
            shutil.copytree(out_root, archive_out_root, dirs_exist_ok=False)
            shutil.rmtree(out_root)
            if verbose:
                print(f"[05_program] staging copy complete -> {archive_out_root}", flush=True)
        except Exception as exc:
            print(
                f"[05_program] ERROR: archive copy failed ({exc}). "
                f"Full run output is still under: {out_root}",
                flush=True,
            )
            raise
        print(f"[05_program] done -> {archive_out_root}")
    else:
        print(f"[05_program] done -> {out_root}")


if __name__ == "__main__":
    main()
