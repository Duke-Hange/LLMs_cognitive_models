"""
神经网络训练与评估
对每种划分与两类模型（Value-Based、Context-Dependent）训练、评估并保存结果。
"""

import sys
import csv
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

# 项目根路径并设置中文字体（避免中文/负号显示为方框）
_p = Path(__file__).resolve().parent
for _ in range(8):
    _p = _p.parent
    if (_p / "shared").is_dir():
        sys.path.insert(0, str(_p))
        break
from shared.visualization import setup_chinese_font
setup_chinese_font()

from config import (
    SEED,
    SEEDS,
    MAX_OUTCOMES,
    SPLIT_TYPES,
    EPOCHS,
    BATCH_SIZE,
    LR,
    EARLY_STOPPING_PATIENCE,
    VAL_FRACTION,
    RESULTS_DIR,
    CURVES_DIR,
    DATA_QUANTITY_N_FRACTIONS,
)
from data_loader import (
    load_standardized_data,
    build_distribution_encodings,
    get_target_vector,
    get_split_data,
)
from models import ValueBasedNet, ContextDependentNet, get_encoding_dims


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_training_curve(history: List[Dict[str, Any]], save_path: Path) -> None:
    """绘制单次训练曲线（epoch vs train_loss / val_mse）并保存。"""
    import matplotlib.pyplot as plt
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_mse = [h["val_mse"] for h in history]
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(epochs, train_loss, label="Train Loss", alpha=0.8)
    ax.plot(epochs, val_mse, label="Val MSE", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / MSE")
    ax.legend()
    ax.set_title("Training curve")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def generate_mean_curves(timestamp: str) -> None:
    """遍历 CURVES_DIR 下本 run 的 JSON（run_{timestamp}_*.json），按 (split_type, model_type) 分组，
    按 epoch 对齐后计算 mean/std，绘制均值±标准差曲线并保存。"""
    import matplotlib.pyplot as plt
    pattern = f"run_{timestamp}_*.json"
    jsons = list(CURVES_DIR.glob(pattern))
    if not jsons:
        return
    by_key: Dict[tuple, List[Dict]] = {}
    for p in jsons:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        key = (data["split_type"], data["model_type"])
        by_key.setdefault(key, []).append(data)
    for (split_type, model_type), group in by_key.items():
        max_epoch = max(len(d["history"]) for d in group if d.get("history"))
        if max_epoch == 0:
            continue
        train_loss_arr = np.full((len(group), max_epoch), np.nan)
        val_mse_arr = np.full((len(group), max_epoch), np.nan)
        for i, d in enumerate(group):
            hist = d.get("history") or []
            for j, h in enumerate(hist):
                if j >= max_epoch:
                    break
                train_loss_arr[i, j] = h["train_loss"]
                val_mse_arr[i, j] = h["val_mse"]
        epochs = list(range(1, max_epoch + 1))
        train_mean = np.nanmean(train_loss_arr, axis=0)
        train_std = np.nanstd(train_loss_arr, axis=0)
        val_mean = np.nanmean(val_mse_arr, axis=0)
        val_std = np.nanstd(val_mse_arr, axis=0)
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(epochs, train_mean, label="Train Loss (mean)", color="C0")
        ax.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.3, color="C0")
        ax.plot(epochs, val_mean, label="Val MSE (mean)", color="C1")
        ax.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.3, color="C1")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss / MSE")
        ax.legend()
        ax.set_title(f"Mean curve ± std — {split_type} / {model_type} (n={len(group)})")
        fig.tight_layout()
        out_path = CURVES_DIR / f"mean_curve_{timestamp}_{split_type}_{model_type}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"均值曲线已保存: {out_path}")


DATA_QUANTITY_SMOOTH_WINDOW = 5
DATA_QUANTITY_DISPLAY_NAMES = {"value_based": "ValueBased", "context_dependent": "Context-Dependent"}


def run_data_quantity_curve(
    standardized_data: List[Dict],
    enc_A: np.ndarray,
    enc_B: np.ndarray,
    enc_full: np.ndarray,
    y: np.ndarray,
    seeds: List[int],
    device: torch.device,
    timestamp: str,
    n_fractions: int = DATA_QUANTITY_N_FRACTIONS,
) -> None:
    """在已加载数据上跑数据量曲线（比例 vs Test MSE），保存 CSV/JSON 并绘图。"""
    fractions = [i / n_fractions for i in range(1, n_fractions + 1)]
    model_types_lc = ["value_based", "context_dependent"]
    rows: List[Dict[str, Any]] = []

    print(f"\n数据量曲线: {n_fractions} 个比例, seeds={seeds}")
    for seed in seeds:
        set_seed(seed)
        for split_type in SPLIT_TYPES:
            split_data = get_split_data(
                standardized_data, enc_A, enc_B, enc_full, y, split_type, random_state=seed
            )
            for fraction in fractions:
                for model_type in model_types_lc:
                    print(f"  [seed={seed}] {split_type} frac={fraction:.2f} {model_type}...")
                    test_mse = run_one_fraction_split(
                        split_data, fraction, model_type, device, seed
                    )
                    rows.append({
                        "split_type": split_type,
                        "model_type": model_type,
                        "fraction": fraction,
                        "seed": seed,
                        "test_mse": test_mse,
                    })

    csv_path = CURVES_DIR / f"learning_curve_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split_type", "model_type", "fraction", "seed", "test_mse"])
        w.writeheader()
        w.writerows(rows)
    print(f"学习曲线数据已保存: {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过数据量曲线图。")
        json_path = CURVES_DIR / f"learning_curve_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"timestamp": timestamp, "rows": rows, "seeds": seeds}, f, indent=2, ensure_ascii=False)
        print(f"学习曲线 JSON 已保存: {json_path}")
        return

    for split_type in SPLIT_TYPES:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        for model_type in model_types_lc:
            sub = [r for r in rows if r["split_type"] == split_type and r["model_type"] == model_type]
            if not sub:
                continue
            by_frac: Dict[float, List[float]] = {}
            for r in sub:
                by_frac.setdefault(r["fraction"], []).append(r["test_mse"])
            fracs_sorted = sorted(by_frac.keys())
            mean_arr = np.array([np.mean(by_frac[f]) for f in fracs_sorted])
            std_arr = np.array([np.std(by_frac[f]) for f in fracs_sorted])
            seeds_here = sorted({r["seed"] for r in sub})
            for s in seeds_here:
                points = [(r["fraction"], r["test_mse"]) for r in sub if r["seed"] == s]
                points.sort(key=lambda x: x[0])
                xs = [p[0] * 100 for p in points]
                ys = [p[1] for p in points]
                ax.plot(xs, ys, alpha=0.4, linewidth=1)
            if len(mean_arr) >= DATA_QUANTITY_SMOOTH_WINDOW:
                mean_smooth = np.convolve(
                    mean_arr, np.ones(DATA_QUANTITY_SMOOTH_WINDOW) / DATA_QUANTITY_SMOOTH_WINDOW, mode="valid"
                )
                std_smooth = np.convolve(
                    std_arr, np.ones(DATA_QUANTITY_SMOOTH_WINDOW) / DATA_QUANTITY_SMOOTH_WINDOW, mode="valid"
                )
                n_smooth = len(mean_smooth)
                x_smooth = np.linspace(0, 100, n_smooth)
                ax.plot(x_smooth, mean_smooth, linewidth=3, alpha=0.9, label=DATA_QUANTITY_DISPLAY_NAMES[model_type])
                ax.fill_between(x_smooth, mean_smooth - std_smooth, mean_smooth + std_smooth, alpha=0.25)
            else:
                ax.plot(
                    [f * 100 for f in fracs_sorted], mean_arr, "o-", linewidth=2,
                    label=DATA_QUANTITY_DISPLAY_NAMES[model_type],
                )
                ax.fill_between(
                    [f * 100 for f in fracs_sorted],
                    mean_arr - std_arr, mean_arr + std_arr, alpha=0.25,
                )
        ax.set_xlabel("Percent training data used (%)", fontsize=12)
        ax.set_ylabel("Test set MSE", fontsize=12)
        ax.set_title(
            f"Data quantity vs test MSE — {split_type}\n"
            f"(Mean ± std across seeds; mean smoothed with window={DATA_QUANTITY_SMOOTH_WINDOW})"
        )
        ax.legend()
        ax.set_xlim(0, 100)
        fig.tight_layout()
        out_path = CURVES_DIR / f"data_quantity_curve_{timestamp}_{split_type}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"数据量曲线已保存: {out_path}")

    json_path = CURVES_DIR / f"learning_curve_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": timestamp, "rows": rows, "seeds": seeds}, f, indent=2, ensure_ascii=False)
    print(f"学习曲线 JSON 已保存: {json_path}")


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """与 01 的 evaluate_model 一致。"""
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
    if len(y_true) > 1:
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        corr = 0.0 if np.isnan(corr) else float(corr)
    else:
        corr = 0.0
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "correlation": corr,
        "n_samples": len(y_true),
    }


def train_value_based(
    enc_A_train: np.ndarray,
    enc_B_train: np.ndarray,
    y_train: np.ndarray,
    enc_A_val: np.ndarray,
    enc_B_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    patience: int = EARLY_STOPPING_PATIENCE,
) -> tuple:
    """训练 ValueBasedNet，返回 (model, scaler_A, scaler_B, best_epoch, history)."""
    from sklearn.preprocessing import StandardScaler

    scaler_A = StandardScaler()
    scaler_B = StandardScaler()
    enc_A_tr = scaler_A.fit_transform(enc_A_train)
    enc_B_tr = scaler_B.fit_transform(enc_B_train)
    enc_A_va = scaler_A.transform(enc_A_val)
    enc_B_va = scaler_B.transform(enc_B_val)

    dim_per_gamble, _ = get_encoding_dims(MAX_OUTCOMES)
    model = ValueBasedNet(input_dim_per_gamble=dim_per_gamble).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    best_val_mse = float("inf")
    best_epoch = 0
    best_state = None
    wait = 0
    history: List[Dict[str, Any]] = []

    n = len(y_train)
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            a = torch.FloatTensor(enc_A_tr[idx]).to(device)
            b = torch.FloatTensor(enc_B_tr[idx]).to(device)
            target = torch.FloatTensor(y_train[idx]).to(device)
            optimizer.zero_grad()
            out = model(a, b)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            a_va = torch.FloatTensor(enc_A_va).to(device)
            b_va = torch.FloatTensor(enc_B_va).to(device)
            pred_val = model(a_va, b_va).cpu().numpy()
        val_mse = float(np.mean((y_val - pred_val) ** 2))
        history.append({"epoch": epoch + 1, "train_loss": float(epoch_loss), "val_mse": val_mse})
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler_A, scaler_B, best_epoch, history


def train_context_dependent(
    enc_full_train: np.ndarray,
    y_train: np.ndarray,
    enc_full_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    patience: int = EARLY_STOPPING_PATIENCE,
) -> tuple:
    """训练 ContextDependentNet，返回 (model, scaler, best_epoch, history)."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    enc_tr = scaler.fit_transform(enc_full_train)
    enc_va = scaler.transform(enc_full_val)

    _, full_dim = get_encoding_dims(MAX_OUTCOMES)
    model = ContextDependentNet(input_dim=full_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    best_val_mse = float("inf")
    best_epoch = 0
    best_state = None
    wait = 0
    history: List[Dict[str, Any]] = []
    n = len(y_train)
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            x = torch.FloatTensor(enc_tr[idx]).to(device)
            target = torch.FloatTensor(y_train[idx]).to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            x_va = torch.FloatTensor(enc_va).to(device)
            pred_val = model(x_va).cpu().numpy()
        val_mse = float(np.mean((y_val - pred_val) ** 2))
        history.append({"epoch": epoch + 1, "train_loss": float(epoch_loss), "val_mse": val_mse})
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler, best_epoch, history


def run_one_fraction_split(
    split_data: Dict[str, Any],
    fraction: float,
    model_type: str,
    device: torch.device,
    seed: int,
) -> float:
    """在训练集的前 fraction 比例上训练，固定测试集评估，返回 test MSE。"""
    set_seed(seed)
    n_train = len(split_data["y_train"])
    n_use = max(1, int(n_train * fraction))
    idx_use = np.arange(n_use)
    n_val = max(1, int(n_use * VAL_FRACTION))
    rng = np.random.default_rng(seed)
    val_idx_local = rng.choice(n_use, size=n_val, replace=False)
    train_idx_local = np.array([i for i in range(n_use) if i not in val_idx_local])

    if model_type == "value_based":
        enc_A_tr = split_data["enc_A_train"][idx_use][train_idx_local]
        enc_B_tr = split_data["enc_B_train"][idx_use][train_idx_local]
        y_tr = split_data["y_train"][idx_use][train_idx_local]
        enc_A_va = split_data["enc_A_train"][idx_use][val_idx_local]
        enc_B_va = split_data["enc_B_train"][idx_use][val_idx_local]
        y_va = split_data["y_train"][idx_use][val_idx_local]
        model, scaler_A, scaler_B, _, _ = train_value_based(
            enc_A_tr, enc_B_tr, y_tr, enc_A_va, enc_B_va, y_va, device
        )
        enc_A_te = scaler_A.transform(split_data["enc_A_test"])
        enc_B_te = scaler_B.transform(split_data["enc_B_test"])
        model.eval()
        with torch.no_grad():
            a = torch.FloatTensor(enc_A_te).to(device)
            b = torch.FloatTensor(enc_B_te).to(device)
            test_pred = model(a, b).cpu().numpy()
    else:
        enc_tr = split_data["enc_full_train"][idx_use][train_idx_local]
        y_tr = split_data["y_train"][idx_use][train_idx_local]
        enc_va = split_data["enc_full_train"][idx_use][val_idx_local]
        y_va = split_data["y_train"][idx_use][val_idx_local]
        model, scaler, _, _ = train_context_dependent(
            enc_tr, y_tr, enc_va, y_va, device
        )
        enc_te = scaler.transform(split_data["enc_full_test"])
        model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(enc_te).to(device)
            test_pred = model(x).cpu().numpy()
    metrics = evaluate_model(split_data["y_test"], test_pred)
    return float(metrics["mse"])


def run_one_split(
    split_data: Dict[str, Any],
    model_type: str,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    """在一种划分上训练指定类型模型，返回指标与预测。seed 用于 train/val 划分与训练可复现。"""
    set_seed(seed)
    n_train = len(split_data["y_train"])
    n_val = max(1, int(n_train * VAL_FRACTION))
    rng = np.random.default_rng(seed)
    val_idx = rng.choice(n_train, size=n_val, replace=False)
    train_idx = np.array([i for i in range(n_train) if i not in val_idx])

    if model_type == "value_based":
        enc_A_tr = split_data["enc_A_train"][train_idx]
        enc_B_tr = split_data["enc_B_train"][train_idx]
        y_tr = split_data["y_train"][train_idx]
        enc_A_va = split_data["enc_A_train"][val_idx]
        enc_B_va = split_data["enc_B_train"][val_idx]
        y_va = split_data["y_train"][val_idx]
        model, scaler_A, scaler_B, best_epoch, history = train_value_based(
            enc_A_tr, enc_B_tr, y_tr, enc_A_va, enc_B_va, y_va, device
        )
        enc_A_te = scaler_A.transform(split_data["enc_A_test"])
        enc_B_te = scaler_B.transform(split_data["enc_B_test"])
        model.eval()
        with torch.no_grad():
            a = torch.FloatTensor(enc_A_te).to(device)
            b = torch.FloatTensor(enc_B_te).to(device)
            test_pred = model(a, b).cpu().numpy()
        enc_A_tr_full = scaler_A.transform(split_data["enc_A_train"])
        enc_B_tr_full = scaler_B.transform(split_data["enc_B_train"])
        with torch.no_grad():
            a = torch.FloatTensor(enc_A_tr_full).to(device)
            b = torch.FloatTensor(enc_B_tr_full).to(device)
            train_pred = model(a, b).cpu().numpy()
        scalers = {"scaler_A": scaler_A, "scaler_B": scaler_B}
    else:
        enc_tr = split_data["enc_full_train"][train_idx]
        y_tr = split_data["y_train"][train_idx]
        enc_va = split_data["enc_full_train"][val_idx]
        y_va = split_data["y_train"][val_idx]
        model, scaler, best_epoch, history = train_context_dependent(
            enc_tr, y_tr, enc_va, y_va, device
        )
        enc_te = scaler.transform(split_data["enc_full_test"])
        model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(enc_te).to(device)
            test_pred = model(x).cpu().numpy()
        enc_tr_full = scaler.transform(split_data["enc_full_train"])
        with torch.no_grad():
            x = torch.FloatTensor(enc_tr_full).to(device)
            train_pred = model(x).cpu().numpy()
        scalers = {"scaler": scaler}

    train_metrics = evaluate_model(split_data["y_train"], train_pred)
    test_metrics = evaluate_model(split_data["y_test"], test_pred)
    return {
        "model_type": model_type,
        "split_info": split_data["split_info"],
        "best_epoch": best_epoch,
        "history": history,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_predictions": train_pred.tolist(),
        "test_predictions": test_pred.tolist(),
        "train_indices": split_data["train_idx"].tolist(),
        "test_indices": split_data["test_idx"].tolist(),
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="神经网络训练与评估（支持多种子）")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS,
        help="随机种子列表，如 --seeds 42 43 44 45 46（默认: 42）",
    )
    parser.add_argument(
        "--curves",
        type=str,
        choices=["all", "mean_only", "none"],
        default="none",
        help="训练曲线图：all=每种子+均值图，mean_only=仅均值图，none=不生成图（仍保存 JSON）",
    )
    parser.add_argument(
        "--data-quantity-curve",
        action="store_true",
        help="训练结束后再跑数据量曲线（训练数据比例 vs Test MSE）并画图",
    )
    parser.add_argument(
        "--n-fractions",
        type=int,
        default=DATA_QUANTITY_N_FRACTIONS,
        help=f"数据量曲线的比例点个数（默认 {DATA_QUANTITY_N_FRACTIONS}）",
    )
    args = parser.parse_args()
    seeds: List[int] = args.seeds

    set_seed(seeds[0])  # 初始环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seeds: {seeds}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("加载数据...")
    standardized_data = load_standardized_data()
    enc_A, enc_B, enc_full = build_distribution_encodings(standardized_data)
    y = get_target_vector(standardized_data)
    print(f"样本数: {len(y)}, enc_A: {enc_A.shape}, enc_full: {enc_full.shape}")

    model_types = ["value_based", "context_dependent"]
    all_results_per_seed: Dict[int, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    for seed in seeds:
        set_seed(seed)
        all_results = {}
        for split_type in SPLIT_TYPES:
            print(f"\n[seed={seed}] 划分: {split_type}")
            split_data = get_split_data(
                standardized_data, enc_A, enc_B, enc_full, y, split_type, random_state=seed
            )
            all_results[split_type] = {}
            for model_type in model_types:
                print(f"  训练 {model_type}...")
                res = run_one_split(split_data, model_type, device, seed=seed)
                all_results[split_type][model_type] = res
                print(
                    f"    test MSE: {res['test_metrics']['mse']:.6f}, R2: {res['test_metrics']['r2']:.4f}"
                )
                curve_json = CURVES_DIR / f"run_{timestamp}_seed{seed}_{split_type}_{model_type}.json"
                curve_png = CURVES_DIR / f"run_{timestamp}_seed{seed}_{split_type}_{model_type}.png"
                with open(curve_json, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "seed": seed,
                            "split_type": split_type,
                            "model_type": model_type,
                            "timestamp": timestamp,
                            "best_epoch": res["best_epoch"],
                            "history": res["history"],
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
                if args.curves == "all":
                    plot_training_curve(res["history"], curve_png)
        all_results_per_seed[seed] = all_results

    def to_serializable(obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(x) for x in obj]
        return obj

    def strip_history(obj: Any) -> Any:
        """深拷贝并移除 history 以减小主结果 JSON 体积；曲线已单独存于 CURVES_DIR。"""
        if isinstance(obj, dict):
            return {k: strip_history(v) for k, v in obj.items() if k != "history"}
        if isinstance(obj, list):
            return [strip_history(x) for x in obj]
        return obj

    # 完整结果 JSON：多种子时每 seed 一文件，单 seed 时单文件（不含 history，曲线在 CURVES_DIR）
    for seed in seeds:
        results_file = RESULTS_DIR / f"neural_models_results_{timestamp}_seed{seed}.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(to_serializable(strip_history(all_results_per_seed[seed])), f, indent=2, ensure_ascii=False)
        print(f"完整结果已保存: {results_file}")

    n_seeds = len(seeds)
    if n_seeds == 1:
        # 单 seed：保持原有列名，与 04 向后兼容
        all_results = all_results_per_seed[seeds[0]]
        summary_rows = []
        for split_type in SPLIT_TYPES:
            for model_type in model_types:
                r = all_results[split_type][model_type]
                summary_rows.append({
                    "model_type": model_type,
                    "split_type": split_type,
                    "train_mse": r["train_metrics"]["mse"],
                    "test_mse": r["test_metrics"]["mse"],
                    "train_r2": r["train_metrics"]["r2"],
                    "test_r2": r["test_metrics"]["r2"],
                    "train_correlation": r["train_metrics"]["correlation"],
                    "test_correlation": r["test_metrics"]["correlation"],
                    "best_epoch": r["best_epoch"],
                })
        summary_path = RESULTS_DIR / f"neural_models_summary_{timestamp}.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
    else:
        # 多种子：聚合 mean/std，输出 *_mean, *_std, n_seeds
        summary_rows = []
        for split_type in SPLIT_TYPES:
            for model_type in model_types:
                mse_list = [all_results_per_seed[s][split_type][model_type]["test_metrics"]["mse"] for s in seeds]
                r2_list = [all_results_per_seed[s][split_type][model_type]["test_metrics"]["r2"] for s in seeds]
                corr_list = [all_results_per_seed[s][split_type][model_type]["test_metrics"]["correlation"] for s in seeds]
                summary_rows.append({
                    "model_type": model_type,
                    "split_type": split_type,
                    "test_mse_mean": float(np.mean(mse_list)),
                    "test_mse_std": float(np.std(mse_list)),
                    "test_r2_mean": float(np.mean(r2_list)),
                    "test_r2_std": float(np.std(r2_list)),
                    "test_correlation_mean": float(np.mean(corr_list)),
                    "test_correlation_std": float(np.std(corr_list)),
                    "n_seeds": n_seeds,
                })
        summary_path = RESULTS_DIR / f"neural_models_summary_{timestamp}.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
    print(f"汇总表已保存: {summary_path}")

    # 可选：保存 Excel 汇总表
    try:
        import pandas as pd
        df_summary = pd.DataFrame(summary_rows)
        excel_path = RESULTS_DIR / f"neural_models_summary_{timestamp}.xlsx"
        df_summary.to_excel(excel_path, index=False)
        print(f"Excel 汇总表已保存: {excel_path}")
    except Exception as e:
        print(f"未保存 Excel（可选）: {e}")

    # 按 (split_type, model_type) 聚合本 run 的曲线 JSON，生成均值±标准差曲线图（--curves none 时跳过）
    if args.curves != "none":
        print("\n生成均值曲线...")
        generate_mean_curves(timestamp)

    # 可选：数据量曲线（训练数据比例 vs Test MSE）
    if args.data_quantity_curve:
        run_data_quantity_curve(
            standardized_data, enc_A, enc_B, enc_full, y,
            seeds, device, timestamp, n_fractions=args.n_fractions,
        )

    # 控制台汇总（多种子时用均值展示）
    all_results = all_results_per_seed[seeds[0]] if n_seeds == 1 else None
    if all_results is not None:
        print("\n" + "=" * 80)
        print("神经网络实验结果汇总")
        print("=" * 80)
        for split_type in SPLIT_TYPES:
            if split_type in all_results:
                print(f"\n划分: {split_type}")
                print("-" * 40)
                for model_type in model_types:
                    r = all_results[split_type][model_type]
                    tm = r["test_metrics"]
                    print(
                        f"  {model_type:20s}: "
                        f"测试 MSE={tm['mse']:.6f}, "
                        f"R^2={tm['r2']:.4f}, "
                        f"相关性={tm['correlation']:.4f}"
                    )
    else:
        print("\n" + "=" * 80)
        print("神经网络实验结果汇总（多种子均值 ± 标准差）")
        print("=" * 80)
        for split_type in SPLIT_TYPES:
            print(f"\n划分: {split_type}")
            print("-" * 40)
            for row in summary_rows:
                if row["split_type"] != split_type:
                    continue
                print(
                    f"  {row['model_type']:20s}: "
                    f"测试 MSE={row['test_mse_mean']:.6f} ± {row['test_mse_std']:.6f}, "
                    f"R^2={row['test_r2_mean']:.4f} ± {row['test_r2_std']:.4f}"
                )


if __name__ == "__main__":
    main()
