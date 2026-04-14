"""
experiments 训练与评估（单一 train_test_split + learning curve）
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from config import (
    SEED,
    MAX_OUTCOMES,
    EPOCHS,
    BATCH_SIZE,
    LR,
    EARLY_STOPPING_PATIENCE,
    VAL_FRACTION,
    DATA_QUANTITY_N_FRACTIONS,
    TEST_SIZE,
    SPLIT_RANDOM_STATE,
    USE_RAW_ENCODING,
)
from models import ValueBasedNet, ContextDependentNet, ContextDependentNetSigmoid, get_encoding_dims


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cross_entropy_error(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7) -> float:
    if len(y_true) == 0:
        return float("nan")
    p = np.clip(y_pred.astype(np.float64), eps, 1.0 - eps)
    ce = -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
    return float(ce)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        nan = float("nan")
        return {"mse": nan, "cross_entropy": nan, "n_samples": 0}
    mse = float(np.mean((y_true - y_pred) ** 2))
    ce = cross_entropy_error(y_true, y_pred)
    return {"mse": mse, "cross_entropy": ce, "n_samples": len(y_true)}


def _split_train_val(
    enc_A_train: np.ndarray,
    enc_B_train: np.ndarray,
    enc_full_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> Dict[str, np.ndarray]:
    n_train = len(y_train)
    n_val = max(1, int(n_train * VAL_FRACTION))
    rng = np.random.default_rng(seed)
    val_idx = rng.choice(n_train, size=n_val, replace=False)
    train_idx = np.array([i for i in range(n_train) if i not in val_idx], dtype=np.int64)
    return {
        "enc_A_tr": enc_A_train[train_idx],
        "enc_B_tr": enc_B_train[train_idx],
        "enc_full_tr": enc_full_train[train_idx],
        "y_tr": y_train[train_idx],
        "enc_A_val": enc_A_train[val_idx],
        "enc_B_val": enc_B_train[val_idx],
        "enc_full_val": enc_full_train[val_idx],
        "y_val": y_train[val_idx],
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
    if USE_RAW_ENCODING:
        scaler_A = None
        scaler_B = None
        enc_A_tr, enc_B_tr = enc_A_train, enc_B_train
        enc_A_va, enc_B_va = enc_A_val, enc_B_val
    else:
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    criterion = torch.nn.MSELoss()

    best_val_mse = float("inf")
    best_state = None
    wait = 0
    n = len(y_train)
    for _ in range(epochs):
        model.train()
        perm = np.random.permutation(n)
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

        model.eval()
        with torch.no_grad():
            a_va = torch.FloatTensor(enc_A_va).to(device)
            b_va = torch.FloatTensor(enc_B_va).to(device)
            pred_val = model(a_va, b_va).cpu().numpy()
        val_mse = float(np.mean((y_val - pred_val) ** 2))
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        scheduler.step(val_mse)
        if wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler_A, scaler_B


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
    use_sigmoid: bool = False,
) -> tuple:
    if USE_RAW_ENCODING:
        scaler = None
        enc_tr = enc_full_train
        enc_va = enc_full_val
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        enc_tr = scaler.fit_transform(enc_full_train)
        enc_va = scaler.transform(enc_full_val)

    _, full_dim = get_encoding_dims(MAX_OUTCOMES)
    model_cls = ContextDependentNetSigmoid if use_sigmoid else ContextDependentNet
    model = model_cls(input_dim=full_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    criterion = torch.nn.MSELoss()

    best_val_mse = float("inf")
    best_state = None
    wait = 0
    n = len(y_train)
    for _ in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            x = torch.FloatTensor(enc_tr[idx]).to(device)
            target = torch.FloatTensor(y_train[idx]).to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            x_va = torch.FloatTensor(enc_va).to(device)
            pred_val = model(x_va).cpu().numpy()
        val_mse = float(np.mean((y_val - pred_val) ** 2))
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        scheduler.step(val_mse)
        if wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler


def _train_test_split_once(
    enc_A: np.ndarray,
    enc_B: np.ndarray,
    enc_full: np.ndarray,
    y: np.ndarray,
    split_seed: int,
) -> Dict[str, np.ndarray]:
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=split_seed, shuffle=True)
    return {
        "enc_A_train": enc_A[tr_idx],
        "enc_B_train": enc_B[tr_idx],
        "enc_full_train": enc_full[tr_idx],
        "y_train": y[tr_idx],
        "enc_A_test": enc_A[te_idx],
        "enc_B_test": enc_B[te_idx],
        "enc_full_test": enc_full[te_idx],
        "y_test": y[te_idx],
    }


def run_one_fraction_split(
    split_data: Dict[str, Any],
    fraction: float,
    model_type: str,
    device: torch.device,
    seed: int,
    epochs: Optional[int] = None,
    patience: Optional[int] = None,
) -> Tuple[float, float]:
    if epochs is None:
        epochs = EPOCHS
    if patience is None:
        patience = EARLY_STOPPING_PATIENCE
    set_seed(seed)

    n_train = len(split_data["y_train"])
    n_use = max(1, int(n_train * fraction))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_train)
    use_idx = perm[:n_use]

    sl = _split_train_val(
        split_data["enc_A_train"][use_idx],
        split_data["enc_B_train"][use_idx],
        split_data["enc_full_train"][use_idx],
        split_data["y_train"][use_idx],
        seed=seed,
    )

    if model_type == "value_based":
        model, scaler_A, scaler_B = train_value_based(
            sl["enc_A_tr"], sl["enc_B_tr"], sl["y_tr"],
            sl["enc_A_val"], sl["enc_B_val"], sl["y_val"],
            device, epochs=epochs, patience=patience,
        )
        if scaler_A is not None:
            enc_A_te = scaler_A.transform(split_data["enc_A_test"])
            enc_B_te = scaler_B.transform(split_data["enc_B_test"])
        else:
            enc_A_te = split_data["enc_A_test"]
            enc_B_te = split_data["enc_B_test"]
        model.eval()
        with torch.no_grad():
            a = torch.FloatTensor(enc_A_te).to(device)
            b = torch.FloatTensor(enc_B_te).to(device)
            test_pred = model(a, b).cpu().numpy()
    elif model_type in ("context_dependent", "context_dependent_sigmoid"):
        model, scaler = train_context_dependent(
            sl["enc_full_tr"], sl["y_tr"], sl["enc_full_val"], sl["y_val"],
            device, epochs=epochs, patience=patience, use_sigmoid=(model_type == "context_dependent_sigmoid"),
        )
        if scaler is not None:
            enc_te = scaler.transform(split_data["enc_full_test"])
        else:
            enc_te = split_data["enc_full_test"]
        model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(enc_te).to(device)
            test_pred = model(x).cpu().numpy()
    else:
        raise ValueError(f"未知 model_type: {model_type}")

    metrics = evaluate_model(split_data["y_test"], test_pred)
    return float(metrics["mse"]), float(metrics["cross_entropy"])


def collect_learning_curve_data(
    enc_A: np.ndarray,
    enc_B: np.ndarray,
    enc_full: np.ndarray,
    y: np.ndarray,
    seeds: List[int],
    device: torch.device,
    n_fractions: int = DATA_QUANTITY_N_FRACTIONS,
    epochs: Optional[int] = None,
    patience: Optional[int] = None,
    timestamp: Optional[str] = None,
    catch_errors: bool = False,
) -> Tuple[List[Dict[str, Any]], str]:
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fractions = [i / n_fractions for i in range(1, n_fractions + 1)]
    model_types_lc = ["value_based", "context_dependent", "context_dependent_sigmoid"]
    rows: List[Dict[str, Any]] = []
    print(f"\n数据量曲线: {n_fractions} 个比例, seeds={seeds}")

    for seed in seeds:
        split_seed = SPLIT_RANDOM_STATE + seed
        split_data = _train_test_split_once(enc_A, enc_B, enc_full, y, split_seed=split_seed)
        for fraction in fractions:
            for model_type in model_types_lc:
                print(f"  [seed={seed}] frac={fraction:.2f} {model_type}...")
                if catch_errors:
                    try:
                        test_mse, test_ce = run_one_fraction_split(
                            split_data, fraction, model_type, device, seed,
                            epochs=epochs, patience=patience,
                        )
                    except Exception as e:
                        print(f"    警告: 该点失败，记为 nan — {e}")
                        test_mse = float("nan")
                        test_ce = float("nan")
                else:
                    test_mse, test_ce = run_one_fraction_split(
                        split_data, fraction, model_type, device, seed,
                        epochs=epochs, patience=patience,
                    )
                rows.append(
                    {
                        "split_type": "train_test",
                        "model_type": model_type,
                        "fraction": fraction,
                        "seed": seed,
                        "test_mse": test_mse,
                        "test_cross_entropy": test_ce,
                    }
                )
    return rows, timestamp
