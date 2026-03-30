from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier, XGBRegressor

from .artifacts import load_overnight_branch_settings, load_overnight_search_budgets
from .features import SEED

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class BranchPredictions:
    name: str
    val_proba: np.ndarray
    val_reg: np.ndarray
    test_proba: np.ndarray
    test_reg: np.ndarray


@dataclass
class PipelineMetrics:
    accuracy: float
    r2: float
    combined: float


def evaluate_predictions(
    y_true_cls: np.ndarray,
    y_true_reg: np.ndarray,
    proba: np.ndarray,
    reg_pred: np.ndarray,
) -> PipelineMetrics:
    pred_cls = np.argmax(proba, axis=1)
    reg_pred = np.clip(reg_pred, 4.99, 35.99)
    accuracy = float(accuracy_score(y_true_cls, pred_cls))
    r2 = float(r2_score(y_true_reg, reg_pred))
    return PipelineMetrics(accuracy=accuracy, r2=r2, combined=0.5 * (accuracy + r2))


def scale_training_budget(params: dict, smoke_test: bool, kind: str) -> dict:
    adjusted = dict(params)
    if not smoke_test:
        return adjusted
    if "n_estimators" in adjusted:
        adjusted["n_estimators"] = max(200, min(600, int(adjusted["n_estimators"] * 0.18)))
    if "iterations" in adjusted:
        adjusted["iterations"] = max(300, min(900, int(adjusted["iterations"] * 0.18)))
    if kind == "regressor":
        adjusted["learning_rate"] = min(0.08, adjusted["learning_rate"] * 1.35)
    else:
        adjusted["learning_rate"] = min(0.06, adjusted["learning_rate"] * 1.30)
    return adjusted


def choose_torch_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def standardize(train: np.ndarray, others: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-6
    return (train - mean) / std, [(other - mean) / std for other in others]


def overnight_branch_settings() -> dict[str, dict]:
    return load_overnight_branch_settings()


def overnight_search_budgets() -> dict:
    return load_overnight_search_budgets()


def softmax_weights(raw_values: list[float]) -> np.ndarray:
    values = np.asarray(raw_values, dtype=float)
    values = values - values.max()
    exp_values = np.exp(values)
    return exp_values / exp_values.sum()


def train_xgb_branch(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_full: np.ndarray,
    X_test: np.ndarray,
    y_train_cls: np.ndarray,
    y_train_reg: np.ndarray,
    y_cls: np.ndarray,
    y_reg: np.ndarray,
    *,
    smoke_test: bool,
) -> BranchPredictions:
    settings = overnight_branch_settings()["xgb"]
    cls_params = scale_training_budget(settings["classifier"], smoke_test, "classifier")
    reg_params = scale_training_budget(settings["regressor"], smoke_test, "regressor")

    clf = XGBClassifier(
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=4,
        **cls_params,
    )
    reg = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=SEED,
        n_jobs=4,
        **reg_params,
    )
    clf.fit(X_train, y_train_cls)
    reg.fit(X_train, y_train_reg)

    clf_full = XGBClassifier(
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=4,
        **cls_params,
    )
    reg_full = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=SEED,
        n_jobs=4,
        **reg_params,
    )
    clf_full.fit(X_full, y_cls)
    reg_full.fit(X_full, y_reg)

    return BranchPredictions(
        name="xgb",
        val_proba=clf.predict_proba(X_val).astype(np.float32),
        val_reg=np.clip(reg.predict(X_val), 4.99, 35.99).astype(np.float32),
        test_proba=clf_full.predict_proba(X_test).astype(np.float32),
        test_reg=np.clip(reg_full.predict(X_test), 4.99, 35.99).astype(np.float32),
    )


def train_lgb_branch(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_full: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_cls: np.ndarray,
    y_train_reg: np.ndarray,
    y_cls: np.ndarray,
    y_reg: np.ndarray,
    *,
    smoke_test: bool,
) -> BranchPredictions:
    settings = overnight_branch_settings()["lgb"]
    cls_params = scale_training_budget(settings["classifier"], smoke_test, "classifier")
    reg_params = scale_training_budget(settings["regressor"], smoke_test, "regressor")

    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=5,
        random_state=SEED,
        n_jobs=4,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
        **cls_params,
    )
    reg = lgb.LGBMRegressor(
        objective="regression",
        random_state=SEED,
        n_jobs=4,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
        **reg_params,
    )
    clf.fit(X_train, y_train_cls)
    reg.fit(X_train, y_train_reg)

    clf_full = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=5,
        random_state=SEED,
        n_jobs=4,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
        **cls_params,
    )
    reg_full = lgb.LGBMRegressor(
        objective="regression",
        random_state=SEED,
        n_jobs=4,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
        **reg_params,
    )
    clf_full.fit(X_full, y_cls)
    reg_full.fit(X_full, y_reg)

    return BranchPredictions(
        name="lgb",
        val_proba=clf.predict_proba(X_val).astype(np.float32),
        val_reg=np.clip(reg.predict(X_val), 4.99, 35.99).astype(np.float32),
        test_proba=clf_full.predict_proba(X_test).astype(np.float32),
        test_reg=np.clip(reg_full.predict(X_test), 4.99, 35.99).astype(np.float32),
    )


def train_catboost_branch(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_full: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_columns: list[str],
    y_train_cls: np.ndarray,
    y_train_reg: np.ndarray,
    y_cls: np.ndarray,
    y_reg: np.ndarray,
    *,
    smoke_test: bool,
) -> BranchPredictions:
    settings = overnight_branch_settings()["cat"]
    cls_params = scale_training_budget(settings["classifier"], smoke_test, "classifier")
    reg_params = scale_training_budget(settings["regressor"], smoke_test, "regressor")
    cat_idx = [X_train.columns.get_loc(col) for col in categorical_columns if col in X_train.columns]

    train_pool_cls = Pool(X_train, y_train_cls, cat_features=cat_idx)
    val_pool_cls = Pool(X_val, cat_features=cat_idx)
    train_pool_reg = Pool(X_train, y_train_reg, cat_features=cat_idx)
    val_pool_reg = Pool(X_val, cat_features=cat_idx)
    full_pool_cls = Pool(X_full, y_cls, cat_features=cat_idx)
    full_pool_reg = Pool(X_full, y_reg, cat_features=cat_idx)
    test_pool = Pool(X_test, cat_features=cat_idx)

    clf = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=SEED,
        thread_count=4,
        allow_writing_files=False,
        task_type="CPU",
        verbose=False,
        **cls_params,
    )
    reg = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=SEED,
        thread_count=4,
        allow_writing_files=False,
        task_type="CPU",
        verbose=False,
        **reg_params,
    )
    clf.fit(train_pool_cls)
    reg.fit(train_pool_reg)

    clf_full = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=SEED,
        thread_count=4,
        allow_writing_files=False,
        task_type="CPU",
        verbose=False,
        **cls_params,
    )
    reg_full = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=SEED,
        thread_count=4,
        allow_writing_files=False,
        task_type="CPU",
        verbose=False,
        **reg_params,
    )
    clf_full.fit(full_pool_cls)
    reg_full.fit(full_pool_reg)

    return BranchPredictions(
        name="cat",
        val_proba=clf.predict_proba(val_pool_cls).astype(np.float32),
        val_reg=np.clip(reg.predict(val_pool_reg), 4.99, 35.99).astype(np.float32),
        test_proba=clf_full.predict_proba(test_pool).astype(np.float32),
        test_reg=np.clip(reg_full.predict(test_pool), 4.99, 35.99).astype(np.float32),
    )


def train_extratrees_branch(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_full: np.ndarray,
    X_test: np.ndarray,
    y_train_cls: np.ndarray,
    y_train_reg: np.ndarray,
    y_cls: np.ndarray,
    y_reg: np.ndarray,
    *,
    smoke_test: bool,
) -> BranchPredictions:
    settings = overnight_branch_settings()["et"]
    n_estimators_cls = 300 if smoke_test else settings["n_estimators_classifier"]
    n_estimators_reg = 350 if smoke_test else settings["n_estimators_regressor"]
    clf = ExtraTreesClassifier(
        n_estimators=n_estimators_cls,
        max_features="sqrt",
        min_samples_leaf=1,
        random_state=SEED,
        n_jobs=8,
    )
    reg = ExtraTreesRegressor(
        n_estimators=n_estimators_reg,
        max_features="sqrt",
        min_samples_leaf=1,
        random_state=SEED,
        n_jobs=8,
    )
    clf.fit(X_train, y_train_cls)
    reg.fit(X_train, y_train_reg)

    clf_full = ExtraTreesClassifier(
        n_estimators=n_estimators_cls,
        max_features="sqrt",
        min_samples_leaf=1,
        random_state=SEED,
        n_jobs=8,
    )
    reg_full = ExtraTreesRegressor(
        n_estimators=n_estimators_reg,
        max_features="sqrt",
        min_samples_leaf=1,
        random_state=SEED,
        n_jobs=8,
    )
    clf_full.fit(X_full, y_cls)
    reg_full.fit(X_full, y_reg)

    return BranchPredictions(
        name="et",
        val_proba=clf.predict_proba(X_val).astype(np.float32),
        val_reg=np.clip(reg.predict(X_val), 4.99, 35.99).astype(np.float32),
        test_proba=clf_full.predict_proba(X_test).astype(np.float32),
        test_reg=np.clip(reg_full.predict(X_test), 4.99, 35.99).astype(np.float32),
    )


class MultitaskMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout * 0.8),
        )
        self.classifier = nn.Linear(hidden_dim // 2, 5)
        self.regressor = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        return self.classifier(hidden), self.regressor(hidden).squeeze(1)


def train_mps_branch(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_full: np.ndarray,
    X_test: np.ndarray,
    y_train_cls: np.ndarray,
    y_val_cls: np.ndarray,
    y_train_reg: np.ndarray,
    y_val_reg: np.ndarray,
    y_cls: np.ndarray,
    y_reg: np.ndarray,
    *,
    smoke_test: bool,
) -> BranchPredictions:
    device = choose_torch_device()
    mps_settings = overnight_branch_settings()["mps"]
    selected_members = mps_settings["selected_members"]
    member_configs = selected_members[:1] if smoke_test else selected_members[:3]
    X_train_s, [X_val_s, X_full_s, X_test_s] = standardize(X_train, [X_val, X_full, X_test])
    X_full_fit_s, [X_test_full_s] = standardize(X_full, [X_test])

    val_proba = np.zeros((X_val.shape[0], 5), dtype=np.float32)
    val_reg = np.zeros(X_val.shape[0], dtype=np.float32)
    test_proba = np.zeros((X_test.shape[0], 5), dtype=np.float32)
    test_reg = np.zeros(X_test.shape[0], dtype=np.float32)

    # The MPS branch is the diversity spice rack, not the whole kitchen.
    for member in member_configs:
        seed = int(member["seed"])
        hidden_dim = int(member["hidden"])
        dropout = float(member["dropout"])
        learning_rate = float(member["lr"])
        reference_epochs = max(12, int(round(0.5 * (member["cls_best_epoch"] + member["reg_best_epoch"]))))
        max_epochs = min(18, reference_epochs) if smoke_test else reference_epochs
        patience = max(5, max_epochs // 3)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = MultitaskMLP(X_train.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=2e-5)
        cls_loss = nn.CrossEntropyLoss()
        reg_loss = nn.MSELoss()

        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_train_s.astype(np.float32)),
                torch.from_numpy(y_train_cls.astype(np.int64)),
                torch.from_numpy(y_train_reg.astype(np.float32)),
            ),
            batch_size=1024 if not smoke_test else 512,
            shuffle=True,
            drop_last=False,
        )
        X_val_tensor = torch.from_numpy(X_val_s.astype(np.float32)).to(device)

        best_state = None
        best_score = -np.inf
        wait = 0

        for _epoch in range(1, max_epochs + 1):
            model.train()
            for xb, yb_cls, yb_reg in loader:
                xb = xb.to(device)
                yb_cls = yb_cls.to(device)
                yb_reg = yb_reg.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits, reg_out = model(xb)
                loss = cls_loss(logits, yb_cls) + 0.35 * reg_loss(reg_out, yb_reg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_logits, val_reg_out = model(X_val_tensor)
                member_val_proba = torch.softmax(val_logits, dim=1).cpu().numpy()
                member_val_reg = np.clip(val_reg_out.cpu().numpy(), 4.99, 35.99)
            score = evaluate_predictions(y_val_cls, y_val_reg, member_val_proba, member_val_reg).combined
            if score > best_score + 1e-6:
                best_score = score
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state is None:
            raise RuntimeError("MPS branch failed to capture a checkpoint.")

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            val_logits, val_reg_out = model(torch.from_numpy(X_val_s.astype(np.float32)).to(device))
        val_proba += torch.softmax(val_logits, dim=1).cpu().numpy().astype(np.float32) / len(member_configs)
        val_reg += np.clip(val_reg_out.cpu().numpy(), 4.99, 35.99).astype(np.float32) / len(member_configs)

        full_model = MultitaskMLP(X_full.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
        full_optimizer = torch.optim.AdamW(full_model.parameters(), lr=learning_rate, weight_decay=2e-5)
        full_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_full_fit_s.astype(np.float32)),
                torch.from_numpy(y_cls.astype(np.int64)),
                torch.from_numpy(y_reg.astype(np.float32)),
            ),
            batch_size=1024 if not smoke_test else 512,
            shuffle=True,
            drop_last=False,
        )
        for _epoch in range(max(8, max_epochs)):
            full_model.train()
            for xb, yb_cls, yb_reg in full_loader:
                xb = xb.to(device)
                yb_cls = yb_cls.to(device)
                yb_reg = yb_reg.to(device)
                full_optimizer.zero_grad(set_to_none=True)
                logits, reg_out = full_model(xb)
                loss = cls_loss(logits, yb_cls) + 0.35 * reg_loss(reg_out, yb_reg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(full_model.parameters(), 1.0)
                full_optimizer.step()
        full_model.eval()
        with torch.no_grad():
            test_logits, test_reg_out = full_model(torch.from_numpy(X_test_full_s.astype(np.float32)).to(device))
        test_proba += torch.softmax(test_logits, dim=1).cpu().numpy().astype(np.float32) / len(member_configs)
        test_reg += np.clip(test_reg_out.cpu().numpy(), 4.99, 35.99).astype(np.float32) / len(member_configs)

    return BranchPredictions(
        name="mps",
        val_proba=val_proba,
        val_reg=val_reg,
        test_proba=test_proba,
        test_reg=test_reg,
    )


def fit_weighted_blend(
    branches: list[BranchPredictions],
    y_val_cls: np.ndarray,
    y_val_reg: np.ndarray,
    *,
    smoke_test: bool,
) -> BranchPredictions:
    budgets = overnight_search_budgets()
    timeout_seconds = 90 if smoke_test else int(budgets["blend_timeout_seconds"])
    trial_cap = 40 if smoke_test else int(budgets["blend_trial_cap"])
    names = [branch.name for branch in branches]
    branch_map = {branch.name: branch for branch in branches}

    def objective(trial: optuna.Trial) -> float:
        cls_weights = softmax_weights([trial.suggest_float(f"cls_{name}", -4.0, 4.0) for name in names])
        reg_weights = softmax_weights([trial.suggest_float(f"reg_{name}", -4.0, 4.0) for name in names])
        proba = np.zeros_like(branches[0].val_proba)
        reg = np.zeros_like(branches[0].val_reg)
        for idx, name in enumerate(names):
            proba += cls_weights[idx] * branch_map[name].val_proba
            reg += reg_weights[idx] * branch_map[name].val_reg
        return evaluate_predictions(y_val_cls, y_val_reg, proba, reg).combined

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=trial_cap, timeout=timeout_seconds, show_progress_bar=False)
    best_params = study.best_params
    cls_weights = softmax_weights([best_params[f"cls_{name}"] for name in names])
    reg_weights = softmax_weights([best_params[f"reg_{name}"] for name in names])

    val_proba = np.zeros_like(branches[0].val_proba)
    val_reg = np.zeros_like(branches[0].val_reg)
    test_proba = np.zeros_like(branches[0].test_proba)
    test_reg = np.zeros_like(branches[0].test_reg)
    for idx, name in enumerate(names):
        val_proba += cls_weights[idx] * branch_map[name].val_proba
        val_reg += reg_weights[idx] * branch_map[name].val_reg
        test_proba += cls_weights[idx] * branch_map[name].test_proba
        test_reg += reg_weights[idx] * branch_map[name].test_reg

    return BranchPredictions(
        name="blend_overnight",
        val_proba=val_proba.astype(np.float32),
        val_reg=np.clip(val_reg, 4.99, 35.99).astype(np.float32),
        test_proba=test_proba.astype(np.float32),
        test_reg=np.clip(test_reg, 4.99, 35.99).astype(np.float32),
    )


def fit_meta_ensemble(
    branches: list[BranchPredictions],
    y_val_cls: np.ndarray,
    y_val_reg: np.ndarray,
    *,
    smoke_test: bool,
) -> BranchPredictions:
    budgets = overnight_search_budgets()
    meta_trials = budgets["meta_trials"]
    cls_trials = 18 if smoke_test else int(meta_trials["classifier"])
    reg_trials = 18 if smoke_test else int(meta_trials["regressor"])
    n_splits = 3 if smoke_test else 5

    cls_val = [branch.val_proba for branch in branches]
    reg_val = [branch.val_reg[:, None] for branch in branches]
    cls_test = [branch.test_proba for branch in branches]
    reg_test = [branch.test_reg[:, None] for branch in branches]

    X_cls_val = np.hstack(cls_val + reg_val).astype(np.float32)
    X_reg_val = np.hstack(reg_val + cls_val).astype(np.float32)
    X_cls_test = np.hstack(cls_test + reg_test).astype(np.float32)
    X_reg_test = np.hstack(reg_test + cls_test).astype(np.float32)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    def cls_objective(trial: optuna.Trial) -> float:
        model = LogisticRegression(
            C=trial.suggest_float("C", 0.005, 60.0, log=True),
            max_iter=5_000,
            solver="lbfgs",
            random_state=SEED,
        )
        scores = []
        for train_idx, val_idx in skf.split(X_cls_val, y_val_cls):
            model.fit(X_cls_val[train_idx], y_val_cls[train_idx])
            pred = model.predict(X_cls_val[val_idx])
            scores.append(accuracy_score(y_val_cls[val_idx], pred))
        return float(np.mean(scores))

    def reg_objective(trial: optuna.Trial) -> float:
        model = Ridge(alpha=trial.suggest_float("alpha", 1e-5, 800.0, log=True), random_state=SEED)
        scores = []
        for train_idx, val_idx in kf.split(X_reg_val):
            model.fit(X_reg_val[train_idx], y_val_reg[train_idx])
            pred = np.clip(model.predict(X_reg_val[val_idx]), 4.99, 35.99)
            scores.append(r2_score(y_val_reg[val_idx], pred))
        return float(np.mean(scores))

    cls_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    reg_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    cls_study.optimize(cls_objective, n_trials=cls_trials, show_progress_bar=False)
    reg_study.optimize(reg_objective, n_trials=reg_trials, show_progress_bar=False)

    cls_model = LogisticRegression(
        C=float(cls_study.best_params["C"]),
        max_iter=5_000,
        solver="lbfgs",
        random_state=SEED,
    )
    reg_model = Ridge(alpha=float(reg_study.best_params["alpha"]), random_state=SEED)
    cls_model.fit(X_cls_val, y_val_cls)
    reg_model.fit(X_reg_val, y_val_reg)

    return BranchPredictions(
        name="meta_overnight",
        val_proba=cls_model.predict_proba(X_cls_val).astype(np.float32),
        val_reg=np.clip(reg_model.predict(X_reg_val), 4.99, 35.99).astype(np.float32),
        test_proba=cls_model.predict_proba(X_cls_test).astype(np.float32),
        test_reg=np.clip(reg_model.predict(X_reg_test), 4.99, 35.99).astype(np.float32),
    )


def build_submission_frame(predictions: BranchPredictions) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Id": np.arange(predictions.test_proba.shape[0]),
            "RiskTier": np.argmax(predictions.test_proba, axis=1).astype(int),
            "InterestRate": np.clip(predictions.test_reg, 4.99, 35.99).round(2),
        }
    )
