"""Model zoo. Every learner is tuned on DEV with the same expanding-window folds.

Protocol (all models):
  1. Hyper-parameters chosen on the DEVELOPMENT period only, using
     expanding-window time-series CV (no shuffling, no future rows in training).
  2. Number of boosting rounds / epochs chosen by early stopping on the last
     `val_months` of DEV.
  3. Final model refit on the full DEV period with the chosen settings.
  4. TEST period touched once.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from xgboost import XGBRegressor

import config as C
from config import SEED, CV_FOLDS, LSTM


# ---------------------------------------------------------------------------
def expanding_folds(dates: pd.Series, n_folds: int = CV_FOLDS):
    """Expanding-window CV split on *calendar dates* (panel-safe)."""
    ud = np.sort(dates.unique())
    blocks = np.array_split(ud, n_folds + 1)
    d = dates.to_numpy()
    folds = []
    for k in range(n_folds):
        tr_end = blocks[k][-1]
        va = blocks[k + 1]
        tr_idx = np.where(d <= tr_end)[0]
        va_idx = np.where((d >= va[0]) & (d <= va[-1]))[0]
        folds.append((tr_idx, va_idx))
    return folds


# ---------------------------------------------------------------------------
def fit_linear(X, y, dates):
    pipe = make_pipeline(StandardScaler(), Ridge())
    gs = GridSearchCV(pipe, {"ridge__alpha": np.logspace(-4, 3, 15)},
                      cv=expanding_folds(dates), scoring="neg_root_mean_squared_error",
                      n_jobs=-1)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_


def fit_rf(X, y, dates):
    space = dict(n_estimators=C.RF_TREES, max_depth=[None, 12, 20],
                 min_samples_leaf=[1, 3, 5, 10], max_features=[0.3, 0.5, "sqrt", 1.0])
    rs = RandomizedSearchCV(RandomForestRegressor(random_state=SEED, n_jobs=-1),
                            space, n_iter=C.N_ITER_SEARCH, cv=expanding_folds(dates),
                            scoring="neg_root_mean_squared_error",
                            random_state=SEED, n_jobs=1)
    rs.fit(X, y)
    return rs.best_estimator_, rs.best_params_


def fit_xgb(X, y, dates, val_mask):
    base = XGBRegressor(n_estimators=600, tree_method="hist", random_state=SEED,
                        n_jobs=-1, objective="reg:squarederror")
    space = dict(learning_rate=[0.02, 0.05, 0.1], max_depth=[3, 4, 6, 8],
                 subsample=[0.7, 0.85, 1.0], colsample_bytree=[0.5, 0.7, 1.0],
                 min_child_weight=[1, 5, 10], reg_lambda=[1.0, 5.0, 10.0])
    rs = RandomizedSearchCV(base, space, n_iter=C.N_ITER_SEARCH, cv=expanding_folds(dates),
                            scoring="neg_root_mean_squared_error",
                            random_state=SEED, n_jobs=1)
    rs.fit(X, y)
    p = rs.best_params_
    # early stopping to pick the number of rounds
    es = XGBRegressor(n_estimators=3000, tree_method="hist", random_state=SEED,
                      n_jobs=-1, early_stopping_rounds=100, **p)
    es.fit(X[~val_mask], y[~val_mask], eval_set=[(X[val_mask], y[val_mask])], verbose=False)
    n_best = int(es.best_iteration) + 1
    final = XGBRegressor(n_estimators=n_best, tree_method="hist", random_state=SEED,
                         n_jobs=-1, **p)
    final.fit(X, y)
    return final, {**p, "n_estimators": n_best}


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------
def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


class _Scaler:
    def fit(self, a, axis):
        self.m = a.mean(axis=axis, keepdims=True)
        self.s = a.std(axis=axis, keepdims=True) + 1e-6
        return self

    def __call__(self, a):
        return (a - self.m) / self.s


def _make_net(d_dyn, d_fut, hp):
    torch, nn = _torch()

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(d_dyn, hp["hidden"], num_layers=LSTM["layers"],
                                batch_first=True)
            self.head = nn.Sequential(nn.Linear(hp["hidden"] + d_fut, 64), nn.ReLU(),
                                      nn.Dropout(hp["dropout"]), nn.Linear(64, 1))

        def forward(self, xs, xf):
            _, (hN, _) = self.lstm(xs)
            return self.head(torch.cat([hN[-1], xf], dim=1)).squeeze(1)
    return Net()


def _train(net, Xs, Xf, y, epochs, seed, hp, Xs_v=None, Xf_v=None, y_v=None):
    """Train for at most `epochs`; with a validation set, stop early and return
    (best_epoch, best_val_loss)."""
    torch, nn = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"])
    lossf = nn.MSELoss()
    Xs, Xf, y = map(torch.from_numpy, (Xs, Xf, y.astype(np.float32)))
    n = len(y)
    best, best_ep, wait = np.inf, 0, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, LSTM["batch"]):
            b = perm[i:i + LSTM["batch"]]
            opt.zero_grad()
            loss = lossf(net(Xs[b], Xf[b]), y[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        if Xs_v is not None:
            net.eval()
            with torch.no_grad():
                vl = lossf(net(torch.from_numpy(Xs_v), torch.from_numpy(Xf_v)),
                           torch.from_numpy(y_v.astype(np.float32))).item()
            if vl < best - 1e-5:
                best, best_ep, wait = vl, ep + 1, 0
            else:
                wait += 1
                if wait >= LSTM["patience"]:
                    break
    return best_ep, best


def _fit_es_then_refit(Xs_tr, Xf_tr, y_tr, es_mask, hp, seed):
    """Early-stop on es_mask rows (scalers fitted on the non-ES rows), then refit on
    ALL rows for exactly the early-stopped number of epochs (no rescaling of the
    epoch count). Returns (net, scalers, best_epoch)."""
    torch, _ = _torch()
    tr = ~es_mask
    ss = _Scaler().fit(Xs_tr[tr], axis=(0, 1)); sf = _Scaler().fit(Xf_tr[tr], axis=0)
    ym, ysd = y_tr[tr].mean(), y_tr[tr].std()
    net = _make_net(Xs_tr.shape[2], Xf_tr.shape[1], hp)
    best_ep, _ = _train(net, ss(Xs_tr[tr]).astype(np.float32), sf(Xf_tr[tr]).astype(np.float32),
                        (y_tr[tr] - ym) / ysd, LSTM["max_epochs"], seed, hp,
                        ss(Xs_tr[es_mask]).astype(np.float32),
                        sf(Xf_tr[es_mask]).astype(np.float32), (y_tr[es_mask] - ym) / ysd)
    best_ep = max(best_ep, 1)
    ss = _Scaler().fit(Xs_tr, axis=(0, 1)); sf = _Scaler().fit(Xf_tr, axis=0)
    ym, ysd = y_tr.mean(), y_tr.std()
    net = _make_net(Xs_tr.shape[2], Xf_tr.shape[1], hp)
    _train(net, ss(Xs_tr).astype(np.float32), sf(Xf_tr).astype(np.float32),
           (y_tr - ym) / ysd, best_ep, seed, hp)
    return net, (ss, sf, ym, ysd), best_ep


def _predict(net, sc, Xs, Xf):
    torch, _ = _torch()
    ss, sf, ym, ysd = sc
    net.eval()
    with torch.no_grad():
        return net(torch.from_numpy(ss(Xs).astype(np.float32)),
                   torch.from_numpy(sf(Xf).astype(np.float32))).numpy() * ysd + ym


def tune_lstm(Xs_dev, Xf_dev, y_dev, dates_dev):
    """Random search with the same budget (N_ITER_SEARCH) and the same
    expanding-window date folds as RF/XGBoost. Inside each fold the last 20 % of
    the fold-training dates are the early-stopping block. Score = mean fold RMSE."""
    rng = np.random.default_rng(SEED)
    keys = list(C.LSTM_SPACE)
    seen, cands = set(), []
    while len(cands) < C.N_ITER_SEARCH:
        hp = {k: C.LSTM_SPACE[k][rng.integers(len(C.LSTM_SPACE[k]))] for k in keys}
        t = tuple(hp[k] for k in keys)
        if t not in seen:
            seen.add(t); cands.append(hp)
    folds = expanding_folds(dates_dev)
    d = dates_dev.to_numpy()
    results = []
    for hp in cands:
        L = hp["window"]
        errs = []
        for tr_idx, va_idx in folds:
            tr_dates = np.sort(np.unique(d[tr_idx]))
            cut = tr_dates[int(len(tr_dates) * 0.8)]
            es = d[tr_idx] >= cut
            net, sc, _ = _fit_es_then_refit(Xs_dev[tr_idx][:, -L:], Xf_dev[tr_idx],
                                            y_dev[tr_idx], es, hp, seed=0)
            p = _predict(net, sc, Xs_dev[va_idx][:, -L:], Xf_dev[va_idx])
            errs.append(float(np.sqrt(np.mean((y_dev[va_idx] - p) ** 2))))
        results.append((float(np.mean(errs)), hp))
    results.sort(key=lambda r: r[0])
    return results[0][1], [dict(cv_rmse=r, **h) for r, h in results]


def fit_predict_lstm(Xs_dev, Xf_dev, y_dev, val_mask, Xs_te, Xf_te, dates_dev=None,
                     tune=True):
    """Tune (optional), then per seed: early-stop on the last `val_months` of DEV,
    refit on full DEV for the early-stopped epoch count, predict TEST.
    Returns test predictions averaged over seeds, per-seed predictions, info."""
    torch, _ = _torch()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    if tune and dates_dev is not None:
        hp, search = tune_lstm(Xs_dev, Xf_dev, y_dev, dates_dev)
    else:
        hp, search = {k: LSTM[k] for k in ("window", "hidden", "dropout", "lr")}, []
    L = hp["window"]
    preds, epochs_used = [], []
    for seed in LSTM["seeds"]:
        net, sc, best_ep = _fit_es_then_refit(Xs_dev[:, -L:], Xf_dev, y_dev, val_mask, hp, seed)
        preds.append(_predict(net, sc, Xs_te[:, -L:], Xf_te))
        epochs_used.append(best_ep)
    preds = np.stack(preds)
    return preds.mean(0), preds, {"epochs": epochs_used, "hp": hp, "search": search}
