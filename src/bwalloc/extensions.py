"""Optional forecasters from Section 3.4: SARIMA (per user) and an LSTM (pooled).

Both produce one-step-ahead forecasts for exactly the same holdout windows and users as
the main pipeline (results/<ds>/holdout_true.csv), so their errors and their effect
on allocation can be compared like-for-like with the Random Forest.

No leakage:
  * SARIMA parameters are fitted on the development period only. The fitted model is then
    run over the full series with those parameters FIXED (statsmodels `apply`), and only
    the one-step-ahead predictions are used. Each prediction for tau uses data <= tau-1.
  * The LSTM is trained on development windows only; each holdout prediction uses the
    previous L windows.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ SARIMA


def sarima_forecast(wide: pd.DataFrame, hold_index: pd.DatetimeIndex, season: int,
                    order=(1, 0, 1), seasonal_order=(1, 0, 1), log=print) -> pd.DataFrame:
    """Per-user SARIMA on log1p(demand). Returns a hold_index x users frame of forecasts."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    start = hold_index[0]
    out = pd.DataFrame(index=hold_index, columns=wide.columns, dtype=float)
    t0 = time.time()
    for k, user in enumerate(wide.columns, 1):
        y = np.log1p(wide[user].astype(float))
        dev = y[y.index < start]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SARIMAX(dev, order=order, seasonal_order=(*seasonal_order, season),
                          trend="c").fit(disp=False, maxiter=50)
            full = res.apply(y)  # same parameters, no refit
            pred = full.get_prediction(start=start, end=hold_index[-1], dynamic=False)
        out[user] = np.expm1(pred.predicted_mean.reindex(hold_index)).clip(lower=0).to_numpy()
        if k % 10 == 0 or k == wide.shape[1]:
            log(f"    SARIMA {k}/{wide.shape[1]} users ({time.time() - t0:.0f}s)")
    return out


# ------------------------------------------------------------------ LSTM


def _sequences(x: np.ndarray, time_feats: np.ndarray, idx: np.ndarray, L: int):
    """x: (T, U) scaled demand; time_feats: (T, F). For target rows idx -> (N, L, 1+F), y."""
    T, U = x.shape
    seqs, ys, users = [], [], []
    for t in idx:
        window = x[t - L:t]                      # (L, U)  data up to t-1
        tf = np.repeat(time_feats[t - L + 1:t + 1][:, None, :], U, axis=1)  # time of next step
        seqs.append(np.concatenate([window[..., None], tf], axis=2).transpose(1, 0, 2))
        ys.append(x[t])
        users.append(np.arange(U))
    return (np.concatenate(seqs).astype(np.float32), np.concatenate(ys).astype(np.float32),
            np.concatenate(users))


def lstm_forecast(wide: pd.DataFrame, hold_index: pd.DatetimeIndex, L: int = 24,
                  hidden: int = 32, epochs: int = 30, patience: int = 4,
                  max_train: int = 60000, seed: int = 42, log=print) -> tuple[pd.DataFrame, dict]:
    """Pooled 1-layer LSTM on demand scaled by each user's development mean."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    start = hold_index[0]
    dev_mask = wide.index < start
    scale = wide[dev_mask].mean().clip(lower=1e-9)
    x = (wide / scale).to_numpy(dtype=float)
    ts = wide.index
    hour = ts.hour + ts.minute / 60
    time_feats = np.c_[np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                       (ts.day_of_week >= 5).astype(float)]

    n_dev = int(dev_mask.sum())
    dev_idx = np.arange(L, n_dev)
    split = int(len(dev_idx) * 0.85)  # last 15% of dev = early-stopping validation
    Xtr, ytr, _ = _sequences(x, time_feats, dev_idx[:split], L)
    Xva, yva, _ = _sequences(x, time_feats, dev_idx[split:], L)
    if len(Xtr) > max_train:
        keep = rng.choice(len(Xtr), max_train, replace=False)
        Xtr, ytr = Xtr[keep], ytr[keep]

    class Net(nn.Module):
        def __init__(self, n_in):
            super().__init__()
            self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, s):
            out, _ = self.lstm(s)
            return self.head(out[:, -1]).squeeze(-1)

    net = Net(Xtr.shape[2])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xva_t, yva_t = torch.from_numpy(Xva), torch.from_numpy(yva)
    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr_t))
        for b in range(0, len(perm), 512):
            i = perm[b:b + 512]
            opt.zero_grad()
            loss = loss_fn(net(Xtr_t[i]), ytr_t[i])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            va = float(loss_fn(net(Xva_t), yva_t))
        hist.append(va)
        if va < best - 1e-5:
            best, bad = va, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    log(f"    LSTM trained {len(hist)} epochs in {time.time() - t0:.0f}s, best val MSE {best:.4f}")

    hold_rows = np.searchsorted(ts, hold_index)
    Xh, _, _ = _sequences(x, time_feats, hold_rows, L)
    with torch.no_grad():
        p = net(torch.from_numpy(Xh)).numpy().reshape(len(hold_rows), -1)
    pred = pd.DataFrame(np.clip(p, 0, None) * scale.to_numpy(), index=hold_index,
                        columns=wide.columns)
    return pred, {"epochs": len(hist), "best_val_mse": best, "train_seqs": len(Xtr),
                  "seq_len": L, "hidden": hidden}
