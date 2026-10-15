"""Modelos para el componente estocástico: ELM, MLP, NPAR (AR aditivo con splines cúbicas),
bosque aleatorio (RF), XGBoost (XGB) y redes recurrentes GRU y LSTM (PyTorch)."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


class _Scaled:
    """Estandariza X e y con estadísticos del entrenamiento."""

    def _fit_scale(self, X, y):
        self.mx, self.sx = X.mean(0), X.std(0)
        self.sx[self.sx == 0] = 1.0
        self.my, self.sy = y.mean(), y.std() or 1.0
        return (X - self.mx) / self.sx, (y - self.my) / self.sy

    def _tx(self, X):
        return (X - self.mx) / self.sx


class ELM(_Scaled):
    """Extreme Learning Machine (Huang et al., 2006/2011): una capa oculta sigmoide,
    pesos de entrada ~ U(low, high) fijos, pesos de salida por pseudoinversa de
    Moore–Penrose (o ridge si ridge > 0). Se promedia un ensamble de semillas para
    que el resultado no dependa de una sola inicialización aleatoria."""

    def __init__(self, n_hidden=50, n_seeds=20, low=-1.0, high=1.0, ridge=0.0, seed=0):
        self.n_hidden, self.n_seeds, self.low, self.high = n_hidden, n_seeds, low, high
        self.ridge, self.seed = ridge, seed

    @staticmethod
    def _sig(Z):
        return 1.0 / (1.0 + np.exp(-np.clip(Z, -50, 50)))

    def fit(self, X, y):
        Xs, ys = self._fit_scale(X, y)
        rng = np.random.default_rng(self.seed)
        self.members = []
        for _ in range(self.n_seeds):
            W = rng.uniform(self.low, self.high, (X.shape[1], self.n_hidden))
            b = rng.uniform(self.low, self.high, self.n_hidden)
            H = self._sig(Xs @ W + b)
            if self.ridge > 0:
                beta = np.linalg.solve(H.T @ H + self.ridge * np.eye(self.n_hidden), H.T @ ys)
            else:
                beta = np.linalg.pinv(H) @ ys
            self.members.append((W, b, beta))
        return self

    def predict(self, X):
        Xs = self._tx(X)
        P = [self._sig(Xs @ W + b) @ beta for W, b, beta in self.members]
        return np.mean(P, axis=0) * self.sy + self.my


class MLP(_Scaled):
    """Perceptrón multicapa autorregresivo: una capa oculta logística, salida lineal.
    Ensamble de semillas. (scikit-learn no implementa Levenberg–Marquardt: el
    manuscrito debe declarar el solver realmente usado, p. ej. L-BFGS.)"""

    def __init__(self, n_hidden=3, solver="lbfgs", max_iter=1000, alpha=1e-4, n_seeds=3, seed=0):
        self.n_hidden, self.solver, self.max_iter = n_hidden, solver, max_iter
        self.alpha, self.n_seeds, self.seed = alpha, n_seeds, seed

    def fit(self, X, y):
        Xs, ys = self._fit_scale(X, y)
        self.members = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for s in range(self.n_seeds):
                kw = dict(hidden_layer_sizes=(self.n_hidden,), activation="logistic",
                          solver=self.solver, max_iter=self.max_iter, alpha=self.alpha,
                          random_state=self.seed + s)
                if self.solver == "adam":
                    kw.update(early_stopping=True, n_iter_no_change=10, validation_fraction=0.1, shuffle=False)
                self.members.append(MLPRegressor(**kw).fit(Xs, ys))
        return self

    def predict(self, X):
        Xs = self._tx(X)
        return np.mean([m.predict(Xs) for m in self.members], axis=0) * self.sy + self.my


class NPAR:
    """AR aditivo no paramétrico: e_t = sum_j q_j(e_{t-j}) + eps, con cada q_j una
    spline cúbica de regresión (B-splines) y penalización ridge. Fuera del rango de
    entrenamiento cada q_j se mantiene constante (sin extrapolación lineal), lo que evita
    pronósticos explosivos al retransformar con exp()."""

    def __init__(self, n_knots=6, degree=3, alpha=1.0):
        self.n_knots, self.degree, self.alpha = n_knots, degree, alpha

    def fit(self, X, y):
        self.model = make_pipeline(
            StandardScaler(),
            SplineTransformer(n_knots=self.n_knots, degree=self.degree, extrapolation="constant"),
            Ridge(alpha=self.alpha),
        ).fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)



class RF:
    """Bosque aleatorio de regresión (Breiman, 2001) sobre los rezagos del residuo."""

    def __init__(self, n_estimators=300, max_depth=None, min_samples_leaf=5, seed=0):
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                           min_samples_leaf=min_samples_leaf, max_features=0.5,
                                           n_jobs=1, random_state=seed)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class XGB:
    """Gradient boosting de árboles. backend="hgb": HistGradientBoostingRegressor de scikit-learn
    (el usado en el manuscrito, rotulado HGB); backend="xgboost"/"auto": XGBoost (Chen y Guestrin,
    2016) y, si no puede cargarse, HGB. El backend efectivo queda en `backend`."""

    def __init__(self, max_depth=4, n_estimators=400, learning_rate=0.05, seed=0, backend="auto"):
        try:
            if backend == "hgb":
                raise ImportError("backend 'hgb' solicitado en config.yaml")
            from xgboost import XGBRegressor
            self.model = XGBRegressor(max_depth=max_depth, n_estimators=n_estimators,
                                      learning_rate=learning_rate, subsample=0.8, colsample_bytree=0.8,
                                      n_jobs=1, random_state=seed, verbosity=0)
            self.backend = "xgboost"
        except Exception:  # paquete ausente o librería OpenMP (libomp) no disponible  # pragma: no cover
            from sklearn.ensemble import HistGradientBoostingRegressor
            self.model = HistGradientBoostingRegressor(max_depth=max_depth, max_iter=n_estimators,
                                                       learning_rate=learning_rate, random_state=seed)
            self.backend = "sklearn-hgb"

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class RNN(_Scaled):
    """Red recurrente (GRU o LSTM) de una capa, implementada en PyTorch. La entrada es la
    secuencia de los p rezagos en orden cronológico; salida lineal. Entrenamiento con Adam,
    lotes de 256 y parada temprana sobre el último 10 % (temporal) del conjunto de entrenamiento.
    Se promedian `n_seeds` redes."""

    def __init__(self, cell="GRU", hidden=32, epochs=60, patience=6, lr=1e-3, n_seeds=1, seed=0):
        self.cell, self.hidden, self.epochs, self.patience = cell, hidden, epochs, patience
        self.lr, self.n_seeds, self.seed = lr, n_seeds, seed

    def _seq(self, X):
        # columnas de X: [lag1, lag2, ..., lagp] -> secuencia cronológica (lagp ... lag1)
        import torch
        return torch.tensor(X[:, ::-1].copy(), dtype=torch.float32).unsqueeze(-1)

    def fit(self, X, y):
        import torch
        from torch import nn
        torch.set_num_threads(1)
        Xs, ys = self._fit_scale(X, y)
        n = len(ys)
        nv = max(1, int(0.1 * n))
        Xt, yt = self._seq(Xs[:-nv]), torch.tensor(ys[:-nv], dtype=torch.float32)
        Xv, yv = self._seq(Xs[-nv:]), torch.tensor(ys[-nv:], dtype=torch.float32)
        Cell = nn.GRU if self.cell == "GRU" else nn.LSTM

        class Net(nn.Module):
            def __init__(s, h):
                super().__init__()
                s.rnn = Cell(1, h, batch_first=True)
                s.out = nn.Linear(h, 1)

            def forward(s, x):
                o, _ = s.rnn(x)
                return s.out(o[:, -1, :]).squeeze(-1)

        self.nets = []
        for k in range(self.n_seeds):
            torch.manual_seed(self.seed + k)
            net = Net(self.hidden)
            opt = torch.optim.Adam(net.parameters(), lr=self.lr)
            best, best_state, wait = np.inf, None, 0
            g = torch.Generator().manual_seed(self.seed + k)
            for _ in range(self.epochs):
                net.train()
                perm = torch.randperm(len(yt), generator=g)
                for b in range(0, len(yt), 256):
                    idx = perm[b:b + 256]
                    opt.zero_grad()
                    loss = ((net(Xt[idx]) - yt[idx]) ** 2).mean()
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    v = float(((net(Xv) - yv) ** 2).mean())
                if v < best - 1e-5:
                    best, wait = v, 0
                    best_state = {kk: vv.clone() for kk, vv in net.state_dict().items()}
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
            net.load_state_dict(best_state)
            net.eval()
            self.nets.append(net)
        return self

    def predict(self, X):
        import torch
        Xs = self._seq(self._tx(X))
        with torch.no_grad():
            P = [net(Xs).numpy() for net in self.nets]
        return np.mean(P, axis=0) * self.sy + self.my


def build(name: str, hp: dict, mcfg: dict, seed: int):
    """Construye un modelo con el hiperparámetro `hp` (seleccionado) y la configuración fija."""
    if name == "ELM":
        return ELM(n_hidden=hp["n_hidden"], n_seeds=mcfg["n_seeds"], low=mcfg["weight_low"],
                   high=mcfg["weight_high"], ridge=hp.get("ridge", 0.0), seed=seed)
    if name == "MLP":
        return MLP(n_hidden=hp["n_hidden"], solver=mcfg["solver"], max_iter=mcfg["max_iter"],
                   alpha=mcfg["alpha"], n_seeds=mcfg["n_seeds"], seed=seed)
    if name == "NPAR":
        return NPAR(n_knots=mcfg["n_knots"], degree=mcfg["degree"], alpha=hp["alpha"])
    if name == "RF":
        return RF(n_estimators=mcfg["n_estimators"], max_depth=hp["max_depth"],
                  min_samples_leaf=hp["min_samples_leaf"], seed=seed)
    if name == "XGB":
        return XGB(max_depth=hp["max_depth"], n_estimators=mcfg["n_estimators"],
                   learning_rate=hp["learning_rate"], seed=seed, backend=mcfg.get("backend", "auto"))
    if name in ("GRU", "LSTM"):
        return RNN(cell=name, hidden=hp["hidden"], epochs=mcfg["epochs"], patience=mcfg["patience"],
                   lr=mcfg["lr"], n_seeds=mcfg["n_seeds"], seed=seed)
    raise ValueError(name)


def grid(name: str, mcfg: dict) -> list[dict]:
    if name == "ELM":
        return [{"n_hidden": int(k), "ridge": float(r)} for k in mcfg["hidden_grid"] for r in mcfg.get("ridge_grid", [0.0])]
    if name == "MLP":
        return [{"n_hidden": int(k)} for k in mcfg["hidden_grid"]]
    if name == "NPAR":
        return [{"alpha": float(a)} for a in mcfg["alpha_grid"]]
    if name == "RF":
        return [{"max_depth": (None if d in (None, "None", 0) else int(d)), "min_samples_leaf": int(l)}
                for d in mcfg["max_depth_grid"] for l in mcfg["min_samples_leaf_grid"]]
    if name == "XGB":
        return [{"max_depth": int(d), "learning_rate": float(r)}
                for d in mcfg["max_depth_grid"] for r in mcfg["learning_rate_grid"]]
    if name in ("GRU", "LSTM"):
        return [{"hidden": int(h)} for h in mcfg["hidden_grid"]]
    raise ValueError(name)
