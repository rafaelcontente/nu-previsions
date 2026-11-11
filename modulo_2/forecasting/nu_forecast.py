# -*- coding: utf-8 -*-
"""
=============================================================================
MÓDULO 2: PREVISÃO — NU HOLDINGS (NU)
=============================================================================
(1) Previsão pontual para além da última observação (modelo campeão);
(2) Distribuição via Monte Carlo condicionado a regimes Markov;
(3) Intervalo conformal calibrado em resíduos de ROLLING-ORIGIN.

-----------------------------------------------------------------------------
SELEÇÃO POR ROLLING-ORIGIN (substitui a seleção por 3 pontos de validação)
-----------------------------------------------------------------------------
Com n_val=3 a escolha do campeão era uma moeda ao ar: em duas execuções
consecutivas o mesmo mecanismo escolheu o pior e o melhor modelo no holdout.
Agora a seleção usa backtest de origens expansivas sobre treino+validação
(1 passo à frente, ~8 folds), gerando ~8 erros por modelo em vez de 3.
O TESTE NUNCA participa na seleção.

SHRINKAGE DOS PESOS: w = λ·(inverso-MAE) + (1−λ)·uniforme,
λ = n_folds/(n_folds+10). Com 8 folds λ≈0.44 — a evidência pesa menos de
metade, que é o correto com esta amostra. Evita winner-takes-all ruidoso.

CONFORMAL: q = quantil(|resíduo rolling-origin do campeão|, 1−α).
Mais pontos de calibração que os 3 de validação. Cobertura reportada
separadamente em validação (in-sample) e em teste (out-of-sample).

PREVISÃO FUTURA: lags do EPS recursivos; restantes covariáveis em
persistência no último valor observado (pressuposto declarado).
=============================================================================
"""
from __future__ import annotations

import hashlib
import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", ValueWarning)
warnings.filterwarnings("ignore", message=".*Ill-conditioned matrix.*")
warnings.filterwarnings("ignore", message=".*Singular matrix.*")

log = logging.getLogger("nu_forecast")

MIN_OBS_ARIMA = 8
MIN_N_CALIBRATION = 10
SHRINKAGE_PRIOR_FOLDS = 10      # λ = n_folds/(n_folds+10)
SELECTION_STABILITY_MIN_FOLDS = 5
SELECTION_STABILITY_MARGIN = 0.10
MODELS = ("naive", "sarimax", "ridge")


@dataclass
class ForecastResult:
    ensemble_forecast: np.ndarray
    ensemble_val: np.ndarray
    mae_val: float

    test_forecast: np.ndarray = field(default_factory=lambda: np.array([]))
    mae_test: float = float("nan")
    maes_test: dict = field(default_factory=dict)
    coverage_test: float = float("nan")

    selected_model: str = "ensemble"
    selection_reason: str = ""
    selection_margin: float = 0.0
    selection_stable: bool = False
    future_features_assumption: str = ""

    # rolling-origin
    n_folds: int = 0
    maes_rolling: dict = field(default_factory=dict)
    shrinkage_lambda: float = 0.0
    n_calibration: int = 0

    sarimax_forecast: np.ndarray = field(default_factory=lambda: np.array([]))
    ridge_forecast: np.ndarray = field(default_factory=lambda: np.array([]))
    naive_forecast: np.ndarray = field(default_factory=lambda: np.array([]))
    sarimax_val: np.ndarray = field(default_factory=lambda: np.array([]))
    ridge_val: np.ndarray = field(default_factory=lambda: np.array([]))
    naive_val: np.ndarray = field(default_factory=lambda: np.array([]))
    sarimax_model_name: str = ""

    weights: dict = field(default_factory=dict)
    maes_individuais: dict = field(default_factory=dict)

    mc_paths: np.ndarray = field(default_factory=lambda: np.array([]))
    mc_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    mc_std: np.ndarray = field(default_factory=lambda: np.array([]))
    transition_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    regime_params: dict = field(default_factory=dict)
    initial_regime: int = 1

    q_conformal: float = 0.0
    interval_lower: np.ndarray = field(default_factory=lambda: np.array([]))
    interval_upper: np.ndarray = field(default_factory=lambda: np.array([]))
    coverage_val: float = 0.0

    ridge_feature_names: list = field(default_factory=list)
    ridge_coefficients: np.ndarray = field(default_factory=lambda: np.array([]))

    horizon: int = 0
    alpha: float = 0.10
    seed: int = 42
    sha256: str = ""
    test_results: dict = field(default_factory=dict)

    def compute_hash(self) -> str:
        self.sha256 = hashlib.sha256(self.mc_paths.tobytes()).hexdigest()
        return self.sha256


# =============================================================================
# HELPERS
# =============================================================================
def _finite(values, fallback) -> np.ndarray:
    a = np.asarray(values, dtype=float).ravel()
    fb = np.asarray(fallback, dtype=float).ravel()
    if fb.size == 0 or not np.all(np.isfinite(fb)):
        fb = np.zeros(1)
    if fb.size == 1:
        fb = np.full(a.size, fb[0])
    elif fb.size != a.size:
        fb = np.resize(fb, a.size)
    return np.where(np.isfinite(a), a, fb)


def _exog_train(X: pd.DataFrame, y: pd.Series, cols=("selic",)):
    avail = [c for c in cols if c in X.columns]
    if not avail:
        return None
    ex = X[avail].reindex(y.index).replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    if len(ex) < 3 or float(ex.var().sum()) <= 1e-12:
        return None
    return ex


def _mae(y, p) -> float:
    yv = _finite(y, 0.0)
    pv = _finite(p, float(yv[-1]) if yv.size else 0.0)
    if pv.size != yv.size:
        pv = np.resize(pv, yv.size)
    return float(mean_absolute_error(yv, pv)) if yv.size else float("nan")


# =============================================================================
# CANDIDATOS
# =============================================================================
def forecast_naive(y_train: pd.Series, horizon: int) -> np.ndarray:
    last = float(y_train.iloc[-1]) if len(y_train) else 0.0
    return np.full(max(1, horizon), last if np.isfinite(last) else 0.0)


def forecast_sarimax(y_train: pd.Series, X_train: pd.DataFrame, horizon: int,
                     verbose: bool = True):
    if verbose:
        log.info("[SARIMAX] A ajustar modelo (1,1,0)...")
    y = y_train.fillna(y_train.mean() if len(y_train) else 0.0).astype(float)
    fill = float(y.iloc[-1]) if len(y) else 0.0

    if len(y) < MIN_OBS_ARIMA:
        mean = float(y.mean()) if len(y) else 0.0
        if verbose:
            log.warning("[SARIMAX] Amostra insuficiente (%d < %d) → média histórica.",
                        len(y), MIN_OBS_ARIMA)
        return np.full(max(1, horizon), mean), "Mean_Baseline"

    ex = _exog_train(X_train, y)
    ex_f = pd.DataFrame([ex.iloc[-1].values] * max(1, horizon), columns=ex.columns) \
        if ex is not None else None
    tent = ([("SARIMAX", (1, 1, 0), ex, ex_f)] if ex is not None else []) + \
           [("ARIMA", (1, 1, 0), None, None),
            ("AR1", (1, 0, 0), None, None),
            ("Mean", (0, 0, 0), None, None)]
    for nome, order, exg, exgf in tent:
        try:
            res = SARIMAX(endog=y, exog=exg, order=order, seasonal_order=(0, 0, 0, 0),
                          enforce_stationarity=False, enforce_invertibility=False
                          ).fit(disp=False, maxiter=200)
            fc = _finite(res.get_forecast(steps=max(1, horizon), exog=exgf).predicted_mean.values, fill)
            if verbose:
                log.info("[SARIMAX] ✓ Ajustado como %s (order=%s).", nome, order)
            return fc, nome
        except Exception as e:
            if verbose:
                log.warning("[SARIMAX] Falhou como %s: %s. Fallback...", nome, type(e).__name__)
    return forecast_naive(y_train, horizon), "Naive_Fallback"


def _fit_ridge(X: pd.DataFrame, y: pd.Series, alpha: float = 1.0) -> dict:
    fills = X.mean(numeric_only=True)
    Xtr = X.fillna(fills).fillna(0.0).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    keep = [c for c in Xtr.columns if float(np.nanstd(Xtr[c].values)) > 1e-12]
    yv = y.fillna(y.mean() if len(y) else 0.0).astype(float)
    y_mean = float(yv.mean()) if len(yv) else 0.0
    bundle = {"model": None, "scaler": None, "keep": keep,
              "fills": fills, "y_mean": y_mean}
    if not keep or len(yv) < 2:
        return bundle
    sc = StandardScaler()
    Xs = sc.fit_transform(Xtr[keep])
    for a in (alpha, 10.0, 100.0, 1000.0):
        m = Ridge(alpha=a)
        m.fit(Xs, yv)
        p = m.predict(Xs)
        if np.all(np.isfinite(m.coef_)) and np.all(np.isfinite(p)):
            bundle.update({"model": m, "scaler": sc})
            return bundle
    return bundle


def _apply_ridge(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    if bundle["model"] is None:
        return np.full(len(X), bundle["y_mean"])
    Xf = X.fillna(bundle["fills"]).fillna(0.0).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    for c in bundle["keep"]:
        if c not in Xf.columns:
            Xf[c] = 0.0
    p = bundle["model"].predict(bundle["scaler"].transform(Xf[bundle["keep"]]))
    return _finite(p, bundle["y_mean"])


def forecast_ridge(X_train, y_train, X_future, alpha: float = 1.0, verbose: bool = True):
    if verbose:
        log.info("[Ridge] A ajustar modelo com alpha=%s", alpha)
    b = _fit_ridge(X_train, y_train, alpha)
    return _apply_ridge(b, X_future), b["model"], b["keep"]


def forecast_ridge_recursive(X_fit, y_fit, X_last: pd.DataFrame,
                             y_hist: pd.Series, horizon: int, alpha: float = 1.0) -> np.ndarray:
    b = _fit_ridge(X_fit, y_fit, alpha)
    if b["model"] is None or X_last.empty:
        return np.full(max(1, horizon), b["y_mean"])
    last_row = X_last.iloc[-1].copy()
    hist = [float(v) for v in y_hist.values if np.isfinite(v)]
    preds = []
    for _ in range(max(1, horizon)):
        row = last_row.copy()
        for k in (1, 2, 3, 4):
            nm = f"eps_lag{k}"
            if nm in row.index and len(hist) >= k:
                row[nm] = hist[-k]
        p = float(_apply_ridge(b, pd.DataFrame([row]))[0])
        preds.append(p)
        hist.append(p)
    return np.asarray(preds, dtype=float)


# =============================================================================
# ROLLING-ORIGIN + SELEÇÃO COM SHRINKAGE
# =============================================================================
def rolling_origin_errors(X: pd.DataFrame, y: pd.Series,
                          min_train: int = MIN_OBS_ARIMA, max_folds: int = 12) -> tuple:
    """
    Backtest de origens expansivas, 1 passo à frente. Usa APENAS os dados
    fornecidos (treino+validação). O teste nunca entra aqui.
    Devolve (erros_por_modelo, previsões_por_modelo_alinhadas_ao_índice).
    """
    n = len(y)
    origins = list(range(min_train, n))
    if not origins:
        return {m: [] for m in MODELS}, {}
    if max_folds and len(origins) > max_folds:
        origins = origins[-max_folds:]

    log.info("[Rolling-origin] %d folds (origens %d..%d, mínimo treino=%d)",
             len(origins), origins[0], origins[-1], min_train)
    errs = {m: [] for m in MODELS}
    for k in origins:
        Xtr, ytr = X.iloc[:k], y.iloc[:k]
        Xte = X.iloc[k:k + 1]
        actual = float(y.iloc[k])
        preds = {
            "naive": float(forecast_naive(ytr, 1)[0]),
            "sarimax": float(forecast_sarimax(ytr, Xtr, 1, verbose=False)[0][0]),
            "ridge": float(forecast_ridge(Xtr, ytr, Xte, verbose=False)[0][0]),
        }
        for m, p in preds.items():
            errs[m].append(abs(actual - p) if np.isfinite(p) else np.nan)
    for m in MODELS:
        e = np.asarray(errs[m], dtype=float)
        e = e[np.isfinite(e)]
        log.info("[Rolling-origin] %-8s MAE=%.4f (n=%d)", m,
                 float(e.mean()) if e.size else float("nan"), e.size)
    return errs, {"origins": origins}


def select_with_shrinkage(maes: dict, n_folds: int) -> tuple:
    """
    Pesos = λ·(inverso-MAE) + (1−λ)·uniforme, λ = n_folds/(n_folds+PRIOR).
    Campeão: melhor individual se bater o ensemble com margem; senão ensemble.
    Estabilidade avaliada por n_folds, não por n_val.
    """
    valid = {k: v for k, v in maes.items() if np.isfinite(v) and v > 0}
    if not valid:
        w = {m: 1.0 / len(MODELS) for m in MODELS}
        return "ensemble", w, "Sem erros válidos → pesos uniformes.", 0.0, False, 0.0

    lam = n_folds / (n_folds + SHRINKAGE_PRIOR_FOLDS) if n_folds > 0 else 0.0
    inv = {m: 1.0 / max(valid[m], 1e-6) for m in valid}
    s = sum(inv.values())
    w_inv = {m: inv[m] / s for m in valid}
    uni = 1.0 / len(valid)
    w = {m: lam * w_inv.get(m, 0.0) + (1 - lam) * uni for m in valid}
    for m in MODELS:
        w.setdefault(m, 0.0)
    tot = sum(w.values())
    w = {m: v / tot for m, v in w.items()} if tot > 0 else {m: uni for m in valid}

    log.info("[Seleção] λ=%.2f (n_folds=%d) | pesos: %s", lam, n_folds,
             " | ".join(f"{m}: {w.get(m, 0):.3f}" for m in MODELS))

    best = min(valid, key=valid.get)
    best_mae = valid[best]
    ens_bound = sum(w[m] * valid[m] for m in valid)   # limite superior do MAE do ensemble
    margin = (ens_bound - best_mae) / best_mae if best_mae > 0 else 0.0
    stable = (n_folds >= SELECTION_STABILITY_MIN_FOLDS
              and margin >= SELECTION_STABILITY_MARGIN)

    if margin <= 1e-9:
        reason = (f"Ensemble (limite {ens_bound:.4f}) não é pior que o melhor candidato "
                  f"'{best}' ({best_mae:.4f}). Mantido ensemble com shrinkage λ={lam:.2f}.")
        log.info("[Seleção] %s", reason)
        return "ensemble", w, reason, 0.0, stable, lam

    reason = (f"Melhor candidato '{best}' (MAE rolling={best_mae:.4f}) vs ensemble "
              f"(limite {ens_bound:.4f}), margem {margin:.1%} sobre {n_folds} folds. ")
    if stable:
        reason += f"Campeão: '{best}' (margem e nº de folds suficientes)."
        log.info("[Seleção] %s", reason)
        w = {m: (1.0 if m == best else 0.0) for m in MODELS}
        return best, w, reason, float(margin), True, lam
    reason += (f"Margem/folds INSUFICIENTES (n_folds={n_folds}) → mantém-se ensemble "
               f"com shrinkage em vez de winner-takes-all ruidoso.")
    log.warning("[Seleção] %s", reason)
    return "ensemble", w, reason, float(margin), False, lam


def _combine(preds: dict, weights: dict, fallback: float) -> np.ndarray:
    return _finite(sum(weights.get(k, 0.0) * _finite(preds[k], fallback) for k in preds), fallback)


# =============================================================================
# MARKOV + MONTE CARLO
# =============================================================================
def estimate_markov_regimes(X: pd.DataFrame, prior: float = 1.0):
    log.info("[Markov] A estimar regimes (growth se rev_yoy > 15%%)...")
    rev = X["rev_yoy"].fillna(0.0) if "rev_yoy" in X.columns else pd.Series(0.0, index=X.index)
    reg = np.where(rev.values > 0.15, 0, 1)
    c = np.zeros((2, 2))
    for t in range(len(reg) - 1):
        c[reg[t], reg[t + 1]] += 1
    sm = c + prior
    tm = sm / sm.sum(axis=1, keepdims=True)
    log.info("[Markov] Matriz de transição:\n%s", tm.round(3))
    return reg, tm


def estimate_regime_params(y: pd.Series, regimes: np.ndarray) -> dict:
    params = {}
    for r in (0, 1):
        v = y.values[regimes == r]
        v = v[np.isfinite(v)] if len(v) else v
        mu = float(np.mean(v)) if len(v) else float(y.mean())
        sg = float(np.std(v, ddof=1)) if len(v) > 1 else \
             (float(y.std(ddof=1)) if len(y) > 1 else 0.01)
        params[r] = (mu, max(sg, 1e-3))
    return params


def monte_carlo_simulation(horizon, tm, regime_params, initial_regime,
                           n_paths: int = 10_000, seed: int = 42) -> np.ndarray:
    log.info("[Monte Carlo] A simular %d caminhos (seed=%d)...", n_paths, seed)
    rng = np.random.default_rng(seed)
    mus = _finite([regime_params[0][0], regime_params[1][0]], 0.0)
    sig = _finite([regime_params[0][1], regime_params[1][1]], 1e-3)
    sig = np.where(sig > 0, sig, 1e-3)
    cur = np.full(n_paths, int(initial_regime), dtype=int)
    paths = np.zeros((n_paths, horizon))
    for t in range(horizon):
        cur = (rng.random(n_paths) >= tm[cur, 0]).astype(int)
        paths[:, t] = rng.normal(mus[cur], sig[cur])
    return np.nan_to_num(paths, nan=0.0)


# =============================================================================
# CONFORMAL
# =============================================================================
def conformal_from_residuals(residuals, forecast, alpha: float = 0.10) -> tuple:
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    q = float(np.quantile(r, 1 - alpha)) if r.size else 0.0
    q = q if np.isfinite(q) else 0.0
    ef = _finite(forecast, 0.0)
    log.info("[Conformal] n_cal=%d | q=%.4f | intervalo T+1: [%.4f, %.4f]",
             r.size, q, ef[0] - q, ef[0] + q)
    return q, ef - q, ef + q, int(r.size)


def coverage_of(y, pred, q) -> float:
    yv, pv = _finite(y, 0.0), _finite(pred, 0.0)
    if pv.size != yv.size:
        pv = np.resize(pv, yv.size)
    return float((np.abs(yv - pv) <= q).mean()) if yv.size else float("nan")


# =============================================================================
# TESTES
# =============================================================================
def test_champion_beats_baselines(maes, mae_selected, n_obs, min_samples=3):
    if n_obs < min_samples:
        msg = f"SKIP (n={n_obs} < {min_samples}): amostra insuficiente."
        log.warning("[Teste 1] %s", msg)
        return True, msg
    best = min(v for v in maes.values() if np.isfinite(v))
    ok = mae_selected <= best + 1e-9
    msg = f"{'PASS' if ok else 'FAIL'} — campeão={mae_selected:.4f} vs melhor={best:.4f} (n={n_obs})"
    log.info("[Teste 1] %s", msg)
    return ok, msg


def test_coverage(y, pred, q, label, min_cov=2 / 3):
    cov = coverage_of(y, pred, q)
    ok = np.isfinite(cov) and cov >= min_cov
    msg = f"{'PASS' if ok else 'FAIL'} — cobertura {label}={cov:.1%} (mínimo {min_cov:.1%})"
    log.info("[Teste 2] %s", msg)
    return ok, cov, msg


def test_reproducibility(horizon, tm, rp, ir, n_paths, seed, ref_hash):
    h2 = hashlib.sha256(
        monte_carlo_simulation(horizon, tm, rp, ir, n_paths, seed).tobytes()).hexdigest()
    ok = h2 == ref_hash
    msg = f"{'PASS' if ok else 'FAIL'} — SHA-256 reproduzido: {h2[:16]}..."
    log.info("[Teste 3] %s", msg)
    return ok, msg


# =============================================================================
# PIPELINE
# =============================================================================
def run_forecast(X_train, y_train, X_val, y_val, X_test, y_test,
                 horizon=4, alpha=0.10, seed=42, n_paths=10_000) -> ForecastResult:
    log.info("=" * 70)
    log.info("PIPELINE DE PREVISÃO — horizon=%d, alpha=%.2f, seed=%d", horizon, alpha, seed)
    log.info("=" * 70)

    X_tv, y_tv = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
    X_all, y_all = pd.concat([X_tv, X_test]), pd.concat([y_tv, y_test])
    fb = float(y_all.iloc[-1]) if len(y_all) else 0.0

    # --- FASE 1: validação (ajuste só em treino) ---------------------------
    log.info("--- FASE 1: Previsões em validação (ajuste em treino) ---")
    hv = max(1, len(y_val))
    val_preds = {
        "naive": forecast_naive(y_train, hv),
        "sarimax": forecast_sarimax(y_train, X_train, hv)[0],
        "ridge": forecast_ridge(X_train, y_train, X_val)[0],
    }
    sname = forecast_sarimax(y_train, X_train, hv, verbose=False)[1]
    maes_val = {m: _mae(y_val, p) for m, p in val_preds.items()}

    # --- FASE 2: rolling-origin + seleção com shrinkage --------------------
    log.info("--- FASE 2: Seleção por rolling-origin (treino+val; teste intocado) ---")
    ro_errs, ro_meta = rolling_origin_errors(X_tv, y_tv)
    n_folds = max((len(v) for v in ro_errs.values()), default=0)
    maes_ro = {m: (float(np.nanmean(v)) if len(v) else float("nan"))
               for m, v in ro_errs.items()}
    selected, weights, reason, margin, stable, lam = select_with_shrinkage(maes_ro, n_folds)
    ev = _combine(val_preds, weights, fb)
    mae_val_sel = _mae(y_val, ev)

    # --- FASE 3: TESTE (holdout verdadeiro) --------------------------------
    log.info("--- FASE 3: Avaliação em TESTE (holdout, ajuste em treino+validação) ---")
    ht = max(1, len(y_test))
    if len(y_test) == 0:
        log.warning("[Teste] Conjunto de teste vazio — não avaliável.")
        test_preds = {m: np.array([]) for m in MODELS}
        maes_test, mae_test_sel, tp_sel = {}, float("nan"), np.array([])
    else:
        test_preds = {
            "naive": forecast_naive(y_tv, ht),
            "sarimax": forecast_sarimax(y_tv, X_tv, ht)[0],
            "ridge": forecast_ridge(X_tv, y_tv, X_test)[0],
        }
        maes_test = {m: _mae(y_test, p) for m, p in test_preds.items()}
        tp_sel = _combine(test_preds, weights, fb)
        mae_test_sel = _mae(y_test, tp_sel)
        log.info("[Teste] MAE teste → %s || Campeão (%s): %.4f",
                 " | ".join(f"{m}: {v:.4f}" for m, v in maes_test.items()),
                 selected, mae_test_sel)
        if np.isfinite(mae_test_sel) and mae_test_sel > mae_val_sel:
            log.warning("[Teste] MAE teste (%.4f) PIOR que validação (%.4f) — "
                        "degradação fora da amostra.", mae_test_sel, mae_val_sel)

    # --- FASE 4: previsão futura -------------------------------------------
    log.info("--- FASE 4: Previsão futura (lags recursivos + persistência) ---")
    assumption = ("Lags do EPS atualizados recursivamente com as próprias previsões; "
                  "restantes covariáveis mantidas no último valor observado (persistência). "
                  "Não existem features observadas para o horizonte futuro.")
    fut_preds = {
        "naive": forecast_naive(y_all, horizon),
        "sarimax": forecast_sarimax(y_all, X_all, horizon)[0],
        "ridge": forecast_ridge_recursive(X_all, y_all, X_all, y_all, horizon),
    }
    ef = _combine(fut_preds, weights, fb)
    log.info("[Futuro] Campeão '%s': %s", selected, ef.round(4))

    # --- FASE 5: Markov + Monte Carlo --------------------------------------
    log.info("--- FASE 5: Monte Carlo condicionado a regimes ---")
    reg, tm = estimate_markov_regimes(X_all)
    rp = estimate_regime_params(y_all, reg)
    ir = int(reg[-1]) if len(reg) else 1
    mc = monte_carlo_simulation(horizon, tm, rp, ir, n_paths, seed)

    # --- FASE 6: conformal (calibra nos resíduos de rolling-origin) --------
    log.info("--- FASE 6: Conformal prediction ---")
    cal_res = ro_errs.get(selected) if selected in ro_errs else None
    if not cal_res or len(cal_res) == 0:
        # ensemble: resíduo = |y - combinação| em cada origem não está guardado;
        # usa-se a combinação ponderada dos erros absolutos como aproximação
        # conservadora (limite superior do erro do ensemble).
        cal_res = [sum(weights.get(m, 0.0) * ro_errs[m][i] for m in MODELS
                       if i < len(ro_errs[m]))
                   for i in range(n_folds)]
        log.info("[Conformal] Calibração no limite superior do erro do ensemble "
                 "(conservador).")
    q, lo, up, n_cal = conformal_from_residuals(cal_res, ef, alpha)
    cov_val = coverage_of(y_val, ev, q)
    cov_test = coverage_of(y_test, tp_sel, q) if len(y_test) else float("nan")
    if np.isfinite(cov_test):
        log.info("[Conformal] Cobertura OUT-OF-SAMPLE em teste: %.1f%% (n=%d)",
                 cov_test * 100, len(y_test))
    if n_cal < MIN_N_CALIBRATION:
        log.warning("[Conformal] n_cal=%d < %d → calibração NÃO avaliável; "
                    "intervalo é NOMINAL.", n_cal, MIN_N_CALIBRATION)

    _, ridge_model, ridge_keep = forecast_ridge(X_all, y_all, X_all.iloc[-1:], verbose=False)

    result = ForecastResult(
        ensemble_forecast=ef, ensemble_val=ev, mae_val=mae_val_sel,
        test_forecast=tp_sel, mae_test=mae_test_sel, maes_test=maes_test,
        coverage_test=cov_test,
        selected_model=selected, selection_reason=reason,
        selection_margin=margin, selection_stable=stable,
        future_features_assumption=assumption,
        n_folds=n_folds, maes_rolling=maes_ro, shrinkage_lambda=lam, n_calibration=n_cal,
        sarimax_forecast=fut_preds["sarimax"], ridge_forecast=fut_preds["ridge"],
        naive_forecast=fut_preds["naive"],
        sarimax_val=val_preds["sarimax"], ridge_val=val_preds["ridge"],
        naive_val=val_preds["naive"], sarimax_model_name=sname,
        weights=weights, maes_individuais=maes_val,
        mc_paths=mc, mc_mean=mc.mean(axis=0), mc_std=mc.std(axis=0),
        transition_matrix=tm, regime_params=rp, initial_regime=ir,
        q_conformal=q, interval_lower=lo, interval_upper=up, coverage_val=cov_val,
        ridge_feature_names=ridge_keep,
        ridge_coefficients=np.asarray(getattr(ridge_model, "coef_", np.array([]))),
        horizon=horizon, alpha=alpha, seed=seed,
    )
    result.compute_hash()

    log.info("--- TESTES DE VALIDAÇÃO ---")
    t1, m1 = test_champion_beats_baselines(maes_ro, maes_ro.get(selected, mae_val_sel)
                                          if selected in maes_ro else mae_val_sel, n_folds)
    t2, _, m2 = test_coverage(y_val, ev, q, "val (in-sample)")
    t3, m3 = test_reproducibility(horizon, tm, rp, ir, n_paths, seed, result.sha256)
    result.test_results = {"teste1": (t1, m1), "teste2": (t2, m2), "teste3": (t3, m3)}
    if np.isfinite(cov_test):
        t4, _, m4 = test_coverage(y_test, tp_sel, q, "teste (out-of-sample)")
        result.test_results["teste4"] = (t4, m4)

    log.info("=" * 70)
    log.info("✓ Módulo 2 concluído | campeão: %s | folds=%d λ=%.2f | "
             "MAE val: %.4f | MAE teste: %.4f",
             selected, n_folds, lam, mae_val_sel, mae_test_sel)
    log.info("=" * 70)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")
    rng = np.random.default_rng(7)
    idx = pd.date_range("2021-03-31", periods=20, freq="QE")
    eps = pd.Series(np.linspace(0.02, 0.24, 20) + rng.normal(0, 0.006, 20), index=idx)
    X = pd.DataFrame({"eps_lag1": eps.shift(1), "eps_lag2": eps.shift(2),
                      "rev_yoy": np.linspace(0.10, 0.25, 20),
                      "selic": np.linspace(0.10, 0.13, 20)}, index=idx).ffill().fillna(0.0)
    r = run_forecast(X.iloc[:14], eps.iloc[:14], X.iloc[14:17], eps.iloc[14:17],
                     X.iloc[17:], eps.iloc[17:], horizon=4)
    print("\n📊 Futuro:", r.ensemble_forecast.round(4))
    print(f"🏆 {r.selected_model} | folds={r.n_folds} λ={r.shrinkage_lambda:.2f} "
          f"| MAE val {r.mae_val:.4f} | MAE teste {r.mae_test:.4f}")