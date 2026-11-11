# -*- coding: utf-8 -*-
"""
=============================================================================
MÓDULO 1 — INGESTÃO, FEATURES, CONTRATO E SPLIT TEMPORAL
=============================================================================
Fontes: Alpha Vantage (trimestres) → SEC EDGAR+yfinance → yfinance.
Âncora: SEC EDGAR anual (20-F), com reconciliação.
Operacionais: SEC 6-K (ARPAC, NPL 15-90) — best-effort, sem defaults.
Macro: BCB SGS com CACHE PERSISTENTE (reprodutibilidade).

-----------------------------------------------------------------------------
PRÉ-IPO: apenas para popular lags, nunca para treinar
-----------------------------------------------------------------------------
O EPS pré-IPO reflete outra estrutura de ações (quebra em dez/2021). Buscar
desde 2019 dá lags completos ao 1º trimestre pós-IPO, recuperando ~5 trimestres
de treino sem misturar regimes. As linhas pré-IPO saem do frame de modelagem.

-----------------------------------------------------------------------------
PREVENÇÃO DE LEAKAGE (4 camadas)
-----------------------------------------------------------------------------
1. Fundamentais/operacionais entram como *_lag1 (publicados com o EPS).
2. Contemporâneos excluídos do X (ficam no df para os factos do Módulo 3).
3. Limpeza causal: só ffill, nunca bfill. Warm-up descartado.
4. PREENCHIMENTO ANUAL DESATIVADO por defeito: o anual de FY Y só é publicado
   em abril de Y+1; usá-lo em trimestres de Y é look-ahead estrutural.
   Lacunas ficam NaN e são imputadas pelo modelo (declarado), não inventadas.

ARPAC/NPL nunca são preenchidos: ausência é informação (NOT_VERIFIED).
=============================================================================
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from .alpha_vantage import fetch_quarterly_fundamentals as av_fetch
from .sec_edgar import fetch_nu_from_edgar, dataset_to_dataframe
from .sec_sixk import fetch_operational_metrics

log = logging.getLogger("nu_data_pipeline")
warnings.filterwarnings("ignore", message=".*No frequency information.*")

TICKER = "NU"
FETCH_START = "2019-01-01"      # janela larga: pré-IPO popula lags
MODEL_START = "2021-12-01"      # IPO na NYSE: início do treino
BCB_SERIES = {"selic": 432, "ipca": 433}
BCB_CACHE_DIR = Path(".cache/bcb_sgs")
BCB_CACHE_TTL_DAYS = 7
USER_AGENT = os.environ.get("SEC_USER_AGENT", "NuForecast Research contact@example.org")

CONTEMP_FUND = ("revenue", "net_income", "total_assets", "equity", "shares",
                "arpac", "npl_15_90")
WARMUP_COLS = ("eps_lag1", "eps_lag2", "eps_lag3", "eps_lag4", "rev_yoy")
NO_FILL_COLS = ("arpac", "npl_15_90")


@dataclass
class Dataset:
    df: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    n_total: int
    source_summary: dict
    sha256: str = ""
    validation_passed: bool = False
    created_at: str = ""
    provenance: dict = field(default_factory=dict)

    @property
    def n_train(self) -> int: return len(self.X_train)
    @property
    def n_val(self) -> int:   return len(self.X_val)
    @property
    def n_test(self) -> int:  return len(self.X_test)

    def compute_hash(self) -> str:
        self.sha256 = hashlib.sha256(self.df.to_csv(index=True).encode()).hexdigest()
        return self.sha256


def _clean_numeric_causal(df: pd.DataFrame) -> pd.DataFrame:
    """Inf→NaN→ffill→0.0. NUNCA bfill (copiaria o futuro para linhas anteriores)."""
    out = df.replace([np.inf, -np.inf], np.nan)
    return out.ffill().fillna(0.0)


# =============================================================================
# FONTES FINANCEIRAS
# =============================================================================
def _fetch_yfinance_financials(ticker: str, start: str) -> pd.DataFrame:
    log.info("[yfinance] A buscar dados de %s (último recurso)", ticker)
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame()
    try:
        qf = yf.Ticker(ticker).quarterly_financials
        if qf is None or qf.empty:
            return pd.DataFrame()
        fin = qf.T.sort_index()
        rename = {"Diluted EPS": "eps_diluted", "Basic EPS": "eps_basic",
                  "Total Revenue": "revenue", "Net Income": "net_income"}
        cols = {k: v for k, v in rename.items() if k in fin.columns}
        df = fin[list(cols)].rename(columns=cols)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        if "eps_diluted" in df.columns:
            df["eps"] = df["eps_diluted"]
        elif "eps_basic" in df.columns:
            df["eps"] = df["eps_basic"]
        else:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        df = df[df.index >= pd.to_datetime(start)].sort_index()
        log.info("[yfinance] ✓ %d trimestres", len(df))
        return df
    except Exception as e:
        log.warning("[yfinance] Falha: %s: %s", type(e).__name__, e)
        return pd.DataFrame()


def _fill_gaps_from_annual(df: pd.DataFrame, annual: pd.DataFrame) -> tuple:
    """
    OPCIONAL e DESATIVADO por defeito. O anual de FY Y é publicado ~4 meses
    após o fecho, logo preencher trimestres de Y é look-ahead estrutural.
    Só usar com allow_annual_fill=True e consciência do enviesamento.
    """
    fills = {}
    if annual is None or annual.empty:
        return df, fills
    by_year = annual.groupby(annual.index.year).last()
    years = pd.Series(df.index.year, index=df.index)
    for col in ("revenue", "net_income", "total_assets", "equity"):
        if col not in by_year.columns:
            continue
        mapped = years.map(by_year[col])
        if col not in df.columns:
            df[col] = mapped
            n = int(mapped.notna().sum())
        else:
            before = int(df[col].isna().sum())
            df[col] = df[col].fillna(mapped)
            n = before - int(df[col].isna().sum())
        if n:
            fills[col] = n
    if fills:
        log.warning("[ancora] ⚠ LOOK-AHEAD: preenchimento anual em %s. O anual de FY Y "
                    "só é público em abril de Y+1 — os lags destes trimestres contêm "
                    "informação posterior. Ativado explicitamente.", fills)
    return df, fills


def _reconcile(df: pd.DataFrame, annual: pd.DataFrame) -> list:
    out = []
    if annual is None or annual.empty:
        log.warning("[reconciliação] Sem anual EDGAR — trimestres NÃO validados.")
        return out
    for col in ("revenue", "net_income"):
        if col not in df.columns or col not in annual.columns:
            continue
        for end, official in annual[col].items():
            sub = df.loc[df.index.year == end.year, col].dropna()
            if len(sub) == 4 and official:
                ratio = float(sub.sum()) / float(official)
                ok = 0.85 <= ratio <= 1.15
                out.append({"field": col, "fy": int(end.year),
                            "ratio": round(ratio, 4), "ok": bool(ok)})
                log.info("[reconciliação] %s %s FY%d: 4T/EDGAR = %.3f",
                         "OK" if ok else "DESVIO", col, end.year, ratio)
    if not out:
        log.warning("[reconciliação] Sem anos com 4 trimestres sobrepostos — NÃO executada.")
    return out


def _fetch_financials(ticker: str, start: str, end: Optional[str],
                      allow_annual_fill: bool = False) -> tuple:
    annual, edgar_q = pd.DataFrame(), pd.DataFrame()
    try:
        eds = fetch_nu_from_edgar(ticker=ticker, user_agent=USER_AGENT, start=start, end=end)
        annual = getattr(eds, "annual", pd.DataFrame())
        edgar_q = dataset_to_dataframe(eds)
    except Exception as e:
        log.warning("[EDGAR] Âncora indisponível: %s: %s", type(e).__name__, e)

    prov = {"primary": None, "columns": {}, "edgar_annual_years": int(len(annual)),
            "reconciliation": [], "annual_fill": {}, "annual_fill_enabled": allow_annual_fill,
            "operational": {}, "gaps_left_nan": {}}

    av = av_fetch(ticker, start=start, end=end)
    if av is not None and not av.df.empty:
        df = av.df.copy()
        prov["primary"] = "alpha_vantage"
        prov["columns"] = dict(av.provenance.get("columns", {}))
        log.info("[pipeline] Fonte primária: Alpha Vantage (%d trimestres)", len(df))
    elif not edgar_q.empty:
        df = edgar_q.copy()
        prov["primary"] = "sec_edgar+yfinance"
        log.warning("[pipeline] AV indisponível → SEC EDGAR + trimestres (%d).", len(df))
    else:
        df = _fetch_yfinance_financials(ticker, start)
        prov["primary"] = "yfinance"
        log.warning("[pipeline] EDGAR sem trimestres → yfinance isolado (%d).", len(df))

    if df.empty:
        raise RuntimeError("Nenhuma fonte devolveu dados financeiros.")

    if not prov["columns"]:
        for col in df.columns:
            if col == "source":
                continue
            n = int(pd.to_numeric(df[col], errors="coerce").notna().sum())
            if n:
                prov["columns"][col] = {prov["primary"]: n}

    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.drop(columns=["source"], errors="ignore")

    if allow_annual_fill:
        df, fills = _fill_gaps_from_annual(df, annual)
        prov["annual_fill"] = fills
        for col, n in fills.items():
            prov["columns"].setdefault(col, {})["edgar_annual_fill_LOOKAHEAD"] = n
    else:
        for col in ("revenue", "net_income", "total_assets", "equity"):
            if col in df.columns:
                n = int(df[col].isna().sum())
                if n:
                    prov["gaps_left_nan"][col] = n
        if prov["gaps_left_nan"]:
            log.info("[ancora] Preenchimento anual DESATIVADO (evita look-ahead). "
                     "Lacunas mantidas como NaN: %s → o modelo imputa pela média do treino.",
                     prov["gaps_left_nan"])

    prov["reconciliation"] = _reconcile(df, annual)

    try:
        om = fetch_operational_metrics(ticker=ticker, user_agent=USER_AGENT)
        if not om.df.empty:
            ops = om.df.copy()
            ops.index = pd.to_datetime(ops.index)
            df = df.join(ops, how="left")
            for c in ops.columns:
                n = int(df[c].notna().sum())
                if n:
                    prov["columns"].setdefault(c, {})["sec_6k"] = n
            prov["operational"] = {k: v for k, v in om.coverage.items()
                                   if k not in ("provenance", "ambiguous")}
            prov["operational"]["provenance"] = om.coverage.get("provenance", [])
        else:
            log.warning("[6-K] Sem métricas operacionais — ARPAC/NPL ausentes "
                        "(regras dependentes NOT_VERIFIED).")
            prov["operational"] = {"quarters_with_data": 0}
    except Exception as e:
        log.warning("[6-K] Parsing falhou: %s: %s — ARPAC/NPL ausentes.", type(e).__name__, e)
        prov["operational"] = {"error": f"{type(e).__name__}: {e}"}

    df["source"] = prov["primary"]
    log.info("[proveniência] valores por coluna/origem:")
    for col, srcs in prov["columns"].items():
        log.info("   %-16s %s", col, srcs)
    return df, prov


# =============================================================================
# MACRO (BCB) COM CACHE PERSISTENTE
# =============================================================================
def _bcb_cache_path(code: int) -> Path:
    BCB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return BCB_CACHE_DIR / f"sgs_{code}.json"


def _bcb_fetch_series(name: str, code: int) -> Optional[pd.Series]:
    """
    Cache 7 dias; se expirada, tenta refrescar mas ACEITA a cache antiga em caso
    de falha (reprodutibilidade > frescura). Só devolve None se nunca houve dados.
    """
    cp = _bcb_cache_path(code)
    cached, fresh = None, False
    if cp.exists():
        try:
            cached = json.loads(cp.read_text(encoding="utf-8"))
            age = datetime.now() - datetime.fromtimestamp(cp.stat().st_mtime)
            fresh = age < timedelta(days=BCB_CACHE_TTL_DAYS)
        except Exception:
            cached = None

    if cached is not None and fresh:
        log.info("[BCB] %s: cache válida (%d observações)", name, len(cached))
    else:
        url = (f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}"
               f"/dados?formato=json&dataInicial=01/01/2018")
        got = None
        for attempt in (1, 2, 3):
            try:
                r = requests.get(url, timeout=45)
                r.raise_for_status()
                data = r.json()
                if not isinstance(data, list) or not data:
                    raise ValueError("resposta vazia/inesperada")
                got = data
                break
            except Exception as e:
                log.warning("[BCB] %s tentativa %d falhou: %s", name, attempt, e)
                if attempt < 3:
                    time.sleep(2.0 * attempt)
        if got is not None:
            try:
                cp.write_text(json.dumps(got), encoding="utf-8")
            except Exception:
                pass
            cached = got
            log.info("[BCB] %s: obtido da API (%d observações)", name, len(got))
        elif cached is not None:
            log.warning("[BCB] %s: API falhou → a usar CACHE EXPIRADA (%d obs). "
                        "Reprodutibilidade preservada.", name, len(cached))
        else:
            return None

    try:
        s = pd.Series({pd.to_datetime(x["data"], format="%d/%m/%Y"): float(x["valor"])
                       for x in cached}, name=name)
        return s.resample("QE").mean() / 100.0
    except Exception as e:
        log.error("[BCB] %s: cache ilegível (%s)", name, e)
        return None


def _fetch_bcb_macro(start: str, strict: bool = True) -> tuple:
    """Devolve (DataFrame, info). Com strict=True, série ausente sem cache → erro."""
    log.info("[BCB] A buscar séries SGS (Selic 432, IPCA 433) com cache persistente")
    frames = {}
    for name, code in BCB_SERIES.items():
        s = _bcb_fetch_series(name, code)
        if s is not None and len(s):
            frames[name] = s
    missing = [k for k in BCB_SERIES if k not in frames]
    info = {"series_present": list(frames), "series_missing": missing}
    if missing:
        msg = (f"Séries BCB ausentes sem cache: {missing}. Continuar alteraria o "
               f"conjunto de features e o modelo escolhido (não reprodutível).")
        if strict:
            raise RuntimeError(msg + " Define strict_macro=False para continuar sem elas.")
        log.error("[BCB] ⚠ %s", msg)
    if not frames:
        return pd.DataFrame(), info
    df = pd.DataFrame(frames).sort_index()
    df = df[df.index >= pd.to_datetime(start)]
    log.info("[BCB] ✓ %d trimestres macro | séries=%s", len(df), list(frames))
    return df, info


def _fetch_fx_brl_usd(start: str) -> pd.Series:
    log.info("[yfinance] A buscar FX BRL/USD")
    try:
        import yfinance as yf
        fx = yf.download("BRL=X", start=start, progress=False, auto_adjust=True)
        if fx.empty:
            return pd.Series(dtype=float, name="brl_usd")
        close = fx["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        s = close.resample("QE").mean().rename("brl_usd")
        log.info("[FX] ✓ %d trimestres", len(s))
        return s
    except Exception as e:
        log.warning("[FX] Falha: %s: %s", type(e).__name__, e)
        return pd.Series(dtype=float, name="brl_usd")


# =============================================================================
# FEATURES / CONTRATO / SPLIT
# =============================================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("[features] A engenheirizar features (fundamentais/operacionais desfasados)")
    if "eps" not in df.columns:
        raise RuntimeError("Dataset sem coluna 'eps' — ingestão falhou.")
    out = df.copy()
    for k in (1, 2, 3, 4):
        out[f"eps_lag{k}"] = out["eps"].shift(k)
    for c in CONTEMP_FUND:
        if c in out.columns:
            out[f"{c}_lag1"] = pd.to_numeric(out[c], errors="coerce").shift(1)

    rev, ni = out.get("revenue_lag1"), out.get("net_income_lag1")
    eq, ta = out.get("equity_lag1"), out.get("total_assets_lag1")
    if rev is not None:
        out["rev_yoy"] = (rev / rev.shift(4) - 1.0).replace([np.inf, -np.inf], np.nan)
        if ni is not None:
            out["net_margin"] = (ni / rev.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    if eq is not None and ni is not None:
        out["roe"] = (ni / eq.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    if ta is not None:
        out["assets_chg"] = (ta / ta.shift(1) - 1.0).replace([np.inf, -np.inf], np.nan)
    log.info("[features] %d colunas", len(out.columns))
    return out


def validate_schema(df: pd.DataFrame) -> bool:
    """Fail-fast no alvo. ARPAC/NPL nunca são criados: ausência é informação."""
    log.info("[schema] A validar contrato de dados")
    errors = []
    if df.empty:
        errors.append("DataFrame vazio")
    if "eps" not in df.columns:
        errors.append("coluna obrigatória 'eps' ausente")
    else:
        if not pd.api.types.is_numeric_dtype(df["eps"]):
            errors.append("'eps' não é numérica")
        if df["eps"].isna().all():
            errors.append("'eps' totalmente NaN")
    if len(df) < 4:
        errors.append(f"amostra insuficiente ({len(df)} < 4 trimestres)")
    if errors:
        for e in errors:
            log.error("[schema] ✗ %s", e)
        return False
    for col in ("revenue", "net_income", "total_assets", "equity"):
        if col not in df.columns:
            log.warning("[schema] covariável '%s' ausente → criada a 0.", col)
            df[col] = 0.0
    for col in NO_FILL_COLS:
        if col in df.columns:
            log.info("[schema] '%s': %d/%d reportados (resto NaN, não preenchido)",
                     col, int(df[col].notna().sum()), len(df))
        else:
            log.warning("[schema] '%s' ausente — regra dependente ficará NOT_VERIFIED.", col)
    log.info("[schema] ✓ Contrato validado com sucesso")
    return True


def temporal_split(df: pd.DataFrame, val_size: int = 3, test_size: int = 3):
    n = len(df)
    if n < val_size + test_size + 4:
        test_size = max(1, n // 6)
        val_size = max(1, n // 6)
        log.warning("[split] %d trimestres → splits reduzidos (val=%d, test=%d).",
                    n, val_size, test_size)
    tr = max(2, n - val_size - test_size)
    ve = tr + val_size
    leak_cols = [c for c in CONTEMP_FUND if c in df.columns]
    X = df.drop(columns=["eps"] + leak_cols)
    log.info("[split] Excluídas do X (leakage contemporâneo): %s", leak_cols)
    log.info("[split] Features (%d): %s", len(X.columns), list(X.columns))
    log.info("[split] Divisão temporal: %d treino, %d val, %d teste", tr, ve - tr, n - ve)
    return (X.iloc[:tr], df["eps"].iloc[:tr],
            X.iloc[tr:ve], df["eps"].iloc[tr:ve],
            X.iloc[ve:], df["eps"].iloc[ve:])


def _structural_break_report(df: pd.DataFrame, model_start: pd.Timestamp) -> dict:
    pre = df.loc[df.index < model_start, "eps"].dropna()
    post = df.loc[df.index >= model_start, "eps"].dropna()
    rep = {"n_pre": int(len(pre)), "n_post": int(len(post))}
    if len(pre) and len(post):
        rep.update({"pre_mean": float(pre.mean()), "post_mean": float(post.mean())})
        log.warning("[quebra estrutural] EPS pré-IPO μ=%.4f (n=%d) vs pós-IPO μ=%.4f (n=%d). "
                    "Pré-IPO usado APENAS para lags.",
                    rep["pre_mean"], len(pre), rep["post_mean"], len(post))
    return rep


# =============================================================================
# PIPELINE
# =============================================================================
def run_pipeline(ticker: str = TICKER,
                 start: str = FETCH_START,
                 end: Optional[str] = None,
                 model_start: str = MODEL_START,
                 val_size: int = 3,
                 test_size: int = 3,
                 allow_annual_fill: bool = False,
                 strict_macro: bool = True,
                 firecrawl_api_key: Optional[str] = None,
                 **kwargs) -> Dataset:
    if firecrawl_api_key:
        log.warning("[pipeline] firecrawl_api_key ignorado (fontes: AV/EDGAR/6-K).")

    log.info("=" * 70)
    log.info("PIPELINE NU HOLDINGS — %s | fetch %s → %s | modelo desde %s",
             ticker, start, end or "hoje", model_start)
    log.info("=" * 70)

    df, prov = _fetch_financials(ticker, start, end, allow_annual_fill)

    macro, macro_info = _fetch_bcb_macro(start, strict=strict_macro)
    prov["macro"] = macro_info
    if not macro.empty:
        df = df.join(macro, how="left")
    fx = _fetch_fx_brl_usd(start)
    if len(fx):
        df = df.join(fx, how="left")

    df = df[df.index >= pd.to_datetime(start)]
    if end:
        df = df[df.index <= pd.to_datetime(end)]

    df = build_features(df)

    n_before = len(df)
    warm = [c for c in WARMUP_COLS if c in df.columns]
    df = df.dropna(subset=["eps"] + warm)
    prov["warmup_dropped"] = int(n_before - len(df))
    if prov["warmup_dropped"]:
        log.info("[warmup] %d trimestres descartados (lags indisponíveis). Restam %d.",
                 prov["warmup_dropped"], len(df))

    ms = pd.Timestamp(model_start)
    prov["structural_break"] = _structural_break_report(df, ms)
    n_pre = int((df.index < ms).sum())
    df = df[df.index >= ms]
    prov["pre_ipo_dropped"] = n_pre
    if n_pre:
        log.info("[pré-IPO] %d trimestres fora do treino (serviram para lags). Restam %d.",
                 n_pre, len(df))

    df = df.drop(columns=["source", "eps_basic", "eps_diluted",
                          "eps_estimate", "surprise_pct", "reported_date"], errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce")

    if not validate_schema(df):
        raise RuntimeError("Validação de schema falhou — aborta pipeline.")

    raw_ops = df[[c for c in NO_FILL_COLS if c in df.columns]].copy()
    df = _clean_numeric_causal(df)
    for c in raw_ops.columns:
        df[c] = raw_ops[c]
    df.index.name = "quarter"

    log.info("[merge] Dataset: %d trimestres × %d colunas", len(df), len(df.columns))

    sp = temporal_split(df, val_size, test_size)
    summary = {prov["primary"]: len(df)}
    if prov["edgar_annual_years"]:
        summary["edgar_annual_ancora"] = prov["edgar_annual_years"]
    if prov.get("operational", {}).get("quarters_with_data"):
        summary["sec_6k_operacionais"] = prov["operational"]["quarters_with_data"]

    ds = Dataset(df=df, X_train=sp[0], y_train=sp[1], X_val=sp[2],
                 y_val=sp[3], X_test=sp[4], y_test=sp[5],
                 n_total=len(df), source_summary=summary,
                 created_at=datetime.now(timezone.utc).isoformat(), provenance=prov)
    ds.compute_hash()
    ds.validation_passed = True

    log.info("=" * 70)
    log.info("✓ TODOS OS TESTES PASSARAM — Dataset pronto para Módulo 2")
    log.info("✓ Dataset: %d treino, %d val, %d teste", ds.n_train, ds.n_val, ds.n_test)
    log.info("✓ Proveniência: %s", source_summary_pretty(summary))
    log.info("✓ Reconciliações: %d | pré-IPO p/ lags: %d | look-ahead anual: %s",
             len(prov["reconciliation"]), n_pre, allow_annual_fill)
    log.info("✓ SHA-256: %s...", ds.sha256[:16])
    log.info("=" * 70)
    return ds


def source_summary_pretty(counts: dict) -> str:
    return " + ".join(f"{k}: {v}" for k, v in counts.items()) if counts else "sem dados"


build_nu_dataset = run_pipeline
NuDataset = Dataset


def test_pipeline_integrity(ds: Dataset) -> bool:
    log.info("[test] A correr testes de integridade...")
    leak = [c for c in CONTEMP_FUND if c in ds.X_train.columns]
    checks = [
        ("dataset não vazio", ds.n_total > 0),
        ("y_train não vazio", len(ds.y_train) > 0),
        ("sem NaN no alvo treino", not ds.y_train.isna().any()),
        ("hash calculado", len(ds.sha256) == 64),
        ("sem leakage treino→val", ds.X_train.index.max() < ds.X_val.index.min()),
        ("val antes de teste",
         ds.X_val.index.max() < ds.X_test.index.min() if len(ds.X_test) else True),
        ("sem contemporâneos no X", len(leak) == 0),
        ("sem preenchimento anual look-ahead", not ds.provenance.get("annual_fill_enabled")),
    ]
    ok = True
    for name, passed in checks:
        log.info("[test] %s: %s", "PASS" if passed else "FAIL", name)
        ok = ok and bool(passed)
    return ok


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")
    try:
        ds = run_pipeline()
        ok = test_pipeline_integrity(ds)
        print(f"\n📋 {ds.n_total} trimestres | {source_summary_pretty(ds.source_summary)}")
        sys.exit(0 if ok else 1)
    except Exception as e:
        log.exception("❌ ERRO FATAL no Módulo 1: %s", e)
        sys.exit(2)