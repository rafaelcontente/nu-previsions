# -*- coding: utf-8 -*-
"""
=============================================================================
MÓDULO 1 · ALPHA VANTAGE — fundamentais trimestrais com histórico longo
=============================================================================
Motivação: a Nu Holdings não publica XBRL trimestral no EDGAR (só 20-F anual)
e o yfinance devolve apenas ~6 trimestres. Com n=5 o pipeline não tem poder
estatístico. O Alpha Vantage fornece histórico trimestral desde o IPO.

Endpoints usados (3 pedidos/dia, cache 24h):
  EARNINGS          -> quarterlyEarnings: reportedEPS, estimatedEPS,
                       surprisePercentage, reportedDate  (alvo + data de publicação)
  INCOME_STATEMENT  -> quarterlyReports: totalRevenue, netIncome
  BALANCE_SHEET     -> quarterlyReports: totalAssets, totalShareholderEquity,
                       commonStockSharesOutstanding

Rate limit (free tier): 25 pedidos/dia, 5 pedidos/minuto.
Chave: variável de ambiente ALPHAVANTAGE_API_KEY (NUNCA hardcoded).

Data Lineage (Taxonomia V11, Pilar 1): cada coluna reporta quantos valores
vieram desta fonte. `reported_date` é preservado para auditoria point-in-time.
=============================================================================
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

log = logging.getLogger("alpha-vantage")

BASE_URL = "https://www.alphavantage.co/query"
CACHE_DIR = Path(".cache/alpha_vantage")
CACHE_TTL_HOURS = 24
MIN_REQUEST_INTERVAL = 12.5      # 5 pedidos/minuto (free tier)
ENV_KEY = "ALPHAVANTAGE_API_KEY"    

_LAST_CALL = 0.0                 # throttle global ao processo


@dataclass
class AlphaVantageDataset:
    symbol: str
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    provenance: dict = field(default_factory=dict)
    fetch_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# =============================================================================
# CHAVE, CACHE E THROTTLE
# =============================================================================
def _api_key(explicit: Optional[str] = None) -> Optional[str]:
    key = (explicit or os.environ.get(ENV_KEY) or "").strip()
    return key or None


def _cache_path(function: str, symbol: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{function}_{symbol.upper()}.json"


def _throttle() -> None:
    global _LAST_CALL
    elapsed = time.time() - _LAST_CALL
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)
    _LAST_CALL = time.time()


def _fetch(function: str, symbol: str, api_key: str) -> dict:
    """GET com cache 24h. Respostas de erro/limite NÃO são cacheadas."""
    cp = _cache_path(function, symbol)
    if cp.exists():
        age = datetime.now() - datetime.fromtimestamp(cp.stat().st_mtime)
        if age < timedelta(hours=CACHE_TTL_HOURS):
            try:
                return json.loads(cp.read_text(encoding="utf-8"))
            except Exception:
                pass

    _throttle()
    try:
        r = requests.get(BASE_URL, timeout=30,
                         params={"function": function, "symbol": symbol, "apikey": api_key})
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning("[AV] %s %s falhou: %s: %s", function, symbol, type(e).__name__, e)
        return {}

    # O AV sinaliza limites/erros com estas chaves (HTTP 200!)
    for k in ("Note", "Information", "Error Message"):
        if k in data:
            log.warning("[AV] %s %s → %s: %s", function, symbol, k, str(data[k])[:220])
            return {}
    if not data:
        log.warning("[AV] %s %s → resposta vazia.", function, symbol)
        return {}

    try:
        cp.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass
    return data


def _to_float(v) -> Optional[float]:
    """O AV usa a string 'None' para valores em falta."""
    if v is None:
        return None
    s = str(v).strip()
    if s in ("", "None", "null", "-", "NaN"):
        return None
    try:
        f = float(s)
        return f if pd.notna(f) else None
    except (TypeError, ValueError):
        return None


# =============================================================================
# ENDPOINTS
# =============================================================================
def fetch_earnings(symbol: str, api_key: str) -> pd.DataFrame:
    """Alvo principal: EPS trimestral reportado, com data de publicação."""
    data = _fetch("EARNINGS", symbol, api_key)
    recs = []
    for q in data.get("quarterlyEarnings") or []:
        fd, eps = q.get("fiscalDateEnding"), _to_float(q.get("reportedEPS"))
        if not fd or eps is None:
            continue
        recs.append({
            "date": pd.Timestamp(fd),
            "eps": eps,
            "eps_estimate": _to_float(q.get("estimatedEPS")),
            "surprise_pct": _to_float(q.get("surprisePercentage")),
            "reported_date": q.get("reportedDate"),
        })
    if not recs:
        log.warning("[AV] EARNINGS sem trimestres utilizáveis para %s.", symbol)
        return pd.DataFrame()
    df = (pd.DataFrame(recs).drop_duplicates("date", keep="last")
          .set_index("date").sort_index())
    log.info("[AV] EARNINGS: %d trimestres com EPS (%s → %s)",
             len(df), df.index.min().date(), df.index.max().date())
    return df


def fetch_income_statement(symbol: str, api_key: str) -> pd.DataFrame:
    """Receita e lucro líquido trimestrais. Pode vir vazio em emissores IFRS."""
    data = _fetch("INCOME_STATEMENT", symbol, api_key)
    recs = []
    for q in data.get("quarterlyReports") or []:
        fd = q.get("fiscalDateEnding")
        if not fd:
            continue
        recs.append({"date": pd.Timestamp(fd),
                     "revenue": _to_float(q.get("totalRevenue")),
                     "net_income": _to_float(q.get("netIncome"))})
    if not recs:
        log.warning("[AV] INCOME_STATEMENT sem trimestres para %s.", symbol)
        return pd.DataFrame()
    df = (pd.DataFrame(recs).drop_duplicates("date", keep="last")
          .set_index("date").sort_index())
    for c in ("revenue", "net_income"):
        log.info("[AV] INCOME_STATEMENT %s: %d/%d valores não nulos",
                 c, int(df[c].notna().sum()), len(df))
    return df


def fetch_balance_sheet(symbol: str, api_key: str) -> pd.DataFrame:
    """Ativos, capitais próprios e nº de ações trimestrais."""
    data = _fetch("BALANCE_SHEET", symbol, api_key)
    recs = []
    for q in data.get("quarterlyReports") or []:
        fd = q.get("fiscalDateEnding")
        if not fd:
            continue
        recs.append({"date": pd.Timestamp(fd),
                     "total_assets": _to_float(q.get("totalAssets")),
                     "equity": _to_float(q.get("totalShareholderEquity")),
                     "shares": _to_float(q.get("commonStockSharesOutstanding"))})
    if not recs:
        log.warning("[AV] BALANCE_SHEET sem trimestres para %s.", symbol)
        return pd.DataFrame()
    df = (pd.DataFrame(recs).drop_duplicates("date", keep="last")
          .set_index("date").sort_index())
    log.info("[AV] BALANCE_SHEET: %d trimestres", len(df))
    return df


# =============================================================================
# API PÚBLICA
# =============================================================================
def fetch_quarterly_fundamentals(symbol: str = "NU",
                                 start: str = "2021-12-01",
                                 end: Optional[str] = None,
                                 api_key: Optional[str] = None) -> AlphaVantageDataset:
    """
    Junta os 3 endpoints num frame trimestral. O índice é o fim do trimestre
    fiscal. Só mantém trimestres com EPS (sem alvo não há previsão).
    Devolve proveniência contabilizada por coluna.
    """
    key = _api_key(api_key)
    if key is None:
        log.warning("[AV] %s não definida — Alpha Vantage desativado. "
                    "Define com: export %s=\"a_tua_chave\"", ENV_KEY, ENV_KEY)
        return AlphaVantageDataset(symbol=symbol)

    log.info("=" * 70)
    log.info("[AV] A extrair trimestres de %s via Alpha Vantage", symbol)
    log.info("=" * 70)

    earnings = fetch_earnings(symbol, key)
    if earnings.empty:
        return AlphaVantageDataset(symbol=symbol)

    income = fetch_income_statement(symbol, key)
    balance = fetch_balance_sheet(symbol, key)

    df = earnings.copy()
    for extra in (income, balance):
        if not extra.empty:
            df = df.join(extra, how="left")

    df = df[~df.index.duplicated(keep="last")].sort_index()
    lo = pd.Timestamp(start) - pd.DateOffset(months=1)
    hi = pd.Timestamp(end) + pd.DateOffset(months=1) if end else pd.Timestamp.now()
    df = df[(df.index >= lo) & (df.index <= hi)]
    df = df.dropna(subset=["eps"])
    df.index.name = "date"

    if df.empty:
        log.warning("[AV] Nenhum trimestre no intervalo pedido.")
        return AlphaVantageDataset(symbol=symbol)

    # Proveniência: contagem REAL de valores por coluna (nada de rótulos vagos)
    cols = {}
    for c in df.columns:
        n = int(df[c].notna().sum())
        if n:
            cols[c] = {"alpha_vantage": n}
    prov = {"source": "alpha_vantage", "columns": cols, "n_quarters": len(df)}

    if "reported_date" in df.columns:
        rd = pd.to_datetime(df["reported_date"], errors="coerce").dropna()
        if len(rd):
            log.info("[AV] Datas de publicação: %s → %s (auditoria point-in-time)",
                     rd.min().date(), rd.max().date())

    log.info("[AV] ✓ frame trimestral: %d trimestres (%s → %s) | colunas=%s",
             len(df), df.index.min().date(), df.index.max().date(), list(df.columns))
    return AlphaVantageDataset(symbol=symbol, df=df, provenance=prov)


def to_dataframe(ds: AlphaVantageDataset) -> pd.DataFrame:
    """Vazio e sem exceção se não houver dados."""
    if ds is None or ds.df is None or ds.df.empty:
        return pd.DataFrame()
    return ds.df.copy()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")
    d = fetch_quarterly_fundamentals()
    print(to_dataframe(d).to_string() if not d.df.empty else "(sem dados — verifica a chave)")