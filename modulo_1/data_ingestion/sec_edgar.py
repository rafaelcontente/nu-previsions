# -*- coding: utf-8 -*-
"""
=============================================================================
MÓDULO 1 · FONTE OFICIAL SEC/EDGAR + TRIMESTRES REAIS
=============================================================================
Factos verificados nas APIs públicas da SEC:
  * CIK da Nu Holdings Ltd. (ticker NU) = 0001691493.
    (0001866581 é Dutch Bros Inc. — nunca usar para NU.)
  * A Nu publica XBRL apenas ANUAL (form 20-F, taxonomia ifrs-full).
    Não existe XBRL trimestral no EDGAR para NU.

Arquitetura (frequências mistas / MIDAS):
  1. EDGAR companyfacts -> fundamentais ANUAIS oficiais (âncora).
  2. yfinance           -> trimestres REAIS.
  3. Lacunas            -> anual oficial do mesmo ano fiscal.
  4. Reconciliação      -> soma de 4 trimestres vs anual EDGAR.
  5. Sem yfinance       -> anual forward-fill por ano fiscal (rotulado).

API pública (estável — não renomear):
    fetch_nu_from_edgar(...) -> EdgarDataset
    dataset_to_dataframe(ds) -> pd.DataFrame
    get_quarterly_fundamentals(ticker, start, end) -> pd.DataFrame
=============================================================================
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

log = logging.getLogger("sec-edgar")

# A SEC exige contacto real. Define: export SEC_USER_AGENT="Nome email@dominio"
USER_AGENT = os.environ.get("SEC_USER_AGENT", "NuForecast Research contact@example.org")
CACHE_DIR = Path(".cache/sec_edgar")
CACHE_TTL_DAYS = 7
MIN_REQUEST_INTERVAL = 0.12
NU_CIK = "0001691493"
TICKER = "NU"

CONCEPTS = {
    "revenue": ["Revenue", "Revenues", "RevenueFromContractsWithCustomers",
                "InterestAndFeeIncomeLoansAndLeases", "InterestIncomeExpenseNet",
                "RevenueFromContractWithCustomerExcludingAssessedTax"],
    "net_income": ["ProfitLoss", "NetIncomeLoss",
                   "ProfitLossAttributableToOwnersOfParent",
                   "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "total_assets": ["Assets", "TotalAssets"],
    "equity": ["Equity", "EquityAttributableToOwnersOfParent",
               "StockholdersEquity", "TotalEquity"],
    "eps_diluted": ["DilutedEarningsLossPerShare", "EarningsPerShareDiluted"],
    "eps_basic": ["BasicEarningsLossPerShare", "EarningsPerShareBasic"],
}
INSTANT_FIELDS = {"total_assets", "equity"}


@dataclass
class EdgarDataset:
    cik: str
    ticker: str
    company_name: str
    df: pd.DataFrame = field(default_factory=pd.DataFrame)        # trimestral
    annual: pd.DataFrame = field(default_factory=pd.DataFrame)    # anual oficial
    fetch_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())


class EdgarClient:
    """Cliente HTTP respeitoso (throttle + cache em disco)."""
    BASE_DATA = "https://data.sec.gov"

    def __init__(self, user_agent: str = USER_AGENT, cache_dir: Path = CACHE_DIR):
        if "example.org" in user_agent or "exemplo.com" in user_agent:
            log.warning("[EDGAR] ⚠ USER_AGENT com email placeholder. A SEC exige contacto "
                        "real e pode bloquear o IP. Define: "
                        "export SEC_USER_AGENT=\"Nome email@dominio.com\"")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json",
        })
        self._last = 0.0

    def _throttle(self) -> None:
        el = time.time() - self._last
        if el < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - el)
        self._last = time.time()

    def _cache_path(self, url: str) -> Path:
        h = hashlib.md5(url.encode()).hexdigest()[:16]
        tail = re.sub(r"[^a-zA-Z0-9._-]", "_", url.split("/")[-1])[:60]
        return self.cache_dir / f"{h}_{tail}"

    def fetch_json(self, url: str) -> dict:
        cp = self._cache_path(url)
        if cp.exists():
            age = datetime.now() - datetime.fromtimestamp(cp.stat().st_mtime)
            if age < timedelta(days=CACHE_TTL_DAYS):
                try:
                    return json.loads(cp.read_text(encoding="utf-8"))
                except Exception:
                    pass
        self._throttle()
        try:
            r = self.session.get(url, timeout=30)
            if r.status_code == 404:
                log.warning("[EDGAR] 404: %s", url)
                return {}
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log.warning("[EDGAR] Falha em %s: %s: %s", url, type(e).__name__, e)
            return {}
        try:
            cp.write_text(json.dumps(data), encoding="utf-8")
        except Exception:
            pass
        return data


def resolve_cik(client: EdgarClient, ticker: str = TICKER) -> str:
    """Nu Holdings tem CIK fixo (o ticker mapping é ambíguo para 'NU')."""
    if ticker.upper() == TICKER:
        log.info("[EDGAR] CIK oficial para %s: %s (Nu Holdings Ltd.)", TICKER, NU_CIK)
        return NU_CIK
    data = client.fetch_json("https://www.sec.gov/files/company_tickers.json")
    for entry in (data or {}).values():
        if str(entry.get("ticker", "")).upper() == ticker.upper():
            cik = str(entry["cik_str"]).zfill(10)
            log.info("[EDGAR] CIK resolvido para %s: %s (%s)", ticker, cik, entry.get("title"))
            return cik
    log.warning("[EDGAR] CIK não resolvido para %s; a usar %s", ticker, NU_CIK)
    return NU_CIK


def fetch_company_facts(client: EdgarClient, cik: str) -> dict:
    url = f"{client.BASE_DATA}/api/xbrl/companyfacts/CIK{str(cik).zfill(10)}.json"
    log.info("[EDGAR] A obter companyfacts de CIK %s", str(cik).zfill(10))
    return client.fetch_json(url)


# =============================================================================
# EXTRAÇÃO ANUAL (XBRL oficial)
# =============================================================================
def _concept_entries(facts: dict, field_name: str) -> list:
    for tax in ("ifrs-full", "us-gaap"):
        block = facts.get("facts", {}).get(tax, {})
        for tag in CONCEPTS[field_name]:
            node = block.get(tag)
            if not node:
                continue
            entries = node.get("units", {}).get("USD") or []
            if entries:
                return entries
    return []


def _dedup_by_end(rows: list) -> pd.Series:
    """Para cada data-fim mantém o valor do filing mais recente (revisões vencem)."""
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows).sort_values(["end", "filed"])
    df = df.drop_duplicates(subset="end", keep="last")
    return df.set_index("end")["val"].sort_index()


def build_annual_df(facts: dict) -> pd.DataFrame:
    """Fundamentais anuais oficiais: fluxos 350–380 dias; balanço instantâneo."""
    out = {}
    for field_name in CONCEPTS:
        rows = []
        for e in _concept_entries(facts, field_name):
            if e.get("val") is None or not e.get("end"):
                continue
            end = pd.Timestamp(e["end"])
            start = e.get("start")
            if field_name in INSTANT_FIELDS:
                if start:
                    continue
            else:
                if not start:
                    continue
                days = (end - pd.Timestamp(start)).days
                if not (350 <= days <= 380):
                    continue
            rows.append({"end": end, "val": float(e["val"]),
                         "filed": e.get("filed", "1900-01-01")})
        s = _dedup_by_end(rows)
        if len(s):
            out[field_name] = s
    annual = pd.DataFrame(out).sort_index()
    if not annual.empty:
        log.info("[EDGAR] ✓ %d anos oficiais (%s → %s) | campos=%s",
                 len(annual), annual.index.min().date(), annual.index.max().date(),
                 list(annual.columns))
    else:
        log.warning("[EDGAR] Nenhum fundamental anual extraído.")
    return annual


# =============================================================================
# TRIMESTRES REAIS (yfinance)
# =============================================================================
def _pick_row(df: Optional[pd.DataFrame], candidates: list) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    for name in candidates:
        if name in df.index:
            s = pd.to_numeric(df.loc[name], errors="coerce").dropna()
            if len(s):
                return s
    return None


def fetch_yfinance_quarterly(ticker: str = TICKER) -> pd.DataFrame:
    """Trimestres reportados (eps, revenue, net_income, assets, equity)."""
    try:
        import yfinance as yf
    except ImportError:
        log.warning("[yfinance] Pacote não instalado.")
        return pd.DataFrame()
    try:
        tkr = yf.Ticker(ticker)
        inc = tkr.quarterly_financials
        bal = tkr.quarterly_balance_sheet
    except Exception as e:
        log.warning("[yfinance] Falha: %s: %s", type(e).__name__, e)
        return pd.DataFrame()

    parts = {
        "eps": _pick_row(inc, ["Diluted EPS", "Basic EPS"]),
        "revenue": _pick_row(inc, ["Total Revenue", "Operating Revenue", "Revenue"]),
        "net_income": _pick_row(inc, ["Net Income", "Net Income Common Stockholders",
                                      "Net Income Including Noncontrolling Interests"]),
        "total_assets": _pick_row(bal, ["Total Assets"]),
        "equity": _pick_row(bal, ["Stockholders Equity", "Common Stock Equity",
                                  "Total Equity Gross Minority Interest"]),
    }
    parts = {k: v for k, v in parts.items() if v is not None}
    if not parts:
        return pd.DataFrame()
    df = pd.DataFrame(parts)
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    log.info("[yfinance] ✓ %d trimestres reais (%s → %s) | campos=%s",
             len(df), df.index.min().date(), df.index.max().date(), list(df.columns))
    return df


# =============================================================================
# MONTAGEM FINAL
# =============================================================================
def _fill_from_annual(q: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    """Preenche lacunas trimestrais com o anual oficial do mesmo ano fiscal."""
    if annual.empty:
        return q
    by_year = annual.groupby(annual.index.year).last()
    for col in ("revenue", "net_income", "total_assets", "equity", "eps"):
        src = col
        if col == "eps":
            src = "eps_diluted" if "eps_diluted" in by_year.columns else (
                  "eps_basic" if "eps_basic" in by_year.columns else None)
        if src is None or src not in by_year.columns:
            continue
        mapped = pd.Series(q.index.year, index=q.index).map(by_year[src])
        if col not in q.columns:
            q[col] = mapped
            log.info("[merge] '%s' trimestral preenchido com anual EDGAR (ano fiscal).", col)
        else:
            before = int(q[col].isna().sum())
            q[col] = q[col].fillna(mapped)
            filled = before - int(q[col].isna().sum())
            if filled:
                log.info("[merge] '%s': %d lacunas preenchidas com anual EDGAR.", col, filled)
    return q


def _reconcile(q: pd.DataFrame, annual: pd.DataFrame) -> None:
    """Auditoria: soma de 4 trimestres vs anual oficial. Não altera dados."""
    if annual.empty:
        return
    for col in ("revenue", "net_income"):
        if col not in q.columns or col not in annual.columns:
            continue
        for end, official in annual[col].items():
            sub = q.loc[q.index.year == end.year, col].dropna()
            if len(sub) == 4 and official:
                ratio = float(sub.sum()) / float(official)
                flag = "OK" if 0.85 <= ratio <= 1.15 else "DESVIO"
                log.info("[reconciliação] %s %s FY%d: 4T/EDGAR = %.3f", flag, col, end.year, ratio)


def _annual_ffill_frame(annual: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Fallback sem yfinance: grelha trimestral com anual oficial por ano fiscal."""
    if annual.empty:
        return pd.DataFrame()
    qidx = pd.period_range(pd.Timestamp(start).to_period("Q"),
                           pd.Timestamp(end).to_period("Q"), freq="Q").to_timestamp(how="E")
    by_year = annual.groupby(annual.index.year).last()
    df = pd.DataFrame(index=qidx)
    years = pd.Series(df.index.year, index=df.index)
    for col in ("revenue", "net_income", "total_assets", "equity"):
        if col in by_year.columns:
            df[col] = years.map(by_year[col])
    eps_src = "eps_diluted" if "eps_diluted" in by_year.columns else (
              "eps_basic" if "eps_basic" in by_year.columns else None)
    if eps_src:
        df["eps"] = years.map(by_year[eps_src])
    if "eps" in df.columns:
        df = df.dropna(subset=["eps"])
    df["source"] = "edgar_annual_ffill"
    log.warning("[SEC] Sem trimestres reais: a usar anual EDGAR por ano fiscal "
                "(estado anual, não trimestral fabricado).")
    return df


def fetch_nu_from_edgar(cik: Optional[str] = None,
                        user_agent: str = USER_AGENT,
                        ticker: str = TICKER,
                        start: str = "2021-12-01",
                        end: Optional[str] = None) -> EdgarDataset:
    """EDGAR anual (oficial) + trimestres reais + reconciliação."""
    end = end or datetime.now().strftime("%Y-%m-%d")
    log.info("=" * 70)
    log.info("[EDGAR] A extrair dados oficiais de %s", ticker)
    log.info("=" * 70)

    client = EdgarClient(user_agent=user_agent)
    resolved = resolve_cik(client, ticker)
    facts = fetch_company_facts(client, resolved)
    entity = facts.get("entityName", "Nu Holdings Ltd.")
    log.info("[EDGAR] entityName oficial: %s", entity)
    if ticker.upper() == TICKER and "NU HOLDINGS" not in entity.upper():
        log.warning("[EDGAR] entityName '%s' não parece Nu Holdings — verifica o CIK.", entity)

    annual = build_annual_df(facts) if facts else pd.DataFrame()
    quarterly = fetch_yfinance_quarterly(ticker)

    if not quarterly.empty:
        q = _fill_from_annual(quarterly.copy(), annual)
        _reconcile(q, annual)
        q["source"] = "yfinance_quarterly+edgar_annual"
    else:
        q = _annual_ffill_frame(annual, start, end)

    if q.empty:
        log.warning("[EDGAR] Dataset trimestral vazio — pipeline usará fallback.")
        return EdgarDataset(cik=resolved, ticker=ticker, company_name=entity, annual=annual)

    q = q.sort_index()
    q = q[(q.index >= pd.Timestamp(start) - pd.DateOffset(months=1)) &
          (q.index <= pd.Timestamp(end) + pd.DateOffset(months=1))]
    if "eps" in q.columns:
        q = q.dropna(subset=["eps"])
    q.index.name = "date"

    log.info("[EDGAR] ✓ frame final: %d trimestres | colunas=%s | source=%s",
             len(q), list(q.columns), q["source"].iloc[0] if "source" in q else "n/d")
    return EdgarDataset(cik=resolved, ticker=ticker, company_name=entity, df=q, annual=annual)


def dataset_to_dataframe(ds: EdgarDataset) -> pd.DataFrame:
    """Devolve o DataFrame trimestral. Vazio e SEM exceção se não houver dados."""
    if ds is None or ds.df is None or ds.df.empty:
        return pd.DataFrame()
    return ds.df.copy()


def get_quarterly_fundamentals(ticker: str = TICKER,
                               start: str = "2021-12-01",
                               end: Optional[str] = None) -> pd.DataFrame:
    """Atalho funcional (mesma lógica, devolve diretamente o DataFrame)."""
    return dataset_to_dataframe(fetch_nu_from_edgar(ticker=ticker, start=start, end=end))


# Aliases de compatibilidade — não remover.
extract_fundamentals = get_quarterly_fundamentals
fetch_fundamentals = get_quarterly_fundamentals
extract_sec_quarterly = get_quarterly_fundamentals
get_sec_data = get_quarterly_fundamentals


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")
    ds = fetch_nu_from_edgar()
    print("\n--- ANUAL OFICIAL (EDGAR) ---")
    print(ds.annual.to_string() if not ds.annual.empty else "(vazio)")
    print("\n--- TRIMESTRAL FINAL ---")
    print(dataset_to_dataframe(ds).to_string())