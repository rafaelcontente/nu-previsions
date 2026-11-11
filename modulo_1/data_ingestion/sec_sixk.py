# -*- coding: utf-8 -*-
"""
=============================================================================
MÓDULO 1 · PARSING DOS 6-K — métricas operacionais não-XBRL
=============================================================================
A Nu Holdings divulga ARPAC e NPL 15-90 nos comunicados trimestrais (6-K),
em HTML/tabelas, NÃO em XBRL. Nenhuma API estruturada as fornece.

Estratégia (best-effort com proveniência estrita):
  1. submissions API → lista os 6-K mais recentes
  2. index.json      → identifica documentos candidatos
  3. HTML → texto    → procura padrões explícitos de ARPAC / NPL 15-90
  4. Validação       → unidade e intervalo plausível; rejeita ambiguidade
  5. Proveniência    → accession, documento, excerto do texto que deu o valor

PRINCÍPIO: se o padrão não for inequívoco, o valor NÃO é extraído. Nunca se
preenche com defaults. Cobertura parcial é reportada como tal — o Módulo 3
marca NOT_VERIFIED nas regras cujos campos faltam.

Limitação declarada: os nomes das métricas variam entre anos; a cobertura
pode ser parcial ou nula. Isso é reportado, não escondido.
=============================================================================
"""
from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from .sec_edgar import EdgarClient, resolve_cik, USER_AGENT

log = logging.getLogger("sec-6k")

# Intervalos de plausibilidade (validação, não calibração estatística)
ARPAC_MIN, ARPAC_MAX = 0.5, 200.0     # USD por cliente ativo / mês
NPL_MIN, NPL_MAX = 0.0, 40.0          # % da carteira

# Padrões: etiqueta → número. Aceita "US$", "$", separador decimal . ou ,
_NUM = r"([0-9]{1,3}(?:[.,][0-9]{1,2})?)"
ARPAC_PATTERNS = [
    rf"ARPAC[^0-9]{{0,120}}?(?:US\$|\$)\s*{_NUM}",
    rf"ARPAC[^0-9]{{0,120}}?{_NUM}\s*(?:US\$|\$)",
    rf"average\s+(?:monthly\s+)?revenue\s+per\s+active\s+customer[^0-9]{{0,120}}?{_NUM}",
]
NPL_PATTERNS = [
    rf"NPL\s*15\s*[-–—to]{1,3}\s*90[^0-9]{{0,120}}?{_NUM}\s*%",
    rf"non[- ]performing\s+loans?\s*15\s*[-–—to]{1,3}\s*90[^0-9]{{0,120}}?{_NUM}\s*%",
    rf"15\s*[-–—to]{1,3}\s*90\s*day[^0-9]{{0,120}}?{_NUM}\s*%",
]


@dataclass
class OperationalMetrics:
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    coverage: dict = field(default_factory=dict)
    fetch_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())


# =============================================================================
# HTML → TEXTO
# =============================================================================
def _to_text(raw: str) -> str:
    """Remove tags/scripts, desescapa entidades, normaliza espaços."""
    t = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    t = html.unescape(t)
    t = t.replace("\xa0", " ")
    return re.sub(r"\s+", " ", t)


def _num(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", "."))
    except (TypeError, ValueError):
        return None


def _extract(text: str, patterns: list, lo: float, hi: float) -> tuple:
    """
    Devolve (valor, excerto) do PRIMEIRO padrão que produza um valor no
    intervalo plausível. Se vários padrões derem valores diferentes,
    considera ambíguo e devolve (None, motivo).
    """
    found = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            v = _num(m.group(1))
            if v is None or not (lo <= v <= hi):
                continue
            snippet = text[max(0, m.start() - 40): m.end() + 20]
            found.append((v, snippet.strip()))
    if not found:
        return None, None
    values = {round(v, 4) for v, _ in found}
    if len(values) > 1:
        return None, f"AMBÍGUO: candidatos {sorted(values)}"
    return found[0]


# =============================================================================
# DESCOBERTA DE DOCUMENTOS
# =============================================================================
def _list_6k(client: EdgarClient, cik: str, max_filings: int) -> list:
    url = f"{client.BASE_DATA}/submissions/CIK{str(cik).zfill(10)}.json"
    data = client.fetch_json(url)
    recent = data.get("filings", {}).get("recent", {})
    out = []
    forms = recent.get("form", [])
    for i in range(len(forms)):
        if forms[i] != "6-K":
            continue
        acc = recent["accessionNumber"][i].replace("-", "")
        out.append({
            "accession": acc,
            "filed": recent["filingDate"][i],
            "report_date": (recent.get("reportDate") or [None] * len(forms))[i]
                           or recent["filingDate"][i],
            "primary": recent["primaryDocument"][i],
            "folder": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}",
        })
        if len(out) >= max_filings:
            break
    log.info("[6-K] %d filings 6-K candidatos (máx %d)", len(out), max_filings)
    return out


def _candidate_docs(client: EdgarClient, f: dict) -> list:
    """Documentos .htm do filing, priorizando exhibits de resultados."""
    idx = client.fetch_json(f"{f['folder']}/index.json")
    names = [it.get("name", "") for it in idx.get("directory", {}).get("item", [])]
    htm = [n for n in names if n.lower().endswith((".htm", ".html"))]

    def score(n: str) -> int:
        s = n.lower()
        return (0 if ("ex99" in s or "ex-99" in s or "earnings" in s or "results" in s)
                else 1 if "6k" in s else 2)

    return sorted(htm, key=score)[:4]      # limita pedidos por filing


# =============================================================================
# EXTRAÇÃO
# =============================================================================
def fetch_operational_metrics(ticker: str = "NU",
                              cik: Optional[str] = None,
                              user_agent: str = USER_AGENT,
                              max_filings: int = 10) -> OperationalMetrics:
    """
    Percorre os 6-K mais recentes e extrai ARPAC / NPL 15-90 quando o padrão
    é inequívoco. Devolve DataFrame por trimestre + relatório de cobertura.
    """
    log.info("=" * 70)
    log.info("[6-K] A procurar métricas operacionais (ARPAC, NPL 15-90) nos 6-K")
    log.info("=" * 70)

    client = EdgarClient(user_agent=user_agent)
    cik = cik or resolve_cik(client, ticker)
    filings = _list_6k(client, cik, max_filings)
    if not filings:
        log.warning("[6-K] Nenhum 6-K encontrado.")
        return OperationalMetrics()

    rows, prov, ambiguous = {}, [], []
    for f in filings:
        docs = _candidate_docs(client, f)
        got = {}
        for name in docs:
            raw = client.fetch_json.__self__.fetch_text(f"{f['folder']}/{name}") \
                if hasattr(client, "fetch_text") else None
            if raw is None:
                # fallback: usa o session diretamente (HTML não é JSON)
                client._throttle()
                try:
                    r = client.session.get(f"{f['folder']}/{name}", timeout=30)
                    r.raise_for_status()
                    raw = r.text
                except Exception as e:
                    log.warning("[6-K] %s falhou: %s", name, type(e).__name__)
                    continue
            text = _to_text(raw)
            if "ARPAC" not in text.upper() and "NPL" not in text.upper():
                continue

            for key, pats, lo, hi in (
                ("arpac", ARPAC_PATTERNS, ARPAC_MIN, ARPAC_MAX),
                ("npl_15_90", NPL_PATTERNS, NPL_MIN, NPL_MAX),
            ):
                if key in got:
                    continue
                v, snip = _extract(text, pats, lo, hi)
                if v is not None:
                    got[key] = v
                    prov.append({"accession": f["accession"], "doc": name,
                                 "field": key, "value": v, "snippet": snip[:200]})
                elif isinstance(snip, str) and snip.startswith("AMBÍGUO"):
                    ambiguous.append({"accession": f["accession"], "doc": name,
                                      "field": key, "reason": snip})
            if len(got) == 2:
                break

        if got:
            try:
                q = pd.Timestamp(f["report_date"]).to_period("Q").end_time.normalize()
            except Exception:
                continue
            prev = rows.get(q, {})
            prev.update(got)
            rows[q] = prev
            log.info("[6-K] %s (%s) → %s", f["filed"], f["accession"][:10], got)

    if not rows:
        log.warning("[6-K] ⚠ Nenhuma métrica extraída. Os nomes/formatos variam entre "
                    "anos; as regras dependentes ficarão NOT_VERIFIED (não se inventa).")
        return OperationalMetrics(coverage={"quarters_with_data": 0,
                                            "ambiguous": ambiguous})

    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index.name = "date"
    cov = {
        "quarters_with_data": int(len(df)),
        "arpac_found": int(df["arpac"].notna().sum()) if "arpac" in df else 0,
        "npl_found": int(df["npl_15_90"].notna().sum()) if "npl_15_90" in df else 0,
        "filings_scanned": len(filings),
        "provenance": prov,
        "ambiguous": ambiguous,
    }
    log.info("[6-K] ✓ cobertura: %d trimestres | ARPAC=%d | NPL=%d | ambíguos=%d",
             cov["quarters_with_data"], cov["arpac_found"], cov["npl_found"], len(ambiguous))
    if ambiguous:
        for a in ambiguous:
            log.warning("[6-K] rejeitado por ambiguidade: %s %s — %s",
                        a["accession"][:10], a["field"], a["reason"])
    return OperationalMetrics(df=df, coverage=cov)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")
    om = fetch_operational_metrics()
    print(om.df.to_string() if not om.df.empty else "(sem métricas extraídas)")
    print("\nCobertura:", {k: v for k, v in om.coverage.items()
                           if k not in ("provenance", "ambiguous")})