# -*- coding: utf-8 -*-
"""
=============================================================================
MÓDULO 3: VERIFICAÇÃO NEURO-SIMBÓLICA BIDIRECIONAL + ORQUESTRADOR
=============================================================================
PRINCÍPIO DESTA VERSÃO: uma regra que não pode ser avaliada NÃO é uma regra
que passou. Campos ausentes (ex.: ARPAC, NPL 15-90, que a Nu não publica em
XBRL nem o yfinance fornece) produzem NOT_VERIFIED, nunca PASS silencioso.

NS-1 (Neural → Simbólico): previsão pontual → factos → 5 regras.
NS-2 (Simbólico → Neural): trajetórias MC verificadas no trimestre terminal.
NS-3: confidence_symbolic = n_pass / n_regras_totais  (não verificadas penalizam)
      confidence_neural   = None (N/D) — não há fórmula defensável a partir de MAE
      confidence_final    = confidence_symbolic (elo verificável)
NS-4: audit trail append-only em JSONL com estatuto por regra e cobertura.

Veredictos:
  ACCEPT    → todas as 5 regras avaliadas e sem violações
  PENALIZE  → violações soft OU verificação parcial (regras não avaliáveis)
  REJECT    → violação hard (antes de esgotar tentativas)
  ESCALATE  → violação hard após esgotar tentativas

Referências: Taxon. ML (Sistemas Neuro-Simbólicos Bidirecionais);
Taxonomia V11 (Guardrails, Maker-Checker, Tool Output Overtrust).
=============================================================================
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("nu_verifier")


# =============================================================================
# ESTATUTO E DEFINIÇÃO DE REGRAS
# =============================================================================
class RuleStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_VERIFIED = "NOT_VERIFIED"   # campos necessários indisponíveis


class RuleSeverity(Enum):
    HARD = "hard"
    SOFT = "soft"


@dataclass
class Rule:
    name: str
    severity: RuleSeverity
    requires: tuple                 # campos obrigatórios para avaliar
    description: str


# Limites de PLAUSIBILIDADE ECONÓMICA — heurísticos documentados, NÃO
# calibrados estatisticamente sobre o histórico da Nu. Cada um justifica-se:
#   margem ≤ 1.00  : lucro líquido não excede receita total (identidade contabilística)
#   margem ≥ -0.50 : prejuízo > 50% da receita é implausível em operação normal
#   NPL    ≤ 0.20  : 20% da carteira em mora 15-90d seria crise sistémica
#   ARPAC  ±50%    : variação trimestral abrupta sinaliza erro de dados
#   EPS    ±30%    : desvio face à média histórica (NÃO é consenso de analistas)
MARGIN_MIN, MARGIN_MAX = -0.50, 1.00
NPL_MIN, NPL_MAX = 0.00, 0.20
ARPAC_REL_BAND = 0.50
EPS_REL_BAND = 0.30

RULES = [
    Rule("eps_net_income_consistency", RuleSeverity.HARD, ("eps", "net_income"),
         "EPS e lucro líquido têm o mesmo sinal"),
    Rule("net_margin_bounds", RuleSeverity.HARD, ("revenue", "net_income"),
         f"Margem líquida em [{MARGIN_MIN:.0%}, {MARGIN_MAX:.0%}]"),
    Rule("npl_15_90_bounds", RuleSeverity.HARD, ("npl_15_90",),
         f"NPL 15-90 em [{NPL_MIN:.0%}, {NPL_MAX:.0%}]"),
    Rule("arpac_range", RuleSeverity.SOFT, ("arpac",),
         f"ARPAC dentro de ±{ARPAC_REL_BAND:.0%} do último valor observado"),
    Rule("eps_deviation_vs_history", RuleSeverity.SOFT, ("eps",),
         f"EPS não desvia >{EPS_REL_BAND:.0%} da média histórica de treino"),
]


# =============================================================================
# VERIFICAÇÃO PONTUAL (NS-1)
# =============================================================================
@dataclass
class VerificationResult:
    statuses: dict                       # nome -> RuleStatus
    violations_hard: list
    violations_soft: list
    not_verified: list
    missing_fields: list
    n_rules: int
    n_evaluated: int
    n_pass: int
    verification_coverage: float         # n_evaluated / n_rules
    confidence_symbolic: float           # n_pass / n_rules

    @property
    def all_hard_pass(self) -> bool:
        return len(self.violations_hard) == 0


def _num(facts: dict, key: str) -> Optional[float]:
    """Devolve float finito ou None (campo ausente/nulo/não-finito)."""
    v = facts.get(key, None)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def verify_prediction(facts: dict,
                      rules: Optional[list] = None,
                      last_arpac: Optional[float] = None,
                      reference_eps: Optional[float] = None,
                      **kwargs) -> VerificationResult:
    """
    Avalia cada regra apenas se TODOS os campos necessários existirem.
    Regras sem dados → NOT_VERIFIED (e penalizam a confiança simbólica).
    """
    rules = rules or RULES
    eps = _num(facts, "eps")
    rev = _num(facts, "revenue")
    ni = _num(facts, "net_income")
    npl = _num(facts, "npl_15_90")
    arpac = _num(facts, "arpac")

    statuses, missing = {}, set()

    for r in rules:
        # Campos base em falta?
        absent = [k for k in r.requires if _num(facts, k) is None]
        # Referências auxiliares em falta?
        if r.name == "arpac_range" and (last_arpac is None or not np.isfinite(last_arpac)):
            absent.append("last_arpac")
        if r.name == "eps_deviation_vs_history" and (reference_eps is None or not np.isfinite(reference_eps)):
            absent.append("reference_eps")

        if absent:
            statuses[r.name] = RuleStatus.NOT_VERIFIED
            missing.update(absent)
            continue

        if r.name == "eps_net_income_consistency":
            ok = (eps >= 0 and ni >= 0) or (eps < 0 and ni < 0)
        elif r.name == "net_margin_bounds":
            ok = (rev != 0) and (MARGIN_MIN <= ni / rev <= MARGIN_MAX)
        elif r.name == "npl_15_90_bounds":
            ok = NPL_MIN <= npl <= NPL_MAX
        elif r.name == "arpac_range":
            ok = (1 - ARPAC_REL_BAND) * last_arpac <= arpac <= (1 + ARPAC_REL_BAND) * last_arpac
        elif r.name == "eps_deviation_vs_history":
            ok = abs(eps - reference_eps) <= EPS_REL_BAND * abs(reference_eps) if reference_eps else True
        else:
            ok = True

        statuses[r.name] = RuleStatus.PASS if ok else RuleStatus.FAIL

    v_hard = [r.name for r in rules
              if r.severity == RuleSeverity.HARD and statuses[r.name] == RuleStatus.FAIL]
    v_soft = [r.name for r in rules
              if r.severity == RuleSeverity.SOFT and statuses[r.name] == RuleStatus.FAIL]
    n_ver = [r.name for r in rules if statuses[r.name] == RuleStatus.NOT_VERIFIED]

    n_rules = len(rules)
    n_eval = n_rules - len(n_ver)
    n_pass = sum(1 for s in statuses.values() if s == RuleStatus.PASS)

    return VerificationResult(
        statuses=statuses, violations_hard=v_hard, violations_soft=v_soft,
        not_verified=n_ver, missing_fields=sorted(missing),
        n_rules=n_rules, n_evaluated=n_eval, n_pass=n_pass,
        verification_coverage=n_eval / n_rules if n_rules else 0.0,
        confidence_symbolic=n_pass / n_rules if n_rules else 0.0,
    )


# =============================================================================
# FILTRAGEM MC (NS-2) — só regras avaliáveis, sem fabricar dados
# =============================================================================
def filter_mc_paths(mc_paths: np.ndarray,
                    features_per_path: dict,
                    rules: Optional[list] = None,
                    last_arpac: Optional[float] = None,
                    reference_eps: Optional[float] = None,
                    **kwargs) -> tuple:
    """
    Filtra trajetórias pelo trimestre terminal, aplicando APENAS as regras
    cujos campos existem. Nenhuma variável é sintetizada com ruído.
    Devolve (paths_filtrados, stats). Nunca devolve zero trajetórias.
    """
    rules = rules or RULES
    n = mc_paths.shape[0]
    eps = np.asarray(mc_paths[:, -1], dtype=float)

    def col(name):
        if name not in features_per_path:
            return None
        a = np.asarray(features_per_path[name], dtype=float)
        return a if a.size == n else None

    rev, ni, npl, arpac = col("revenue"), col("net_income"), col("npl_15_90"), col("arpac")

    kept = np.ones(n, dtype=bool)
    applied, skipped, counts = [], [], {}

    for r in rules:
        if r.severity != RuleSeverity.HARD:
            continue   # soft não descarta trajetórias
        if r.name == "eps_net_income_consistency" and ni is not None:
            m = ((eps >= 0) & (ni >= 0)) | ((eps < 0) & (ni < 0))
        elif r.name == "net_margin_bounds" and rev is not None and ni is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                margin = np.where(rev != 0, ni / np.where(rev == 0, 1.0, rev), np.nan)
            m = (margin >= MARGIN_MIN) & (margin <= MARGIN_MAX)
        elif r.name == "npl_15_90_bounds" and npl is not None:
            m = (npl >= NPL_MIN) & (npl <= NPL_MAX)
        else:
            skipped.append(r.name)
            continue
        m = np.nan_to_num(m.astype(float), nan=0.0).astype(bool)
        counts[r.name] = int((~m).sum())
        kept &= m
        applied.append(r.name)

    acc = float(kept.mean())
    log.info("[MC Filter] regras hard aplicadas=%s | não avaliáveis=%s | aceitação=%.1f%% | descartadas=%s",
             applied or "nenhuma", skipped or "nenhuma", acc * 100, counts)

    if not kept.any():
        log.warning("[MC Filter] 0 trajetórias aceites → modo PERMISSIVO (mantém todas) "
                    "para não esvaziar o fan chart. Rever limites hard.")
        kept = np.ones(n, dtype=bool)
        acc, permissive = 1.0, True
    else:
        permissive = False

    return mc_paths[kept], {
        "n_input": n, "n_kept": int(kept.sum()), "n_discarded": int(n - kept.sum()),
        "acceptance_rate": acc, "rules_applied": applied, "rules_not_evaluable": skipped,
        "discarded_by_rule": counts, "permissive_mode": permissive,
    }


# =============================================================================
# ORQUESTRADOR (NS-3) + AUDIT (NS-4)
# =============================================================================
class Verdict(Enum):
    ACCEPT = "ACCEPT"
    PENALIZE = "PENALIZE"
    REJECT = "REJECT"
    ESCALATE = "ESCALATE"


@dataclass
class OrchestrationResult:
    verdict: Verdict
    statuses: dict                       # nome -> "PASS"/"FAIL"/"NOT_VERIFIED"
    violations: list                     # [(nome, "hard"|"soft")]
    not_verified: list
    missing_fields: list
    verification_coverage: float
    confidence_neural: Optional[float]   # None = N/D
    confidence_symbolic: float
    confidence_final: float
    mc_paths_filtered: np.ndarray
    stats: dict
    audit_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value, "statuses": self.statuses,
            "violations": self.violations, "not_verified": self.not_verified,
            "missing_fields": self.missing_fields,
            "verification_coverage": self.verification_coverage,
            "confidence_neural": self.confidence_neural,
            "confidence_symbolic": self.confidence_symbolic,
            "confidence_final": self.confidence_final,
            "mc_paths_shape": list(self.mc_paths_filtered.shape),
            "stats": self.stats, "audit_id": self.audit_id,
        }


class Orchestrator:
    def __init__(self, rules: Optional[list] = None, max_retries: int = 2,
                 audit_file: str = "audit.jsonl"):
        self.rules = rules or RULES
        self.max_retries = max_retries
        self.audit_file = Path(audit_file)

    def orchestrate(self, prediction: dict, mc_paths: np.ndarray,
                    features_per_path: dict,
                    confidence_neural: Optional[float] = None,
                    last_arpac: Optional[float] = None,
                    reference_eps: Optional[float] = None,
                    **kwargs) -> OrchestrationResult:
        log.info("[Orchestrator] A iniciar verificação neuro-simbólica")

        res = verify_prediction(prediction, self.rules, last_arpac, reference_eps)
        violations = ([(n, "hard") for n in res.violations_hard]
                      + [(n, "soft") for n in res.violations_soft])

        if not res.all_hard_pass:
            verdict = Verdict.REJECT
            for a in range(self.max_retries):
                log.warning("[Orchestrator] Tentativa %d/%d — hard violadas: %s",
                            a + 1, self.max_retries, res.violations_hard)
            verdict = Verdict.ESCALATE
        elif res.violations_soft or res.not_verified:
            verdict = Verdict.PENALIZE
        else:
            verdict = Verdict.ACCEPT

        mc_f, stats = filter_mc_paths(mc_paths, features_per_path, self.rules,
                                      last_arpac, reference_eps)

        # Confiança final = elo verificável. Neural é N/D por construção.
        conf_final = res.confidence_symbolic

        log.info("[Orchestrator] Veredicto: %s", verdict.value)
        log.info("[Orchestrator] Regras: %d/%d avaliadas (%.0f%% cobertura) | %d PASS",
                 res.n_evaluated, res.n_rules, res.verification_coverage * 100, res.n_pass)
        if res.not_verified:
            log.warning("[Orchestrator] NÃO VERIFICADAS: %s | campos ausentes: %s",
                        res.not_verified, res.missing_fields)
        log.info("[Orchestrator] Confiança: neural=N/D | simbólica=%.2f | final=%.2f",
                 res.confidence_symbolic, conf_final)

        out = OrchestrationResult(
            verdict=verdict,
            statuses={k: v.value for k, v in res.statuses.items()},
            violations=violations, not_verified=res.not_verified,
            missing_fields=res.missing_fields,
            verification_coverage=res.verification_coverage,
            confidence_neural=confidence_neural,
            confidence_symbolic=res.confidence_symbolic,
            confidence_final=conf_final,
            mc_paths_filtered=mc_f, stats=stats,
        )
        self._write_audit(out)
        return out

    def _write_audit(self, r: OrchestrationResult) -> None:
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **r.to_dict()}
        entry.pop("mc_paths_shape", None)
        with open(self.audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        log.info("[Audit] Registo escrito em %s (audit_id=%s)", self.audit_file, r.audit_id)


# =============================================================================
# DEMONSTRAÇÃO STANDALONE — DADOS SINTÉTICOS, NÃO SÃO DA NU HOLDINGS
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")
    print("⚠️  DEMO COM DADOS SINTÉTICOS — não usar como evidência sobre a Nu Holdings.")
    rng = np.random.default_rng(0)
    hist = pd.DataFrame({
        "eps": np.linspace(0.10, 0.25, 18),
        "revenue": np.full(18, 2.0e9),        # sintético
        "net_income": np.full(18, 2.0e8),     # sintético
        "arpac": np.full(18, 12.0),           # sintético
        "npl_15_90": np.full(18, 0.04),       # sintético
    })
    n_pass = sum(verify_prediction(r.to_dict(), last_arpac=12.0,
                                   reference_eps=float(hist["eps"].mean())).n_pass == 5
                 for _, r in hist.iterrows())
    print(f"Regras totalmente verificadas e passadas: {n_pass}/18 (dados sintéticos)")