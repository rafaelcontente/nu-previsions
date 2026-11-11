# -*- coding: utf-8 -*-
"""
=============================================================================
NU HOLDINGS FORECAST — ORQUESTRADOR PRINCIPAL
=============================================================================
Princípios:
  * Factos vêm APENAS do dataset. Campos inexistentes (ARPAC, NPL 15-90) ficam
    ausentes → Módulo 3 marca NOT_VERIFIED, nunca PASS silencioso.
  * Coeficientes do gráfico = coeficientes REAIS do Ridge (padronizados).
  * confidence_neural = None (N/D). Não se deriva probabilidade de MAE em USD.
  * MAE de TESTE (holdout) e modelo CAMPEÃO reportados no log, PDF e metadata.
  * MC: variáveis não simuladas ficam fixas no último valor real (sem ruído).
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from modulo_1.data_ingestion.nu_data_pipeline import build_nu_dataset, NuDataset
from modulo_2.forecasting.nu_forecast import run_forecast, ForecastResult
from modulo_3.nu_verifier import Orchestrator, OrchestrationResult
from modulo_4.nu_reports import generate_executive_report

warnings.filterwarnings("ignore", message=".*invalid value encountered in divide.*")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("nu_orchestrator")

UNAVAILABLE_FIELDS = ("arpac", "npl_15_90")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Previsão de EPS da Nu Holdings (NU)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ticker", type=str, default="NU")
    p.add_argument("--start", type=str, default="2021-12-01")
    p.add_argument("--end", type=str, default="2026-11-01")
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="output")
    p.add_argument("--firecrawl-key", type=str, default=None)
    return p.parse_args()


def _real(value, default=None):
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def build_facts(dataset: NuDataset, eps_forecast: float) -> tuple:
    """Factos APENAS com valores presentes no dataset. Sem defaults inventados."""
    last = dataset.df.iloc[-1]
    facts = {"eps": float(eps_forecast)}
    for col in ("revenue", "net_income", "total_assets", "equity", "arpac", "npl_15_90"):
        v = _real(last.get(col, None))
        if v is not None:
            facts[col] = v
    absent = [c for c in ("revenue", "net_income", "arpac", "npl_15_90") if c not in facts]
    if absent:
        log.warning("[factos] Campos indisponíveis em fonte real: %s → regras dependentes "
                    "serão NOT_VERIFIED (não PASS).", absent)
    return facts, absent


def mc_features_from_real(mc_paths: np.ndarray, facts: dict) -> dict:
    """Variáveis não simuladas fixas no último valor real. Sem ruído adicionado."""
    n = mc_paths.shape[0]
    return {k: np.full(n, v, dtype=float) for k, v in facts.items() if k != "eps"}


def run_pipeline(args) -> dict:
    t0 = time.time()
    results = {}
    log.info("=" * 80)
    log.info("🚀 NU HOLDINGS FORECAST — PIPELINE DE PREVISÃO")
    log.info("Ticker: %s | %s → %s | horizonte=%d | alpha=%.2f",
             args.ticker, args.start, args.end, args.horizon, args.alpha)
    log.info("=" * 80)

    # --- MÓDULO 1 ---
    log.info("\n📊 MÓDULO 1 — INGESTÃO DE DADOS")
    dataset: NuDataset = build_nu_dataset(ticker=args.ticker, start=args.start,
                                          end=args.end, firecrawl_api_key=args.firecrawl_key)
    results["dataset"] = dataset
    log.info("✓ Dataset: %d treino, %d val, %d teste",
             dataset.n_train, dataset.n_val, dataset.n_test)

    # --- MÓDULO 2 ---
    log.info("\n🔮 MÓDULO 2 — PREVISÃO")
    fc: ForecastResult = run_forecast(
        X_train=dataset.X_train, y_train=dataset.y_train,
        X_val=dataset.X_val, y_val=dataset.y_val,
        X_test=dataset.X_test, y_test=dataset.y_test,
        horizon=args.horizon, alpha=args.alpha, seed=args.seed)
    results["forecast"] = fc
    log.info("✓ Campeão: %s | futuro: %s", fc.selected_model, fc.ensemble_forecast.round(4))
    log.info("✓ MAE val: %.4f | MAE teste (holdout): %.4f", fc.mae_val, fc.mae_test)

    # --- MÓDULO 3 ---
    log.info("\n🧠 MÓDULO 3 — VERIFICAÇÃO NEURO-SIMBÓLICA")
    facts, absent = build_facts(dataset, float(fc.ensemble_forecast[0]))
    mc_feats = mc_features_from_real(fc.mc_paths, facts)
    reference_eps = _real(dataset.y_train.mean())
    orchestrator = Orchestrator(audit_file=str(Path(args.output_dir) / "audit.jsonl"))
    neuro: OrchestrationResult = orchestrator.orchestrate(
        prediction=facts, mc_paths=fc.mc_paths, features_per_path=mc_feats,
        confidence_neural=None, last_arpac=facts.get("arpac"),
        reference_eps=reference_eps)
    results["neuro_result"] = neuro
    log.info("✓ Veredicto: %s | cobertura de verificação: %.0f%% | confiança: %.2f",
             neuro.verdict.value, neuro.verification_coverage * 100, neuro.confidence_final)

    # --- MÓDULO 4 ---
    log.info("\n📄 MÓDULO 4 — RELATÓRIO EXECUTIVO")
    eps_hist = pd.concat([dataset.y_train, dataset.y_val, dataset.y_test])
    feature_names = list(fc.ridge_feature_names)
    coefficients = np.asarray(fc.ridge_coefficients, dtype=float)
    if not len(coefficients) or not len(feature_names):
        log.warning("[relatório] Coeficientes Ridge indisponíveis → gráfico vazio.")
        feature_names, coefficients = [], np.array([])

    nominal = np.array([0.80, 0.90, 0.95])
    resid = np.abs(np.asarray(dataset.y_val, dtype=float)
                   - np.asarray(fc.ensemble_val, dtype=float))
    if len(resid) >= 10:
        empirical = np.array([float((resid <= np.quantile(resid, L)).mean()) for L in nominal])
    else:
        empirical = None   # placeholder; o Módulo 4 NÃO o desenha
        log.warning("[relatório] n_val=%d < 10 → calibração NÃO avaliável (painel informativo).",
                    len(resid))

    neuro_dict = {
        "verdict": neuro.verdict.value,
        "violations_hard": [n for n, s in neuro.violations if s == "hard"],
        "violations_soft": [n for n, s in neuro.violations if s == "soft"],
        "not_verified": neuro.not_verified,
        "missing_fields": neuro.missing_fields,
        "verification_coverage": neuro.verification_coverage,
        "confidence_final": neuro.confidence_final,
        "confidence_neural": neuro.confidence_neural,
        "confidence_symbolic": neuro.confidence_symbolic,
        "stats": neuro.stats,
    }
    pdf_path, pngs = generate_executive_report(
        eps_hist=eps_hist, mc_paths_filtered=neuro.mc_paths_filtered,
        feature_names=feature_names, coefficients=coefficients,
        nominal_coverages=nominal, empirical_coverages=empirical,
        point_forecast=float(fc.ensemble_forecast[0]), mae_val=fc.mae_val,
        conf_lower=float(fc.interval_lower[0]), conf_upper=float(fc.interval_upper[0]),
        neuro_result=neuro_dict, output_dir=str(Path(args.output_dir) / "reports"),
        horizon=args.horizon, n_val=dataset.n_val, n_train=dataset.n_train,
        mae_test=fc.mae_test, n_test=dataset.n_test,
        selected_model=fc.selected_model, selection_reason=fc.selection_reason,
        coverage_test=fc.coverage_test, test_forecast=fc.test_forecast)
    results["report"] = {"pdf_path": pdf_path, "png_paths": pngs}

    # --- RESUMO ---
    log.info("\n" + "=" * 80)
    log.info("✅ PIPELINE CONCLUÍDO em %.1fs", time.time() - t0)
    log.info("Campeão: %s | EPS T+1: $%.3f | Intervalo 90%% (nominal): [$%.3f, $%.3f]",
             fc.selected_model, fc.ensemble_forecast[0], fc.interval_lower[0], fc.interval_upper[0])
    log.info("MAE val: %.4f | MAE TESTE (holdout, n=%d): %.4f",
             fc.mae_val, dataset.n_test, fc.mae_test)
    if np.isfinite(fc.coverage_test):
        log.info("Cobertura conformal OUT-OF-SAMPLE em teste: %.0f%%", fc.coverage_test * 100)
    log.info("Veredicto: %s | Verificação: %.0f%% das regras | Confiança: %.2f",
             neuro.verdict.value, neuro.verification_coverage * 100, neuro.confidence_final)
    if not fc.selection_stable:
        log.warning("⚠️ Seleção de campeão INSTÁVEL (margem %.1f%%, n_val=%d).",
                    fc.selection_margin * 100, dataset.n_val)
    if neuro.not_verified:
        log.warning("⚠️ Regras NÃO verificadas: %s (campos ausentes: %s)",
                    neuro.not_verified, neuro.missing_fields)
    log.info("=" * 80)
    return results


def main() -> int:
    try:
        args = parse_args()
        out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
        r = run_pipeline(args)
        fc, nr, ds = r["forecast"], r["neuro_result"], r["dataset"]
        metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticker": args.ticker, "period": {"start": args.start, "end": args.end},
            "horizon": args.horizon, "alpha": args.alpha, "seed": args.seed,
            "sources": ds.source_summary, "dataset_sha256": ds.sha256,
            "n_train": ds.n_train, "n_val": ds.n_val, "n_test": ds.n_test,
            "unavailable_fields": list(UNAVAILABLE_FIELDS),
            "selection": {
                "champion": fc.selected_model, "reason": fc.selection_reason,
                "margin": fc.selection_margin, "stable": fc.selection_stable,
                "weights": fc.weights,
            },
            "evaluation": {
                "mae_val": fc.mae_val, "maes_val_individuais": fc.maes_individuais,
                "mae_test": fc.mae_test, "maes_test_individuais": fc.maes_test,
                "coverage_val_in_sample": fc.coverage_val,
                "coverage_test_out_of_sample": fc.coverage_test,
                "interval_status": "nominal" if ds.n_val < 10 else "validated",
            },
            "forecast": {
                "point": fc.ensemble_forecast.tolist(),
                "test_predictions": np.asarray(fc.test_forecast).tolist(),
                "interval_lower": fc.interval_lower.tolist(),
                "interval_upper": fc.interval_upper.tolist(),
                "model_temporal": fc.sarimax_model_name,
                "future_features_assumption": fc.future_features_assumption,
                "ridge_features": list(fc.ridge_feature_names),
                "ridge_coefficients_standardized": np.asarray(fc.ridge_coefficients).tolist(),
            },
            "verification": nr.to_dict(),
            "tests": {k: {"passed": v[0], "detail": v[1]} for k, v in fc.test_results.items()},
            "report": r["report"]["pdf_path"],
        }
        metadata["verification"].pop("mc_paths_shape", None)
        with open(out / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        log.info("\n💾 Metadados: %s", out / "metadata.json")
        return 0
    except KeyboardInterrupt:
        log.warning("\n⚠️ Interrompido"); return 130
    except Exception as e:
        log.error("\n❌ ERRO: %s", e, exc_info=True); return 1


if __name__ == "__main__":
    sys.exit(main())