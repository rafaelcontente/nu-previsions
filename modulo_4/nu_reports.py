# -*- coding: utf-8 -*-
"""
=============================================================================
MÓDULO 4: EXPLICAÇÃO + RELATÓRIO EXECUTIVO — NU HOLDINGS (NU)
=============================================================================
3 gráficos + 1 parágrafo. Princípios de honestidade:
  * MAE de VALIDAÇÃO e de TESTE reportados separadamente (teste = holdout).
  * Modelo CAMPEÃO identificado. Se não for o Ridge, o gráfico de coeficientes
    é rotulado como descrevendo apenas o candidato Ridge.
  * Coeficientes em unidades de desvio-padrão (features padronizadas), não USD.
  * Calibração: com n < 10 não se desenha nominal-vs-empírico; declara-se que
    não é avaliável. Se não houver cobertura medida, não se plota nada.
  * MC VAZIO → RECUSA desenhar bandas. Não se sintetizam trajetórias:
    uma banda inventada pareceria precisão inexistente.
Referências: Taxon. ML (XAI); Taxonomia V11 (Self-Explaining Systems).
=============================================================================
"""
from __future__ import annotations

import logging
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

log = logging.getLogger("nu_report")

MIN_N_CALIBRATION = 10
SOURCE_NOTE = "Fonte: SEC EDGAR (anual oficial) + Alpha Vantage (trimestres) | Módulos 2-3"


def _fmt(v, spec=".3f", nd="N/D"):
    try:
        return format(float(v), spec) if np.isfinite(float(v)) else nd
    except (TypeError, ValueError):
        return nd


# =============================================================================
# FAN CHART — recusa fabricar bandas
# =============================================================================
def _draw_fan(ax, eps_hist, mc_paths, conf_lower, conf_upper, horizon, point,
              test_pred=None, n_test=0):
    """
    Devolve (p5, p50, p95, matriz_usada, desenhado).
    Se não houver trajetórias MC, NÃO sintetiza: mostra painel informativo e
    devolve desenhado=False (o teste de bandas passa a SKIP, não a PASS falso).
    """
    hist = pd.Series(eps_hist).dropna()
    horizon = max(1, int(horizon))

    if mc_paths is None or np.size(mc_paths) == 0:
        ax.axis("off")
        ax.set_title("Previsão de EPS Trimestral (Nu Holdings)", fontweight="bold")
        ax.text(0.02, 0.92,
                "SEM BANDAS DE INCERTEZA\n\n"
                "O filtro neuro-simbólico descartou todas as trajetórias\n"
                "Monte Carlo. Não se sintetizam bandas: uma banda inventada\n"
                "pareceria precisão inexistente.\n\n"
                f"Previsão pontual T+1: ${_fmt(point)}\n"
                f"Intervalo conformal (nominal): [${_fmt(conf_lower)}, ${_fmt(conf_upper)}]\n\n"
                "Ação: rever os limites das regras hard do Módulo 3.",
                va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", fc="whitesmoke", ec="black"))
        ax.text(1.0, -0.10, SOURCE_NOTE + " | MC VAZIO — bandas não desenhadas",
                transform=ax.transAxes, ha="right", fontsize=6, style="italic")
        z = np.zeros(horizon)
        return z, z, z, np.zeros((1, horizon)), False

    mc = np.asarray(mc_paths, dtype=float)
    horizon = mc.shape[1]
    p5, p50, p95 = (np.percentile(mc, q, axis=0) for q in (5, 50, 95))

    h_idx = np.arange(len(hist))
    f_idx = np.arange(len(hist), len(hist) + horizon)
    ax.plot(h_idx, hist.values, color="black", lw=2, label="Histórico (EPS reportado)")

    if test_pred is not None and n_test > 0 and len(test_pred) == n_test:
        ax.plot(h_idx[-n_test:], np.asarray(test_pred, dtype=float), color="dimgray",
                ls=":", lw=1.8, marker="o", ms=3,
                label=f"Previsão holdout (n={n_test})")

    ax.fill_between(f_idx, p5, p95, color="gray", alpha=0.30, hatch="//",
                    label="Banda P5–P95 (MC filtrado)")
    ax.plot(f_idx, p50, color="black", ls="--", lw=2, label="Mediana (P50)")
    ax.annotate(f"Conformal 90% (nominal)\n[${conf_lower:.3f}, ${conf_upper:.3f}]",
                xy=(f_idx[0], p50[0]), xytext=(f_idx[0] - 0.8, p50[0] + 0.02),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"), fontsize=8)
    ax.set_title("Previsão de EPS Trimestral (Nu Holdings)", fontweight="bold")
    ax.set_xlabel("Trimestre"); ax.set_ylabel("EPS (USD)")
    ax.set_xticks(list(h_idx) + list(f_idx))
    ax.set_xticklabels([f"T-{len(hist)-i}" for i in range(len(hist), 0, -1)]
                       + [f"T+{i+1}" for i in range(horizon)], fontsize=7)
    ax.legend(loc="upper left", frameon=True, edgecolor="black", fontsize=6.5)
    ax.grid(axis="y", ls=":", alpha=0.7)
    ax.text(1.0, -0.16, SOURCE_NOTE, transform=ax.transAxes,
            ha="right", fontsize=6, style="italic")
    return p5, p50, p95, mc, True


# =============================================================================
# COEFICIENTES
# =============================================================================
def _draw_coefs(ax, feature_names, coefficients, n_train=None, selected_model="ensemble"):
    coefs = np.nan_to_num(np.asarray(coefficients, dtype=float), nan=0.0)
    names = list(feature_names)[:len(coefs)]
    if not len(coefs):
        ax.axis("off")
        ax.set_title("Coeficientes Ridge — indisponíveis", fontweight="bold")
        ax.text(0.5, 0.5, "Sem coeficientes para mostrar", ha="center", va="center")
        return
    o = np.argsort(np.abs(coefs))[::-1]
    names, coefs = [names[i] for i in o], coefs[o]
    ax.barh(names, coefs, color=["black" if c >= 0 else "dimgray" for c in coefs],
            edgecolor="black")
    for i, c in enumerate(coefs):
        ax.text(c + (0.002 if c >= 0 else -0.002), i, f"{c:+.3f}", va="center",
                ha="left" if c >= 0 else "right", fontsize=7)
    is_sel = selected_model in ("ridge", "ensemble")
    ax.set_title("Coeficientes Ridge (padronizados)"
                 + ("" if is_sel else " — CANDIDATO, não o modelo usado"),
                 fontweight="bold", fontsize=10)
    ax.set_xlabel("Coeficiente (unidades de desvio-padrão)")
    ax.set_ylabel("Feature")
    ax.axvline(0, color="black", lw=1.2)
    ax.grid(axis="x", ls=":", alpha=0.7)
    ax.tick_params(labelsize=7)
    note = ("Coeficientes exatos do Ridge em features padronizadas — não são efeitos em USD "
            "nem aproximações SHAP.")
    if not is_sel:
        note += (f" Modelo selecionado: '{selected_model}' — estes coeficientes NÃO "
                 f"explicam a previsão publicada.")
    if n_train:
        note += f" Estabilidade limitada (n treino = {n_train})."
    ax.text(1.0, -0.16, note, transform=ax.transAxes, ha="right", fontsize=6, style="italic")


# =============================================================================
# CALIBRAÇÃO — não plota dados inexistentes
# =============================================================================
def _draw_calib(ax, nominal, empirical, n_val, coverage_test=float("nan"), n_test=0):
    ax.set_title("Calibração do Intervalo (Conformal)", fontweight="bold")

    if n_val is None or n_val < MIN_N_CALIBRATION:
        ax.axis("off")
        msg = (f"CALIBRAÇÃO NÃO AVALIÁVEL\n\n"
               f"Conjunto de calibração: n = {n_val if n_val is not None else 'N/D'} "
               f"(mínimo necessário: {MIN_N_CALIBRATION}).\n\n"
               f"O intervalo de 90% é NOMINAL. A garantia de cobertura exige\n"
               f"permutabilidade e um conjunto de calibração suficiente. Qualquer\n"
               f"cobertura medida em validação é in-sample e trivialmente 100%.\n\n")
        if np.isfinite(coverage_test):
            msg += (f"Cobertura OUT-OF-SAMPLE em teste: {coverage_test:.0%} "
                    f"(n={n_test}; q calibrado em validação).\n")
            if coverage_test >= 0.999:
                msg += ("100% com nominal 90% = intervalo SOBRE-CONSERVADOR neste\n"
                        "período, não 'bem calibrado'.\n")
            msg += f"Com n={n_test} não se distingue calibração de sorte."
        else:
            msg += "Sem conjunto de teste pontuado — nenhuma validação out-of-sample."
        ax.text(0.02, 0.95, msg, va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.5", fc="whitesmoke", ec="black"))
        return False

    if empirical is None:
        ax.axis("off")
        ax.text(0.02, 0.95,
                "SEM COBERTURA EMPÍRICA MEDIDA\n\nNão há dados para plotar. "
                "Não se preenche com valores nominais.",
                va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", fc="whitesmoke", ec="black"))
        return False

    emp = np.asarray(empirical, dtype=float)
    nom = np.asarray(nominal, dtype=float)
    ax.plot([0.75, 1.0], [0.75, 1.0], "k--", lw=2, label="Calibração perfeita")
    ax.scatter(nom, emp, color="black", s=50, zorder=5, edgecolor="white")
    for x, y in zip(nom, emp):
        ax.annotate(f"{y:.0%}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_xlabel("Cobertura nominal"); ax.set_ylabel("Cobertura empírica")
    ax.set_xlim(0.75, 1.0); ax.set_ylim(0.75, 1.0)
    ax.grid(ls=":", alpha=0.7); ax.legend(loc="lower right", fontsize=7)
    ax.text(1.0, -0.16, f"Nota: cobertura medida em validação (n={n_val}).",
            transform=ax.transAxes, ha="right", fontsize=6, style="italic")
    return True


# =============================================================================
# PARÁGRAFO
# =============================================================================
def generate_summary_paragraph(point, mae, lower, upper, driver_name, driver_coef,
                               neuro, n_val=None, n_train=None,
                               mae_test=float("nan"), n_test=0,
                               selected_model="ensemble", selection_reason="",
                               coverage_test=float("nan")) -> str:
    st = neuro.get("stats", {})
    nh = len(neuro.get("violations_hard", []))
    ns = len(neuro.get("violations_soft", []))
    nv = neuro.get("not_verified", [])
    missing = neuro.get("missing_fields", [])
    cov = neuro.get("verification_coverage", 0.0)

    cal_note = ("nominal, não validado" if (n_val is None or n_val < MIN_N_CALIBRATION)
                else "validado em validação")
    nv_txt = (f"{len(nv)} NÃO VERIFICADAS (campos ausentes: {', '.join(missing) or 'n/d'})"
              if nv else "0 não verificadas")
    test_txt = (f"MAE de teste (holdout, n={n_test}): ${_fmt(mae_test)}"
                if np.isfinite(mae_test) else "MAE de teste: N/D (holdout não pontuado)")
    if np.isfinite(coverage_test):
        covt = f"cobertura out-of-sample em teste: {coverage_test:.0%}"
        if coverage_test >= 0.999:
            covt += " (sobre-conservador, não 'calibrado')"
    else:
        covt = "cobertura out-of-sample: N/D"

    return (
        f"Previsão: EPS de ${_fmt(point)} (T+1), modelo campeão '{selected_model}'. "
        f"MAE de validação: ${_fmt(mae)} (n_val={n_val if n_val is not None else 'N/D'}). "
        f"{test_txt}. "
        f"Intervalo conformal 90%: [${_fmt(lower)}, ${_fmt(upper)}] — {cal_note}; {covt}. "
        f"Driver principal (Ridge): {driver_name} (coef. padronizado {_fmt(driver_coef, '+.3f')}). "
        f"Verificação simbólica: 5 regras, {int(round(cov*5))} avaliadas ({cov:.0%} cobertura), "
        f"{nh} violações hard, {ns} soft, {nv_txt}. "
        f"Trajetórias MC: {st.get('n_discarded', 0)} de {st.get('n_input', 0)} descartadas. "
        f"Confiança de verificação: {_fmt(neuro.get('confidence_final'), '.2f')} "
        f"(neural: N/D — não é probabilidade de acerto). "
        f"Veredicto: {neuro.get('verdict', 'N/D')}. "
        f"Seleção: {selection_reason}"
    )


# =============================================================================
# RELATÓRIO
# =============================================================================
def generate_executive_report(eps_hist, mc_paths_filtered, feature_names, coefficients,
                              nominal_coverages, empirical_coverages, point_forecast,
                              mae_val, conf_lower, conf_upper, neuro_result,
                              output_dir="reports", horizon=4,
                              n_val=None, n_train=None,
                              mae_test=float("nan"), n_test=0,
                              selected_model="ensemble", selection_reason="",
                              coverage_test=float("nan"), test_forecast=None):
    os.makedirs(output_dir, exist_ok=True)
    coefs = np.nan_to_num(np.asarray(coefficients, dtype=float), nan=0.0)
    idx = int(np.argmax(np.abs(coefs))) if len(coefs) else 0
    driver = feature_names[idx] if len(feature_names) > idx else "n/d"
    driver_coef = float(coefs[idx]) if len(coefs) else float("nan")

    summary = generate_summary_paragraph(
        point_forecast, mae_val, conf_lower, conf_upper, driver, driver_coef,
        neuro_result, n_val, n_train, mae_test, n_test,
        selected_model, selection_reason, coverage_test)

    # --- PNGs individuais ---
    fig1, a = plt.subplots(figsize=(8, 4))
    _draw_fan(a, eps_hist, mc_paths_filtered, conf_lower, conf_upper, horizon,
              point_forecast, test_forecast, n_test)
    plt.tight_layout()
    fig2, a2 = plt.subplots(figsize=(8, 4))
    _draw_coefs(a2, feature_names, coefs, n_train, selected_model)
    plt.tight_layout()
    fig3, a3 = plt.subplots(figsize=(7, 4))
    _draw_calib(a3, nominal_coverages, empirical_coverages, n_val, coverage_test, n_test)
    plt.tight_layout()

    pngs = []
    for fig, name in ((fig1, "fan_chart"), (fig2, "coefficients"), (fig3, "calibration")):
        p = os.path.join(output_dir, f"nu_eps_{name}.png")
        fig.savefig(p, dpi=200, bbox_inches="tight"); pngs.append(p); plt.close(fig)

    # --- PDF A4 ---
    pdf_path = os.path.join(output_dir, "nu_eps_forecast.pdf")
    fig = plt.figure(figsize=(8.27, 11.69))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 0.9, 0.9, 1.15], hspace=0.6)
    a1, a2, a3, a4 = (fig.add_subplot(gs[i]) for i in range(4))
    p5, p50, p95, used, fan_drawn = _draw_fan(
        a1, eps_hist, mc_paths_filtered, conf_lower, conf_upper,
        horizon, point_forecast, test_forecast, n_test)
    _draw_coefs(a2, feature_names, coefs, n_train, selected_model)
    calib_drawn = _draw_calib(a3, nominal_coverages, empirical_coverages, n_val,
                              coverage_test, n_test)
    a4.axis("off")
    a4.text(0.0, 1.00, "RESUMO EXECUTIVO — COMITÉ DE INVESTIMENTO",
            fontweight="bold", fontsize=11, va="top")
    a4.text(0.0, 0.90, "\n".join(textwrap.wrap(summary, width=95)),
            fontsize=7.8, va="top", linespacing=1.5)
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # --- Testes ---
    t1 = bool(a1.get_title() and (a1.get_xlabel() or not fan_drawn))
    if fan_drawn:
        e5, e50, e95 = (np.percentile(used, q, axis=0) for q in (5, 50, 95))
        t2 = bool(np.allclose(p5, e5) and np.allclose(p50, e50) and np.allclose(p95, e95))
        m2 = "PASS — bandas = percentis das trajetórias usadas" if t2 else "FAIL — bandas divergem"
    else:
        t2, m2 = True, "SKIP — MC vazio, bandas não desenhadas (não se fabricam)"
    t3 = all(m in summary for m in ["Previsão:", "MAE", "Intervalo", "Verificação simbólica:"])
    log.info("[Teste 1] %s — legibilidade", "PASS" if t1 else "FAIL")
    log.info("[Teste 2] %s", m2)
    log.info("[Teste 3] %s — parágrafo com marcadores obrigatórios", "PASS" if t3 else "FAIL")
    log.info("[Relatório] Campeão: %s | MAE val=%s | MAE teste=%s | fan=%s | calibração=%s",
             selected_model, _fmt(mae_val, ".4f"), _fmt(mae_test, ".4f"),
             fan_drawn, calib_drawn)
    log.info("✓ Relatório gerado: %s", pdf_path)
    return pdf_path, pngs