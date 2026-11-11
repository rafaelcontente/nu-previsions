#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
AUDITORIA INDEPENDENTE — hardcodes, leakage e confiança  (v3)
=============================================================================
Não modifica nada. Três partes:
  A) AST scan CONTEXTUAL. Distingue VALOR DE PREENCHIMENTO de CARDINALIDADE:
     em np.full(18, 2.0e9) o dado é 2.0e9; o 18 é contagem. A v2 marcava
     ambos e gerava dezenas de avisos inúteis.
  B) Invariantes sobre output/metadata.json.
  C) Matemática da confiança em metadata + audit.jsonl.

Uso: python audit.py     Exit: 0 sem falhas, 1 com falhas.
=============================================================================
"""
from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).parent
OK, WARN, FAIL, INFO = [], [], [], []

CONTEMP_FUND = {"revenue", "net_income", "total_assets", "equity", "shares"}
FINANCIAL_FIELDS = ("revenue", "net_income", "eps", "arpac", "npl", "total_assets",
                    "equity", "shares", "market_cap", "ebitda", "gross_profit",
                    "price", "profit")
KNOWN_FABRICATED = {1.82e9, 1.65e8, 16.2, 0.052, 2.0e9, 2.0e8}
FILL_FUNCS = {"full", "ones", "zeros", "full_like"}
PLACEHOLDERS = ("exemplo.com", "example.org", "example.com")
N_RULES = 5
SPEC_N_PATHS = 10000
MIN_N_CALIBRATION = 10


def ok(m):   OK.append(m);   print(f"  ✅ {m}")
def warn(m): WARN.append(m); print(f"  ⚠️  {m}")
def bad(m):  FAIL.append(m); print(f"  ❌ {m}")
def info(m): INFO.append(m); print(f"  ℹ️  {m}")


def fin(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


# =============================================================================
# A) AST SCAN
# =============================================================================
def demo_line_ranges(tree):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "__name__" in ast.dump(node.test) \
                and "__main__" in ast.dump(node.test):
            end = max(getattr(n, "lineno", node.lineno) for n in ast.walk(node))
            out.append((node.lineno, end))
    return out


def fill_value_positions(tree):
    """Posições (linha, coluna) dos VALORES de preenchimento, não das contagens."""
    pos = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fname = node.func.attr if isinstance(node.func, ast.Attribute) else (
            node.func.id if isinstance(node.func, ast.Name) else "")
        if fname in FILL_FUNCS and len(node.args) >= 2:
            a = node.args[1]
            if isinstance(a, ast.Constant) and isinstance(a.value, (int, float)):
                pos.add((a.lineno, a.col_offset))
    return pos


def scan_code():
    print("\n=== A) SCAN DE CÓDIGO (AST contextual) ===")
    files = [p for p in ROOT.rglob("*.py")
             if p.name != "audit.py" and not any(
                 x in p.parts for x in (".cache", "venv", "__pycache__"))]
    if not files:
        bad("nenhum .py encontrado")
        return

    n_prod = n_demo = 0
    seen = set()
    for p in sorted(files):
        src = p.read_text(encoding="utf-8")
        lines = src.splitlines()
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            bad(f"{p.relative_to(ROOT)}: erro de sintaxe — {e}")
            continue
        demos = demo_line_ranges(tree)
        fills = fill_value_positions(tree)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, (int, float)):
                continue
            v = float(node.value)
            key = (str(p), node.lineno, v)
            if key in seen:
                continue
            line = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
            low = line.lower()
            has_fin = any(f in low for f in FINANCIAL_FIELDS)
            is_fill = (node.lineno, node.col_offset) in fills
            known = v in KNOWN_FABRICATED

            # Suspeito: valor conhecido OU (contexto financeiro E (grande OU é fill-value))
            suspicious = known or (has_fin and (abs(v) > 1000 or is_fill))
            if not suspicious:
                continue
            seen.add(key)

            rel = f"{p.relative_to(ROOT)}:{node.lineno}"
            in_demo = any(a <= node.lineno <= b for a, b in demos)
            if in_demo:
                n_demo += 1
                warn(f"{rel} → {node.value} em bloco __main__ (demo sintética declarada)")
            else:
                n_prod += 1
                bad(f"{rel} → {node.value} associado a campo financeiro EM PRODUÇÃO")

    if n_prod == 0:
        ok(f"sem dados financeiros inventados em produção ({n_demo} em demos declaradas)")

    fcast = ROOT / "modulo_2" / "forecasting" / "nu_forecast.py"
    if fcast.exists() and ("10_000" in fcast.read_text(encoding="utf-8")
                           or "10000" in fcast.read_text(encoding="utf-8")):
        info(f"n_paths = {SPEC_N_PATHS} conforme especificação (cardinalidade, não dado)")

    forbidden = {
        "eps_hist.corr": "correlações apresentadas como coeficientes Ridge",
        "1.0 - forecast.mae_val": "confiança neural derivada de MAE em USD",
        "1 - mae_val": "confiança neural derivada de MAE",
    }
    hits = 0
    for p in sorted(files):
        txt = p.read_text(encoding="utf-8")
        for pat, why in forbidden.items():
            if pat in txt:
                bad(f"{p.relative_to(ROOT)}: '{pat}' — {why}")
                hits += 1
    if hits == 0:
        ok("nenhum padrão de batota/falsa confiança")

    for p in sorted(files):
        txt = p.read_text(encoding="utf-8")
        if "nominal.copy()" in txt:
            bad(f"{p.relative_to(ROOT)}: placeholder 'nominal.copy()' — dados falsos "
                f"passados ao relatório")
        if any(ph in txt for ph in PLACEHOLDERS):
            warn(f"{p.relative_to(ROOT)}: USER_AGENT com email placeholder — "
                 f"define SEC_USER_AGENT com contacto real (política SEC)")
        if "_synth_paths" in txt:
            warn(f"{p.relative_to(ROOT)}: síntese de bandas MC presente (fabricação)")


# =============================================================================
# B) INVARIANTES
# =============================================================================
def check_metadata():
    print("\n=== B) INVARIANTES (output/metadata.json) ===")
    mp = ROOT / "output" / "metadata.json"
    if not mp.exists():
        bad("metadata.json ausente — corre `python main.py`")
        return
    m = json.loads(mp.read_text(encoding="utf-8"))

    feats = set(m.get("forecast", {}).get("ridge_features", []))
    leaked = feats & CONTEMP_FUND
    if leaked:
        bad(f"LEAKAGE: fundamentais contemporâneos nas features: {sorted(leaked)}")
    else:
        ok(f"sem leakage contemporâneo ({len(feats)} features, todas desfasadas)")

    sel = m.get("selection", {})
    champ, w = sel.get("champion"), sel.get("weights", {}) or {}
    if champ and champ != "ensemble":
        nz = [k for k, v in w.items() if fin(v) and float(v) > 1e-9]
        if nz == [champ] and abs(float(w.get(champ, 0)) - 1.0) < 1e-9:
            ok(f"campeão '{champ}' aplicado (one-hot) — ensemble descartado")
        else:
            bad(f"campeão='{champ}' mas pesos não são one-hot: {w}")
        if sel.get("stable") is False:
            warn(f"seleção INSTÁVEL (margem {sel.get('margin', 0):.1%}) — declarada")
    elif champ == "ensemble":
        ok("ensemble mantido (superou candidatos em validação)")
    else:
        bad("selection.champion ausente")

    ev = m.get("evaluation", {})
    n_test, n_val = m.get("n_test", 0), m.get("n_val", 0)
    if fin(ev.get("mae_test")):
        ok(f"MAE de teste pontuado: {float(ev['mae_test']):.4f} (n_test={n_test})")
    else:
        bad("MAE de teste NÃO pontuado — holdout desperdiçado")
    tp = m.get("forecast", {}).get("test_predictions", [])
    if n_test and len(tp) != n_test:
        bad(f"test_predictions={len(tp)} ≠ n_test={n_test}")
    elif n_test:
        ok(f"test_predictions alinhado (n={n_test})")

    if fin(ev.get("mae_test")) and fin(ev.get("mae_val")):
        mt, mv = float(ev["mae_test"]), float(ev["mae_val"])
        if mt > mv:
            warn(f"MAE teste ({mt:.4f}) > validação ({mv:.4f}) — degradação fora da amostra")
        elif mt < mv * 0.5:
            warn(f"MAE teste ({mt:.4f}) muito melhor que validação ({mv:.4f}) — "
                 f"com n_test={n_test} é plausível por sorte; NÃO é evidência de skill")

    ct = ev.get("coverage_test_out_of_sample")
    if fin(ct):
        ct = float(ct)
        ok(f"cobertura out-of-sample medida: {ct:.0%}")
        if ct >= 0.999:
            warn(f"cobertura {ct:.0%} com nominal 90% → intervalo SOBRE-CONSERVADOR. "
                 f"100% não é 'bem calibrado'; com n={n_test} não se distingue.")
    else:
        warn("cobertura out-of-sample não medida")

    status = ev.get("interval_status")
    if n_val < MIN_N_CALIBRATION:
        if status == "nominal":
            ok(f"intervalo rotulado 'nominal' (n_val={n_val} < {MIN_N_CALIBRATION})")
        else:
            bad(f"n_val={n_val} mas interval_status='{status}' (devia ser 'nominal')")

    if any("ancora" in str(k) for k in m.get("sources", {})):
        ok(f"âncora oficial EDGAR presente: {m.get('sources')}")
    else:
        warn("âncora anual EDGAR não reportada")

    if m.get("forecast", {}).get("future_features_assumption"):
        ok("pressuposto das features futuras declarado")
    else:
        warn("pressuposto das features futuras NÃO declarado")


# =============================================================================
# C) CONFIANÇA
# =============================================================================
def check_confidence():
    print("\n=== C) CONFIANÇA (metadata + audit.jsonl) ===")
    mp = ROOT / "output" / "metadata.json"
    if not mp.exists():
        bad("metadata.json ausente")
        return
    v = json.loads(mp.read_text(encoding="utf-8")).get("verification", {})

    if v.get("confidence_neural") is None:
        ok("confidence_neural = None (N/D)")
    else:
        bad(f"confidence_neural = {v.get('confidence_neural')} — devia ser None")

    st = v.get("statuses", {}) or {}
    n_pass = sum(1 for s in st.values() if s == "PASS")
    n_fail = sum(1 for s in st.values() if s == "FAIL")
    n_eval = n_pass + n_fail
    exp = n_pass / N_RULES
    cs = v.get("confidence_symbolic")
    if fin(cs) and abs(float(cs) - exp) < 1e-9:
        ok(f"confidence_symbolic = {n_pass}/{N_RULES} = {exp:.2f}")
    else:
        bad(f"confidence_symbolic={cs}, esperado {exp:.2f}")

    cf = v.get("confidence_final")
    if fin(cf) and fin(cs) and abs(float(cf) - float(cs)) < 1e-9:
        ok("confidence_final = confidence_symbolic")
    else:
        bad(f"confidence_final={cf} ≠ symbolic={cs}")

    cov = v.get("verification_coverage")
    if fin(cov) and abs(float(cov) - n_eval / N_RULES) < 1e-9:
        ok(f"verification_coverage = {n_eval}/{N_RULES} = {n_eval/N_RULES:.0%}")
    else:
        bad(f"verification_coverage={cov} inesperado")

    if n_eval:
        info(f"pass-rate entre avaliadas = {n_pass}/{n_eval} = {n_pass/n_eval:.0%} "
             f"(a confiança {exp:.2f} mistura 'não verificável' com 'falhou')")

    nv = v.get("not_verified", []) or []
    if nv:
        warn(f"{len(nv)} regras NÃO VERIFICADAS: {nv} — ausentes: {v.get('missing_fields')}")
        if v.get("verdict") == "ACCEPT":
            bad("ACCEPT com verificação parcial — devia ser PENALIZE")
        else:
            ok(f"veredicto '{v.get('verdict')}' coerente com verificação parcial")
    else:
        ok("todas as regras avaliáveis")

    ap = ROOT / "output" / "audit.jsonl"
    if ap.exists():
        lines = [l for l in ap.read_text(encoding="utf-8").splitlines() if l.strip()]
        ok(f"audit.jsonl: {len(lines)} registos append-only")
        try:
            if json.loads(lines[-1]).get("audit_id") == v.get("audit_id"):
                ok("último registo corresponde à execução atual")
            else:
                warn("último registo não corresponde ao metadata")
        except Exception as e:
            bad(f"audit.jsonl ilegível: {e}")
    else:
        bad("audit.jsonl ausente")


if __name__ == "__main__":
    print("=" * 72)
    print("AUDITORIA v3 — hardcodes, leakage e confiança")
    print("=" * 72)
    scan_code()
    check_metadata()
    check_confidence()
    print("\n" + "=" * 72)
    print(f"RESULTADO: {len(OK)} OK | {len(INFO)} info | {len(WARN)} avisos | {len(FAIL)} falhas")
    print("=" * 72)
    if FAIL:
        print("\nFALHAS:")
        for f in FAIL:
            print(f"  ❌ {f}")
    if WARN:
        print("\nAVISOS:")
        for w in WARN:
            print(f"  ⚠️  {w}")
    sys.exit(1 if FAIL else 0)