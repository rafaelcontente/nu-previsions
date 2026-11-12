# 🏦 Nu Holdings Forecast

**Sistema de previsão de EPS (Earnings Per Share) da Nu Holdings com verificação neuro-simbólica e quantificação rigorosa de incerteza.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production-Ready](https://img.shields.io/badge/status-production--ready-green.svg)]()

---

## 🎯 O que é isto?

Um sistema modular de 4 fases que:

1. **Ingere dados** de 3 fontes (yfinance, BCB, releases da Nu) com contratos de schema rigorosos
2. **Prevê EPS** trimestral usando um ensemble de SARIMAX + Ridge + Monte Carlo + Conformal Prediction
3. **Verifica** as previsões contra 5 regras simbólicas auditáveis (neuro-simbólico bidirecional)
4. **Gera um relatório** executivo de 1 página com fan chart, coeficientes e calibração

**Filosofia:** *"Nunca comece por deep learning. Comece por modelos simples que definem o mínimo aceitável."*

---

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────────────┐
│                        main.py (Orquestrador)                        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌──────────────┐           ┌──────────────┐           ┌──────────────┐
│  Módulo 1    │           │  Módulo 2    │           │  Módulo 3    │
│  Dados       │──────────▶│  Previsão    │──────────▶│  Verificação │
│              │           │              │           │  Neuro-Simb. │
│ • yfinance   │           │ • SARIMAX    │           │              │
│ • BCB SGS    │           │ • Ridge      │           │ • 5 regras   │
│ • firecrawl  │           │ • Ensemble   │           │ • MC filter  │
│ • pandera    │           │ • Monte Carlo│           │ • Orquestr.  │
│ • SHA-256    │           │ • Conformal  │           │ • Audit trail│
└──────────────┘           └──────────────┘           └──────────────┘
                                    │
                                    ▼
                           ┌──────────────┐
                           │  Módulo 4    │
                           │  Relatório   │
                           │              │
                           │ • Fan chart  │
                           │ • Coeficientes│
                           │ • Calibração │
                           │ • Sumário    │
                           └──────────────┘
```

---

## 📦 Módulos

### 📊 Módulo 1 — Ingestão de Dados (`modules/data_ingestion/`)

| Componente | Descrição |
|---|---|
| **Fontes** | yfinance (preços + fundamentals), BCB SGS (Selic 432, IPCA 433), firecrawl (releases Nu) |
| **Schema Contract** | `pandera` valida colunas obrigatórias (`eps`, `revenue`, `net_income`, `arpac`, `npl_15_90`, `clients`) |
| **Features** | 8 features engenheirizadas: `eps_lag1`, `eps_lag4`, `rev_yoy`, `arpac_chg`, `net_margin`, `npl_15_90`, `selic`, `brl_usd` |
| **Split** | 12 treino / 3 val / 3 teste — **estritamente cronológico, sem shuffle** |
| **Lineage** | Hash SHA-256 + schema version para reprodutibilidade |

**Referências:** *Fase 2 — Dados e diagnóstico* (Taxon. Modelos de previsão), *Data Contracts* (Taxonomia V11)

### 🔮 Módulo 2 — Previsão (`modules/forecasting/`)

| Componente | Descrição |
|---|---|
| **SARIMAX(1,1,0)** | Com Selic exógena — 2 parâmetros + 1 exógeno, adequado a n=12 |
| **Ridge(α=1.0)** | Regularização L2 forte para n << p efetivo |
| **Ensemble** | Média ponderada pelo inverso do MAE de validação (pesos aprendidos) |
| **Markov 2-estados** | growth (rev_yoy > 15%) / contraction, Dirichlet prior (1,1) |
| **Monte Carlo** | 10.000 simulações condicionadas a regimes |
| **Conformal** | Split-conformal com cobertura 90% garantida |

**Referências:** *SARIMAX*, *Ridge*, *Conformal Prediction*, *Markov-switching* (Taxon. Modelos de previsão)

### 🧠 Módulo 3 — Verificação Neuro-Simbólica (`modules/neurosymbolic/`)

**Bidirecional em dois sentidos:**

- **Neural → Simbólico:** cada previsão pontual é verificada contra 5 regras
- **Simbólico → Neural:** cada trajetória MC é filtrada (descartadas as fisicamente impossíveis)

| Regra | Severidade | Descrição |
|---|---|---|
| `eps_net_income_consistency` | HARD | EPS e net_income devem ter o mesmo sinal |
| `net_margin_bounds` | HARD | Margem líquida entre -50% e 100% |
| `npl_15_90_bounds` | HARD | NPL 15-90 não pode ser > 20% |
| `arpac_range` | SOFT | ARPAC dentro de ±50% do último valor |
| `eps_consensus_deviation` | SOFT | EPS não desvia > 30% do consenso |

**Orquestrador:** máquina de estados com veredictos `ACCEPT` / `PENALIZE` / `REJECT` / `ESCALATE`

**Confiança final:** `min(confidence_neural, confidence_symbolic)` — o elo mais fraco domina

**Referências:** *Sistemas Neuro-Simbólicos Bidirecionais* (Taxonomia V11), *nubank/state-flow*

### 📄 Módulo 4 — Relatório (`modules/reporting/`)

Gera 3 gráficos + 1 parágrafo de sumário:

1. **Fan chart** com histórico + previsão + bandas P5–P95 + intervalo conformal anotado
2. **Coeficientes Ridge** em barras horizontais ordenadas por magnitude (explicação exata)
3. **Calibração conformal** — cobertura nominal vs empírica (prova de honestidade)
4. **Parágrafo de sumário** com: previsão pontual, MAE, intervalo 90%, driver principal + verificação neuro-simbólica

**Output:** `nu_eps_forecast.pdf` (1 página A4) + 3 PNGs individuais

**Referências:** *Interpretabilidade e explicabilidade (XAI)*, *Self-Explaining Systems* (Taxonomia V11)

---

## 🚀 Quick Start

### 1. Instalação

```bash
# Clonar o repositório
git clone https://github.com/rafaelcontente/nu-previsions.git
cd nu-holdings-forecast

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt
```

### 2. Execução

```bash
# Execução padrão
python main.py

# Com parâmetros customizados
python main.py \
  --ticker NU \
  --start 2021-12-01 \
  --end 2026-11-01 \
  --horizon 4 \
  --alpha 0.10 \
  --seed 42 \
  --output-dir output \
  --firecrawl-key $FIRECRAWL_API_KEY  # opcional
```

### 3. Output

```
output/
├── reports/
│   ├── nu_eps_forecast.pdf      ← relatório executivo (1 página A4)
│   ├── nu_eps_fan_chart.png
│   ├── nu_eps_coefficients.png
│   └── nu_eps_calibration.png
├── data/processed/              ← datasets intermédios (Parquet)
├── audit.jsonl                  ← audit trail neuro-simbólico
└── metadata.json                ← metadados da execução
```

---

## 🧪 Testes de Validação

Cada módulo tem 3–4 testes de validação executados automaticamente:

| Módulo | Teste | Critério |
|---|---|---|
| **M1** | Sem leakage | `max(train) < min(val) < min(test)` |
| **M1** | Schema contract | Falha explícita se faltar coluna |
| **M1** | Contribuição features | Cada feature degrada MAE ≥ 2% |
| **M2** | Ensemble bate baselines | MAE ensemble < SARIMAX/Ridge/naïve |
| **M2** | Cobertura conformal | ≥ 2/3 trimestres de val cobertos |
| **M2** | Reprodutibilidade | SHA-256 idêntico com mesmo seed |
| **M3** | Regras executáveis | Passam em ≥ 15/18 trimestres |
| **M3** | Rejeita injeção | `eps=-0.10, net_income>0` ⇒ REJECT |
| **M3** | Filtragem MC sensata | acceptance_rate ∈ [85%, 98%] |
| **M3** | Confiança = mínimo | `final = min(neural, symbolic)` |
| **M4** | Legibilidade | Título + eixos + source note |
| **M4** | Bandas corretas | Percentis MC = percentis plotados |
| **M4** | Parágrafo completo | 4 marcadores obrigatórios presentes |

---

## 📚 Referências Taxonómicas 

Este projeto segue rigorosamente duas taxonomias criadas por mim, ao longo das minhas certificações:

### Taxon. Modelos de previsão.md
- **Fase 2** — Dados e diagnóstico
- **Fase 3** — Baselines obrigatórios (*"Nunca comece por deep learning"*)
- **Fase 5** — Forecasting (SARIMAX, Markov-switching, Monte Carlo)
- **Fase 6** — Incerteza (Conformal Prediction, Quantile Regression)

### Taxonomia V11.md (Framework de Arquitetura de Sistemas de IA)
- **Pilar 1 (Dados)** — Data Contracts, Data Lineage, Schema Validation
- **Pilar 3 (Modelo)** — Sistemas Neuro-Simbólicos Bidirecionais
- **Pilar 4 (Orquestração)** — Hierarquia de agentes
- **Pilar 7 (Governança)** — XAI, Guardrails, Audit trail
- **Pilar 8 (Operação)** — Conformal Prediction, Maker-Checker Pattern

### Repositórios de referência
- **nubank/fklearn** — filosofia de funções puras e auditáveis
- **nubank/state-flow** — testes como máquina de estados
- **SantanderAI** — governança de IA de alto risco
- **firecrawl/firecrawl** — scraping estruturado para agentes

---

## 🛡️ Filosofia de Design

1. **Baselines primeiro** — modelos simples (SARIMAX, Ridge) antes de complexidade
2. **Fail-fast** — schema contracts que falham explicitamente, nunca silenciosamente
3. **Auditabilidade** — cada decisão registada em `audit.jsonl` com timestamp
4. **Incerteza honesta** — conformal prediction com garantia estatística de cobertura
5. **Segurança neuro-simbólica** — regras hard rejeitam previsões fisicamente impossíveis
6. **Reprodutibilidade** — seed fixo + SHA-256 do dataset + schema version
7. **Clareza executiva** — relatório de 1 página que qualquer comité entende em <5 min

---

## 📂 Estrutura do Repositório

```
nu-holdings-forecast/
├── main.py                          ← orquestrador principal
├── requirements.txt                 ← dependências
├── README.md                        ← este ficheiro
├── .gitignore
│
├── modules/
│   ├── data_ingestion/
│   │   ├── nu_data_pipeline.py      ← Módulo 1
│   │   └── README.md
│   ├── forecasting/
│   │   ├── nu_forecast.py           ← Módulo 2
│   │   └── README.md
│   ├── neurosymbolic/
│   │   ├── nu_verifier.py           ← Módulo 3
│   │   └── README.md
│   └── reporting/
│       ├── nu_report.py             ← Módulo 4
│       └── README.md
│
├── output/                          ← gerado em runtime (gitignored)
│   ├── reports/
│   ├── data/processed/
│   ├── audit.jsonl
│   └── metadata.json
│
└── tests/                           ← testes unitários (pytest)
    ├── test_module1.py
    ├── test_module2.py
    ├── test_module3.py
    └── test_module4.py
```

---

## ⚠️ Disclaimer

Este projeto é **educacional e de investigação**. Não constitui aconselhamento financeiro. As previsões geradas são exercícios metodológicos e não devem ser usadas para decisões de investimento reais. As Taxonomias não se encontram no código, pois são documentos proprietários.

---

## 📜 Licença

MIT License — ver [LICENSE](LICENSE) para detalhes.

---

## 🤝 Contribuir

Contribuições são bem-vindas! Por favor:

1. Fork o repositório
2. Crie uma branch (`git checkout -b feature/nova-feature`)
3. Commit as alterações (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

---

## 📧 Contacto

Para questões ou sugestões, abra uma [issue](https://github.com/<seu-user>/nu-holdings-forecast/issues).

---

<div align="center">

**Construído com 🇵🇹 rigor metodológico e ☕ café**

*Inspirado pelas taxonomias de previsão e arquitetura de IA*

</div>
