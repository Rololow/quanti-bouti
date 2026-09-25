# quanti-bouti

# Real-Time Adaptive Quantitative Portfolio Engine

> **Real-time, multi-horizon, adaptive portfolio monitoring and decision engine combining quantitative market signals, online learning, regime detection, fundamentals, news and structured AI extraction.**

---

## Démarrage rapide

```bash
pip install -e ".[dev]"

# Lance la boucle Core sur un flux simulé (config/config.yaml)
python -m trading_engine.main

# Tests
pytest
```

État actuel : Phase 1 (Core) + volatilité EWMA et barres 5m / 1h / 1d, alimentées
par un flux simulé déterministe. Le flux Alpaca WebSocket (Phase 2) se branchera
derrière la même interface `MarketFeed`.

---

## 1. Vision

L'objectif du projet est de construire un **moteur quantitatif de monitoring et de décision de portefeuille en temps réel**.

Le système suit en permanence :

* chaque position ;
* son exposition ;
* son PnL ;
* sa volatilité ;
* sa contribution au risque ;
* son momentum ;
* ses signaux multi-horizon ;
* le régime de marché ;
* les données fondamentales ;
* les événements/news ;
* les changements de corrélation ;
* l'écart entre allocation actuelle et allocation cible.

Le système doit pouvoir répondre à tout moment à :

> **"Quelle est la situation actuelle de chaque position, quel risque représente-t-elle, pourquoi sa cible devrait-elle changer et quels événements ont provoqué cette évolution ?"**

Le système est conçu autour de **l'apprentissage online**.

Le backtesting n'est donc **pas le moteur principal** du système. Il reste un outil de validation et d'analyse historique.

---

# 2. Philosophie générale

Architecture principale :

```text
                         REAL-TIME DATA
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
        Market Data       Fundamentals         News
             │                 │                 │
             ▼                 ▼                 ▼
       Feature Engine    Fundamental Engine   AI/NLP Engine
             │                 │                 │
             └─────────────────┼─────────────────┘
                               ▼
                      ONLINE LEARNING
                               │
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
           HMM             Volatility        Predictive
         Regimes             Models            Models
             │                 │                 │
             └─────────────────┼─────────────────┘
                               ▼
                      SIGNAL FUSION
                               │
                               ▼
                       RISK ENGINE
                               │
                               ▼
                     TARGET ALLOCATION
                               │
                               ▼
                     DECISION ENGINE
                               │
                  ┌────────────┴────────────┐
                  ▼                         ▼
              Dashboard                  Alerts
                  │
                  ▼
              Human / Execution
```

---

# 3. Principe fondamental : monitoring ≠ trading automatique

Le système ne doit pas directement transformer un signal en ordre.

Pipeline :

```text
SIGNAL
   ↓
TARGET POSITION
   ↓
RISK CHECK
   ↓
APPROVE / REJECT
   ↓
TRADE PROPOSAL
   ↓
HUMAN / EXECUTION
```

Un `TRADE_PROPOSAL` est une recommandation technique du moteur, pas nécessairement un ordre envoyé au broker.

Cela permet de développer et valider tout le système sans risque d'exécution involontaire.

---

# 4. Real-Time Market Data

## 4.1 Fournisseur initial

Le fournisseur initial prévu est **Alpaca**.

Le flux WebSocket peut fournir notamment :

* trades ;
* quotes ;
* bars ;
* news.

Architecture :

```text
Alpaca WebSocket
       │
       ▼
 MarketDataFeed
       │
       ▼
    EventBus
```

L'utilisation d'un WebSocket est privilégiée à du polling périodique.

---

## 4.2 Types d'événements

Le système doit manipuler des événements génériques :

```python
MarketEvent
TradeEvent
QuoteEvent
BarEvent
NewsEvent
FundamentalEvent
PortfolioEvent
RiskEvent
DecisionEvent
```

Chaque événement doit posséder au minimum :

```python
timestamp
symbol
event_type
source
payload
```

---

# 5. Event Bus

Le `EventBus` est le cœur logiciel permettant de découpler les différents modules.

Exemple :

```text
MarketDataFeed
      │
      ▼
   EventBus
      │
      ├── Feature Engine
      ├── Portfolio Engine
      ├── Risk Engine
      ├── News Engine
      ├── HMM
      └── Storage
```

Les composants ne doivent pas directement dépendre les uns des autres.

Exemple :

```python
await event_bus.publish(event)
```

Puis :

```python
@event_bus.subscribe("trade")
async def on_trade(event):
    ...
```

Cela permet d'ajouter ultérieurement :

* nouvelles sources de données ;
* nouveaux modèles ;
* nouveaux brokers ;
* nouveaux dashboards ;

sans réécrire le moteur central.

---

# 6. Multi-Time-Horizon Architecture

Le système doit analyser plusieurs horizons simultanément.

## High Frequency / Fast

Exemples :

```text
tick
5 min
30 min
```

Features :

* returns ;
* spread ;
* volume ;
* short momentum ;
* short volatility ;
* mean reversion.

---

## Short Term

```text
1h
4h
1d
```

Features :

* momentum ;
* volatility ;
* correlation ;
* volume ;
* regime.

---

## Medium Term

```text
5d
20d
60d
```

Features :

* trend ;
* momentum ;
* volatility ;
* relative strength ;
* fundamental reaction.

---

## Long Term

```text
120d
252d+
```

Features :

* long-term momentum ;
* drawdown ;
* fundamentals ;
* valuation ;
* structural regime.

---

# 7. Important : les données n'ont pas toutes la même fréquence

Chaque feature doit avoir son propre rythme d'actualisation.

```text
Tick
 │
 ├── price
 ├── spread
 └── microstructure

5 min
 │
 ├── volatility
 ├── momentum
 └── HMM-HF

1 hour
 │
 ├── medium momentum
 ├── correlation
 └── HMM-MT

1 day
 │
 ├── long momentum
 ├── drawdown
 └── HMM-LT

Event
 │
 ├── earnings
 ├── guidance
 ├── news
 └── filings
```

Il ne faut donc **jamais recalculer un modèle lent inutilement à chaque tick**.

---

# 8. Online Learning

## 8.1 Principe

Le système est initialisé avec un petit historique de bootstrap puis fonctionne en mode online.

```text
Historical Seed
       │
       ▼
Initial Model
       │
       ▼
REAL-TIME
       │
       ├── inference
       │
       └── online update
```

Le but n'est pas :

```text
historique → entraînement → modèle figé
```

mais :

```text
historique minimal
      ↓
modèle initial
      ↓
observation
      ↓
prediction
      ↓
nouvelle observation
      ↓
model update
      ↓
prediction améliorée
      ↓
...
```

---

# 9. HMM Online

Le HMM sert principalement à détecter les régimes.

Exemples de régimes :

```text
TREND_LOW_VOL
TREND_HIGH_VOL
SIDEWAYS
HIGH_VOL
CRISIS
```

Le modèle ne doit pas simplement retourner :

```text
regime = CRISIS
```

Il doit conserver une distribution :

```text
TREND_LOW_VOL     0.72
SIDEWAYS          0.18
HIGH_VOL          0.08
CRISIS            0.02
```

Ainsi les décisions peuvent prendre en compte l'incertitude.

---

## 9.1 Observation vector

Exemple :

$$
x_t =
[
r_t,
\sigma_t,
volume_t,
momentum_t,
correlation_t,
drawdown_t
]
$$

Le modèle estime :

$$
P(S_t \mid x_{1:t})
$$

---

## 9.2 HMM par horizon

On prévoit plusieurs HMM :

```text
HMM-HF
    5m / 30m

HMM-MT
    1h / 4h

HMM-LT
    1d
```

Exemple :

```text
HF:
    trend       72%
    sideways    21%
    crisis       7%

MT:
    trend       81%
    sideways    14%
    crisis       5%

LT:
    trend       64%
    sideways    29%
    crisis       7%
```

Le système peut donc détecter un désaccord entre horizons.

---

# 10. Online Volatility

Une première version utilise EWMA.

$$
\sigma_t^2 =
\lambda\sigma_{t-1}^2
+
(1-\lambda)r_t^2
$$

Avantages :

* aucun retraining ;
* très peu de calcul ;
* mise à jour en temps réel ;
* état compact.

---

# 11. Momentum

Plusieurs horizons :

```text
momentum_5m
momentum_30m
momentum_1h
momentum_1d
momentum_20d
momentum_60d
momentum_252d
```

Les signaux doivent idéalement être normalisés par la volatilité.

Exemple :

$$
M_{i,h} =
\frac{R_{i,h}}{\sigma_{i,h}}
$$

---

# 12. Mean Reversion

Principalement utile sur les horizons courts.

Exemples de features :

* distance à moyenne mobile ;
* z-score ;
* short-term reversal ;
* distance à VWAP ;
* deviation from expected range.

---

# 13. Correlation Engine

Le système maintient une matrice :

$$
\Sigma_t
$$

ou une matrice de covariance/volatilité équivalente.

Elle sert à calculer :

* portfolio volatility ;
* diversification ;
* concentration ;
* risk contribution ;
* correlation spikes.

---

# 14. Portfolio State

Chaque position possède un état complet.

Exemple :

```python
PositionState(
    symbol="SPY",

    quantity=10,
    price=650.20,
    avg_price=620.00,

    market_value=6502,
    pnl=302,

    weight=0.21,
    target_weight=0.18,

    volatility=0.17,
    risk_contribution=0.23,

    signal_hf=0.42,
    signal_st=0.61,
    signal_mt=0.73,
    signal_lt=0.81,

    regime="TREND_LOW_VOL",
    regime_probability=0.82,
)
```

---

# 15. Risk Engine

Le Risk Engine doit être indépendant des modèles de signal.

Il surveille notamment :

* position limits ;
* weight limits ;
* target drift ;
* volatility ;
* portfolio volatility ;
* drawdown ;
* concentration ;
* correlation ;
* leverage ;
* risk contribution ;
* signal reversal ;
* regime changes ;
* signal disagreement.

---

# 16. Risk Contribution

Pour une covariance \(\Sigma\) et des poids \(w\) :

$$
\sigma_p =
\sqrt{w^T\Sigma w}
$$

Contribution marginale :

$$
MRC_i =
\frac{(\Sigma w)_i}{\sigma_p}
$$

Risk contribution :

$$
RC_i =
w_i MRC_i
$$

Contribution relative :

$$
RC_i^{relative}
=
\frac{w_i(\Sigma w)_i}
{w^T\Sigma w}
$$

Cela permet d'éviter de regarder uniquement le poids nominal.

Une position de 10 % peut représenter beaucoup plus que 10 % du risque du portefeuille.

---

# 17. Target Allocation

Le moteur doit distinguer :

```text
CURRENT WEIGHT
TARGET WEIGHT
DRIFT
```

Exemple :

```text
Current = 21%
Target  = 18%
Drift   = +3%
```

Le target weight peut dépendre de :

```text
signal
regime
volatility
correlation
risk budget
portfolio state
qualitative score
```

---

# 18. Risk Parity / HRP

Une méthode possible pour transformer les signaux en allocation est :

```text
Signal
   ↓
Risk Budget
   ↓
HRP / Risk Allocation
   ↓
Target Weights
```

Le système doit pouvoir utiliser :

* volatility targeting ;
* risk parity ;
* hierarchical risk parity ;
* concentration constraints.

---

# 19. Qualitative / Fundamental Engine

Le système ne doit pas être purement technique.

Il doit également intégrer les informations concernant les entreprises.

Exemples :

### Financial performance

```text
Revenue growth
EPS growth
Free cash flow growth
Margin change
Debt change
ROIC
```

### Earnings

```text
earnings surprise
EPS surprise
revenue surprise
guidance
guidance revision
```

### Valuation

```text
P/E
EV/EBITDA
P/S
FCF yield
```

### Analyst / market information

```text
estimate revisions
earnings revisions
consensus changes
```

---

# 20. Fundamental data est event-driven

Une donnée trimestrielle ne doit pas être considérée comme une feature realtime classique.

Exemple :

```text
Quarterly Earnings
       ↓
Fundamental Event
       ↓
Feature Update
       ↓
Online Model Update
```

Une donnée fondamentale doit conserver :

```text
period
publication_time
available_time
source
value
```

---

# 21. Information timing

C'est une contrainte critique.

Il faut distinguer :

```text
period
    ↓
quand la donnée concerne

publication_time
    ↓
quand elle est publiée

received_time
    ↓
quand notre système la reçoit

decision_time
    ↓
quand le modèle l'utilise
```

Le moteur ne doit jamais utiliser une information avant sa disponibilité réelle.

Cela évite le look-ahead bias.

---

# 22. News Engine

Les news sont traitées comme des événements.

Pipeline :

```text
News
 ↓
AI / NLP
 ↓
Structured Extraction
 ↓
Validation
 ↓
Quantitative Features
 ↓
Online Model
```

Exemples de catégories :

```text
earnings
guidance
product
management
M&A
regulation
litigation
macro
other
```

---

# 23. Structured AI Output

Le LLM ne doit pas produire une dissertation.

Il doit transformer :

```text
article / filing / news
```

en :

```text
structured financial event
```

Exemple conceptuel :

```json
{
  "symbol": "XXX",
  "event_type": "earnings",
  "sentiment": 0.82,
  "relevance": 0.97,
  "novelty": 0.91,
  "revenue_growth": 0.18,
  "earnings_surprise": 0.07,
  "guidance_change": 0.65,
  "risk_impact": -0.12,
  "horizon_days": 60,
  "confidence": 0.96
}
```

Les Structured Outputs / JSON Schema permettent de contraindre directement la sortie du modèle à une structure définie.

---

# 24. LLM ≠ Decision Maker

Le LLM ne doit pas faire :

```text
article
 ↓
BUY
```

Il doit faire :

```text
article
 ↓
structured information
 ↓
features
 ↓
online quantitative model
 ↓
signal
```

Le LLM sert principalement de :

> **text → structured financial information converter**

---

# 25. Novelty

Le système doit éviter de compter plusieurs fois la même information.

Exemple :

```text
Article 1:
"NVIDIA raises guidance"

Article 2:
"NVIDIA outlook improves"

Article 3:
"NVIDIA raises forecast"
```

Ces articles peuvent représenter le même événement.

Le moteur doit donc estimer :

```text
novelty
event_similarity
event_relevance
```

Un événement déjà connu ne doit pas générer artificiellement trois nouveaux signaux.

---

# 26. Confidence

Chaque extraction AI doit comporter une confiance :

```text
confidence = 0.94
```

Cette confiance peut ensuite pondérer l'information :

$$
S_\text{event}
=
confidence
\times
relevance
\times
novelty
\times
impact
$$

---

# 27. Qualitative Score

Une première représentation peut être :

$$
Q_i =
\sum_k w_k f_{i,k}
$$

avec :

```text
earnings
revenue growth
margin
FCF
guidance
news sentiment
analyst revisions
balance sheet
...
```

Mais les poids ne doivent pas nécessairement rester fixes.

Le système doit pouvoir apprendre progressivement leur pouvoir prédictif.

---

# 28. Online Predictive Learning

Pour chaque facteur, on peut mesurer sa relation avec les rendements futurs :

$$
E[r_{t+h}|X_t]
$$

pour plusieurs horizons \(h\).

Exemple :

```text
                    Earnings Surprise
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
             +1d          +5d          +20d
              │            │            │
            return       return       return
```

Le système peut ainsi apprendre qu'une feature est :

```text
short-term
medium-term
long-term
```

---

# 29. Signal Fusion

Le signal final ne doit pas être uniquement basé sur le prix.

On a :

```text
Market Signal
Regime Signal
Fundamental Signal
Event Signal
```

Pour chaque actif \(i\) et horizon \(h\) :

$$
S_{i,h}
=
F(
S^{market}_{i,h},
S^{regime}_{i,h},
S^{fundamental}_{i,h},
S^{event}_{i,h}
)
$$

La fonction \(F\) peut progressivement devenir un modèle appris.

---

# 30. Signal Disagreement

Un indicateur important est le désaccord entre horizons.

Exemple :

```text
HF      +0.72
ST      +0.41
MT      -0.18
LT      -0.43
```

Le système doit identifier :

```text
SHORT TERM BULLISH
LONG TERM BEARISH
```

Cela peut être une information de risque à part entière.

---

# 31. Decision Engine

Le Decision Engine combine :

```text
Signals
+
Regime
+
Portfolio
+
Risk
+
Target weights
```

et produit :

```python
TradeProposal(
    symbol="AAPL",
    current_weight=0.21,
    target_weight=0.17,
    delta=-0.04,

    reason=[
        "target drift",
        "risk contribution elevated",
        "medium-term signal weakened"
    ],

    confidence=0.78
)
```

Le proposal ne devient pas automatiquement un ordre.

---

# 32. Alerts

Exemples :

```text
POSITION_DRIFT
VOLATILITY_SPIKE
CORRELATION_SPIKE
DRAWDOWN
RISK_LIMIT
REGIME_CHANGE
SIGNAL_REVERSAL
SIGNAL_DISAGREEMENT
NEW_EARNINGS
GUIDANCE_CHANGE
IMPORTANT_NEWS
```

---

# 33. Dashboard

Le dashboard doit permettre de voir en temps réel :

### Portfolio

```text
Total value
PnL
Daily PnL
Portfolio volatility
Drawdown
Leverage
```

### Positions

```text
Symbol
Weight
Target
Drift
PnL
Volatility
Risk contribution
```

### Signals

```text
HF
ST
MT
LT
```

### Regime

```text
HMM state probabilities
```

### Qualitative

```text
Fundamental score
News score
Latest events
Confidence
```

---

# 34. Architecture logicielle

```text
trading_engine/
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── market_feed.py
│   ├── news_feed.py
│   ├── fundamental_feed.py
│   ├── event_bus.py
│   ├── market_state.py
│   └── bar_builder.py
│
├── portfolio/
│   ├── positions.py
│   └── portfolio.py
│
├── features/
│   ├── volatility.py
│   ├── momentum.py
│   ├── mean_reversion.py
│   ├── correlations.py
│   └── fundamentals.py
│
├── models/
│   ├── hmm.py
│   ├── online_regression.py
│   ├── online_factors.py
│   └── regime.py
│
├── ai/
│   ├── structured_extraction.py
│   ├── news_classifier.py
│   └── schemas.py
│
├── signals/
│   ├── market.py
│   ├── fundamental.py
│   ├── event.py
│   └── fusion.py
│
├── risk/
│   ├── portfolio_risk.py
│   ├── position_risk.py
│   ├── covariance.py
│   └── limits.py
│
├── allocation/
│   ├── risk_parity.py
│   ├── hrp.py
│   └── targeting.py
│
├── decision/
│   └── engine.py
│
├── storage/
│   ├── database.py
│   └── event_log.py
│
├── api/
│   └── server.py
│
├── dashboard/
│
├── tests/
│
└── main.py
```

---

# 35. Core data model

Chaque objet important doit être immutable ou versionné autant que possible.

### Position

```python
Position(
    symbol,
    quantity,
    avg_price
)
```

### Market State

```python
MarketState(
    symbol,
    price,
    timestamp
)
```

### Signal

```python
Signal(
    symbol,
    horizon,
    value,
    confidence,
    timestamp,
    source
)
```

### Fundamental Event

```python
FundamentalEvent(
    symbol,
    event_type,
    value,
    period,
    available_at,
    source
)
```

### News Event

```python
NewsEvent(
    symbol,
    event_type,
    sentiment,
    relevance,
    novelty,
    confidence,
    timestamp,
    source
)
```

### Risk State

```python
RiskState(
    weight,
    target_weight,
    volatility,
    risk_contribution,
    drawdown
)
```

---

# 36. Asynchronous architecture

Le système doit être conçu autour de `asyncio`.

Exemple :

```python
async def main():

    await asyncio.gather(
        market_feed.run(),
        news_feed.run(),
        portfolio_monitor.run(),
        risk_engine.run(),
        dashboard.run(),
    )
```

Chaque composant communique via l'EventBus.

---

# 37. Exemple de boucle complète

```python
async for event in feed:

    await event_bus.publish(event)

    if event.type == "TRADE":
        market_state.update(event)

        features.update(event)

        hmm_hf.update(features.fast)

        portfolio.update_market(event)

        risk.update(portfolio)

    if event.type == "BAR_1H":
        hmm_mt.update(features.medium)

    if event.type == "BAR_1D":
        hmm_lt.update(features.long)

    if event.type == "NEWS":
        structured_event = await ai.extract(event)

        qualitative.update(structured_event)

        online_model.update(structured_event)

    decision_engine.evaluate()
```

---

# 38. Storage

Même si le moteur est realtime, **tout doit être enregistré**.

À conserver :

```text
raw market events
features
model states
HMM probabilities
fundamental events
news
AI outputs
signals
risk states
target weights
trade proposals
```

Cela permet ensuite de répondre à :

> "Pourquoi le modèle voulait réduire cette position à 14:32 ?"

---

# 39. Reproductibilité

Chaque décision doit pouvoir être reconstruite à partir de :

```text
timestamp
market state
features
model version
model parameters
signals
risk state
fundamental information available at that time
news available at that time
```

Un `decision_id` doit idéalement relier tout cela.

---

# 40. Backtesting

Le backtesting n'est pas le moteur principal.

Il sert à :

* validation ;
* analyse historique ;
* comparaison de modèles ;
* détection de bugs ;
* mesure de performance ;
* ablation studies ;
* analyse des facteurs.

Le moteur live reste :

```text
online inference
+
online learning
```

Le log historique permet ensuite de reconstruire le comportement passé.

---

# 41. Anti-look-ahead

Le système doit respecter strictement :

$$
information\_available(t)
$$

Une information avec :

```text
period = Q2
publication = 2026-08-10
```

ne peut pas être utilisée pour une décision du :

```text
2026-07-20
```

Même si la base de données historique contient déjà la valeur finale.

---

# 42. Initialisation

Le système ne démarre pas totalement "from zero".

Pour les modèles nécessitant une distribution initiale :

```text
historical seed
        ↓
initial parameters
        ↓
online adaptation
```

Cela concerne notamment :

* HMM ;
* covariance ;
* volatility ;
* normalisation des features.

---

# 43. Online vs retraining

Tous les modèles ne doivent pas être entraînés à la même fréquence.

| Modèle                 | Update            |
| ---------------------- | ----------------- |
| Price state            | tick              |
| EWMA volatility        | tick/bar          |
| Momentum               | bar               |
| Correlation            | 5m/1h             |
| HMM-HF                 | 5m                |
| HMM-MT                 | 1h                |
| HMM-LT                 | 1d                |
| Fundamentals           | event             |
| News NLP               | event             |
| Long-term factor model | daily             |
| Allocation             | event / scheduled |

---

# 44. AI Architecture

Le LLM doit être utilisé principalement pour les données non structurées :

```text
News
Filings
Earnings calls
Management statements
```

Pipeline :

```text
RAW TEXT
   ↓
LLM
   ↓
STRUCTURED OUTPUT
   ↓
VALIDATION
   ↓
FEATURE STORE
   ↓
ONLINE MODEL
```

Le modèle quantitatif reste responsable de l'interprétation statistique.

---

# 45. Modèle AI économique

Il n'est pas nécessaire d'utiliser un gros modèle pour chaque événement.

Architecture :

```text
                New event
                    │
                    ▼
             Cheap / fast model
                    │
             ┌──────┴──────┐
             │             │
         confident       uncertain
             │             │
             ▼             ▼
          accept       stronger model
```

Cela permet de réduire :

* latence ;
* coût ;
* consommation de tokens.

---

# 46. V1 Development Roadmap

## Phase 1 — Core

```text
[x] Project structure
[x] Configuration
[x] EventBus
[x] MarketState
[x] PositionState
[x] PortfolioState
```

## Phase 2 — Realtime

```text
[ ] Alpaca WebSocket
[ ] Trade events
[ ] Quote events
[ ] Bar events
[ ] Reconnection
[ ] Heartbeat
[ ] Event timestamps
```

## Phase 3 — Features

```text
[ ] Returns
[ ] Momentum
[x] EWMA volatility
[ ] Mean reversion
[ ] Correlation
```

## Phase 4 — Online Models

```text
[ ] HMM-HF
[ ] HMM-MT
[ ] HMM-LT
[ ] Online factor model
```

## Phase 5 — Risk

```text
[ ] Portfolio volatility
[ ] Covariance
[ ] Risk contribution
[ ] Drawdown
[ ] Concentration
[ ] Limits
```

## Phase 6 — Allocation

```text
[ ] Signal → target weight
[ ] Volatility targeting
[ ] Risk parity
[ ] HRP
[ ] Drift monitoring
```

## Phase 7 — Qualitative

```text
[ ] Fundamental data
[ ] Earnings events
[ ] Guidance
[ ] SEC/filings
[ ] News feed
```

## Phase 8 — AI

```text
[ ] Structured output schema
[ ] News extraction
[ ] Event classification
[ ] Sentiment
[ ] Novelty
[ ] Confidence
[ ] Fundamental extraction
```

## Phase 9 — Decision Engine

```text
[ ] Signal fusion
[ ] Risk checks
[ ] Trade proposals
[ ] Reason generation
[ ] Alerts
```

## Phase 10 — Dashboard

```text
[ ] Portfolio overview
[ ] Position monitor
[ ] Signal monitor
[ ] Regime monitor
[ ] Risk monitor
[ ] News/events
[ ] Decision history
```

---

# 47. Final target architecture

```text
                           ┌───────────────────────┐
                           │       ALPACA          │
                           │                       │
                           │ Market │ News │ Data │
                           └───────────┬───────────┘
                                       │
                                       ▼
                              ┌────────────────┐
                              │    EVENT BUS   │
                              └───────┬────────┘
                                      │
             ┌────────────────────────┼────────────────────────┐
             │                        │                        │
             ▼                        ▼                        ▼
       MARKET ENGINE            FUNDAMENTAL               NEWS ENGINE
             │                    ENGINE                       │
             │                        │                        ▼
             │                        │                  Structured AI
             │                        │                        │
             └──────────────┬─────────┴────────────────────────┘
                            │
                            ▼
                     FEATURE ENGINE
                            │
            ┌───────────────┼────────────────┐
            ▼               ▼                ▼
          HMM-HF          HMM-MT           HMM-LT
            │               │                │
            └───────────────┼────────────────┘
                            ▼
                    ONLINE MODELS
                            │
                            ▼
                      SIGNAL FUSION
                            │
                            ▼
                     PORTFOLIO ENGINE
                            │
                            ▼
                       RISK ENGINE
                            │
                            ▼
                    ALLOCATION ENGINE
                            │
                            ▼
                    DECISION ENGINE
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
             DASHBOARD             ALERTS
                  │
                  ▼
            TRADE PROPOSAL
                  │
                  ▼
          HUMAN / EXECUTION
```

---

# 48. Design principles

Le projet doit respecter les principes suivants :

1. **Realtime first**
2. **Online learning first**
3. **Multi-horizon by design**
4. **Risk before execution**
5. **Signal ≠ order**
6. **LLM ≠ trading decision**
7. **Every event is timestamped**
8. **No look-ahead**
9. **All model states are observable**
10. **All decisions are reproducible**
11. **Qualitative information becomes structured data**
12. **Historical data is primarily used for initialization and validation**
13. **Models update at frequencies appropriate to their timescale**
14. **Portfolio risk is evaluated globally, not position-by-position only**
15. **Every target allocation must have an explainable reason**

---

# 49. First implementation milestone

La première milestone concrète est volontairement petite :

```text
Alpaca WebSocket
       ↓
EventBus
       ↓
MarketState
       ↓
PositionState
       ↓
PortfolioState
       ↓
EWMA Volatility
       ↓
5m / 1h / 1d bars
       ↓
Basic HMM
       ↓
Risk Engine
       ↓
Console output
```

Une fois cette boucle stable, on ajoute :

```text
Fundamentals
      +
News
      ↓
Structured AI
      ↓
Qualitative Features
      ↓
Online Predictive Model
```

Puis seulement :

```text
Target Allocation
      ↓
Trade Proposal
      ↓
Dashboard
      ↓
Optional Execution
```

**Objectif final : construire une sorte de "control tower" quantitative du portefeuille : le système observe continuellement le marché, apprend progressivement, maintient une représentation probabiliste de l'état de chaque actif et du portefeuille, intègre les informations fondamentales et qualitatives, mesure le risque global et explique en temps réel pourquoi l'allocation cible évolue.**
